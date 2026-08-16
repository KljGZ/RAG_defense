import os
import sys
from pathlib import Path

import pytest

from rgrd.experiments.gpu_admission import (
    GpuMemory,
    log_segment_has_cuda_oom,
    parse_nvidia_smi_memory,
    rank_eligible_gpus,
)
from rgrd.experiments import run as run_module
from rgrd.experiments.run import ExperimentRunner, ScheduledGpuJob


def test_parse_and_rank_gpu_memory() -> None:
    memory = parse_nvidia_smi_memory("0, 24564, 12000\n1, 24564, 22000\n2, 24564, 22000\n")
    assert memory[0] == GpuMemory(total_mib=24564, free_mib=12000)
    assert rank_eligible_gpus(
        memory,
        minimum_free_mib=19000,
        busy={1},
    ) == [2]
    assert rank_eligible_gpus(memory, minimum_free_mib=19000) == [1, 2]
    assert rank_eligible_gpus(
        memory,
        minimum_free_mib=19000,
        allowed={2},
    ) == [2]
    assert rank_eligible_gpus(
        memory,
        minimum_free_mib=19000,
        allowed=set(),
    ) == []


@pytest.mark.parametrize(
    "output",
    [
        "",
        "0, 24564\n",
        "gpu0, 24564, 22000\n",
        "0, 24564, 25000\n",
        "0, 24564, 22000\n0, 24564, 21000\n",
    ],
)
def test_parse_rejects_invalid_gpu_memory(output: str) -> None:
    with pytest.raises(ValueError):
        parse_nvidia_smi_memory(output)


def test_oom_detection_is_attempt_scoped(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    stale = b"torch.OutOfMemoryError: CUDA out of memory\n"
    log.write_bytes(stale + b"current attempt completed normally\n")
    assert log_segment_has_cuda_oom(log)
    assert not log_segment_has_cuda_oom(log, len(stale))


def test_gpu_queue_retries_only_oom_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counter = tmp_path / "attempts.txt"
    script = (
        "import pathlib,sys; "
        "path=pathlib.Path(sys.argv[1]); "
        "attempt=int(path.read_text()) if path.exists() else 0; "
        "path.write_text(str(attempt+1)); "
        "print('torch.OutOfMemoryError: CUDA out of memory' if attempt == 0 else 'ok'); "
        "raise SystemExit(1 if attempt == 0 else 0)"
    )
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.root = tmp_path
    runner.base_environment = os.environ.copy()
    runner.gpu_allowed_physical_gpus = (4,)
    runner.gpu_admission_poll_seconds = 0
    runner.gpu_oom_retry_limit = 2
    runner.gpu_oom_retry_cooldown_seconds = 0
    runner.gpu_oom_retry_headroom_mib = 1024
    progress: list[dict[str, object]] = []
    runner._progress = lambda **values: progress.append(values)  # type: ignore[method-assign]
    monkeypatch.setattr(
        run_module,
        "query_gpu_memory",
        lambda: {
            0: GpuMemory(total_mib=24564, free_mib=24500),
            4: GpuMemory(total_mib=24564, free_mib=24000),
        },
    )

    result = runner._run_gpu_queue(
        "fault-injection",
        [
            ScheduledGpuJob(
                name="shard-00",
                command=[sys.executable, "-c", script, counter],
            )
        ],
        minimum_free_mib=19000,
    )

    assert result == {"shard-00": 0}
    assert counter.read_text() == "2"
    assert any(item.get("last_oom_retry") for item in progress)
    assert any(
        item.get("gpu_admission", {}).get("allowed_physical_gpus") == [4]
        for item in progress
    )
    selected_gpus = [
        details["physical_gpu"]
        for item in progress
        for details in item.get("gpu_admission", {}).get("running", {}).values()
    ]
    assert selected_gpus
    assert set(selected_gpus) == {4}
