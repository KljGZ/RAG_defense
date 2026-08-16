import pytest

from rgrd.experiments.v01_worker import _assert_gpu_contract


def _config(allowed=(4, 5, 6, 7), excluded=(0, 1, 2, 3)):
    return {
        "runtime": {
            "gpu_admission": {
                "allowed_physical_gpus": list(allowed),
                "excluded_physical_gpus": list(excluded),
            }
        }
    }


def test_v01_worker_rejects_physical_gpu_zero(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="violates"):
        _assert_gpu_contract(_config())


def test_v01_worker_accepts_one_allowed_physical_gpu(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    assert _assert_gpu_contract(_config()) == 6


def test_v01_worker_rejects_mutated_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    with pytest.raises(RuntimeError, match="exactly"):
        _assert_gpu_contract(_config(allowed=(0, 4, 5, 6, 7)))
