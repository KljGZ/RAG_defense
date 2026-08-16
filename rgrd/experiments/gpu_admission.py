from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GpuMemory:
    total_mib: int
    free_mib: int


_CUDA_OOM_MARKERS = (
    "cuda out of memory",
    "torch.outofmemoryerror",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)


def parse_nvidia_smi_memory(output: str) -> dict[int, GpuMemory]:
    """Parse index,total,free MiB rows emitted by nvidia-smi."""

    memory: dict[int, GpuMemory] = {}
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 3:
            raise ValueError(f"invalid nvidia-smi row {line_number}: {raw_line!r}")
        try:
            index, total_mib, free_mib = (int(column) for column in columns)
        except ValueError as exc:
            raise ValueError(f"non-integer nvidia-smi row {line_number}: {raw_line!r}") from exc
        if index in memory:
            raise ValueError(f"duplicate GPU index in nvidia-smi output: {index}")
        if total_mib <= 0 or free_mib < 0 or free_mib > total_mib:
            raise ValueError(f"invalid GPU memory values in row {line_number}: {raw_line!r}")
        memory[index] = GpuMemory(total_mib=total_mib, free_mib=free_mib)
    if not memory:
        raise ValueError("nvidia-smi returned no GPU memory rows")
    return memory


def query_gpu_memory() -> dict[int, GpuMemory]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nvidia_smi_memory(completed.stdout)


def rank_eligible_gpus(
    memory: dict[int, GpuMemory],
    *,
    minimum_free_mib: int,
    busy: Iterable[int] = (),
    quarantined: Iterable[int] = (),
) -> list[int]:
    """Return safe GPUs ordered by most free memory, then stable physical index."""

    unavailable = set(busy) | set(quarantined)
    return sorted(
        (
            index
            for index, value in memory.items()
            if index not in unavailable and value.free_mib >= minimum_free_mib
        ),
        key=lambda index: (-memory[index].free_mib, index),
    )


def log_segment_has_cuda_oom(path: Path, start_offset: int = 0) -> bool:
    """Inspect only the current process attempt, excluding stale appended failures."""

    with path.open("rb") as handle:
        handle.seek(start_offset)
        text = handle.read().decode("utf-8", errors="replace").casefold()
    return any(marker in text for marker in _CUDA_OOM_MARKERS)
