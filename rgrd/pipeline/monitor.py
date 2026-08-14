from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from rgrd.pipeline.state import StateStore


def _tail(path: Path, lines: int) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readlines()[-lines:]


def _gpu_snapshot() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    values: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            values.append(
                dict(
                    zip(
                        ("index", "util_pct", "memory_mib", "total_mib", "temp_c"),
                        parts,
                        strict=True,
                    )
                )
            )
    return values


def snapshot(state_path: Path, tail_lines: int = 30) -> dict[str, Any]:
    state = StateStore(state_path).load()
    process_alive = False
    if state.pid is not None:
        try:
            os.kill(state.pid, 0)
            process_alive = True
        except OSError:
            process_alive = False
    return {
        "state": state.model_dump(mode="json"),
        "process_alive": process_alive,
        "gpu": _gpu_snapshot(),
        "log_tail": _tail(Path(state.log_path), tail_lines) if state.log_path else [],
        "terminal": state.status in {"passed", "failed", "stopped"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only RGRD run monitor")
    parser.add_argument("--state", type=Path, default=Path("artifacts/run_state.json"))
    parser.add_argument("--tail-lines", type=int, default=30)
    arguments = parser.parse_args()
    print(json.dumps(snapshot(arguments.state, arguments.tail_lines), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
