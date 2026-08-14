from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from rgrd.provenance import utc_now


class StateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GateDecision(StateModel):
    gate: Literal["gate_1", "gate_2", "gate_3", "gate_robustness"]
    passed: bool
    decided_at: str = Field(default_factory=utc_now)
    evidence_paths: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class PhaseRecord(StateModel):
    phase: str
    status: Literal["pending", "running", "passed", "failed", "skipped"] = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class RunState(StateModel):
    schema_version: int = 1
    run_id: str
    status: Literal["created", "running", "passed", "failed", "stopped"] = "created"
    root: str
    detector_code_commit: str | None = None
    model_revisions: dict[str, str] = Field(default_factory=dict)
    pipeline_config_sha256: str | None = None
    pid: int | None = None
    log_path: str | None = None
    current_phase: str | None = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    phases: dict[str, PhaseRecord] = Field(default_factory=dict)
    gates: dict[str, GateDecision] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class StateStore:
    """Atomic, single-writer run state used by the runner and read-only monitor."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> RunState:
        return RunState.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, state: RunState) -> None:
        state.updated_at = utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(state.model_dump_json(indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def mutate(self, callback: Any) -> RunState:
        state = self.load()
        callback(state)
        self.save(state)
        return state


def require_gate(state: RunState, gate: str) -> None:
    decision = state.gates.get(gate)
    if decision is None:
        raise RuntimeError(f"{gate} has not been decided")
    if not decision.passed:
        raise RuntimeError(f"{gate} failed: {'; '.join(decision.reasons)}")
