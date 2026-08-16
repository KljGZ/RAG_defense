from __future__ import annotations

import json
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Iterable

from rgrd.provenance import sha256_file

import yaml


PROVENANCE_FIELDS = (
    "schema_version",
    "detector_code_commit",
    "model_revisions",
    "pipeline_config_sha256",
)


def event_provenance(
    root: Path,
    *,
    detector_code_commit: str,
    model_revisions: dict[str, str],
    pipeline_config: Path | None = None,
) -> dict[str, Any]:
    config_path = pipeline_config or (root / "configs/pipeline/v0.yaml")
    return {
        "schema_version": 2,
        "detector_code_commit": detector_code_commit,
        "model_revisions": dict(sorted(model_revisions.items())),
        "pipeline_config_sha256": sha256_file(config_path),
    }


def v01_event_provenance(
    root: Path,
    *,
    detector_code_commit: str,
    model_revisions: dict[str, str],
    selection_manifest: Path,
) -> dict[str, Any]:
    value = event_provenance(
        root,
        detector_code_commit=detector_code_commit,
        model_revisions=model_revisions,
        pipeline_config=root / "configs/pipeline/v0_1.yaml",
    )
    value.update(
        {
            "schema_version": 3,
            "protocol_id": "RGRD-V0.1-oracle-mechanism-audit",
            "experiment_config_sha256": sha256_file(
                root / "configs/experiments/v0_1_preregistration.yaml"
            ),
            "selection_manifest_sha256": sha256_file(selection_manifest),
        }
    )
    return value


def project_provenance(root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config = yaml.safe_load((root / "configs/pipeline/v0.yaml").read_text(encoding="utf-8"))
    revisions = {role: str(value["revision"]) for role, value in config["models"].items()}
    return event_provenance(
        root,
        detector_code_commit=commit,
        model_revisions=revisions,
    )


def validate_rows_provenance(rows: Iterable[dict[str, Any]], expected: dict[str, Any]) -> None:
    for index, row in enumerate(rows):
        mismatches = [name for name in PROVENANCE_FIELDS if row.get(name) != expected.get(name)]
        if mismatches:
            raise RuntimeError(
                f"result row {index} has stale or mixed provenance fields: {mismatches}"
            )


def prepare_jsonl_resume(
    path: Path,
    *,
    key_fields: Iterable[str],
    expected_provenance: dict[str, Any],
) -> set[tuple[str, ...]]:
    """Keep only rows from the exact current detector and return their keys."""

    fields = tuple(key_fields)
    if not fields:
        raise ValueError("resume key must contain at least one field")
    if not path.is_file():
        return set()
    valid: list[dict[str, Any]] = []
    completed: set[tuple[str, ...]] = set()
    changed = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                changed = True
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL resume row {path}:{line_number}") from exc
            matches = all(row.get(name) == value for name, value in expected_provenance.items())
            if not matches:
                changed = True
                continue
            key = tuple(str(row[name]) for name in fields)
            if key in completed:
                changed = True
                continue
            completed.add(key)
            valid.append(row)
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for row in valid:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return completed
