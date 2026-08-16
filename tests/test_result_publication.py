from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from rgrd.publishing.github import publish_terminal_results


def _git(cwd: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, Path, str]:
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    remote.mkdir()
    source.mkdir()
    _git(remote, "init", "--bare", "--quiet")
    _git(source, "init", "--quiet")
    _git(source, "config", "user.name", "Test User")
    _git(source, "config", "user.email", "test@example.com")
    (source / "README.md").write_text("test repository\n", encoding="utf-8")
    _git(source, "add", "README.md")
    _git(source, "commit", "--quiet", "-m", "initial")
    _git(source, "branch", "-M", "results-test")
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "--quiet", "-u", "origin", "results-test")
    return source, remote, _git(source, "rev-parse", "HEAD")


def _write_terminal_state(root: Path, commit: str) -> Path:
    report = root / "artifacts/v01/reports/V0_1_final_report.md"
    result = root / "artifacts/v01/statistics/v0_1_results.json"
    event = root / "artifacts/v01/events/shard-00.jsonl"
    log = root / "artifacts/logs/runner.log"
    report.parent.mkdir(parents=True)
    result.parent.mkdir(parents=True)
    event.parent.mkdir(parents=True)
    log.parent.mkdir(parents=True)
    report.write_text("# Final report\n\nFAIL_TESTED\n", encoding="utf-8")
    result.write_text('{"gate_v01":{"status":"FAIL_TESTED"}}\n', encoding="utf-8")
    event.write_text('{"query":"must not publish"}\n', encoding="utf-8")
    log.write_text("runner completed with a tested gate failure\n", encoding="utf-8")
    state = {
        "schema_version": 3,
        "run_id": "rgrd-v0.1-test-run",
        "status": "failed",
        "detector_code_commit": commit,
        "pipeline_config_sha256": "a" * 64,
        "experiment_config_sha256": "b" * 64,
        "model_revisions": {"generator": "c" * 40},
        "created_at": "2026-08-16T00:00:00+00:00",
        "updated_at": "2026-08-16T01:00:00+00:00",
        "log_path": str(log),
        "phases": {
            "combine": {
                "status": "passed",
                "started_at": "2026-08-16T00:30:00+00:00",
                "finished_at": "2026-08-16T01:00:00+00:00",
                "error": None,
                "progress": {"report": str(report), "result_json": str(result)},
            }
        },
        "gates": {
            "gate_v01": {
                "status": "FAIL_TESTED",
                "passed": False,
                "evidence_paths": [str(report), str(result)],
                "reasons": ["tested failure"],
            }
        },
        "errors": ["terminal gate status: primary=FAIL_TESTED"],
    }
    state_path = root / "artifacts/run_state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path


def _policy(root: Path) -> Path:
    path = root / "configs/publication.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "enabled": True,
                "remote": "origin",
                "branch": "current",
                "github_repository": None,
                "destination_prefix": "results/runs",
                "maximum_file_bytes": 1024 * 1024,
                "maximum_total_bytes": 2 * 1024 * 1024,
                "failure_log_tail_lines": 20,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_terminal_publication_is_minimal_and_idempotent(tmp_path: Path) -> None:
    source, remote, commit = _repository(tmp_path)
    state = _write_terminal_state(source, commit)
    policy = _policy(source)

    first = publish_terminal_results(source, state, policy_path=policy)
    assert first["status"] == "published"

    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "--quiet", "--branch", "results-test", str(remote), str(checkout))
    destination = checkout / "results/runs/rgrd-v0.1-test-run"
    assert (destination / "README.md").is_file()
    assert (destination / "run_summary.json").is_file()
    assert (destination / "publication_manifest.json").is_file()
    assert (destination / "evidence/artifacts/v01/reports/V0_1_final_report.md").is_file()
    assert (destination / "evidence/artifacts/v01/statistics/v0_1_results.json").is_file()
    assert not (destination / "evidence/artifacts/v01/events/shard-00.jsonl").exists()

    (source / "artifacts/logs/runner.log").write_text(
        "runner completed\npublication completed\n", encoding="utf-8"
    )
    second = publish_terminal_results(source, state, policy_path=policy)
    assert second["status"] == "already_published"


def test_publication_rejects_secret_in_key_evidence(tmp_path: Path) -> None:
    source, _, commit = _repository(tmp_path)
    state = _write_terminal_state(source, commit)
    policy = _policy(source)
    report = source / "artifacts/v01/reports/V0_1_final_report.md"
    report.write_text("github_pat_abcdefghijklmnopqrstuvwxyz0123456789\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="contain a secret"):
        publish_terminal_results(source, state, policy_path=policy)


def test_publication_requires_terminal_state(tmp_path: Path) -> None:
    source, _, commit = _repository(tmp_path)
    state = _write_terminal_state(source, commit)
    policy = _policy(source)
    value = json.loads(state.read_text(encoding="utf-8"))
    value["status"] = "running"
    state.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(RuntimeError, match="terminal state"):
        publish_terminal_results(source, state, policy_path=policy)
