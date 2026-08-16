from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from rgrd.provenance import sha256_file, utc_now


_TERMINAL = {"passed", "failed", "stopped"}
_TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml"}
_PUBLIC_PREFIXES = (
    Path("artifacts/reports"),
    Path("artifacts/statistics"),
    Path("artifacts/audit"),
    Path("artifacts/v01/reports"),
    Path("artifacts/v01/statistics"),
    Path("artifacts/v0_legacy"),
)
_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(
        rb"(?i)\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\b"
        rb"\s*[:=]\s*['\"]?[A-Za-z0-9_./+\-=]{16,}"
    ),
)


@dataclass(frozen=True)
class PublicationPolicy:
    enabled: bool
    remote: str
    branch: str
    github_repository: str | None
    destination_prefix: Path
    maximum_file_bytes: int
    maximum_total_bytes: int
    failure_log_tail_lines: int

    @classmethod
    def load(cls, path: Path) -> "PublicationPolicy":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            remote=str(raw.get("remote", "origin")),
            branch=str(raw.get("branch", "current")),
            github_repository=(
                str(raw["github_repository"]) if raw.get("github_repository") else None
            ),
            destination_prefix=Path(str(raw.get("destination_prefix", "results/runs"))),
            maximum_file_bytes=int(raw.get("maximum_file_bytes", 20 * 1024 * 1024)),
            maximum_total_bytes=int(raw.get("maximum_total_bytes", 50 * 1024 * 1024)),
            failure_log_tail_lines=int(raw.get("failure_log_tail_lines", 200)),
        )


def _git(cwd: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        env=environment,
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _public_relative(path: Path, root: Path) -> Path | None:
    resolved = path.resolve()
    if not _within(resolved, root):
        return None
    relative = resolved.relative_to(root)
    if relative.suffix.lower() not in _TEXT_SUFFIXES:
        return None
    if not any(_within(relative, prefix) for prefix in _PUBLIC_PREFIXES):
        return None
    return relative


def _walk_path_values(value: Any, *, key: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk_path_values(child, key=str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_path_values(child, key=key)
    elif isinstance(value, str) and any(
        marker in key.lower() for marker in ("report", "result", "evidence", "statistic", "audit")
    ):
        yield value


def _candidate_evidence(root: Path, state: dict[str, Any]) -> list[Path]:
    values: set[Path] = set()
    for path_value in _walk_path_values(state):
        candidate = Path(path_value)
        values.add(candidate if candidate.is_absolute() else root / candidate)
    run_id = str(state.get("run_id", ""))
    if run_id.startswith("rgrd-v0.1-"):
        values.update(
            {
                root / "artifacts/v01/reports/V0_1_final_report.md",
                root / "artifacts/v01/statistics/v0_1_results.json",
            }
        )
    else:
        values.update(
            {
                root / "artifacts/reports/V0_final_report.md",
                root / "artifacts/reports/pipeline_report.md",
            }
        )
    result: list[Path] = []
    for candidate in sorted(values):
        relative = _public_relative(candidate, root)
        if relative is not None and candidate.is_file() and not candidate.is_symlink():
            result.append(candidate.resolve())
    return result


def _validate_evidence(paths: list[Path], policy: PublicationPolicy) -> None:
    total = 0
    for path in paths:
        size = path.stat().st_size
        if size > policy.maximum_file_bytes:
            raise RuntimeError(f"publication evidence exceeds per-file limit: {path} ({size} bytes)")
        total += size
        if total > policy.maximum_total_bytes:
            raise RuntimeError(f"publication evidence exceeds total limit ({total} bytes)")
        payload = path.read_bytes()
        _ensure_no_secret(payload, str(path))


def _ensure_no_secret(payload: bytes, label: str) -> None:
    if any(pattern.search(payload) for pattern in _SECRET_PATTERNS):
        raise RuntimeError(f"publication evidence appears to contain a secret: {label}")


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    phases = {
        name: {
            "status": value.get("status"),
            "started_at": value.get("started_at"),
            "finished_at": value.get("finished_at"),
            "error": value.get("error"),
        }
        for name, value in state.get("phases", {}).items()
    }
    return {
        "schema_version": 1,
        "run_id": state.get("run_id"),
        "status": state.get("status"),
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at"),
        "detector_code_commit": state.get("detector_code_commit"),
        "pipeline_config_sha256": state.get("pipeline_config_sha256"),
        "experiment_config_sha256": state.get("experiment_config_sha256"),
        "model_revisions": state.get("model_revisions", {}),
        "phases": phases,
        "gates": state.get("gates", {}),
        "errors": state.get("errors", []),
    }


def _readme(state: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    gates = state.get("gates", {})
    lines = [
        f"# Experiment result: {state['run_id']}",
        "",
        f"- Terminal status: `{state['status']}`",
        f"- Detector code commit: `{state.get('detector_code_commit') or 'unknown'}`",
        f"- Published at: `{utc_now()}`",
        "",
        "## Gates",
        "",
    ]
    if gates:
        lines.extend(["| Gate | Status |", "|---|---|"])
        for name, decision in sorted(gates.items()):
            status = decision.get("status")
            if hasattr(status, "value"):
                status = status.value
            lines.append(f"| {name} | {status or decision.get('passed')} |")
    else:
        lines.append("No gate decision was available before termination.")
    lines.extend(["", "## Published evidence", ""])
    lines.extend(f"- `{item['destination']}`" for item in evidence)
    lines.extend(
        [
            "",
            "Raw per-query events, model files, indexes, attack artifacts, full logs, and secrets are excluded.",
            "",
        ]
    )
    return "\n".join(lines)


def _tail_log(root: Path, state: dict[str, Any], lines: int) -> str | None:
    if state.get("status") == "passed" or lines <= 0:
        return None
    raw = state.get("log_path")
    if not raw:
        return None
    path = Path(str(raw)).resolve()
    if not _within(path, root) or not path.is_file() or path.is_symlink():
        return None
    values = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(values[-lines:]) + "\n"


def _validate_remote(remote_url: str, expected: str | None) -> None:
    if expected is None:
        return
    slug = re.escape(expected.removesuffix(".git"))
    if not re.search(rf"github\.com(?::|/){slug}(?:\.git)?$", remote_url):
        raise RuntimeError(
            f"publication remote {remote_url!r} does not match github repository {expected!r}"
        )


def _write_bundle(
    bundle: Path,
    *,
    root: Path,
    state: dict[str, Any],
    evidence_paths: list[Path],
    policy: PublicationPolicy,
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for source in evidence_paths:
        relative = source.relative_to(root)
        destination = Path("evidence") / relative
        target = bundle / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        evidence.append(
            {
                "source": relative.as_posix(),
                "destination": destination.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    log_tail = _tail_log(root, state, policy.failure_log_tail_lines)
    if log_tail:
        _ensure_no_secret(log_tail.encode("utf-8"), "terminal runner log tail")
        target = bundle / "failure_runner_tail.log"
        target.write_text(log_tail, encoding="utf-8")
        evidence.append(
            {
                "source": "terminal runner log tail",
                "destination": target.name,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    summary_path = bundle / "run_summary.json"
    _atomic_json(summary_path, _summary(state))
    readme = bundle / "README.md"
    readme.write_text(_readme(state, evidence), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "run_id": state["run_id"],
        "terminal_status": state["status"],
        "published_at": utc_now(),
        "detector_code_commit": state.get("detector_code_commit"),
        "pipeline_config_sha256": state.get("pipeline_config_sha256"),
        "experiment_config_sha256": state.get("experiment_config_sha256"),
        "evidence": evidence,
        "excluded_classes": [
            "raw per-query events",
            "models and indexes",
            "attack artifacts",
            "full logs",
            "credentials and secrets",
        ],
    }
    _atomic_json(bundle / "publication_manifest.json", manifest)
    return manifest


def publish_terminal_results(
    root: Path,
    state_path: Path,
    *,
    policy_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    state_path = state_path.resolve()
    policy_path = (policy_path or root / "configs/publication.yaml").resolve()
    policy = PublicationPolicy.load(policy_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    status_path = root / "artifacts/publications" / f"{state.get('run_id', 'unknown')}.json"
    if not policy.enabled:
        result = {"status": "disabled", "run_id": state.get("run_id"), "at": utc_now()}
        _atomic_json(status_path, result)
        return result
    if state.get("status") not in _TERMINAL:
        raise RuntimeError("results may be published only after the run reaches a terminal state")
    run_id = str(state.get("run_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise RuntimeError(f"unsafe run_id for publication: {run_id!r}")
    if policy.destination_prefix.is_absolute() or ".." in policy.destination_prefix.parts:
        raise RuntimeError("publication destination_prefix must be a safe relative path")

    remote_url = _git(root, "remote", "get-url", policy.remote).stdout.strip()
    _validate_remote(remote_url, policy.github_repository)
    branch = (
        _git(root, "branch", "--show-current").stdout.strip()
        if policy.branch == "current"
        else policy.branch
    )
    if not branch or not re.fullmatch(r"[A-Za-z0-9._/-]+", branch) or ".." in branch:
        raise RuntimeError(f"unsafe publication branch: {branch!r}")
    code_commit = str(state.get("detector_code_commit") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise RuntimeError("terminal state lacks a full detector_code_commit")
    _git(root, "fetch", "--quiet", policy.remote, branch)
    ancestor = _git(root, "merge-base", "--is-ancestor", code_commit, "FETCH_HEAD", check=False)
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"detector code commit {code_commit} is not present on {policy.remote}/{branch}"
        )

    evidence_paths = _candidate_evidence(root, state)
    _validate_evidence(evidence_paths, policy)
    destination = policy.destination_prefix / run_id
    try:
        with tempfile.TemporaryDirectory(prefix="rgrd-publication-") as temporary:
            temporary_root = Path(temporary)
            bundle = temporary_root / "bundle"
            checkout = temporary_root / "checkout"
            bundle.mkdir()
            manifest = _write_bundle(
                bundle,
                root=root,
                state=state,
                evidence_paths=evidence_paths,
                policy=policy,
            )
            _git(
                temporary_root,
                "clone",
                "--quiet",
                "--single-branch",
                "--branch",
                branch,
                remote_url,
                str(checkout),
            )
            target = checkout / destination
            if target.exists():
                existing = target / "publication_manifest.json"
                if existing.is_file():
                    old = json.loads(existing.read_text(encoding="utf-8"))
                    stable_old = [
                        item
                        for item in old.get("evidence", [])
                        if item.get("source") != "terminal runner log tail"
                    ]
                    stable_new = [
                        item
                        for item in manifest.get("evidence", [])
                        if item.get("source") != "terminal runner log tail"
                    ]
                    comparable = ("run_id", "terminal_status", "detector_code_commit")
                    if all(old.get(key) == manifest.get(key) for key in comparable) and (
                        stable_old == stable_new
                    ):
                        result = {
                            "status": "already_published",
                            "run_id": run_id,
                            "branch": branch,
                            "destination": destination.as_posix(),
                            "commit": _git(checkout, "rev-parse", "HEAD").stdout.strip(),
                            "at": utc_now(),
                        }
                        _atomic_json(status_path, result)
                        return result
                raise RuntimeError(f"publication destination already exists with different content: {destination}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(bundle, target)
            _git(checkout, "add", "--", destination.as_posix())
            _git(
                checkout,
                "-c",
                "user.name=RGRD Experiment Publisher",
                "-c",
                "user.email=rgrd-publisher@users.noreply.github.com",
                "commit",
                "--quiet",
                "-m",
                f"results: publish {run_id} ({state['status']})",
            )
            _git(checkout, "push", "--quiet", "origin", f"HEAD:refs/heads/{branch}")
            result = {
                "status": "published",
                "run_id": run_id,
                "branch": branch,
                "remote": remote_url,
                "destination": destination.as_posix(),
                "commit": _git(checkout, "rev-parse", "HEAD").stdout.strip(),
                "evidence_count": len(manifest["evidence"]),
                "at": utc_now(),
            }
            _atomic_json(status_path, result)
            return result
    except Exception as exc:
        failure = {
            "status": "failed",
            "run_id": run_id,
            "branch": branch,
            "destination": destination.as_posix(),
            "error": f"{type(exc).__name__}: {exc}",
            "at": utc_now(),
        }
        _atomic_json(status_path, failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish terminal RGRD evidence to GitHub")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    arguments = parser.parse_args()
    result = publish_terminal_results(
        arguments.root,
        arguments.state,
        policy_path=arguments.policy,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
