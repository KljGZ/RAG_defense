from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def run(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_url(value: str) -> str:
    return value.removesuffix(".git").removesuffix("/")


def pin(root: Path, config_path: Path, output_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    projects: list[dict[str, object]] = []
    for name, expected in config["projects"].items():
        repo = root / name
        if not (repo / ".git").exists():
            raise FileNotFoundError(f"missing full clone: {repo}")
        remote = run(repo, "remote", "get-url", "origin")
        if normalized_url(remote) != normalized_url(str(expected["url"])):
            raise ValueError(f"remote mismatch for {name}: {remote}")
        shallow = run(repo, "rev-parse", "--is-shallow-repository") == "true"
        if shallow:
            raise ValueError(f"shallow repository is forbidden: {name}")
        missing = run(repo, "rev-list", "--objects", "--all", "--missing=print")
        missing_objects = [line for line in missing.splitlines() if line.startswith("?")]
        if missing_objects:
            raise ValueError(
                f"partial repository has {len(missing_objects)} missing objects: {name}"
            )
        status = run(repo, "status", "--porcelain=v1")
        license_files = sorted(
            path
            for pattern in ("LICENSE*", "COPYING*")
            for path in repo.glob(pattern)
            if path.is_file()
        )
        projects.append(
            {
                "name": name,
                "url": str(expected["url"]),
                "commit": run(repo, "rev-parse", "HEAD"),
                "branch": run(repo, "branch", "--show-current"),
                "license": {
                    "spdx": str(expected["license"]),
                    "files": [
                        {"path": str(path.relative_to(repo)), "sha256": file_hash(path)}
                        for path in license_files
                    ],
                },
                "environment": str(expected["environment"]),
                "purpose": str(expected["purpose"]),
                "local_modifications": {
                    "dirty": bool(status),
                    "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
                    "entries": status.splitlines(),
                },
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    manifest = {"schema_version": 1, "projects": projects}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Pin complete upstream Git repositories")
    parser.add_argument("--root", type=Path, default=Path("third_party"))
    parser.add_argument("--config", type=Path, default=Path("configs/upstream_projects.yaml"))
    parser.add_argument("--output", type=Path, default=Path("third_party/manifest.lock.yaml"))
    args = parser.parse_args()
    manifest = pin(args.root, args.config, args.output)
    print(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
