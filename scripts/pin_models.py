from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from rgrd.provenance import sha256_file, utc_now


def _atomic_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _selected_files(root: Path, patterns: list[str]) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    relative_files = [path.relative_to(root) for path in root.rglob("*") if path.is_file()]
    selected: set[Path] = set()
    missing: list[str] = []
    for pattern in patterns:
        matches = [
            relative for relative in relative_files if fnmatch.fnmatch(relative.as_posix(), pattern)
        ]
        if matches:
            selected.update(matches)
        elif not pattern.startswith(("special_tokens", "vocab", "merges")):
            missing.append(pattern)
    if missing:
        raise FileNotFoundError(f"required model files absent under {root}: {missing}")
    return sorted(root / relative for relative in selected)


def _content_hash(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, value in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def pin(source_path: Path, pipeline_path: Path, output_path: Path) -> dict[str, Any]:
    sources = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    entries: dict[str, Any] = {}
    for role, source in sources["models"].items():
        configured = pipeline["models"].get(role)
        if configured is None:
            raise KeyError(f"pipeline has no model role {role}")
        for field, expected in (
            ("name", source["repository"]),
            ("revision", source["revision"]),
            ("local_path", source["local_path"]),
        ):
            if str(configured[field]) != str(expected):
                raise ValueError(
                    f"{role}.{field} differs between model_sources and pipeline: "
                    f"{expected!r} != {configured[field]!r}"
                )
        root = Path(source["local_path"]).resolve()
        selected = _selected_files(root, list(source["include"]))
        hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in selected}
        entries[role] = {
            "repository": source["repository"],
            "revision": source["revision"],
            "local_path": str(root),
            "transport": source["transport"],
            "content_sha256": _content_hash(hashes),
            "files": hashes,
        }
    result = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "models": entries,
    }
    _atomic_yaml(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Pin exact files loaded by RGRD models")
    parser.add_argument("--sources", type=Path, default=Path("configs/model_sources.yaml"))
    parser.add_argument("--pipeline", type=Path, default=Path("configs/pipeline/v0.yaml"))
    parser.add_argument("--output", type=Path, default=Path("configs/models.lock.yaml"))
    arguments = parser.parse_args()
    result = pin(arguments.sources, arguments.pipeline, arguments.output)
    print(
        json.dumps(
            {role: value["content_sha256"] for role, value in result["models"].items()},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
