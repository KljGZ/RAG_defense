from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rgrd.provenance import sha256_file, utc_now
from rgrd.v01.protocol import canonical_per_query
from rgrd.v01.samples import ACTIVE_FAMILIES, load_v01_samples


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_manifest(
    root: Path,
    attack_root: Path,
    *,
    family: str,
    dataset: str,
    output: Path,
) -> dict[str, Any]:
    samples = load_v01_samples(root, attack_root, family, dataset=dataset)
    canonical, audit = canonical_per_query(samples)
    rows = [
        {"query_id": sample.query_id, "sample_id": sample.sample_id}
        for sample in canonical
    ]
    canonical_payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    source_paths = sorted({sample.provenance.get("result_path", "") for sample in samples})
    result = {
        "schema_version": 1,
        "protocol_id": "RGRD-V0.1-oracle-mechanism-audit",
        "created_at": utc_now(),
        "family": family,
        "dataset": dataset,
        "selection": asdict(audit),
        "selection_depends_on_outcomes": False,
        "forbidden_selection_fields": [
            "original_retrieval_hit",
            "original_end_to_end_success",
            "output_poison",
            "retrieval_score",
            "generation_score",
            "oracle_contrast",
        ],
        "canonical_sha256": hashlib.sha256(canonical_payload).hexdigest(),
        "source_artifacts": [
            {
                "path": value,
                "sha256": sha256_file(Path(value)) if value and Path(value).is_file() else None,
            }
            for value in source_paths
        ],
        "rows": rows,
    }
    _atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze outcome-independent V0.1 samples")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attack-root", type=Path, required=True)
    parser.add_argument("--family", choices=ACTIVE_FAMILIES, required=True)
    parser.add_argument("--dataset", default="nq")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = prepare_manifest(
        arguments.root.resolve(),
        arguments.attack_root.resolve(),
        family=arguments.family,
        dataset=arguments.dataset,
        output=arguments.output.resolve(),
    )
    print(json.dumps({key: result[key] for key in ("family", "dataset", "selection")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
