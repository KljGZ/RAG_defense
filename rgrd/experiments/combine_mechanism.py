from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rgrd.evaluation import MechanismObservation, run_mechanism_tests, write_mechanism_outputs
from rgrd.experiments.resume import project_provenance, validate_rows_provenance


REQUIRED = {"PoisonedRAG-B": 100, "PoisonedRAG-W": 100, "Phantom": 50}


def combine(input_dir: Path, root: Path, seed: int) -> dict[str, object]:
    provenance = project_provenance(root)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                sample_id = str(value["sample_id"])
                if sample_id not in seen:
                    rows.append(value)
                    seen.add(sample_id)
    validate_rows_provenance(rows, provenance)
    selected: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for family, required in REQUIRED.items():
        family_rows = [row for row in rows if row["family"] == family]
        counts[family] = len(family_rows)
        selected.extend(family_rows[:required])
    missing = {
        family: {"required": required, "available": counts.get(family, 0)}
        for family, required in REQUIRED.items()
        if counts.get(family, 0) < required
    }
    if missing:
        failure = {
            **provenance,
            "gate_3": {
                "passed": False,
                "reason": "successful co-chunk/reranked/end-to-end event quota not met",
                "missing": missing,
            },
            "available_counts": counts,
        }
        output = root / "artifacts/statistics/mechanism_tests.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        report = root / "artifacts/reports/mechanism_report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "# RGRD-V0 Oracle Mechanism Report\n\n"
            "Gate 3: **FAIL**. Confirmatory event quotas were not met; no H1-H3 claim was made.\n\n"
            + "\n".join(
                f"- {family}: required {item['required']}, available {item['available']}"
                for family, item in missing.items()
            )
            + "\n",
            encoding="utf-8",
        )
        return failure
    observations = [
        MechanismObservation(
            query_id=str(row["query_id"]),
            family=str(row["family"]),
            anchor_p_retrieval=float(row["anchor_p_retrieval"]),
            anchor_p_generation=float(row["anchor_p_generation"]),
            payload_p_retrieval=float(row["payload_p_retrieval"]),
            payload_p_generation=float(row["payload_p_generation"]),
            poison_t_rg=float(row["poison_t_rg"]),
            clean_t_rg=float(row["clean_t_rg"]),
        )
        for row in selected
    ]
    result = run_mechanism_tests(
        observations,
        permutations=10000,
        bootstrap_replicates=10000,
        seed=seed,
    )
    result.update(provenance)
    result["quota"] = {"required": REQUIRED, "available": counts, "selected": REQUIRED}
    write_mechanism_outputs(
        result,
        root / "artifacts/statistics/mechanism_tests.json",
        root / "artifacts/reports/mechanism_report.md",
    )
    role_path = root / "artifacts/role_maps/oracle_role_maps.parquet"
    role_path.parent.mkdir(parents=True, exist_ok=True)
    pd.json_normalize(selected).to_parquet(role_path, index=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine E02 workers and decide Gate 3")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    arguments = parser.parse_args()
    result = combine(arguments.input, arguments.root, arguments.seed)
    print(json.dumps(result["gate_3"], sort_keys=True))
    return 0 if result["gate_3"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
