from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgrd.evaluation import DirectionComparison, evaluate_direction_stability
from rgrd.experiments.resume import project_provenance, validate_rows_provenance


def combine(root: Path, input_dir: Path) -> dict[str, object]:
    provenance = project_provenance(root)
    rows = {}
    for path in input_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows.setdefault(row["sample_id"], row)
    validate_rows_provenance(rows.values(), provenance)
    comparisons = []
    for row in rows.values():
        for role in ("anchor", "payload"):
            comparisons.append(
                DirectionComparison(
                    query_id=str(row["query_id"]),
                    oracle_role=role,
                    mask_direction=int(row[role]["mask_direction"]),
                    donor_direction=int(row[role]["donor_direction"]),
                )
            )
    result = evaluate_direction_stability(
        comparisons, minimum_agreement=0.80, minimum_ci_lower=0.70
    )
    result.update(provenance)
    output = root / "artifacts/statistics/intervention_robustness.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine RGRD E03 robustness")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    result = combine(args.root, args.input)
    print(json.dumps(result["gate_robustness"], sort_keys=True))
    return 0 if result["gate_robustness"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
