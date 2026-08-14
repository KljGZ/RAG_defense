from __future__ import annotations

import argparse
import json
from pathlib import Path

from rgrd.evaluation import ScoredQuery, evaluate_conformal_detector
from rgrd.experiments.resume import project_provenance, validate_rows_provenance


def _rows(input_dir: Path, partition: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["query_id"]), str(row["family"]))
                if row["partition"] == partition and key not in seen:
                    rows.append(row)
                    seen.add(key)
    return rows


def combine(root: Path, input_dir: Path) -> dict[str, object]:
    provenance = project_provenance(root)
    calibration_rows = _rows(input_dir, "calibration")
    clean_rows = _rows(input_dir, "clean_test")
    attack_rows = _rows(input_dir, "attack_test")
    validate_rows_provenance(calibration_rows + clean_rows + attack_rows, provenance)
    family_counts: dict[str, int] = {}
    for row in attack_rows:
        family = str(row["family"])
        family_counts[family] = family_counts.get(family, 0) + 1
    required_families = {"PoisonedRAG-B": 50, "PoisonedRAG-W": 50, "Phantom": 50}
    shortages = {
        **({"calibration": [200, len(calibration_rows)]} if len(calibration_rows) < 200 else {}),
        **({"clean_test": [100, len(clean_rows)]} if len(clean_rows) < 100 else {}),
        **{
            family: [required, family_counts.get(family, 0)]
            for family, required in required_families.items()
            if family_counts.get(family, 0) < required
        },
    }
    if shortages:
        raise RuntimeError(f"E04 sample quota not met: {shortages}")
    calibration = [
        ScoredQuery(str(row["query_id"]), float(row["score"]), bool(row["role_identifiable"]))
        for row in calibration_rows[:200]
    ]
    clean = [
        ScoredQuery(str(row["query_id"]), float(row["score"]), bool(row["role_identifiable"]))
        for row in clean_rows[:100]
    ]
    attacks: list[ScoredQuery] = []
    for family, required in required_families.items():
        selected = [row for row in attack_rows if row["family"] == family][:required]
        attacks.extend(
            ScoredQuery(
                str(row["query_id"]),
                float(row["score"]),
                bool(row["role_identifiable"]),
                family,
            )
            for row in selected
        )
    result = evaluate_conformal_detector(calibration, clean, attacks, alpha=0.05)
    result.update(provenance)
    statistics_path = root / "artifacts/statistics/detection.json"
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = root / "artifacts/reports/detection_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    clean_summary = result["ordinary_clean"]
    lines = [
        "# RGRD-V0 Conformal Detection Report",
        "",
        "Calibration used only 200 clean NQ queries. No attack score selected or adjusted the threshold.",
        "",
        f"Ordinary-clean FPR: **{clean_summary['rate']:.3%}** "
        f"(exact 95% CI {clean_summary['exact_95_ci'][0]:.3%}–{clean_summary['exact_95_ci'][1]:.3%}).",
        "",
        "| Family | n | TPR @ query-FPR 5% | Role-identifiable |",
        "|---|---:|---:|---:|",
    ]
    for family, value in result["per_family"].items():
        lines.append(
            f"| {family} | {value['n']} | {value['rate']:.3%} | {value['role_identifiable_rate']:.3%} |"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine E04 conformal detection results")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    result = combine(args.root, args.input)
    print(
        json.dumps(
            {"ordinary_clean": result["ordinary_clean"], "per_family": result["per_family"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
