from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from rgrd.provenance import utc_now


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _legacy_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    event_root = root / "artifacts/events/e02_mechanism"
    for path in sorted(event_root.glob("**/*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def write_legacy_correction(root: Path) -> tuple[Path, Path]:
    output_root = root / "artifacts/v0_legacy"
    corrected = output_root / "V0_corrected_final_report.md"
    salvage = output_root / "V0_exploratory_salvage_report.md"
    mechanism_path = root / "artifacts/statistics/mechanism_tests.json"
    mechanism = (
        json.loads(mechanism_path.read_text(encoding="utf-8"))
        if mechanism_path.is_file()
        else {}
    )
    counts = mechanism.get("family_counts", {})
    missing = mechanism.get("missing_quotas", {})
    corrected_text = f"""# Corrected interpretation of the frozen RGRD-V0 run

Generated at `{utc_now()}`. The original `artifacts/reports/V0_final_report.md` is
preserved byte-for-byte and is not replaced by this correction.

## Corrected Gate 3 status: NOT_ESTIMABLE

Gate 3 was not estimable because the preregistered family quotas were not reached.
H1--H3 were not executed. The confirmatory mechanism claim is therefore inconclusive.

- Observed successful-event counts: `{json.dumps(counts, sort_keys=True)}`
- Recorded quota shortfalls: `{json.dumps(missing, sort_keys=True)}`

This is not evidence that the mechanism hypothesis was tested and failed. It is an
estimability failure caused by the V0 success-conditioned quota design.

## Estimator limitation

The legacy `T_RG` statistic applies positive clipping and separately normalizes the
retrieval and generation maps. It therefore discards absolute effect scale: a chunk
with negligible generation mass can still obtain an extreme `T_RG`. The observed
legacy role direction was mixed, and no confirmatory conclusion may be drawn from it.

The frozen evidence remains useful only for explicitly exploratory diagnostics. V0.1
uses scale-preserving Oracle A/P Shapley contrasts and an outcome-independent query
sample instead.
"""
    _atomic_text(corrected, corrected_text)

    rows = _legacy_rows(root)
    by_query: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault((str(row["family"]), str(row["query_id"])), []).append(row)
    aggregated: list[dict[str, Any]] = []
    for (family, query_id), group in sorted(by_query.items()):
        aggregated.append(
            {
                "family": family,
                "query_id": query_id,
                "poison_t_rg": float(np.median([row["poison_t_rg"] for row in group])),
                "poison_generation_mass": float(
                    np.median([row["poison_role"]["generation_mass"] for row in group])
                ),
                "clean_t_rg": float(np.median([row["clean_t_rg"] for row in group])),
                "clean_generation_mass": float(
                    np.median([row["clean_role"]["generation_mass"] for row in group])
                ),
            }
        )
    noise_path = root / "artifacts/statistics/noop_noise_floor.json"
    noise = (
        float(json.loads(noise_path.read_text(encoding="utf-8")).get("generation", 1e-8))
        if noise_path.is_file()
        else 1e-8
    )
    family_lines: list[str] = []
    for family in sorted({row["family"] for row in aggregated}):
        values = [row for row in aggregated if row["family"] == family]
        poison_rho = spearmanr(
            [row["poison_t_rg"] for row in values],
            [row["poison_generation_mass"] for row in values],
        ).statistic
        clean_rho = spearmanr(
            [row["clean_t_rg"] for row in values],
            [row["clean_generation_mass"] for row in values],
        ).statistic
        low_extreme = sum(
            row["clean_generation_mass"] <= noise and row["clean_t_rg"] > 0.8
            for row in values
        )
        family_lines.append(
            f"| {family} | {len(values)} | {poison_rho:.4g} | {clean_rho:.4g} | "
            f"{low_extreme} |"
        )
    salvage_text = f"""# V0 exploratory salvage (`EXPLORATORY_ONLY`)

This document aggregates the frozen, success-conditioned V0 events by independent
query. It is not a gate, does not select a threshold, and cannot support a
confirmatory mechanism claim.

| Family | Query n | Spearman(T_RG, poison generation mass) | Spearman(T_RG, clean generation mass) | Clean low-mass and T_RG>0.8 |
|---|---:|---:|---:|---:|
{chr(10).join(family_lines)}

"Low generation mass" is defined here, before counting, as no larger than the frozen
V0 deterministic no-op generation floor (`{noise:.6g}`). The diagnostic directly
checks the scale pathology: an extreme normalized total-variation score can coexist
with negligible absolute generation effect.

Rows read: {len(rows)}; query-level rows after aggregation: {len(aggregated)}.
"""
    _atomic_text(salvage, salvage_text)
    return corrected, salvage


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze corrected V0 interpretation")
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    corrected, salvage = write_legacy_correction(arguments.root.resolve())
    print(json.dumps({"corrected": str(corrected), "salvage": str(salvage)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
