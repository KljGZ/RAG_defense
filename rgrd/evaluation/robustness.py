from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rgrd.statistics import clopper_pearson_interval


@dataclass(frozen=True)
class DirectionComparison:
    query_id: str
    oracle_role: str
    mask_direction: int
    donor_direction: int


def evaluate_direction_stability(
    comparisons: Iterable[DirectionComparison],
    *,
    minimum_agreement: float = 0.80,
    minimum_ci_lower: float = 0.70,
) -> dict[str, object]:
    rows = list(comparisons)
    groups: dict[str, dict[str, object]] = {}
    for role in ("anchor", "payload"):
        selected = [row for row in rows if row.oracle_role == role]
        if not selected:
            raise ValueError(f"no robustness comparisons for {role}")
        agreements = sum(
            row.mask_direction == row.donor_direction and row.mask_direction != 0
            for row in selected
        )
        interval = clopper_pearson_interval(agreements, len(selected))
        rate = agreements / len(selected)
        groups[role] = {
            "n": len(selected),
            "agreements": agreements,
            "agreement_rate": rate,
            "exact_95_ci": list(interval),
            "passes": rate >= minimum_agreement and interval[0] >= minimum_ci_lower,
        }
    passed = all(bool(group["passes"]) for group in groups.values())
    return {
        "schema_version": 1,
        "groups": groups,
        "gate_robustness": {
            "passed": passed,
            "minimum_agreement": minimum_agreement,
            "minimum_ci_lower": minimum_ci_lower,
            "reason": "both oracle role groups must satisfy preregistered agreement and CI criteria",
        },
    }
