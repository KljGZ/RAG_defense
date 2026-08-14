from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from rgrd.statistics import clopper_pearson_interval, flag_from_calibration


@dataclass(frozen=True)
class ScoredQuery:
    query_id: str
    score: float
    role_identifiable: bool
    family: str = "clean"


def _group_summary(rows: list[tuple[ScoredQuery, float, bool]]) -> dict[str, object]:
    trials = len(rows)
    flags = sum(flag for _, _, flag in rows)
    identifiable = sum(row.role_identifiable for row, _, _ in rows)
    interval = clopper_pearson_interval(flags, trials)
    return {
        "n": trials,
        "flagged": flags,
        "rate": flags / trials,
        "exact_95_ci": [interval[0], interval[1]],
        "role_identifiable_rate": identifiable / trials,
    }


def evaluate_conformal_detector(
    calibration: Iterable[ScoredQuery],
    clean_test: Iterable[ScoredQuery],
    attack_test: Iterable[ScoredQuery],
    *,
    alpha: float = 0.05,
) -> dict[str, object]:
    calibration_rows = list(calibration)
    clean_rows = list(clean_test)
    attack_rows = list(attack_test)
    if not calibration_rows or not clean_rows or not attack_rows:
        raise ValueError("calibration, clean test, and attack test must all be non-empty")
    calibration_scores = [row.score for row in calibration_rows]

    def score(rows: list[ScoredQuery]) -> list[tuple[ScoredQuery, float, bool]]:
        return [
            (row, *flag_from_calibration(calibration_scores, row.score, alpha=alpha))
            for row in rows
        ]

    clean_scored = score(clean_rows)
    attack_scored = score(attack_rows)
    family_rows: dict[str, list[tuple[ScoredQuery, float, bool]]] = defaultdict(list)
    for row in attack_scored:
        family_rows[row[0].family].append(row)
    return {
        "schema_version": 1,
        "alpha": alpha,
        "calibration_n": len(calibration_rows),
        "selection_rule": "Top-5 retrieval attribution; generation attribution for top-2 positive-retrieval chunks; query score=max T_RG",
        "ordinary_clean": _group_summary(clean_scored),
        "all_attacks": _group_summary(attack_scored),
        "per_family": {
            family: _group_summary(rows) for family, rows in sorted(family_rows.items())
        },
        "test_rows": [
            {
                "query_id": row.query_id,
                "family": row.family,
                "score": row.score,
                "role_identifiable": row.role_identifiable,
                "p_value": p_value,
                "flagged": flagged,
            }
            for row, p_value, flagged in clean_scored + attack_scored
        ],
    }
