from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from rgrd.statistics import cluster_bootstrap_median_ci, holm_adjust, paired_sign_flip_test


@dataclass(frozen=True)
class MechanismObservation:
    query_id: str
    family: str
    anchor_p_retrieval: float
    anchor_p_generation: float
    payload_p_retrieval: float
    payload_p_generation: float
    poison_t_rg: float
    clean_t_rg: float


def run_mechanism_tests(
    observations: Iterable[MechanismObservation],
    *,
    permutations: int = 10000,
    bootstrap_replicates: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, object]:
    rows = list(observations)
    if not rows:
        raise ValueError("mechanism validation requires observations")
    differences = {
        "H1": [row.anchor_p_retrieval - row.anchor_p_generation for row in rows],
        "H2": [row.payload_p_generation - row.payload_p_retrieval for row in rows],
        "H3": [row.poison_t_rg - row.clean_t_rg for row in rows],
    }
    clusters = [row.query_id for row in rows]
    tests: dict[str, dict[str, object]] = {}
    raw_p_values: dict[str, float] = {}
    for offset, name in enumerate(("H1", "H2", "H3")):
        permutation = paired_sign_flip_test(
            differences[name], permutations=permutations, seed=seed + offset
        )
        interval = cluster_bootstrap_median_ci(
            differences[name],
            clusters,
            replicates=bootstrap_replicates,
            alpha=alpha,
            seed=seed + 100 + offset,
        )
        raw_p_values[name] = permutation.p_value
        tests[name] = {
            "paired_median_difference": permutation.statistic,
            "raw_one_sided_p": permutation.p_value,
            "permutations": permutation.permutations,
            "cluster_bootstrap_95_ci": [interval.lower, interval.upper],
            "bootstrap_replicates": interval.replicates,
        }
    adjusted = holm_adjust(raw_p_values)
    for name in tests:
        tests[name]["holm_corrected_p"] = adjusted[name]
        tests[name]["passes"] = bool(
            adjusted[name] < alpha and tests[name]["cluster_bootstrap_95_ci"][0] > 0.0
        )
    family_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row.family] = family_counts.get(row.family, 0) + 1
    gate_pass = all(bool(test["passes"]) for test in tests.values())
    return {
        "schema_version": 1,
        "estimand": "paired median difference",
        "alternative": "greater than zero",
        "observations": len(rows),
        "query_clusters": len(set(clusters)),
        "family_counts": family_counts,
        "tests": tests,
        "gate_3": {
            "passed": gate_pass,
            "criterion": "all Holm-corrected p < alpha and all cluster-bootstrap CI lower > 0",
            "alpha": alpha,
            "reasons": [
                f"{name}: corrected_p={test['holm_corrected_p']:.6g}, "
                f"ci_lower={test['cluster_bootstrap_95_ci'][0]:.6g}, pass={test['passes']}"
                for name, test in tests.items()
            ],
        },
        "rows": [asdict(row) for row in rows],
    }


def write_mechanism_outputs(result: dict[str, object], json_path: Path, report_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    tests = result["tests"]
    lines = [
        "# RGRD-V0 Oracle Mechanism Report",
        "",
        f"Observations: `{result['observations']}`; query clusters: `{result['query_clusters']}`.",
        "",
        "Oracle boundaries were used only in this mechanism evaluator, after the label-free detector produced role maps.",
        "",
        "| Hypothesis | Paired median | One-sided p | Holm p | Cluster bootstrap 95% CI | Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for name in ("H1", "H2", "H3"):
        test = tests[name]
        interval = test["cluster_bootstrap_95_ci"]
        lines.append(
            f"| {name} | {test['paired_median_difference']:.6f} | "
            f"{test['raw_one_sided_p']:.6g} | {test['holm_corrected_p']:.6g} | "
            f"[{interval[0]:.6f}, {interval[1]:.6f}] | {test['passes']} |"
        )
    lines.extend(
        [
            "",
            f"Gate 3: **{'PASS' if result['gate_3']['passed'] else 'FAIL'}**.",
            "",
            *[f"- {reason}" for reason in result["gate_3"]["reasons"]],
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
