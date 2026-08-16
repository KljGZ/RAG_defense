from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from rgrd.pipeline.state import GateStatus
from rgrd.provenance import utc_now
from rgrd.statistics.tests import holm_adjust
from rgrd.statistics.v01 import (
    exact_one_sided_sign_test,
    outcome_association,
    outcome_label_permutation_test,
    query_bootstrap_association_ci,
    query_bootstrap_median_ci,
)


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


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(input_dir.glob("**/*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                if sample_id in seen:
                    raise RuntimeError(f"duplicate V0.1 sample_id {sample_id} at {path}:{line_number}")
                seen.add(sample_id)
                rows.append(row)
    if not rows:
        raise RuntimeError("V0.1 combine received no worker rows")
    provenance_fields = (
        "schema_version",
        "protocol_id",
        "detector_code_commit",
        "model_revisions",
        "pipeline_config_sha256",
        "experiment_config_sha256",
    )
    reference = {field: rows[0].get(field) for field in provenance_fields}
    for index, row in enumerate(rows[1:], 1):
        mismatches = [field for field in provenance_fields if row.get(field) != reference[field]]
        if mismatches:
            raise RuntimeError(f"mixed V0.1 provenance at row {index}: {mismatches}")
    return rows


def _attrition(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = (
        "source_ranges_valid",
        "anchor_payload_cochunk",
        "gold_aliases_valid",
        "target_distinct_from_gold",
        "actual_retrieval_hit",
        "natural_end_to_end_success",
        "forced_context_success",
        "mechanism_scores_finite",
    )
    result = dict(manifest["selection"])
    result["worker_rows"] = len(rows)
    result["eligible_rows"] = sum(row["status"] == "ELIGIBLE" for row in rows)
    result["ineligible_rows"] = sum(row["status"] == "INELIGIBLE" for row in rows)
    result["checks"] = {
        check: sum(bool(row.get("attrition_checks", {}).get(check)) for row in rows)
        for check in checks
    }
    result["ineligible_reasons"] = {}
    for row in rows:
        if row["status"] != "INELIGIBLE":
            continue
        reason = str(row.get("ineligible_stage", "unknown"))
        result["ineligible_reasons"][reason] = result["ineligible_reasons"].get(reason, 0) + 1
    return result


def _primary_test(
    rows: list[dict[str, Any]],
    field: str,
    *,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    values = [float(row["primary_estimate"][field]) for row in rows]
    sign = exact_one_sided_sign_test(values)
    interval = query_bootstrap_median_ci(
        values, replicates=bootstrap_replicates, seed=seed
    )
    return {
        "n": len(values),
        "positive": sign.positive,
        "negative": sign.negative,
        "zero": sign.zero,
        "positive_proportion": sign.positive_proportion,
        "median": interval.estimate,
        "bootstrap_ci_95": [interval.lower, interval.upper],
        "raw_p_value": sign.p_value,
        "holm_adjusted_p_value": None,
        "passed": None,
    }


def _association(
    rows: list[dict[str, Any]],
    *,
    field: str,
    outcome: str,
    seed: int,
    permutations: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    values = [float(row["primary_estimate"][field]) for row in rows]
    labels = [bool(row["outcomes"][outcome]) for row in rows]
    try:
        descriptive = outcome_association(values, labels)
        permutation = outcome_label_permutation_test(
            values, labels, permutations=permutations, seed=seed
        )
        interval = query_bootstrap_association_ci(
            values,
            labels,
            replicates=bootstrap_replicates,
            seed=seed + 1,
        )
    except ValueError as exc:
        return {"status": GateStatus.NOT_ESTIMABLE.value, "reason": str(exc)}
    return {
        "status": "ESTIMATED_SECONDARY",
        "n": len(values),
        "successes": descriptive.successes,
        "failures": descriptive.failures,
        "difference_in_medians": descriptive.difference_in_medians,
        "bootstrap_ci_95": [interval.lower, interval.upper],
        "cliffs_delta": descriptive.cliffs_delta,
        "one_sided_permutation_p_value": permutation.p_value,
        "inferential_role": "secondary association only; not part of the mechanism gate",
    }


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}g}"


def _report(result: dict[str, Any]) -> str:
    primary_rows = []
    for name, value in result["primary_tests"].items():
        interval = value.get("bootstrap_ci_95") or [None, None]
        primary_rows.append(
            "| {name} | {n} | {prop} | {median} | [{low}, {high}] | {raw} | {holm} | {passed} |".format(
                name=name,
                n=value.get("n", 0),
                prop=_format_number(value.get("positive_proportion")),
                median=_format_number(value.get("median")),
                low=_format_number(interval[0]),
                high=_format_number(interval[1]),
                raw=_format_number(value.get("raw_p_value")),
                holm=_format_number(value.get("holm_adjusted_p_value")),
                passed=value.get("passed", "NA"),
            )
        )
    attrition_rows = []
    for family, value in result["attrition"].items():
        checks = value["checks"]
        attrition_rows.append(
            f"| {family} | {value['loaded_samples']} | {value['unique_query_ids']} | "
            f"{value['canonical_samples']} | {checks['source_ranges_valid']} | "
            f"{checks['anchor_payload_cochunk']} | {checks['gold_aliases_valid']} | "
            f"{checks['target_distinct_from_gold']} | {checks['mechanism_scores_finite']} |"
        )
    association_rows = []
    for name, value in result["outcome_associations"].items():
        association_rows.append(
            f"| {name} | {value['status']} | {value.get('successes', 'NA')} | "
            f"{value.get('failures', 'NA')} | {_format_number(value.get('difference_in_medians'))} | "
            f"{_format_number(value.get('cliffs_delta'))} | "
            f"{_format_number(value.get('one_sided_permutation_p_value'))} |"
        )
    robustness_rows = []
    for family, value in result["mask_direction_robustness"]["families"].items():
        robustness_rows.append(
            f"| {family} | {value['n']} | {_format_number(value['retrieval_mean_agreement'])} | "
            f"{_format_number(value['generation_mean_agreement'])} | {value['passed']} |"
        )
    conclusion = {
        GateStatus.PASS.value: "The preregistered PoisonedRAG-B/W Oracle mechanism claim passed V0.1.",
        GateStatus.FAIL_TESTED.value: "The preregistered Oracle mechanism claim was tested and did not pass every corrected criterion.",
        GateStatus.NOT_ESTIMABLE.value: "The preregistered Oracle mechanism claim remains not estimable under V0.1.",
        GateStatus.SKIPPED.value: "The V0.1 mechanism audit was skipped.",
    }[result["gate_v01"]["status"]]
    return f"""# RGRD-V0.1 Oracle mechanism-audit report

## Decision

**Gate V0.1: {result['gate_v01']['status']}**

{conclusion}

This is a mechanism audit, not a deployable detector result. It does not establish a
runtime threshold, false-positive rate, or generalization beyond the frozen NQ
PoisonedRAG-B/W scope.

## Correction to V0

Legacy Gate 3 was not estimable because the preregistered family quotas were not
reached. H1--H3 were not executed. The confirmatory mechanism claim was therefore
inconclusive; it was not a tested failure. Legacy `T_RG` remains exploratory because
positive clipping and separate normalization discard absolute generation scale.

## Attrition ledger

| Family | Loaded samples | Unique queries | Canonical | Valid ranges | A/P co-chunk | Valid gold | Target distinct | Finite mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(attrition_rows)}

Canonicalization selected the lexicographically smallest `sample_id` per `query_id`
before observing retrieval, natural generation, forced generation, or any mechanism
score. Retrieval/generation outcomes were retained as labels and never as filters.

## Primary family-specific tests

| Test | n | Positive proportion | Median | Query-bootstrap 95% CI | Exact sign p | Holm p | Pass |
|---|---:|---:|---:|---|---:|---:|---|
{chr(10).join(primary_rows)}

Each test requires both Holm-adjusted p < 0.05 and a strictly positive lower
query-bootstrap confidence bound. `C_R = phi_A^R - phi_P^R`; `C_G = phi_P^G - phi_A^G`.
No clipping, per-channel normalization, cross-channel comparison, or combined score
was used.

## Attention-hole direction robustness

**Robustness gate: {result['mask_direction_robustness']['status']}**

| Family | Mask-valid n | Retrieval agreement | Generation agreement | Pass |
|---|---:|---:|---:|---|
{chr(10).join(robustness_rows)}

The attention-hole analysis is direction-only. The eight same-token-length benign
donor pairs remain the primary estimator.

## Outcome association (secondary)

| Contrast / outcome | Status | Outcome=1 | Outcome=0 | Median difference | Cliff's delta | Permutation p |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(association_rows)}

These family-specific associations test whether `C_R` tracks actual reranker survival
and whether `C_G` tracks fixed-position forced-context attack success. They do not
replace the primary mechanism tests.

## Frozen contracts and claim boundary

- Unit: one final chunk per independent query.
- Context: clean Top-K with the poison fixed at zero-based index 1 for generation attribution.
- Models: frozen Contriever, MiniLM cross-encoder, and Qwen2.5-7B-Instruct.
- Generator: explicit BF16, eager attention, strict deterministic algorithms, and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- GPU boundary: physical GPUs 4--7 only; physical GPUs 0--3 excluded.
- Excluded claims: Phantom, Joint-GCG, multi-chunk attacks, latent backdoors,
  GraphRAG, multimodal RAG, and other attack families.

Generated at `{result['generated_at']}` from commit
`{result['provenance']['detector_code_commit']}`.
"""


def combine(
    root: Path,
    *,
    input_dir: Path,
    manifests: dict[str, Path],
    output_json: Path,
    output_report: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(
        (root / "configs/experiments/v0_1_preregistration.yaml").read_text(encoding="utf-8")
    )
    pipeline = yaml.safe_load(
        (root / "configs/pipeline/v0_1.yaml").read_text(encoding="utf-8")
    )
    rows = _load_rows(input_dir)
    families = list(config["scope"]["active_attack_families"])
    manifest_values = {
        family: json.loads(manifests[family].read_text(encoding="utf-8"))
        for family in families
    }
    rows_by_family = {
        family: [row for row in rows if row["family"] == family] for family in families
    }
    for family in families:
        expected = {row["sample_id"] for row in manifest_values[family]["rows"]}
        observed = {row["sample_id"] for row in rows_by_family[family]}
        if expected != observed:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise RuntimeError(
                f"V0.1 family {family} does not cover its canonical manifest; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
    attrition = {
        family: _attrition(manifest_values[family], rows_by_family[family])
        for family in families
    }
    identifiable = {
        family: [
            row
            for row in rows_by_family[family]
            if row["status"] == "ELIGIBLE"
            and bool(row["primary_estimate"]["role_identifiable"])
        ]
        for family in families
    }
    minimum = int(config["sampling"]["minimum_queries_per_family"])
    all_estimable = all(len(identifiable[family]) >= minimum for family in families)
    bootstrap_replicates = int(config["statistics"]["bootstrap"]["replicates"])
    seed = int(pipeline["seed"])
    primary_tests: dict[str, dict[str, Any]] = {}
    if all_estimable:
        raw_p: dict[str, float] = {}
        for family_index, family in enumerate(families):
            for field, label in (
                ("c_retrieval", "C_R"),
                ("c_generation", "C_G"),
            ):
                name = f"{family}/{label}"
                value = _primary_test(
                    identifiable[family],
                    field,
                    seed=seed + family_index * 10 + (0 if field == "c_retrieval" else 1),
                    bootstrap_replicates=bootstrap_replicates,
                )
                primary_tests[name] = value
                raw_p[name] = float(value["raw_p_value"])
        adjusted = holm_adjust(raw_p)
        for name, value in primary_tests.items():
            value["holm_adjusted_p_value"] = adjusted[name]
            value["passed"] = bool(
                adjusted[name] < float(config["statistics"]["alpha"])
                and float(value["bootstrap_ci_95"][0]) > 0.0
            )
        gate_status = (
            GateStatus.PASS
            if all(value["passed"] for value in primary_tests.values())
            else GateStatus.FAIL_TESTED
        )
        gate_reasons = [
            "all four family-specific corrected tests passed"
            if gate_status == GateStatus.PASS
            else "one or more corrected family-specific tests failed"
        ]
    else:
        for family in families:
            for label in ("C_R", "C_G"):
                primary_tests[f"{family}/{label}"] = {
                    "n": len(identifiable[family]),
                    "status": GateStatus.NOT_ESTIMABLE.value,
                    "reason": f"requires >= {minimum} identifiable independent queries",
                }
        gate_status = GateStatus.NOT_ESTIMABLE
        gate_reasons = [
            f"{family}: {len(identifiable[family])}/{minimum} identifiable queries"
            for family in families
            if len(identifiable[family]) < minimum
        ]

    robustness_families: dict[str, Any] = {}
    robustness_estimable = True
    robustness_passed = True
    threshold = float(pipeline["oracle_mechanism"]["direction_agreement_minimum"])
    for family in families:
        values = [
            row
            for row in identifiable[family]
            if row["mask_robustness"].get("valid")
            and row["direction_agreement"]["retrieval"] is not None
            and row["direction_agreement"]["generation"] is not None
            and np.isfinite(float(row["direction_agreement"]["retrieval"]))
            and np.isfinite(float(row["direction_agreement"]["generation"]))
        ]
        if len(values) < minimum:
            robustness_estimable = False
        retrieval_agreement = (
            float(np.mean([row["direction_agreement"]["retrieval"] for row in values]))
            if values
            else float("nan")
        )
        generation_agreement = (
            float(np.mean([row["direction_agreement"]["generation"] for row in values]))
            if values
            else float("nan")
        )
        passed = bool(
            len(values) >= minimum
            and retrieval_agreement >= threshold
            and generation_agreement >= threshold
        )
        robustness_passed &= passed
        robustness_families[family] = {
            "n": len(values),
            "retrieval_mean_agreement": retrieval_agreement,
            "generation_mean_agreement": generation_agreement,
            "minimum": threshold,
            "passed": passed,
        }
    robustness_status = (
        GateStatus.NOT_ESTIMABLE
        if not robustness_estimable
        else (GateStatus.PASS if robustness_passed else GateStatus.FAIL_TESTED)
    )

    associations: dict[str, Any] = {}
    for family_index, family in enumerate(families):
        for offset, (field, outcome, label) in enumerate(
            (
                ("c_retrieval", "actual_retrieval_hit", "C_R~R"),
                ("c_generation", "forced_context_success", "C_G~G_f"),
            )
        ):
            associations[f"{family}/{label}"] = _association(
                identifiable[family],
                field=field,
                outcome=outcome,
                seed=seed + 100 + family_index * 10 + offset,
                permutations=10000,
                bootstrap_replicates=bootstrap_replicates,
            )
    strata = {
        family: {
            stratum: sum(
                row["status"] == "ELIGIBLE" and row["outcomes"]["stratum"] == stratum
                for row in rows_by_family[family]
            )
            for stratum in ("RG", "R_only", "G_only", "neither")
        }
        for family in families
    }
    result = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "generated_at": utc_now(),
        "gate_v01": {
            "status": gate_status.value,
            "reasons": gate_reasons,
            "minimum_queries_per_family": minimum,
        },
        "attrition": attrition,
        "identifiable_queries": {
            family: len(identifiable[family]) for family in families
        },
        "outcome_strata": strata,
        "primary_tests": primary_tests,
        "mask_direction_robustness": {
            "status": robustness_status.value,
            "families": robustness_families,
        },
        "outcome_associations": associations,
        "provenance": {
            key: rows[0][key]
            for key in (
                "detector_code_commit",
                "model_revisions",
                "pipeline_config_sha256",
                "experiment_config_sha256",
            )
        },
        "claim_boundary": config["claim"],
    }
    safe = _json_safe(result)
    _atomic_json(output_json, safe)
    _atomic_text(output_report, _report(safe))
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine RGRD-V0.1 mechanism audit")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    arguments = parser.parse_args()
    manifests = {
        family: Path(path) for family, path in (item.split("=", 1) for item in arguments.manifest)
    }
    result = combine(
        arguments.root.resolve(),
        input_dir=arguments.input_dir.resolve(),
        manifests=manifests,
        output_json=arguments.output_json.resolve(),
        output_report=arguments.output_report.resolve(),
    )
    print(json.dumps(result["gate_v01"], ensure_ascii=False, sort_keys=True))
    # A tested scientific failure is a valid completed analysis, not a CLI crash.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
