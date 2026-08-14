from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from rgrd.provenance import sha256_file, utc_now


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_current_result(
    root: Path,
    state: dict[str, Any],
    *,
    relative_path: str,
    phase: str,
    allow_failed_phase: bool = False,
) -> dict[str, Any] | None:
    phase_status = (state.get("phases", {}).get(phase) or {}).get("status")
    allowed = {"passed", "failed"} if allow_failed_phase else {"passed"}
    if phase_status not in allowed:
        return None
    value = _load_json(root / relative_path)
    if not value:
        return None
    expected_commit = state.get("detector_code_commit")
    expected_config = state.get("pipeline_config_sha256")
    if expected_commit and value.get("detector_code_commit") != expected_commit:
        return None
    if expected_config and value.get("pipeline_config_sha256") != expected_config:
        return None
    return value


def _format_rate(value: Any) -> str:
    try:
        return f"{float(value):.3%}"
    except (TypeError, ValueError):
        return "not estimable"


def _gate_line(name: str, value: Any) -> str:
    if not isinstance(value, dict):
        return f"- {name}: not decided"
    passed = value.get("passed")
    if passed is None and "status" in value:
        passed = value["status"] == "PASS"
    return f"- {name}: {'PASS' if passed else 'FAIL'}"


def write_pipeline_report(root: Path, state: dict[str, Any]) -> Path:
    pipeline = yaml.safe_load((root / "configs/pipeline/v0.yaml").read_text(encoding="utf-8"))
    track = pipeline["track_b"]
    model_lock = root / "configs/models.lock.yaml"
    upstream_lock = root / "third_party/manifest.lock.yaml"
    lines = [
        "# RGRD-V0 Pipeline Report",
        "",
        f"Generated: `{utc_now()}`",
        f"Run ID: `{state.get('run_id', 'unknown')}`; status: **{str(state.get('status', 'unknown')).upper()}**.",
        "",
        "## Fixed Track-B pipeline",
        "",
        f"- Generator-token chunks: {track['chunk_size']} tokens, overlap {track['chunk_overlap']}.",
        f"- Retrieval: exact `{track['dense_index']}`, dense Top-{track['dense_top_b']}, reranked Top-{track['rerank_top_k']}.",
        f"- Interventions: {track['span_size']}-token spans at offsets {track['span_offset_views']}.",
        f"- Generation: temperature={track['generator_temperature']}, do_sample={track['generator_do_sample']}, max_new_tokens={track['generator_max_new_tokens']}.",
        "- Decision point: after reranking and deterministic shadow generation, before answer release.",
        "- Runtime detector view excludes attack labels, target/gold answers, and all oracle ranges.",
        "",
        "## Pinned models",
        "",
    ]
    for role, model in pipeline["models"].items():
        lines.append(f"- {role}: `{model['name']}` at revision `{model['revision']}`.")
    lines.extend(
        [
            "",
            "## Provenance locks",
            "",
            f"- Model lock: `{model_lock}` (SHA-256 `{sha256_file(model_lock) if model_lock.is_file() else 'missing'}`).",
            f"- Upstream lock: `{upstream_lock}` (SHA-256 `{sha256_file(upstream_lock) if upstream_lock.is_file() else 'missing'}`).",
            "- `/home/jkl/RAG_attack` was used as read-only attack evidence.",
            "- Each upstream paper project has a separate Conda environment declaration.",
            "",
            "## Gate decisions",
            "",
        ]
    )
    gates = state.get("gates", {})
    for name in ("gate_1", "gate_2", "gate_3", "gate_robustness"):
        lines.append(_gate_line(name, gates.get(name)))
        decision = gates.get(name) or {}
        for reason in decision.get("reasons", []):
            lines.append(f"  - {reason}")
    lines.extend(
        ["", "## Phase ledger", "", "| Phase | Status | Progress / error |", "|---|---|---|"]
    )
    for name, phase in state.get("phases", {}).items():
        progress = phase.get("progress") or {}
        detail = json.dumps(progress, ensure_ascii=False, sort_keys=True)
        if phase.get("error"):
            detail = f"{detail}; error={phase['error']}"
        lines.append(f"| {name} | {phase.get('status', 'unknown')} | `{detail}` |")
    output = root / "artifacts/reports/pipeline_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _placeholder(path: Path, title: str, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {title}\n\nNot estimable: {reason}. No numerical result is asserted.\n",
        encoding="utf-8",
    )


def _conclusion(state: dict[str, Any], joint: dict[str, Any] | None) -> str:
    gates = state.get("gates", {})
    for name in ("gate_1", "gate_2", "gate_3", "gate_robustness"):
        decision = gates.get(name)
        if decision is not None and not bool(decision.get("passed")):
            if name in {"gate_1", "gate_2"}:
                return f"Terminate this V0 run at {name}; the engineering evidence is insufficient for a mechanism claim."
            if name == "gate_3":
                return "Terminate the confirmatory detector path: the preregistered role-decoupling mechanism was not established."
            return "Narrow or redesign the intervention before further confirmatory detector claims; donor robustness failed."
    if joint:
        judgments = joint.get("judgments", {})
        if judgments.get("role_separation_weakened_by_joint_optimization"):
            return "Narrow the theory to explicitly modular attacks; continue only as a scoped structural screener."
        if judgments.get("role_separation_persists"):
            return "Continue with an independently preregistered replication; V0 supports a scoped structural screener, not a universal malicious-content classifier."
    if state.get("status") == "passed":
        return "Continue cautiously as a scoped structural screener; the fused-attack boundary remains not estimable."
    return "No continue/narrow/terminate judgment is estimable until the active gate-controlled run reaches a terminal state."


def write_final_report(root: Path, state: dict[str, Any]) -> Path:
    audit_phase = (state.get("phases", {}).get("phase_1_attack_audit") or {}).get("status")
    audit = (
        _load_json(root / "artifacts/audit/reproduction_audit.json")
        if audit_phase in {"passed", "failed"}
        else None
    )
    mechanism = _load_current_result(
        root,
        state,
        relative_path="artifacts/statistics/mechanism_tests.json",
        phase="phase_5_oracle_mechanism",
        allow_failed_phase=True,
    )
    detection = _load_current_result(
        root,
        state,
        relative_path="artifacts/statistics/detection.json",
        phase="phase_7_conformal_detection",
    )
    hard = _load_current_result(
        root,
        state,
        relative_path="artifacts/statistics/hard_benign.json",
        phase="phase_8_hard_benign",
    )
    joint = _load_current_result(
        root,
        state,
        relative_path="artifacts/statistics/joint_gcg_boundary.json",
        phase="phase_9_joint_gcg_boundary",
    )
    stopped = state.get("status") in {"failed", "stopped"}
    stop_reason = (
        "; ".join(state.get("errors", [])) or f"run status is {state.get('status', 'unknown')}"
    )
    if mechanism is None:
        _placeholder(
            root / "artifacts/reports/mechanism_report.md",
            "RGRD-V0 Oracle Mechanism Report",
            stop_reason,
        )
    if detection is None:
        _placeholder(
            root / "artifacts/reports/detection_report.md",
            "RGRD-V0 Conformal Detection Report",
            stop_reason,
        )
    if hard is None:
        _placeholder(
            root / "artifacts/reports/hard_benign_report.md",
            "RGRD-V0 Hard-Benign False-Positive Report",
            stop_reason,
        )
    if joint is None:
        _placeholder(
            root / "artifacts/reports/joint_gcg_boundary_report.md",
            "RGRD-V0 Joint-GCG Boundary Report",
            stop_reason,
        )

    if audit:
        poison_status = audit.get("components", {}).get("poisonedrag", {}).get("status", "unknown")
        phantom_status = audit.get("components", {}).get("phantom", {}).get("status", "unknown")
        gate1 = audit.get("gate_1", {}).get("status", "unknown")
        answer1 = f"PoisonedRAG={poison_status}, Phantom={phantom_status}, Gate 1={gate1}."
    else:
        answer1 = "Not estimable: reproduction audit did not complete."

    tests = mechanism.get("tests", {}) if mechanism else {}
    gate3 = mechanism.get("gate_3", {}) if mechanism else {}
    answer2 = (
        f"H1 pass={tests['H1']['passes']}, paired median={tests['H1']['paired_median_difference']:.6f}, "
        f"Holm p={tests['H1']['holm_corrected_p']:.6g}."
        if "H1" in tests
        else "Not estimable: Gate-3 H1 was not completed."
    )
    answer3 = (
        f"H2 pass={tests['H2']['passes']}, paired median={tests['H2']['paired_median_difference']:.6f}, "
        f"Holm p={tests['H2']['holm_corrected_p']:.6g}."
        if "H2" in tests
        else "Not estimable: Gate-3 H2 was not completed."
    )
    answer4 = (
        f"H3 pass={tests['H3']['passes']}, paired median={tests['H3']['paired_median_difference']:.6f}, "
        f"Holm p={tests['H3']['holm_corrected_p']:.6g}; Gate 3 pass={gate3.get('passed')}."
        if "H3" in tests
        else "Not estimable: poison-versus-matched-clean H3 was not completed."
    )

    if detection:
        clean = detection.get("ordinary_clean", {})
        family_parts = [
            f"{family}={_format_rate(value.get('rate'))}"
            for family, value in detection.get("per_family", {}).items()
        ]
        answer5 = (
            f"Ordinary-clean FPR={_format_rate(clean.get('rate'))}; per-family TPR: "
            + (", ".join(family_parts) or "not estimable")
            + "."
        )
    else:
        answer5 = "Not estimable: conformal detection was not reached or did not finish."

    if hard:
        groups = hard.get("groups", {})
        answer6 = (
            "; ".join(
                f"{name}: FPR={_format_rate(value.get('fpr'))}, n={value.get('n')}"
                for name, value in groups.items()
            )
            or "Not estimable: no hard-benign groups completed."
        )
    else:
        answer6 = "Not estimable: hard-benign evaluation was not reached or did not finish."

    if joint:
        judgments = joint.get("judgments", {})
        answer7 = (
            f"weakened={judgments.get('role_separation_weakened_by_joint_optimization')}, "
            f"persists={judgments.get('role_separation_persists')}, "
            f"TPR@5%={_format_rate(joint.get('detection_at_query_fpr_5pct', {}).get('rate'))}."
        )
    else:
        answer7 = "Not estimable: the Joint-GCG boundary phase was not reached or did not finish."
    answer8 = _conclusion(state, joint)
    lines = [
        "# RGRD-V0 Final Report",
        "",
        f"Generated: `{utc_now()}`",
        f"Run status: **{str(state.get('status', 'unknown')).upper()}**.",
        "",
        "RGRD is evaluated here as a structural anomaly screener at the post-reranking/pre-release decision point, not as a universal malicious-content classifier.",
        "",
        "## Eight required decisions",
        "",
        f"1. **Are PoisonedRAG/Phantom reproductions credible?** {answer1}",
        f"2. **Is the oracle anchor retrieval-dominant?** {answer2}",
        f"3. **Is the oracle payload generation-dominant?** {answer3}",
        f"4. **Is poison T_RG higher than matched clean?** {answer4}",
        f"5. **What is detection at 5% query-FPR?** {answer5}",
        f"6. **Which benign structures cause false positives?** {answer6}",
        f"7. **Does Joint-GCG weaken role separation?** {answer7}",
        f"8. **Continue, narrow, or terminate?** {answer8}",
        "",
        "## Gate discipline",
        "",
        "No result from a phase after a failed gate is promoted as confirmatory. Missing values are reported as not estimable rather than imputed.",
    ]
    if stopped:
        lines.extend(["", "## Terminal reason", "", stop_reason])
    output = root / "artifacts/reports/V0_final_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_all(root: Path, state: dict[str, Any]) -> tuple[Path, Path]:
    return write_pipeline_report(root, state), write_final_report(root, state)
