from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rgrd.attribution import aggregate_overlapping_span_effects
from rgrd.data import load_queries
from rgrd.experiments.detection_worker import _pipeline
from rgrd.ingestion import tokenize_with_offsets
from rgrd.intervention import build_span_views
from rgrd.pipeline.track_b import stable_event_hash
from rgrd.provenance import utc_now


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _config(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    pipeline = yaml.safe_load((root / "configs/pipeline/v0.yaml").read_text(encoding="utf-8"))
    datasets = yaml.safe_load((root / "configs/datasets.yaml").read_text(encoding="utf-8"))[
        "datasets"
    ]
    revisions = {role: str(value["revision"]) for role, value in pipeline["models"].items()}
    return pipeline, datasets, revisions


def _finite_positive_total(values: list[float], *, label: str) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{label} must be a non-empty effect vector")
    nonfinite = np.flatnonzero(~np.isfinite(array)).tolist()
    if nonfinite:
        raise FloatingPointError(f"{label} contains non-finite effects at indices {nonfinite}")
    return float(np.clip(array, 0.0, None).sum())


def _lineage_errors(selected: list[Any], generator_tokenizer: Any) -> list[str]:
    errors: list[str] = []
    for candidate in selected:
        source = candidate.source
        chunk = candidate.chunk
        source_slice = source.text[chunk.source_chars.start : chunk.source_chars.end]
        if source_slice != chunk.text:
            errors.append(f"{chunk.chunk_id}: source character projection differs from chunk text")
            continue
        source_tokens = tokenize_with_offsets(source.text, generator_tokenizer)
        token_slice = source_tokens[chunk.source_tokens.start : chunk.source_tokens.end]
        if not token_slice:
            errors.append(f"{chunk.chunk_id}: source token range is empty")
            continue
        if token_slice[0].char_start != chunk.source_chars.start:
            errors.append(f"{chunk.chunk_id}: source token start does not match source_chars")
        if token_slice[-1].char_end != chunk.source_chars.end:
            errors.append(f"{chunk.chunk_id}: source token end does not match source_chars")
        chunk_tokens = tokenize_with_offsets(chunk.text, generator_tokenizer)
        if len(chunk_tokens) != chunk.source_tokens.length:
            errors.append(
                f"{chunk.chunk_id}: retokenized chunk length differs from source token range"
            )
    return errors


def determinism_gate(
    root: Path,
    *,
    dataset: str,
    count: int,
    device: str,
    detector_commit: str,
) -> dict[str, Any]:
    pipeline_config, datasets, revisions = _config(root)
    dataset_config = datasets[dataset]
    queries = sorted(load_queries(Path(dataset_config["queries"])).items())[:count]
    if len(queries) < count:
        raise RuntimeError(f"Gate 2 requires {count} queries; found {len(queries)}")
    _, bundle, pipeline = _pipeline(root, dataset, device, revisions)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        for query_id, query in queries:
            repeats: list[dict[str, Any]] = []
            for _ in range(2):
                _, selected, frozen = pipeline.retrieve(query)
                event, layout = pipeline.build_detector_event(
                    query_id=query_id,
                    query=query,
                    dataset=dataset,
                    split=str(dataset_config["split"]),
                    selected=selected,
                    frozen=frozen,
                    detector_code_commit=detector_commit,
                    seed=int(pipeline_config["seed"]),
                )
                lineage = _lineage_errors(selected, pipeline.generator.tokenizer)
                teacher_score: float | None
                teacher_score_error: str | None = None
                try:
                    teacher_score = pipeline.generator.teacher_score(layout, event.shadow_answer)
                except FloatingPointError as exc:
                    teacher_score = None
                    teacher_score_error = str(exc)
                repeats.append(
                    {
                        "context_order": list(event.context_order),
                        "shadow_answer": event.shadow_answer,
                        "event_sha256": stable_event_hash(event),
                        "lineage_errors": lineage,
                        "teacher_score": teacher_score,
                        "teacher_score_error": teacher_score_error,
                    }
                )
            exact_top_k = repeats[0]["context_order"] == repeats[1]["context_order"]
            exact_shadow = repeats[0]["shadow_answer"] == repeats[1]["shadow_answer"]
            lineage_ok = not repeats[0]["lineage_errors"] and not repeats[1]["lineage_errors"]
            teacher_scores_finite = all(
                repeat["teacher_score"] is not None and np.isfinite(float(repeat["teacher_score"]))
                for repeat in repeats
            )
            if not exact_top_k:
                failures.append(f"{query_id}: final Top-K differs across repeats")
            if not exact_shadow:
                failures.append(f"{query_id}: shadow answer differs across repeats")
            if not lineage_ok:
                failures.append(f"{query_id}: source/chunk/token lineage validation failed")
            if not teacher_scores_finite:
                failures.append(f"{query_id}: teacher-forced shadow score is non-finite")
            rows.append(
                {
                    "query_id": query_id,
                    "exact_top_k": exact_top_k,
                    "exact_shadow_answer": exact_shadow,
                    "lineage_ok": lineage_ok,
                    "teacher_scores_finite": teacher_scores_finite,
                    "repeats": repeats,
                }
            )
    finally:
        bundle.close()
    result = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "dataset": dataset,
        "queries": len(rows),
        "repeats": 2,
        "gate_2": {
            "passed": not failures and len(rows) == count,
            "criteria": [
                "same-seed final Top-K is exactly identical",
                "deterministic shadow answer is exactly identical",
                "source-to-chunk-to-generator-token offsets validate",
                "teacher-forced shadow-answer scores are finite",
            ],
            "reasons": failures or [f"all {count} queries passed every exact check"],
        },
        "rows": rows,
    }
    _atomic_json(root / "artifacts/statistics/gate2_determinism.json", result)
    return result


def calibrate_noop(
    root: Path,
    *,
    dataset: str,
    count: int,
    device: str,
    detector_commit: str,
) -> dict[str, Any]:
    config, datasets, revisions = _config(root)
    dataset_config = datasets[dataset]
    queries = sorted(load_queries(Path(dataset_config["queries"])).items())[:count]
    _, bundle, pipeline = _pipeline(root, dataset, device, revisions)
    retrieval_values: list[float] = []
    generation_values: list[float] = []
    rows: list[dict[str, Any]] = []
    try:
        for query_id, query in queries:
            query_embedding, selected, frozen = pipeline.retrieve(query)
            event, layout = pipeline.build_detector_event(
                query_id=query_id,
                query=query,
                dataset=dataset,
                split=str(dataset_config["split"]),
                selected=selected,
                frozen=frozen,
                detector_code_commit=detector_commit,
                seed=int(config["seed"]),
            )
            for candidate in selected:
                offsets = [
                    (token.char_start, token.char_end)
                    for token in tokenize_with_offsets(
                        candidate.chunk.text, pipeline.generator.tokenizer
                    )
                ]
                spans = build_span_views(
                    offsets,
                    span_size=int(config["track_b"]["span_size"]),
                    offsets=tuple(config["track_b"]["span_offset_views"]),
                )
                retrieval_delta, generation_delta = pipeline.no_op_deltas(
                    query=query,
                    query_embedding=query_embedding,
                    candidate=candidate,
                    frozen=frozen,
                    layout=layout,
                    shadow_answer=event.shadow_answer,
                )
                atoms = aggregate_overlapping_span_effects(
                    (span.char_range, retrieval_delta, generation_delta) for span in spans
                )
                sample_label = f"no-op {query_id}/{candidate.chunk.chunk_id}"
                retrieval_total = _finite_positive_total(
                    [atom.retrieval_effect for atom in atoms],
                    label=f"{sample_label} retrieval",
                )
                generation_total = _finite_positive_total(
                    [atom.generation_effect for atom in atoms],
                    label=f"{sample_label} generation",
                )
                retrieval_values.append(retrieval_total)
                generation_values.append(generation_total)
                rows.append(
                    {
                        "query_id": query_id,
                        "chunk_id": candidate.chunk.chunk_id,
                        "atomic_spans": len(atoms),
                        "retrieval_total": retrieval_total,
                        "generation_total": generation_total,
                    }
                )
    finally:
        bundle.close()
    if len(rows) < count:
        raise RuntimeError("no-op calibration produced too few observations")
    result = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "protocol": "99th percentile of total positive effect under exact repeated no-op scoring",
        "quantile": 0.99,
        "observations": len(rows),
        "retrieval": max(1e-8, float(np.quantile(retrieval_values, 0.99))),
        "generation": max(1e-8, float(np.quantile(generation_values, 0.99))),
        "maximum_observed": {
            "retrieval": max(retrieval_values),
            "generation": max(generation_values),
        },
        "rows": rows,
    }
    _atomic_json(root / "artifacts/statistics/noop_noise_floor.json", result)
    return result


def attribution_pilot(
    root: Path,
    *,
    dataset: str,
    count: int,
    device: str,
    detector_commit: str,
) -> dict[str, Any]:
    config, datasets, revisions = _config(root)
    dataset_config = datasets[dataset]
    queries = sorted(load_queries(Path(dataset_config["queries"])).items())[:count]
    _, bundle, pipeline = _pipeline(root, dataset, device, revisions)
    rows: list[dict[str, Any]] = []
    try:
        for query_id, query in queries:
            query_embedding, selected, frozen = pipeline.retrieve(query)
            event, layout = pipeline.build_detector_event(
                query_id=query_id,
                query=query,
                dataset=dataset,
                split=str(dataset_config["split"]),
                selected=selected,
                frozen=frozen,
                detector_code_commit=detector_commit,
                seed=int(config["seed"]),
            )
            role = pipeline.score_detector_event(
                event,
                candidates=pipeline.runtime_candidates(selected),
                query_embedding=query_embedding,
                frozen=frozen,
                layout=layout,
            )
            if not 0.0 <= role.query_score <= 1.0:
                raise AssertionError(f"T_RG outside [0,1] for {query_id}")
            rows.append(
                {
                    "query_id": query_id,
                    "query_score": role.query_score,
                    "role_identifiable": any(chunk.role_identifiable for chunk in role.chunks),
                    "selected_for_generation": list(role.selected_for_generation),
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "t_rg": chunk.t_rg,
                            "role_identifiable": chunk.role_identifiable,
                            "retrieval_mass": chunk.retrieval_mass,
                            "generation_mass": chunk.generation_mass,
                        }
                        for chunk in role.chunks
                    ],
                }
            )
    finally:
        bundle.close()
    result = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "dataset": dataset,
        "queries": len(rows),
        "detector_inputs_label_free": True,
        "rows": rows,
    }
    _atomic_json(root / "artifacts/statistics/attribution_pilot.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="RGRD deterministic gates and pilots")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("determinism", "noop", "pilot"):
        command = subparsers.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--dataset", default="nq")
        command.add_argument("--count", type=int, required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--detector-commit", required=True)
    arguments = parser.parse_args()
    if arguments.command == "determinism":
        result = determinism_gate(
            arguments.root,
            dataset=arguments.dataset,
            count=arguments.count,
            device=arguments.device,
            detector_commit=arguments.detector_commit,
        )
        print(json.dumps(result["gate_2"], sort_keys=True))
        return 0 if result["gate_2"]["passed"] else 2
    if arguments.command == "noop":
        result = calibrate_noop(
            arguments.root,
            dataset=arguments.dataset,
            count=arguments.count,
            device=arguments.device,
            detector_commit=arguments.detector_commit,
        )
    else:
        result = attribution_pilot(
            arguments.root,
            dataset=arguments.dataset,
            count=arguments.count,
            device=arguments.device,
            detector_commit=arguments.detector_commit,
        )
    print(json.dumps({key: result[key] for key in result if key not in {"rows"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
