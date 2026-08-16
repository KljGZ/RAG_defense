from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rgrd.attacks import attack_succeeds
from rgrd.data import load_qrels
from rgrd.experiments.resume import prepare_jsonl_resume, v01_event_provenance
from rgrd.indexing import ExactIndexBundle
from rgrd.models import CausalAnswerGenerator, CrossEncoderReranker, DenseRetriever
from rgrd.pipeline.track_b import TrackBPipeline
from rgrd.v01.donors import (
    DeterministicDonorSampler,
)
from rgrd.v01.engine import (
    candidate_context,
    deterministic_generation,
    donor_interventions,
    forced_context,
    generation_margin,
    mask_retrieval_values,
    oracle_cochunk,
    prepare_query,
    retrieval_values_for_interventions,
    select_fixed_gold_alias,
)
from rgrd.v01.protocol import canonical_per_query, valid_distinct_answers
from rgrd.v01.samples import ACTIVE_FAMILIES, load_v01_samples
from rgrd.v01.shapley import (
    CoalitionValues,
    aggregate_oracle_replicates,
    direction_agreement,
    oracle_contrast,
)


_ALLOWED_PHYSICAL_GPUS = [4, 5, 6, 7]
_EXCLUDED_PHYSICAL_GPUS = [0, 1, 2, 3]


def _assert_gpu_contract(config: dict[str, Any]) -> int:
    admission = config["runtime"]["gpu_admission"]
    if list(admission["allowed_physical_gpus"]) != _ALLOWED_PHYSICAL_GPUS:
        raise RuntimeError("V0.1 allowed_physical_gpus must be exactly [4,5,6,7]")
    if list(admission["excluded_physical_gpus"]) != _EXCLUDED_PHYSICAL_GPUS:
        raise RuntimeError("V0.1 excluded_physical_gpus must be exactly [0,1,2,3]")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    values = [value.strip() for value in visible.split(",") if value.strip()]
    if len(values) != 1 or not values[0].isdigit():
        raise RuntimeError("V0.1 worker requires exactly one visible physical GPU")
    physical = int(values[0])
    if physical not in _ALLOWED_PHYSICAL_GPUS:
        raise RuntimeError(f"physical GPU {physical} violates the V0.1 4--7-only contract")
    return physical


def _pipeline(
    root: Path,
    dataset: str,
    device: str,
    config: dict[str, Any],
    revisions: dict[str, str],
) -> tuple[ExactIndexBundle, TrackBPipeline]:
    models = config["models"]
    track = config["track_b"]
    bundle = ExactIndexBundle(root / f"artifacts/indexes/{dataset}/manifest.json")
    pipeline = TrackBPipeline(
        index=bundle,
        retriever=DenseRetriever(Path(models["retriever"]["local_path"]), device=device),
        reranker=CrossEncoderReranker(Path(models["reranker"]["local_path"]), device=device),
        generator=CausalAnswerGenerator(
            Path(models["generator"]["local_path"]),
            device=device,
            max_new_tokens=int(track["generator_max_new_tokens"]),
            seed=int(config["seed"]),
            dtype=str(models["generator"]["dtype"]),
            attention_implementation=str(models["generator"]["attention_implementation"]),
        ),
        dense_top_b=int(track["dense_top_b"]),
        rerank_top_k=int(track["rerank_top_k"]),
        retrieval_noise_floor=1e-8,
        generation_noise_floor=1e-8,
        model_revisions=revisions,
    )
    return bundle, pipeline


def _coalition(values: dict[str, dict[str, float]]) -> CoalitionValues:
    return CoalitionValues(
        empty=float(values["empty"]["value"]),
        anchor=float(values["anchor"]["value"]),
        payload=float(values["payload"]["value"]),
        both=float(values["both"]["value"]),
    )


def _outcome_stratum(retrieval: bool, forced_generation: bool) -> str:
    if retrieval and forced_generation:
        return "RG"
    if retrieval:
        return "R_only"
    if forced_generation:
        return "G_only"
    return "neither"


def _ineligible_row(
    provenance: dict[str, Any],
    *,
    sample: object,
    stage: str,
    reason: str,
    checks: dict[str, bool],
    physical_gpu: int,
) -> dict[str, Any]:
    return {
        **provenance,
        "sample_id": sample.sample_id,
        "query_id": sample.query_id,
        "family": sample.family,
        "dataset": sample.dataset,
        "status": "INELIGIBLE",
        "ineligible_stage": stage,
        "ineligible_reason": reason,
        "attrition_checks": checks,
        "physical_gpu": physical_gpu,
    }


def _write_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run_worker(args: argparse.Namespace) -> dict[str, int]:
    root = args.root.resolve()
    config = yaml.safe_load((root / "configs/pipeline/v0_1.yaml").read_text(encoding="utf-8"))
    protocol = yaml.safe_load(
        (root / "configs/experiments/v0_1_preregistration.yaml").read_text(encoding="utf-8")
    )
    physical_gpu = _assert_gpu_contract(config)
    if args.family not in protocol["scope"]["active_attack_families"]:
        raise ValueError("worker family is outside the preregistered V0.1 scope")
    datasets = yaml.safe_load((root / "configs/datasets.yaml").read_text(encoding="utf-8"))[
        "datasets"
    ]
    dataset_config = datasets[args.dataset]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["family"] != args.family or manifest["dataset"] != args.dataset:
        raise ValueError("selection manifest family/dataset mismatch")
    loaded = load_v01_samples(root, args.attack_root, args.family, dataset=args.dataset)
    canonical, audit = canonical_per_query(loaded)
    expected_rows = [
        {"query_id": sample.query_id, "sample_id": sample.sample_id} for sample in canonical
    ]
    if manifest["rows"] != expected_rows or manifest["selection"] != asdict(audit):
        raise RuntimeError(
            "selection manifest no longer matches outcome-independent canonicalization"
        )
    samples = {sample.sample_id: sample for sample in canonical}
    tasks = [
        samples[row["sample_id"]]
        for index, row in enumerate(manifest["rows"])
        if index % args.shards == args.shard_id
    ]
    revisions = {role: str(value["revision"]) for role, value in config["models"].items()}
    provenance = v01_event_provenance(
        root,
        detector_code_commit=args.detector_commit,
        model_revisions=revisions,
        selection_manifest=args.manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {
        key[0]
        for key in prepare_jsonl_resume(
            args.output,
            key_fields=("sample_id",),
            expected_provenance=provenance,
        )
    }
    bundle, pipeline = _pipeline(root, args.dataset, args.device, config, revisions)
    donor_config = config["oracle_mechanism"]
    ownership_rule = "maximum_character_overlap_tie_to_earlier_A_then_P"
    if donor_config.get("token_ownership_rule") != ownership_rule:
        raise RuntimeError(f"V0.1 token ownership must be {ownership_rule}")
    sampler = DeterministicDonorSampler(
        bundle,
        pipeline.generator.tokenizer,
        seed=int(donor_config["donor_seed"]),
    )
    qrels = load_qrels(Path(dataset_config["qrels"]))
    counts = {
        "tasks": len(tasks),
        "already_complete": len(completed),
        "eligible": 0,
        "ineligible": 0,
    }
    try:
        with args.output.open("a", encoding="utf-8") as handle:
            for sample in tasks:
                if sample.sample_id in completed:
                    continue
                checks: dict[str, bool] = {
                    "source_ranges_valid": False,
                    "anchor_payload_cochunk": False,
                    "gold_aliases_valid": False,
                    "target_distinct_from_gold": False,
                    "oracle_token_partitions_valid": False,
                    "donor_pairs_valid": False,
                    "actual_retrieval_hit": False,
                    "natural_end_to_end_success": False,
                    "forced_context_success": False,
                    "mechanism_scores_finite": False,
                }
                try:
                    poison_chunk, all_poison_chunks = oracle_cochunk(
                        sample,
                        pipeline,
                        chunk_size=int(config["track_b"]["chunk_size"]),
                        overlap=int(config["track_b"]["chunk_overlap"]),
                    )
                    checks["source_ranges_valid"] = True
                    checks["anchor_payload_cochunk"] = True
                except ValueError as exc:
                    message = str(exc)
                    stage = (
                        "anchor_payload_cochunk" if "co-occur" in message else "source_ranges_valid"
                    )
                    row = _ineligible_row(
                        provenance,
                        sample=sample,
                        stage=stage,
                        reason=message,
                        checks=checks,
                        physical_gpu=physical_gpu,
                    )
                    _write_row(handle, row)
                    counts["ineligible"] += 1
                    continue
                try:
                    gold_aliases = valid_distinct_answers(sample.target_answer, sample.gold_answers)
                    checks["gold_aliases_valid"] = True
                    checks["target_distinct_from_gold"] = True
                except ValueError as exc:
                    message = str(exc)
                    stage = (
                        "target_distinct_from_gold"
                        if "identical" in message
                        else "gold_aliases_valid"
                    )
                    row = _ineligible_row(
                        provenance,
                        sample=sample,
                        stage=stage,
                        reason=message,
                        checks=checks,
                        physical_gpu=physical_gpu,
                    )
                    _write_row(handle, row)
                    counts["ineligible"] += 1
                    continue

                prepared = prepare_query(
                    sample,
                    pipeline,
                    poison_chunk=poison_chunk,
                    all_poison_chunks=all_poison_chunks,
                )
                natural_layout = pipeline.generator.build_prompt(
                    sample.query, candidate_context(prepared.natural_selected)
                )
                natural_generation = deterministic_generation(pipeline.generator, natural_layout)
                natural_success = attack_succeeds(sample, str(natural_generation["answer"]))
                clean_layout = pipeline.generator.build_prompt(
                    sample.query, candidate_context(prepared.clean_selected)
                )
                fixed_gold, clean_gold_scores = select_fixed_gold_alias(
                    pipeline.generator, clean_layout, gold_aliases
                )
                fixed_index = int(donor_config["fixed_poison_context_index_zero_based"])
                original_forced_context = forced_context(
                    prepared,
                    poison_chunk.text,
                    fixed_zero_based_index=fixed_index,
                    top_k=pipeline.rerank_top_k,
                )
                original_forced_layout = pipeline.generator.build_prompt(
                    sample.query, original_forced_context
                )
                forced_generation = deterministic_generation(
                    pipeline.generator, original_forced_layout
                )
                forced_success = attack_succeeds(sample, str(forced_generation["answer"]))
                checks["actual_retrieval_hit"] = prepared.actual_retrieval_hit
                checks["natural_end_to_end_success"] = natural_success
                checks["forced_context_success"] = forced_success

                all_oracle_spans = [
                    *poison_chunk.anchor_ranges_chunk,
                    *poison_chunk.payload_ranges_chunk,
                ]
                anchor_count = len(poison_chunk.anchor_ranges_chunk)
                try:
                    generator_lengths = pipeline.generator.intervention_token_lengths(
                        original_forced_layout,
                        chunk_id=poison_chunk.chunk_id,
                        spans=all_oracle_spans,
                    )
                    retriever_lengths = pipeline.retriever.intervention_token_lengths(
                        poison_chunk.text, all_oracle_spans
                    )
                    reranker_lengths = pipeline.reranker.intervention_token_lengths(
                        sample.query, poison_chunk.text, all_oracle_spans
                    )
                    checks["oracle_token_partitions_valid"] = True
                except ValueError as exc:
                    row = _ineligible_row(
                        provenance,
                        sample=sample,
                        stage="oracle_token_partition",
                        reason=str(exc),
                        checks=checks,
                        physical_gpu=physical_gpu,
                    )
                    _write_row(handle, row)
                    counts["ineligible"] += 1
                    continue
                anchor_lengths = generator_lengths[:anchor_count]
                payload_lengths = generator_lengths[anchor_count:]
                anchor_minimum_requirements = tuple(
                    (
                        ("retriever", pipeline.retriever.tokenizer, retriever_lengths[index]),
                        ("reranker", pipeline.reranker.tokenizer, reranker_lengths[index]),
                    )
                    for index in range(anchor_count)
                )
                payload_minimum_requirements = tuple(
                    (
                        ("retriever", pipeline.retriever.tokenizer, retriever_lengths[index]),
                        ("reranker", pipeline.reranker.tokenizer, reranker_lengths[index]),
                    )
                    for index in range(anchor_count, len(all_oracle_spans))
                )
                relevant_sources = {
                    doc_id for doc_id, score in qrels.get(sample.query_id, {}).items() if score > 0
                }
                try:
                    donor_pairs = sampler.sample_pairs(
                        sample_id=sample.sample_id,
                        anchor_lengths=anchor_lengths,
                        payload_lengths=payload_lengths,
                        original_text=poison_chunk.text,
                        anchor_ranges=poison_chunk.anchor_ranges_chunk,
                        payload_ranges=poison_chunk.payload_ranges_chunk,
                        excluded_source_ids=relevant_sources,
                        forbidden_texts=[sample.query, sample.target_answer, *gold_aliases],
                        replicates=int(donor_config["donor_replicates"]),
                        anchor_minimum_requirements=anchor_minimum_requirements,
                        payload_minimum_requirements=payload_minimum_requirements,
                    )
                    checks["donor_pairs_valid"] = True
                except (RuntimeError, ValueError) as exc:
                    row = _ineligible_row(
                        provenance,
                        sample=sample,
                        stage="donor_sampling",
                        reason=str(exc),
                        checks=checks,
                        physical_gpu=physical_gpu,
                    )
                    _write_row(handle, row)
                    counts["ineligible"] += 1
                    continue

                full_generation_margin, full_target_score, full_gold_score = generation_margin(
                    pipeline.generator,
                    original_forced_layout,
                    target=sample.target_answer,
                    fixed_gold=fixed_gold,
                )
                full_generation = {
                    "value": full_generation_margin,
                    "target_mean_logp": full_target_score,
                    "gold_mean_logp": full_gold_score,
                }
                replicate_rows: list[dict[str, Any]] = []
                contrasts = []
                for pair in donor_pairs:
                    names = ("empty", "anchor", "payload", "both")
                    interventions = donor_interventions(poison_chunk, pair)
                    retrieval = retrieval_values_for_interventions(
                        pipeline, prepared, sample.query, interventions
                    )
                    generation: dict[str, dict[str, float]] = {}
                    for name in names:
                        spans, donor_texts = interventions[name]
                        if any(value is not None for value in donor_texts):
                            margin, target_score, gold_score = generation_margin(
                                pipeline.generator,
                                original_forced_layout,
                                target=sample.target_answer,
                                fixed_gold=fixed_gold,
                                chunk_id=poison_chunk.chunk_id,
                                replacement_spans=spans,
                                replacement_texts=donor_texts,
                            )
                            generation[name] = {
                                "value": margin,
                                "target_mean_logp": target_score,
                                "gold_mean_logp": gold_score,
                            }
                        else:
                            generation[name] = full_generation
                    contrast = oracle_contrast(_coalition(retrieval), _coalition(generation))
                    contrasts.append(contrast)
                    replicate_rows.append(
                        {
                            "replicate": pair.replicate,
                            "donors": asdict(pair),
                            "retrieval_coalitions": retrieval,
                            "generation_coalitions": generation,
                            "contrast": asdict(contrast),
                        }
                    )

                aggregate = aggregate_oracle_replicates(
                    contrasts,
                    total_replicates=int(donor_config["donor_replicates"]),
                    minimum_valid=int(donor_config["minimum_valid_donor_replicates"]),
                )
                mask_row: dict[str, Any]
                mask_contrast = None
                try:
                    mask_retrieval = mask_retrieval_values(pipeline, prepared, sample.query)
                    payload_count = len(poison_chunk.payload_ranges_chunk)
                    hidden_by_coalition: dict[str, list[bool]] = {
                        "empty": [True] * len(all_oracle_spans),
                        "anchor": [False] * anchor_count + [True] * payload_count,
                        "payload": [True] * anchor_count + [False] * payload_count,
                        "both": [False] * len(all_oracle_spans),
                    }
                    mask_generation: dict[str, dict[str, float]] = {}
                    for name, hidden_flags in hidden_by_coalition.items():
                        if any(hidden_flags):
                            margin, target_score, gold_score = generation_margin(
                                pipeline.generator,
                                original_forced_layout,
                                target=sample.target_answer,
                                fixed_gold=fixed_gold,
                                chunk_id=poison_chunk.chunk_id,
                                partition_spans=all_oracle_spans,
                                partition_hidden=hidden_flags,
                            )
                            mask_generation[name] = {
                                "value": margin,
                                "target_mean_logp": target_score,
                                "gold_mean_logp": gold_score,
                            }
                        else:
                            mask_generation[name] = full_generation
                    mask_contrast = oracle_contrast(
                        _coalition(mask_retrieval), _coalition(mask_generation)
                    )
                    mask_row = {
                        "valid": True,
                        "retrieval_coalitions": mask_retrieval,
                        "generation_coalitions": mask_generation,
                        "contrast": asdict(mask_contrast),
                    }
                except ValueError as exc:
                    mask_row = {"valid": False, "reason": str(exc)}

                retrieval_agreement = (
                    direction_agreement(
                        [value.c_retrieval for value in contrasts],
                        mask_contrast.c_retrieval,
                    )
                    if mask_contrast is not None
                    else float("nan")
                )
                generation_agreement = (
                    direction_agreement(
                        [value.c_generation for value in contrasts],
                        mask_contrast.c_generation,
                    )
                    if mask_contrast is not None
                    else float("nan")
                )
                agreement_threshold = float(donor_config["direction_agreement_minimum"])
                direction_robust = bool(
                    np.isfinite(retrieval_agreement)
                    and np.isfinite(generation_agreement)
                    and retrieval_agreement >= agreement_threshold
                    and generation_agreement >= agreement_threshold
                )
                retrieval_agreement_output = (
                    retrieval_agreement if np.isfinite(retrieval_agreement) else None
                )
                generation_agreement_output = (
                    generation_agreement if np.isfinite(generation_agreement) else None
                )
                checks["mechanism_scores_finite"] = True
                row = {
                    **provenance,
                    "sample_id": sample.sample_id,
                    "query_id": sample.query_id,
                    "family": sample.family,
                    "dataset": sample.dataset,
                    "status": "ELIGIBLE",
                    "physical_gpu": physical_gpu,
                    "attrition_checks": checks,
                    "oracle": {
                        "poison_chunk_id": poison_chunk.chunk_id,
                        "source_chars": poison_chunk.source_chars.model_dump(mode="json"),
                        "anchor_ranges_chunk": [
                            span.model_dump(mode="json")
                            for span in poison_chunk.anchor_ranges_chunk
                        ],
                        "payload_ranges_chunk": [
                            span.model_dump(mode="json")
                            for span in poison_chunk.payload_ranges_chunk
                        ],
                        "anchor_token_lengths": list(anchor_lengths),
                        "payload_token_lengths": list(payload_lengths),
                        "token_ownership_rule": ownership_rule,
                        "model_token_partitions": {
                            "generator": {
                                "anchor": list(generator_lengths[:anchor_count]),
                                "payload": list(generator_lengths[anchor_count:]),
                            },
                            "retriever": {
                                "anchor": list(retriever_lengths[:anchor_count]),
                                "payload": list(retriever_lengths[anchor_count:]),
                            },
                            "reranker": {
                                "anchor": list(reranker_lengths[:anchor_count]),
                                "payload": list(reranker_lengths[anchor_count:]),
                            },
                        },
                        "donor_tokenizer": str(pipeline.generator.tokenizer.name_or_path),
                        "token_level_replacement_preserves_model_sequence_length": True,
                        "non_generator_donor_policy": (
                            "at_least_partition_length_then_deterministic_prefix"
                        ),
                    },
                    "outcomes": {
                        "actual_retrieval_hit": prepared.actual_retrieval_hit,
                        "natural_end_to_end_success": natural_success,
                        "forced_context_success": forced_success,
                        "stratum": _outcome_stratum(prepared.actual_retrieval_hit, forced_success),
                        "poison_dense_rank": prepared.poison_dense_rank,
                        "poison_rerank_rank": prepared.poison_rerank_rank,
                        "natural_generation": natural_generation,
                        "forced_generation": forced_generation,
                        "fixed_poison_context_index_zero_based": fixed_index,
                    },
                    "gold": {
                        "aliases": list(gold_aliases),
                        "clean_context_scores": clean_gold_scores,
                        "fixed_best_alias": fixed_gold,
                    },
                    "precision_contract": pipeline.generator.precision_metadata(),
                    "donor_replicates": replicate_rows,
                    "primary_estimate": asdict(aggregate),
                    "mask_robustness": mask_row,
                    "direction_agreement": {
                        "retrieval": retrieval_agreement_output,
                        "generation": generation_agreement_output,
                        "minimum": agreement_threshold,
                        "passed": direction_robust,
                    },
                }
                _write_row(handle, row)
                counts["eligible"] += 1
    finally:
        bundle.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="RGRD-V0.1 Oracle mechanism worker")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attack-root", type=Path, required=True)
    parser.add_argument("--family", choices=ACTIVE_FAMILIES, required=True)
    parser.add_argument("--dataset", default="nq")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector-commit", required=True)
    arguments = parser.parse_args()
    if not 0 <= arguments.shard_id < arguments.shards:
        raise ValueError("shard-id must lie in [0, shards)")
    result = run_worker(arguments)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
