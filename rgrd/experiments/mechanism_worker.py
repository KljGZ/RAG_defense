from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import yaml

from rgrd.attacks import (
    attack_succeeds,
    load_phantom_samples,
    load_poisonedrag_blackbox,
    load_poisonedrag_whitebox,
)
from rgrd.attribution import oracle_mass
from rgrd.data import load_qrels
from rgrd.indexing import ExactIndexBundle
from rgrd.models import CausalAnswerGenerator, CrossEncoderReranker, DenseRetriever
from rgrd.pipeline.track_b import TrackBPipeline
from rgrd.schema import CharRange
from rgrd.experiments.resume import event_provenance, prepare_jsonl_resume


def _load_samples(family: str, attack_root: Path, root: Path):
    if family == "PoisonedRAG-B":
        return load_poisonedrag_blackbox(attack_root / "results/query_results/main")
    if family == "PoisonedRAG-W":
        return load_poisonedrag_whitebox(root / "artifacts/attacks/poisonedrag_w_nq100.json")
    if family == "Phantom":
        return load_phantom_samples(attack_root / "phantom-results")
    raise ValueError(f"unknown mechanism family: {family}")


def _whole_range_present(projected: list[CharRange], originals: tuple[CharRange, ...]) -> bool:
    return all(
        any(item.start == original.start and item.end == original.end for item in projected)
        for original in originals
    )


def _role_row(role_result, chunk_id: str):
    return next(item for item in role_result.chunks if item.chunk_id == chunk_id)


def _oracle_values(role, anchor_ranges: list[CharRange], payload_ranges: list[CharRange]):
    atomic = [CharRange(start=span.start, end=span.end) for span in role.spans]
    p_r = [span.p_retrieval for span in role.spans]
    p_g = [span.p_generation for span in role.spans]
    return {
        "anchor_p_retrieval": oracle_mass(atomic, p_r, anchor_ranges),
        "anchor_p_generation": oracle_mass(atomic, p_g, anchor_ranges),
        "payload_p_retrieval": oracle_mass(atomic, p_r, payload_ranges),
        "payload_p_generation": oracle_mass(atomic, p_g, payload_ranges),
    }


def run_worker(args: argparse.Namespace) -> dict[str, int]:
    root = args.root.resolve()
    pipeline_config = yaml.safe_load(
        (root / "configs/pipeline/v0.yaml").read_text(encoding="utf-8")
    )
    datasets = yaml.safe_load((root / "configs/datasets.yaml").read_text(encoding="utf-8"))[
        "datasets"
    ]
    dataset_config = datasets[args.dataset]
    samples = [
        sample
        for sample in _load_samples(args.family, args.attack_root, root)
        if sample.dataset == args.dataset
        and sample.original_retrieval_hit
        and sample.original_end_to_end_success
    ]
    samples = samples[: args.max_candidates]
    samples = [
        sample for index, sample in enumerate(samples) if index % args.shards == args.shard_id
    ]
    index = ExactIndexBundle(root / f"artifacts/indexes/{args.dataset}/manifest.json")
    models = pipeline_config["models"]
    track = pipeline_config["track_b"]
    retriever = DenseRetriever(Path(models["retriever"]["local_path"]), device=args.device)
    reranker = CrossEncoderReranker(Path(models["reranker"]["local_path"]), device=args.device)
    generator = CausalAnswerGenerator(
        Path(models["generator"]["local_path"]),
        device=args.device,
        max_new_tokens=int(track["generator_max_new_tokens"]),
        seed=int(pipeline_config["seed"]),
    )
    noise_path = root / "artifacts/statistics/noop_noise_floor.json"
    if not noise_path.is_file():
        raise FileNotFoundError("mechanism validation requires the frozen no-op noise floor")
    noise = json.loads(noise_path.read_text(encoding="utf-8"))
    pipeline = TrackBPipeline(
        index=index,
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        dense_top_b=int(track["dense_top_b"]),
        rerank_top_k=int(track["rerank_top_k"]),
        span_size=int(track["span_size"]),
        span_offsets=tuple(track["span_offset_views"]),
        retrieval_noise_floor=float(noise["retrieval"]),
        generation_noise_floor=float(noise["generation"]),
        model_revisions=args.model_revision,
    )
    qrels = load_qrels(Path(dataset_config["qrels"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = event_provenance(
        root,
        detector_code_commit=args.detector_commit,
        model_revisions=args.model_revision,
    )
    completed = {
        key[0]
        for key in prepare_jsonl_resume(
            args.output,
            key_fields=("sample_id",),
            expected_provenance=provenance,
        )
    }
    counts = {"attempted": 0, "eligible": 0, "skipped": len(completed)}
    with args.output.open("a", encoding="utf-8") as handle:
        for sample in samples:
            if sample.sample_id in completed:
                continue
            counts["attempted"] += 1
            query_embedding, selected, frozen = pipeline.retrieve(
                sample.query,
                injected_sources=[
                    (sample.source, list(sample.anchor_ranges), list(sample.payload_ranges))
                ],
            )
            poison_candidates = [
                candidate
                for candidate in selected
                if candidate.source.source_doc_id == sample.source.source_doc_id
                and _whole_range_present(candidate.chunk.anchor_ranges_source, sample.anchor_ranges)
                and _whole_range_present(
                    candidate.chunk.payload_ranges_source, sample.payload_ranges
                )
            ]
            if not poison_candidates:
                continue
            poison = poison_candidates[0]
            event, layout = pipeline.build_detector_event(
                query_id=sample.query_id,
                query=sample.query,
                dataset=sample.dataset,
                split=str(dataset_config["split"]),
                selected=selected,
                frozen=frozen,
                detector_code_commit=args.detector_commit,
                seed=int(pipeline_config["seed"]),
            )
            if not attack_succeeds(sample, event.shadow_answer):
                continue
            relevant = {
                doc_id for doc_id, score in qrels.get(sample.query_id, {}).items() if score > 0
            }
            if not relevant:
                continue
            clean_query_embedding, clean_selected, clean_frozen = pipeline.retrieve(sample.query)
            clean_candidates = [
                candidate
                for candidate in clean_selected
                if candidate.source.source_doc_id in relevant
            ]
            if not clean_candidates:
                continue
            clean = clean_candidates[0]
            clean_event, clean_layout = pipeline.build_detector_event(
                query_id=sample.query_id,
                query=sample.query,
                dataset=sample.dataset,
                split=str(dataset_config["split"]),
                selected=clean_selected,
                frozen=clean_frozen,
                detector_code_commit=args.detector_commit,
                seed=int(pipeline_config["seed"]),
            )
            poison_roles = pipeline.score_mechanism_chunks(
                event,
                candidates=selected,
                query_embedding=query_embedding,
                frozen=frozen,
                layout=layout,
                evaluated_chunk_ids=[poison.chunk.chunk_id],
            )
            clean_roles = pipeline.score_mechanism_chunks(
                clean_event,
                candidates=clean_selected,
                query_embedding=clean_query_embedding,
                frozen=clean_frozen,
                layout=clean_layout,
                evaluated_chunk_ids=[clean.chunk.chunk_id],
            )
            poison_role = _role_row(poison_roles, poison.chunk.chunk_id)
            clean_role = _role_row(clean_roles, clean.chunk.chunk_id)
            oracle_values = _oracle_values(
                poison_role,
                poison.chunk.anchor_ranges_chunk,
                poison.chunk.payload_ranges_chunk,
            )
            value = {
                **provenance,
                "sample_id": sample.sample_id,
                "query_id": sample.query_id,
                "family": sample.family,
                "dataset": sample.dataset,
                "poison_chunk_id": poison.chunk.chunk_id,
                "clean_chunk_id": clean.chunk.chunk_id,
                "shadow_answer": event.shadow_answer,
                "clean_shadow_answer": clean_event.shadow_answer,
                "poison_t_rg": poison_role.t_rg,
                "clean_t_rg": clean_role.t_rg,
                **oracle_values,
                "poison_role": asdict(poison_role),
                "clean_role": asdict(clean_role),
            }
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            counts["eligible"] += 1
    index.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="RGRD E02 mechanism worker")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attack-root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector-commit", required=True)
    parser.add_argument("--model-revision", action="append", default=[])
    arguments = parser.parse_args()
    arguments.model_revision = dict(item.split("=", 1) for item in arguments.model_revision)
    counts = run_worker(arguments)
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
