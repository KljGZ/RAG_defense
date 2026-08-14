from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from rgrd.attacks import (
    load_phantom_samples,
    load_poisonedrag_blackbox,
    load_poisonedrag_whitebox,
)
from rgrd.data import load_queries
from rgrd.indexing import ExactIndexBundle
from rgrd.models import CausalAnswerGenerator, CrossEncoderReranker, DenseRetriever
from rgrd.pipeline.track_b import TrackBPipeline
from rgrd.experiments.resume import event_provenance, prepare_jsonl_resume


def _pipeline(root: Path, dataset: str, device: str, model_revision: dict[str, str]):
    config = yaml.safe_load((root / "configs/pipeline/v0.yaml").read_text(encoding="utf-8"))
    models = config["models"]
    track = config["track_b"]
    noise_path = root / "artifacts/statistics/noop_noise_floor.json"
    noise = (
        json.loads(noise_path.read_text(encoding="utf-8"))
        if noise_path.is_file()
        else {"retrieval": 1e-8, "generation": 1e-8}
    )
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
        ),
        dense_top_b=int(track["dense_top_b"]),
        rerank_top_k=int(track["rerank_top_k"]),
        span_size=int(track["span_size"]),
        span_offsets=tuple(track["span_offset_views"]),
        retrieval_noise_floor=float(noise["retrieval"]),
        generation_noise_floor=float(noise["generation"]),
        model_revisions=model_revision,
    )
    return config, bundle, pipeline


def _successful_sample_ids(input_dir: Path, family: str) -> set[str]:
    values: set[str] = set()
    for path in input_dir.glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row["family"] == family:
                        values.add(str(row["sample_id"]))
    return values


def _load_attack_samples(root: Path, attack_root: Path, family: str):
    if family == "PoisonedRAG-B":
        return load_poisonedrag_blackbox(attack_root / "results/query_results/main")
    if family == "PoisonedRAG-W":
        return load_poisonedrag_whitebox(root / "artifacts/attacks/poisonedrag_w_nq100.json")
    if family == "Phantom":
        return load_phantom_samples(attack_root / "phantom-results")
    raise ValueError(family)


def _score_one(
    pipeline: TrackBPipeline,
    *,
    query_id: str,
    query: str,
    dataset: str,
    split: str,
    detector_commit: str,
    seed: int,
    injected_sources=(),
) -> dict[str, object]:
    query_embedding, selected, frozen = pipeline.retrieve(query, injected_sources=injected_sources)
    event, layout = pipeline.build_detector_event(
        query_id=query_id,
        query=query,
        dataset=dataset,
        split=split,
        selected=selected,
        frozen=frozen,
        detector_code_commit=detector_commit,
        seed=seed,
    )
    roles = pipeline.score_detector_event(
        event,
        candidates=pipeline.runtime_candidates(selected),
        query_embedding=query_embedding,
        frozen=frozen,
        layout=layout,
    )
    return {
        "query_id": query_id,
        "score": roles.query_score,
        "role_identifiable": any(chunk.role_identifiable for chunk in roles.chunks),
        "shadow_answer": roles.shadow_answer,
        "selected_for_generation": list(roles.selected_for_generation),
    }


def run(args: argparse.Namespace) -> dict[str, int]:
    root = args.root.resolve()
    datasets = yaml.safe_load((root / "configs/datasets.yaml").read_text(encoding="utf-8"))[
        "datasets"
    ]
    dataset_config = datasets[args.dataset]
    config, bundle, pipeline = _pipeline(root, args.dataset, args.device, args.model_revision)
    tasks: list[tuple[str, str, str, object]] = []
    if args.mode == "clean":
        queries = sorted(load_queries(Path(dataset_config["queries"])).items())
        selected = queries[args.start : args.start + args.limit]
        tasks = [(qid, query, "clean", ()) for qid, query in selected]
    else:
        successful = _successful_sample_ids(args.mechanism_input, args.family)
        samples = [
            sample
            for sample in _load_attack_samples(root, args.attack_root, args.family)
            if sample.sample_id in successful and sample.dataset == args.dataset
        ][: args.limit]
        tasks = [
            (
                sample.sample_id,
                sample.query,
                sample.family,
                [(sample.source, list(sample.anchor_ranges), list(sample.payload_ranges))],
            )
            for sample in samples
        ]
    tasks = [task for index, task in enumerate(tasks) if index % args.shards == args.shard_id]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = event_provenance(
        root,
        detector_code_commit=args.detector_commit,
        model_revisions=args.model_revision,
    )
    completed = prepare_jsonl_resume(
        args.output,
        key_fields=("query_id", "family"),
        expected_provenance=provenance,
    )
    written = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for query_id, query, family, injected in tasks:
            if (query_id, family) in completed:
                continue
            value = _score_one(
                pipeline,
                query_id=query_id,
                query=query,
                dataset=args.dataset,
                split=str(dataset_config["split"]),
                detector_commit=args.detector_commit,
                seed=int(config["seed"]),
                injected_sources=injected,
            )
            value["family"] = family
            value["partition"] = args.partition
            value.update(provenance)
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
    bundle.close()
    return {"tasks": len(tasks), "already_complete": len(completed), "written": written}


def main() -> int:
    parser = argparse.ArgumentParser(description="RGRD E04 label-free detector worker")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attack-root", type=Path, default=Path("/home/jkl/RAG_attack"))
    parser.add_argument("--mode", choices=["clean", "attack"], required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--family", default="clean")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--mechanism-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector-commit", required=True)
    parser.add_argument("--model-revision", action="append", default=[])
    arguments = parser.parse_args()
    arguments.model_revision = dict(item.split("=", 1) for item in arguments.model_revision)
    result = run(arguments)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
