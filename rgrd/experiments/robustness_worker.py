from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from rgrd.attacks import (
    load_phantom_samples,
    load_poisonedrag_blackbox,
    load_poisonedrag_whitebox,
)
from rgrd.attribution import oracle_mass
from rgrd.experiments.detection_worker import _pipeline
from rgrd.ingestion import tokenize_with_offsets
from rgrd.intervention import build_span_views
from rgrd.schema import CharRange
from rgrd.experiments.resume import event_provenance, prepare_jsonl_resume


QUOTAS = {"PoisonedRAG-B": 100, "PoisonedRAG-W": 100, "Phantom": 50}


def _selected_rows(input_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in sorted(input_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["sample_id"] not in seen:
                rows.append(row)
                seen.add(row["sample_id"])
    selected: list[dict[str, object]] = []
    for family, quota in QUOTAS.items():
        selected.extend([row for row in rows if row["family"] == family][:quota])
    selected.sort(key=lambda row: hashlib.sha256(str(row["sample_id"]).encode()).hexdigest())
    count = max(1, round(len(selected) * 0.10))
    return selected[:count]


def _samples(root: Path, attack_root: Path):
    values = [
        *load_poisonedrag_blackbox(attack_root / "results/query_results/main"),
        *load_poisonedrag_whitebox(root / "artifacts/attacks/poisonedrag_w_nq100.json"),
        *load_phantom_samples(attack_root / "phantom-results"),
    ]
    return {sample.sample_id: sample for sample in values}


def _oracle_direction(role, ranges: list[CharRange], retrieval_minus_generation: bool) -> int:
    atoms = [CharRange(start=span.start, end=span.end) for span in role.spans]
    p_r = oracle_mass(atoms, [span.p_retrieval for span in role.spans], ranges)
    p_g = oracle_mass(atoms, [span.p_generation for span in role.spans], ranges)
    value = p_r - p_g if retrieval_minus_generation else p_g - p_r
    return int(value > 0) - int(value < 0)


def _overlap_count(offsets, span: CharRange, eligible: set[int] | None = None) -> int:
    return sum(
        1
        for index, (start, end) in enumerate(offsets)
        if end > start
        and start < span.end
        and end > span.start
        and (eligible is None or index in eligible)
    )


def _required_donor_lengths(pipeline, candidate, event, layout) -> tuple[int, int, int]:
    offsets = [
        (token.char_start, token.char_end)
        for token in tokenize_with_offsets(candidate.chunk.text, pipeline.generator.tokenizer)
    ]
    spans = build_span_views(offsets, span_size=pipeline.span_size, offsets=pipeline.span_offsets)
    if not spans:
        raise RuntimeError("donor robustness target has no intervention spans")

    retriever_encoded = pipeline.retriever.tokenizer(
        candidate.chunk.text,
        truncation=True,
        max_length=pipeline.retriever.max_length,
        return_offsets_mapping=True,
    )
    retriever_offsets = retriever_encoded["offset_mapping"]
    reranker_encoded = pipeline.reranker.tokenizer(
        event.query.query,
        candidate.chunk.text,
        truncation=True,
        max_length=pipeline.reranker.max_length,
        return_offsets_mapping=True,
    )
    reranker_offsets = reranker_encoded["offset_mapping"]
    reranker_chunk = {
        index
        for index, sequence_id in enumerate(reranker_encoded.sequence_ids())
        if sequence_id == 1
    }
    full_text = layout.prompt + event.shadow_answer
    generator_offsets = pipeline.generator.tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )["offset_mapping"]
    chunk_range = layout.chunk_ranges[candidate.chunk.chunk_id]
    generator_counts = []
    retriever_counts = []
    reranker_counts = []
    for span in spans:
        prompt_span = CharRange(
            start=chunk_range.start + span.char_range.start,
            end=chunk_range.start + span.char_range.end,
        )
        generator_counts.append(_overlap_count(generator_offsets, prompt_span))
        retriever_counts.append(_overlap_count(retriever_offsets, span.char_range))
        reranker_counts.append(_overlap_count(reranker_offsets, span.char_range, reranker_chunk))
    required = (
        max(retriever_counts),
        max(reranker_counts),
        max(generator_counts),
    )
    if min(required) <= 0:
        raise RuntimeError("an intervention span did not map into every model tokenizer")
    return required


def _donor_is_long_enough(pipeline, text: str, required: tuple[int, int, int]) -> bool:
    tokenizers = (
        pipeline.retriever.tokenizer,
        pipeline.reranker.tokenizer,
        pipeline.generator.tokenizer,
    )
    for tokenizer, minimum in zip(tokenizers, required, strict=True):
        values = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if values and isinstance(values[0], list):
            values = values[0]
        if len(values) < minimum:
            return False
    return True


def run(args: argparse.Namespace) -> dict[str, int]:
    root = args.root.resolve()
    rows = _selected_rows(args.mechanism_input)
    rows = [row for index, row in enumerate(rows) if index % args.shards == args.shard_id]
    samples = _samples(root, args.attack_root)
    if args.dataset:
        rows = [row for row in rows if samples[str(row["sample_id"])].dataset == args.dataset]
    configs = yaml.safe_load((root / "configs/datasets.yaml").read_text(encoding="utf-8"))[
        "datasets"
    ]
    pipelines = {}
    outputs = 0
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
    with args.output.open("a", encoding="utf-8") as handle:
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in completed:
                continue
            sample = samples[sample_id]
            if sample.dataset not in pipelines:
                pipelines[sample.dataset] = _pipeline(
                    root, sample.dataset, args.device, args.model_revision
                )
            config, bundle, pipeline = pipelines[sample.dataset]
            query_embedding, selected, frozen = pipeline.retrieve(
                sample.query,
                injected_sources=[
                    (sample.source, list(sample.anchor_ranges), list(sample.payload_ranges))
                ],
            )
            poison = next(
                candidate
                for candidate in selected
                if candidate.chunk.chunk_id == row["poison_chunk_id"]
            )
            event, layout = pipeline.build_detector_event(
                query_id=sample.query_id,
                query=sample.query,
                dataset=sample.dataset,
                split=str(configs[sample.dataset]["split"]),
                selected=selected,
                frozen=frozen,
                detector_code_commit=args.detector_commit,
                seed=int(config["seed"]),
            )
            mask_result = pipeline.score_mechanism_chunks(
                event,
                candidates=selected,
                query_embedding=query_embedding,
                frozen=frozen,
                layout=layout,
                evaluated_chunk_ids=[poison.chunk.chunk_id],
            ).chunks[0]
            required_lengths = _required_donor_lengths(pipeline, poison, event, layout)
            _, donors_indexed = bundle.search(query_embedding, 512)
            donors: list[str] = []
            donor_ids: set[str] = set()
            for indexed in donors_indexed:
                text = indexed.chunk.text
                if indexed.source.source_doc_id == sample.source.source_doc_id:
                    continue
                if indexed.chunk.chunk_id in donor_ids:
                    continue
                if not _donor_is_long_enough(pipeline, text, required_lengths):
                    continue
                donors.append(text)
                donor_ids.add(indexed.chunk.chunk_id)
                if len(donors) == 8:
                    break
            if len(donors) != 8:
                raise RuntimeError(f"fewer than eight valid clean donors for {sample_id}")
            donor_result = pipeline.score_donor_replacements(
                event,
                candidate=poison,
                query_embedding=query_embedding,
                frozen=frozen,
                layout=layout,
                donor_texts=donors,
            )
            value = {
                **provenance,
                "sample_id": sample_id,
                "query_id": sample.query_id,
                "family": sample.family,
                "anchor": {
                    "mask_direction": _oracle_direction(
                        mask_result, poison.chunk.anchor_ranges_chunk, True
                    ),
                    "donor_direction": _oracle_direction(
                        donor_result, poison.chunk.anchor_ranges_chunk, True
                    ),
                },
                "payload": {
                    "mask_direction": _oracle_direction(
                        mask_result, poison.chunk.payload_ranges_chunk, False
                    ),
                    "donor_direction": _oracle_direction(
                        donor_result, poison.chunk.payload_ranges_chunk, False
                    ),
                },
            }
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            outputs += 1
    for _, bundle, _ in pipelines.values():
        bundle.close()
    return {"tasks": len(rows), "written": outputs}


def main() -> int:
    parser = argparse.ArgumentParser(description="RGRD E03 donor robustness worker")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attack-root", type=Path, required=True)
    parser.add_argument("--mechanism-input", type=Path, required=True)
    parser.add_argument("--dataset", choices=["nq", "msmarco"], default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector-commit", required=True)
    parser.add_argument("--model-revision", action="append", default=[])
    arguments = parser.parse_args()
    arguments.model_revision = dict(item.split("=", 1) for item in arguments.model_revision)
    print(json.dumps(run(arguments), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
