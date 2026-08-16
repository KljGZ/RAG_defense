from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from rgrd.attacks.artifacts import AttackSample
from rgrd.ingestion import chunk_source
from rgrd.pipeline.track_b import Candidate, TrackBPipeline
from rgrd.retrieval import freeze_thresholds, pipeline_margin
from rgrd.schema import CharRange, ChunkLineage
from rgrd.v01.donors import DonorPair, validate_disjoint_ranges


@dataclass(frozen=True)
class PreparedQuery:
    query_embedding: np.ndarray
    poison_chunk: ChunkLineage
    all_poison_chunks: tuple[ChunkLineage, ...]
    frozen: object
    poison_dense_score: float
    poison_rerank_score: float
    poison_dense_rank: int | None
    poison_rerank_rank: int | None
    actual_retrieval_hit: bool
    natural_selected: tuple[Candidate, ...]
    clean_selected: tuple[Candidate, ...]


class GenerationFormatError(RuntimeError):
    """A deterministic sample output that cannot satisfy the frozen answer format."""

    def __init__(self, code: str, audit: object) -> None:
        self.code = code
        self.audit = audit
        values = asdict(audit)
        super().__init__(
            f"{code}: generated_tokens={values['generated_tokens']} "
            f"termination_token_id={values['termination_token_id']}"
        )

    def evidence(self) -> dict[str, object]:
        return {"code": self.code, "audit": asdict(self.audit)}


def _ranges_valid(text: str, ranges: Sequence[CharRange]) -> bool:
    ordered = sorted(ranges, key=lambda span: (span.start, span.end))
    return bool(ordered) and all(
        span.start >= 0
        and span.end <= len(text)
        and (index == 0 or ordered[index - 1].end <= span.start)
        for index, span in enumerate(ordered)
    )


def _full_projection(projected: Sequence[CharRange], originals: Sequence[CharRange]) -> bool:
    return sorted((span.start, span.end) for span in projected) == sorted(
        (span.start, span.end) for span in originals
    )


def oracle_cochunk(
    sample: AttackSample,
    pipeline: TrackBPipeline,
    *,
    chunk_size: int,
    overlap: int,
) -> tuple[ChunkLineage, tuple[ChunkLineage, ...]]:
    """Choose the lexicographically first chunk containing every Oracle range."""

    if not _ranges_valid(sample.source.text, sample.anchor_ranges):
        raise ValueError("source anchor ranges are invalid or overlapping")
    if not _ranges_valid(sample.source.text, sample.payload_ranges):
        raise ValueError("source payload ranges are invalid or overlapping")
    for anchor in sample.anchor_ranges:
        if any(anchor.overlap(payload) > 0 for payload in sample.payload_ranges):
            raise ValueError("source anchor and payload ranges overlap")
    chunks = tuple(
        chunk_source(
            sample.source,
            pipeline.generator.tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
            anchor_ranges=list(sample.anchor_ranges),
            payload_ranges=list(sample.payload_ranges),
        )
    )
    candidates = [
        chunk
        for chunk in chunks
        if _full_projection(chunk.anchor_ranges_source, sample.anchor_ranges)
        and _full_projection(chunk.payload_ranges_source, sample.payload_ranges)
    ]
    if not candidates:
        raise ValueError("Oracle anchor and payload do not co-occur in one final chunk")
    chosen = sorted(candidates, key=lambda chunk: chunk.chunk_id)[0]
    validate_disjoint_ranges(chosen.text, chosen.anchor_ranges_chunk, chosen.payload_ranges_chunk)
    return chosen, chunks


def prepare_query(
    sample: AttackSample,
    pipeline: TrackBPipeline,
    *,
    poison_chunk: ChunkLineage,
    all_poison_chunks: Sequence[ChunkLineage],
) -> PreparedQuery:
    """Freeze actual retrieval, while retaining the Oracle chunk outside Top-B if needed."""

    query_embedding = pipeline.retriever.encode([sample.query])[0]
    clean_scores, clean_entries = pipeline.index.search(query_embedding, pipeline.dense_top_b)
    raw: list[tuple[object, ChunkLineage, float, bool]] = [
        (entry.source, entry.chunk, float(score), False)
        for score, entry in zip(clean_scores, clean_entries, strict=True)
    ]
    poison_embeddings = pipeline.retriever.encode([chunk.text for chunk in all_poison_chunks])
    poison_score_by_id: dict[str, float] = {}
    for chunk, embedding in zip(all_poison_chunks, poison_embeddings, strict=True):
        score = pipeline.retriever.score(query_embedding, embedding)
        poison_score_by_id[chunk.chunk_id] = score
        raw.append((sample.source, chunk, score, True))
    if poison_chunk.chunk_id not in poison_score_by_id:
        raise AssertionError("chosen Oracle chunk is absent from injected chunk set")
    raw.sort(key=lambda item: (-item[2], item[1].chunk_id))
    dense_top = raw[: pipeline.dense_top_b]
    rerank_scores = pipeline.reranker.score_pairs(
        sample.query, [item[1].text for item in dense_top]
    )
    if not np.all(np.isfinite(rerank_scores)):
        raise FloatingPointError("actual retrieval reranker produced non-finite scores")
    rerank_order = sorted(
        range(len(dense_top)),
        key=lambda index: (-float(rerank_scores[index]), dense_top[index][1].chunk_id),
    )
    rerank_ranks = {dense_index: rank + 1 for rank, dense_index in enumerate(rerank_order)}
    candidates = tuple(
        Candidate(
            source=source,
            chunk=chunk,
            dense_score=float(dense_score),
            dense_rank=index + 1,
            rerank_score=float(rerank_scores[index]),
            rerank_rank=rerank_ranks[index],
            injected=injected,
        )
        for index, (source, chunk, dense_score, injected) in enumerate(dense_top)
    )
    frozen = freeze_thresholds(
        np.asarray([candidate.dense_score for candidate in candidates]),
        np.asarray([candidate.rerank_score for candidate in candidates]),
        dense_top_b=min(pipeline.dense_top_b, len(candidates)),
        rerank_top_k=min(pipeline.rerank_top_k, len(candidates)),
    )
    natural_selected = tuple(
        sorted(candidates, key=lambda candidate: candidate.rerank_rank)[: pipeline.rerank_top_k]
    )
    poison_in_top = next(
        (
            candidate
            for candidate in candidates
            if candidate.chunk.chunk_id == poison_chunk.chunk_id
        ),
        None,
    )
    if poison_in_top is None:
        poison_dense_rank = None
        poison_rerank_rank = None
        poison_rerank_score = float(
            pipeline.reranker.score_pairs(sample.query, [poison_chunk.text])[0]
        )
    else:
        poison_dense_rank = poison_in_top.dense_rank
        poison_rerank_rank = poison_in_top.rerank_rank
        poison_rerank_score = poison_in_top.rerank_score
    if not np.isfinite(poison_rerank_score):
        raise FloatingPointError("forced Oracle reranker score is non-finite")
    _, clean_selected, _ = pipeline.retrieve(sample.query)
    actual_hit = bool(
        poison_dense_rank is not None
        and poison_rerank_rank is not None
        and poison_rerank_rank <= pipeline.rerank_top_k
    )
    return PreparedQuery(
        query_embedding=query_embedding,
        poison_chunk=poison_chunk,
        all_poison_chunks=tuple(all_poison_chunks),
        frozen=frozen,
        poison_dense_score=float(poison_score_by_id[poison_chunk.chunk_id]),
        poison_rerank_score=float(poison_rerank_score),
        poison_dense_rank=poison_dense_rank,
        poison_rerank_rank=poison_rerank_rank,
        actual_retrieval_hit=actual_hit,
        natural_selected=natural_selected,
        clean_selected=tuple(clean_selected),
    )


def candidate_context(candidates: Sequence[Candidate]) -> list[tuple[str, str]]:
    return [
        (candidate.chunk.chunk_id, candidate.chunk.text)
        for candidate in sorted(candidates, key=lambda item: item.rerank_rank)
    ]


def forced_context(
    prepared: PreparedQuery,
    poison_text: str,
    *,
    fixed_zero_based_index: int,
    top_k: int,
) -> list[tuple[str, str]]:
    clean = candidate_context(prepared.clean_selected)
    if not 0 <= fixed_zero_based_index < top_k:
        raise ValueError("fixed poison position must lie within Top-K")
    position = min(fixed_zero_based_index, len(clean))
    result = clean[:position] + [(prepared.poison_chunk.chunk_id, poison_text)] + clean[position:]
    result = result[:top_k]
    if len(result) != top_k:
        raise RuntimeError("forced context could not preserve the fixed Top-K size")
    if result[position][0] != prepared.poison_chunk.chunk_id:
        raise AssertionError("forced poison chunk moved from its fixed context position")
    return result


def deterministic_generation(generator: object, layout: object) -> dict[str, object]:
    first = generator.generate_shadow_audited(layout)
    second = generator.generate_shadow_audited(layout)
    if first != second:
        raise RuntimeError("strict deterministic generation contract was violated")
    if first.truncated:
        raise GenerationFormatError("max_new_tokens_without_eos", first)
    if not first.terminated_by_eos:
        raise GenerationFormatError("missing_configured_eos", first)
    if not first.strict_single_line:
        raise GenerationFormatError("not_exactly_one_nonempty_line", first)
    return asdict(first)


def select_fixed_gold_alias(
    generator: object,
    clean_layout: object,
    aliases: Sequence[str],
) -> tuple[str, dict[str, float]]:
    values = {alias: float(generator.teacher_score(clean_layout, alias)) for alias in aliases}
    if not values or not np.all(np.isfinite(list(values.values()))):
        raise FloatingPointError("clean-context gold alias scores are empty or non-finite")
    selected = sorted(values, key=lambda alias: (-values[alias], alias))[0]
    return selected, values


def generation_margin(
    generator: object,
    layout: object,
    *,
    target: str,
    fixed_gold: str,
    chunk_id: str | None = None,
    hidden_spans: Sequence[CharRange] | None = None,
    replacement_spans: Sequence[CharRange] | None = None,
    replacement_texts: Sequence[str | None] | None = None,
    partition_spans: Sequence[CharRange] | None = None,
    partition_hidden: Sequence[bool] | None = None,
) -> tuple[float, float, float]:
    modes = sum(value is not None for value in (hidden_spans, replacement_spans, partition_spans))
    if modes > 1:
        raise ValueError("generation coalition must use exactly one intervention mode")
    if hidden_spans:
        kwargs = {"chunk_id": chunk_id, "hidden_spans": hidden_spans}
    elif replacement_spans is not None:
        kwargs = {
            "chunk_id": chunk_id,
            "replacement_spans": replacement_spans,
            "replacement_texts": replacement_texts,
        }
    elif partition_spans is not None:
        kwargs = {
            "chunk_id": chunk_id,
            "partition_spans": partition_spans,
            "partition_hidden": partition_hidden,
        }
    else:
        kwargs = {}
    target_score = float(generator.teacher_score(layout, target, **kwargs))
    gold_score = float(generator.teacher_score(layout, fixed_gold, **kwargs))
    margin = target_score - gold_score
    if not np.all(np.isfinite([target_score, gold_score, margin])):
        raise FloatingPointError("generation coalition contains non-finite values")
    return float(margin), target_score, gold_score


def donor_interventions(
    chunk: ChunkLineage,
    pair: DonorPair,
) -> dict[str, tuple[list[CharRange], list[str | None]]]:
    anchor = [segment.text for segment in pair.anchor]
    payload = [segment.text for segment in pair.payload]
    all_spans = [*chunk.anchor_ranges_chunk, *chunk.payload_ranges_chunk]
    return {
        "empty": (all_spans, [*anchor, *payload]),
        "anchor": (all_spans, [None] * len(anchor) + payload),
        "payload": (all_spans, anchor + [None] * len(payload)),
        "both": (all_spans, [None] * len(all_spans)),
    }


def retrieval_values_for_interventions(
    pipeline: TrackBPipeline,
    prepared: PreparedQuery,
    query: str,
    interventions: Mapping[str, tuple[Sequence[CharRange], Sequence[str | None]]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for key, (spans, donor_texts) in interventions.items():
        if any(value is not None for value in donor_texts):
            embedding = pipeline.retriever.encode_replaced_ranges(
                prepared.poison_chunk.text, spans, donor_texts
            )
            rerank_score = pipeline.reranker.score_replaced_ranges(
                query, prepared.poison_chunk.text, spans, donor_texts
            )
            dense_score = pipeline.retriever.score(prepared.query_embedding, embedding)
        else:
            dense_score = prepared.poison_dense_score
            rerank_score = prepared.poison_rerank_score
        margin = pipeline_margin(dense_score, float(rerank_score), prepared.frozen)
        if not np.all(np.isfinite([dense_score, rerank_score, margin])):
            raise FloatingPointError("retrieval coalition contains non-finite values")
        result[key] = {
            "value": float(margin),
            "dense_score": float(dense_score),
            "rerank_score": float(rerank_score),
        }
    return result


def mask_retrieval_values(
    pipeline: TrackBPipeline,
    prepared: PreparedQuery,
    query: str,
) -> dict[str, dict[str, float]]:
    chunk = prepared.poison_chunk
    all_spans = [*chunk.anchor_ranges_chunk, *chunk.payload_ranges_chunk]
    anchor_count = len(chunk.anchor_ranges_chunk)
    payload_count = len(chunk.payload_ranges_chunk)
    absent = {
        "empty": [True] * len(all_spans),
        "anchor": [False] * anchor_count + [True] * payload_count,
        "payload": [True] * anchor_count + [False] * payload_count,
        "both": [False] * len(all_spans),
    }
    result: dict[str, dict[str, float]] = {}
    for coalition, hidden_flags in absent.items():
        if any(hidden_flags):
            embedding = pipeline.retriever.encode_hidden_partition(
                chunk.text, all_spans, hidden_flags
            )
            rerank_score = pipeline.reranker.score_hidden_partition(
                query, chunk.text, all_spans, hidden_flags
            )
            dense_score = pipeline.retriever.score(prepared.query_embedding, embedding)
        else:
            dense_score = prepared.poison_dense_score
            rerank_score = prepared.poison_rerank_score
        margin = pipeline_margin(dense_score, rerank_score, prepared.frozen)
        if not np.all(np.isfinite([dense_score, rerank_score, margin])):
            raise FloatingPointError("mask retrieval coalition contains non-finite values")
        result[coalition] = {
            "value": float(margin),
            "dense_score": float(dense_score),
            "rerank_score": float(rerank_score),
        }
    return result
