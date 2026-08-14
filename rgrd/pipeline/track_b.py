from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from rgrd.attribution import (
    aggregate_overlapping_span_effects,
    attribute_generation,
    attribute_retrieval,
    compute_role_map,
)
from rgrd.generation import generation_effect
from rgrd.indexing import ExactIndexBundle
from rgrd.ingestion import chunk_source, tokenize_with_offsets
from rgrd.intervention import build_span_views
from rgrd.models import CausalAnswerGenerator, CrossEncoderReranker, DenseRetriever
from rgrd.provenance import canonical_hash
from rgrd.retrieval import freeze_thresholds, pipeline_margin
from rgrd.schema import (
    CharRange,
    ChunkLineage,
    DetectorChunk,
    DetectorEvent,
    DetectorQuery,
    ModelReference,
    RetrievalTrace,
    SourceDocument,
)


@dataclass(frozen=True)
class Candidate:
    source: SourceDocument
    chunk: ChunkLineage
    dense_score: float
    dense_rank: int
    rerank_score: float
    rerank_rank: int
    injected: bool


@dataclass(frozen=True)
class RuntimeCandidate:
    """Label-free candidate accepted by the runtime detector boundary."""

    chunk: DetectorChunk
    dense_score: float
    dense_rank: int
    rerank_score: float
    rerank_rank: int


@dataclass(frozen=True)
class SpanEffectRecord:
    start: int
    end: int
    retrieval_effect: float
    generation_effect: float
    p_retrieval: float
    p_generation: float
    role: str


@dataclass(frozen=True)
class ChunkRoleResult:
    chunk_id: str
    retrieval_mass: float
    generation_mass: float
    t_rg: float
    role_identifiable: bool
    spans: tuple[SpanEffectRecord, ...]


@dataclass(frozen=True)
class QueryRoleResult:
    query_id: str
    shadow_answer: str
    query_score: float
    selected_for_generation: tuple[str, ...]
    chunks: tuple[ChunkRoleResult, ...]


class TrackBPipeline:
    """Fixed exact dense -> cross-encoder -> open generator RGRD pipeline."""

    def __init__(
        self,
        *,
        index: ExactIndexBundle,
        retriever: DenseRetriever,
        reranker: CrossEncoderReranker,
        generator: CausalAnswerGenerator,
        dense_top_b: int = 50,
        rerank_top_k: int = 5,
        span_size: int = 32,
        span_offsets: tuple[int, ...] = (0, 16),
        retrieval_noise_floor: float = 1e-8,
        generation_noise_floor: float = 1e-8,
        model_revisions: dict[str, str] | None = None,
    ) -> None:
        self.index = index
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.dense_top_b = dense_top_b
        self.rerank_top_k = rerank_top_k
        self.span_size = span_size
        self.span_offsets = span_offsets
        self.retrieval_noise_floor = retrieval_noise_floor
        self.generation_noise_floor = generation_noise_floor
        revisions = model_revisions or {}
        self.models = [
            ModelReference(
                role="retriever",
                name=str(retriever.model_path),
                revision=revisions.get("retriever", "unresolved"),
                config_hash=canonical_hash(
                    {"max_length": retriever.max_length, "normalize": retriever.normalize}
                ),
                local_path=str(retriever.model_path),
            ),
            ModelReference(
                role="reranker",
                name=str(reranker.model_path),
                revision=revisions.get("reranker", "unresolved"),
                config_hash=canonical_hash({"max_length": reranker.max_length}),
                local_path=str(reranker.model_path),
            ),
            ModelReference(
                role="generator",
                name=str(generator.model_path),
                revision=revisions.get("generator", "unresolved"),
                config_hash=canonical_hash(
                    {
                        "temperature": 0.0,
                        "do_sample": False,
                        "max_new_tokens": generator.max_new_tokens,
                        "attention_implementation": "eager",
                    }
                ),
                local_path=str(generator.model_path),
            ),
        ]

    @staticmethod
    def runtime_candidates(candidates: Sequence[Candidate]) -> tuple[RuntimeCandidate, ...]:
        """Strip source metadata, attack flags, and oracle ranges before detection."""

        return tuple(
            RuntimeCandidate(
                chunk=DetectorChunk(
                    chunk_id=candidate.chunk.chunk_id,
                    source_doc_id=candidate.chunk.source_doc_id,
                    text=candidate.chunk.text,
                    source_chars=candidate.chunk.source_chars,
                    source_tokens=candidate.chunk.source_tokens,
                    chunker_name=candidate.chunk.chunker_name,
                    chunker_hash=candidate.chunk.chunker_hash,
                ),
                dense_score=candidate.dense_score,
                dense_rank=candidate.dense_rank,
                rerank_score=candidate.rerank_score,
                rerank_rank=candidate.rerank_rank,
            )
            for candidate in candidates
        )

    def retrieve(
        self,
        query: str,
        *,
        injected_sources: Sequence[tuple[SourceDocument, list[CharRange], list[CharRange]]] = (),
    ) -> tuple[np.ndarray, list[Candidate], object]:
        query_embedding = self.retriever.encode([query])[0]
        clean_scores, indexed = self.index.search(query_embedding, self.dense_top_b)
        raw: list[tuple[SourceDocument, object, float, bool]] = [
            (entry.source, entry.chunk, float(score), False)
            for score, entry in zip(clean_scores, indexed, strict=True)
        ]
        for source, anchor_ranges, payload_ranges in injected_sources:
            chunks = chunk_source(
                source,
                self.generator.tokenizer,
                chunk_size=256,
                overlap=32,
                anchor_ranges=anchor_ranges,
                payload_ranges=payload_ranges,
            )
            embeddings = self.retriever.encode([chunk.text for chunk in chunks])
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                raw.append(
                    (
                        source,
                        chunk,
                        self.retriever.score(query_embedding, embedding),
                        True,
                    )
                )
        raw.sort(key=lambda item: (-item[2], str(item[1].chunk_id)))
        dense = raw[: self.dense_top_b]
        rerank_scores = self.reranker.score_pairs(query, [item[1].text for item in dense])
        rerank_order = sorted(
            range(len(dense)),
            key=lambda index: (-float(rerank_scores[index]), str(dense[index][1].chunk_id)),
        )
        rerank_ranks = {dense_index: rank + 1 for rank, dense_index in enumerate(rerank_order)}
        candidates = [
            Candidate(
                source=source,
                chunk=chunk,
                dense_score=float(score),
                dense_rank=index + 1,
                rerank_score=float(rerank_scores[index]),
                rerank_rank=rerank_ranks[index],
                injected=injected,
            )
            for index, (source, chunk, score, injected) in enumerate(dense)
        ]
        frozen = freeze_thresholds(
            np.asarray([candidate.dense_score for candidate in candidates]),
            np.asarray([candidate.rerank_score for candidate in candidates]),
            dense_top_b=min(self.dense_top_b, len(candidates)),
            rerank_top_k=min(self.rerank_top_k, len(candidates)),
        )
        selected = sorted(candidates, key=lambda item: item.rerank_rank)[: self.rerank_top_k]
        return query_embedding, selected, frozen

    def build_detector_event(
        self,
        *,
        query_id: str,
        query: str,
        dataset: str,
        split: str,
        selected: Sequence[Candidate],
        frozen: object,
        detector_code_commit: str,
        seed: int,
    ) -> tuple[DetectorEvent, object]:
        ordered = sorted(selected, key=lambda item: item.rerank_rank)
        layout = self.generator.build_prompt(
            query, [(candidate.chunk.chunk_id, candidate.chunk.text) for candidate in ordered]
        )
        shadow_answer, _ = self.generator.generate_shadow(layout)
        traces = [
            RetrievalTrace(
                chunk_id=candidate.chunk.chunk_id,
                dense_score=candidate.dense_score,
                dense_rank=candidate.dense_rank,
                rerank_score=candidate.rerank_score,
                rerank_rank=candidate.rerank_rank,
                dense_top_b=self.dense_top_b,
                rerank_top_k=self.rerank_top_k,
                dense_threshold=frozen.dense_threshold,
                rerank_threshold=frozen.rerank_threshold,
                context_position=position,
            )
            for position, candidate in enumerate(ordered)
        ]
        event = DetectorEvent(
            query=DetectorQuery(query_id=query_id, dataset=dataset, query=query, split=split),
            chunks=[
                DetectorChunk(
                    chunk_id=candidate.chunk.chunk_id,
                    source_doc_id=candidate.chunk.source_doc_id,
                    text=candidate.chunk.text,
                    source_chars=candidate.chunk.source_chars,
                    source_tokens=candidate.chunk.source_tokens,
                    chunker_name=candidate.chunk.chunker_name,
                    chunker_hash=candidate.chunk.chunker_hash,
                )
                for candidate in ordered
            ],
            retrieval_trace=traces,
            context_order=[candidate.chunk.chunk_id for candidate in ordered],
            shadow_answer=shadow_answer,
            models=self.models,
            detector_code_commit=detector_code_commit,
            seed=seed,
        )
        return event, layout

    def _retrieval_effects(
        self,
        *,
        query: str,
        query_embedding: np.ndarray,
        candidate: Candidate | RuntimeCandidate,
        frozen: object,
        spans: Sequence[object],
    ) -> list[float]:
        def hidden_score(span: object) -> tuple[float, float]:
            hidden_embedding = self.retriever.encode_hidden(candidate.chunk.text, span.char_range)
            dense_score = self.retriever.score(query_embedding, hidden_embedding)
            rerank_score = self.reranker.score_hidden(query, candidate.chunk.text, span.char_range)
            return dense_score, rerank_score

        values = attribute_retrieval(
            spans,
            full_dense_score=candidate.dense_score,
            full_rerank_score=candidate.rerank_score,
            hidden_score=hidden_score,
            frozen=frozen,
        )
        return [value.effect for value in values]

    def _score_role_maps(
        self,
        event: DetectorEvent,
        *,
        candidates: Sequence[Candidate | RuntimeCandidate],
        query_embedding: np.ndarray,
        frozen: object,
        layout: object,
        generation_chunk_ids: set[str] | None,
    ) -> QueryRoleResult:
        candidate_by_id = {candidate.chunk.chunk_id: candidate for candidate in candidates}
        retrieval_by_id: dict[str, tuple[list[object], list[float]]] = {}
        retrieval_totals: list[tuple[float, str]] = []
        for chunk_id in event.context_order:
            candidate = candidate_by_id[chunk_id]
            offsets = [
                (token.char_start, token.char_end)
                for token in tokenize_with_offsets(candidate.chunk.text, self.generator.tokenizer)
            ]
            spans = build_span_views(offsets, span_size=self.span_size, offsets=self.span_offsets)
            effects = self._retrieval_effects(
                query=event.query.query,
                query_embedding=query_embedding,
                candidate=candidate,
                frozen=frozen,
                spans=spans,
            )
            retrieval_by_id[chunk_id] = (spans, effects)
            retrieval_totals.append((sum(max(0.0, value) for value in effects), chunk_id))
        retrieval_totals.sort(key=lambda item: (-item[0], item[1]))
        generation_ids = (
            {chunk_id for _, chunk_id in retrieval_totals[:2]}
            if generation_chunk_ids is None
            else set(generation_chunk_ids)
        )
        results: list[ChunkRoleResult] = []
        full_mean_logp = self.generator.teacher_score(layout, event.shadow_answer)
        for chunk_id in event.context_order:
            if chunk_id not in generation_ids:
                continue
            spans, retrieval_effects = retrieval_by_id[chunk_id]
            generation = attribute_generation(
                spans,
                full_mean_logp=full_mean_logp,
                hidden_teacher_forced_score=lambda span, cid=chunk_id: self.generator.teacher_score(
                    layout,
                    event.shadow_answer,
                    chunk_id=cid,
                    hidden_span=span.char_range,
                ),
            )
            combined = aggregate_overlapping_span_effects(
                (
                    span.char_range,
                    retrieval_effect,
                    generation_effect_value.effect,
                )
                for span, retrieval_effect, generation_effect_value in zip(
                    spans, retrieval_effects, generation, strict=True
                )
            )
            role_map = compute_role_map(
                [item.retrieval_effect for item in combined],
                [item.generation_effect for item in combined],
                retrieval_noise_floor=self.retrieval_noise_floor,
                generation_noise_floor=self.generation_noise_floor,
            )
            span_records = tuple(
                SpanEffectRecord(
                    start=item.char_range.start,
                    end=item.char_range.end,
                    retrieval_effect=item.retrieval_effect,
                    generation_effect=item.generation_effect,
                    p_retrieval=float(role_map.p_retrieval[index]),
                    p_generation=float(role_map.p_generation[index]),
                    role=role_map.roles[index],
                )
                for index, item in enumerate(combined)
            )
            results.append(
                ChunkRoleResult(
                    chunk_id=chunk_id,
                    retrieval_mass=role_map.retrieval_mass,
                    generation_mass=role_map.generation_mass,
                    t_rg=role_map.t_rg,
                    role_identifiable=role_map.role_identifiable,
                    spans=span_records,
                )
            )
        return QueryRoleResult(
            query_id=event.query.query_id,
            shadow_answer=event.shadow_answer,
            query_score=max((result.t_rg for result in results), default=0.0),
            selected_for_generation=tuple(sorted(generation_ids)),
            chunks=tuple(results),
        )

    def score_detector_event(
        self,
        event: DetectorEvent,
        *,
        candidates: Sequence[RuntimeCandidate],
        query_embedding: np.ndarray,
        frozen: object,
        layout: object,
    ) -> QueryRoleResult:
        """Score the label-free runtime view using the frozen Top-5/Top-2 rule."""

        if any(not isinstance(candidate, RuntimeCandidate) for candidate in candidates):
            raise TypeError("runtime detector accepts only sanitized RuntimeCandidate objects")
        event_ids = set(event.context_order)
        candidate_ids = {candidate.chunk.chunk_id for candidate in candidates}
        if event_ids != candidate_ids:
            raise ValueError("runtime candidates do not exactly match the detector event")

        return self._score_role_maps(
            event,
            candidates=candidates,
            query_embedding=query_embedding,
            frozen=frozen,
            layout=layout,
            generation_chunk_ids=None,
        )

    def score_mechanism_chunks(
        self,
        event: DetectorEvent,
        *,
        candidates: Sequence[Candidate],
        query_embedding: np.ndarray,
        frozen: object,
        layout: object,
        evaluated_chunk_ids: Iterable[str],
    ) -> QueryRoleResult:
        """Mechanism-only evaluator; the runtime detector never calls this method."""

        identifiers = set(evaluated_chunk_ids)
        unknown = identifiers - {candidate.chunk.chunk_id for candidate in candidates}
        if unknown:
            raise ValueError(f"mechanism evaluator requested unknown chunks: {sorted(unknown)}")
        return self._score_role_maps(
            event,
            candidates=candidates,
            query_embedding=query_embedding,
            frozen=frozen,
            layout=layout,
            generation_chunk_ids=identifiers,
        )

    def no_op_deltas(
        self,
        *,
        query: str,
        query_embedding: np.ndarray,
        candidate: Candidate,
        frozen: object,
        layout: object,
        shadow_answer: str,
    ) -> tuple[float, float]:
        """Measure deterministic numerical noise without altering any token."""

        full_margin = pipeline_margin(candidate.dense_score, candidate.rerank_score, frozen)
        repeated_embedding = self.retriever.encode([candidate.chunk.text])[0]
        repeated_dense = self.retriever.score(query_embedding, repeated_embedding)
        repeated_rerank = float(self.reranker.score_pairs(query, [candidate.chunk.text])[0])
        repeated_margin = pipeline_margin(repeated_dense, repeated_rerank, frozen)
        retrieval_delta = abs(full_margin - repeated_margin)
        generation_a = self.generator.teacher_score(layout, shadow_answer)
        generation_b = self.generator.teacher_score(layout, shadow_answer)
        return retrieval_delta, abs(generation_effect(generation_a, generation_b))

    def score_donor_replacements(
        self,
        event: DetectorEvent,
        *,
        candidate: Candidate,
        query_embedding: np.ndarray,
        frozen: object,
        layout: object,
        donor_texts: Sequence[str],
    ) -> ChunkRoleResult:
        """Repeat attribution with exactly eight clean, same-token-length donors."""

        if len(donor_texts) != 8:
            raise ValueError("E03 requires exactly eight clean donors per span")
        offsets = [
            (token.char_start, token.char_end)
            for token in tokenize_with_offsets(candidate.chunk.text, self.generator.tokenizer)
        ]
        spans = build_span_views(offsets, span_size=self.span_size, offsets=self.span_offsets)
        full_margin = pipeline_margin(candidate.dense_score, candidate.rerank_score, frozen)
        full_logp = self.generator.teacher_score(layout, event.shadow_answer)
        averaged: list[tuple[CharRange, float, float]] = []
        for span in spans:
            retrieval_values: list[float] = []
            generation_values: list[float] = []
            for donor_text in donor_texts:
                embedding = self.retriever.encode_replaced(
                    candidate.chunk.text, span.char_range, donor_text
                )
                dense_score = self.retriever.score(query_embedding, embedding)
                rerank_score = self.reranker.score_replaced(
                    event.query.query, candidate.chunk.text, span.char_range, donor_text
                )
                retrieval_values.append(
                    full_margin - pipeline_margin(dense_score, rerank_score, frozen)
                )
                replacement_logp = self.generator.teacher_score(
                    layout,
                    event.shadow_answer,
                    chunk_id=candidate.chunk.chunk_id,
                    hidden_span=span.char_range,
                    donor_text=donor_text,
                )
                generation_values.append(generation_effect(full_logp, replacement_logp))
            averaged.append(
                (
                    span.char_range,
                    float(np.mean(retrieval_values)),
                    float(np.mean(generation_values)),
                )
            )
        combined = aggregate_overlapping_span_effects(averaged)
        role_map = compute_role_map(
            [item.retrieval_effect for item in combined],
            [item.generation_effect for item in combined],
            retrieval_noise_floor=self.retrieval_noise_floor,
            generation_noise_floor=self.generation_noise_floor,
        )
        return ChunkRoleResult(
            chunk_id=candidate.chunk.chunk_id,
            retrieval_mass=role_map.retrieval_mass,
            generation_mass=role_map.generation_mass,
            t_rg=role_map.t_rg,
            role_identifiable=role_map.role_identifiable,
            spans=tuple(
                SpanEffectRecord(
                    start=item.char_range.start,
                    end=item.char_range.end,
                    retrieval_effect=item.retrieval_effect,
                    generation_effect=item.generation_effect,
                    p_retrieval=float(role_map.p_retrieval[index]),
                    p_generation=float(role_map.p_generation[index]),
                    role=role_map.roles[index],
                )
                for index, item in enumerate(combined)
            ),
        )


def stable_event_hash(event: DetectorEvent) -> str:
    return hashlib.sha256(event.model_dump_json().encode("utf-8")).hexdigest()
