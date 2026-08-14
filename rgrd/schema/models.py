from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AuditStatus(str, Enum):
    PASS_EXACT = "PASS_EXACT"
    PASS_FUNCTIONAL = "PASS_FUNCTIONAL"
    WARN = "WARN"
    FAIL = "FAIL"


class CharRange(StrictModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "CharRange":
        if self.end <= self.start:
            raise ValueError("range end must be greater than start")
        return self

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlap(self, other: "CharRange") -> int:
        return max(0, min(self.end, other.end) - max(self.start, other.start))


class TokenRange(CharRange):
    pass


class QueryMetadata(StrictModel):
    query_id: str
    dataset: str
    query: str
    split: str = "unknown"
    gold_answers: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectorQuery(StrictModel):
    query_id: str
    dataset: str
    query: str
    split: str


class SourceDocument(StrictModel):
    source_doc_id: str
    text: str
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    sha256: str | None = None

    @model_validator(mode="after")
    def fill_or_validate_hash(self) -> "SourceDocument":
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.sha256 is None:
            self.sha256 = digest
        elif self.sha256 != digest:
            raise ValueError("source sha256 does not match text")
        return self


class ChunkLineage(StrictModel):
    chunk_id: str
    source_doc_id: str
    text: str
    source_chars: CharRange
    source_tokens: TokenRange
    anchor_ranges_source: list[CharRange] = Field(default_factory=list)
    payload_ranges_source: list[CharRange] = Field(default_factory=list)
    anchor_ranges_chunk: list[CharRange] = Field(default_factory=list)
    payload_ranges_chunk: list[CharRange] = Field(default_factory=list)
    chunker_name: str
    chunker_hash: str

    @model_validator(mode="after")
    def validate_ranges(self) -> "ChunkLineage":
        if self.source_chars.length != len(self.text):
            raise ValueError("chunk text length must equal source character interval length")
        for name in ("anchor_ranges_chunk", "payload_ranges_chunk"):
            for span in getattr(self, name):
                if span.end > len(self.text):
                    raise ValueError(f"{name} exceeds chunk text")
        return self


class DetectorChunk(StrictModel):
    chunk_id: str
    source_doc_id: str
    text: str
    source_chars: CharRange
    source_tokens: TokenRange
    chunker_name: str
    chunker_hash: str


class RetrievalTrace(StrictModel):
    chunk_id: str
    dense_score: float
    dense_rank: int = Field(ge=1)
    rerank_score: float
    rerank_rank: int = Field(ge=1)
    dense_top_b: int = Field(ge=1)
    rerank_top_k: int = Field(ge=1)
    dense_threshold: float
    rerank_threshold: float
    context_position: int | None = Field(default=None, ge=0)


class ModelReference(StrictModel):
    role: Literal["chunker", "retriever", "reranker", "generator"]
    name: str
    revision: str
    config_hash: str
    local_path: str | None = None


class AttackAnnotations(StrictModel):
    attack_family: str | None = None
    attack_variant: str | None = None
    attack_target: str | None = None
    target_answer: str | None = None
    poison_hit: bool | None = None
    conditional_attack_success: bool | None = None
    end_to_end_attack_success: bool | None = None


class DetectorEvent(StrictModel):
    query: DetectorQuery
    chunks: list[DetectorChunk]
    retrieval_trace: list[RetrievalTrace]
    context_order: list[str]
    shadow_answer: str
    models: list[ModelReference]
    detector_code_commit: str
    seed: int


class RGRDEvent(StrictModel):
    query: QueryMetadata
    sources: list[SourceDocument]
    chunks: list[ChunkLineage]
    retrieval_trace: list[RetrievalTrace]
    context_order: list[str]
    clean_answer: str | None = None
    shadow_answer: str
    attacked_answer: str | None = None
    attack: AttackAnnotations = Field(default_factory=AttackAnnotations)
    models: list[ModelReference]
    attack_code_commit: str | None = None
    detector_code_commit: str
    seed: int

    @computed_field
    @property
    def source_ids(self) -> set[str]:
        return {source.source_doc_id for source in self.sources}

    @model_validator(mode="after")
    def validate_references(self) -> "RGRDEvent":
        chunk_ids = {chunk.chunk_id for chunk in self.chunks}
        if set(self.context_order) - chunk_ids:
            raise ValueError("context_order contains unknown chunk ids")
        if {trace.chunk_id for trace in self.retrieval_trace} - chunk_ids:
            raise ValueError("retrieval trace contains unknown chunk ids")
        if {chunk.source_doc_id for chunk in self.chunks} - self.source_ids:
            raise ValueError("chunk references an unknown source document")
        return self

    def detector_view(self) -> DetectorEvent:
        """Return the only event view accepted by runtime detector code.

        Gold/target answers, attack labels, and oracle spans are intentionally absent.
        """

        return DetectorEvent(
            query=DetectorQuery(
                query_id=self.query.query_id,
                dataset=self.query.dataset,
                query=self.query.query,
                split=self.query.split,
            ),
            chunks=[
                DetectorChunk(
                    chunk_id=chunk.chunk_id,
                    source_doc_id=chunk.source_doc_id,
                    text=chunk.text,
                    source_chars=chunk.source_chars,
                    source_tokens=chunk.source_tokens,
                    chunker_name=chunk.chunker_name,
                    chunker_hash=chunk.chunker_hash,
                )
                for chunk in self.chunks
            ],
            retrieval_trace=self.retrieval_trace,
            context_order=self.context_order,
            shadow_answer=self.shadow_answer,
            models=self.models,
            detector_code_commit=self.detector_code_commit,
            seed=self.seed,
        )
