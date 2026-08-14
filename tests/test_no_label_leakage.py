from dataclasses import fields

from rgrd.pipeline.track_b import Candidate, RuntimeCandidate, TrackBPipeline
from rgrd.schema import (
    AttackAnnotations,
    CharRange,
    ChunkLineage,
    ModelReference,
    QueryMetadata,
    RGRDEvent,
    RetrievalTrace,
    SourceDocument,
    TokenRange,
)


FORBIDDEN = {
    "attack_family",
    "attack_variant",
    "attack_target",
    "target_answer",
    "gold_answers",
    "anchor_ranges_source",
    "payload_ranges_source",
    "anchor_ranges_chunk",
    "payload_ranges_chunk",
    "clean_answer",
    "attacked_answer",
    "end_to_end_attack_success",
}


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def test_runtime_detector_view_has_no_label_or_oracle_fields() -> None:
    source = SourceDocument(source_doc_id="d", text="anchor payload")
    chunk = ChunkLineage(
        chunk_id="c",
        source_doc_id="d",
        text="anchor payload",
        source_chars=CharRange(start=0, end=14),
        source_tokens=TokenRange(start=0, end=2),
        anchor_ranges_source=[CharRange(start=0, end=6)],
        payload_ranges_source=[CharRange(start=7, end=14)],
        anchor_ranges_chunk=[CharRange(start=0, end=6)],
        payload_ranges_chunk=[CharRange(start=7, end=14)],
        chunker_name="fixture",
        chunker_hash="a" * 64,
    )
    event = RGRDEvent(
        query=QueryMetadata(
            query_id="q",
            dataset="nq",
            query="question",
            gold_answers=["gold"],
            metadata={"target_answer": "hidden", "attack_family": "hidden"},
        ),
        sources=[source],
        chunks=[chunk],
        retrieval_trace=[
            RetrievalTrace(
                chunk_id="c",
                dense_score=1,
                dense_rank=1,
                rerank_score=1,
                rerank_rank=1,
                dense_top_b=50,
                rerank_top_k=5,
                dense_threshold=0,
                rerank_threshold=0,
                context_position=0,
            )
        ],
        context_order=["c"],
        clean_answer="gold",
        shadow_answer="target",
        attacked_answer="target",
        attack=AttackAnnotations(
            attack_family="PoisonedRAG",
            target_answer="target",
            end_to_end_attack_success=True,
        ),
        models=[
            ModelReference(
                role="generator",
                name="fixture",
                revision="r",
                config_hash="b" * 64,
            )
        ],
        detector_code_commit="deadbeef",
        seed=1,
    )
    detector_dump = event.detector_view().model_dump()
    assert not (FORBIDDEN & _all_keys(detector_dump))


def test_runtime_candidate_boundary_strips_oracle_and_source_metadata() -> None:
    source = SourceDocument(
        source_doc_id="d",
        text="anchor payload",
        metadata={"attack_family": "hidden", "target_answer": "hidden"},
    )
    chunk = ChunkLineage(
        chunk_id="c",
        source_doc_id="d",
        text=source.text,
        source_chars=CharRange(start=0, end=len(source.text)),
        source_tokens=TokenRange(start=0, end=2),
        anchor_ranges_source=[CharRange(start=0, end=6)],
        payload_ranges_source=[CharRange(start=7, end=14)],
        anchor_ranges_chunk=[CharRange(start=0, end=6)],
        payload_ranges_chunk=[CharRange(start=7, end=14)],
        chunker_name="fixture",
        chunker_hash="a" * 64,
    )
    candidate = Candidate(
        source=source,
        chunk=chunk,
        dense_score=1.0,
        dense_rank=1,
        rerank_score=1.0,
        rerank_rank=1,
        injected=True,
    )
    runtime = TrackBPipeline.runtime_candidates([candidate])
    assert len(runtime) == 1
    assert isinstance(runtime[0], RuntimeCandidate)
    assert {field.name for field in fields(RuntimeCandidate)} == {
        "chunk",
        "dense_score",
        "dense_rank",
        "rerank_score",
        "rerank_rank",
    }
    assert not (FORBIDDEN & _all_keys(runtime[0].chunk.model_dump()))
    assert not hasattr(runtime[0], "source")
    assert not hasattr(runtime[0], "injected")
