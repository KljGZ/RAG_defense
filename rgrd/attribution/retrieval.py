from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

from rgrd.retrieval import FrozenRetrievalThresholds, pipeline_margin


SpanT = TypeVar("SpanT")


@dataclass(frozen=True)
class RetrievalAttribution:
    span: object
    full_margin: float
    hidden_margin: float
    effect: float


def attribute_retrieval(
    spans: Iterable[SpanT],
    *,
    full_dense_score: float,
    full_rerank_score: float,
    hidden_score: Callable[[SpanT], tuple[float, float]],
    frozen: FrozenRetrievalThresholds,
) -> list[RetrievalAttribution]:
    """Attribute the frozen two-stage retrieval bottleneck to each span.

    `hidden_score` must return both dense and reranker scores even when the hidden
    chunk would leave dense Top-B. This prevents selection-induced score leakage.
    """

    full_margin = pipeline_margin(full_dense_score, full_rerank_score, frozen)
    results: list[RetrievalAttribution] = []
    for span in spans:
        dense_score, rerank_score = hidden_score(span)
        hidden_margin = pipeline_margin(dense_score, rerank_score, frozen)
        results.append(
            RetrievalAttribution(
                span=span,
                full_margin=full_margin,
                hidden_margin=hidden_margin,
                effect=full_margin - hidden_margin,
            )
        )
    return results
