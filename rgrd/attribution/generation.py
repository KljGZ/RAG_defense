from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

from rgrd.generation import generation_effect


SpanT = TypeVar("SpanT")


@dataclass(frozen=True)
class GenerationAttribution:
    span: object
    full_mean_logp: float
    hidden_mean_logp: float
    effect: float


def attribute_generation(
    spans: Iterable[SpanT],
    *,
    full_mean_logp: float,
    hidden_teacher_forced_score: Callable[[SpanT], float],
) -> list[GenerationAttribution]:
    """Attribute support for one fixed, parsed shadow answer to each span."""

    results: list[GenerationAttribution] = []
    for span in spans:
        hidden_logp = float(hidden_teacher_forced_score(span))
        results.append(
            GenerationAttribution(
                span=span,
                full_mean_logp=float(full_mean_logp),
                hidden_mean_logp=hidden_logp,
                effect=generation_effect(full_mean_logp, hidden_logp),
            )
        )
    return results
