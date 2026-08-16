from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol, TypeVar


class SampleLike(Protocol):
    sample_id: str
    query_id: str


SampleT = TypeVar("SampleT", bound=SampleLike)


def normalize_answer(value: str) -> str:
    """Normalization used only for validity/exclusion checks, never for scoring."""

    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


@dataclass(frozen=True)
class CanonicalSelection:
    loaded_samples: int
    unique_query_ids: int
    canonical_samples: int
    duplicate_samples_discarded: int
    rule: str = "lexicographically smallest sample_id within each query_id"


def canonical_per_query(samples: Iterable[SampleT]) -> tuple[list[SampleT], CanonicalSelection]:
    """Select one outcome-independent sample per query.

    The only tie-breaker is ``sample_id``.  Retrieval score, model output, attack
    success, family contrast, and artifact order are intentionally unavailable to
    this function.
    """

    rows = list(samples)
    by_query: dict[str, list[SampleT]] = {}
    for sample in rows:
        query_id = str(sample.query_id).strip()
        sample_id = str(sample.sample_id).strip()
        if not query_id or not sample_id:
            raise ValueError("canonical selection requires non-empty query_id and sample_id")
        by_query.setdefault(query_id, []).append(sample)
    canonical = [
        sorted(group, key=lambda item: str(item.sample_id))[0]
        for _, group in sorted(by_query.items())
    ]
    audit = CanonicalSelection(
        loaded_samples=len(rows),
        unique_query_ids=len(by_query),
        canonical_samples=len(canonical),
        duplicate_samples_discarded=len(rows) - len(canonical),
    )
    return canonical, audit


def valid_distinct_answers(target: str, gold_answers: Iterable[str]) -> tuple[str, ...]:
    """Return valid gold aliases and reject target/gold identity."""

    normalized_target = normalize_answer(target)
    if not normalized_target:
        raise ValueError("attack target is empty after normalization")
    aliases: list[str] = []
    seen: set[str] = set()
    for answer in gold_answers:
        normalized = normalize_answer(answer)
        if not normalized or normalized in seen:
            continue
        if normalized == normalized_target:
            raise ValueError("attack target is identical to a gold answer after normalization")
        seen.add(normalized)
        aliases.append(str(answer).strip())
    if not aliases:
        raise ValueError("no valid gold answer aliases")
    return tuple(aliases)
