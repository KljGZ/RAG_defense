from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from rgrd.indexing import ExactIndexBundle
from rgrd.ingestion import tokenize_with_offsets
from rgrd.schema import CharRange
from rgrd.v01.protocol import normalize_answer


@dataclass(frozen=True)
class DonorSegment:
    faiss_id: int
    chunk_id: str
    source_doc_id: str
    text: str
    token_length: int
    chunk_token_start: int
    chunk_token_end: int


@dataclass(frozen=True)
class DonorPair:
    replicate: int
    anchor: tuple[DonorSegment, ...]
    payload: tuple[DonorSegment, ...]


def token_length(tokenizer: object, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    values = encoded["input_ids"]
    if values and isinstance(values[0], list):
        values = values[0]
    return len(values)


def oracle_token_lengths(
    tokenizer: object,
    text: str,
    ranges: Sequence[CharRange],
) -> tuple[int, ...]:
    lengths = tuple(token_length(tokenizer, text[span.start : span.end]) for span in ranges)
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError("every Oracle group must contain at least one tokenizer token")
    return lengths


def validate_disjoint_ranges(
    text: str,
    anchor_ranges: Sequence[CharRange],
    payload_ranges: Sequence[CharRange],
) -> None:
    labelled = [(span, "anchor") for span in anchor_ranges] + [
        (span, "payload") for span in payload_ranges
    ]
    if not anchor_ranges or not payload_ranges:
        raise ValueError("both Oracle groups must be non-empty")
    ordered = sorted(labelled, key=lambda item: (item[0].start, item[0].end, item[1]))
    for span, _ in ordered:
        if span.end > len(text):
            raise ValueError("Oracle range exceeds the co-chunk text")
    for (left, left_name), (right, right_name) in zip(ordered, ordered[1:]):
        if left.end > right.start:
            raise ValueError(f"Oracle ranges overlap: {left_name} and {right_name}")


def replace_oracle_groups(
    text: str,
    *,
    anchor_ranges: Sequence[CharRange],
    payload_ranges: Sequence[CharRange],
    anchor_replacements: Sequence[str] | None,
    payload_replacements: Sequence[str] | None,
) -> str:
    """Replace absent Oracle players while retaining all residual chunk text."""

    validate_disjoint_ranges(text, anchor_ranges, payload_ranges)
    replacements: list[tuple[CharRange, str]] = []
    for ranges, values, label in (
        (anchor_ranges, anchor_replacements, "anchor"),
        (payload_ranges, payload_replacements, "payload"),
    ):
        if values is None:
            continue
        if len(values) != len(ranges):
            raise ValueError(f"{label} donor count does not match Oracle ranges")
        replacements.extend(zip(ranges, (str(value) for value in values), strict=True))
    result = text
    for span, replacement in sorted(replacements, key=lambda item: item[0].start, reverse=True):
        result = result[: span.start] + replacement + result[span.end :]
    return result


class DeterministicDonorSampler:
    """Select benign same-token-length donors from different indexed documents."""

    def __init__(
        self,
        index: ExactIndexBundle,
        tokenizer: object,
        *,
        seed: int,
        maximum_attempts: int = 100000,
    ) -> None:
        self.index = index
        self.tokenizer = tokenizer
        self.seed = int(seed)
        self.maximum_attempts = int(maximum_attempts)

    def _rng(self, sample_id: str) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}\0{sample_id}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _segment(
        self,
        *,
        length: int,
        rng: random.Random,
        used_sources: set[str],
        excluded_sources: set[str],
        forbidden_phrases: tuple[str, ...],
        attempted_ids: set[int],
    ) -> DonorSegment:
        for _ in range(self.maximum_attempts):
            faiss_id = rng.randrange(self.index.chunk_count)
            if faiss_id in attempted_ids:
                continue
            attempted_ids.add(faiss_id)
            entry = self.index.fetch_chunks([faiss_id])[0]
            source_id = entry.source.source_doc_id
            if source_id in used_sources or source_id in excluded_sources:
                continue
            tokens = tokenize_with_offsets(entry.chunk.text, self.tokenizer)
            if len(tokens) < length:
                continue
            possible = list(range(0, len(tokens) - length + 1))
            rng.shuffle(possible)
            for start in possible[: min(32, len(possible))]:
                window = tokens[start : start + length]
                donor = entry.chunk.text[window[0].char_start : window[-1].char_end]
                normalized = normalize_answer(donor)
                if not normalized or any(value in normalized for value in forbidden_phrases):
                    continue
                if token_length(self.tokenizer, donor) != length:
                    continue
                used_sources.add(source_id)
                return DonorSegment(
                    faiss_id=faiss_id,
                    chunk_id=entry.chunk.chunk_id,
                    source_doc_id=source_id,
                    text=donor,
                    token_length=length,
                    chunk_token_start=start,
                    chunk_token_end=start + length,
                )
        raise RuntimeError(
            f"could not find an exact {length}-token benign donor after "
            f"{self.maximum_attempts} attempts"
        )

    def sample_pairs(
        self,
        *,
        sample_id: str,
        anchor_lengths: Sequence[int],
        payload_lengths: Sequence[int],
        excluded_source_ids: Iterable[str],
        forbidden_texts: Iterable[str],
        replicates: int = 8,
    ) -> tuple[DonorPair, ...]:
        if replicates != 8:
            raise ValueError("RGRD-V0.1 requires exactly eight donor pairs")
        if not anchor_lengths or not payload_lengths:
            raise ValueError("donor sampling requires both Oracle groups")
        if any(int(value) <= 0 for value in (*anchor_lengths, *payload_lengths)):
            raise ValueError("donor token lengths must be positive")
        forbidden = tuple(
            value
            for value in (normalize_answer(text) for text in forbidden_texts)
            if value
        )
        excluded = {str(value) for value in excluded_source_ids}
        used_sources: set[str] = set()
        attempted_ids: set[int] = set()
        rng = self._rng(sample_id)
        pairs: list[DonorPair] = []
        for replicate in range(replicates):
            anchor = tuple(
                self._segment(
                    length=int(length),
                    rng=rng,
                    used_sources=used_sources,
                    excluded_sources=excluded,
                    forbidden_phrases=forbidden,
                    attempted_ids=attempted_ids,
                )
                for length in anchor_lengths
            )
            payload = tuple(
                self._segment(
                    length=int(length),
                    rng=rng,
                    used_sources=used_sources,
                    excluded_sources=excluded,
                    forbidden_phrases=forbidden,
                    attempted_ids=attempted_ids,
                )
                for length in payload_lengths
            )
            pairs.append(DonorPair(replicate=replicate, anchor=anchor, payload=payload))
        return tuple(pairs)
