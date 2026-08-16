from __future__ import annotations

import hashlib
import random
import re
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
    model_token_lengths: dict[str, int]
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
        target_text: str,
        rng: random.Random,
        used_sources: set[str],
        excluded_sources: set[str],
        forbidden_phrases: tuple[str, ...],
        attempted_ids: set[int],
        minimum_requirements: Sequence[tuple[str, object, int]],
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
            leading = re.match(r"\s*", target_text).group(0)
            trailing = re.search(r"\s*$", target_text).group(0)
            candidate_lengths = range(max(1, length - 2), length + 3)
            if len(tokens) < min(candidate_lengths):
                continue
            lengths = list(candidate_lengths)
            rng.shuffle(lengths)
            for window_length in lengths:
                if len(tokens) < window_length:
                    continue
                possible = list(range(0, len(tokens) - window_length + 1))
                rng.shuffle(possible)
                for start in possible[: min(32, len(possible))]:
                    window = tokens[start : start + window_length]
                    core = entry.chunk.text[window[0].char_start : window[-1].char_end]
                    donor = leading + core + trailing
                    normalized = normalize_answer(donor)
                    if not normalized or any(value in normalized for value in forbidden_phrases):
                        continue
                    generator_length = token_length(self.tokenizer, donor)
                    if generator_length != length:
                        continue
                    model_lengths = {"generator": generator_length}
                    requirement_failed = False
                    for name, tokenizer, minimum in minimum_requirements:
                        observed = token_length(tokenizer, donor)
                        model_lengths[name] = observed
                        if observed < int(minimum):
                            requirement_failed = True
                            break
                    if requirement_failed:
                        continue
                    used_sources.add(source_id)
                    return DonorSegment(
                        faiss_id=faiss_id,
                        chunk_id=entry.chunk.chunk_id,
                        source_doc_id=source_id,
                        text=donor,
                        token_length=length,
                        model_token_lengths=model_lengths,
                        chunk_token_start=start,
                        chunk_token_end=start + window_length,
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
        original_text: str,
        anchor_ranges: Sequence[CharRange],
        payload_ranges: Sequence[CharRange],
        excluded_source_ids: Iterable[str],
        forbidden_texts: Iterable[str],
        replicates: int = 8,
        anchor_minimum_requirements: Sequence[Sequence[tuple[str, object, int]]] | None = None,
        payload_minimum_requirements: Sequence[Sequence[tuple[str, object, int]]] | None = None,
    ) -> tuple[DonorPair, ...]:
        if replicates != 8:
            raise ValueError("RGRD-V0.1 requires exactly eight donor pairs")
        if not anchor_lengths or not payload_lengths:
            raise ValueError("donor sampling requires both Oracle groups")
        if any(int(value) <= 0 for value in (*anchor_lengths, *payload_lengths)):
            raise ValueError("donor token lengths must be positive")
        forbidden = tuple(
            value for value in (normalize_answer(text) for text in forbidden_texts) if value
        )
        excluded = {str(value) for value in excluded_source_ids}
        used_sources: set[str] = set()
        attempted_ids: set[int] = set()
        rng = self._rng(sample_id)
        pairs: list[DonorPair] = []
        if len(anchor_lengths) != len(anchor_ranges) or len(payload_lengths) != len(payload_ranges):
            raise ValueError("Oracle token lengths and ranges disagree")
        anchor_requirements = (
            tuple(() for _ in anchor_lengths)
            if anchor_minimum_requirements is None
            else tuple(tuple(values) for values in anchor_minimum_requirements)
        )
        payload_requirements = (
            tuple(() for _ in payload_lengths)
            if payload_minimum_requirements is None
            else tuple(tuple(values) for values in payload_minimum_requirements)
        )
        if len(anchor_requirements) != len(anchor_lengths) or len(payload_requirements) != len(
            payload_lengths
        ):
            raise ValueError("donor minimum requirements and Oracle ranges disagree")
        if any(
            int(minimum) <= 0
            for requirements in (*anchor_requirements, *payload_requirements)
            for _, _, minimum in requirements
        ):
            raise ValueError("donor minimum token requirements must be positive")
        validate_disjoint_ranges(original_text, anchor_ranges, payload_ranges)
        anchor_targets = [original_text[span.start : span.end] for span in anchor_ranges]
        payload_targets = [original_text[span.start : span.end] for span in payload_ranges]
        for replicate in range(replicates):
            anchor = tuple(
                self._segment(
                    length=int(length),
                    target_text=target,
                    rng=rng,
                    used_sources=used_sources,
                    excluded_sources=excluded,
                    forbidden_phrases=forbidden,
                    attempted_ids=attempted_ids,
                    minimum_requirements=requirements,
                )
                for length, target, requirements in zip(
                    anchor_lengths,
                    anchor_targets,
                    anchor_requirements,
                    strict=True,
                )
            )
            payload = tuple(
                self._segment(
                    length=int(length),
                    target_text=target,
                    rng=rng,
                    used_sources=used_sources,
                    excluded_sources=excluded,
                    forbidden_phrases=forbidden,
                    attempted_ids=attempted_ids,
                    minimum_requirements=requirements,
                )
                for length, target, requirements in zip(
                    payload_lengths,
                    payload_targets,
                    payload_requirements,
                    strict=True,
                )
            )
            pairs.append(DonorPair(replicate=replicate, anchor=anchor, payload=payload))
        return tuple(pairs)
