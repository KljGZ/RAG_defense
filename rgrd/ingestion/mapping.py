from __future__ import annotations

from dataclasses import dataclass

from rgrd.ingestion.chunker import TokenizerProtocol, tokenize_with_offsets
from rgrd.schema import CharRange


@dataclass(frozen=True)
class PromptTokenMap:
    chunk_chars_in_prompt: CharRange
    chunk_token_indices: tuple[int, ...]
    token_char_ranges_in_chunk: tuple[CharRange, ...]


def project_source_ranges(
    source_ranges: list[CharRange], chunk_source_range: CharRange
) -> list[CharRange]:
    projected: list[CharRange] = []
    for span in source_ranges:
        start = max(span.start, chunk_source_range.start)
        end = min(span.end, chunk_source_range.end)
        if end > start:
            projected.append(
                CharRange(
                    start=start - chunk_source_range.start,
                    end=end - chunk_source_range.start,
                )
            )
    return projected


def build_prompt_token_map(
    prompt: str,
    chunk_text: str,
    tokenizer: TokenizerProtocol,
    *,
    occurrence: int = 0,
) -> PromptTokenMap:
    if occurrence < 0:
        raise ValueError("occurrence must be non-negative")
    starts: list[int] = []
    cursor = 0
    while True:
        found = prompt.find(chunk_text, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + max(1, len(chunk_text))
    if occurrence >= len(starts):
        raise ValueError("chunk text occurrence not found in prompt")
    chunk_start = starts[occurrence]
    chunk_end = chunk_start + len(chunk_text)
    prompt_tokens = tokenize_with_offsets(prompt, tokenizer)
    indices: list[int] = []
    chunk_ranges: list[CharRange] = []
    for token in prompt_tokens:
        overlap_start = max(token.char_start, chunk_start)
        overlap_end = min(token.char_end, chunk_end)
        if overlap_end <= overlap_start:
            continue
        indices.append(token.token_index)
        chunk_ranges.append(
            CharRange(start=overlap_start - chunk_start, end=overlap_end - chunk_start)
        )
    return PromptTokenMap(
        chunk_chars_in_prompt=CharRange(start=chunk_start, end=chunk_end),
        chunk_token_indices=tuple(indices),
        token_char_ranges_in_chunk=tuple(chunk_ranges),
    )
