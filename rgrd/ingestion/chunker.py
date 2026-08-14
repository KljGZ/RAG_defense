from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from rgrd.schema import CharRange, ChunkLineage, SourceDocument, TokenRange


class TokenizerProtocol(Protocol):
    name_or_path: str

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class OffsetToken:
    token_id: int
    token_index: int
    char_start: int
    char_end: int


class WhitespaceOffsetTokenizer:
    """Deterministic tokenizer fixture with a Hugging Face-compatible call shape."""

    name_or_path = "rgrd/whitespace-offset-v1"

    def __call__(self, text: str, **_: Any) -> dict[str, Any]:
        offsets: list[tuple[int, int]] = []
        ids: list[int] = []
        for match in re.finditer(r"\S+", text):
            token = match.group(0)
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            ids.append(int.from_bytes(digest, "big"))
            offsets.append((match.start(), match.end()))
        return {"input_ids": ids, "offset_mapping": offsets}


def tokenize_with_offsets(text: str, tokenizer: TokenizerProtocol) -> list[OffsetToken]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("tokenizer must provide offset_mapping")
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    if len(ids) != len(offsets):
        raise ValueError("input_ids and offset_mapping lengths differ")
    tokens = [
        OffsetToken(int(token_id), index, int(start), int(end))
        for index, (token_id, (start, end)) in enumerate(zip(ids, offsets, strict=True))
        if int(end) > int(start)
    ]
    return tokens


def _chunker_hash(tokenizer_name: str, chunk_size: int, overlap: int) -> str:
    payload = json.dumps(
        {"tokenizer": tokenizer_name, "chunk_size": chunk_size, "overlap": overlap},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _project_ranges(
    source_ranges: list[CharRange], chunk_start: int, chunk_end: int
) -> tuple[list[CharRange], list[CharRange]]:
    source_overlap: list[CharRange] = []
    chunk_overlap: list[CharRange] = []
    for span in source_ranges:
        start = max(span.start, chunk_start)
        end = min(span.end, chunk_end)
        if end <= start:
            continue
        source_overlap.append(CharRange(start=start, end=end))
        chunk_overlap.append(CharRange(start=start - chunk_start, end=end - chunk_start))
    return source_overlap, chunk_overlap


def chunk_source(
    source: SourceDocument,
    tokenizer: TokenizerProtocol,
    *,
    chunk_size: int = 256,
    overlap: int = 32,
    anchor_ranges: list[CharRange] | None = None,
    payload_ranges: list[CharRange] | None = None,
) -> list[ChunkLineage]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    tokens = tokenize_with_offsets(source.text, tokenizer)
    if not tokens:
        return []
    anchor_ranges = anchor_ranges or []
    payload_ranges = payload_ranges or []
    step = chunk_size - overlap
    tokenizer_name = str(getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__))
    config_hash = _chunker_hash(tokenizer_name, chunk_size, overlap)
    chunks: list[ChunkLineage] = []
    for token_start in range(0, len(tokens), step):
        token_end = min(token_start + chunk_size, len(tokens))
        window = tokens[token_start:token_end]
        char_start = window[0].char_start
        char_end = window[-1].char_end
        text = source.text[char_start:char_end]
        anchor_source, anchor_chunk = _project_ranges(anchor_ranges, char_start, char_end)
        payload_source, payload_chunk = _project_ranges(payload_ranges, char_start, char_end)
        identity = f"{source.source_doc_id}:{token_start}:{token_end}:{config_hash}"
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        chunks.append(
            ChunkLineage(
                chunk_id=chunk_id,
                source_doc_id=source.source_doc_id,
                text=text,
                source_chars=CharRange(start=char_start, end=char_end),
                source_tokens=TokenRange(start=token_start, end=token_end),
                anchor_ranges_source=anchor_source,
                payload_ranges_source=payload_source,
                anchor_ranges_chunk=anchor_chunk,
                payload_ranges_chunk=payload_chunk,
                chunker_name=tokenizer_name,
                chunker_hash=config_hash,
            )
        )
        if token_end == len(tokens):
            break
    return chunks
