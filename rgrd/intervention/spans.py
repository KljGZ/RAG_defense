from __future__ import annotations

from dataclasses import dataclass

from rgrd.schema import CharRange, TokenRange


@dataclass(frozen=True)
class SpanWindow:
    token_range: TokenRange
    char_range: CharRange
    view_offset: int


def build_span_views(
    token_offsets: list[tuple[int, int]],
    *,
    span_size: int = 32,
    offsets: tuple[int, ...] = (0, 16),
) -> list[SpanWindow]:
    if span_size <= 0:
        raise ValueError("span_size must be positive")
    if any(offset < 0 or offset >= span_size for offset in offsets):
        raise ValueError("view offsets must be in [0, span_size)")
    n_tokens = len(token_offsets)
    spans: list[SpanWindow] = []
    seen: set[tuple[int, int, int]] = set()
    for view_offset in offsets:
        start = view_offset
        while start < n_tokens:
            end = min(start + span_size, n_tokens)
            if end <= start:
                break
            key = (start, end, view_offset)
            if key not in seen:
                char_start = int(token_offsets[start][0])
                char_end = int(token_offsets[end - 1][1])
                if char_end > char_start:
                    spans.append(
                        SpanWindow(
                            token_range=TokenRange(start=start, end=end),
                            char_range=CharRange(start=char_start, end=char_end),
                            view_offset=view_offset,
                        )
                    )
                    seen.add(key)
            if end == n_tokens:
                break
            start += span_size
    return spans
