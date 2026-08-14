from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def _clone(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone()
    if hasattr(value, "copy"):
        return value.copy()
    return list(value)


def _length(value: Any) -> int:
    if hasattr(value, "shape") and len(value.shape) > 1:
        return int(value.shape[-1])
    return len(value)


def _set_range(value: Any, start: int, end: int, replacement: int) -> None:
    if hasattr(value, "shape") and len(value.shape) > 1:
        value[..., start:end] = replacement
    else:
        value[start:end] = [replacement] * (end - start)


@dataclass(frozen=True)
class HiddenInputs:
    input_ids: Any
    attention_mask: Any
    position_ids: Any
    hidden_indices: tuple[int, ...]


def hide_token_span(
    input_ids: Any,
    attention_mask: Any,
    position_ids: Any,
    *,
    start: int,
    end: int,
    neutral_token_id: int,
    allowed_indices: Iterable[int] | None = None,
) -> HiddenInputs:
    sequence_length = _length(input_ids)
    if not (0 <= start < end <= sequence_length):
        raise ValueError("invalid hidden span")
    if _length(attention_mask) != sequence_length or _length(position_ids) != sequence_length:
        raise ValueError("input, attention, and position lengths must match")
    hidden = set(range(start, end))
    if allowed_indices is not None and not hidden.issubset(set(allowed_indices)):
        raise ValueError("hidden span includes protected query/special tokens")
    ids_out = _clone(input_ids)
    mask_out = _clone(attention_mask)
    positions_out = _clone(position_ids)
    _set_range(ids_out, start, end, neutral_token_id)
    _set_range(mask_out, start, end, 0)
    if _length(ids_out) != sequence_length or _length(mask_out) != sequence_length:
        raise AssertionError("span hiding changed sequence length")
    return HiddenInputs(
        input_ids=ids_out,
        attention_mask=mask_out,
        position_ids=positions_out,
        hidden_indices=tuple(range(start, end)),
    )
