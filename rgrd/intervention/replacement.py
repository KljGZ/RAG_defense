from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .mask import _clone, _length


@dataclass(frozen=True)
class ReplacedInputs:
    input_ids: Any
    attention_mask: Any
    position_ids: Any
    replaced_indices: tuple[int, ...]


def replace_token_span(
    input_ids: Any,
    attention_mask: Any,
    position_ids: Any,
    *,
    start: int,
    end: int,
    donor_token_ids: Sequence[int],
) -> ReplacedInputs:
    sequence_length = _length(input_ids)
    if not (0 <= start < end <= sequence_length):
        raise ValueError("invalid replacement span")
    if len(donor_token_ids) != end - start:
        raise ValueError("donor replacement must have exactly the target token length")
    if _length(attention_mask) != sequence_length or _length(position_ids) != sequence_length:
        raise ValueError("input, attention, and position lengths must match")
    ids_out = _clone(input_ids)
    mask_out = _clone(attention_mask)
    positions_out = _clone(position_ids)
    if hasattr(ids_out, "shape") and len(ids_out.shape) > 1:
        ids_out[..., start:end] = ids_out.new_tensor(donor_token_ids)
    else:
        ids_out[start:end] = list(donor_token_ids)
    if _length(ids_out) != sequence_length:
        raise AssertionError("donor intervention changed sequence length")
    return ReplacedInputs(
        input_ids=ids_out,
        attention_mask=mask_out,
        position_ids=positions_out,
        replaced_indices=tuple(range(start, end)),
    )
