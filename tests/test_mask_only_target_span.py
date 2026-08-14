import pytest

from rgrd.intervention import hide_token_span


def test_only_target_span_is_hidden() -> None:
    original_ids = [101, 11, 12, 13, 14, 102]
    original_mask = [1] * len(original_ids)
    positions = list(range(len(original_ids)))
    hidden = hide_token_span(
        original_ids,
        original_mask,
        positions,
        start=2,
        end=4,
        neutral_token_id=0,
        allowed_indices=range(1, 5),
    )
    assert hidden.input_ids == [101, 11, 0, 0, 14, 102]
    assert hidden.attention_mask == [1, 1, 0, 0, 1, 1]
    assert original_ids == [101, 11, 12, 13, 14, 102]
    assert hidden.hidden_indices == (2, 3)
    with pytest.raises(ValueError):
        hide_token_span(
            original_ids,
            original_mask,
            positions,
            start=0,
            end=2,
            neutral_token_id=0,
            allowed_indices=range(1, 5),
        )
