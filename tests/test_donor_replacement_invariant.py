from rgrd.intervention import replace_token_span


def test_donor_replacement_preserves_length_mask_and_positions() -> None:
    result = replace_token_span(
        [10, 11, 12, 13, 14],
        [1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4],
        start=1,
        end=3,
        donor_token_ids=[91, 92],
    )
    assert result.input_ids == [10, 91, 92, 13, 14]
    assert result.attention_mask == [1, 1, 1, 1, 1]
    assert result.position_ids == [0, 1, 2, 3, 4]
