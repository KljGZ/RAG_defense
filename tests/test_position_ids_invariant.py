from rgrd.intervention import hide_token_span


def test_position_ids_and_later_positions_are_invariant() -> None:
    ids = list(range(10))
    mask = [1] * 10
    positions = list(range(100, 110))
    hidden = hide_token_span(
        ids,
        mask,
        positions,
        start=3,
        end=6,
        neutral_token_id=99,
    )
    assert hidden.position_ids == positions
    assert hidden.input_ids[6:] == ids[6:]
    assert len(hidden.input_ids) == len(ids)
