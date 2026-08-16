import pytest

from rgrd.models.adapters import _partition_overlap_indices
from rgrd.schema import CharRange


def test_boundary_token_has_one_owner_and_tie_goes_to_earlier_span() -> None:
    offsets = [(0, 0), (0, 2), (2, 4), (4, 6)]
    spans = [CharRange(start=0, end=3), CharRange(start=3, end=6)]

    partitions = _partition_overlap_indices(offsets, spans)

    assert partitions == ((1, 2), (3,))
    assert set(partitions[0]).isdisjoint(partitions[1])


def test_boundary_token_uses_maximum_character_overlap() -> None:
    offsets = [(0, 2), (2, 5), (5, 7)]
    spans = [CharRange(start=0, end=3), CharRange(start=3, end=7)]

    assert _partition_overlap_indices(offsets, spans) == ((0,), (1, 2))


def test_partition_rejects_a_span_without_a_unique_model_token() -> None:
    offsets = [(0, 4)]
    spans = [CharRange(start=0, end=2), CharRange(start=2, end=4)]

    with pytest.raises(ValueError, match="missing ordered span indices"):
        _partition_overlap_indices(offsets, spans)
