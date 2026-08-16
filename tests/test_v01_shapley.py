import pytest

from rgrd.v01.shapley import (
    CoalitionValues,
    aggregate_oracle_replicates,
    direction_agreement,
    oracle_contrast,
)


def test_two_player_shapley_and_contrasts_preserve_signed_scale() -> None:
    retrieval = CoalitionValues(empty=0.0, anchor=4.0, payload=-1.0, both=5.0)
    generation = CoalitionValues(empty=-2.0, anchor=-1.0, payload=3.0, both=6.0)
    result = oracle_contrast(retrieval, generation)
    assert result.phi_anchor_retrieval == pytest.approx(5.0)
    assert result.phi_payload_retrieval == pytest.approx(0.0)
    assert result.c_retrieval == pytest.approx(5.0)
    assert result.phi_anchor_generation == pytest.approx(2.0)
    assert result.phi_payload_generation == pytest.approx(6.0)
    assert result.c_generation == pytest.approx(4.0)
    # Efficiency is a direct invariant of the four coalition values.
    assert result.phi_anchor_retrieval + result.phi_payload_retrieval == pytest.approx(5.0)
    assert result.phi_anchor_generation + result.phi_payload_generation == pytest.approx(8.0)


def test_donor_aggregation_requires_six_of_eight() -> None:
    row = oracle_contrast(
        CoalitionValues(0.0, 2.0, 0.0, 2.0),
        CoalitionValues(0.0, 0.0, 3.0, 3.0),
    )
    assert aggregate_oracle_replicates([row] * 6).role_identifiable
    assert not aggregate_oracle_replicates([row] * 5).role_identifiable
    assert direction_agreement([1.0] * 7 + [-1.0], 2.0) == pytest.approx(0.875)
