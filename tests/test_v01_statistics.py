import pytest

from rgrd.statistics.v01 import (
    cliffs_delta,
    exact_one_sided_sign_test,
    outcome_association,
    query_bootstrap_median_ci,
)


def test_exact_sign_test_is_binomial_and_excludes_zeros() -> None:
    result = exact_one_sided_sign_test([1, 2, 3, 4, 5, 0])
    assert result.positive == 5
    assert result.zero == 1
    assert result.trials == 5
    assert result.p_value == pytest.approx(1 / 32)


def test_query_bootstrap_and_outcome_effects() -> None:
    interval = query_bootstrap_median_ci([1.0, 2.0, 3.0], replicates=200, seed=7)
    assert interval.estimate == 2.0
    association = outcome_association([4.0, 3.0, 0.0, -1.0], [True, True, False, False])
    assert association.difference_in_medians == pytest.approx(4.0)
    assert association.cliffs_delta == 1.0
    assert cliffs_delta([0.0], [0.0]) == 0.0
