import numpy as np
import pytest

from rgrd.attribution import compute_role_map
from rgrd.experiments.gates import _finite_positive_total


def test_positive_total_clips_negative_finite_effects() -> None:
    assert _finite_positive_total([-2.0, 0.5, 1.5], label="test") == pytest.approx(2.0)


def test_positive_total_rejects_nan_instead_of_silently_zeroing() -> None:
    with pytest.raises(FloatingPointError, match=r"indices \[1\]"):
        _finite_positive_total([0.0, np.nan], label="generation")


def test_role_map_reports_nonfinite_channel_and_index() -> None:
    with pytest.raises(ValueError, match=r"generation_nonfinite=\[1\]"):
        compute_role_map([1.0, 2.0], [1.0, np.inf])
