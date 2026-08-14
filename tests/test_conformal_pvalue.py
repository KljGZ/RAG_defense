import pytest

from rgrd.statistics import conformal_pvalue, flag_from_calibration


def test_conformal_upper_tail_with_ties() -> None:
    calibration = [0.1, 0.2, 0.2, 0.4]
    assert conformal_pvalue(calibration, 0.2) == pytest.approx(4 / 5)
    assert conformal_pvalue(calibration, 0.5) == pytest.approx(1 / 5)
    p_value, flagged = flag_from_calibration(calibration, 0.5, alpha=0.2)
    assert p_value == pytest.approx(0.2)
    assert flagged
