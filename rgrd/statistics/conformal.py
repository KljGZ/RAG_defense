from __future__ import annotations

import numpy as np


def conformal_pvalue(calibration_scores: list[float] | np.ndarray, test_score: float) -> float:
    calibration = np.asarray(calibration_scores, dtype=float)
    if calibration.ndim != 1 or len(calibration) == 0:
        raise ValueError("calibration scores must be a non-empty vector")
    if not np.all(np.isfinite(calibration)) or not np.isfinite(test_score):
        raise ValueError("conformal scores must be finite")
    return float((1 + np.count_nonzero(calibration >= float(test_score))) / (len(calibration) + 1))


def flag_from_calibration(
    calibration_scores: list[float] | np.ndarray,
    test_score: float,
    *,
    alpha: float = 0.05,
) -> tuple[float, bool]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    p_value = conformal_pvalue(calibration_scores, test_score)
    return p_value, p_value <= alpha
