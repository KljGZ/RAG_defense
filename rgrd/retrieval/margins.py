from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _mad(values: np.ndarray, epsilon: float) -> float:
    center = float(np.median(values))
    value = float(np.median(np.abs(values - center)))
    return max(value, epsilon)


def _descending_boundary(values: np.ndarray, k: int) -> float:
    if values.ndim != 1 or len(values) < k or k <= 0:
        raise ValueError("scores must be one-dimensional and contain at least k values")
    return float(np.sort(values)[::-1][k - 1])


@dataclass(frozen=True)
class FrozenRetrievalThresholds:
    dense_threshold: float
    rerank_threshold: float
    dense_mad: float
    rerank_mad: float
    dense_top_b: int
    rerank_top_k: int


def freeze_thresholds(
    dense_scores: list[float] | np.ndarray,
    original_top_b_rerank_scores: list[float] | np.ndarray,
    *,
    dense_top_b: int,
    rerank_top_k: int,
    epsilon: float = 1e-8,
) -> FrozenRetrievalThresholds:
    dense = np.asarray(dense_scores, dtype=float)
    rerank = np.asarray(original_top_b_rerank_scores, dtype=float)
    if not np.all(np.isfinite(dense)) or not np.all(np.isfinite(rerank)):
        raise ValueError("retrieval scores must be finite")
    return FrozenRetrievalThresholds(
        dense_threshold=_descending_boundary(dense, dense_top_b),
        rerank_threshold=_descending_boundary(rerank, rerank_top_k),
        dense_mad=_mad(dense, epsilon),
        rerank_mad=_mad(rerank, epsilon),
        dense_top_b=dense_top_b,
        rerank_top_k=rerank_top_k,
    )


def pipeline_margin(
    dense_score: float,
    rerank_score: float,
    frozen: FrozenRetrievalThresholds,
) -> float:
    """Robust bottleneck margin against thresholds frozen on the full candidates."""

    dense_z = (float(dense_score) - frozen.dense_threshold) / frozen.dense_mad
    rerank_z = (float(rerank_score) - frozen.rerank_threshold) / frozen.rerank_mad
    return min(dense_z, rerank_z)
