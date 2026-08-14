from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class PermutationResult:
    statistic: float
    p_value: float
    permutations: int


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    replicates: int


def paired_sign_flip_test(
    differences: Iterable[float],
    *,
    permutations: int = 10000,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = lambda values: float(np.median(values)),
) -> PermutationResult:
    values = np.asarray(list(differences), dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a finite non-empty vector")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    observed = float(statistic(values))
    rng = np.random.default_rng(seed)
    exceedances = 0
    # Chunking bounds memory while keeping the preregistered number of draws exact.
    remaining = permutations
    while remaining:
        batch = min(2048, remaining)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(batch, len(values)))
        permuted = signs * values
        permuted_stats = np.apply_along_axis(statistic, 1, permuted)
        exceedances += int(np.count_nonzero(permuted_stats >= observed))
        remaining -= batch
    return PermutationResult(
        statistic=observed,
        p_value=float((1 + exceedances) / (permutations + 1)),
        permutations=permutations,
    )


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    for value in p_values.values():
        if not 0 <= value <= 1:
            raise ValueError("p-values must lie in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted_ordered: list[tuple[str, float]] = []
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        adjusted = min(1.0, (count - index) * value)
        running = max(running, adjusted)
        adjusted_ordered.append((name, running))
    return dict(adjusted_ordered)


def cluster_bootstrap_median_ci(
    differences: Iterable[float],
    cluster_ids: Iterable[str],
    *,
    replicates: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapInterval:
    values = np.asarray(list(differences), dtype=float)
    clusters = np.asarray(list(cluster_ids), dtype=object)
    if values.shape != clusters.shape or len(values) == 0:
        raise ValueError("differences and cluster_ids must have equal non-zero length")
    unique = np.unique(clusters)
    grouped = {cluster: values[clusters == cluster] for cluster in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([grouped[cluster] for cluster in selected])
        draws[index] = float(np.median(sample))
    return BootstrapInterval(
        estimate=float(np.median(values)),
        lower=float(np.quantile(draws, alpha / 2)),
        upper=float(np.quantile(draws, 1 - alpha / 2)),
        replicates=replicates,
    )
