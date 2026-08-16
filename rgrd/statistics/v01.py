from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from rgrd.statistics.tests import BootstrapInterval, PermutationResult


@dataclass(frozen=True)
class ExactSignResult:
    positive: int
    negative: int
    zero: int
    trials: int
    positive_proportion: float
    p_value: float


@dataclass(frozen=True)
class AssociationResult:
    difference_in_medians: float
    cliffs_delta: float
    successes: int
    failures: int


def exact_one_sided_sign_test(values: Iterable[float]) -> ExactSignResult:
    """Exact H0: P(X>0)=0.5 against H1: P(X>0)>0.5; zeros are excluded."""

    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("sign-test values must be a finite non-empty vector")
    positive = int(np.count_nonzero(array > 0.0))
    negative = int(np.count_nonzero(array < 0.0))
    zero = int(np.count_nonzero(array == 0.0))
    trials = positive + negative
    if trials == 0:
        return ExactSignResult(positive, negative, zero, 0, float("nan"), 1.0)
    tail = sum(math.comb(trials, k) for k in range(positive, trials + 1)) / (2**trials)
    return ExactSignResult(
        positive=positive,
        negative=negative,
        zero=zero,
        trials=trials,
        positive_proportion=float(positive / trials),
        p_value=float(min(1.0, tail)),
    )


def query_bootstrap_median_ci(
    values: Iterable[float],
    *,
    replicates: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapInterval:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be a finite non-empty vector")
    if replicates < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    draws = np.median(
        rng.choice(array, size=(replicates, len(array)), replace=True), axis=1
    )
    return BootstrapInterval(
        estimate=float(np.median(array)),
        lower=float(np.quantile(draws, alpha / 2.0)),
        upper=float(np.quantile(draws, 1.0 - alpha / 2.0)),
        replicates=replicates,
    )


def cliffs_delta(success: Iterable[float], failure: Iterable[float]) -> float:
    first = np.asarray(list(success), dtype=float)
    second = np.asarray(list(failure), dtype=float)
    if len(first) == 0 or len(second) == 0:
        raise ValueError("Cliff's delta requires both outcome groups")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Cliff's delta requires finite values")
    comparisons = first[:, None] - second[None, :]
    return float((np.count_nonzero(comparisons > 0) - np.count_nonzero(comparisons < 0)) / comparisons.size)


def outcome_association(values: Iterable[float], outcomes: Iterable[bool]) -> AssociationResult:
    array = np.asarray(list(values), dtype=float)
    labels = np.asarray(list(outcomes), dtype=bool)
    if array.shape != labels.shape or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("association values/outcomes must be finite vectors of equal length")
    success = array[labels]
    failure = array[~labels]
    if len(success) == 0 or len(failure) == 0:
        raise ValueError("association is not estimable without both outcome classes")
    return AssociationResult(
        difference_in_medians=float(np.median(success) - np.median(failure)),
        cliffs_delta=cliffs_delta(success, failure),
        successes=len(success),
        failures=len(failure),
    )


def outcome_label_permutation_test(
    values: Iterable[float],
    outcomes: Iterable[bool],
    *,
    permutations: int = 10000,
    seed: int = 0,
) -> PermutationResult:
    array = np.asarray(list(values), dtype=float)
    labels = np.asarray(list(outcomes), dtype=bool)
    observed = outcome_association(array, labels).difference_in_medians
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        statistic = float(np.median(array[permuted]) - np.median(array[~permuted]))
        exceedances += int(statistic >= observed)
    return PermutationResult(
        statistic=observed,
        p_value=float((exceedances + 1) / (permutations + 1)),
        permutations=permutations,
    )


def query_bootstrap_association_ci(
    values: Iterable[float],
    outcomes: Iterable[bool],
    *,
    replicates: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapInterval:
    array = np.asarray(list(values), dtype=float)
    labels = np.asarray(list(outcomes), dtype=bool)
    observed = outcome_association(array, labels).difference_in_medians
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    attempts = 0
    maximum_attempts = replicates * 20
    while len(draws) < replicates and attempts < maximum_attempts:
        indices = rng.integers(0, len(array), size=len(array))
        sampled_labels = labels[indices]
        attempts += 1
        if sampled_labels.all() or (~sampled_labels).all():
            continue
        sampled = array[indices]
        draws.append(
            float(np.median(sampled[sampled_labels]) - np.median(sampled[~sampled_labels]))
        )
    if len(draws) < replicates:
        raise RuntimeError("could not draw enough bootstrap samples containing both outcomes")
    distribution = np.asarray(draws, dtype=float)
    return BootstrapInterval(
        estimate=observed,
        lower=float(np.quantile(distribution, alpha / 2.0)),
        upper=float(np.quantile(distribution, 1.0 - alpha / 2.0)),
        replicates=replicates,
    )
