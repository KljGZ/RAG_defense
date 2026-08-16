from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CoalitionValues:
    """Value function for two players: anchor A and payload P."""

    empty: float
    anchor: float
    payload: float
    both: float

    def finite(self) -> bool:
        return bool(np.all(np.isfinite([self.empty, self.anchor, self.payload, self.both])))


@dataclass(frozen=True)
class OracleContrast:
    phi_anchor_retrieval: float
    phi_payload_retrieval: float
    phi_anchor_generation: float
    phi_payload_generation: float
    c_retrieval: float
    c_generation: float


@dataclass(frozen=True)
class AggregatedOracleContrast:
    c_retrieval: float
    c_generation: float
    valid_replicates: int
    total_replicates: int
    role_identifiable: bool
    retrieval_positive_fraction: float
    generation_positive_fraction: float


def _shapley(values: CoalitionValues) -> tuple[float, float]:
    if not values.finite():
        raise FloatingPointError("coalition value contains a non-finite score")
    phi_anchor = 0.5 * (
        (values.anchor - values.empty) + (values.both - values.payload)
    )
    phi_payload = 0.5 * (
        (values.payload - values.empty) + (values.both - values.anchor)
    )
    if not np.all(np.isfinite([phi_anchor, phi_payload])):
        raise FloatingPointError("Shapley attribution is non-finite")
    return float(phi_anchor), float(phi_payload)


def oracle_contrast(
    retrieval: CoalitionValues,
    generation: CoalitionValues,
) -> OracleContrast:
    """Compute scale-preserving Oracle contrasts without clipping or normalization."""

    phi_ar, phi_pr = _shapley(retrieval)
    phi_ag, phi_pg = _shapley(generation)
    return OracleContrast(
        phi_anchor_retrieval=phi_ar,
        phi_payload_retrieval=phi_pr,
        phi_anchor_generation=phi_ag,
        phi_payload_generation=phi_pg,
        c_retrieval=float(phi_ar - phi_pr),
        c_generation=float(phi_pg - phi_ag),
    )


def aggregate_oracle_replicates(
    replicates: Iterable[OracleContrast],
    *,
    total_replicates: int = 8,
    minimum_valid: int = 6,
) -> AggregatedOracleContrast:
    """Median aggregation for the preregistered donor-replacement analysis."""

    rows = [
        row
        for row in replicates
        if np.all(np.isfinite([row.c_retrieval, row.c_generation]))
    ]
    if total_replicates <= 0 or not 1 <= minimum_valid <= total_replicates:
        raise ValueError("invalid donor replicate contract")
    if len(rows) > total_replicates:
        raise ValueError("more valid replicates than the preregistered total")
    if rows:
        retrieval = np.asarray([row.c_retrieval for row in rows], dtype=float)
        generation = np.asarray([row.c_generation for row in rows], dtype=float)
        c_retrieval = float(np.median(retrieval))
        c_generation = float(np.median(generation))
        retrieval_positive = float(np.mean(retrieval > 0.0))
        generation_positive = float(np.mean(generation > 0.0))
    else:
        c_retrieval = float("nan")
        c_generation = float("nan")
        retrieval_positive = float("nan")
        generation_positive = float("nan")
    identifiable = (
        len(rows) >= minimum_valid
        and np.isfinite(c_retrieval)
        and np.isfinite(c_generation)
    )
    return AggregatedOracleContrast(
        c_retrieval=c_retrieval,
        c_generation=c_generation,
        valid_replicates=len(rows),
        total_replicates=total_replicates,
        role_identifiable=bool(identifiable),
        retrieval_positive_fraction=retrieval_positive,
        generation_positive_fraction=generation_positive,
    )


def direction_agreement(values: Iterable[float], reference: float) -> float:
    """Fraction of non-zero donor directions matching a non-zero mask direction."""

    array = np.asarray(list(values), dtype=float)
    if not np.isfinite(reference) or reference == 0.0:
        return float("nan")
    array = array[np.isfinite(array) & (array != 0.0)]
    if len(array) == 0:
        return float("nan")
    return float(np.mean(np.sign(array) == np.sign(reference)))
