from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from rgrd.schema import CharRange


@dataclass(frozen=True)
class AtomicSpanEffect:
    char_range: CharRange
    retrieval_effect: float
    generation_effect: float


@dataclass(frozen=True)
class RoleMap:
    p_retrieval: np.ndarray
    p_generation: np.ndarray
    retrieval_mass: float
    generation_mass: float
    t_rg: float
    role_identifiable: bool
    roles: tuple[Literal["anchor-like", "payload-like", "mixed"], ...]


def compute_role_map(
    retrieval_effects: Iterable[float],
    generation_effects: Iterable[float],
    *,
    retrieval_noise_floor: float = 0.0,
    generation_noise_floor: float = 0.0,
    comparison_tolerance: float = 1e-12,
) -> RoleMap:
    retrieval = np.asarray(list(retrieval_effects), dtype=float)
    generation = np.asarray(list(generation_effects), dtype=float)
    if retrieval.ndim != 1 or retrieval.shape != generation.shape or len(retrieval) == 0:
        raise ValueError("retrieval and generation effects must be equal non-empty vectors")
    retrieval_nonfinite = np.flatnonzero(~np.isfinite(retrieval)).tolist()
    generation_nonfinite = np.flatnonzero(~np.isfinite(generation)).tolist()
    if retrieval_nonfinite or generation_nonfinite:
        raise ValueError(
            "role effects must be finite; "
            f"retrieval_nonfinite={retrieval_nonfinite}; "
            f"generation_nonfinite={generation_nonfinite}"
        )
    positive_r = np.clip(retrieval, 0.0, None)
    positive_g = np.clip(generation, 0.0, None)
    total_r = float(positive_r.sum())
    total_g = float(positive_g.sum())
    identifiable = total_r > retrieval_noise_floor and total_g > generation_noise_floor
    if not identifiable:
        zeros = np.zeros_like(positive_r)
        return RoleMap(
            p_retrieval=zeros,
            p_generation=zeros.copy(),
            retrieval_mass=total_r,
            generation_mass=total_g,
            t_rg=0.0,
            role_identifiable=False,
            roles=tuple("mixed" for _ in positive_r),
        )
    p_r = positive_r / total_r
    p_g = positive_g / total_g
    t_rg = float(0.5 * np.abs(p_r - p_g).sum())
    t_rg = min(1.0, max(0.0, t_rg))
    roles: list[Literal["anchor-like", "payload-like", "mixed"]] = []
    for r_value, g_value in zip(p_r, p_g, strict=True):
        if r_value > g_value + comparison_tolerance:
            roles.append("anchor-like")
        elif g_value > r_value + comparison_tolerance:
            roles.append("payload-like")
        else:
            roles.append("mixed")
    return RoleMap(
        p_retrieval=p_r,
        p_generation=p_g,
        retrieval_mass=total_r,
        generation_mass=total_g,
        t_rg=t_rg,
        role_identifiable=True,
        roles=tuple(roles),
    )


def aggregate_overlapping_span_effects(
    spans: Iterable[tuple[CharRange, float, float]],
) -> list[AtomicSpanEffect]:
    spans = list(spans)
    if not spans:
        return []
    boundaries = sorted({point for span, _, _ in spans for point in (span.start, span.end)})
    atoms: list[AtomicSpanEffect] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end <= start:
            continue
        covering = [entry for entry in spans if entry[0].start < end and entry[0].end > start]
        if not covering:
            continue
        atoms.append(
            AtomicSpanEffect(
                char_range=CharRange(start=start, end=end),
                retrieval_effect=float(np.mean([entry[1] for entry in covering])),
                generation_effect=float(np.mean([entry[2] for entry in covering])),
            )
        )
    return atoms


def oracle_mass(
    atomic_ranges: Iterable[CharRange],
    distribution: Iterable[float],
    oracle_ranges: Iterable[CharRange],
) -> float:
    atoms = list(atomic_ranges)
    probabilities = np.asarray(list(distribution), dtype=float)
    oracle = list(oracle_ranges)
    if len(atoms) != len(probabilities):
        raise ValueError("atomic range and probability lengths differ")
    mass = 0.0
    for atom, probability in zip(atoms, probabilities, strict=True):
        covered = sum(atom.overlap(span) for span in oracle)
        covered = min(covered, atom.length)
        mass += float(probability) * (covered / atom.length)
    return float(mass)
