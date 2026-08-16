"""RGRD-V0.1 mechanism-audit primitives.

The V0.1 package is deliberately separate from the legacy V0 detector code.  It
implements an Oracle A/P mechanism audit and must not be imported by the
label-free runtime detector.
"""

from rgrd.v01.protocol import CanonicalSelection, canonical_per_query, normalize_answer
from rgrd.v01.shapley import (
    CoalitionValues,
    OracleContrast,
    aggregate_oracle_replicates,
    oracle_contrast,
)

__all__ = [
    "CanonicalSelection",
    "CoalitionValues",
    "OracleContrast",
    "aggregate_oracle_replicates",
    "canonical_per_query",
    "normalize_answer",
    "oracle_contrast",
]
