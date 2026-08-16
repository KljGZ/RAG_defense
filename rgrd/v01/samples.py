from __future__ import annotations

from pathlib import Path

from rgrd.attacks import (
    load_poisonedrag_blackbox,
    load_poisonedrag_whitebox,
)
from rgrd.attacks.artifacts import AttackSample


ACTIVE_FAMILIES = ("PoisonedRAG-B", "PoisonedRAG-W")


def load_v01_samples(
    root: Path,
    attack_root: Path,
    family: str,
    *,
    dataset: str = "nq",
) -> list[AttackSample]:
    if family == "PoisonedRAG-B":
        samples = load_poisonedrag_blackbox(attack_root / "results/query_results/main")
    elif family == "PoisonedRAG-W":
        samples = load_poisonedrag_whitebox(
            root / "artifacts/attacks/poisonedrag_w_nq100.json"
        )
    else:
        raise ValueError(f"V0.1 excludes non-active family {family!r}")
    return [sample for sample in samples if sample.dataset == dataset]
