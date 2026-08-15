from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


FAMILY_DATASETS = {
    "PoisonedRAG-B": "nq",
    "PoisonedRAG-W": "nq",
    "Phantom": "msmarco",
}


@dataclass(frozen=True)
class ProtocolScope:
    protocol_id: str
    amendment_id: str
    prior_failed_run: str
    amendment_reason: str
    claim_boundary: str
    active_attack_families: tuple[str, ...]
    excluded_attack_families: tuple[str, ...]
    mechanism_quotas: dict[str, int]
    detection_quotas: dict[str, int]

    @property
    def family_datasets(self) -> tuple[tuple[str, str], ...]:
        return tuple((family, FAMILY_DATASETS[family]) for family in self.active_attack_families)

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "amendment_id": self.amendment_id,
            "prior_failed_run": self.prior_failed_run,
            "amendment_reason": self.amendment_reason,
            "claim_boundary": self.claim_boundary,
            "active_attack_families": list(self.active_attack_families),
            "excluded_attack_families": list(self.excluded_attack_families),
            "mechanism_quotas": dict(self.mechanism_quotas),
            "detection_quotas": dict(self.detection_quotas),
        }


def _positive_quotas(value: object, *, field: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a family-to-positive-integer mapping")
    quotas = {str(family): int(count) for family, count in value.items()}
    if any(count <= 0 for count in quotas.values()):
        raise ValueError(f"{field} quotas must all be positive")
    return quotas


def load_protocol_scope(root: Path) -> ProtocolScope:
    config = yaml.safe_load(
        (root / "configs/experiments/v0_preregistration.yaml").read_text(encoding="utf-8")
    )
    protocol = config.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("missing explicit protocol scope")
    active = tuple(str(value) for value in protocol.get("active_attack_families", []))
    excluded = tuple(str(value) for value in protocol.get("excluded_attack_families", []))
    known = set(FAMILY_DATASETS)
    if not active:
        raise ValueError("protocol must retain at least one active attack family")
    if len(active) != len(set(active)) or len(excluded) != len(set(excluded)):
        raise ValueError("protocol family lists contain duplicates")
    if set(active) & set(excluded):
        raise ValueError("active and excluded attack families overlap")
    if set(active) | set(excluded) != known:
        raise ValueError("protocol scope must account for every known modular attack family")
    mechanism = _positive_quotas(
        config["e02_oracle_mechanism"].get("family_success_quotas"),
        field="e02_oracle_mechanism.family_success_quotas",
    )
    detection = _positive_quotas(
        config["e04_conformal"].get("attack_success_quotas"),
        field="e04_conformal.attack_success_quotas",
    )
    if set(mechanism) != set(active):
        raise ValueError("mechanism quotas must exactly match active attack families")
    if set(detection) != set(active):
        raise ValueError("detection quotas must exactly match active attack families")
    return ProtocolScope(
        protocol_id=str(protocol["id"]),
        amendment_id=str(protocol["amendment_id"]),
        prior_failed_run=str(protocol["prior_failed_run"]),
        amendment_reason=str(protocol["amendment_reason"]),
        claim_boundary=str(protocol["claim_boundary"]),
        active_attack_families=active,
        excluded_attack_families=excluded,
        mechanism_quotas=mechanism,
        detection_quotas=detection,
    )
