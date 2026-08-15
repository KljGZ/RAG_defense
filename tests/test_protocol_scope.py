from pathlib import Path

from rgrd.experiments.scope import load_protocol_scope


def test_posthoc_scope_excludes_phantom_consistently() -> None:
    root = Path(__file__).resolve().parents[1]
    scope = load_protocol_scope(root)

    assert scope.amendment_id == "PA-001-exclude-phantom"
    assert scope.active_attack_families == ("PoisonedRAG-B", "PoisonedRAG-W")
    assert scope.excluded_attack_families == ("Phantom",)
    assert scope.family_datasets == (("PoisonedRAG-B", "nq"), ("PoisonedRAG-W", "nq"))
    assert scope.mechanism_quotas == {"PoisonedRAG-B": 100, "PoisonedRAG-W": 100}
    assert scope.detection_quotas == {"PoisonedRAG-B": 50, "PoisonedRAG-W": 50}


def test_scope_metadata_preserves_prior_failed_run_and_claim_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = load_protocol_scope(root).metadata()

    assert metadata["prior_failed_run"] == "rgrd-v0-20260814T190433Z"
    assert "Post-hoc" in metadata["claim_boundary"]
    assert metadata["excluded_attack_families"] == ["Phantom"]
