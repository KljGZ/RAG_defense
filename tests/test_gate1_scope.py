from rgrd.audit.scan import ComponentAudit, _gate_1_decision


def _component(status: str) -> ComponentAudit:
    return ComponentAudit(status=status, reasons=[], evidence={})


def test_phantom_warn_is_informational_under_pa001() -> None:
    decision = _gate_1_decision(
        {
            "poisonedrag": _component("PASS_FUNCTIONAL"),
            "phantom": _component("WARN"),
        },
        {
            "protocol_amendment_id": "PA-001-exclude-phantom",
            "required_components": ["poisonedrag"],
            "informational_components": ["phantom"],
        },
    )

    assert decision["status"] == "PASS"
    assert decision["required_components"] == ["poisonedrag"]
    assert decision["informational_components"] == ["phantom"]
    assert any("informational-only" in reason for reason in decision["reasons"])


def test_original_two_component_gate_still_fails_for_phantom_warn() -> None:
    decision = _gate_1_decision(
        {
            "poisonedrag": _component("PASS_FUNCTIONAL"),
            "phantom": _component("WARN"),
        },
        None,
    )

    assert decision["status"] == "FAIL"
