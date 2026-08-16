from dataclasses import dataclass

import pytest

from rgrd.pipeline.state import GateDecision, GateStatus
from rgrd.v01.protocol import canonical_per_query, valid_distinct_answers


@dataclass(frozen=True)
class _Sample:
    query_id: str
    sample_id: str
    retrieval_success: bool
    generation_success: bool


def test_canonical_selection_is_sample_id_only() -> None:
    rows = [
        _Sample("q1", "z", True, True),
        _Sample("q1", "a", False, False),
        _Sample("q2", "b", False, True),
    ]
    selected, audit = canonical_per_query(rows)
    assert [(row.query_id, row.sample_id) for row in selected] == [("q1", "a"), ("q2", "b")]
    assert audit.loaded_samples == 3
    assert audit.unique_query_ids == 2
    assert audit.duplicate_samples_discarded == 1


def test_target_must_be_distinct_from_gold() -> None:
    assert valid_distinct_answers("wrong", ["right", "RIGHT", "also right"]) == (
        "right",
        "also right",
    )
    with pytest.raises(ValueError, match="identical"):
        valid_distinct_answers("New-York", ["new york"])


def test_gate_status_preserves_legacy_flag_but_supports_not_estimable() -> None:
    legacy = GateDecision(gate="gate_3", passed=False)
    assert legacy.status == GateStatus.FAIL_TESTED
    corrected = GateDecision(gate="gate_v01", status=GateStatus.NOT_ESTIMABLE)
    assert corrected.passed is False
    with pytest.raises(ValueError, match="disagree"):
        GateDecision(gate="gate_v01", status=GateStatus.NOT_ESTIMABLE, passed=True)
