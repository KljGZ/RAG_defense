from types import SimpleNamespace

import pytest

from rgrd.models.adapters import (
    GenerationAudit,
    _resolve_generation_eos_token_ids,
    _token_id_tuple,
)
from rgrd.v01.engine import GenerationFormatError, deterministic_generation


def _audit(
    *,
    continuation: str = " Charles Darwin",
    token_ids: tuple[int, ...] = (100, 151643),
    terminated: bool = True,
    truncated: bool = False,
) -> GenerationAudit:
    return GenerationAudit(
        answer=continuation.strip(),
        continuation=continuation,
        generated_tokens=len(token_ids),
        generated_token_ids=token_ids,
        accepted_eos_token_ids=(151645, 151643),
        termination_token_id=token_ids[-1] if terminated else None,
        terminated_by_eos=terminated,
        truncated=truncated,
        strict_single_line=True,
    )


def test_checkpoint_multi_eos_contract_is_not_narrowed_to_tokenizer_eos() -> None:
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[151645, 151643]))
    tokenizer = SimpleNamespace(eos_token_id=151645)
    assert _resolve_generation_eos_token_ids(model, tokenizer) == (151645, 151643)


def test_tokenizer_eos_is_used_only_as_fallback_and_ids_are_deduplicated() -> None:
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=None))
    tokenizer = SimpleNamespace(eos_token_id=[7, 7, 8])
    assert _resolve_generation_eos_token_ids(model, tokenizer) == (7, 8)
    assert _token_id_tuple(None) == ()
    with pytest.raises(ValueError, match="non-negative"):
        _token_id_tuple(-1)


def test_deterministic_generation_accepts_checkpoint_secondary_eos() -> None:
    audit = _audit()
    generator = SimpleNamespace(generate_shadow_audited=lambda _layout: audit)
    result = deterministic_generation(generator, object())
    assert result["termination_token_id"] == 151643
    assert result["accepted_eos_token_ids"] == (151645, 151643)


def test_deterministic_format_failure_carries_forensic_token_evidence() -> None:
    audit = _audit(
        continuation=" Albert Einstein continued beyond the answer budget",
        token_ids=tuple(range(32)),
        terminated=False,
        truncated=True,
    )
    generator = SimpleNamespace(generate_shadow_audited=lambda _layout: audit)
    with pytest.raises(GenerationFormatError) as raised:
        deterministic_generation(generator, object())
    assert raised.value.code == "max_new_tokens_without_eos"
    assert raised.value.evidence()["audit"]["generated_token_ids"] == tuple(range(32))


def test_determinism_mismatch_remains_fatal_instead_of_query_attrition() -> None:
    audits = iter((_audit(), _audit(continuation=" Albert Einstein")))
    generator = SimpleNamespace(generate_shadow_audited=lambda _layout: next(audits))
    with pytest.raises(RuntimeError, match="strict deterministic") as raised:
        deterministic_generation(generator, object())
    assert not isinstance(raised.value, GenerationFormatError)
