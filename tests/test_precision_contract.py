from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rgrd.attacks.poisonedrag_whitebox import (
    CUBLAS_WORKSPACE_CONFIG,
    GENERATION_PROTOCOL,
    GENERATOR_ATTENTION_IMPLEMENTATION,
    GENERATOR_DTYPE,
)
from rgrd.models.adapters import CausalAnswerGenerator, _resolve_model_dtype


class _FakeCuda:
    def __init__(self, supported: bool) -> None:
        self.supported = supported

    def is_bf16_supported(self) -> bool:
        return self.supported


class _FakeTorch:
    bfloat16 = object()
    float16 = object()
    float32 = object()

    def __init__(self, supported: bool = True) -> None:
        self.cuda = _FakeCuda(supported)


def test_pipeline_declares_native_qwen_precision() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/pipeline/v0.yaml").read_text(encoding="utf-8"))
    generator = config["models"]["generator"]
    assert generator["dtype"] == "bfloat16"
    assert generator["attention_implementation"] == "eager"


def test_track_a_declares_same_strict_precision_contract() -> None:
    assert GENERATOR_DTYPE == "bfloat16"
    assert GENERATOR_ATTENTION_IMPLEMENTATION == "eager"
    assert CUBLAS_WORKSPACE_CONFIG == ":4096:8"
    assert GENERATION_PROTOCOL == "rgrd-track-a-qwen-chat-v3-bf16-deterministic"


def test_precision_metadata_records_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    generator = object.__new__(CausalAnswerGenerator)
    generator.requested_dtype = "bfloat16"
    generator.model_dtype = "bfloat16"
    generator.attention_implementation = "eager"
    assert generator.precision_metadata() == {
        "requested_dtype": "bfloat16",
        "loaded_dtype": "bfloat16",
        "attention_implementation": "eager",
        "deterministic_algorithms": "strict",
        "cublas_workspace_config": ":4096:8",
    }


def test_bfloat16_alias_resolves_on_supported_cuda() -> None:
    torch = _FakeTorch()
    name, value = _resolve_model_dtype(torch, "bf16", SimpleNamespace(type="cuda"))
    assert name == "bfloat16"
    assert value is torch.bfloat16


def test_bfloat16_rejects_unsupported_cuda() -> None:
    with pytest.raises(RuntimeError, match="lacks BF16 support"):
        _resolve_model_dtype(_FakeTorch(False), "bfloat16", SimpleNamespace(type="cuda"))


def test_unknown_generator_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported model dtype"):
        _resolve_model_dtype(_FakeTorch(), "float8", SimpleNamespace(type="cuda"))
