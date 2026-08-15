import math
from types import SimpleNamespace

import numpy as np
import pytest

from rgrd.generation import (
    generation_effect,
    mean_causal_answer_logprob,
    parse_final_answer,
    teacher_forced_mean_logp,
)


def test_teacher_forced_logprob_scores_answer_tokens_only() -> None:
    logits = np.zeros((4, 3), dtype=float)
    logits[1] = [0.0, 2.0, 0.0]  # predicts label at position 2
    logits[2] = [0.0, 0.0, 3.0]  # predicts label at position 3
    labels = np.array([0, 0, 1, 2])
    answer_mask = np.array([False, False, True, True])
    observed = mean_causal_answer_logprob(logits, labels, answer_mask)
    expected_1 = 2.0 - math.log(math.exp(0) + math.exp(2) + math.exp(0))
    expected_2 = 3.0 - math.log(math.exp(0) + math.exp(0) + math.exp(3))
    assert observed == pytest.approx((expected_1 + expected_2) / 2)
    assert parse_final_answer("analysis\nFINAL_ANSWER: short answer\n") == "short answer"


def test_teacher_forced_logprob_rejects_nonfinite_logits() -> None:
    logits = np.zeros((3, 2), dtype=float)
    logits[1, 0] = np.nan
    labels = np.array([0, 0, 1])
    answer_mask = np.array([False, False, True])
    with pytest.raises(FloatingPointError, match="logits must be finite"):
        mean_causal_answer_logprob(logits, labels, answer_mask)


@pytest.mark.parametrize("full,hidden", [(np.nan, -1.0), (-1.0, np.inf)])
def test_generation_effect_rejects_nonfinite_scores(full: float, hidden: float) -> None:
    with pytest.raises(FloatingPointError, match="generation scores must be finite"):
        generation_effect(full, hidden)


def test_torch_teacher_score_rejects_nonfinite_answer_logprob() -> None:
    torch = pytest.importorskip("torch")

    class _Model:
        def __call__(self, **_: object) -> SimpleNamespace:
            logits = torch.zeros((1, 3, 2), dtype=torch.float32)
            logits[0, 1, 0] = torch.nan
            return SimpleNamespace(logits=logits)

    input_ids = torch.tensor([[0, 0, 0]])
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(3).unsqueeze(0)
    answer_mask = torch.tensor([[False, False, True]])
    with pytest.raises(FloatingPointError, match="answer log-probabilities"):
        teacher_forced_mean_logp(
            _Model(),
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            answer_mask=answer_mask,
        )
