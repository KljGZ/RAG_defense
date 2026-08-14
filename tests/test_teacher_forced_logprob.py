import math

import numpy as np
import pytest

from rgrd.generation import mean_causal_answer_logprob, parse_final_answer


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
