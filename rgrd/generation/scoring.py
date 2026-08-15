from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


FINAL_ANSWER_PATTERN = re.compile(r"(?:^|\n)\s*FINAL_ANSWER\s*:\s*(.+?)\s*(?:\n|$)", re.I)


def parse_final_answer(text: str) -> str:
    match = FINAL_ANSWER_PATTERN.search(text)
    if not match:
        raise ValueError("generation did not contain a FINAL_ANSWER line")
    answer = match.group(1).strip()
    if not answer:
        raise ValueError("FINAL_ANSWER is empty")
    return answer


def _logsumexp(vector: np.ndarray) -> float:
    maximum = float(np.max(vector))
    return maximum + math.log(float(np.exp(vector - maximum).sum()))


def mean_causal_answer_logprob(
    logits: np.ndarray,
    labels: np.ndarray,
    answer_mask: np.ndarray,
) -> float:
    """Score only answer tokens using causal next-token alignment.

    `answer_mask[..., t]` marks whether label token at position `t` belongs to the
    parsed answer. Logits at `t-1` are used to score that label.
    """

    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    answer_mask = np.asarray(answer_mask, dtype=bool)
    if not np.all(np.isfinite(logits)):
        raise FloatingPointError("causal answer logits must be finite")
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("only batch size one is supported by the scalar scorer")
        logits = logits[0]
    if labels.ndim == 2:
        if labels.shape[0] != 1:
            raise ValueError("only batch size one is supported")
        labels = labels[0]
    if answer_mask.ndim == 2:
        if answer_mask.shape[0] != 1:
            raise ValueError("only batch size one is supported")
        answer_mask = answer_mask[0]
    if logits.shape[0] != labels.shape[0] or labels.shape != answer_mask.shape:
        raise ValueError("sequence dimensions must agree")
    positions = np.flatnonzero(answer_mask)
    positions = positions[positions > 0]
    if len(positions) == 0:
        raise ValueError("answer mask contains no causally scoreable tokens")
    scores: list[float] = []
    for position in positions:
        target = int(labels[position])
        if target < 0 or target >= logits.shape[-1]:
            raise ValueError("answer label is outside vocabulary")
        vector = logits[position - 1]
        scores.append(float(vector[target]) - _logsumexp(vector))
    return float(np.mean(scores))


def teacher_forced_mean_logp(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any,
    position_ids: Any,
    answer_mask: Any,
) -> float:
    """Run a deterministic teacher-forced model pass and score answer tokens only."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only in model environment
        raise RuntimeError("torch is required for model scoring") from exc
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        raw_logits = outputs.logits
        logits = raw_logits.float()
        labels = input_ids
        shifted_logits = logits[..., :-1, :]
        shifted_labels = labels[..., 1:]
        shifted_mask = answer_mask[..., 1:].bool()
        log_probs = torch.log_softmax(shifted_logits, dim=-1)
        selected = log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
        selected = selected[shifted_mask]
        if selected.numel() == 0:
            raise ValueError("answer mask contains no scoreable tokens")
        finite = torch.isfinite(selected)
        if not bool(finite.all()):
            nonfinite = int((~finite).sum().item())
            raise FloatingPointError(
                "teacher-forced answer log-probabilities are non-finite: "
                f"{nonfinite}/{selected.numel()} answer tokens; "
                f"logits_dtype={raw_logits.dtype}; sequence_length={input_ids.shape[-1]}"
            )
        return float(selected.mean().cpu())


def generation_effect(full_mean_logp: float, hidden_mean_logp: float) -> float:
    full = float(full_mean_logp)
    hidden = float(hidden_mean_logp)
    if not math.isfinite(full) or not math.isfinite(hidden):
        raise FloatingPointError(
            f"generation scores must be finite, observed full={full}, hidden={hidden}"
        )
    effect = full - hidden
    if not math.isfinite(effect):
        raise FloatingPointError(f"generation effect must be finite, observed {effect}")
    return effect
