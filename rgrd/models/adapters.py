from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from rgrd.generation import parse_final_answer, teacher_forced_mean_logp
from rgrd.schema import CharRange


_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - model environment only
        raise RuntimeError("model adapters require torch") from exc
    return torch


_MODEL_DTYPE_ALIASES = {
    "auto": "auto",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float32": "float32",
}


def _resolve_model_dtype(torch: Any, requested: str, device: Any) -> tuple[str, Any]:
    normalized = _MODEL_DTYPE_ALIASES.get(str(requested).strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(_MODEL_DTYPE_ALIASES))
        raise ValueError(f"unsupported model dtype {requested!r}; expected one of: {supported}")
    if normalized == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 was requested but the selected CUDA device lacks BF16 support")
    resolved = "auto" if normalized == "auto" else getattr(torch, normalized)
    return normalized, resolved


def _loaded_model_dtype(model: Any) -> str:
    dtypes = {
        str(parameter.dtype).removeprefix("torch.")
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if len(dtypes) != 1:
        raise RuntimeError(f"generator must use one floating dtype, observed {sorted(dtypes)}")
    return next(iter(dtypes))


def set_deterministic(seed: int) -> None:
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    denominator = mask.sum(dim=1).clamp(min=1.0)
    return (last_hidden_state * mask).sum(dim=1) / denominator


def _neutral_token_id(tokenizer: Any) -> int:
    for value in (tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.eos_token_id):
        if value is not None:
            return int(value)
    raise ValueError("tokenizer has no neutral pad/unk/eos token")


def _overlap_indices(
    offsets: Sequence[Sequence[int]],
    span: CharRange,
    *,
    eligible: Iterable[int] | None = None,
) -> list[int]:
    allowed = None if eligible is None else set(eligible)
    indices: list[int] = []
    for index, pair in enumerate(offsets):
        start, end = int(pair[0]), int(pair[1])
        if end <= start or (allowed is not None and index not in allowed):
            continue
        if start < span.end and end > span.start:
            indices.append(index)
    if not indices:
        raise ValueError(f"span {span} did not map to any model tokens")
    return indices


def _overlap_indices_many(
    offsets: Sequence[Sequence[int]],
    spans: Sequence[CharRange],
    *,
    eligible: Iterable[int] | None = None,
) -> list[int]:
    if not spans:
        raise ValueError("at least one intervention span is required")
    values: set[int] = set()
    for span in spans:
        values.update(_overlap_indices(offsets, span, eligible=eligible))
    return sorted(values)


def _partition_overlap_indices(
    offsets: Sequence[Sequence[int]],
    spans: Sequence[CharRange],
    *,
    eligible: Iterable[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Assign each overlapping token to exactly one ordered character span.

    Fast tokenizers can emit a boundary token that overlaps two adjacent Oracle
    spans.  Treating that token as part of both players violates the two-player
    coalition design.  We therefore assign a token to the span with the largest
    character overlap; ties are resolved by the input order (A spans before P
    spans in V0.1).  Tokens outside every span remain untouched.
    """

    if not spans:
        raise ValueError("at least one intervention span is required")
    allowed = None if eligible is None else set(eligible)
    partitions: list[list[int]] = [[] for _ in spans]
    for index, pair in enumerate(offsets):
        start, end = int(pair[0]), int(pair[1])
        if end <= start or (allowed is not None and index not in allowed):
            continue
        overlaps = [max(0, min(end, span.end) - max(start, span.start)) for span in spans]
        maximum = max(overlaps)
        if maximum > 0:
            partitions[overlaps.index(maximum)].append(index)
    missing = [index for index, values in enumerate(partitions) if not values]
    if missing:
        raise ValueError(
            "Oracle spans did not map to distinct model-token partitions: "
            f"missing ordered span indices {missing}"
        )
    return tuple(tuple(values) for values in partitions)


def _replacement_ids(tokenizer: Any, donor_text: str, length: int) -> list[int]:
    encoded = tokenizer(donor_text, add_special_tokens=False, truncation=False)
    values = encoded["input_ids"]
    if values and isinstance(values[0], list):
        values = values[0]
    if len(values) < length:
        raise ValueError(f"clean donor has {len(values)} tokens but intervention requires {length}")
    return [int(value) for value in values[:length]]


def _exact_replacement_ids(tokenizer: Any, donor_text: str, length: int) -> list[int]:
    values = _replacement_ids(tokenizer, donor_text, length)
    encoded = tokenizer(donor_text, add_special_tokens=False, truncation=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if len(encoded) != length:
        raise ValueError(
            f"V0.1 donor has {len(encoded)} tokens but intervention requires exactly {length}"
        )
    return values


class DenseRetriever:
    """Contriever-compatible dense encoder with intervention-aware mean pooling."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cuda:0",
        max_length: int = 512,
        normalize: bool = False,
    ) -> None:
        torch = _torch()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers is required") from exc
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.max_length = max_length
        self.normalize = normalize
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, use_fast=True
        )
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
        self.model.eval().to(self.device)
        self.dimension = int(self.model.config.hidden_size)

    def _forward(self, encoded: Any) -> Any:
        torch = _torch()
        encoded = {
            key: value.to(self.device) for key, value in encoded.items() if key != "offset_mapping"
        }
        position_ids = torch.arange(encoded["input_ids"].shape[-1], device=self.device).unsqueeze(0)
        if encoded["input_ids"].shape[0] > 1:
            position_ids = position_ids.expand(encoded["input_ids"].shape[0], -1)
        with torch.inference_mode():
            output = self.model(**encoded, position_ids=position_ids, return_dict=True)
            embeddings = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings.float()

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        values: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            values.append(self._forward(encoded).cpu().numpy())
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(values), dtype=np.float32)

    def encode_hidden(self, text: str, span: CharRange) -> np.ndarray:
        return self.encode_hidden_ranges(text, [span])

    def encode_hidden_ranges(self, text: str, spans: Sequence[CharRange]) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        indices = _overlap_indices_many(offsets, spans)
        encoded["input_ids"] = encoded["input_ids"].clone()
        encoded["attention_mask"] = encoded["attention_mask"].clone()
        encoded["input_ids"][0, indices] = _neutral_token_id(self.tokenizer)
        encoded["attention_mask"][0, indices] = 0
        if int(encoded["attention_mask"].sum()) == 0:
            raise ValueError("intervention hid every retriever token")
        return self._forward(encoded)[0].cpu().numpy().astype(np.float32, copy=False)

    def intervention_token_lengths(self, text: str, spans: Sequence[CharRange]) -> tuple[int, ...]:
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        if (
            offsets
            and isinstance(offsets[0], list)
            and offsets[0]
            and isinstance(offsets[0][0], list)
        ):
            offsets = offsets[0]
        return tuple(len(values) for values in _partition_overlap_indices(offsets, spans))

    def encode_hidden_partition(
        self,
        text: str,
        spans: Sequence[CharRange],
        hidden: Sequence[bool],
    ) -> np.ndarray:
        if len(spans) != len(hidden) or not spans:
            raise ValueError("partition spans and hidden flags must have equal non-zero length")
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        partitions = _partition_overlap_indices(offsets, spans)
        indices = sorted(
            index
            for values, should_hide in zip(partitions, hidden, strict=True)
            if should_hide
            for index in values
        )
        if not indices:
            return self._forward(encoded)[0].cpu().numpy().astype(np.float32, copy=False)
        encoded["input_ids"] = encoded["input_ids"].clone()
        encoded["attention_mask"] = encoded["attention_mask"].clone()
        encoded["input_ids"][0, indices] = _neutral_token_id(self.tokenizer)
        encoded["attention_mask"][0, indices] = 0
        if int(encoded["attention_mask"].sum()) == 0:
            raise ValueError("intervention hid every retriever token")
        return self._forward(encoded)[0].cpu().numpy().astype(np.float32, copy=False)

    def encode_replaced(self, text: str, span: CharRange, donor_text: str) -> np.ndarray:
        return self.encode_replaced_ranges(text, [span], [donor_text])

    def encode_replaced_ranges(
        self,
        text: str,
        spans: Sequence[CharRange],
        donor_texts: Sequence[str | None],
    ) -> np.ndarray:
        torch = _torch()
        if len(spans) != len(donor_texts) or not spans:
            raise ValueError("replacement spans and donor texts must have equal non-zero length")
        encoded = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded["input_ids"] = encoded["input_ids"].clone()
        partitions = _partition_overlap_indices(offsets, spans)
        for indices, donor_text in zip(partitions, donor_texts, strict=True):
            if donor_text is None:
                continue
            replacements = _replacement_ids(self.tokenizer, donor_text, len(indices))
            encoded["input_ids"][0, indices] = torch.tensor(replacements, dtype=torch.long)
        return self._forward(encoded)[0].cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def score(query_embedding: np.ndarray, chunk_embedding: np.ndarray) -> float:
        return float(np.dot(query_embedding.astype(np.float32), chunk_embedding.astype(np.float32)))


class CrossEncoderReranker:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cuda:0",
        max_length: int = 512,
    ) -> None:
        torch = _torch()
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers is required") from exc
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, use_fast=True
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.model_path, local_files_only=True
            )
            .eval()
            .to(self.device)
        )

    def _score_encoded(self, encoded: Any) -> np.ndarray:
        torch = _torch()
        encoded = {
            key: value.to(self.device) for key, value in encoded.items() if key != "offset_mapping"
        }
        position_ids = torch.arange(encoded["input_ids"].shape[-1], device=self.device).unsqueeze(0)
        if encoded["input_ids"].shape[0] > 1:
            position_ids = position_ids.expand(encoded["input_ids"].shape[0], -1)
        with torch.inference_mode():
            logits = self.model(
                **encoded, position_ids=position_ids, return_dict=True
            ).logits.float()
            if logits.shape[-1] == 1:
                scores = logits[:, 0]
            elif logits.shape[-1] == 2:
                scores = logits[:, 1]
            else:
                raise ValueError(f"unsupported reranker output width {logits.shape[-1]}")
        return scores.cpu().numpy()

    def score_pairs(self, query: str, chunks: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        scores: list[np.ndarray] = []
        for start in range(0, len(chunks), batch_size):
            batch = list(chunks[start : start + batch_size])
            encoded = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            scores.append(self._score_encoded(encoded))
        return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)

    def score_hidden(self, query: str, chunk: str, span: CharRange) -> float:
        return self.score_hidden_ranges(query, chunk, [span])

    def score_hidden_ranges(self, query: str, chunk: str, spans: Sequence[CharRange]) -> float:
        encoded = self.tokenizer(
            query,
            chunk,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        sequence_ids = encoded.sequence_ids(0)
        chunk_indices = [
            index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1
        ]
        indices = _overlap_indices_many(offsets, spans, eligible=chunk_indices)
        encoded["input_ids"] = encoded["input_ids"].clone()
        encoded["attention_mask"] = encoded["attention_mask"].clone()
        encoded["input_ids"][0, indices] = _neutral_token_id(self.tokenizer)
        encoded["attention_mask"][0, indices] = 0
        return float(self._score_encoded(encoded)[0])

    def intervention_token_lengths(
        self,
        query: str,
        chunk: str,
        spans: Sequence[CharRange],
    ) -> tuple[int, ...]:
        encoded = self.tokenizer(
            query,
            chunk,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        if (
            offsets
            and isinstance(offsets[0], list)
            and offsets[0]
            and isinstance(offsets[0][0], list)
        ):
            offsets = offsets[0]
        sequence_ids = encoded.sequence_ids(0)
        chunk_indices = [
            index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1
        ]
        return tuple(
            len(values)
            for values in _partition_overlap_indices(offsets, spans, eligible=chunk_indices)
        )

    def score_hidden_partition(
        self,
        query: str,
        chunk: str,
        spans: Sequence[CharRange],
        hidden: Sequence[bool],
    ) -> float:
        if len(spans) != len(hidden) or not spans:
            raise ValueError("partition spans and hidden flags must have equal non-zero length")
        encoded = self.tokenizer(
            query,
            chunk,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        sequence_ids = encoded.sequence_ids(0)
        chunk_indices = [
            index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1
        ]
        partitions = _partition_overlap_indices(offsets, spans, eligible=chunk_indices)
        indices = sorted(
            index
            for values, should_hide in zip(partitions, hidden, strict=True)
            if should_hide
            for index in values
        )
        if not indices:
            return float(self._score_encoded(encoded)[0])
        encoded["input_ids"] = encoded["input_ids"].clone()
        encoded["attention_mask"] = encoded["attention_mask"].clone()
        encoded["input_ids"][0, indices] = _neutral_token_id(self.tokenizer)
        encoded["attention_mask"][0, indices] = 0
        return float(self._score_encoded(encoded)[0])

    def score_replaced(self, query: str, chunk: str, span: CharRange, donor_text: str) -> float:
        return self.score_replaced_ranges(query, chunk, [span], [donor_text])

    def score_replaced_ranges(
        self,
        query: str,
        chunk: str,
        spans: Sequence[CharRange],
        donor_texts: Sequence[str | None],
    ) -> float:
        torch = _torch()
        if len(spans) != len(donor_texts) or not spans:
            raise ValueError("replacement spans and donor texts must have equal non-zero length")
        encoded = self.tokenizer(
            query,
            chunk,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        sequence_ids = encoded.sequence_ids(0)
        chunk_indices = [
            index for index, sequence_id in enumerate(sequence_ids) if sequence_id == 1
        ]
        encoded["input_ids"] = encoded["input_ids"].clone()
        partitions = _partition_overlap_indices(offsets, spans, eligible=chunk_indices)
        for indices, donor_text in zip(partitions, donor_texts, strict=True):
            if donor_text is None:
                continue
            replacements = _replacement_ids(self.tokenizer, donor_text, len(indices))
            encoded["input_ids"][0, indices] = torch.tensor(replacements, dtype=torch.long)
        return float(self._score_encoded(encoded)[0])


@dataclass(frozen=True)
class PromptLayout:
    prompt: str
    chunk_ranges: dict[str, CharRange]


@dataclass(frozen=True)
class GenerationAudit:
    answer: str
    continuation: str
    generated_tokens: int
    terminated_by_eos: bool
    truncated: bool
    strict_single_line: bool


class CausalAnswerGenerator:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cuda:0",
        max_new_tokens: int = 32,
        seed: int = 0,
        dtype: str = "auto",
        attention_implementation: str = "eager",
    ) -> None:
        torch = _torch()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers is required") from exc
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.requested_dtype, torch_dtype = _resolve_model_dtype(torch, dtype, self.device)
        self.attention_implementation = attention_implementation
        set_deterministic(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True,
                torch_dtype=torch_dtype,
                attn_implementation=self.attention_implementation,
                low_cpu_mem_usage=True,
            )
            .eval()
            .to(self.device)
        )
        self.model_dtype = _loaded_model_dtype(self.model)
        if self.requested_dtype != "auto" and self.model_dtype != self.requested_dtype:
            raise RuntimeError(
                f"generator dtype mismatch: requested {self.requested_dtype}, "
                f"loaded {self.model_dtype}"
            )

    def precision_metadata(self) -> dict[str, str]:
        return {
            "requested_dtype": self.requested_dtype,
            "loaded_dtype": self.model_dtype,
            "attention_implementation": self.attention_implementation,
            "deterministic_algorithms": "strict",
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        }

    def build_prompt(self, query: str, ordered_chunks: Sequence[tuple[str, str]]) -> PromptLayout:
        parts = [
            "Answer the query using only the supplied contexts. Be short and factual.\n",
            "Return exactly one non-empty line in this form: FINAL_ANSWER: <answer>\n\n",
        ]
        for position, (chunk_id, text) in enumerate(ordered_chunks):
            header = f"CONTEXT[{position}]:\n"
            parts.append(header)
            parts.append(text)
            parts.append("\n\n")
        parts.append(f"QUERY:\n{query}")
        content = "".join(parts)
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = content + "\n\nASSISTANT:\n"
        prompt += "FINAL_ANSWER: "
        ranges: dict[str, CharRange] = {}
        cursor = 0
        for chunk_id, text in ordered_chunks:
            start = prompt.find(text, cursor)
            if start < 0:
                raise ValueError(f"chat template did not preserve context chunk {chunk_id}")
            end = start + len(text)
            ranges[chunk_id] = CharRange(start=start, end=end)
            cursor = end
        return PromptLayout(prompt=prompt, chunk_ranges=ranges)

    def generate_shadow_audited(self, layout: PromptLayout) -> GenerationAudit:
        torch = _torch()
        set_deterministic(self.seed)
        encoded = self.tokenizer(layout.prompt, return_tensors="pt", add_special_tokens=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        continuation_ids = output[0, encoded["input_ids"].shape[-1] :]
        continuation = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        answer = parse_final_answer("FINAL_ANSWER: " + continuation)
        eos_values = self.tokenizer.eos_token_id
        eos_ids = (
            {int(value) for value in eos_values}
            if isinstance(eos_values, (list, tuple, set))
            else ({int(eos_values)} if eos_values is not None else set())
        )
        generated_ids = [int(value) for value in continuation_ids.tolist()]
        terminated_by_eos = bool(generated_ids and generated_ids[-1] in eos_ids)
        truncated = len(generated_ids) >= self.max_new_tokens and not terminated_by_eos
        nonempty_lines = [line for line in continuation.splitlines() if line.strip()]
        strict_single_line = len(nonempty_lines) == 1
        return GenerationAudit(
            answer=answer,
            continuation=continuation,
            generated_tokens=len(generated_ids),
            terminated_by_eos=terminated_by_eos,
            truncated=truncated,
            strict_single_line=strict_single_line,
        )

    def generate_shadow(self, layout: PromptLayout) -> tuple[str, str]:
        audit = self.generate_shadow_audited(layout)
        return audit.answer, audit.continuation

    def intervention_token_lengths(
        self,
        layout: PromptLayout,
        *,
        chunk_id: str,
        spans: Sequence[CharRange],
    ) -> tuple[int, ...]:
        if chunk_id not in layout.chunk_ranges or not spans:
            raise ValueError("intervention token lengths require a mapped chunk and spans")
        encoded = self.tokenizer(
            layout.prompt,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offsets = encoded["offset_mapping"]
        if (
            offsets
            and isinstance(offsets[0], list)
            and offsets[0]
            and isinstance(offsets[0][0], list)
        ):
            offsets = offsets[0]
        chunk_range = layout.chunk_ranges[chunk_id]
        prompt_spans: list[CharRange] = []
        for span in spans:
            prompt_span = CharRange(
                start=chunk_range.start + span.start,
                end=chunk_range.start + span.end,
            )
            if prompt_span.end > chunk_range.end:
                raise ValueError("intervention span exceeds the target chunk")
            prompt_spans.append(prompt_span)
        return tuple(len(values) for values in _partition_overlap_indices(offsets, prompt_spans))

    def teacher_score(
        self,
        layout: PromptLayout,
        answer: str,
        *,
        chunk_id: str | None = None,
        hidden_span: CharRange | None = None,
        hidden_spans: Sequence[CharRange] | None = None,
        donor_text: str | None = None,
        replacement_spans: Sequence[CharRange] | None = None,
        replacement_texts: Sequence[str | None] | None = None,
        partition_spans: Sequence[CharRange] | None = None,
        partition_hidden: Sequence[bool] | None = None,
    ) -> float:
        torch = _torch()
        suffix = answer
        full_text = layout.prompt + suffix
        encoded = self.tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        answer_chars = CharRange(start=len(layout.prompt), end=len(full_text))
        answer_indices = _overlap_indices(offsets, answer_chars)
        answer_mask = torch.zeros_like(encoded["input_ids"], dtype=torch.bool)
        answer_mask[0, answer_indices] = True
        if hidden_span is not None and hidden_spans is not None:
            raise ValueError("use hidden_span or hidden_spans, not both")
        has_replacements = replacement_spans is not None or replacement_texts is not None
        has_partition_mask = partition_spans is not None or partition_hidden is not None
        if has_replacements and (
            hidden_span is not None
            or hidden_spans is not None
            or donor_text is not None
            or has_partition_mask
        ):
            raise ValueError("V0.1 replacements cannot be combined with legacy interventions")
        if has_partition_mask and (
            hidden_span is not None or hidden_spans is not None or donor_text is not None
        ):
            raise ValueError("partition masking cannot be combined with legacy interventions")
        if has_replacements:
            if replacement_spans is None or replacement_texts is None:
                raise ValueError("replacement_spans and replacement_texts are both required")
            if len(replacement_spans) != len(replacement_texts) or not replacement_spans:
                raise ValueError("replacement spans/texts must have equal non-zero length")
            if chunk_id is None or chunk_id not in layout.chunk_ranges:
                raise ValueError("replacement intervention requires a mapped chunk_id")
            chunk_range = layout.chunk_ranges[chunk_id]
            encoded["input_ids"] = encoded["input_ids"].clone()
            prompt_spans: list[CharRange] = []
            for span in replacement_spans:
                prompt_span = CharRange(
                    start=chunk_range.start + span.start,
                    end=chunk_range.start + span.end,
                )
                if prompt_span.end > chunk_range.end:
                    raise ValueError("replacement span exceeds the target chunk")
                prompt_spans.append(prompt_span)
            partitions = _partition_overlap_indices(offsets, prompt_spans)
            for indices, replacement_text in zip(partitions, replacement_texts, strict=True):
                if replacement_text is None:
                    continue
                if set(indices) & set(answer_indices):
                    raise AssertionError("chunk replacement overlaps answer tokens")
                replacements = _exact_replacement_ids(
                    self.tokenizer, replacement_text, len(indices)
                )
                encoded["input_ids"][0, indices] = torch.tensor(replacements, dtype=torch.long)
        if has_partition_mask:
            if partition_spans is None or partition_hidden is None:
                raise ValueError("partition_spans and partition_hidden are both required")
            if len(partition_spans) != len(partition_hidden) or not partition_spans:
                raise ValueError("partition spans/flags must have equal non-zero length")
            if chunk_id is None or chunk_id not in layout.chunk_ranges:
                raise ValueError("partition mask requires a mapped chunk_id")
            chunk_range = layout.chunk_ranges[chunk_id]
            prompt_spans = [
                CharRange(
                    start=chunk_range.start + span.start,
                    end=chunk_range.start + span.end,
                )
                for span in partition_spans
            ]
            if any(span.end > chunk_range.end for span in prompt_spans):
                raise ValueError("partition span exceeds the target chunk")
            partitions = _partition_overlap_indices(offsets, prompt_spans)
            hidden_indices = sorted(
                index
                for values, should_hide in zip(partitions, partition_hidden, strict=True)
                if should_hide
                for index in values
            )
            if set(hidden_indices) & set(answer_indices):
                raise AssertionError("chunk intervention overlaps answer tokens")
            if hidden_indices:
                encoded["input_ids"] = encoded["input_ids"].clone()
                encoded["attention_mask"] = encoded["attention_mask"].clone()
                encoded["input_ids"][0, hidden_indices] = _neutral_token_id(self.tokenizer)
                encoded["attention_mask"][0, hidden_indices] = 0
        intervention_spans = (
            list(hidden_spans)
            if hidden_spans is not None
            else ([] if hidden_span is None else [hidden_span])
        )
        if intervention_spans and not has_replacements and not has_partition_mask:
            if chunk_id is None or chunk_id not in layout.chunk_ranges:
                raise ValueError("hidden intervention requires a mapped chunk_id")
            chunk_range = layout.chunk_ranges[chunk_id]
            prompt_spans = [
                CharRange(
                    start=chunk_range.start + span.start,
                    end=chunk_range.start + span.end,
                )
                for span in intervention_spans
            ]
            if any(span.end > chunk_range.end for span in prompt_spans):
                raise ValueError("hidden span exceeds the target chunk")
            hidden_indices = _overlap_indices_many(offsets, prompt_spans)
            if set(hidden_indices) & set(answer_indices):
                raise AssertionError("chunk intervention overlaps answer tokens")
            encoded["input_ids"] = encoded["input_ids"].clone()
            encoded["attention_mask"] = encoded["attention_mask"].clone()
            if donor_text is None:
                encoded["input_ids"][0, hidden_indices] = _neutral_token_id(self.tokenizer)
                encoded["attention_mask"][0, hidden_indices] = 0
            elif len(intervention_spans) == 1:
                replacements = _replacement_ids(self.tokenizer, donor_text, len(hidden_indices))
                encoded["input_ids"][0, hidden_indices] = torch.tensor(
                    replacements, dtype=torch.long
                )
            else:
                raise ValueError("donor_text replacement supports one span; use coalition text")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        answer_mask = answer_mask.to(self.device)
        position_ids = torch.arange(input_ids.shape[-1], device=self.device).unsqueeze(0)
        return teacher_forced_mean_logp(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            answer_mask=answer_mask,
        )
