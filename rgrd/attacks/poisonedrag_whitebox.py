from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_targets(path: Path, count: int) -> list[dict[str, Any]]:
    values = list(json.loads(path.read_text(encoding="utf-8")).values())
    if len(values) < count:
        raise ValueError(f"requested {count} targets but only {len(values)} are available")
    return values[:count]


def run_whitebox_smoke(
    *,
    attack_root: Path,
    retriever_path: Path,
    generator_path: Path,
    output_path: Path,
    count: int = 20,
    gpu: int = 0,
    seed: int = 12,
) -> list[dict[str, Any]]:
    """Run the locally audited PoisonedRAG HotFlip path without writing attack_root."""

    import torch
    from transformers import AutoTokenizer

    attack_root = attack_root.resolve()
    if not attack_root.is_dir():
        raise FileNotFoundError(attack_root)
    output_path = output_path.resolve()
    if attack_root in output_path.parents:
        raise ValueError("smoke output must not be written under the read-only attack root")
    torch.cuda.set_device(gpu)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sys.path.insert(0, str(attack_root))
    from src.attack import Attacker
    from src.contriever_src.contriever import Contriever
    from src.utils import contriever_get_emb, load_beir_datasets

    tokenizer = AutoTokenizer.from_pretrained(retriever_path, local_files_only=True, use_fast=True)
    retriever = Contriever.from_pretrained(retriever_path, local_files_only=True).eval().cuda()
    target_path = attack_root / "results/adv_targeted_results/nq.json"
    targets = _load_targets(target_path, count)
    retrieval_scores = json.loads(
        (attack_root / "results/beir_results/nq-contriever.json").read_text(encoding="utf-8")
    )
    args = SimpleNamespace(
        attack_method="hotflip",
        adv_per_query=5,
        eval_dataset="nq",
        score_function="dot",
    )
    with working_directory(attack_root):
        attacker = Attacker(
            args,
            model=retriever,
            c_model=retriever,
            tokenizer=tokenizer,
            get_emb=contriever_get_emb,
        )
    cache_path = output_path.with_suffix(".attack-cache.json")
    generated: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        generated = json.loads(cache_path.read_text(encoding="utf-8"))
    for target in targets:
        query_id = str(target["id"])
        if query_id in generated:
            continue
        top1 = float(next(iter(retrieval_scores[query_id].values())))
        with working_directory(attack_root):
            group = attacker.get_attack(
                [{"query": str(target["question"]), "top1_score": top1, "id": query_id}]
            )[0]
        oracle: dict[str, dict[str, list[int]]] = {}
        payload_sources = target["adv_texts"][: len(group)]
        for attack_text, payload_source in zip(group, payload_sources, strict=True):
            payload_ids = tokenizer(
                payload_source,
                max_length=128,
                truncation=True,
                padding=False,
            )["input_ids"]
            payload_text = tokenizer.decode(
                payload_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            payload_start = attack_text.rfind(payload_text)
            if payload_start <= 0:
                raise ValueError("unable to recover HotFlip anchor/payload boundary")
            oracle[attack_text] = {
                "anchor": [0, payload_start],
                "payload": [payload_start, len(attack_text)],
            }
        generated[query_id] = {"attacks": group, "oracle_ranges": oracle}
        _save(cache_path, generated)

    with working_directory(attack_root):
        corpus, _, qrels = load_beir_datasets("nq", "test")
    rows: list[dict[str, Any]] = []
    for target in targets:
        query_id = str(target["id"])
        query = str(target["question"])
        group = generated[query_id]["attacks"]
        query_inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
        query_inputs = {key: value.cuda() for key, value in query_inputs.items()}
        attack_inputs = tokenizer(
            group, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        attack_inputs = {key: value.cuda() for key, value in attack_inputs.items()}
        with torch.inference_mode():
            query_embedding = contriever_get_emb(retriever, query_inputs)
            attack_embeddings = contriever_get_emb(retriever, attack_inputs)
        clean_top = list(retrieval_scores[query_id].items())[:5]
        candidates = [
            {"score": float(score), "text": corpus[doc_id]["text"], "attack": False}
            for doc_id, score in clean_top
        ]
        for attack_text, embedding in zip(group, attack_embeddings, strict=True):
            score = float(torch.mm(embedding.unsqueeze(0), query_embedding.T).cpu().item())
            candidates.append({"score": score, "text": attack_text, "attack": True})
        candidates.sort(key=lambda item: (-item["score"], item["text"]))
        selected = candidates[:5]
        injected = [item["text"] for item in selected if item["attack"]]
        ground_truth = [corpus[doc_id]["text"] for doc_id in qrels[query_id]]
        rows.append(
            {
                "id": query_id,
                "question": query,
                "injected_adv": injected,
                "contexts": [item["text"] for item in selected],
                "oracle_ranges": {
                    text: generated[query_id]["oracle_ranges"][text] for text in injected
                },
                "incorrect_answer": str(target["incorrect answer"]),
                "answer": str(target["correct answer"]),
                "ground_truth_context_count": len(ground_truth),
            }
        )
    del attacker, retriever
    torch.cuda.empty_cache()

    from rgrd.models import CausalAnswerGenerator

    generator_path = generator_path.resolve()
    generator = CausalAnswerGenerator(
        generator_path,
        device=f"cuda:{gpu}",
        max_new_tokens=32,
        seed=seed,
    )
    completed_rows: list[dict[str, Any]] = []
    existing: dict[str, dict[str, Any]] = {}
    if output_path.is_file():
        for iteration in json.loads(output_path.read_text(encoding="utf-8")):
            for values in iteration.values():
                for row in values:
                    if row.get("generation_protocol") == "rgrd-track-a-qwen-chat-v1" and row.get(
                        "generator_path"
                    ) == str(generator_path):
                        existing[str(row["id"])] = row
    for row in rows:
        if row["id"] in existing:
            completed_rows.append(existing[row["id"]])
            continue
        contexts = row.pop("contexts")
        layout = generator.build_prompt(
            row["question"],
            [(f"track-a-context-{index}", text) for index, text in enumerate(contexts)],
        )
        answer, continuation = generator.generate_shadow(layout)
        row["input_prompt"] = layout.prompt
        row["output_poison"] = answer
        row["output_continuation"] = continuation
        row["generation_protocol"] = "rgrd-track-a-qwen-chat-v1"
        row["generator_path"] = str(generator_path)
        completed_rows.append(row)
        _save(output_path, [{"iter_0": completed_rows}])
    return completed_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="PoisonedRAG-W 20-query functional smoke")
    parser.add_argument("--attack-root", type=Path, required=True)
    parser.add_argument("--retriever", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()
    rows = run_whitebox_smoke(
        attack_root=args.attack_root,
        retriever_path=args.retriever,
        generator_path=args.generator,
        output_path=args.output,
        count=args.count,
        gpu=args.gpu,
        seed=args.seed,
    )
    print(json.dumps({"completed": len(rows), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
