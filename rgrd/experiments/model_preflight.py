from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import yaml

from rgrd.models import CausalAnswerGenerator, CrossEncoderReranker, DenseRetriever
from rgrd.provenance import utc_now
from rgrd.schema import CharRange


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def preflight(root: Path, device: str) -> dict[str, object]:
    config = yaml.safe_load((root / "configs/pipeline/v0.yaml").read_text(encoding="utf-8"))
    models = config["models"]
    track = config["track_b"]
    retriever = DenseRetriever(Path(models["retriever"]["local_path"]), device=device)
    texts = ["William Shakespeare wrote Hamlet.", "Paris is the capital of France."]
    dense = retriever.encode(texts)
    hidden_dense = retriever.encode_hidden(texts[0], CharRange(start=0, end=7))
    if dense.shape != (2, retriever.dimension) or hidden_dense.shape != (retriever.dimension,):
        raise AssertionError("dense retriever preflight returned unexpected shapes")
    reranker = CrossEncoderReranker(Path(models["reranker"]["local_path"]), device=device)
    rerank = reranker.score_pairs("Who wrote Hamlet?", texts)
    hidden_rerank = reranker.score_hidden("Who wrote Hamlet?", texts[0], CharRange(start=0, end=7))
    if rerank.shape != (2,):
        raise AssertionError("reranker preflight returned unexpected shape")
    generator = CausalAnswerGenerator(
        Path(models["generator"]["local_path"]),
        device=device,
        max_new_tokens=int(track["generator_max_new_tokens"]),
        seed=int(config["seed"]),
    )
    layout = generator.build_prompt("Who wrote Hamlet?", [("c1", texts[0]), ("c2", texts[1])])
    answer, continuation = generator.generate_shadow(layout)
    full_logp = generator.teacher_score(layout, answer)
    hidden_logp = generator.teacher_score(
        layout, answer, chunk_id="c1", hidden_span=CharRange(start=0, end=7)
    )
    if not answer:
        raise AssertionError("generator produced an empty parsed FINAL_ANSWER")
    result = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "device": device,
        "models": {
            role: {
                "name": value["name"],
                "revision": value["revision"],
                "local_path": value["local_path"],
            }
            for role, value in models.items()
        },
        "checks": {
            "dense_shape": list(dense.shape),
            "dense_hidden_finite": bool(float(hidden_dense[0]) == float(hidden_dense[0])),
            "reranker_scores": [float(value) for value in rerank],
            "reranker_hidden_score": float(hidden_rerank),
            "shadow_answer": answer,
            "continuation": continuation,
            "teacher_full_mean_logp": full_logp,
            "teacher_hidden_mean_logp": hidden_logp,
        },
        "passed": True,
    }
    _atomic_json(root / "artifacts/statistics/model_preflight.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real local-model RGRD preflight")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    result = preflight(arguments.root.resolve(), arguments.device)
    print(
        json.dumps(
            {"passed": result["passed"], "shadow_answer": result["checks"]["shadow_answer"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
