from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from rgrd.provenance import sha256_file, utc_now


def _atomic_json(path: Path, value: Any) -> None:
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


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _task_id(value: dict[str, Any]) -> str:
    payload = "\0".join(
        (
            str(value.get("query", "")),
            str(value.get("incorrect_answer", "")),
            str(value.get("adv_text", "")),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def prepare_tasks(
    *,
    official_root: Path,
    output: Path,
    count: int,
    shards: int,
) -> dict[str, Any]:
    source_path = official_root / "attack_data/poisoned_rag_nq/data.json"
    values = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) < count:
        raise RuntimeError(f"Joint-GCG NQ assets contain fewer than {count} tasks")
    selected = values[:count]
    records: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    for shard_id in range(shards):
        shard_values = [value for index, value in enumerate(selected) if index % shards == shard_id]
        shard_dir = output / f"shard-{shard_id:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(shard_dir / "data.json", shard_values)
        shard_tasks = []
        for cluster_index, value in enumerate(shard_values):
            record = {
                "task_id": _task_id(value),
                "shard_id": shard_id,
                "cluster_index": cluster_index,
                "query": str(value["query"]),
                "incorrect_answer": str(value["incorrect_answer"]),
                "adv_text": str(value["adv_text"]),
            }
            records.append(record)
            shard_tasks.append(record)
        _atomic_json(shard_dir / "task_manifest.json", {"tasks": shard_tasks})
    manifest = {
        "schema_version": 1,
        "prepared_at": utc_now(),
        "official_asset": str(source_path.resolve()),
        "official_asset_sha256": sha256_file(source_path),
        "count": len(records),
        "shards": shards,
        "tasks": records,
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def run_official_worker(
    *,
    official_root: Path,
    assets: Path,
    output: Path,
    retriever: Path,
    generator: Path,
    transfer_matrix: Path,
    corpus: Path,
    retrieval_results: Path,
    epochs: int,
    n_samples: int,
    topk: int,
    tag_length: int,
) -> dict[str, Any]:
    """Run the pinned optimizer with two non-algorithmic compatibility guards.

    The upstream loader leaves all model parameters trainable even though GCG only
    differentiates the one-hot control tensor. Freezing those parameters preserves
    the input gradient while avoiding a second 7B gradient copy. The upstream step
    also returns before writing JSON when an attempt succeeds; the wrapper persists
    that already-computed terminal state so every preregistered attempt is auditable.
    Neither guard changes losses, gradients, candidate sampling, or stopping rules.
    """

    complete = output / "worker_complete.json"
    if complete.is_file():
        return json.loads(complete.read_text(encoding="utf-8"))
    manifest = json.loads((assets / "task_manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(official_root.resolve()))
    with _working_directory(official_root):
        import attack_rag
        from rag.poisionedrag import PoisionedRAGJointAttackCore

        upstream_load_model = attack_rag.load_model
        upstream_step = PoisionedRAGJointAttackCore.step

        def load_frozen_model(*arguments: Any, **keywords: Any):
            model, tokenizer = upstream_load_model(*arguments, **keywords)
            model.requires_grad_(False)
            return model, tokenizer

        def step_with_terminal_record(self: Any, epoch: int, eval_only: bool = False):
            success = upstream_step(self, epoch, eval_only=eval_only)
            log_folder = Path(self.log_path) / self.tag
            log_path = log_folder / f"{epoch}.json"
            if self.verbose > 0 and not log_path.is_file():
                value = dict(getattr(self, "log_data_json", {}))
                value.update(
                    {
                        "fake_corpus": self.fake_corpus,
                        "adv_tag": self.adv_tag,
                        "terminal_success": bool(success),
                        "rgrd_compatibility_record": True,
                    }
                )
                _atomic_json(log_path, value)
            return success

        attack_rag.load_model = load_frozen_model
        PoisionedRAGJointAttackCore.step = step_with_terminal_record
        try:
            attack_rag.attack_joint(
                tag="rag_v2",
                model_llm=str(generator.resolve()),
                model_retriever=str(retriever.resolve()),
                assets_path=str(assets.resolve()),
                transfer_matrix_path=str(transfer_matrix.resolve()),
                corpus_jsonl_path=str(corpus.resolve()),
                queries_top50_hits_json_path=str(retrieval_results.resolve()),
                use_adaptive_ratio=True,
                log_path=str(output.resolve()),
                start_cluster_idx=0,
                max_new_tokens=32,
                infer_batch_size=16,
                use_query_as_suffix=True,
                n_samples=n_samples,
                topk=topk,
                epochs=epochs,
                tag_length=tag_length,
                ascii_only=True,
                fixed_ratio=0.2,
                joint_loss_only=False,
                joint_grad_only=False,
                eval_gap=0,
            )
        finally:
            attack_rag.load_model = upstream_load_model
            PoisionedRAGJointAttackCore.step = upstream_step
    result = {
        "schema_version": 1,
        "completed_at": utc_now(),
        "official_commit": "8707acb23954d9683483b9dc2406684c301a39d2",
        "tasks": len(manifest["tasks"]),
        "epochs": epochs,
        "n_samples": n_samples,
        "topk": topk,
        "tag_length": tag_length,
        "transfer_matrix_sha256": sha256_file(transfer_matrix),
        "compatibility_guards": [
            "freeze model parameters while preserving control-token input gradients",
            "persist an already-computed terminal state when upstream returns before JSON logging",
        ],
    }
    _atomic_json(complete, result)
    return result


def collect_outputs(*, tasks_root: Path, logs_root: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((tasks_root / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for task in manifest["tasks"]:
        shard_id = int(task["shard_id"])
        cluster_index = int(task["cluster_index"])
        cluster = logs_root / f"shard-{shard_id:02d}/log/rag_v2" / f"cluster_{cluster_index}"
        epoch_files = []
        for path in cluster.glob("*.json"):
            try:
                epoch_files.append((int(path.stem), path))
            except ValueError:
                continue
        if not epoch_files:
            missing.append(str(task["task_id"]))
            continue
        epoch, path = max(epoch_files)
        value = json.loads(path.read_text(encoding="utf-8"))
        optimized_text = str(value.get("fake_corpus", ""))
        if not optimized_text:
            missing.append(str(task["task_id"]))
            continue
        rows.append(
            {
                "schema_version": 1,
                **task,
                "family": "Joint-GCG",
                "dataset": "nq",
                "optimized_text": optimized_text,
                "final_epoch": epoch,
                "official_log": str(path.resolve()),
                "official_log_sha256": sha256_file(path),
            }
        )
    if missing:
        raise RuntimeError(
            f"Joint-GCG output collection is missing {len(missing)} tasks: {missing[:10]}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return {"completed": len(rows), "output": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare, run, or collect pinned official Joint-GCG"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--official-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--count", type=int, required=True)
    prepare.add_argument("--shards", type=int, required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--official-root", type=Path, required=True)
    worker.add_argument("--assets", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--retriever", type=Path, required=True)
    worker.add_argument("--generator", type=Path, required=True)
    worker.add_argument("--transfer-matrix", type=Path, required=True)
    worker.add_argument("--corpus", type=Path, required=True)
    worker.add_argument("--retrieval-results", type=Path, required=True)
    worker.add_argument("--epochs", type=int, default=64)
    worker.add_argument("--n-samples", type=int, default=128)
    worker.add_argument("--topk", type=int, default=16)
    worker.add_argument("--tag-length", type=int, default=32)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--tasks-root", type=Path, required=True)
    collect.add_argument("--logs-root", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        result = prepare_tasks(
            official_root=arguments.official_root,
            output=arguments.output,
            count=arguments.count,
            shards=arguments.shards,
        )
    elif arguments.command == "worker":
        result = run_official_worker(
            official_root=arguments.official_root,
            assets=arguments.assets,
            output=arguments.output,
            retriever=arguments.retriever,
            generator=arguments.generator,
            transfer_matrix=arguments.transfer_matrix,
            corpus=arguments.corpus,
            retrieval_results=arguments.retrieval_results,
            epochs=arguments.epochs,
            n_samples=arguments.n_samples,
            topk=arguments.topk,
            tag_length=arguments.tag_length,
        )
    else:
        result = collect_outputs(
            tasks_root=arguments.tasks_root,
            logs_root=arguments.logs_root,
            output=arguments.output,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
