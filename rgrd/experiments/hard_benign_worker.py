from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from rgrd.experiments.detection_worker import _pipeline
from rgrd.experiments.resume import event_provenance, prepare_jsonl_resume
from rgrd.schema import SourceDocument


def run(args: argparse.Namespace) -> dict[str, int]:
    root = args.root.resolve()
    tasks = [
        json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line
    ]
    tasks = [task for task in tasks if task["dataset"] == args.dataset]
    tasks = [task for index, task in enumerate(tasks) if index % args.shards == args.shard_id]
    config = yaml.safe_load((root / "configs/pipeline/v0.yaml").read_text(encoding="utf-8"))
    dataset_config = yaml.safe_load((root / "configs/datasets.yaml").read_text(encoding="utf-8"))[
        "datasets"
    ][args.dataset]
    _, bundle, pipeline = _pipeline(root, args.dataset, args.device, args.model_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = event_provenance(
        root,
        detector_code_commit=args.detector_commit,
        model_revisions=args.model_revision,
    )
    completed = {
        key[0]
        for key in prepare_jsonl_resume(
            args.output,
            key_fields=("task_id",),
            expected_provenance=provenance,
        )
    }
    written = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for task in tasks:
            if task["task_id"] in completed:
                continue
            source = SourceDocument(
                source_doc_id=f"hard-benign:{task['task_id']}",
                text=task["text"],
                uri=f"constructed://{task['group']}/{task['task_id']}",
                metadata={"hard_benign_group": task["group"]},
            )
            query_embedding, selected, frozen = pipeline.retrieve(
                task["query"], injected_sources=[(source, [], [])]
            )
            event, layout = pipeline.build_detector_event(
                query_id=task["query_id"],
                query=task["query"],
                dataset=args.dataset,
                split=str(dataset_config["split"]),
                selected=selected,
                frozen=frozen,
                detector_code_commit=args.detector_commit,
                seed=int(config["seed"]),
            )
            roles = pipeline.score_detector_event(
                event,
                candidates=pipeline.runtime_candidates(selected),
                query_embedding=query_embedding,
                frozen=frozen,
                layout=layout,
            )
            value = {
                **provenance,
                "task_id": task["task_id"],
                "query_id": task["query_id"],
                "group": task["group"],
                "dataset": args.dataset,
                "score": roles.query_score,
                "role_identifiable": any(chunk.role_identifiable for chunk in roles.chunks),
                "injected_survives": any(
                    candidate.source.source_doc_id == source.source_doc_id for candidate in selected
                ),
            }
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
    bundle.close()
    return {"tasks": len(tasks), "written": written}


def main() -> int:
    parser = argparse.ArgumentParser(description="RGRD E05 hard-benign worker")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector-commit", required=True)
    parser.add_argument("--model-revision", action="append", default=[])
    args = parser.parse_args()
    args.model_revision = dict(item.split("=", 1) for item in args.model_revision)
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
