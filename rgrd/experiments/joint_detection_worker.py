from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import yaml

from rgrd.experiments.detection_worker import _pipeline
from rgrd.schema import SourceDocument
from rgrd.experiments.resume import event_provenance, prepare_jsonl_resume


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def run(args: argparse.Namespace) -> dict[str, int]:
    root = args.root.resolve()
    rows = [
        json.loads(line)
        for line in args.tasks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for index, row in enumerate(rows) if index % args.shards == args.shard_id]
    config = yaml.safe_load((root / "configs/pipeline/v0.yaml").read_text(encoding="utf-8"))
    datasets = yaml.safe_load((root / "configs/datasets.yaml").read_text(encoding="utf-8"))[
        "datasets"
    ]
    _, bundle, pipeline = _pipeline(root, "nq", args.device, args.model_revision)
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
        for row in rows:
            if str(row["task_id"]) in completed:
                continue
            source = SourceDocument(
                source_doc_id=f"joint-gcg:{row['task_id']}",
                text=str(row["optimized_text"]),
                uri=f"artifact://joint-gcg/{row['task_id']}",
                metadata={"family": "Joint-GCG", "official_log_sha256": row["official_log_sha256"]},
            )
            query_embedding, selected, frozen = pipeline.retrieve(
                str(row["query"]), injected_sources=[(source, [], [])]
            )
            event, layout = pipeline.build_detector_event(
                query_id=str(row["task_id"]),
                query=str(row["query"]),
                dataset="nq",
                split=str(datasets["nq"]["split"]),
                selected=selected,
                frozen=frozen,
                detector_code_commit=args.detector_commit,
                seed=int(config["seed"]),
            )
            role = pipeline.score_detector_event(
                event,
                candidates=pipeline.runtime_candidates(selected),
                query_embedding=query_embedding,
                frozen=frozen,
                layout=layout,
            )
            target = _normalized(str(row["incorrect_answer"]))
            shadow = _normalized(event.shadow_answer)
            value = {
                **provenance,
                "task_id": str(row["task_id"]),
                "query_id": str(row["task_id"]),
                "family": "Joint-GCG",
                "partition": "joint_gcg_test",
                "score": role.query_score,
                "role_identifiable": any(chunk.role_identifiable for chunk in role.chunks),
                "retrieval_survived": any(
                    candidate.source.source_doc_id == source.source_doc_id for candidate in selected
                ),
                "end_to_end_success": bool(target) and target in shadow,
                "shadow_answer": event.shadow_answer,
                "selected_for_generation": list(role.selected_for_generation),
                "official_log_sha256": row["official_log_sha256"],
            }
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
    bundle.close()
    return {"tasks": len(rows), "written": written}


def main() -> int:
    parser = argparse.ArgumentParser(description="Label-free RGRD evaluator for Joint-GCG")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector-commit", required=True)
    parser.add_argument("--model-revision", action="append", default=[])
    arguments = parser.parse_args()
    arguments.model_revision = dict(item.split("=", 1) for item in arguments.model_revision)
    print(json.dumps(run(arguments), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
