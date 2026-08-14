from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class BEIRRecord:
    record_id: str
    text: str
    title: str
    metadata: dict[str, object]


def _record_id(value: dict[str, object]) -> str:
    for key in ("_id", "id", "docid", "query_id"):
        if key in value:
            return str(value[key])
    raise ValueError("BEIR JSONL record is missing an id")


def iter_corpus(path: Path) -> Iterator[BEIRRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            text = str(value.get("text", ""))
            title = str(value.get("title", ""))
            if title and title not in text:
                combined = f"{title}\n{text}" if text else title
            else:
                combined = text
            metadata = {key: item for key, item in value.items() if key not in {"text", "title"}}
            yield BEIRRecord(_record_id(value), combined, title, metadata)


def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    for record in iter_corpus(path):
        queries[record.record_id] = record.text
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"qrels file has no header: {path}")
        query_key = next(
            (key for key in ("query-id", "query_id", "qid") if key in reader.fieldnames), None
        )
        corpus_key = next(
            (key for key in ("corpus-id", "corpus_id", "docid") if key in reader.fieldnames), None
        )
        score_key = next((key for key in ("score", "relevance") if key in reader.fieldnames), None)
        if not query_key or not corpus_key or not score_key:
            raise ValueError(f"unsupported qrels header: {reader.fieldnames}")
        for row in reader:
            qrels.setdefault(str(row[query_key]), {})[str(row[corpus_key])] = int(row[score_key])
    return qrels
