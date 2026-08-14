from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rgrd.data import iter_corpus
from rgrd.ingestion import chunk_source
from rgrd.models import DenseRetriever
from rgrd.provenance import sha256_file, utc_now
from rgrd.schema import ChunkLineage, SourceDocument


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


def _flush_embedding_batch(
    encoder: DenseRetriever,
    texts: list[str],
    records: list[ChunkLineage],
    sources: list[SourceDocument],
    embedding_handle: Any,
    metadata_handle: Any,
    *,
    batch_size: int,
) -> int:
    if not texts:
        return 0
    embeddings = encoder.encode(texts, batch_size=batch_size)
    if embeddings.shape != (len(records), encoder.dimension):
        raise AssertionError("encoder returned an unexpected embedding matrix")
    embeddings.astype("<f4", copy=False).tofile(embedding_handle)
    if len(sources) != len(records):
        raise AssertionError("chunk and source queues disagree")
    for chunk, source in zip(records, sources, strict=True):
        metadata_handle.write(
            json.dumps(
                {"chunk": chunk.model_dump(mode="json"), "source": source.model_dump(mode="json")},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    count = len(records)
    texts.clear()
    records.clear()
    sources.clear()
    return count


def build_shard(
    *,
    corpus_path: Path,
    generator_tokenizer_path: Path,
    retriever_path: Path,
    output_dir: Path,
    shard_id: int,
    shards: int,
    device: str,
    chunk_size: int = 256,
    overlap: int = 32,
    batch_size: int = 96,
    flush_chunks: int = 768,
) -> dict[str, Any]:
    if not 0 <= shard_id < shards:
        raise ValueError("shard_id must lie in [0, shards)")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("transformers is required") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard-{shard_id:02d}-of-{shards:02d}"
    final_embeddings = output_dir / f"{stem}.f32"
    final_metadata = output_dir / f"{stem}.jsonl"
    complete_path = output_dir / f"{stem}.complete.json"
    if complete_path.is_file() and final_embeddings.is_file() and final_metadata.is_file():
        return json.loads(complete_path.read_text(encoding="utf-8"))
    partial_embeddings = output_dir / f".{stem}.f32.partial"
    partial_metadata = output_dir / f".{stem}.jsonl.partial"
    for path in (partial_embeddings, partial_metadata):
        if path.exists():
            path.unlink()
    tokenizer = AutoTokenizer.from_pretrained(
        generator_tokenizer_path, local_files_only=True, use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValueError("Track B chunking requires a fast tokenizer with offsets")
    encoder = DenseRetriever(retriever_path, device=device, normalize=False)
    pending_texts: list[str] = []
    pending_chunks: list[ChunkLineage] = []
    pending_sources: list[SourceDocument] = []
    document_count = 0
    chunk_count = 0
    progress_path = output_dir / f"{stem}.progress.json"
    with (
        partial_embeddings.open("wb") as embedding_handle,
        partial_metadata.open("w", encoding="utf-8") as metadata_handle,
    ):
        for document_index, record in enumerate(iter_corpus(corpus_path)):
            if document_index % shards != shard_id:
                continue
            source = SourceDocument(
                source_doc_id=record.record_id,
                text=record.text,
                uri=f"beir://{corpus_path.parent.name}/{record.record_id}",
                metadata={"title": record.title, **record.metadata},
            )
            chunks = chunk_source(
                source,
                tokenizer,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            document_count += 1
            for chunk in chunks:
                pending_texts.append(chunk.text)
                pending_chunks.append(chunk)
                pending_sources.append(source)
            if len(pending_texts) >= flush_chunks:
                chunk_count += _flush_embedding_batch(
                    encoder,
                    pending_texts,
                    pending_chunks,
                    pending_sources,
                    embedding_handle,
                    metadata_handle,
                    batch_size=batch_size,
                )
            if document_count % 5000 == 0:
                _atomic_json(
                    progress_path,
                    {
                        "captured_at": utc_now(),
                        "shard_id": shard_id,
                        "documents": document_count,
                        "chunks": chunk_count + len(pending_chunks),
                    },
                )
        chunk_count += _flush_embedding_batch(
            encoder,
            pending_texts,
            pending_chunks,
            pending_sources,
            embedding_handle,
            metadata_handle,
            batch_size=batch_size,
        )
        embedding_handle.flush()
        os.fsync(embedding_handle.fileno())
        metadata_handle.flush()
        os.fsync(metadata_handle.fileno())
    os.replace(partial_embeddings, final_embeddings)
    os.replace(partial_metadata, final_metadata)
    expected_bytes = chunk_count * encoder.dimension * np.dtype("<f4").itemsize
    if final_embeddings.stat().st_size != expected_bytes:
        raise AssertionError("raw embedding shard has an invalid byte count")
    result = {
        "schema_version": 1,
        "completed_at": utc_now(),
        "shard_id": shard_id,
        "shards": shards,
        "documents": document_count,
        "chunks": chunk_count,
        "dimension": encoder.dimension,
        "embedding_path": str(final_embeddings),
        "metadata_path": str(final_metadata),
        "embedding_sha256": sha256_file(final_embeddings),
        "metadata_sha256": sha256_file(final_metadata),
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    _atomic_json(complete_path, result)
    _atomic_json(progress_path, result)
    return result


def _batched_rows(handle: Any, dimension: int, rows: int = 65536) -> Iterable[np.ndarray]:
    values_per_batch = dimension * rows
    while True:
        values = np.fromfile(handle, dtype="<f4", count=values_per_batch)
        if len(values) == 0:
            return
        if len(values) % dimension:
            raise ValueError("embedding file ends mid-row")
        yield np.ascontiguousarray(values.reshape(-1, dimension), dtype=np.float32)


def merge_shards(output_dir: Path, *, shards: int) -> dict[str, Any]:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("faiss-cpu is required") from exc
    manifests: list[dict[str, Any]] = []
    for shard_id in range(shards):
        path = output_dir / f"shard-{shard_id:02d}-of-{shards:02d}.complete.json"
        if not path.is_file():
            raise FileNotFoundError(f"incomplete index shard: {path}")
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    dimensions = {int(item["dimension"]) for item in manifests}
    if len(dimensions) != 1:
        raise ValueError(f"shard dimensions disagree: {dimensions}")
    dimension = dimensions.pop()
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(dimension)
    database_partial = output_dir / ".chunks.sqlite3.partial"
    index_partial = output_dir / ".chunks.index.partial"
    for path in (database_partial, index_partial):
        if path.exists():
            path.unlink()
    connection = sqlite3.connect(database_partial)
    try:
        connection.execute(
            "CREATE TABLE chunks (faiss_id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL, "
            "source_doc_id TEXT NOT NULL, text TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE sources (source_doc_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        connection.execute("CREATE INDEX chunks_source_idx ON chunks(source_doc_id)")
        next_id = 0
        for manifest in manifests:
            embedding_path = Path(manifest["embedding_path"])
            metadata_path = Path(manifest["metadata_path"])
            shard_rows = 0
            with embedding_path.open("rb") as embedding_handle:
                for matrix in _batched_rows(embedding_handle, dimension):
                    index.add(matrix)
                    shard_rows += len(matrix)
            rows: list[tuple[int, str, str, str, str]] = []
            source_rows: dict[str, str] = {}
            with metadata_path.open("r", encoding="utf-8") as metadata_handle:
                for line in metadata_handle:
                    value = json.loads(line)
                    chunk_value = value["chunk"]
                    source_value = value["source"]
                    rows.append(
                        (
                            next_id,
                            str(chunk_value["chunk_id"]),
                            str(chunk_value["source_doc_id"]),
                            str(chunk_value["text"]),
                            json.dumps(chunk_value, ensure_ascii=False, sort_keys=True),
                        )
                    )
                    source_rows[str(source_value["source_doc_id"])] = json.dumps(
                        source_value, ensure_ascii=False, sort_keys=True
                    )
                    next_id += 1
                    if len(rows) >= 10000:
                        connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", rows)
                        connection.executemany(
                            "INSERT OR IGNORE INTO sources VALUES (?, ?)", source_rows.items()
                        )
                        rows.clear()
                        source_rows.clear()
                if rows:
                    connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", rows)
                    connection.executemany(
                        "INSERT OR IGNORE INTO sources VALUES (?, ?)", source_rows.items()
                    )
            if shard_rows != int(manifest["chunks"]):
                raise AssertionError("embedding and manifest row counts disagree")
        connection.commit()
        if index.ntotal != next_id:
            raise AssertionError("FAISS and metadata row counts disagree")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("VACUUM")
    finally:
        connection.close()
    faiss.write_index(index, str(index_partial))
    final_index = output_dir / "chunks.index"
    final_database = output_dir / "chunks.sqlite3"
    os.replace(index_partial, final_index)
    os.replace(database_partial, final_database)
    manifest = {
        "schema_version": 1,
        "completed_at": utc_now(),
        "index_type": "IndexFlatIP",
        "dimension": dimension,
        "chunks": int(index.ntotal),
        "shards": shards,
        "index_path": str(final_index),
        "database_path": str(final_database),
        "index_sha256": sha256_file(final_index),
        "database_sha256": sha256_file(final_database),
        "source_shards": manifests,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class IndexedChunk:
    faiss_id: int
    chunk: ChunkLineage
    source: SourceDocument


class ExactIndexBundle:
    def __init__(self, manifest_path: Path):
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faiss-cpu is required") from exc
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        faiss.omp_set_num_threads(1)
        self.index = faiss.read_index(self.manifest["index_path"])
        if not isinstance(self.index, faiss.IndexFlatIP):
            raise ValueError("RGRD-V0 only accepts an exact IndexFlatIP")
        self.connection = sqlite3.connect(
            f"file:{self.manifest['database_path']}?mode=ro", uri=True
        )

    def close(self) -> None:
        self.connection.close()

    def search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, list[IndexedChunk]]:
        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        scores, identifiers = self.index.search(query, top_k)
        ids = [int(value) for value in identifiers[0] if value >= 0]
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(
            f"SELECT c.faiss_id, c.payload, s.payload FROM chunks c JOIN sources s "
            f"ON c.source_doc_id=s.source_doc_id WHERE c.faiss_id IN ({placeholders})",
            ids,
        ).fetchall()
        mapped = {
            int(row[0]): IndexedChunk(
                faiss_id=int(row[0]),
                chunk=ChunkLineage.model_validate_json(row[1]),
                source=SourceDocument.model_validate_json(row[2]),
            )
            for row in rows
        }
        chunks = [mapped[value] for value in ids]
        return scores[0, : len(ids)].astype(np.float32), chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact RGRD Track-B index")
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--corpus", type=Path, required=True)
    shard.add_argument("--generator-tokenizer", type=Path, required=True)
    shard.add_argument("--retriever", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--shard-id", type=int, required=True)
    shard.add_argument("--shards", type=int, required=True)
    shard.add_argument("--device", required=True)
    shard.add_argument("--batch-size", type=int, default=96)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--shards", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.command == "shard":
        result = build_shard(
            corpus_path=arguments.corpus,
            generator_tokenizer_path=arguments.generator_tokenizer,
            retriever_path=arguments.retriever,
            output_dir=arguments.output,
            shard_id=arguments.shard_id,
            shards=arguments.shards,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
    else:
        result = merge_shards(arguments.output, shards=arguments.shards)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
