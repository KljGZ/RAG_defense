from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

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


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _wordpiece_to_bpe(token: str) -> str:
    return token[2:] if token.startswith("##") else "Ġ" + token


def _bpe_to_wordpiece(token: str) -> str:
    return "##" + token if not token.startswith("Ġ") else token[1:]


def _common_embedding_pairs(
    retriever_path: Path, generator_path: Path, device: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    retriever_tokenizer = AutoTokenizer.from_pretrained(
        retriever_path, local_files_only=True, use_fast=True
    )
    generator_tokenizer = AutoTokenizer.from_pretrained(
        generator_path, local_files_only=True, use_fast=True
    )
    retriever_vocab = retriever_tokenizer.get_vocab()
    generator_vocab = generator_tokenizer.get_vocab()
    mapped = {
        _wordpiece_to_bpe(token): token
        for token in retriever_vocab
        if _wordpiece_to_bpe(token) in generator_vocab
    }
    common_generator_tokens = sorted(mapped)
    if len(common_generator_tokens) < 1000:
        raise RuntimeError(
            f"Joint-GCG projection found only {len(common_generator_tokens)} common tokens"
        )
    generator_ids = generator_tokenizer.convert_tokens_to_ids(common_generator_tokens)
    retriever_ids = retriever_tokenizer.convert_tokens_to_ids(
        [_bpe_to_wordpiece(token) for token in common_generator_tokens]
    )
    retriever = (
        AutoModel.from_pretrained(
            retriever_path,
            local_files_only=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to(device)
    )
    generator = (
        AutoModel.from_pretrained(
            generator_path,
            local_files_only=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to(device)
    )
    with torch.inference_mode():
        generator_values = (
            generator.get_input_embeddings()(torch.tensor(generator_ids, device=device))
            .float()
            .cpu()
            .numpy()
        )
        retriever_values = (
            retriever.get_input_embeddings()(torch.tensor(retriever_ids, device=device))
            .float()
            .cpu()
            .numpy()
        )
    dimensions = {
        "common_tokens": len(common_generator_tokens),
        "generator_dimension": int(generator_values.shape[1]),
        "retriever_dimension": int(retriever_values.shape[1]),
        "generator_vocab": len(generator_vocab),
        "retriever_vocab": len(retriever_vocab),
    }
    del generator, retriever
    gc.collect()
    torch.cuda.empty_cache()
    return generator_values, retriever_values, dimensions


def _train_projection(
    generator_values: np.ndarray,
    retriever_values: np.ndarray,
    output: Path,
    *,
    device: str,
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[Path, dict[str, Any]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    class Autoencoder(nn.Module):
        def __init__(self, generator_dimension: int, retriever_dimension: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(generator_dimension, 2048),
                nn.ReLU(True),
                nn.Linear(2048, 1024),
                nn.ReLU(True),
                nn.Linear(1024, retriever_dimension),
            )
            self.decoder = nn.Sequential(
                nn.Linear(retriever_dimension, 1024),
                nn.ReLU(True),
                nn.Linear(1024, 2048),
                nn.ReLU(True),
                nn.Linear(2048, generator_dimension),
            )

        def forward(self, values: Any) -> tuple[Any, Any]:
            encoded = self.encoder(values)
            return encoded, self.decoder(encoded)

    values = TensorDataset(
        torch.from_numpy(generator_values).float(), torch.from_numpy(retriever_values).float()
    )
    train_count = int(math.floor(0.9 * len(values)))
    validation_count = len(values) - train_count
    split_generator = torch.Generator().manual_seed(seed)
    train, validation = random_split(
        values, [train_count, validation_count], generator=split_generator
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train, batch_size=64, shuffle=True, generator=loader_generator, num_workers=0
    )
    validation_loader = DataLoader(validation, batch_size=64, shuffle=False, num_workers=0)
    model = Autoencoder(generator_values.shape[1], retriever_values.shape[1]).to(device)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.MSELoss()
    best = float("inf")
    stale = 0
    history: list[dict[str, float | int]] = []
    checkpoint = output / "autoencoder-best.pt"
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(max_epochs):
        model.train()
        training_losses: list[float] = []
        for generator_batch, retriever_batch in train_loader:
            generator_batch = generator_batch.to(device)
            retriever_batch = retriever_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            encoded, decoded = model(generator_batch)
            generator_loss = torch.sqrt(criterion(decoded, generator_batch))
            retriever_loss = torch.sqrt(criterion(encoded, retriever_batch))
            loss = 0.25 * generator_loss + 0.75 * retriever_loss
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        model.eval()
        validation_losses: list[float] = []
        with torch.inference_mode():
            for generator_batch, retriever_batch in validation_loader:
                generator_batch = generator_batch.to(device)
                retriever_batch = retriever_batch.to(device)
                encoded, decoded = model(generator_batch)
                generator_loss = torch.sqrt(criterion(decoded, generator_batch))
                retriever_loss = torch.sqrt(criterion(encoded, retriever_batch))
                validation_losses.append(
                    float((0.25 * generator_loss + 0.75 * retriever_loss).cpu())
                )
        train_loss = float(np.mean(training_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best:
            best = validation_loss
            stale = 0
            torch.save(
                {
                    "encoder": model.encoder.state_dict(),
                    "generator_dimension": generator_values.shape[1],
                    "retriever_dimension": retriever_values.shape[1],
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    return checkpoint, {
        "epochs_completed": len(history),
        "best_validation_loss": best,
        "early_stopping_patience": patience,
        "history": history,
    }


def _project_full_generator_vocab(
    generator_path: Path,
    checkpoint: Path,
    output: Path,
    *,
    device: str,
) -> tuple[Path, dict[str, int]]:
    import torch
    from torch import nn
    from transformers import AutoModel

    saved = torch.load(checkpoint, map_location="cpu")
    encoder = nn.Sequential(
        nn.Linear(int(saved["generator_dimension"]), 2048),
        nn.ReLU(True),
        nn.Linear(2048, 1024),
        nn.ReLU(True),
        nn.Linear(1024, int(saved["retriever_dimension"])),
    )
    encoder.load_state_dict(saved["encoder"])
    encoder.eval().to(device)
    generator = (
        AutoModel.from_pretrained(
            generator_path,
            local_files_only=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to(device)
    )
    embeddings = generator.get_input_embeddings().weight
    projection_path = output / "full_generator_projection.npy"
    projected = np.lib.format.open_memmap(
        projection_path,
        mode="w+",
        dtype=np.float32,
        shape=(embeddings.shape[0], int(saved["retriever_dimension"])),
    )
    with torch.inference_mode():
        for start in range(0, embeddings.shape[0], 2048):
            values = embeddings[start : start + 2048].float()
            projected[start : start + len(values)] = encoder(values).float().cpu().numpy()
    projected.flush()
    dimensions = {
        "generator_vocab": int(embeddings.shape[0]),
        "generator_dimension": int(embeddings.shape[1]),
        "retriever_dimension": int(saved["retriever_dimension"]),
    }
    del projected, generator, encoder
    gc.collect()
    torch.cuda.empty_cache()
    return projection_path, dimensions


def _calculate_transfer_matrix(
    retriever_path: Path,
    projection_path: Path,
    output: Path,
    *,
    device: str,
    row_block: int,
) -> tuple[Path, dict[str, Any]]:
    import torch
    from transformers import AutoModel

    projected = np.load(projection_path, mmap_mode="r")
    projection = torch.from_numpy(np.asarray(projected)).float().to(device)
    retriever = (
        AutoModel.from_pretrained(
            retriever_path,
            local_files_only=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to(device)
    )
    retriever_embeddings = retriever.get_input_embeddings().weight.float()
    gram = projection.T @ projection
    inverse = torch.linalg.pinv(gram, hermitian=True)
    middle = retriever_embeddings @ inverse
    matrix_path = output / "transfer_matrix.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=np.float32,
        shape=(retriever_embeddings.shape[0], projection.shape[0]),
    )
    with torch.inference_mode():
        for start in range(0, middle.shape[0], row_block):
            values = middle[start : start + row_block] @ projection.T
            matrix[start : start + len(values)] = values.cpu().numpy()
    matrix.flush()
    validation_indices = [0, int(middle.shape[0] // 2), int(middle.shape[0] - 1)]
    reconstruction_errors = []
    with torch.inference_mode():
        for index in validation_indices:
            coefficients = torch.from_numpy(np.asarray(matrix[index])).to(device)
            reconstructed = coefficients @ projection
            target = retriever_embeddings[index]
            reconstruction_errors.append(
                float(
                    torch.linalg.vector_norm(reconstructed - target)
                    / torch.linalg.vector_norm(target)
                )
            )
    result = {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "dtype": str(matrix.dtype),
        "method": "vectorized minimum-norm least squares: B @ pinv(P.T @ P) @ P.T",
        "equivalence": "same minimum-norm estimand as the official per-token torch.linalg.lstsq loop",
        "validation_indices": validation_indices,
        "relative_reconstruction_errors": reconstruction_errors,
    }
    del matrix, retriever, retriever_embeddings, projection, middle, inverse, gram
    gc.collect()
    torch.cuda.empty_cache()
    return matrix_path, result


def build_projection(
    *,
    retriever_path: Path,
    generator_path: Path,
    output: Path,
    official_source: Path,
    device: str,
    seed: int = 42,
    max_epochs: int = 500,
    patience: int = 10,
    row_block: int = 128,
) -> dict[str, Any]:
    manifest_path = output / "projection_manifest.json"
    transfer_path = output / "transfer_matrix.npy"
    if manifest_path.is_file() and transfer_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("transfer_matrix_sha256") == sha256_file(transfer_path):
            return manifest
    _seed(seed)
    generator_values, retriever_values, common = _common_embedding_pairs(
        retriever_path, generator_path, device
    )
    checkpoint, training = _train_projection(
        generator_values,
        retriever_values,
        output,
        device=device,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
    )
    del generator_values, retriever_values
    gc.collect()
    projection_path, dimensions = _project_full_generator_vocab(
        generator_path, checkpoint, output, device=device
    )
    transfer_path, calculation = _calculate_transfer_matrix(
        retriever_path,
        projection_path,
        output,
        device=device,
        row_block=row_block,
    )
    manifest = {
        "schema_version": 1,
        "completed_at": utc_now(),
        "seed": seed,
        "retriever_path": str(retriever_path.resolve()),
        "generator_path": str(generator_path.resolve()),
        "official_train_source": str(official_source.resolve()),
        "official_train_source_sha256": sha256_file(official_source),
        "implementation_note": (
            "The pinned official training file is syntactically invalid under Python 3.10 because of "
            "nested quotes in two f-strings. This module preserves its architecture, loss, optimizer, "
            "scheduler, 90/10 split, 500-epoch ceiling, and patience-10 stopping; token ordering is "
            "sorted for determinism. The official per-token least-squares loop is evaluated in its "
            "vectorized minimum-norm form. The third-party clone remains unmodified."
        ),
        "common_token_mapping": common,
        "training": training,
        "dimensions": dimensions,
        "calculation": calculation,
        "checkpoint_sha256": sha256_file(checkpoint),
        "full_projection_sha256": sha256_file(projection_path),
        "transfer_matrix_sha256": sha256_file(transfer_path),
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the pinned Joint-GCG CVP transfer matrix")
    parser.add_argument("--retriever", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    result = build_projection(
        retriever_path=arguments.retriever,
        generator_path=arguments.generator,
        output=arguments.output,
        official_source=arguments.official_source,
        device=arguments.device,
        seed=arguments.seed,
    )
    print(json.dumps({"transfer_matrix_sha256": result["transfer_matrix_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
