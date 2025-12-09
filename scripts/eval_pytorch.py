#!/usr/bin/env python3
"""Evaluate a trained PyTorch CESNET-QUIC22 model on held-out parquet files."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from scripts.train_pytorch import (
    ChunkManifest,
    HybridCNN,
    SingleChunkDataset,
    prepare_split_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved PyTorch QUIC classifier")
    parser.add_argument("--data-dir", type=str, default="datasets", help="Base data directory")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="datasets/cache_pytorch",
        help="Directory containing cached tensors (same as training)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--test-files",
        type=str,
        nargs="+",
        help="List of parquet files (relative to --data-dir unless absolute) used for evaluation",
    )
    group.add_argument(
        "--test-dir",
        type=str,
        help="Directory whose .parquet files (relative to --data-dir unless absolute) are evaluated",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--split-name", type=str, default="test", help="Cache split name to use (default: test)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers per chunk (0 avoids duplicating chunk tensors per worker)",
    )
    parser.add_argument("--cache-batch-rows", type=int, default=65536)
    parser.add_argument("--cache-workers", type=int, default=18, help="Parallel workers for cache building")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--sequence-only", action="store_true", help="Zero out static/tabular inputs like sequence-only checkpoint")
    return parser.parse_args()


def resolve_files(data_dir: Path, files: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    for item in files:
        path = Path(item)
        if not path.is_absolute():
            path = data_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Missing parquet file: {path}")
        resolved.append(path)
    return resolved


def resolve_directory(data_dir: Path, directory: str) -> List[Path]:
    path = Path(directory)
    if not path.is_absolute():
        path = data_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Test directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected directory but got: {path}")
    parquet_files = sorted(path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return parquet_files


def load_training_meta(cache_dir: Path) -> Dict:
    meta_path = cache_dir / "train" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Training cache metadata missing: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return meta


def build_test_cache(
    split_name: str,
    files: Sequence[Path],
    cache_args: SimpleNamespace,
    label_map: Dict[str, int],
    version_values: Sequence[int],
    normalization: Dict[str, List[float]],
    rebuild: bool,
) -> None:
    prepare_split_cache(
        split=split_name,
        files=files,
        args=cache_args,
        label_map=label_map,
        version_values=version_values,
        normalization=normalization,
        rebuild=rebuild,
    )


def release_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_loader(model: HybridCNN, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            sequences = batch["sequences"].to(device, non_blocking=True)
            tabular = batch["tabular"].to(device, non_blocking=True)
            version_idx = batch["version_idx"].to(device, non_blocking=True).long()
            labels = batch["label"].to(device, non_blocking=True).long()
            if args.sequence_only:
                tabular = torch.zeros_like(tabular)
                version_idx = torch.zeros_like(version_idx)
            logits = model(sequences, tabular, version_idx)
            pred = logits.argmax(dim=1)
            preds.append(pred.cpu().numpy())
            targets.append(labels.cpu().numpy())
    if not preds:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(preds), np.concatenate(targets)


def evaluate_manifest(
    model: HybridCNN,
    manifest: ChunkManifest,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    preds_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    for chunk_idx in range(manifest.num_chunks):
        chunk_path = manifest.chunk_path(chunk_idx)
        dataset = SingleChunkDataset(chunk_path, manifest.normalization)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        chunk_preds, chunk_targets = evaluate_loader(model, loader, device, args)
        if chunk_preds.size == 0:
            continue
        preds_all.append(chunk_preds)
        targets_all.append(chunk_targets)
        del dataset, loader
        gc.collect()
        release_cuda_cache()
    if not preds_all:
        raise ValueError(f"No samples found in split '{manifest.split}'")
    return np.concatenate(preds_all), np.concatenate(targets_all)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    if args.test_files:
        test_files = resolve_files(data_dir, args.test_files)
    else:
        assert args.test_dir is not None
        test_files = resolve_directory(data_dir, args.test_dir)
    cache_root = Path(args.cache_dir)
    train_meta = load_training_meta(cache_root)
    label_map = {k: int(v) for k, v in train_meta["label_to_index"].items()}
    version_values = train_meta.get("version_values", [])
    normalization = train_meta["normalization"]

    cache_args = SimpleNamespace(
        cache_dir=str(args.cache_dir),
        max_seq_len=train_meta["seq_len"],
        cache_batch_rows=args.cache_batch_rows,
        cache_workers=args.cache_workers,
    )

    build_test_cache(
        split_name=args.split_name,
        files=test_files,
        cache_args=cache_args,
        label_map=label_map,
        version_values=version_values,
        normalization=normalization,
        rebuild=args.rebuild_cache,
    )

    manifest = ChunkManifest(cache_root, args.split_name)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint.get("args")
    if saved_args is None:
        raise ValueError("Checkpoint missing training args metadata; please evaluate a checkpoint produced by the chunked trainer")
    model_args = SimpleNamespace(
        seq_hidden=saved_args["seq_hidden"],
        mlp_hidden=saved_args["mlp_hidden"],
        dropout=saved_args["dropout"],
        version_embed_dim=saved_args.get("version_embed_dim", 16),
    )

    model = HybridCNN(
        seq_len=manifest.seq_len,
        tab_dim=manifest.tab_dim,
        num_classes=manifest.num_classes,
        num_versions=max(getattr(manifest, "num_versions", 0), 1),
        args=model_args,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    preds, targets = evaluate_manifest(model, manifest, args.batch_size, args.num_workers, device, args)

    overall_acc = (preds == targets).mean()
    macro_f1 = f1_score(targets, preds, average="macro")

    label_names = manifest.meta["index_to_label"]
    supports = np.bincount(targets, minlength=len(label_names))
    correct = np.bincount(targets[preds == targets], minlength=len(label_names))
    per_class_acc = np.divide(correct, supports, out=np.zeros_like(correct, dtype=np.float64), where=supports > 0)
    f1_per_class = f1_score(targets, preds, labels=np.arange(len(label_names)), average=None)

    order = np.argsort(-per_class_acc)
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")
    print("Service performance (sorted by accuracy):")
    print(f"{'Service':35s} acc  f1    support")
    for idx in order:
        if supports[idx] == 0:
            continue
        print(
            f"{label_names[idx]:35s} {per_class_acc[idx]:.4f} {f1_per_class[idx]:.4f} {supports[idx]:6d}"
        )


if __name__ == "__main__":
    main()
