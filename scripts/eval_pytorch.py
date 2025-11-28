#!/usr/bin/env python3
"""Evaluate a trained PyTorch CESNET-QUIC22 model on held-out parquet files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from scripts.train_pytorch import (
    CesnetQuicDataset,
    HybridCNN,
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
    parser.add_argument(
        "--test-files",
        type=str,
        nargs="+",
        required=True,
        help="List of parquet files (relative to --data-dir unless absolute) used for evaluation",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--split-name", type=str, default="test", help="Cache split name to use (default: test)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache-batch-rows", type=int, default=65536)
    parser.add_argument("--rebuild-cache", action="store_true")
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


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[HybridCNN, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = HybridCNN(
        seq_len=ckpt_args["max_seq_len"],
        tab_dim=ckpt_args["tab_dim"] if "tab_dim" in ckpt_args else checkpoint["model"]["head.0.weight"].shape[1],
        num_classes=ckpt_args["num_classes"],
        num_versions=ckpt_args.get("num_versions", 1),
        args=SimpleNamespace(
            seq_hidden=ckpt_args["seq_hidden"],
            mlp_hidden=ckpt_args["mlp_hidden"],
            dropout=ckpt_args["dropout"],
            sni_embed_dim=ckpt_args["sni_embed_dim"],
            ua_embed_dim=ckpt_args["ua_embed_dim"],
            version_embed_dim=ckpt_args["version_embed_dim"],
            sni_hash_size=ckpt_args["sni_hash_size"],
            ua_hash_size=ckpt_args["ua_hash_size"],
        ),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, ckpt_args


def evaluate(model: HybridCNN, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            sequences = torch.as_tensor(batch["sequences"], device=device)
            tabular = torch.as_tensor(batch["tabular"], device=device)
            sni_idx = torch.as_tensor(batch["sni_idx"], device=device).long()
            ua_idx = torch.as_tensor(batch["ua_idx"], device=device).long()
            version_idx = torch.as_tensor(batch["version_idx"], device=device).long()
            labels = torch.as_tensor(batch["label"], device=device).long()
            logits = model(sequences, tabular, sni_idx, ua_idx, version_idx)
            pred = logits.argmax(dim=1)
            preds.append(pred.cpu().numpy())
            targets.append(labels.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_files = resolve_files(Path(args.data_dir), args.test_files)
    cache_root = Path(args.cache_dir)
    train_meta = load_training_meta(cache_root)
    label_map = {k: int(v) for k, v in train_meta["label_to_index"].items()}
    version_values = train_meta.get("version_values", [])
    normalization = train_meta["normalization"]

    ckpt_args = train_meta  # placeholder to capture dims from meta if needed

    cache_args = SimpleNamespace(
        cache_dir=str(args.cache_dir),
        max_seq_len=train_meta["seq_len"],
        sni_hash_size=train_meta["sni_hash_size"],
        ua_hash_size=train_meta["ua_hash_size"],
        cache_batch_rows=args.cache_batch_rows,
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

    dataset = CesnetQuicDataset(cache_root, args.split_name)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    saved_args = checkpoint.get("args", train_meta)
    model_args = SimpleNamespace(
        seq_hidden=saved_args["seq_hidden"],
        mlp_hidden=saved_args["mlp_hidden"],
        dropout=saved_args["dropout"],
        sni_embed_dim=saved_args["sni_embed_dim"],
        ua_embed_dim=saved_args["ua_embed_dim"],
        version_embed_dim=saved_args["version_embed_dim"],
        sni_hash_size=saved_args["sni_hash_size"],
        ua_hash_size=saved_args["ua_hash_size"],
    )

    model = HybridCNN(
        seq_len=train_meta["seq_len"],
        tab_dim=dataset.tabular.shape[1],
        num_classes=dataset.num_classes,
        num_versions=dataset.num_versions,
        args=model_args,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    preds, targets = evaluate(model, loader, device)

    overall_acc = (preds == targets).mean()
    macro_f1 = f1_score(targets, preds, average="macro")

    label_names = train_meta["index_to_label"]
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
