#!/usr/bin/env python3
"""Distributed PyTorch training pipeline for the CESNET-QUIC22 dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import amp, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid CNN trainer for CESNET-QUIC22")
    parser.add_argument("--data-dir", type=str, default="datasets/training", help="Directory with parquet files")
    parser.add_argument("--output-dir", type=str, default="artifacts/cnn_ddp", help="Directory for checkpoints")
    parser.add_argument("--cache-dir", type=str, default="datasets/cache_pytorch", help="On-disk cache for processed tensors")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--val-batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of files for validation if --val-files not set")
    parser.add_argument("--train-files", type=str, nargs="*", default=None, help="Optional subset of parquet files for training")
    parser.add_argument("--val-files", type=str, nargs="*", default=None, help="Optional subset of parquet files for validation")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force regeneration of cached tensors")
    parser.add_argument("--cache-batch-rows", type=int, default=65536, help="Rows per batch when parsing parquet for cache")
    parser.add_argument("--sni-hash-size", type=int, default=65536)
    parser.add_argument("--ua-hash-size", type=int, default=16384)
    parser.add_argument("--sni-embed-dim", type=int, default=64)
    parser.add_argument("--ua-embed-dim", type=int, default=64)
    parser.add_argument("--version-embed-dim", type=int, default=16)
    parser.add_argument("--mlp-hidden", type=int, default=512)
    parser.add_argument("--seq-hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true", dest="use_amp", default=True, help="Enable mixed precision (default)")
    parser.add_argument("--no-amp", action="store_false", dest="use_amp", help="Disable mixed precision")
    parser.add_argument("--amp-dtype", type=str, choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--world-size", type=int, default=None, help="Number of GPUs/processes (defaults to torch.cuda.device_count())")
    parser.add_argument("--dist-backend", type=str, default="nccl")
    parser.add_argument("--dist-url", type=str, default="env://")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=0, help="Optional checkpoint frequency in epochs")
    parser.add_argument("--amp-loss-scale", type=float, default=1024.0)
    parser.add_argument("--cache-wait-seconds", type=int, default=7200, help="Seconds non-primary ranks wait for cache files")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_hash(text: Optional[str], num_buckets: int) -> int:
    if not text or num_buckets <= 1:
        return 0
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    value = int(digest, 16)
    return (value % (num_buckets - 1)) + 1


def log1p_safe(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return math.log1p(max(float(value), 0.0))


def wait_for_paths(paths: Sequence[Path], timeout: int, poll_seconds: int = 5) -> None:
    if not paths:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(path.exists() for path in paths):
            return
        time.sleep(poll_seconds)
    missing = [str(path) for path in paths if not path.exists()]
    raise TimeoutError(f"Timed out waiting for cache files: {missing}")


def build_caches_if_needed(args: argparse.Namespace, is_primary: bool) -> Tuple[List[Path], List[Path]]:
    data_dir = Path(args.data_dir)
    train_files, val_files = resolve_file_lists(data_dir, args.train_files, args.val_files, args.val_split)
    cache_root = Path(args.cache_dir)
    train_meta_path = cache_root / "train" / "meta.json"
    val_meta_path = cache_root / "val" / "meta.json"

    need_train = args.rebuild_cache or not train_meta_path.exists()
    need_val = bool(val_files) and (args.rebuild_cache or not val_meta_path.exists())

    if is_primary:
        if need_train or need_val:
            print(
                f"Preparing caches (train files={len(train_files)}, val files={len(val_files)})",
                flush=True,
            )
            label_map: Dict[str, int]
            version_values: Sequence[int]
            normalization: Optional[Dict[str, List[float]]] = None
            if need_train:
                label_map, version_values, _ = collect_label_and_version_info(train_files)
                train_meta = prepare_split_cache(
                    "train",
                    train_files,
                    args,
                    label_map,
                    version_values,
                    None,
                    args.rebuild_cache,
                )
                normalization = train_meta["normalization"]
            else:
                with open(train_meta_path, "r", encoding="utf-8") as handle:
                    cached_meta = json.load(handle)
                label_map = {k: int(v) for k, v in cached_meta["label_to_index"].items()}
                version_values = cached_meta.get("version_values", [])
                normalization = cached_meta["normalization"]

            if need_val and val_files:
                prepare_split_cache(
                    "val",
                    val_files,
                    args,
                    label_map,
                    version_values,
                    normalization,
                    args.rebuild_cache,
                )
    else:
        wait_targets: List[Path] = []
        if need_train:
            wait_targets.append(train_meta_path)
        if need_val:
            wait_targets.append(val_meta_path)
        wait_for_paths(wait_targets, timeout=args.cache_wait_seconds)

    setattr(args, "has_val_split", bool(val_files))
    return train_files, val_files


# -----------------------------------------------------------------------------
# Dataset preprocessing and caching
# -----------------------------------------------------------------------------


def resolve_file_lists(
    data_dir: Path,
    train_files: Optional[Sequence[str]],
    val_files: Optional[Sequence[str]],
    val_split: float,
) -> Tuple[List[Path], List[Path]]:
    def normalize(files: Sequence[str]) -> List[Path]:
        resolved = []
        for entry in files:
            path = Path(entry)
            if not path.is_absolute():
                path = data_dir / entry
            if not path.exists():
                raise FileNotFoundError(f"Missing parquet file: {path}")
            resolved.append(path)
        return sorted(resolved)

    all_parquet = sorted(data_dir.glob("*.parquet"))
    if not all_parquet:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    if train_files:
        train_paths = normalize(train_files)
    else:
        train_paths = list(all_parquet)

    if val_files:
        val_paths = normalize(val_files)
        train_paths = [path for path in train_paths if path not in val_paths]
    elif val_split > 0.0:
        num_val = max(1, int(len(train_paths) * val_split))
        val_paths = train_paths[-num_val:]
        train_paths = train_paths[:-num_val]
    else:
        val_paths = []

    if not train_paths:
        raise ValueError("Training file list is empty")
    return train_paths, val_paths


def collect_label_and_version_info(files: Sequence[Path], batch_rows: int = 500_000) -> Tuple[Dict[str, int], List[int], Counter]:
    label_counts: Counter = Counter()
    version_values: set[int] = set()
    columns = ["APP", "QUIC_VERSION"]
    for path in files:
        parquet_file = pq.ParquetFile(str(path))
        for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_rows):
            data = batch.to_pydict()
            for label in data["APP"]:
                if label is not None:
                    label_counts[label] += 1
            for version in data["QUIC_VERSION"]:
                if version is not None:
                    version_values.add(int(version))
    labels_sorted = sorted(label_counts.keys())
    label_map = {label: idx for idx, label in enumerate(labels_sorted)}
    version_list = sorted(version_values)
    return label_map, version_list, label_counts


class SplitCacheBuilder:
    """Builds numpy caches from parquet files for a given split."""

    columns: Tuple[str, ...] = (
        "APP",
        "PPI",
        "PHIST_SRC_SIZES",
        "PHIST_DST_SIZES",
        "PHIST_SRC_IPT",
        "PHIST_DST_IPT",
        "DURATION",
        "BYTES",
        "BYTES_REV",
        "PACKETS",
        "PACKETS_REV",
        "PPI_LEN",
        "PPI_DURATION",
        "PPI_ROUNDTRIPS",
        "FLOW_ENDREASON_IDLE",
        "FLOW_ENDREASON_ACTIVE",
        "FLOW_ENDREASON_OTHER",
        "PROTOCOL",
        "DST_ASN",
        "SRC_PORT",
        "DST_PORT",
        "QUIC_SNI",
        "QUIC_USERAGENT",
        "QUIC_VERSION",
    )

    def __init__(
        self,
        split: str,
        files: Sequence[Path],
        cache_dir: Path,
        seq_len: int,
        label_map: Dict[str, int],
        version_values: Sequence[int],
        normalization: Optional[Dict[str, List[float]]],
        sni_hash_size: int,
        ua_hash_size: int,
        batch_rows: int,
    ) -> None:
        self.split = split
        self.files = list(files)
        self.cache_dir = cache_dir
        self.seq_len = seq_len
        self.label_map = label_map
        self.version_map = {value: idx + 1 for idx, value in enumerate(version_values)}
        self.normalization = normalization
        self.sni_hash_size = sni_hash_size
        self.ua_hash_size = ua_hash_size
        self.batch_rows = batch_rows
        self.tab_dim: Optional[int] = None
        self._seq_sum = np.zeros(3, dtype=np.float64)
        self._seq_sumsq = np.zeros(3, dtype=np.float64)
        self._seq_counts = np.zeros(3, dtype=np.float64)
        self._tab_sum: Optional[np.ndarray] = None
        self._tab_sumsq: Optional[np.ndarray] = None
        self.class_counts = np.zeros(len(label_map), dtype=np.int64)
        self.processed_rows = 0

    def build(self) -> Dict[str, object]:
        total_rows = self._count_rows()
        if total_rows == 0:
            raise ValueError(f"No rows found for split {self.split}")

        sequences = np.zeros((total_rows, 3, self.seq_len), dtype=np.float32)
        tabular: Optional[np.ndarray] = None
        sni_idx = np.zeros(total_rows, dtype=np.int32)
        ua_idx = np.zeros(total_rows, dtype=np.int32)
        version_idx = np.zeros(total_rows, dtype=np.int32)
        labels = np.zeros(total_rows, dtype=np.int64)

        row_ptr = 0
        t_start = time.perf_counter()
        for path in self.files:
            parquet_file = pq.ParquetFile(str(path))
            for batch in parquet_file.iter_batches(columns=self.columns, batch_size=self.batch_rows):
                data = batch.to_pydict()
                batch_rows = len(data["APP"])
                for local_idx in range(batch_rows):
                    label_name = data["APP"][local_idx]
                    if label_name is None or label_name not in self.label_map:
                        continue

                    seq_tensor, valid_len = self._parse_sequence(data["PPI"][local_idx])
                    if self.normalization is None and valid_len > 0:
                        self._seq_sum += seq_tensor[:, :valid_len].sum(axis=1)
                        self._seq_sumsq += (seq_tensor[:, :valid_len] ** 2).sum(axis=1)
                        self._seq_counts += valid_len
                    elif self.normalization is not None:
                        seq_tensor = self._apply_seq_norm(seq_tensor)
                    sequences[row_ptr] = seq_tensor

                    tab_vec = self._build_tabular_features(data, local_idx)
                    if self.tab_dim is None:
                        self.tab_dim = len(tab_vec)
                        tabular = np.zeros((total_rows, self.tab_dim), dtype=np.float32)
                        self._tab_sum = np.zeros(self.tab_dim, dtype=np.float64)
                        self._tab_sumsq = np.zeros(self.tab_dim, dtype=np.float64)
                    assert tabular is not None
                    if self.normalization is None:
                        self._tab_sum += tab_vec
                        self._tab_sumsq += tab_vec ** 2
                    else:
                        tab_vec = self._apply_tab_norm(tab_vec)
                    tabular[row_ptr] = tab_vec

                    sni_idx[row_ptr] = stable_hash(data["QUIC_SNI"][local_idx], self.sni_hash_size)
                    ua_idx[row_ptr] = stable_hash(data["QUIC_USERAGENT"][local_idx], self.ua_hash_size)
                    version_value = data["QUIC_VERSION"][local_idx]
                    version_idx[row_ptr] = self.version_map.get(int(version_value), 0) if version_value is not None else 0

                    label_id = self.label_map[label_name]
                    labels[row_ptr] = label_id
                    self.class_counts[label_id] += 1

                    row_ptr += 1
                    if row_ptr % 500_000 == 0:
                        elapsed = time.perf_counter() - t_start
                        print(f"[{self.split}] Cached {row_ptr:,} rows in {elapsed:.1f}s", flush=True)

        sequences = sequences[:row_ptr]
        tabular = tabular[:row_ptr] if tabular is not None else np.zeros((row_ptr, 1), dtype=np.float32)
        sni_idx = sni_idx[:row_ptr]
        ua_idx = ua_idx[:row_ptr]
        version_idx = version_idx[:row_ptr]
        labels = labels[:row_ptr]
        self.processed_rows = row_ptr

        if self.normalization is None:
            seq_mean, seq_std = self._finalize_seq_stats()
            tab_mean, tab_std = self._finalize_tab_stats()
            sequences = self._normalize_sequences_array(sequences, seq_mean, seq_std)
            tabular = self._normalize_tabular_array(tabular, tab_mean, tab_std)
            normalization = {
                "seq_mean": seq_mean.tolist(),
                "seq_std": seq_std.tolist(),
                "tab_mean": tab_mean.tolist(),
                "tab_std": tab_std.tolist(),
            }
        else:
            normalization = self.normalization

        meta = {
            "split": self.split,
            "num_samples": int(row_ptr),
            "seq_len": self.seq_len,
            "tab_dim": int(tabular.shape[1]),
            "num_classes": len(self.label_map),
            "label_to_index": self.label_map,
            "index_to_label": sorted(self.label_map, key=self.label_map.get),
            "version_values": list(self.version_map.keys()),
            "class_counts": self.class_counts[: len(self.label_map)].tolist(),
            "normalization": normalization,
            "sni_hash_size": self.sni_hash_size,
            "ua_hash_size": self.ua_hash_size,
            "schema_version": 1,
        }

        self._save_arrays(sequences, tabular, sni_idx, ua_idx, version_idx, labels)
        with open(self.cache_dir / "meta.json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        return meta

    def _save_arrays(
        self,
        sequences: np.ndarray,
        tabular: np.ndarray,
        sni_idx: np.ndarray,
        ua_idx: np.ndarray,
        version_idx: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_dir / "sequences.npy", sequences.astype(np.float32))
        np.save(self.cache_dir / "tabular.npy", tabular.astype(np.float32))
        np.save(self.cache_dir / "sni_idx.npy", sni_idx.astype(np.int32))
        np.save(self.cache_dir / "ua_idx.npy", ua_idx.astype(np.int32))
        np.save(self.cache_dir / "version_idx.npy", version_idx.astype(np.int32))
        np.save(self.cache_dir / "labels.npy", labels.astype(np.int64))

    def _count_rows(self) -> int:
        total = 0
        for path in self.files:
            parquet_file = pq.ParquetFile(str(path))
            total += parquet_file.metadata.num_rows
        return total

    def _parse_sequence(self, seq_value: Optional[str]) -> Tuple[np.ndarray, int]:
        if not seq_value:
            return np.zeros((3, self.seq_len), dtype=np.float32), 0
        triplet = json.loads(seq_value)
        ipt = self._pad_sequence(triplet[0])
        ipt = np.log1p(np.maximum(ipt, 0.0))
        direction = self._pad_sequence(triplet[1])
        sizes = self._pad_sequence(triplet[2])
        stacked = np.stack([ipt, direction, sizes], axis=0).astype(np.float32)
        valid_len = min(len(triplet[0]), self.seq_len)
        return stacked, valid_len

    def _pad_sequence(self, values: Sequence[float]) -> np.ndarray:
        arr = np.zeros(self.seq_len, dtype=np.float32)
        if not values:
            return arr
        clipped = values[: self.seq_len]
        arr[: len(clipped)] = np.array(clipped, dtype=np.float32)
        return arr

    def _build_tabular_features(self, data: Dict[str, List], idx: int) -> np.ndarray:
        duration = float(data["DURATION"][idx] or 0.0)
        bytes_fwd = float(data["BYTES"][idx] or 0.0)
        bytes_rev = float(data["BYTES_REV"][idx] or 0.0)
        packets_fwd = float(data["PACKETS"][idx] or 0.0)
        packets_rev = float(data["PACKETS_REV"][idx] or 0.0)
        ppi_len = float(data["PPI_LEN"][idx] or 0.0)
        ppi_duration = float(data["PPI_DURATION"][idx] or 0.0)
        ppi_roundtrips = float(data["PPI_ROUNDTRIPS"][idx] or 0.0)
        flow_idle = 1.0 if data["FLOW_ENDREASON_IDLE"][idx] else 0.0
        flow_active = 1.0 if data["FLOW_ENDREASON_ACTIVE"][idx] else 0.0
        flow_other = 1.0 if data["FLOW_ENDREASON_OTHER"][idx] else 0.0
        protocol = float(data["PROTOCOL"][idx] or 0.0)
        dst_asn = float(data["DST_ASN"][idx] or 0.0)
        src_port = float(data["SRC_PORT"][idx] or 0.0)
        dst_port = float(data["DST_PORT"][idx] or 0.0)

        features = [
            duration,
            log1p_safe(bytes_fwd),
            log1p_safe(bytes_rev),
            log1p_safe(bytes_fwd + bytes_rev),
            log1p_safe(packets_fwd),
            log1p_safe(packets_rev),
            log1p_safe(packets_fwd + packets_rev),
            ppi_len / max(self.seq_len, 1),
            ppi_duration,
            ppi_roundtrips,
            flow_idle,
            flow_active,
            flow_other,
            protocol,
            log1p_safe(dst_asn),
            src_port / 65535.0,
            dst_port / 65535.0,
        ]

        features.extend(self._parse_histogram(data["PHIST_SRC_SIZES"][idx]))
        features.extend(self._parse_histogram(data["PHIST_DST_SIZES"][idx]))
        features.extend(self._parse_histogram(data["PHIST_SRC_IPT"][idx]))
        features.extend(self._parse_histogram(data["PHIST_DST_IPT"][idx]))
        return np.array(features, dtype=np.float32)

    def _parse_histogram(self, value: Optional[str]) -> List[float]:
        if not value:
            return [0.0] * 8
        parsed = json.loads(value)
        return [log1p_safe(float(x)) for x in parsed[:8]]

    def _apply_seq_norm(self, seq: np.ndarray) -> np.ndarray:
        assert self.normalization is not None
        seq_mean = np.array(self.normalization["seq_mean"], dtype=np.float32)
        seq_std = np.array(self.normalization["seq_std"], dtype=np.float32)
        seq_std = np.where(seq_std == 0.0, 1.0, seq_std)
        return (seq - seq_mean[:, None]) / seq_std[:, None]

    def _apply_tab_norm(self, tab: np.ndarray) -> np.ndarray:
        assert self.normalization is not None
        tab_mean = np.array(self.normalization["tab_mean"], dtype=np.float32)
        tab_std = np.array(self.normalization["tab_std"], dtype=np.float32)
        tab_std = np.where(tab_std == 0.0, 1.0, tab_std)
        return (tab - tab_mean) / tab_std

    def _finalize_seq_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        counts = np.maximum(self._seq_counts, 1.0)
        mean = self._seq_sum / counts
        var = self._seq_sumsq / counts - mean ** 2
        std = np.sqrt(np.clip(var, 1e-6, None))
        return mean.astype(np.float32), std.astype(np.float32)

    def _finalize_tab_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self._tab_sum is not None and self._tab_sumsq is not None
        sample_count = max(self.processed_rows, 1)
        mean = self._tab_sum / sample_count
        var = self._tab_sumsq / sample_count - mean ** 2
        std = np.sqrt(np.clip(var, 1e-6, None))
        return mean.astype(np.float32), std.astype(np.float32)

    def _normalize_sequences_array(self, arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        std = np.where(std == 0.0, 1.0, std)
        mean = mean.reshape(1, -1, 1)
        std = std.reshape(1, -1, 1)
        arr -= mean
        arr /= std
        return arr

    def _normalize_tabular_array(self, arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        std = np.where(std == 0.0, 1.0, std)
        mean = mean.reshape(1, -1)
        std = std.reshape(1, -1)
        arr -= mean
        arr /= std
        return arr


class CesnetQuicDataset(Dataset):
    """Memory-mapped dataset backed by cached numpy arrays."""

    def __init__(self, cache_root: Path, split: str) -> None:
        self.split_dir = cache_root / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Cache directory missing for split {split}: {self.split_dir}")
        with open(self.split_dir / "meta.json", "r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.sequences = np.load(self.split_dir / "sequences.npy", mmap_mode="r")
        self.tabular = np.load(self.split_dir / "tabular.npy", mmap_mode="r")
        self.sni_idx = np.load(self.split_dir / "sni_idx.npy", mmap_mode="r")
        self.ua_idx = np.load(self.split_dir / "ua_idx.npy", mmap_mode="r")
        self.version_idx = np.load(self.split_dir / "version_idx.npy", mmap_mode="r")
        self.labels = np.load(self.split_dir / "labels.npy", mmap_mode="r")
        self.length = int(self.meta["num_samples"])

    def __len__(self) -> int:
        return self.length

    @property
    def num_classes(self) -> int:
        return int(self.meta["num_classes"])

    @property
    def num_versions(self) -> int:
        return len(self.meta["version_values"])

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return {
            "sequences": np.array(self.sequences[index], copy=True),
            "tabular": np.array(self.tabular[index], copy=True),
            "sni_idx": int(self.sni_idx[index]),
            "ua_idx": int(self.ua_idx[index]),
            "version_idx": int(self.version_idx[index]),
            "label": int(self.labels[index]),
        }


def prepare_split_cache(
    split: str,
    files: Sequence[Path],
    args: argparse.Namespace,
    label_map: Dict[str, int],
    version_values: Sequence[int],
    normalization: Optional[Dict[str, List[float]]],
    rebuild: bool,
) -> Dict[str, object]:
    split_dir = Path(args.cache_dir) / split
    meta_path = split_dir / "meta.json"
    if split_dir.exists() and meta_path.exists() and not rebuild:
        with open(meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    builder = SplitCacheBuilder(
        split=split,
        files=files,
        cache_dir=split_dir,
        seq_len=args.max_seq_len,
        label_map=label_map,
        version_values=version_values,
        normalization=normalization,
        sni_hash_size=args.sni_hash_size,
        ua_hash_size=args.ua_hash_size,
        batch_rows=args.cache_batch_rows,
    )
    return builder.build()


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------


class HybridCNN(nn.Module):
    def __init__(
        self,
        seq_len: int,
        tab_dim: int,
        num_classes: int,
        num_versions: int,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.seq_branch = nn.Sequential(
            nn.Conv1d(3, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, args.seq_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(args.seq_hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(args.seq_hidden, args.seq_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(args.seq_hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )

        static_in = tab_dim + args.sni_embed_dim + args.ua_embed_dim + args.version_embed_dim
        self.sni_emb = nn.Embedding(args.sni_hash_size, args.sni_embed_dim, padding_idx=0)
        self.ua_emb = nn.Embedding(args.ua_hash_size, args.ua_embed_dim, padding_idx=0)
        self.version_emb = nn.Embedding(max(num_versions + 1, 1), args.version_embed_dim, padding_idx=0)

        self.static_branch = nn.Sequential(
            nn.Linear(static_in, args.mlp_hidden),
            nn.BatchNorm1d(args.mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.mlp_hidden, args.mlp_hidden),
            nn.BatchNorm1d(args.mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.dropout),
        )

        fusion_in = args.seq_hidden + args.mlp_hidden
        self.head = nn.Sequential(
            nn.Linear(fusion_in, args.mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.mlp_hidden, num_classes),
        )

        self.seq_len = seq_len

    def forward(
        self,
        sequences: torch.Tensor,
        tabular: torch.Tensor,
        sni_idx: torch.Tensor,
        ua_idx: torch.Tensor,
        version_idx: torch.Tensor,
    ) -> torch.Tensor:
        seq_feat = self.seq_branch(sequences)
        sni_feat = self.sni_emb(sni_idx)
        ua_feat = self.ua_emb(ua_idx)
        version_feat = self.version_emb(version_idx)
        static_input = torch.cat([tabular, sni_feat, ua_feat, version_feat], dim=1)
        static_feat = self.static_branch(static_input)
        fused = torch.cat([seq_feat, static_feat], dim=1)
        return self.head(fused)


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------


def create_dataloaders(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    has_val: bool,
) -> Tuple[CesnetQuicDataset, DataLoader, Optional[DataLoader], DistributedSampler, Optional[DistributedSampler]]:
    cache_root = Path(args.cache_dir)
    train_dataset = CesnetQuicDataset(cache_root, "train")
    val_dataset = CesnetQuicDataset(cache_root, "val") if has_val else None

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        if val_dataset is not None
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            sampler=val_sampler,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
        if val_dataset is not None
        else None
    )
    return train_dataset, train_loader, val_loader, train_sampler, val_sampler


def prepare_model(
    args: argparse.Namespace,
    device: torch.device,
    num_classes: int,
    num_versions: int,
    tab_dim: int,
) -> nn.Module:
    model = HybridCNN(
        seq_len=args.max_seq_len,
        tab_dim=tab_dim,
        num_classes=num_classes,
        num_versions=num_versions,
        args=args,
    )
    model.to(device)
    return model


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[amp.GradScaler],
    device: torch.device,
) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint.get("epoch", 0)
    best_acc = checkpoint.get("best_acc", 0.0)
    return start_epoch, best_acc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    sampler: DistributedSampler,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    rank: int,
) -> Dict[str, float]:
    model.train()
    sampler.set_epoch(epoch)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    autocast_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    for step, batch in enumerate(loader):
        sequences = batch["sequences"].to(device, non_blocking=True)
        tabular = batch["tabular"].to(device, non_blocking=True)
        sni_idx = batch["sni_idx"].to(device, non_blocking=True).long()
        ua_idx = batch["ua_idx"].to(device, non_blocking=True).long()
        version_idx = batch["version_idx"].to(device, non_blocking=True).long()
        targets = batch["label"].to(device, non_blocking=True).long()

        with amp.autocast("cuda", dtype=autocast_dtype, enabled=args.use_amp):
            logits = model(sequences, tabular, sni_idx, ua_idx, version_idx)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += preds.eq(targets).sum().item()
        total_samples += batch_size

        if rank == 0 and step % args.log_interval == 0:
            avg_loss = total_loss / max(total_samples, 1)
            avg_acc = total_correct / max(total_samples, 1)
            print(f"Epoch {epoch} Step {step}: loss={avg_loss:.4f} acc={avg_acc:.4f}", flush=True)

    metrics = torch.tensor([total_loss, total_correct, total_samples], device=device)
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    avg_loss = (metrics[0] / metrics[2]).item()
    avg_acc = (metrics[1] / metrics[2]).item()
    return {"loss": avg_loss, "acc": avg_acc}


def gather_predictions(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    preds_list: List[Optional[List[int]]] = [None for _ in range(dist.get_world_size())]
    targets_list: List[Optional[List[int]]] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(preds_list, preds.tolist())
    dist.all_gather_object(targets_list, targets.tolist())
    flat_preds = np.array([item for sublist in preds_list if sublist for item in sublist], dtype=np.int64)
    flat_targets = np.array([item for sublist in targets_list if sublist for item in sublist], dtype=np.int64)
    return flat_preds, flat_targets


def compute_macro_f1(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    eps = 1e-9
    f1_scores = []
    for cls in range(num_classes):
        tp = np.logical_and(preds == cls, targets == cls).sum()
        fp = np.logical_and(preds == cls, targets != cls).sum()
        fn = np.logical_and(preds != cls, targets == cls).sum()
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    sampler: Optional[DistributedSampler],
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    rank: int,
    num_classes: int,
) -> Optional[Dict[str, float]]:
    if loader is None or sampler is None:
        return None
    model.eval()
    sampler.set_epoch(0)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    preds_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    autocast_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    with torch.no_grad():
        for batch in loader:
            sequences = batch["sequences"].to(device, non_blocking=True)
            tabular = batch["tabular"].to(device, non_blocking=True)
            sni_idx = batch["sni_idx"].to(device, non_blocking=True).long()
            ua_idx = batch["ua_idx"].to(device, non_blocking=True).long()
            version_idx = batch["version_idx"].to(device, non_blocking=True).long()
            targets = batch["label"].to(device, non_blocking=True).long()
            with amp.autocast("cuda", dtype=autocast_dtype, enabled=args.use_amp):
                logits = model(sequences, tabular, sni_idx, ua_idx, version_idx)
                loss = criterion(logits, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += preds.eq(targets).sum().item()
            total_samples += batch_size
            preds_all.append(preds.cpu())
            targets_all.append(targets.cpu())

    metrics = torch.tensor([total_loss, total_correct, total_samples], device=device)
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    avg_loss = (metrics[0] / metrics[2]).item()
    avg_acc = (metrics[1] / metrics[2]).item()

    macro_f1 = None
    preds_tensor = torch.cat(preds_all) if preds_all else torch.empty(0, dtype=torch.int64)
    targets_tensor = torch.cat(targets_all) if targets_all else torch.empty(0, dtype=torch.int64)
    if dist.is_initialized():
        gathered_preds, gathered_targets = gather_predictions(preds_tensor, targets_tensor)
        if rank == 0 and gathered_preds.size > 0:
            macro_f1 = compute_macro_f1(gathered_preds, gathered_targets, num_classes)
    else:
        if preds_tensor.numel() > 0:
            macro_f1 = compute_macro_f1(preds_tensor.numpy(), targets_tensor.numpy(), num_classes)

    return {"loss": avg_loss, "acc": avg_acc, "macro_f1": macro_f1}


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    best_acc: float,
    args: argparse.Namespace,
    tag: str,
) -> None:
    if dist.get_rank() != 0:
        return
    state = {
        "epoch": epoch,
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_acc": best_acc,
        "args": vars(args),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint_{tag}.pt"
    torch.save(state, path)
    print(f"Saved checkpoint: {path}", flush=True)


# -----------------------------------------------------------------------------
# Distributed training entrypoint
# -----------------------------------------------------------------------------


def setup_process(rank: int, world_size: int, args: argparse.Namespace) -> None:
    if args.dist_url == "env://":
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(args.dist_backend, init_method=args.dist_url, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_process() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    setup_process(rank, world_size, args)
    set_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True

    dist.barrier()

    cache_root = Path(args.cache_dir)
    train_dataset = CesnetQuicDataset(cache_root, "train")
    val_dataset_present = bool(getattr(args, "has_val_split", False))
    _, train_loader, val_loader, train_sampler, val_sampler = create_dataloaders(
        args=args,
        rank=rank,
        world_size=world_size,
        has_val=val_dataset_present,
    )

    device = torch.device("cuda", rank)
    model = prepare_model(
        args=args,
        device=device,
        num_classes=train_dataset.num_classes,
        num_versions=train_dataset.num_versions,
        tab_dim=train_dataset.tabular.shape[1],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=args.use_amp, init_scale=args.amp_loss_scale)

    if args.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1)),
            ],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    class_weights = None
    train_meta_path = cache_root / "train" / "meta.json"
    with open(train_meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
        if args.use_class_weights:
            counts = np.array(meta["class_counts"], dtype=np.float32)
            counts = np.where(counts == 0, 1.0, counts)
            weights = counts.sum() / (len(counts) * counts)
            class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing).to(device)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        if rank == 0:
            print(f"Resuming from checkpoint {args.resume}", flush=True)
        start_epoch, best_acc = load_checkpoint(args.resume, ddp_model.module, optimizer, scaler, device)

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(ddp_model, train_loader, train_sampler, criterion, optimizer, scaler, device, epoch, args, rank)
        scheduler.step()

        val_metrics = evaluate(ddp_model, val_loader, val_sampler, criterion, device, args, rank, train_dataset.num_classes)
        if rank == 0:
            msg = f"Epoch {epoch} | train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f}"
            if val_metrics:
                msg += f" | val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f}"
                if val_metrics.get("macro_f1") is not None:
                    msg += f" val_macro_f1={val_metrics['macro_f1']:.4f}"
            print(msg, flush=True)

        current_acc = val_metrics["acc"] if val_metrics else train_metrics["acc"]
        if rank == 0 and current_acc > best_acc:
            best_acc = current_acc
            save_checkpoint(Path(args.output_dir), epoch, ddp_model, optimizer, scaler, best_acc, args, tag="best")
        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(Path(args.output_dir), epoch, ddp_model, optimizer, scaler, best_acc, args, tag=f"epoch{epoch+1}")

    cleanup_process()


def launch_training(args: argparse.Namespace) -> None:
    use_external_launcher = "LOCAL_RANK" in os.environ
    rank_env = int(os.environ.get("RANK", "0")) if use_external_launcher else 0
    is_primary = rank_env == 0
    build_caches_if_needed(args, is_primary)

    if use_external_launcher:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        main_worker(local_rank, world_size, args)
    else:
        world_size = args.world_size or torch.cuda.device_count()
        if world_size < 1:
            raise RuntimeError("At least one CUDA device is required")
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    cli_args = parse_args()
    launch_training(cli_args)
