#!/usr/bin/env python3
"""Distributed PyTorch trainer with sequential chunk loading for CESNET-QUIC22."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid CNN trainer for CESNET-QUIC22 (chunked)")
    parser.add_argument("--data-dir", type=str, default="datasets/training", help="Directory with parquet files")
    parser.add_argument("--output-dir", type=str, default="artifacts/cnn_ddp", help="Directory for checkpoints")
    parser.add_argument("--cache-dir", type=str, default="datasets/cache_pytorch", help="Cache root for processed chunks")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--val-batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--max-seq-len", type=int, default=30)
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of files for validation if --val-files not set")
    parser.add_argument("--train-files", type=str, nargs="*", default=None, help="Optional subset of parquet files for training")
    parser.add_argument("--val-files", type=str, nargs="*", default=None, help="Optional subset of parquet files for validation")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force regeneration of cached chunks")
    parser.add_argument("--cache-batch-rows", type=int, default=65536, help="Rows per batch when parsing parquet for cache")
    parser.add_argument("--cache-workers", type=int, default=7, help="Parallel workers for cache building (max 10)")
    parser.add_argument("--mlp-hidden", type=int, default=512)
    parser.add_argument("--seq-hidden", type=int, default=256)
    parser.add_argument("--version-embed-dim", type=int, default=16)
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
    parser.add_argument("--sequence-only", action="store_true", help="Zero out static/tabular inputs so the model trains on sequences only")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            print(f"Preparing caches (train files={len(train_files)}, val files={len(val_files)})", flush=True)
            label_map: Dict[str, int]
            version_values: List[int]
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


def parse_sequence_value(seq_value: Optional[str], seq_len: int) -> Tuple[np.ndarray, int]:
    if not seq_value:
        return np.zeros((3, seq_len), dtype=np.float32), 0
    triplet = json.loads(seq_value)
    ipt = _pad_sequence(triplet[0], seq_len)
    ipt = np.log1p(np.maximum(ipt, 0.0))
    direction = _pad_sequence(triplet[1], seq_len)
    sizes = _pad_sequence(triplet[2], seq_len)
    stacked = np.stack([ipt, direction, sizes], axis=0).astype(np.float32)
    valid_len = min(len(triplet[0]), seq_len)
    return stacked, valid_len


def build_tabular_features_value(data: Dict[str, List], idx: int, seq_len: int) -> np.ndarray:
    duration = float(data["DURATION"][idx] or 0.0)
    bytes_fwd = float(data["BYTES"][idx] or 0.0)
    bytes_rev = float(data["BYTES_REV"][idx] or 0.0)
    packets_fwd = float(data["PACKETS"][idx] or 0.0)
    packets_rev = float(data["PACKETS_REV"][idx] or 0.0)
    ppi_len = float(data["PPI_LEN"][idx] or 0.0)
    ppi_duration = float(data["PPI_DURATION"][idx] or 0.0)
    ppi_roundtrips = float(data["PPI_ROUNDTRIPS"][idx] or 0.0)
    protocol = float(data["PROTOCOL"][idx] or 0.0)
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
        ppi_len / max(seq_len, 1),
        ppi_duration,
        ppi_roundtrips,
        protocol,
        src_port / 65535.0,
        dst_port / 65535.0,
    ]

    features.extend(_parse_histogram(data["PHIST_SRC_SIZES"][idx]))
    features.extend(_parse_histogram(data["PHIST_DST_SIZES"][idx]))
    features.extend(_parse_histogram(data["PHIST_SRC_IPT"][idx]))
    features.extend(_parse_histogram(data["PHIST_DST_IPT"][idx]))
    return np.array(features, dtype=np.float32)


def _pad_sequence(values: Sequence[float], seq_len: int) -> np.ndarray:
    arr = np.zeros(seq_len, dtype=np.float32)
    if not values:
        return arr
    clipped = values[:seq_len]
    arr[: len(clipped)] = np.array(clipped, dtype=np.float32)
    return arr


def _parse_histogram(value: Optional[str]) -> List[float]:
    if not value:
        return [0.0] * 8
    parsed = json.loads(value)
    return [log1p_safe(float(x)) for x in parsed[:8]]


class SplitCacheBuilder:
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
        "PROTOCOL",
        "SRC_PORT",
        "DST_PORT",
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
        batch_rows: int,
        num_workers: int,
        rebuild: bool,
    ) -> None:
        self.split = split
        self.files = list(files)
        self.cache_dir = cache_dir
        self.chunk_dir = cache_dir / "chunks"
        self.seq_len = seq_len
        self.label_map = label_map
        self.version_map = {value: idx + 1 for idx, value in enumerate(version_values)}
        self.normalization = normalization
        self.batch_rows = batch_rows
        self.num_workers = max(1, min(num_workers, 18))
        self.rebuild = rebuild

    def build(self) -> Dict[str, object]:
        if self.rebuild and self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for idx, path in enumerate(self.files):
            chunk_name = f"chunk_{idx:05d}_{path.stem.replace(' ', '_')}.pt"
            tasks.append(
                {
                    "parquet_path": str(path),
                    "chunk_path": str(self.chunk_dir / chunk_name),
                    "seq_len": self.seq_len,
                    "label_map": self.label_map,
                    "version_map": self.version_map,
                    "batch_rows": self.batch_rows,
                }
            )

        results: List[Dict[str, object]] = []
        with ProcessPoolExecutor(max_workers=min(len(tasks), self.num_workers)) as executor:
            futures = [executor.submit(_cache_worker_entry, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                chunk_name = Path(result["chunk_path"]).name
                print(
                    f"[{self.split}] Wrote {result['num_rows']:,} samples to {chunk_name}",
                    flush=True,
                )

        results.sort(key=lambda item: item["chunk_path"])
        chunk_entries = []
        total_rows = 0
        tab_dim = None
        class_counts = np.zeros(len(self.label_map), dtype=np.int64)
        seq_sum = np.zeros(3, dtype=np.float64)
        seq_sumsq = np.zeros(3, dtype=np.float64)
        seq_counts = np.zeros(3, dtype=np.float64)
        tab_sum: Optional[np.ndarray] = None
        tab_sumsq: Optional[np.ndarray] = None

        for item in results:
            num_rows = int(item["num_rows"])
            if num_rows == 0:
                chunk_path = Path(item["chunk_path"])
                if chunk_path.exists():
                    chunk_path.unlink()
                continue
            chunk_entries.append({"file": Path(item["chunk_path"]).name, "num_samples": num_rows})
            total_rows += num_rows
            class_counts += np.array(item["class_counts"], dtype=np.int64)
            if tab_dim is None and int(item["tab_dim"]) > 0:
                tab_dim = int(item["tab_dim"])
            if self.normalization is None:
                seq_sum += np.array(item["seq_sum"], dtype=np.float64)
                seq_sumsq += np.array(item["seq_sumsq"], dtype=np.float64)
                seq_counts += np.array(item["seq_counts"], dtype=np.float64)
                tab_array = np.array(item["tab_sum"], dtype=np.float64)
                tab_sq_array = np.array(item["tab_sumsq"], dtype=np.float64)
                tab_sum = tab_array if tab_sum is None else tab_sum + tab_array
                tab_sumsq = tab_sq_array if tab_sumsq is None else tab_sumsq + tab_sq_array

        if not chunk_entries:
            raise ValueError(f"No usable samples found for split {self.split}")

        if tab_dim is None:
            tab_dim = 0

        if self.normalization is None:
            seq_counts = np.maximum(seq_counts, 1.0)
            seq_mean = (seq_sum / seq_counts).astype(np.float32)
            seq_var = np.clip(seq_sumsq / seq_counts - seq_mean ** 2, 1e-6, None)
            seq_std = np.sqrt(seq_var).astype(np.float32)
            sample_count = max(total_rows, 1)
            assert tab_sum is not None and tab_sumsq is not None
            tab_mean = (tab_sum / sample_count).astype(np.float32)
            tab_var = np.clip(tab_sumsq / sample_count - tab_mean ** 2, 1e-6, None)
            tab_std = np.sqrt(tab_var).astype(np.float32)
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
            "num_samples": int(total_rows),
            "seq_len": self.seq_len,
            "tab_dim": int(tab_dim),
            "num_classes": len(self.label_map),
            "label_to_index": self.label_map,
            "index_to_label": sorted(self.label_map, key=self.label_map.get),
            "version_values": list(self.version_map.keys()),
            "class_counts": class_counts.tolist(),
            "normalization": normalization,
            "chunks": chunk_entries,
            "schema_version": 2,
        }

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_dir / "meta.json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        return meta


def _cache_worker_entry(config: Dict[str, object]) -> Dict[str, object]:
    parquet_path = Path(config["parquet_path"])
    chunk_path = Path(config["chunk_path"])
    seq_len = int(config["seq_len"])
    label_map: Dict[str, int] = config["label_map"]
    version_map: Dict[int, int] = config["version_map"]
    batch_rows = int(config["batch_rows"])

    parquet_file = pq.ParquetFile(str(parquet_path))
    total_rows = parquet_file.metadata.num_rows
    total_rows = max(int(total_rows), 1)
    sequences = np.zeros((total_rows, 3, seq_len), dtype=np.float32)
    tabular: Optional[np.ndarray] = None
    version_idx = np.zeros(total_rows, dtype=np.int32)
    labels = np.zeros(total_rows, dtype=np.int64)

    seq_sum = np.zeros(3, dtype=np.float64)
    seq_sumsq = np.zeros(3, dtype=np.float64)
    seq_counts = np.zeros(3, dtype=np.float64)
    tab_sum: Optional[np.ndarray] = None
    tab_sumsq: Optional[np.ndarray] = None
    class_counts = np.zeros(len(label_map), dtype=np.int64)
    tab_dim = 0

    row_ptr = 0
    for batch in parquet_file.iter_batches(columns=SplitCacheBuilder.columns, batch_size=batch_rows):
        data = batch.to_pydict()
        batch_rows_local = len(data["APP"])
        for local_idx in range(batch_rows_local):
            label_name = data["APP"][local_idx]
            if label_name is None or label_name not in label_map:
                continue

            seq_tensor, valid_len = parse_sequence_value(data["PPI"][local_idx], seq_len)
            if valid_len > 0:
                seq_sum += seq_tensor[:, :valid_len].sum(axis=1)
                seq_sumsq += (seq_tensor[:, :valid_len] ** 2).sum(axis=1)
                seq_counts += valid_len
            sequences[row_ptr] = seq_tensor

            tab_vec = build_tabular_features_value(data, local_idx, seq_len)
            if tabular is None:
                tab_dim = len(tab_vec)
                tabular = np.zeros((total_rows, tab_dim), dtype=np.float32)
                tab_sum = np.zeros(tab_dim, dtype=np.float64)
                tab_sumsq = np.zeros(tab_dim, dtype=np.float64)
            assert tabular is not None and tab_sum is not None and tab_sumsq is not None
            tabular[row_ptr] = tab_vec
            tab_sum += tab_vec
            tab_sumsq += tab_vec ** 2

            version_value = data["QUIC_VERSION"][local_idx]
            version_idx[row_ptr] = version_map.get(int(version_value), 0) if version_value is not None else 0

            label_id = label_map[label_name]
            labels[row_ptr] = label_id
            class_counts[label_id] += 1
            row_ptr += 1

    sequences = np.ascontiguousarray(sequences[:row_ptr])
    if tabular is None:
        tab_dim = 0
        tabular = np.zeros((row_ptr, 1), dtype=np.float32)
        tab_sum = np.zeros(1, dtype=np.float64)
        tab_sumsq = np.zeros(1, dtype=np.float64)
    else:
        tabular = np.ascontiguousarray(tabular[:row_ptr])
    version_idx = np.ascontiguousarray(version_idx[:row_ptr])
    labels = np.ascontiguousarray(labels[:row_ptr])

    if row_ptr > 0:
        payload = {
            "sequences": torch.from_numpy(sequences.copy()),
            "tabular": torch.from_numpy(tabular.copy()),
            "version_idx": torch.from_numpy(version_idx.copy()),
            "labels": torch.from_numpy(labels.copy()),
        }
        torch.save(payload, chunk_path)
    else:
        if chunk_path.exists():
            chunk_path.unlink()

    return {
        "chunk_path": str(chunk_path),
        "num_rows": int(row_ptr),
        "tab_dim": int(tab_dim),
        "seq_sum": seq_sum.tolist(),
        "seq_sumsq": seq_sumsq.tolist(),
        "seq_counts": seq_counts.tolist(),
        "tab_sum": (tab_sum.tolist() if tab_sum is not None else [0.0]),
        "tab_sumsq": (tab_sumsq.tolist() if tab_sumsq is not None else [0.0]),
        "class_counts": class_counts.tolist(),
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
        batch_rows=args.cache_batch_rows,
        num_workers=args.cache_workers,
        rebuild=rebuild,
    )
    return builder.build()


class ChunkManifest:
    def __init__(self, cache_root: Path, split: str) -> None:
        self.split = split
        self.split_dir = cache_root / split
        meta_path = self.split_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing cache metadata for split {split}: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.chunk_dir = self.split_dir / "chunks"
        if not self.chunk_dir.exists():
            raise FileNotFoundError(f"Missing chunk directory for split {split}: {self.chunk_dir}")
        self.chunks: List[Dict[str, object]] = list(self.meta.get("chunks", []))
        if not self.chunks:
            raise ValueError(f"No chunks registered for split {split}")

    @property
    def normalization(self) -> Dict[str, List[float]]:
        norm = self.meta.get("normalization")
        if norm is None:
            raise ValueError(f"Normalization stats missing for split {self.split}")
        return norm

    @property
    def seq_len(self) -> int:
        return int(self.meta["seq_len"])

    @property
    def tab_dim(self) -> int:
        return int(self.meta["tab_dim"])

    @property
    def num_classes(self) -> int:
        return int(self.meta["num_classes"])

    @property
    def num_versions(self) -> int:
        return len(self.meta.get("version_values", []))

    @property
    def class_counts(self) -> Sequence[int]:
        return self.meta.get("class_counts", [])

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    def chunk_path(self, index: int) -> Path:
        entry = self.chunks[index]
        chunk_file = entry.get("file")
        if not chunk_file:
            raise ValueError(f"Chunk entry at index {index} missing file field")
        path = self.chunk_dir / str(chunk_file)
        if not path.exists():
            raise FileNotFoundError(f"Expected chunk file missing: {path}")
        return path

    def chunk_size(self, index: int) -> int:
        entry = self.chunks[index]
        return int(entry.get("num_samples", 0))


class SingleChunkDataset(Dataset):
    def __init__(self, chunk_path: Path, normalization: Dict[str, List[float]]) -> None:
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk tensor missing: {chunk_path}")
        payload = torch.load(chunk_path, map_location="cpu")
        self.sequences: torch.Tensor = payload["sequences"].float()
        self.tabular: torch.Tensor = payload["tabular"].float()
        self.version_idx: torch.Tensor = payload["version_idx"].long()
        self.labels: torch.Tensor = payload["labels"].long()

        seq_mean = torch.tensor(normalization["seq_mean"], dtype=torch.float32).view(1, -1, 1)
        seq_std = torch.tensor(normalization["seq_std"], dtype=torch.float32).clamp_min_(1e-6).view(1, -1, 1)
        seq_mean = seq_mean.to(self.sequences.device)
        seq_std = seq_std.to(self.sequences.device)
        self.sequences.sub_(seq_mean).div_(seq_std)

        tab_mean = torch.tensor(normalization["tab_mean"], dtype=torch.float32).view(1, -1)
        tab_std = torch.tensor(normalization["tab_std"], dtype=torch.float32).clamp_min_(1e-6).view(1, -1)
        tab_mean = tab_mean.to(self.tabular.device)
        tab_std = tab_std.to(self.tabular.device)
        self.tabular.sub_(tab_mean).div_(tab_std)

        self.length = self.labels.shape[0]
        if self.length == 0:
            raise ValueError(f"Chunk {chunk_path} has no usable samples")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "sequences": self.sequences[index],
            "tabular": self.tabular[index],
            "version_idx": self.version_idx[index],
            "label": self.labels[index],
        }


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

        static_in = tab_dim + args.version_embed_dim

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
        version_idx: torch.Tensor,
    ) -> torch.Tensor:
        seq_feat = self.seq_branch(sequences)
        version_feat = self.version_emb(version_idx)
        static_input = torch.cat([tabular, version_feat], dim=1)
        static_feat = self.static_branch(static_input)
        fused = torch.cat([seq_feat, static_feat], dim=1)
        return self.head(fused)


def broadcast_chunk_order(num_chunks: int, epoch: int, shuffle: bool, seed: int, rank: int) -> List[int]:
    if num_chunks == 0:
        return []
    order: Optional[List[int]] = None
    if not dist.is_initialized() or dist.get_world_size() == 1:
        order = list(range(num_chunks))
        if shuffle:
            rng = random.Random(seed + epoch)
            rng.shuffle(order)
        return order

    if rank == 0:
        order = list(range(num_chunks))
        if shuffle:
            rng = random.Random(seed + epoch)
            rng.shuffle(order)
    obj_list: List[Optional[List[int]]] = [order]
    dist.broadcast_object_list(obj_list, src=0)
    received = obj_list[0]
    if received is None:
        raise RuntimeError("Failed to broadcast chunk order")
    return received


def ddp_barrier(rank: int) -> None:
    if dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def release_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_chunk_loader(
    manifest: ChunkManifest,
    chunk_idx: int,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    is_train: bool,
    epoch: int,
) -> Tuple[SingleChunkDataset, DataLoader, DistributedSampler]:
    chunk_path = manifest.chunk_path(chunk_idx)
    dataset = SingleChunkDataset(chunk_path, manifest.normalization)
    batch_size = args.batch_size if is_train else args.val_batch_size
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=is_train, drop_last=False)
    sampler.set_epoch(epoch * 10 + chunk_idx if is_train else 0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader, sampler


def train_single_chunk(
    model: DDP,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    rank: int,
    epoch: int,
    chunk_idx: int,
) -> Tuple[float, int, int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    autocast_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    for step, batch in enumerate(loader):
        sequences = batch["sequences"].to(device, non_blocking=True)
        tabular = batch["tabular"].to(device, non_blocking=True)
        version_idx = batch["version_idx"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)

        if args.sequence_only:
            tabular = torch.zeros_like(tabular)
            version_idx = torch.zeros_like(version_idx)

        with amp.autocast("cuda", dtype=autocast_dtype, enabled=args.use_amp):
            logits = model(sequences, tabular, version_idx)
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
            print(
                f"Epoch {epoch} Chunk {chunk_idx} Step {step}: loss={avg_loss:.4f} acc={avg_acc:.4f}",
                flush=True,
            )

    return total_loss, total_correct, total_samples


def train_epoch(
    model: DDP,
    manifest: ChunkManifest,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    epoch: int,
) -> Dict[str, float]:
    chunk_order = broadcast_chunk_order(manifest.num_chunks, epoch, shuffle=True, seed=args.seed, rank=rank)
    loss_sum = 0.0
    correct_sum = 0
    sample_sum = 0
    for chunk_idx in chunk_order:
        ddp_barrier(rank)
        dataset, loader, _ = create_chunk_loader(manifest, chunk_idx, args, rank, world_size, True, epoch)
        chunk_loss, chunk_correct, chunk_samples = train_single_chunk(
            model,
            loader,
            criterion,
            optimizer,
            scaler,
            device,
            args,
            rank,
            epoch,
            chunk_idx,
        )
        loss_sum += chunk_loss
        correct_sum += chunk_correct
        sample_sum += chunk_samples
        del dataset, loader
        gc.collect()
        release_cuda_cache()
        ddp_barrier(rank)

    metrics = torch.tensor([loss_sum, correct_sum, sample_sum], device=device)
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    if metrics[2].item() == 0:
        raise ValueError("No training samples were processed; check chunk cache contents")
    avg_loss = (metrics[0] / metrics[2]).item()
    avg_acc = (metrics[1] / metrics[2]).item()
    return {"loss": avg_loss, "acc": avg_acc}


def evaluate_single_chunk(
    model: DDP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[float, int, int, List[torch.Tensor], List[torch.Tensor]]:
    model.eval()
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
            version_idx = batch["version_idx"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            if args.sequence_only:
                tabular = torch.zeros_like(tabular)
                version_idx = torch.zeros_like(version_idx)
            with amp.autocast("cuda", dtype=autocast_dtype, enabled=args.use_amp):
                logits = model(sequences, tabular, version_idx)
                loss = criterion(logits, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += preds.eq(targets).sum().item()
            total_samples += batch_size
            preds_all.append(preds.cpu())
            targets_all.append(targets.cpu())

    return total_loss, total_correct, total_samples, preds_all, targets_all


def gather_predictions(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    world_size = dist.get_world_size()
    preds_list: List[Optional[List[int]]] = [None for _ in range(world_size)]
    targets_list: List[Optional[List[int]]] = [None for _ in range(world_size)]
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


def evaluate_split(
    model: DDP,
    manifest: Optional[ChunkManifest],
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> Optional[Dict[str, float]]:
    if manifest is None:
        return None
    chunk_order = broadcast_chunk_order(manifest.num_chunks, 0, shuffle=False, seed=0, rank=rank)
    loss_sum = 0.0
    correct_sum = 0
    sample_sum = 0
    preds_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []

    for chunk_idx in chunk_order:
        ddp_barrier(rank)
        dataset, loader, _ = create_chunk_loader(manifest, chunk_idx, args, rank, world_size, False, epoch=0)
        chunk_loss, chunk_correct, chunk_samples, preds_chunk, targets_chunk = evaluate_single_chunk(
            model, loader, criterion, device, args
        )
        loss_sum += chunk_loss
        correct_sum += chunk_correct
        sample_sum += chunk_samples
        preds_all.extend(preds_chunk)
        targets_all.extend(targets_chunk)
        del dataset, loader
        gc.collect()
        release_cuda_cache()
        ddp_barrier(rank)

    metrics = torch.tensor([loss_sum, correct_sum, sample_sum], device=device)
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    if metrics[2].item() == 0:
        return None
    avg_loss = (metrics[0] / metrics[2]).item()
    avg_acc = (metrics[1] / metrics[2]).item()

    macro_f1 = None
    preds_tensor = torch.cat(preds_all) if preds_all else torch.empty(0, dtype=torch.int64)
    targets_tensor = torch.cat(targets_all) if targets_all else torch.empty(0, dtype=torch.int64)
    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered_preds, gathered_targets = gather_predictions(preds_tensor, targets_tensor)
        if rank == 0 and gathered_preds.size > 0:
            macro_f1 = compute_macro_f1(gathered_preds, gathered_targets, manifest.num_classes)
    else:
        if preds_tensor.numel() > 0:
            macro_f1 = compute_macro_f1(preds_tensor.numpy(), targets_tensor.numpy(), manifest.num_classes)

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
    if dist.is_initialized() and dist.get_rank() != 0:
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
    device = torch.device("cuda", rank)
    ddp_barrier(rank)

    cache_root = Path(args.cache_dir)
    train_manifest = ChunkManifest(cache_root, "train")
    val_manifest = ChunkManifest(cache_root, "val") if getattr(args, "has_val_split", False) else None

    model = HybridCNN(
        seq_len=train_manifest.seq_len,
        tab_dim=train_manifest.tab_dim,
        num_classes=train_manifest.num_classes,
        num_versions=train_manifest.num_versions,
        args=args,
    ).to(device)
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
    if args.use_class_weights:
        counts = np.array(train_manifest.class_counts, dtype=np.float32)
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
        checkpoint = torch.load(args.resume, map_location=device)
        ddp_model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint.get("optimizer", optimizer.state_dict()))
        scaler.load_state_dict(checkpoint.get("scaler", scaler.state_dict()))
        start_epoch = checkpoint.get("epoch", 0)
        best_acc = checkpoint.get("best_acc", 0.0)

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(ddp_model, train_manifest, criterion, optimizer, scaler, device, args, rank, world_size, epoch)
        scheduler.step()
        val_metrics = evaluate_split(ddp_model, val_manifest, criterion, device, args, rank, world_size)
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

    ddp_barrier(rank)
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


