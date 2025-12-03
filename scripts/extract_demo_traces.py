#!/usr/bin/env python3
"""Extract hand-picked CESNET-QUIC22 samples for the demo replay client."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

TARGET_LABELS: List[str] = [
    "youtube",
    "tiktok",
    "spotify",
    "instagram",
    "microsoft-outlook",
    "gmail",
    "google-www",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract cached traces for demo replay")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("datasets/cache_pytorch"),
        help="Root directory that contains split caches",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Cache split to sample from (default: val)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo/demo_traces.json"),
        help="Destination JSON file",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=15,
        help="Number of samples to keep per target class (must be between 10 and 20)",
    )
    return parser.parse_args()


def load_metadata(cache_root: Path) -> Dict[str, object]:
    meta_path = cache_root / "train" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing training metadata: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_traces(
    chunks_dir: Path,
    label_lookup: List[str],
    target_labels: List[str],
    samples_per_label: int,
) -> List[Dict[str, object]]:
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunks_dir}")
    collected: Dict[str, List[Dict[str, object]]] = {label: [] for label in target_labels}
    target_set = set(target_labels)
    chunk_paths = sorted(chunks_dir.glob("*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk tensors found under {chunks_dir}")

    for chunk_path in chunk_paths:
        payload = torch.load(chunk_path, map_location="cpu")
        sequences: torch.Tensor = payload["sequences"]
        tabular: torch.Tensor = payload["tabular"]
        sni_idx: torch.Tensor = payload["sni_idx"]
        ua_idx: torch.Tensor = payload["ua_idx"]
        version_idx: torch.Tensor = payload["version_idx"]
        labels: torch.Tensor = payload["labels"]

        for idx in range(labels.shape[0]):
            label_id = int(labels[idx].item())
            if label_id >= len(label_lookup):
                continue
            label_name = label_lookup[label_id]
            if label_name not in target_set:
                continue
            bucket = collected[label_name]
            if len(bucket) >= samples_per_label:
                continue
            sample = {
                "ground_truth": label_name,
                "sequences": sequences[idx].tolist(),
                "tabular": tabular[idx].tolist(),
                "sni_idx": int(sni_idx[idx].item()),
                "ua_idx": int(ua_idx[idx].item()),
                "version_idx": int(version_idx[idx].item()),
            }
            bucket.append(sample)
        if all(len(bucket) >= samples_per_label for bucket in collected.values()):
            break

    missing = [label for label, rows in collected.items() if len(rows) < samples_per_label]
    if missing:
        raise RuntimeError(
            "Not enough samples collected for: "
            + ", ".join(f"{label} ({len(collected[label])})" for label in missing)
        )

    merged: List[Dict[str, object]] = []
    for label in target_labels:
        merged.extend(collected[label])
    return merged


def save_traces(records: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def main() -> None:
    args = parse_args()
    if not 10 <= args.per_class <= 20:
        raise ValueError("--per-class must be between 10 and 20 to satisfy the demo requirements")

    cache_root = args.cache_root
    split_dir = cache_root / args.split
    chunks_dir = split_dir / "chunks"

    meta = load_metadata(cache_root)
    label_lookup = list(meta.get("index_to_label", []))
    if not label_lookup:
        raise ValueError("index_to_label missing in metadata")

    traces = collect_traces(chunks_dir, label_lookup, TARGET_LABELS, args.per_class)
    save_traces(traces, args.output)
    print(f"Wrote {len(traces)} demo traces to {args.output}")


if __name__ == "__main__":
    main()
