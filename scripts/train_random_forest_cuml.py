"""Train a week-long Random Forest using RAPIDS cuML and evaluate on another week."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

try:
    import cudf
    from cuml.ensemble import RandomForestClassifier
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "RAPIDS cuML is not installed. Install RAPIDS (cuml, cudf) for GPU training."
    ) from exc

from scripts.load_day import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_cached_week_features_with_labels,
)

DEFAULT_DATASET_ROOT = Path("datasets/cesnet-quic22")
DEFAULT_TRAIN_WEEK = "W-2022-47"
DEFAULT_EVAL_WEEK = "W-2022-46"
DEFAULT_EVAL_SAMPLE_LIMIT: Optional[int] = 1_000_000
DEFAULT_USE_CACHE = True
RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 16
DEFAULT_MIN_SAMPLES_LEAF = 5
DEFAULT_N_BINS = 64
DEFAULT_N_STREAMS = 8
DEFAULT_MAX_BATCH_SIZE = 50


def maybe_sample_aligned(
    X: pd.DataFrame,
    y: pd.Series,
    limit: Optional[int],
) -> Tuple[pd.DataFrame, pd.Series]:
    if limit is None or len(X) <= limit:
        return X, y
    sampled_idx = X.sample(n=limit, random_state=RANDOM_STATE).index
    return X.loc[sampled_idx], y.loc[sampled_idx]


def to_cudf_frame(X: pd.DataFrame) -> cudf.DataFrame:
    # cuML training expects float32 features for best performance.
    X_float32 = X.astype(np.float32, copy=False)
    return cudf.from_pandas(X_float32)  # type: ignore[return-value]


def to_cudf_series(y: pd.Series) -> cudf.Series:
    # cuDF prefers 0-based contiguous integers
    return cudf.Series(y.values.astype(np.int32))


def list_week_files(dataset_root: Path, week: str) -> List[Path]:
    week_dir = dataset_root / week
    if not week_dir.exists():
        raise FileNotFoundError(f"Week directory does not exist: {week_dir}")

    day_files: List[Path] = []
    for day_dir in sorted(p for p in week_dir.iterdir() if p.is_dir()):
        parquet_files = sorted(day_dir.glob("flows-*.parquet"))
        if parquet_files:
            day_files.append(parquet_files[0])
            continue
        csv_files = sorted(day_dir.glob("flows-*.csv.gz"))
        if csv_files:
            day_files.append(csv_files[0])
            continue
        raise FileNotFoundError(f"No CSV or Parquet file found in {day_dir}")

    if not day_files:
        raise FileNotFoundError(f"No daily files discovered under {week_dir}")
    return day_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a cuML Random Forest on QUIC flow features."
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Number of trees in the forest (default: %(default)s).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Maximum tree depth (default: %(default)s).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=DEFAULT_MIN_SAMPLES_LEAF,
        help="Minimum samples per leaf (default: %(default)s).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=DEFAULT_N_BINS,
        help="Histogram bin count for split finding (default: %(default)s).",
    )
    parser.add_argument(
        "--n-streams",
        type=int,
        default=DEFAULT_N_STREAMS,
        help="Number of CUDA streams used during training (default: %(default)s).",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help="Maximum tree nodes processed per batch (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing weekly subfolders (default: %(default)s).",
    )
    parser.add_argument(
        "--train-week",
        type=str,
        default=DEFAULT_TRAIN_WEEK,
        help="Week folder name for training data (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-week",
        type=str,
        default=DEFAULT_EVAL_WEEK,
        help="Week folder name for evaluation data (default: %(default)s).",
    )
    parser.add_argument(
        "--train-sample-limit",
        type=int,
        default=-1,
        help="Optional cap on training rows (-1 to use full week).",
    )
    parser.add_argument(
        "--eval-sample-limit",
        type=int,
        default=DEFAULT_EVAL_SAMPLE_LIMIT if DEFAULT_EVAL_SAMPLE_LIMIT is not None else -1,
        help="Rows sampled for evaluation (-1 to use all rows).",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Recompute features instead of using cached matrices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cache = not args.disable_cache and DEFAULT_USE_CACHE

    dataset_root = args.dataset_root
    train_week = args.train_week
    eval_week = args.eval_week

    train_paths = list_week_files(dataset_root, train_week)
    eval_paths = list_week_files(dataset_root, eval_week)

    train_sample_limit: Optional[int] = None if args.train_sample_limit < 0 else args.train_sample_limit
    eval_sample_limit: Optional[int] = (
        None if args.eval_sample_limit is not None and args.eval_sample_limit < 0 else args.eval_sample_limit
    )

    # Step 1: load/cached features for the entire training week.
    X_train_full, y_train_labels = load_cached_week_features_with_labels(
        train_paths,
        FEATURE_COLUMNS,
        cache_prefix=f"train_{train_week}",
        use_cache=use_cache,
    )
    X_train_full = X_train_full.astype(np.float32, copy=False)
    y_train_labels = y_train_labels.astype(str)

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_labels)
    y_train_series = pd.Series(
        np.asarray(y_train_encoded, dtype=np.int32),
        name=TARGET_COLUMN,
    )

    total_training_rows = len(X_train_full)

    X_train, y_train = maybe_sample_aligned(
        X_train_full,
        y_train_series,
        train_sample_limit,
    )

    X_train_gpu = to_cudf_frame(X_train)
    y_train_gpu = to_cudf_series(y_train)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features="sqrt",
        min_samples_leaf=args.min_samples_leaf,
        n_streams=args.n_streams,
        n_bins=args.n_bins,
        max_batch_size=args.max_batch_size,
        random_state=RANDOM_STATE,
    )

    print("Training week:", train_week)
    print("Training files:")
    for path in train_paths:
        print(" -", path)
    print("Training samples:", len(X_train))
    if train_sample_limit is not None:
        print("Total training rows before sampling:", total_training_rows)
    print("Unique classes:", len(encoder.classes_))
    print(
        f"Parameters: n_estimators={args.n_estimators} max_depth={args.max_depth} "
        f"min_samples_leaf={args.min_samples_leaf} n_bins={args.n_bins} "
        f"n_streams={args.n_streams} max_batch_size={args.max_batch_size}"
    )

    class_counts = y_train_labels.value_counts().head(10)
    print("Top 10 classes by training support:")
    for label, count in class_counts.items():
        print(f" {label:25s} {count:>10d}")

    start = perf_counter()
    model.fit(X_train_gpu, y_train_gpu)
    train_time = perf_counter() - start

    print(f"Training time (GPU): {train_time:.1f}s")

    eval_features_full, eval_labels_raw = load_cached_week_features_with_labels(
        eval_paths,
        FEATURE_COLUMNS,
        cache_prefix=f"eval_{eval_week}",
        use_cache=use_cache,
    )
    eval_features_full = eval_features_full.astype(np.float32, copy=False)
    eval_labels_raw = eval_labels_raw.astype(str)
    overlap_mask = eval_labels_raw.isin(encoder.classes_)
    if not overlap_mask.any():
        raise ValueError("Evaluation dataframe has no overlapping labels with training set.")

    eval_features_full = eval_features_full.loc[overlap_mask]
    eval_labels_raw = eval_labels_raw.loc[overlap_mask]
    total_eval_rows = len(eval_features_full)

    eval_features, eval_labels_raw = maybe_sample_aligned(
        eval_features_full,
        eval_labels_raw,
        eval_sample_limit,
    )
    encoded_eval = np.asarray(encoder.transform(eval_labels_raw), dtype=np.int32)
    eval_labels = pd.Series(encoded_eval.tolist(), index=eval_labels_raw.index, name=TARGET_COLUMN)

    eval_features_gpu = to_cudf_frame(eval_features)

    y_pred_gpu = model.predict(eval_features_gpu)
    if hasattr(y_pred_gpu, "to_pandas"):
        y_pred = y_pred_gpu.to_pandas().astype(np.int32)
    else:
        y_pred = pd.Series(np.asarray(y_pred_gpu, dtype=np.int32))

    accuracy = accuracy_score(eval_labels, y_pred)
    macro_f1 = f1_score(eval_labels, y_pred, average="macro")

    print("Evaluation week:", eval_week)
    print("Evaluation files:")
    for path in eval_paths:
        print(" -", path)
    print("Evaluation samples:", len(eval_features))
    if eval_sample_limit is not None:
        print("Total evaluation rows before sampling:", total_eval_rows)
    print("Accuracy:", round(float(accuracy), 4))
    print("Macro F1:", round(float(macro_f1), 4))
    print("Classification report (top 10 classes by support):")
    unique_labels = np.unique(eval_labels)
    report_dict = cast(
        Dict[str, Dict[str, float]],
        classification_report(
            eval_labels,
            y_pred,
            labels=unique_labels,
            target_names=encoder.classes_[unique_labels],
            output_dict=True,
            zero_division=0,
        ),
    )
    per_class = [
        (label, metrics)
        for label, metrics in report_dict.items()
        if label not in {"accuracy", "macro avg", "weighted avg"}
    ]
    per_class.sort(key=lambda item: item[1]["support"], reverse=True)
    for label, metrics in per_class[:10]:
        print(
            f"{label:25s} prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} "
            f"f1={metrics['f1-score']:.3f} support={int(metrics['support'])}"
        )


if __name__ == "__main__":
    main()
