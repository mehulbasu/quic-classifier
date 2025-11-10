"""Train a Random Forest using RAPIDS cuML on GPU and evaluate on another day."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

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
    load_cached_features_with_labels,
    load_cached_training_matrices,
)

TRAIN_DAY = Path("datasets/cesnet-quic22/W-2022-47/1_Mon/flows-20221121.parquet")
EVAL_DAY = Path("datasets/cesnet-quic22/W-2022-47/3_Wed/flows-20221123.parquet")

DEFAULT_SAMPLE_LIMIT: Optional[int] = 1_000_000
DEFAULT_USE_CACHE = True
RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 400
DEFAULT_MAX_DEPTH = 16
DEFAULT_MIN_SAMPLES_LEAF = 10
DEFAULT_N_BINS = 64
DEFAULT_N_STREAMS = 8


def sample_matrices(
    X: pd.DataFrame, y: pd.Series, limit: Optional[int]
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
        "--sample-limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT if DEFAULT_SAMPLE_LIMIT is not None else -1,
        help="Maximum rows sampled from each day (-1 to use all rows).",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Recompute features instead of using cached matrices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_limit: Optional[int] = (
        None if args.sample_limit is not None and args.sample_limit < 0 else args.sample_limit
    )
    use_cache = not args.disable_cache and DEFAULT_USE_CACHE

    X_train_full, y_train_full, encoder = load_cached_training_matrices(
        TRAIN_DAY,
        FEATURE_COLUMNS,
        use_cache=use_cache,
    )
    X_train, y_train = sample_matrices(X_train_full, y_train_full, sample_limit)

    X_train_gpu = to_cudf_frame(X_train)
    y_train_gpu = to_cudf_series(y_train)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features="sqrt",
        min_samples_leaf=args.min_samples_leaf,
        n_streams=args.n_streams,
        n_bins=args.n_bins,
        random_state=RANDOM_STATE,
    )

    print("Training samples:", len(X_train))
    print("Unique classes:", len(encoder.classes_))
    print(
        f"Parameters: n_estimators={args.n_estimators} max_depth={args.max_depth} "
        f"min_samples_leaf={args.min_samples_leaf} n_bins={args.n_bins} "
        f"n_streams={args.n_streams}"
    )

    start = perf_counter()
    model.fit(X_train_gpu, y_train_gpu)
    train_time = perf_counter() - start

    print(f"Training time (GPU): {train_time:.1f}s")

    eval_features_full, eval_labels_raw = load_cached_features_with_labels(
        EVAL_DAY,
        FEATURE_COLUMNS,
        use_cache=use_cache,
    )
    overlap_mask = eval_labels_raw.isin(encoder.classes_)
    if not overlap_mask.any():
        raise ValueError("Evaluation dataframe has no overlapping labels with training set.")

    eval_features_full = eval_features_full.loc[overlap_mask]
    eval_labels_raw = eval_labels_raw.loc[overlap_mask]
    eval_features, eval_labels_raw = sample_matrices(
        eval_features_full,
        eval_labels_raw,
        sample_limit,
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

    print("Evaluation samples:", len(eval_features))
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
