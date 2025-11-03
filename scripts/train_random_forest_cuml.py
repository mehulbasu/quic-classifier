"""Train a Random Forest using RAPIDS cuML on GPU and evaluate on another day."""
from __future__ import annotations

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

SAMPLE_LIMIT: Optional[int] = 1_000_000
USE_CACHE = True
RANDOM_STATE = 42
N_ESTIMATORS = 400
N_BINS = 64


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


def main() -> None:
    X_train_full, y_train_full, encoder = load_cached_training_matrices(
        TRAIN_DAY,
        FEATURE_COLUMNS,
        use_cache=USE_CACHE,
    )
    X_train, y_train = sample_matrices(X_train_full, y_train_full, SAMPLE_LIMIT)

    X_train_gpu = to_cudf_frame(X_train)
    y_train_gpu = to_cudf_series(y_train)

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=16,
        max_features="sqrt",
        min_samples_leaf=10,
        n_streams=8,
        n_bins=N_BINS,
        random_state=RANDOM_STATE,
    )

    start = perf_counter()
    model.fit(X_train_gpu, y_train_gpu)
    train_time = perf_counter() - start

    print("Training samples:", len(X_train))
    print("Unique classes:", len(encoder.classes_))
    print(f"Training time (GPU): {train_time:.1f}s")

    eval_features_full, eval_labels_raw = load_cached_features_with_labels(
        EVAL_DAY,
        FEATURE_COLUMNS,
        use_cache=USE_CACHE,
    )
    overlap_mask = eval_labels_raw.isin(encoder.classes_)
    if not overlap_mask.any():
        raise ValueError("Evaluation dataframe has no overlapping labels with training set.")

    eval_features_full = eval_features_full.loc[overlap_mask]
    eval_labels_raw = eval_labels_raw.loc[overlap_mask]
    eval_features, eval_labels_raw = sample_matrices(
        eval_features_full,
        eval_labels_raw,
        SAMPLE_LIMIT,
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
