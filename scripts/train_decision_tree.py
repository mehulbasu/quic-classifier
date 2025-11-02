"""Train a baseline Decision Tree on one day's flows and evaluate on another."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier

from scripts.load_day import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_cached_features_with_labels,
    load_cached_training_matrices,
)

TRAIN_DAY = Path("datasets/cesnet-quic22/W-2022-47/1_Mon/flows-20221121.csv.gz")
EVAL_DAY = Path("datasets/cesnet-quic22/W-2022-47/3_Wed/flows-20221123.csv.gz")

SAMPLE_LIMIT: Optional[int] = 1_000_000
USE_CACHE = True
RANDOM_STATE = 42

def sample_matrices(
    X: pd.DataFrame, y: pd.Series, limit: Optional[int]
) -> Tuple[pd.DataFrame, pd.Series]:
    if limit is None or len(X) <= limit:
        return X, y
    sampled_idx = X.sample(n=limit, random_state=RANDOM_STATE).index
    return X.loc[sampled_idx], y.loc[sampled_idx]


def main() -> None:
    X_train_full, y_train_full, encoder = load_cached_training_matrices(
        TRAIN_DAY,
        FEATURE_COLUMNS,
        use_cache=USE_CACHE,
    )
    X_train, y_train = sample_matrices(X_train_full, y_train_full, SAMPLE_LIMIT)

    model = DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=200,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    print("Training samples:", len(X_train))
    print("Unique classes:", len(encoder.classes_))

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
        eval_features_full, eval_labels_raw, SAMPLE_LIMIT
    )
    encoded_eval = np.asarray(encoder.transform(eval_labels_raw), dtype=np.int32)
    eval_labels = pd.Series(encoded_eval.tolist(), index=eval_labels_raw.index, name=TARGET_COLUMN)

    y_pred = model.predict(eval_features)

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
