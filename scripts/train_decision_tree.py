"""Train a baseline Decision Tree on one day's flows and evaluate on another."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier

from scripts.load_day import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_day,
    prepare_matrices,
)

TRAIN_DAY = Path("datasets/cesnet-quic22/W-2022-47/1_Mon/flows-20221121.csv.gz")
EVAL_DAY = Path("datasets/cesnet-quic22/W-2022-47/3_Wed/flows-20221123.csv.gz")

SAMPLE_LIMIT: Optional[int] = 1_000_000
RANDOM_STATE = 42

@dataclass
class DatasetBundle:
    features: pd.DataFrame
    labels: pd.Series


def sample_rows(df: pd.DataFrame, limit: Optional[int]) -> pd.DataFrame:
    if limit is None or len(df) <= limit:
        return df
    return df.sample(n=limit, random_state=RANDOM_STATE)


def encode_eval_labels(
    df: pd.DataFrame,
    encoder,
) -> DatasetBundle:
    mask = df[TARGET_COLUMN].isin(encoder.classes_)
    filtered = df.loc[mask].copy()
    if filtered.empty:
        raise ValueError("Evaluation dataframe has no overlapping labels with training set.")
    encoded = pd.Series(encoder.transform(filtered[TARGET_COLUMN].astype(str)), index=filtered.index)
    X = filtered[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return DatasetBundle(features=X, labels=encoded)


def main() -> None:
    train_df = load_day(TRAIN_DAY, FEATURE_COLUMNS)
    train_df = sample_rows(train_df, SAMPLE_LIMIT)
    X_train, y_train, encoder = prepare_matrices(train_df, FEATURE_COLUMNS)

    model = DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=200,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    print("Training samples:", len(X_train))
    print("Unique classes:", len(encoder.classes_))

    eval_df = load_day(EVAL_DAY, FEATURE_COLUMNS)
    eval_df = sample_rows(eval_df, SAMPLE_LIMIT)
    eval_bundle = encode_eval_labels(eval_df, encoder)

    y_pred = model.predict(eval_bundle.features)

    accuracy = accuracy_score(eval_bundle.labels, y_pred)
    macro_f1 = f1_score(eval_bundle.labels, y_pred, average="macro")

    print("Evaluation samples:", len(eval_bundle.features))
    print("Accuracy:", round(float(accuracy), 4))
    print("Macro F1:", round(float(macro_f1), 4))
    print("Classification report (top 10 classes by support):")
    unique_labels = np.unique(eval_bundle.labels)
    report_dict = cast(
        Dict[str, Dict[str, float]],
        classification_report(
            eval_bundle.labels,
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
