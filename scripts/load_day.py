"""Utility helpers for loading a single-day QUIC flow CSV and preparing
features/labels for modeling experiments.

Update DATA_PATH before running this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Adjust this path before running. Keep it pointed at one day's flows CSV.
DATA_PATH = Path("datasets/cesnet-quic22/W-2022-47/1_Mon/flows-20221121.csv.gz")

# Numeric, flow-level columns that are cheap to parse and broadly available.
FEATURE_COLUMNS: List[str] = [
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
]

TARGET_COLUMN = "APP"

def load_day(path: Path, feature_columns: List[str]) -> pd.DataFrame:
    """Read a gzipped daily CSV and keep only the selected columns plus the label."""
    cols_to_keep = feature_columns + [TARGET_COLUMN]
    print("Loading data from:", path)
    df = pd.read_csv(path, compression="gzip", usecols=cols_to_keep, low_memory=False)
    return df.dropna(subset=[TARGET_COLUMN])

def prepare_matrices(
    df: pd.DataFrame, feature_columns: List[str]
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Split the dataframe into numeric features and label-encoded targets."""
    print("Preparing feature matrix and labels.")
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_raw = df[TARGET_COLUMN].astype(str)

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y_raw), index=y_raw.index, name=TARGET_COLUMN)
    return X, y, encoder

def train_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/validation split with stratification."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

def main() -> None:
    df = load_day(DATA_PATH, FEATURE_COLUMNS)
    X, y, encoder = prepare_matrices(df, FEATURE_COLUMNS)
    X_train, X_val, y_train, y_val = train_validation_split(X, y)

    print("Loaded rows:", len(df))
    print("Feature columns:", list(X.columns))
    print("Label classes:", list(encoder.classes_))
    print("Train/validation sizes:", len(X_train), len(X_val))

if __name__ == "__main__":
    main()
