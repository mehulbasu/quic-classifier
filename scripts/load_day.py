"""Utility helpers for loading a single-day QUIC flow CSV and preparing
features/labels for modeling experiments.

Update DATA_PATH before running this module.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

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
CACHE_DIR = Path("datasets/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _parquet_candidate(path: Path) -> Path:
    if path.suffix == ".parquet":
        return path
    if path.name.endswith(".csv.gz"):
        return path.with_suffix("").with_suffix(".parquet")
    return path.with_suffix(".parquet")


def _resolve_input_path(path: Path) -> Path:
    candidate = _parquet_candidate(path)
    if candidate.suffix == ".parquet" and candidate.exists():
        return candidate
    return path

def load_day(path: Path, feature_columns: List[str]) -> pd.DataFrame:
    """Read a gzipped daily CSV and keep only the selected columns plus the label."""
    cols_to_keep = feature_columns + [TARGET_COLUMN]
    actual_path = _resolve_input_path(path)
    if actual_path.suffix == ".parquet":
        print("Loading data from Parquet:", actual_path)
        df = pd.read_parquet(actual_path, columns=cols_to_keep)
    else:
        compression = "gzip" if ".gz" in actual_path.suffixes else "infer"
        print("Loading data from CSV:", actual_path)
        df = pd.read_csv(
            actual_path,
            compression=compression,
            usecols=cols_to_keep,
            low_memory=False,
        )
    return df.dropna(subset=[TARGET_COLUMN])

def prepare_matrices(
    df: pd.DataFrame, feature_columns: List[str]
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Split the dataframe into numeric features and label-encoded targets."""
    print("Preparing feature matrix and labels.")
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_raw = df[TARGET_COLUMN].astype(str)

    encoder = LabelEncoder()
    encoded_array = np.asarray(encoder.fit_transform(y_raw), dtype=np.int32)
    encoded_list = encoded_array.tolist()
    y = pd.Series(encoded_list, index=y_raw.index, name=TARGET_COLUMN)
    return X, y, encoder

def train_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/validation split with stratification."""
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_val, y_train, y_val

def _cache_key(path: Path, feature_columns: List[str]) -> str:
    joined = ",".join(feature_columns)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}"


def load_cached_training_matrices(
    path: Path,
    feature_columns: List[str],
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    key = _cache_key(path, feature_columns)
    X_path = CACHE_DIR / f"{key}_X.joblib"
    y_path = CACHE_DIR / f"{key}_y.joblib"
    enc_path = CACHE_DIR / f"{key}_encoder.joblib"

    if use_cache and X_path.exists() and y_path.exists() and enc_path.exists():
        X = load(X_path)
        y = load(y_path)
        encoder = load(enc_path)
        return X, y, encoder

    df = load_day(path, feature_columns)
    X, y, encoder = prepare_matrices(df, feature_columns)

    if use_cache:
        dump(X, X_path)
        dump(y, y_path)
        dump(encoder, enc_path)

    return X, y, encoder


def load_cached_features_with_labels(
    path: Path,
    feature_columns: List[str],
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    key = _cache_key(path, feature_columns)
    X_path = CACHE_DIR / f"{key}_eval_X.joblib"
    label_path = CACHE_DIR / f"{key}_eval_labels.joblib"

    if use_cache and X_path.exists() and label_path.exists():
        return load(X_path), load(label_path)

    df = load_day(path, feature_columns)
    features = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = df[TARGET_COLUMN].astype(str)

    if use_cache:
        dump(features, X_path)
        dump(labels, label_path)

    return features, labels

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
