"""Utility helpers for loading a single-day QUIC flow CSV and preparing
features/labels for modeling experiments.

Update DATA_PATH before running this module.
"""
from pathlib import Path
import hashlib
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Adjust this path before running. Keep it pointed at one day's flows CSV.
DATA_PATH = Path("datasets/cesnet-quic22/W-2022-47/1_Mon/flows-20221121.csv.gz")

# Raw columns required to engineer numeric features.
RAW_INPUT_COLUMNS: List[str] = [
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
    "DST_ASN",
    "SRC_PORT",
    "DST_PORT",
    "PROTOCOL",
    "QUIC_VERSION",
    "TIME_FIRST",
    "TIME_LAST",
    "PHIST_SRC_SIZES",
    "PHIST_DST_SIZES",
    "PHIST_SRC_IPT",
    "PHIST_DST_IPT",
]

HISTOGRAM_COLUMNS: Tuple[str, ...] = (
    "PHIST_SRC_SIZES",
    "PHIST_DST_SIZES",
    "PHIST_SRC_IPT",
    "PHIST_DST_IPT",
)

HISTOGRAM_BINS = 8

DERIVED_FEATURES: List[str] = [
    "TOTAL_BYTES",
    "TOTAL_PACKETS",
    "BYTES_RATIO",
    "PACKETS_RATIO",
    "BYTES_BALANCE",
    "PACKETS_BALANCE",
    "MEAN_PACKET_SIZE_FWD",
    "MEAN_PACKET_SIZE_REV",
    "BYTES_PER_SECOND",
    "PACKETS_PER_SECOND",
    "PPI_DENSITY",
    "PPI_ROUNDTRIP_RATIO",
    "LOG_TOTAL_BYTES",
    "LOG_TOTAL_PACKETS",
    "START_HOUR",
    "START_MINUTE_OF_DAY",
    "START_DAY_OF_WEEK",
    "IS_WEEKEND",
    "DURATION_FROM_TIMESTAMPS",
    "QUIC_VERSION_CODE",
]

# Histogram-derived feature names are generated dynamically below.
HISTOGRAM_FEATURES: List[str] = []
for column in HISTOGRAM_COLUMNS:
    HISTOGRAM_FEATURES.extend(
        [f"{column}_BIN_{idx}" for idx in range(HISTOGRAM_BINS)]
    )

# Base numeric columns retained directly from the input.
BASE_FEATURE_COLUMNS: List[str] = [
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
    "DST_ASN",
    "SRC_PORT",
    "DST_PORT",
    "PROTOCOL",
]

FEATURE_COLUMNS: List[str] = (
    BASE_FEATURE_COLUMNS + DERIVED_FEATURES + HISTOGRAM_FEATURES
)

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


def _ensure_histogram_lists(values: pd.Series) -> pd.Series:
    def normalize(entry: object) -> List[float]:
        if isinstance(entry, (list, tuple)) and len(entry) == HISTOGRAM_BINS:
            return list(entry)
        if isinstance(entry, np.ndarray):
            return entry.tolist()[:HISTOGRAM_BINS]
        return [0.0] * HISTOGRAM_BINS

    return values.apply(normalize)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    # Expand histogram columns into individual numeric features.
    for column in HISTOGRAM_COLUMNS:
        if column in enriched:
            normalized = _ensure_histogram_lists(enriched[column])
            hist_df = pd.DataFrame(
                normalized.tolist(),
                columns=[f"{column}_BIN_{idx}" for idx in range(HISTOGRAM_BINS)],
                index=enriched.index,
            )
            enriched = pd.concat([enriched, hist_df], axis=1)

    # Convert raw histogram arrays to avoid interfering with numeric casting.
    enriched.drop(columns=list(HISTOGRAM_COLUMNS), inplace=True, errors="ignore")

    # Basic flow aggregates and ratios.
    fwd_bytes = enriched["BYTES"].astype(np.float64)
    rev_bytes = enriched["BYTES_REV"].astype(np.float64)
    fwd_packets = enriched["PACKETS"].astype(np.float64)
    rev_packets = enriched["PACKETS_REV"].astype(np.float64)
    duration = enriched["DURATION"].astype(np.float64)
    ppi_len = enriched["PPI_LEN"].astype(np.float64)
    ppi_duration = enriched["PPI_DURATION"].astype(np.float64)
    ppi_roundtrips = enriched["PPI_ROUNDTRIPS"].astype(np.float64)

    total_bytes = fwd_bytes + rev_bytes
    total_packets = fwd_packets + rev_packets

    enriched["TOTAL_BYTES"] = total_bytes
    enriched["TOTAL_PACKETS"] = total_packets
    enriched["BYTES_RATIO"] = (fwd_bytes + 1.0) / (rev_bytes + 1.0)
    enriched["PACKETS_RATIO"] = (fwd_packets + 1.0) / (rev_packets + 1.0)
    enriched["BYTES_BALANCE"] = (fwd_bytes - rev_bytes) / np.where(total_bytes == 0, 1.0, total_bytes)
    enriched["PACKETS_BALANCE"] = (fwd_packets - rev_packets) / np.where(total_packets == 0, 1.0, total_packets)

    enriched["MEAN_PACKET_SIZE_FWD"] = np.divide(
        fwd_bytes,
        np.where(fwd_packets == 0, np.nan, fwd_packets),
    )
    enriched["MEAN_PACKET_SIZE_REV"] = np.divide(
        rev_bytes,
        np.where(rev_packets == 0, np.nan, rev_packets),
    )
    enriched["MEAN_PACKET_SIZE_FWD"] = enriched["MEAN_PACKET_SIZE_FWD"].fillna(0.0)
    enriched["MEAN_PACKET_SIZE_REV"] = enriched["MEAN_PACKET_SIZE_REV"].fillna(0.0)

    enriched["BYTES_PER_SECOND"] = np.divide(
        total_bytes,
        np.where(duration <= 0.0, np.nan, duration),
    )
    enriched["PACKETS_PER_SECOND"] = np.divide(
        total_packets,
        np.where(duration <= 0.0, np.nan, duration),
    )
    enriched["BYTES_PER_SECOND"] = enriched["BYTES_PER_SECOND"].fillna(0.0)
    enriched["PACKETS_PER_SECOND"] = enriched["PACKETS_PER_SECOND"].fillna(0.0)

    enriched["PPI_DENSITY"] = np.divide(
        ppi_len,
        np.where(ppi_duration <= 0.0, np.nan, ppi_duration),
    )
    enriched["PPI_ROUNDTRIP_RATIO"] = np.divide(
        ppi_roundtrips,
        np.where(ppi_len <= 0.0, np.nan, ppi_len),
    )
    enriched["PPI_DENSITY"] = enriched["PPI_DENSITY"].fillna(0.0)
    enriched["PPI_ROUNDTRIP_RATIO"] = enriched["PPI_ROUNDTRIP_RATIO"].fillna(0.0)

    enriched["LOG_TOTAL_BYTES"] = np.log1p(np.clip(total_bytes, a_min=0.0, a_max=None))
    enriched["LOG_TOTAL_PACKETS"] = np.log1p(np.clip(total_packets, a_min=0.0, a_max=None))

    # Timestamp-derived features.
    start_ts = pd.to_datetime(enriched.get("TIME_FIRST"), errors="coerce")
    end_ts = pd.to_datetime(enriched.get("TIME_LAST"), errors="coerce")

    enriched["DURATION_FROM_TIMESTAMPS"] = (
        (end_ts - start_ts).dt.total_seconds().fillna(0.0)
    )

    start_hour = start_ts.dt.hour.fillna(-1).astype(np.int16)
    start_minute = start_ts.dt.minute.fillna(0).astype(np.int16)
    start_dayofweek = start_ts.dt.dayofweek.fillna(-1).astype(np.int16)

    enriched["START_HOUR"] = start_hour
    enriched["START_MINUTE_OF_DAY"] = start_hour * 60 + start_minute
    enriched["START_DAY_OF_WEEK"] = start_dayofweek
    enriched["IS_WEEKEND"] = (start_dayofweek >= 5).astype(np.int8)

    # QUIC version encoding.
    version_codes, _ = pd.factorize(enriched.get("QUIC_VERSION").fillna("UNKNOWN"))
    enriched["QUIC_VERSION_CODE"] = version_codes.astype(np.int32)

    # Ensure all expected feature columns exist.
    for column in FEATURE_COLUMNS:
        if column not in enriched:
            enriched[column] = 0.0

    return enriched


def load_day(path: Path, raw_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read a daily file and keep only the necessary columns plus the label."""
    columns = list(raw_columns or RAW_INPUT_COLUMNS)
    if TARGET_COLUMN not in columns:
        columns.append(TARGET_COLUMN)

    actual_path = _resolve_input_path(path)
    if actual_path.suffix == ".parquet":
        print("Loading data from Parquet:", actual_path)
        df = pd.read_parquet(actual_path, columns=columns)
    else:
        compression = "gzip" if ".gz" in actual_path.suffixes else "infer"
        print("Loading data from CSV:", actual_path)
        df = pd.read_csv(
            actual_path,
            compression=compression,
            usecols=columns,
            low_memory=False,
        )
    return df.dropna(subset=[TARGET_COLUMN])

def prepare_matrices(
    df: pd.DataFrame, feature_columns: List[str]
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Split the dataframe into numeric features and label-encoded targets."""
    print("Preparing feature matrix and labels.")
    enriched = engineer_features(df)
    X = (
        enriched[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
    )
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

    df = load_day(path, RAW_INPUT_COLUMNS)
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

    df = load_day(path, RAW_INPUT_COLUMNS)
    enriched = engineer_features(df)
    features = (
        enriched[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
    )
    labels = df[TARGET_COLUMN].astype(str)

    if use_cache:
        dump(features, X_path)
        dump(labels, label_path)

    return features, labels

def main() -> None:
    df = load_day(DATA_PATH, RAW_INPUT_COLUMNS)
    X, y, encoder = prepare_matrices(df, FEATURE_COLUMNS)
    X_train, X_val, y_train, y_val = train_validation_split(X, y)

    print("Loaded rows:", len(df))
    print("Feature columns:", list(X.columns))
    print("Label classes:", list(encoder.classes_))
    print("Train/validation sizes:", len(X_train), len(X_val))

if __name__ == "__main__":
    main()
