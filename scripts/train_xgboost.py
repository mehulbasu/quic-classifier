"""Train a multi-GPU XGBoost classifier on QUIC flow features using Dask.

This script mirrors the experimentation patterns used in the RAPIDS random
forest trainer while scaling to datasets that exceed individual GPU memory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

import cudf
import dask.array as da
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask_cuda.utils import get_n_gpus
from dask.distributed import Client
import xgboost as xgb
from xgboost.dask import (
    DaskQuantileDMatrix,
    predict as xgb_dask_predict,
    train as xgb_dask_train,
)

from scripts.load_day import (
    FEATURE_COLUMNS,
    HISTOGRAM_COLUMNS,
    HISTOGRAM_BINS,
    RAW_INPUT_COLUMNS,
    TARGET_COLUMN,
)

DEFAULT_TRAIN_ROOT = Path("datasets/training")
DEFAULT_EVAL_PATH = Path("datasets/cesnet-quic22/W-2022-46/1_Mon/flows-20221114.parquet")
DEFAULT_MODEL_PATH = Path("datasets/cache/models/xgboost_quic.json")
DEFAULT_PARTITION_SIZE = "256MB"
DEFAULT_DEVICE_MEMORY_LIMIT = "14GB"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost classifier on QUIC features with multi-GPU Dask.",
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=DEFAULT_TRAIN_ROOT,
        help="Directory or Parquet file containing training data (default: %(default)s).",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help="Parquet file or directory for evaluation data (default: %(default)s).",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to save the trained model (default: %(default)s).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (auto-detected if omitted).",
    )
    parser.add_argument(
        "--partition-size",
        type=str,
        default=DEFAULT_PARTITION_SIZE,
        help="Desired partition size for Dask repartition (default: %(default)s).",
    )
    parser.add_argument(
        "--device-memory-limit",
        type=str,
        default=DEFAULT_DEVICE_MEMORY_LIMIT,
        help="Per-worker device memory limit passed to LocalCUDACluster (default: %(default)s).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Maximum training samples (useful for debugging).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth (default: %(default)s).",
    )
    parser.add_argument(
        "--max-bin",
        type=int,
        default=256,
        help="Maximum number of histogram bins (default: %(default)s).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Number of boosting rounds (default: %(default)s).",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Row subsample ratio (default: %(default)s).",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Column subsample ratio per tree (default: %(default)s).",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1.0,
        help="L2 regularization term (default: %(default)s).",
    )
    return parser.parse_args()


def detect_gpu_count(requested: int | None) -> int:
    if requested is not None and requested > 0:
        return requested
    try:
        return max(get_n_gpus(), 0)
    except Exception:
        return 0


def use_parquet_source(path: Path) -> str:
    if path.is_dir():
        return str(path / "**" / "*.parquet")
    if path.suffix == ".parquet":
        return str(path)
    raise FileNotFoundError(f"Unsupported training source: {path}")


def load_dask_frame(source: Path, columns: Iterable[str], partition_size: str) -> dask_cudf.DataFrame:
    pattern = use_parquet_source(source)
    ddf = dask_cudf.read_parquet(
        pattern,
        columns=list(columns),
        aggregate_files=False,
    )
    if partition_size:
        ddf = ddf.repartition(partition_size=partition_size)
    return ddf


def summarize_partitions(ddf: dask_cudf.DataFrame, name: str) -> None:
    def _stats(pdf: cudf.DataFrame) -> cudf.DataFrame:
        rows = len(pdf)
        if rows:
            try:
                bytes_total = int(pdf.memory_usage(deep=True).sum())
            except Exception:
                bytes_total = 0
        else:
            bytes_total = 0
        return cudf.DataFrame({"rows": [rows], "bytes": [bytes_total]})

    meta = {"rows": "int64", "bytes": "int64"}
    stats = ddf.map_partitions(_stats, meta=meta).compute()
    if len(stats) == 0:
        print(f"{name}: empty Dask collection")
        return

    rows_per_partition = stats["rows"].to_pandas().astype(int)
    bytes_per_partition = stats["bytes"].to_pandas().astype(int)
    total_rows = int(rows_per_partition.sum())
    total_bytes_gb = bytes_per_partition.sum() / (1024 ** 3)
    print(
        f"{name}: npartitions={ddf.npartitions} total_rows={total_rows} "
        f"total_size={total_bytes_gb:.2f}GB rows/part[min={rows_per_partition.min()} max={rows_per_partition.max()} mean={rows_per_partition.mean():.1f}]"
    )

def _engineer_partition(df: cudf.DataFrame) -> cudf.DataFrame:
    enriched = df.copy(deep=False)

    for column in HISTOGRAM_COLUMNS:
        if column not in enriched.columns:
            continue
        for idx in range(HISTOGRAM_BINS):
            bin_col = f"{column}_BIN_{idx}"
            try:
                enriched[bin_col] = enriched[column].list.get(idx).fillna(0.0).astype("float32")
            except AttributeError:
                enriched[bin_col] = 0.0
        enriched.drop(column, axis=1, inplace=True)

    fwd_bytes = enriched["BYTES"].astype("float64")
    rev_bytes = enriched["BYTES_REV"].astype("float64")
    fwd_packets = enriched["PACKETS"].astype("float64")
    rev_packets = enriched["PACKETS_REV"].astype("float64")
    duration = enriched["DURATION"].astype("float64")
    ppi_len = enriched["PPI_LEN"].astype("float64")
    ppi_duration = enriched["PPI_DURATION"].astype("float64")
    ppi_roundtrips = enriched["PPI_ROUNDTRIPS"].astype("float64")

    total_bytes = fwd_bytes + rev_bytes
    total_packets = fwd_packets + rev_packets

    enriched["TOTAL_BYTES"] = total_bytes
    enriched["TOTAL_PACKETS"] = total_packets
    enriched["BYTES_RATIO"] = (fwd_bytes + 1.0) / (rev_bytes + 1.0)
    enriched["PACKETS_RATIO"] = (fwd_packets + 1.0) / (rev_packets + 1.0)

    bytes_safe = total_bytes.where(total_bytes != 0, 1.0)
    packets_safe = total_packets.where(total_packets != 0, 1.0)
    enriched["BYTES_BALANCE"] = (fwd_bytes - rev_bytes) / bytes_safe
    enriched["PACKETS_BALANCE"] = (fwd_packets - rev_packets) / packets_safe

    fwd_packets_safe = fwd_packets.where(fwd_packets != 0, np.nan)
    rev_packets_safe = rev_packets.where(rev_packets != 0, np.nan)
    duration_safe = duration.where(duration != 0, np.nan)
    ppi_duration_safe = ppi_duration.where(ppi_duration != 0, np.nan)
    ppi_len_safe = ppi_len.where(ppi_len != 0, np.nan)

    enriched["MEAN_PACKET_SIZE_FWD"] = (fwd_bytes / fwd_packets_safe).fillna(0.0)
    enriched["MEAN_PACKET_SIZE_REV"] = (rev_bytes / rev_packets_safe).fillna(0.0)
    enriched["BYTES_PER_SECOND"] = (total_bytes / duration_safe).fillna(0.0)
    enriched["PACKETS_PER_SECOND"] = (total_packets / duration_safe).fillna(0.0)
    enriched["PPI_DENSITY"] = (ppi_len / ppi_duration_safe).fillna(0.0)
    enriched["PPI_ROUNDTRIP_RATIO"] = (ppi_roundtrips / ppi_len_safe).fillna(0.0)

    enriched["LOG_TOTAL_BYTES"] = np.log1p(np.clip(enriched["TOTAL_BYTES"].values, a_min=0.0, a_max=None))
    enriched["LOG_TOTAL_PACKETS"] = np.log1p(np.clip(enriched["TOTAL_PACKETS"].values, a_min=0.0, a_max=None))

    time_first = enriched["TIME_FIRST"].astype("str")
    time_last = enriched["TIME_LAST"].astype("str")
    start_date = time_first.str.slice(0, 10)
    end_date = time_last.str.slice(0, 10)
    start_time = time_first.str.slice(11).str.replace("-", ":")
    end_time = time_last.str.slice(11).str.replace("-", ":")
    
    try:
        start_dt = cudf.to_datetime(start_date.str.cat(start_time, sep=" "))
        end_dt = cudf.to_datetime(end_date.str.cat(end_time, sep=" "))
    except Exception:
        start_dt = None
        end_dt = None

    if start_dt is not None and end_dt is not None:
        duration_from_ts = (end_dt - start_dt).dt.total_seconds().fillna(0.0)
        start_hour = start_dt.dt.hour.fillna(-1).astype("int16")
        start_minute = start_dt.dt.minute.fillna(0).astype("int16")
        start_dayofweek = start_dt.dt.weekday.fillna(-1).astype("int16")
    else:
        # Fallback: use default values
        row_count = len(enriched)
        duration_from_ts = cudf.Series([0.0] * row_count, index=enriched.index)
        start_hour = cudf.Series([-1] * row_count, dtype="int16", index=enriched.index)
        start_minute = cudf.Series([0] * row_count, dtype="int16", index=enriched.index)
        start_dayofweek = cudf.Series([-1] * row_count, dtype="int16", index=enriched.index)

    enriched["DURATION_FROM_TIMESTAMPS"] = duration_from_ts
    enriched["START_HOUR"] = start_hour
    enriched["START_MINUTE_OF_DAY"] = start_hour.astype("int32") * 60 + start_minute.astype("int32")
    enriched["START_DAY_OF_WEEK"] = start_dayofweek
    enriched["IS_WEEKEND"] = (start_dayofweek >= 5).astype("int8")

    enriched["QUIC_VERSION_CODE"] = (
        enriched["QUIC_VERSION"].fillna("UNKNOWN").hash_values().astype("int32")
    )

    for column in FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = np.float32(0.0)

    return enriched


def engineer_features(ddf: dask_cudf.DataFrame) -> dask_cudf.DataFrame:
    meta = _engineer_partition(ddf._meta)
    return ddf.map_partitions(_engineer_partition, meta=meta)


def sample_limit_ddf(ddf: dask_cudf.DataFrame, limit: int | None) -> dask_cudf.DataFrame:
    if limit is None or limit <= 0:
        return ddf
    total_rows = int(ddf.shape[0].compute())
    if total_rows <= limit:
        return ddf
    fraction = limit / total_rows
    return ddf.sample(frac=fraction, random_state=RANDOM_STATE)


def build_label_map(ddf: dask_cudf.DataFrame) -> Dict[str, int]:
    unique_labels = ddf[TARGET_COLUMN].dropna().unique().compute()
    label_list = sorted(unique_labels.to_pandas().tolist())
    return {label: idx for idx, label in enumerate(label_list)}


def encode_labels(ddf: dask_cudf.DataFrame, label_map: Dict[str, int]) -> Tuple[dask_cudf.DataFrame, Dict[int, str]]:
    def map_partition(s: cudf.Series) -> cudf.Series:
        return s.map(label_map)
    
    encoded = ddf[TARGET_COLUMN].map_partitions(map_partition, meta=(None, "int64"))
    ddf["LABEL"] = encoded
    ddf = ddf.dropna(subset=["LABEL"])
    ddf["LABEL"] = ddf["LABEL"].astype("int32")
    inverse = {idx: label for label, idx in label_map.items()}
    return ddf, inverse


def prepare_features_and_labels(
    ddf: dask_cudf.DataFrame,
    label_map: Dict[str, int],
) -> Tuple[dask_cudf.DataFrame, dask_cudf.Series]:
    engineered = engineer_features(ddf)
    engineered, _ = encode_labels(engineered, label_map)
    features = engineered[FEATURE_COLUMNS].astype("float32")
    labels = engineered["LABEL"].astype("int32")
    return features, labels


def compute_numpy(array_like) -> np.ndarray:
    result = array_like
    if hasattr(result, "compute"):
        result = result.compute()
    if hasattr(result, "to_numpy"):
        return result.to_numpy()
    if hasattr(result, "values"):
        return np.asarray(result.values)
    if hasattr(result, "get"):
        return result.get()
    return np.asarray(result)


def train_model(
    client: Client,
    features: dask_cudf.DataFrame,
    labels: dask_cudf.Series,
    params: Dict[str, float | int | str],
    num_boost_round: int,
) -> xgb.Booster:
    dtrain = DaskQuantileDMatrix(client, features, labels)
    train_params = dict(params)

    start = perf_counter()
    result = xgb_dask_train(client, train_params, dtrain, num_boost_round=num_boost_round)
    duration = perf_counter() - start
    print(f"Training time (GPU): {duration:.1f}s")

    booster = result["booster"]
    history = result.get("history", {})
    if history:
        first_metric = next(iter(history))
        final_value = history[first_metric][-1]
        print(f"Final training {first_metric}: {final_value:.4f}")
    return booster


def print_model_params(params: Dict[str, float | int | str]) -> None:
    print("Model parameters:")
    for key, value in params.items():
        print(f"  {key}={value}")


def evaluate_model(
    client: Client,
    booster: xgb.Booster,
    features: dask_cudf.DataFrame,
    labels: dask_cudf.Series,
    inverse_labels: Dict[int, str],
    num_classes: int,
) -> None:
    dtest = DaskQuantileDMatrix(client, features, labels)
    proba = xgb_dask_predict(client, booster, dtest)
    if proba.ndim != 2:
        raise ValueError(f"Expected 2D probability output, received ndim={proba.ndim}")

    class_axis = proba.shape[1]
    def _block_argmax(block: np.ndarray) -> np.ndarray:
        return block.argmax(axis=1).astype(np.int32)

    class_preds = proba.map_blocks(_block_argmax, dtype=np.int32, drop_axis=1)
    y_pred = compute_numpy(class_preds)
    y_true = compute_numpy(labels).astype(np.int32)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Evaluation samples: {len(y_true)}")
    print("Accuracy:", round(float(accuracy), 4))
    print("Macro F1:", round(float(macro_f1), 4))
    print("Classification report (top 10 classes by support):")

    unique_labels = np.unique(y_true)
    target_names = [inverse_labels[int(idx)] for idx in unique_labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=unique_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = [
        (label, metrics)
        for label, metrics in report.items()
        if label not in {"accuracy", "macro avg", "weighted avg"}
    ]
    per_class.sort(key=lambda item: item[1]["support"], reverse=True)
    for label, metrics in per_class[:10]:
        print(
            f"{label:25s} prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} "
            f"f1={metrics['f1-score']:.3f} support={int(metrics['support'])}"
        )


def main() -> None:
    args = parse_args()

    num_gpus = detect_gpu_count(args.num_gpus)
    if num_gpus <= 0:
        raise SystemExit("No GPUs detected. Cannot start LocalCUDACluster.")

    model_params: Dict[str, float | int | str] = {
        "tree_method": "hist",
        "device": "cuda",
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "max_depth": args.max_depth,
        "max_bin": args.max_bin,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "seed": RANDOM_STATE,
    }
    num_boost_round = args.n_estimators

    cluster = LocalCUDACluster(
        n_workers=num_gpus,
        threads_per_worker=1,
        device_memory_limit=args.device_memory_limit,
    )
    client = Client(cluster)
    client.wait_for_workers(num_gpus)

    try:
        print_model_params({**model_params, "n_estimators": num_boost_round})
        print(f"Using {num_gpus} GPU(s).")

        train_columns = list(RAW_INPUT_COLUMNS) + [TARGET_COLUMN]
        train_ddf = load_dask_frame(args.train_root, train_columns, args.partition_size)
        train_ddf = sample_limit_ddf(train_ddf, args.sample_limit)
        summarize_partitions(train_ddf, "Loaded training data")

        label_map = build_label_map(train_ddf)
        inverse_labels = {idx: label for label, idx in label_map.items()}
        num_classes = len(label_map)
        if num_classes <= 1:
            raise ValueError("Training data must contain at least two classes.")
        model_params["num_class"] = num_classes

        train_features, train_labels = prepare_features_and_labels(train_ddf, label_map)
        summarize_partitions(train_features, "Training feature matrix")
        summarize_partitions(train_labels.to_frame("LABEL"), "Training label vector")
        print(f"Training samples: {int(train_labels.count().compute())}")
        print(f"Unique classes: {num_classes}")

        output_model_path = args.output_model
        output_model_path.parent.mkdir(parents=True, exist_ok=True)

        booster = train_model(client, train_features, train_labels, model_params, num_boost_round)
        booster.save_model(str(output_model_path))
        print(f"Saved model to {output_model_path}")

        del train_features, train_labels, train_ddf

        eval_columns = list(RAW_INPUT_COLUMNS) + [TARGET_COLUMN]
        eval_ddf = load_dask_frame(args.eval_path, eval_columns, args.partition_size)
        eval_features, eval_labels = prepare_features_and_labels(eval_ddf, label_map)
        summarize_partitions(eval_features, "Evaluation feature matrix")
        summarize_partitions(eval_labels.to_frame("LABEL"), "Evaluation label vector")

        evaluate_model(client, booster, eval_features, eval_labels, inverse_labels, num_classes)

    finally:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
