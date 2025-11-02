"""Convert QUIC daily flow CSV files to Parquet for faster reuse.

Usage examples:

    # Convert the default example file defined below
    python -m scripts.convert_to_parquet

    # Convert one or more explicit CSV paths
    python -m scripts.convert_to_parquet datasets/cesnet-quic22/W-2022-47/1_Mon/flows-20221121.csv.gz \
        datasets/cesnet-quic22/W-2022-47/3_Wed/flows-20221123.csv.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional

import pandas as pd

# Expose a default so the script can be run without arguments during prototyping.
DEFAULT_INPUTS = [
    Path("datasets/cesnet-quic22/W-2022-47/1_Mon/flows-20221121.csv.gz"),
]

CHUNKSIZE = 1_000_000
PARQUET_COMPRESSION = "snappy"


def _candidate_parquet_path(csv_path: Path, output_dir: Optional[Path]) -> Path:
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = csv_path.name
        if base_name.endswith(".csv.gz"):
            base_name = base_name[:-7]
        elif base_name.endswith(".csv"):
            base_name = base_name[:-4]
        return output_dir / f"{base_name}.parquet"

    if csv_path.name.endswith(".csv.gz"):
        return csv_path.with_suffix("").with_suffix(".parquet")
    return csv_path.with_suffix(".parquet")


def _ensure_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - informative runtime guard
        raise SystemExit(
            "pyarrow is required to write Parquet files. Install it with 'pip install pyarrow'."
        ) from exc


def convert_csv_to_parquet(
    csv_path: Path,
    *,
    output_path: Optional[Path] = None,
    chunksize: int = CHUNKSIZE,
    columns: Optional[Iterable[str]] = None,
) -> Path:
    _ensure_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    resolved_csv = csv_path.expanduser()
    if not resolved_csv.exists():
        raise FileNotFoundError(resolved_csv)

    parquet_path = output_path or _candidate_parquet_path(resolved_csv, None)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    compression = "gzip" if ".gz" in resolved_csv.suffixes else "infer"
    reader = pd.read_csv(
        resolved_csv,
        compression=compression,
        chunksize=chunksize,
        low_memory=False,
        usecols=list(columns) if columns else None,
    )

    writer: Optional[pq.ParquetWriter] = None
    total_rows = 0
    start = perf_counter()

    for chunk in reader:
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, compression=PARQUET_COMPRESSION)
        writer.write_table(table)
        total_rows += len(chunk)

    if writer is not None:
        writer.close()
    else:  # No rows were read; create an empty Parquet file.
        empty_table = pa.Table.from_pandas(pd.DataFrame(columns=columns or []))
        pq.write_table(empty_table, parquet_path, compression=PARQUET_COMPRESSION)

    elapsed = perf_counter() - start
    print(
        f"Converted {resolved_csv} -> {parquet_path} | rows={total_rows} | "
        f"elapsed={elapsed:.1f}s"
    )
    return parquet_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert daily QUIC flow CSVs to Parquet.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=None,
        help="Paths to .csv or .csv.gz files. Defaults to the sample list in the script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to store the Parquet files.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        help="Optional subset of columns to keep. By default all columns are preserved.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=CHUNKSIZE,
        help=f"Number of rows per chunk when streaming the CSV (default={CHUNKSIZE}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inputs = args.inputs if args.inputs else DEFAULT_INPUTS
    outputs = []
    for csv_path in inputs:
        parquet_path = convert_csv_to_parquet(
            csv_path,
            output_path=_candidate_parquet_path(csv_path, args.output_dir) if args.output_dir else None,
            chunksize=args.chunksize,
            columns=args.columns,
        )
        outputs.append(parquet_path)

    print("Generated Parquet files:")
    for path in outputs:
        print(" -", path)


if __name__ == "__main__":
    main()
