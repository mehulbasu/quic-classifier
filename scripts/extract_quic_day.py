#!/usr/bin/env python3
"""Extract a single day's CSV (and optional metadata) from the CESNET-QUIC22 archive."""

import argparse
import os
import sys
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a specific day from the CESNET-QUIC22 dataset ZIP without decompressing everything."
    )
    parser.add_argument(
        "zip_path",
        help="Path to cesnet-quic22.zip",
    )
    parser.add_argument(
        "output_dir",
        help="Directory where the selected files will be extracted",
    )
    parser.add_argument(
        "--week",
        default="W-2022-44",
        help="Week folder to pull from (default: W-2022-44).",
    )
    parser.add_argument(
        "--day",
        default="1_Mon",
        help="Day subfolder inside the week (default: 1_Mon).",
    )
    parser.add_argument(
        "--include-stats",
        action="store_true",
        help="Also extract the day's stats JSON file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.isfile(args.zip_path):
        print(f"ZIP file not found: {args.zip_path}", file=sys.stderr)
        return 1

    target_prefix = f"cesnet-quic22/{args.week}/{args.day}/"

    os.makedirs(args.output_dir, exist_ok=True)

    to_extract = []
    with zipfile.ZipFile(args.zip_path) as zf:
        for info in zf.infolist():
            if not info.filename.startswith(target_prefix):
                continue
            if info.is_dir():
                continue
            if info.filename.endswith(".csv.gz"):
                to_extract.append(info)
            elif args.include_stats and info.filename.endswith(".json"):
                to_extract.append(info)

        if not to_extract:
            print(
                f"No matching files found for prefix {target_prefix}. "
                "Check the week/day values.",
                file=sys.stderr,
            )
            return 1

        for info in to_extract:
            zf.extract(info, path=args.output_dir)
            print(f"Extracted {info.filename} -> {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
