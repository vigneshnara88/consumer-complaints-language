from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def refresh_reduced_csv(source_csv: Path, target_csv: Path, chunksize: int = 100_000) -> None:
    header_written = False
    for chunk in pd.read_csv(source_csv, chunksize=chunksize, low_memory=False):
        chunk = chunk.dropna(subset=["Consumer complaint narrative"])
        if chunk.empty:
            continue
        chunk.to_csv(target_csv, mode="a", header=not header_written, index=False)
        header_written = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a current CFPB reduced CSV containing narrative complaints only."
    )
    parser.add_argument("--source-csv", default="complaints.csv", help="Full CFPB CSV downloaded from CFPB.")
    parser.add_argument(
        "--target-csv",
        default="complaints_reduced_latest.csv",
        help="Output path for the refreshed narrative-only CSV.",
    )
    parser.add_argument("--chunksize", type=int, default=100_000, help="Chunk size for streaming the source CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = Path(args.target_csv)
    if target.exists():
        target.unlink()
    refresh_reduced_csv(Path(args.source_csv), target, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
