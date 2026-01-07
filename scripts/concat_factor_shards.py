#!/usr/bin/env python3
"""
Concatenate MultiIndex factor shards into ONE parquet file (streaming, low memory).

Input:  a directory containing shard files like `factors_batch_0001.parquet`
Output: a single parquet file containing columns: date, ticker, <factors...>

Note:
- We store `date` and `ticker` as normal columns to allow streaming writes.
- You can restore MultiIndex on read:
    df = pd.read_parquet(out).set_index(["date","ticker"]).sort_index()
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, default="data/factor_exports/factors")
    p.add_argument("--pattern", type=str, default="factors_batch_*.parquet")
    p.add_argument("--output", type=str, default="data/factor_exports/factors/factors_all.parquet")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    input_dir = Path(args.input_dir)
    files = sorted(glob.glob(str(input_dir / args.pattern)))
    if not files:
        raise SystemExit(f"No files found: {input_dir}/{args.pattern}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise SystemExit(f"pyarrow is required for streaming concat: {e}")

    writer = None
    total_rows = 0

    for idx, f in enumerate(files, start=1):
        logger.info(f"[{idx}/{len(files)}] reading {f}")
        df = pd.read_parquet(f)

        # Ensure MultiIndex -> columns for streaming write
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        # Normalize column names
        if "date" not in df.columns or "ticker" not in df.columns:
            raise ValueError(f"Shard missing date/ticker columns after reset_index: {f}")

        # Convert date to datetime64[ns] (safe)
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str)

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema, compression="snappy")
        writer.write_table(table)
        total_rows += len(df)

    if writer is not None:
        writer.close()

    logger.info(f"WROTE {out_path} rows={total_rows:,}")


if __name__ == "__main__":
    main()


