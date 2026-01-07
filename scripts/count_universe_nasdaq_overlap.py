#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count how many tickers in the allfac universe are NASDAQ-listed (based on a provided ticker list file).

This scans the allfac batch parquet files listed in manifest.parquet and only reads the 'ticker' column.

Usage:
  python scripts/count_universe_nasdaq_overlap.py \
    --allfac-dir data/factor_exports/allfac \
    --nasdaq-file D:/trade/us_nasdaq_tickers.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Set

import pandas as pd


def load_ticker_file(path: str) -> Set[str]:
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            for token in s.replace(",", " ").split():
                t = token.strip().strip("'\"").upper()
                if t:
                    out.add(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allfac-dir", type=str, default="data/factor_exports/factors")
    ap.add_argument("--nasdaq-file", type=str, required=True)
    ap.add_argument("--limit-batches", type=int, default=None, help="Debug: only scan first N batches")
    args = ap.parse_args()

    allfac_dir = Path(args.allfac_dir)
    manifest_path = allfac_dir / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))

    nasdaq = load_ticker_file(args.nasdaq_file)
    if not nasdaq:
        raise RuntimeError(f"NASDAQ file empty or unreadable: {args.nasdaq_file}")

    manifest = pd.read_parquet(manifest_path)
    tickers: Set[str] = set()

    n_scanned = 0
    for _, row in manifest.iterrows():
        batch_id = int(row["batch_id"])
        pf = allfac_dir / f"polygon_factors_batch_{batch_id:04d}.parquet"
        if not pf.exists():
            continue
        # NOTE: parquet stores (date,ticker) as index in many of our batch files, and pandas cannot
        # select index levels via read_parquet(columns=[...]). So we read the file and extract tickers
        # from index when present.
        df = pd.read_parquet(pf)
        if isinstance(df.index, pd.MultiIndex):
            names = [n.lower() if isinstance(n, str) else "" for n in df.index.names]
            if "ticker" in names:
                vals = df.index.get_level_values(names.index("ticker"))
            elif "symbol" in names:
                vals = df.index.get_level_values(names.index("symbol"))
            else:
                # fallback to 2nd level for (date,ticker)
                vals = df.index.get_level_values(1) if df.index.nlevels >= 2 else df.index.get_level_values(0)
        else:
            col = "ticker" if "ticker" in df.columns else ("symbol" if "symbol" in df.columns else None)
            if not col:
                continue
            vals = df[col]
        vals = pd.Series(vals).astype(str).str.upper().str.strip()
        tickers.update(vals.unique().tolist())
        n_scanned += 1
        if args.limit_batches and n_scanned >= int(args.limit_batches):
            break

    tickers.discard("")
    overlap = tickers.intersection(nasdaq)

    total = len(tickers)
    n_overlap = len(overlap)
    pct = (n_overlap / total * 100.0) if total else 0.0
    sample = sorted(list(overlap))[:20]

    print(f"TOTAL_UNIVERSE_TICKERS {total}")
    print(f"NASDAQ_TICKERS_IN_UNIVERSE {n_overlap}")
    print(f"NASDAQ_SHARE_PCT {pct:.2f}")
    print("SAMPLE", ",".join(sample))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


