#!/usr/bin/env python3
"""
Filter a factor dataset (MultiIndex or flat) to a given tickers list and write a new parquet.

Supports tickers files in either format:
- one ticker per line
- or a JSON-ish CSV line format like: "A", "AA", "AAPL", ...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def load_tickers(path: str) -> list[str]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8", errors="ignore")
    # extract tickers as quoted tokens if present, else split lines
    toks = re.findall(r'"([A-Za-z0-9\.\-\^=]+)"', txt)
    if toks:
        out = [t.strip().upper() for t in toks if t.strip()]
        return list(dict.fromkeys(out))
    # line-based fallback
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for tok in line.replace(",", " ").split():
            t = tok.strip().strip("'\"").upper()
            if t:
                out.append(t)
    return list(dict.fromkeys(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_file", type=str, default="data/factor_exports/factors/factors_all.parquet")
    ap.add_argument("--tickers-file", type=str, required=True)
    ap.add_argument("--out", dest="out_file", type=str, required=True)
    ap.add_argument("--require-date-ticker-cols", action="store_true", help="Assert output has date,ticker columns (flat format).")
    args = ap.parse_args()

    tickers = set(load_tickers(args.tickers_file))
    if not tickers:
        raise SystemExit(f"Empty tickers list: {args.tickers_file}")

    df = pd.read_parquet(args.in_file)
    # normalize to flat df with date,ticker columns for compatibility with existing scripts
    if isinstance(df.index, pd.MultiIndex):
        if {"date", "ticker"}.issubset(df.index.names):
            df = df.reset_index()
        else:
            df = df.reset_index()
    if not {"date", "ticker"}.issubset(df.columns):
        raise SystemExit("Input factors must have columns date,ticker or a MultiIndex including them.")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    before = int(df["ticker"].nunique())
    df = df[df["ticker"].isin(tickers)].copy()
    after = int(df["ticker"].nunique()) if len(df) else 0

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    if args.require_date_ticker_cols:
        df2 = pd.read_parquet(out)
        assert {"date", "ticker"}.issubset(df2.columns)

    print(f"WROTE {out} rows={len(df):,} tickers {before} -> {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




