#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find the best-performing contiguous rank window, e.g. best 5 consecutive ranks within 31-50.

We compute per-date predicted rank (1 = highest prediction), then for each sliding window
[start, start+window-1] we compute mean(actual) per date and average across dates.

Usage:
  python scripts/find_best_rank_window.py --predictions PATH_OR_GLOB --start 31 --end 50 --window 5
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_predictions(path_or_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(path_or_glob)) if any(ch in path_or_glob for ch in "*?[]") else [path_or_glob]
    if not paths:
        raise FileNotFoundError(f"No files match: {path_or_glob}")
    path = paths[-1]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    for c in ["date", "ticker", "prediction", "actual"]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {path}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=str, required=True, help="Predictions parquet/csv path or glob")
    ap.add_argument("--start", type=int, default=31)
    ap.add_argument("--end", type=int, default=50)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min_universe", type=int, default=200, help="Skip dates with too few stocks")
    args = ap.parse_args()

    preds = load_predictions(args.predictions).dropna(subset=["prediction", "actual"])
    start = int(args.start)
    end = int(args.end)
    w = int(args.window)
    if w <= 0:
        raise ValueError("--window must be > 0")
    if start > end:
        raise ValueError("--start must be <= --end")
    if (end - start + 1) < w:
        raise ValueError("Range length must be >= window")

    windows: List[Tuple[int, int]] = [(s, s + w - 1) for s in range(start, end - w + 2)]
    sums = {win: 0.0 for win in windows}
    cnts = {win: 0 for win in windows}

    for d, g in preds.groupby("date", sort=True):
        if len(g) < int(args.min_universe):
            continue
        g = g.sort_values("prediction", ascending=False).reset_index(drop=True)
        n = len(g)
        for s, e in windows:
            if s > n:
                continue
            ee = min(e, n)
            sl = g.iloc[s - 1 : ee]["actual"]
            if len(sl) != (e - s + 1):
                continue
            m = float(sl.mean())
            if np.isfinite(m):
                sums[(s, e)] += m
                cnts[(s, e)] += 1

    rows = []
    for win in windows:
        c = cnts[win]
        avg = (sums[win] / c) if c > 0 else float("nan")
        rows.append({"start": win[0], "end": win[1], "avg_return": avg, "n_dates": c})
    out = pd.DataFrame(rows).sort_values("avg_return", ascending=False)

    best = out.iloc[0]
    print(f"BEST window {int(best['start'])}-{int(best['end'])} avg_return={best['avg_return']:.6f}% over {int(best['n_dates'])} dates")
    # Print top 5 for visibility
    print("\nTop 5 windows:")
    for _, r in out.head(5).iterrows():
        print(f"  {int(r['start'])}-{int(r['end'])}: {r['avg_return']:.6f}% (n_dates={int(r['n_dates'])})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


