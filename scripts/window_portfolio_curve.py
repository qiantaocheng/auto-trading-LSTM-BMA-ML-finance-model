#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a weekly-rebalanced "rank window" portfolio equity curve from saved predictions.

Interpretation:
  - For each week (date), sort by prediction desc.
  - Select ranks [start, end] (1-based, inclusive).
  - Weekly return = mean(actual) of selected names for that date.
  - actual is assumed to be in PERCENT units (e.g. 0.35 means +0.35%).
  - Cumulative NAV uses: nav *= (1 + weekly_return_pct/100).

Usage:
  python scripts/window_portfolio_curve.py \
    --predictions results/.../ridge_stacking_predictions_*.parquet \
    --start 43 --end 47 \
    --output results/ridge_rank_43_47_curve.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def _latest(path_or_glob: str) -> str:
    paths = sorted(glob.glob(path_or_glob)) if any(ch in path_or_glob for ch in "*?[]") else [path_or_glob]
    if not paths:
        raise FileNotFoundError(f"No files match: {path_or_glob}")
    p = paths[-1]
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def load_predictions(path_or_glob: str) -> pd.DataFrame:
    p = _latest(path_or_glob)
    if p.lower().endswith(".parquet"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    for c in ["date", "ticker", "prediction", "actual"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {p}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def compute_window_curve(preds: pd.DataFrame, start: int, end: int, min_universe: int = 200) -> pd.DataFrame:
    rows: List[dict] = []
    for d, g in preds.groupby("date", sort=True):
        g = g.dropna(subset=["prediction", "actual"])
        if len(g) < min_universe:
            continue
        g = g.sort_values("prediction", ascending=False).reset_index(drop=True)
        if len(g) < end:
            continue
        sl = g.iloc[start - 1 : end]
        r = float(sl["actual"].mean())  # percent units
        rows.append(
            {
                "date": pd.to_datetime(d),
                "window_start": int(start),
                "window_end": int(end),
                "n_stocks": int(len(sl)),
                "weekly_return_pct": r,
            }
        )
    out = pd.DataFrame(rows).sort_values("date")
    if out.empty:
        return out
    out["weekly_return_decimal"] = out["weekly_return_pct"] / 100.0
    out["cum_nav"] = (1.0 + out["weekly_return_decimal"]).cumprod()
    out["cum_return_pct"] = (out["cum_nav"] - 1.0) * 100.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=str, required=True, help="Predictions parquet/csv path or glob")
    ap.add_argument("--start", type=int, default=43)
    ap.add_argument("--end", type=int, default=47)
    ap.add_argument("--min-universe", type=int, default=200)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    start = int(args.start)
    end = int(args.end)
    if start <= 0 or end <= 0 or start > end:
        raise ValueError("--start/--end must be positive and start<=end")

    preds = load_predictions(args.predictions)
    curve = compute_window_curve(preds, start=start, end=end, min_universe=int(args.min_universe))
    if curve.empty:
        raise RuntimeError("No curve produced (check dates/universe size/inputs).")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    curve.to_csv(args.output, index=False)

    first = curve.iloc[0]
    last = curve.iloc[-1]
    print(f"WROTE {args.output}")
    print(f"WEEKS {len(curve)} | START {first['date'].date()} | END {last['date'].date()}")
    print(f"FINAL_NAV {float(last['cum_nav']):.6f} | FINAL_RETURN {float(last['cum_return_pct']):.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


