#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run best contiguous rank-window search for all model prediction files in a backtest output directory.

Example:
  python scripts/find_best_rank_window_all_models.py ^
    --output-dir results/full_bucket_10x150_<SID> ^
    --start 31 --end 50 --window 5
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


MODEL_ORDER = ["elastic_net", "xgboost", "catboost", "lambdarank", "ridge_stacking"]


def _latest(glob_pat: str) -> str:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        return ""
    return paths[-1]


def _best_windows_for_predictions(
    df: pd.DataFrame,
    start: int,
    end: int,
    window: int,
    min_universe: int = 200,
) -> pd.DataFrame:
    df = df.dropna(subset=["prediction", "actual"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df = df.dropna(subset=["prediction", "actual"])

    windows: List[Tuple[int, int]] = [(s, s + window - 1) for s in range(start, end - window + 2)]
    sums = {win: 0.0 for win in windows}
    cnts = {win: 0 for win in windows}

    for _, g in df.groupby("date", sort=True):
        if len(g) < min_universe:
            continue
        g = g.sort_values("prediction", ascending=False).reset_index(drop=True)
        n = len(g)
        for s, e in windows:
            if e > n:
                continue
            sl = g.iloc[s - 1 : e]["actual"]
            m = float(sl.mean())
            if np.isfinite(m):
                sums[(s, e)] += m
                cnts[(s, e)] += 1

    rows = []
    for (s, e) in windows:
        c = cnts[(s, e)]
        avg = (sums[(s, e)] / c) if c > 0 else float("nan")
        rows.append({"start": s, "end": e, "avg_return": avg, "n_dates": c})
    out = pd.DataFrame(rows).sort_values("avg_return", ascending=False).reset_index(drop=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, required=True, help="Backtest output directory containing *_predictions_*.parquet")
    ap.add_argument("--start", type=int, default=31)
    ap.add_argument("--end", type=int, default=50)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-universe", type=int, default=200)
    args = ap.parse_args()

    start = int(args.start)
    end = int(args.end)
    window = int(args.window)
    if window <= 0:
        raise ValueError("--window must be > 0")
    if start > end:
        raise ValueError("--start must be <= --end")
    if (end - start + 1) < window:
        raise ValueError("Range length must be >= window")

    outdir = args.output_dir
    results: List[Dict[str, object]] = []

    for model in MODEL_ORDER:
        pred_path = _latest(os.path.join(outdir, f"{model}_predictions_*.parquet"))
        if not pred_path:
            pred_path = _latest(os.path.join(outdir, f"{model}_predictions_*.csv"))
        if not pred_path:
            continue

        df = pd.read_parquet(pred_path) if pred_path.lower().endswith(".parquet") else pd.read_csv(pred_path)
        out = _best_windows_for_predictions(df, start=start, end=end, window=window, min_universe=int(args.min_universe))
        if out.empty:
            continue
        best = out.iloc[0]
        results.append(
            {
                "model": model,
                "best_window": f"{int(best['start'])}-{int(best['end'])}",
                "best_avg_return": float(best["avg_return"]),
                "n_dates": int(best["n_dates"]),
                "pred_file": os.path.basename(pred_path),
            }
        )

        print(f"\n{model}")
        print(f"  BEST {int(best['start'])}-{int(best['end'])}: {float(best['avg_return']):.6f}% (n_dates={int(best['n_dates'])})")
        print("  Top 5:")
        for _, r in out.head(5).iterrows():
            print(f"    {int(r['start'])}-{int(r['end'])}: {float(r['avg_return']):.6f}% (n_dates={int(r['n_dates'])})")

    if results:
        summary = pd.DataFrame(results).sort_values("best_avg_return", ascending=False)
        summary_path = os.path.join(outdir, f"best_windows_{start}_{end}_w{window}.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\nWROTE summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


