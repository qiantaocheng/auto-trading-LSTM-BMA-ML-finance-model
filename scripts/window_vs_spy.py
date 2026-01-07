#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the cumulative return of a contiguous rank window (e.g., ranks 43-47)
against SPY on a weekly basis.

Usage:
  python scripts/window_vs_spy.py \
    --predictions results/.../ridge_stacking_predictions_*.parquet \
    --start 43 --end 47 \
    --spy SPY \
    --output results/window_vs_spy.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf


def load_predictions(path_or_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(path_or_glob)) if any(ch in path_or_glob for ch in "*?[]") else [path_or_glob]
    if not paths:
        raise FileNotFoundError(f"No files match {path_or_glob}")
    path = paths[-1]
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    for col in ["date", "ticker", "prediction", "actual"]:
        if col not in df.columns:
            raise ValueError(f"Predictions missing column {col}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df


def compute_window_returns(preds: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    target = []
    for date, group in preds.groupby("date", sort=True):
        group = group.dropna(subset=["prediction", "actual"]).sort_values("prediction", ascending=False)
        n = len(group)
        if n < end:
            continue
        window = group.iloc[start - 1 : end]
        ret = float(window["actual"].mean())
        target.append({"date": date, "window_return": ret, "n_stocks": len(window)})
    return pd.DataFrame(target)


def fetch_spy(start: pd.Timestamp, end: pd.Timestamp, ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=(end + pd.Timedelta(days=7)).strftime("%Y-%m-%d"), interval="1wk", auto_adjust=True, progress=False)
    df = df[~df.index.duplicated(keep="last")]
    df = df[["Close"]].copy()
    df["return"] = df["Close"].pct_change()
    df = df.dropna()
    df.index = pd.to_datetime(df.index).normalize()
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=str, required=True)
    ap.add_argument("--start", type=int, default=43)
    ap.add_argument("--end", type=int, default=47)
    ap.add_argument("--spy", type=str, default="SPY")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    preds = load_predictions(args.predictions)
    window = compute_window_returns(preds, start=args.start, end=args.end)
    if window.empty:
        raise RuntimeError("No window returns computed")

    # SPY covering same weeks
    spy_series = fetch_spy(window["date"].min() - pd.Timedelta(days=1), window["date"].max() + pd.Timedelta(days=1), args.spy)
    merged = window.merge(spy_series["return"].rename("spy_return"), left_on="date", right_index=True, how="inner")
    merged["cum_window"] = (1 + merged["window_return"]).cumprod()
    merged["cum_spy"] = (1 + merged["spy_return"]).cumprod()
    merged["excess"] = merged["cum_window"] - merged["cum_spy"]

    summary = {
        "window_mean": merged["window_return"].mean(),
        "spy_mean": merged["spy_return"].mean(),
        "window_std": merged["window_return"].std(),
        "spy_std": merged["spy_return"].std(),
        "total_weeks": len(merged),
        "window_final": float(merged["cum_window"].iloc[-1]),
        "spy_final": float(merged["cum_spy"].iloc[-1]),
        "excess_final": float(merged["cum_window"].iloc[-1] - merged["cum_spy"].iloc[-1]),
    }

    print("SUMMARY", summary)
    if args.output:
        merged.to_csv(args.output, index=False)
        print("WROTE", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


