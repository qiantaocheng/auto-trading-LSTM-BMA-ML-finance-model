#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot a portfolio equity curve (from window_portfolio_curve.py) against a benchmark (e.g., Nasdaq).

Inputs:
  - curve CSV with columns: date, cum_nav, weekly_return_decimal (or weekly_return_pct)
  - benchmark ticker (default: ^IXIC for Nasdaq Composite)

Outputs:
  - PNG plot
  - merged CSV (optional)
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import yfinance as yf


def fetch_weekly_returns(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    # Add a bit of buffer around the date range.
    start_s = (start - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_s = (end + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start_s, end=end_s, interval="1wk", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned empty data for {ticker}")
    df = df[~df.index.duplicated(keep="last")].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    close = df["Close"]
    # yfinance may return MultiIndex columns even for a single ticker -> Close can be a DataFrame
    if isinstance(close, pd.DataFrame):
        # pick the first (and usually only) column
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce")
    ret = close.pct_change().dropna()
    if not isinstance(ret, pd.Series):
        # last resort
        ret = pd.Series(ret, index=df.index[1:])
    ret.name = f"{ticker}_weekly_return"
    return ret


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve-csv", type=str, required=True)
    ap.add_argument("--benchmark", type=str, default="^IXIC", help="Benchmark ticker (e.g., ^IXIC or QQQ)")
    ap.add_argument("--output-png", type=str, required=True)
    ap.add_argument("--output-csv", type=str, default=None, help="Optional merged output CSV")
    args = ap.parse_args()

    curve = pd.read_csv(args.curve_csv)
    curve["date"] = pd.to_datetime(curve["date"]).dt.tz_localize(None).dt.normalize()
    if "cum_nav" not in curve.columns:
        raise ValueError("curve CSV missing 'cum_nav'")

    start = curve["date"].min()
    end = curve["date"].max()
    bench_ret = fetch_weekly_returns(args.benchmark, start=start, end=end)
    # IMPORTANT: Series.rename("x") is interpreted as an index mapper in some pandas versions.
    # Set name explicitly for safe merges.
    bench_ret = bench_ret.copy()
    bench_ret.name = "benchmark_return"

    # Merge safely by converting to a 1-col DataFrame with a stable column name.
    bench_df = pd.DataFrame({"benchmark_return": bench_ret})
    merged = curve.merge(bench_df, left_on="date", right_index=True, how="inner")
    merged = merged.sort_values("date")
    merged["benchmark_nav"] = (1.0 + merged["benchmark_return"]).cumprod()

    # Plot
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(args.output_png) or ".", exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(merged["date"], merged["cum_nav"], label="Portfolio NAV", linewidth=2.0)
    plt.plot(merged["date"], merged["benchmark_nav"], label=f"Benchmark NAV ({args.benchmark})", linewidth=2.0)
    plt.title("Portfolio vs Benchmark (Weekly Rebalance, No Costs)")
    plt.ylabel("NAV")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=170)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        merged.to_csv(args.output_csv, index=False)

    # Print a short summary
    p_final = float(merged["cum_nav"].iloc[-1])
    b_final = float(merged["benchmark_nav"].iloc[-1])
    print(f"WROTE {args.output_png}")
    if args.output_csv:
        print(f"WROTE {args.output_csv}")
    print(f"FINAL: portfolio={p_final:.6f}, benchmark={b_final:.6f}, weeks={len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


