#!/usr/bin/env python3
"""
Plot per-model Top-N performance vs benchmark, including net-of-cost returns.

This consumes the outputs of scripts/comprehensive_model_backtest.py:
- <model>_weekly_returns_<ts>.csv (contains date, top_return, top_return_net, etc.)

Outputs:
- per_model_topN_vs_benchmark.png
- per_model_topN_vs_benchmark_cum.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _cum_pct(s_pct: pd.Series) -> pd.Series:
    r = pd.to_numeric(s_pct, errors="coerce").fillna(0.0) / 100.0
    return (1.0 + r).cumprod() - 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest-outdir", type=str, required=True, help="Directory containing *_weekly_returns_*.csv files.")
    # Paper scope: CatBoost excluded.
    ap.add_argument("--models", nargs="+", default=["elastic_net", "xgboost", "lambdarank", "ridge_stacking"])
    ap.add_argument("--benchmark", type=str, default="QQQ")
    ap.add_argument("--benchmark-col", type=str, default="benchmark_return", help="Column name used for benchmark in merged CSV (percent).")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("plot_single_models")

    outdir = Path(args.backtest_outdir)
    if not outdir.exists():
        raise FileNotFoundError(outdir)

    files = sorted(outdir.glob("*_weekly_returns_*.csv"))
    if not files:
        raise RuntimeError(f"No *_weekly_returns_*.csv found in {outdir}")

    # Load the latest per model by filename sort (timestamps in names)
    model_df = {}
    for m in args.models:
        m = str(m).strip()
        m_files = [f for f in files if f.name.lower().startswith(m.lower() + "_weekly_returns_")]
        if not m_files:
            logger.warning("Missing weekly returns for model=%s", m)
            continue
        f = m_files[-1]
        df = pd.read_csv(f)
        if "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        model_df[m] = df.sort_values("date")

    if not model_df:
        raise RuntimeError("No model weekly returns loaded.")

    # Merge all models on date
    dates = None
    for df in model_df.values():
        d = df["date"]
        dates = d if dates is None else dates[dates.isin(set(d))]
    if dates is None or len(dates) == 0:
        raise RuntimeError("No overlapping dates across models.")

    # Build plotting frame
    plot = pd.DataFrame({"date": sorted(pd.to_datetime(dates).unique())})
    for m, df in model_df.items():
        df = df[df["date"].isin(plot["date"])].copy()
        # percent series
        if "top_return" in df.columns:
            plot[f"{m}_top_return"] = pd.to_numeric(df["top_return"], errors="coerce") * 100.0
        if "top_return_net" in df.columns:
            plot[f"{m}_top_return_net"] = pd.to_numeric(df["top_return_net"], errors="coerce") * 100.0

    # Benchmark via yfinance on the same rebalance dates (T+10 forward return)
    try:
        import yfinance as yf  # type: ignore
        start = (plot["date"].min() - pd.Timedelta(days=30)).date().isoformat()
        end = (plot["date"].max() + pd.Timedelta(days=30)).date().isoformat()
        px = yf.download(args.benchmark, start=start, end=end, progress=False, auto_adjust=False)
        close = px["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
        close = close.sort_index()
        trading_days = close.index

        def _tplus(d: pd.Timestamp, horizon: int = 10) -> float:
            base_candidates = trading_days[trading_days <= d]
            if len(base_candidates) == 0:
                return float("nan")
            base = pd.Timestamp(base_candidates[-1])
            base_pos = int(trading_days.get_indexer([base])[0])
            tgt_pos = base_pos + horizon
            if tgt_pos >= len(trading_days):
                return float("nan")
            tgt = pd.Timestamp(trading_days[tgt_pos])
            b = float(close.loc[base])
            t = float(close.loc[tgt])
            return (t - b) / b if b else float("nan")

        plot[args.benchmark_col] = plot["date"].map(lambda d: _tplus(pd.Timestamp(d)) * 100.0)
    except Exception as e:
        logger.warning("Benchmark fetch failed: %s", e)
        plot[args.benchmark_col] = float("nan")

    out_plot_dir = Path(args.output_dir)
    out_plot_dir.mkdir(parents=True, exist_ok=True)

    # Period returns plot
    plt.figure(figsize=(14, 7))
    for m in args.models:
        if f"{m}_top_return" in plot.columns:
            plt.plot(plot["date"], plot[f"{m}_top_return"], linewidth=1.6, label=f"{m} gross")
        if f"{m}_top_return_net" in plot.columns:
            plt.plot(plot["date"], plot[f"{m}_top_return_net"], linewidth=1.2, linestyle=":", label=f"{m} net")
    plt.plot(plot["date"], plot[args.benchmark_col], linewidth=2.0, linestyle="--", color="black", label=f"{args.benchmark} (T+10)")
    plt.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
    plt.title("Per-model Top-N forward returns vs benchmark (gross vs net)")
    plt.xlabel("Rebalance date")
    plt.ylabel("Return over next horizon (%)")
    plt.legend(ncol=2)
    plt.tight_layout()
    p1 = out_plot_dir / "per_model_topN_vs_benchmark.png"
    plt.savefig(p1, dpi=160)
    plt.close()

    # Cumulative plot
    plt.figure(figsize=(14, 7))
    for m in args.models:
        if f"{m}_top_return" in plot.columns:
            plt.plot(plot["date"], _cum_pct(plot[f"{m}_top_return"]) * 100.0, linewidth=1.6, label=f"{m} cum gross")
        if f"{m}_top_return_net" in plot.columns:
            plt.plot(plot["date"], _cum_pct(plot[f"{m}_top_return_net"]) * 100.0, linewidth=1.2, linestyle=":", label=f"{m} cum net")
    plt.plot(plot["date"], _cum_pct(plot[args.benchmark_col]) * 100.0, linewidth=2.2, linestyle="--", color="black", label=f"{args.benchmark} cum")
    plt.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
    plt.title("Per-model cumulative return vs benchmark (gross vs net)")
    plt.xlabel("Rebalance date")
    plt.ylabel("Cumulative return (%)")
    plt.legend(ncol=2)
    plt.tight_layout()
    p2 = out_plot_dir / "per_model_topN_vs_benchmark_cum.png"
    plt.savefig(p2, dpi=160)
    plt.close()

    plot.to_csv(out_plot_dir / "per_model_topN_vs_benchmark.csv", index=False, encoding="utf-8")
    logger.info("WROTE %s", out_plot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


