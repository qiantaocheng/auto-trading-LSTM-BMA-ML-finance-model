#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize T+10 bucket returns vs NASDAQ proxy (QQQ by default).

Requirement:
  - Rebalance every T+10 trading days (non-overlapping): use --rebalance-mode horizon --target-horizon-days 10
  - Plot top buckets (1-10, 11-20, 21-30) as separate lines
  - Compare to NASDAQ (QQQ) T+10 forward return on the same rebalance dates

Outputs:
  - buckets_vs_nasdaq.csv
  - buckets_vs_nasdaq.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-id", type=str, default=None, help="Snapshot id to backtest (default: latest).")
    p.add_argument("--model", type=str, default="ridge_stacking", help="Model to plot buckets for (default ridge_stacking).")
    p.add_argument("--benchmark", type=str, default="QQQ", help="Benchmark ticker (NASDAQ proxy). Default QQQ.")
    p.add_argument("--data-dir", type=str, default="data/factor_exports/factors")
    p.add_argument("--data-file", type=str, default="data/factor_exports/factors/factors_all.parquet")
    p.add_argument("--tickers-file", type=str, default=None, help="Restrict universe to tickers in this file (e.g., >$1B list).")
    p.add_argument("--max-weeks", type=int, default=260)
    p.add_argument("--rebalance-mode", type=str, default="horizon", choices=["horizon", "weekly"])
    p.add_argument("--target-horizon-days", type=int, default=10)
    p.add_argument("--cost-bps", type=float, default=0.0, help="Transaction cost (bps) applied to bucket portfolios via turnover * cost_bps/1e4.")
    p.add_argument("--output-dir", type=str, default="results/t10_bucket_vs_nasdaq")
    p.add_argument("--benchmark-source", type=str, default="auto", choices=["auto", "dataset", "yfinance"],
                   help="How to get benchmark returns. auto=dataset if ticker exists else yfinance.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _bucket_cols(buckets: List[Tuple[int, int]]) -> List[str]:
    return [f"top_{a}_{b}_return" for a, b in buckets]


def _compute_benchmark_t10_from_yfinance(
    bench: str,
    rebalance_dates: pd.Series,
    horizon_days: int,
    logger: logging.Logger,
) -> pd.Series:
    """
    Compute forward returns for benchmark on the same rebalance_dates, using benchmark trading calendar.
    Aligns each rebalance date to the latest available benchmark trading day <= date, then uses +horizon_days trading days.
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        logger.warning("yfinance not available: %s", e)
        return pd.Series(dtype=float)

    dates = pd.to_datetime(rebalance_dates).dropna().sort_values()
    if len(dates) == 0:
        return pd.Series(dtype=float)

    start = (dates.min() - pd.Timedelta(days=30)).date().isoformat()
    end = (dates.max() + pd.Timedelta(days=30)).date().isoformat()
    logger.info("Fetching benchmark %s via yfinance (%s -> %s)...", bench, start, end)
    px = yf.download(
        tickers=bench,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if px is None or px.empty:
        logger.warning("yfinance returned empty for %s", bench)
        return pd.Series(dtype=float)

    close = px["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    trading_days = close.index

    def _tplus_return(d: pd.Timestamp) -> float:
        # base date is last trading day <= d
        base_candidates = trading_days[trading_days <= d]
        if len(base_candidates) == 0:
            return float("nan")
        base = pd.Timestamp(base_candidates[-1])
        base_pos = int(trading_days.get_indexer([base])[0])
        tgt_pos = base_pos + int(horizon_days)
        if tgt_pos >= len(trading_days):
            return float("nan")
        tgt = pd.Timestamp(trading_days[tgt_pos])
        b = float(close.loc[base])
        t = float(close.loc[tgt])
        return (t - b) / b if b else float("nan")

    out = pd.Series({_d: _tplus_return(pd.Timestamp(_d)) for _d in dates})
    out.index = pd.to_datetime(out.index)
    out.name = "benchmark_return"
    return out


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("plot_buckets_vs_nasdaq")

    # Ensure repo modules import
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.comprehensive_model_backtest import ComprehensiveModelBacktest

    bt = ComprehensiveModelBacktest(
        data_dir=str(Path(args.data_dir)),
        snapshot_id=args.snapshot_id,
        data_file=str(Path(args.data_file)),
        tickers_file=args.tickers_file,
    )
    bt._rebalance_mode = args.rebalance_mode
    bt._target_horizon_days = int(args.target_horizon_days)
    bt._cost_bps = float(args.cost_bps or 0.0)

    logger.info("Loading factor data (for benchmark target returns)...")
    data = bt.load_factor_data()

    logger.info("Running rolling prediction + bucket returns...")
    all_results, report_df, weekly_details = bt.run_backtest(max_weeks=int(args.max_weeks))

    model_name = str(args.model).strip()
    if model_name not in all_results:
        raise RuntimeError(f"Model '{model_name}' not found in backtest results. Available: {list(all_results.keys())}")

    preds = all_results[model_name]
    if preds.empty:
        raise RuntimeError(f"No predictions found for model '{model_name}'.")

    # Desired buckets: top 1-10, 11-20, 21-30
    buckets = [(1, 10), (11, 20), (21, 30)]
    bucket_summary, bucket_ts = bt.calculate_bucket_returns(
        predictions=preds,
        top_buckets=buckets,
        bottom_buckets=[],
        cost_bps=float(args.cost_bps or 0.0),
    )
    if bucket_ts.empty:
        raise RuntimeError("Bucket time series is empty.")

    # Benchmark: use the same rebalance dates and pull T+10 forward return from dataset 'target'
    bench = str(args.benchmark).upper().strip()
    if "target" not in data.columns:
        raise RuntimeError("Factor dataset does not contain 'target' column; cannot compute benchmark T+10 return.")

    # data is MultiIndex(date,ticker). Pull benchmark target per date if available.
    bench_series = pd.Series(dtype=float)
    source = str(args.benchmark_source).lower().strip()
    if source in ("auto", "dataset"):
        try:
            bench_slice = data.xs(bench, level="ticker", drop_level=False)
            bench_series = bench_slice["target"].copy()
            if isinstance(bench_series.index, pd.MultiIndex):
                bench_series = bench_series.reset_index().set_index("date")["target"]
            logger.info("Benchmark %s found in dataset; using dataset target as benchmark return.", bench)
        except Exception:
            bench_series = pd.Series(dtype=float)

    if (source in ("auto", "yfinance")) and (bench_series.empty or bench_series.isna().all()):
        bench_series = _compute_benchmark_t10_from_yfinance(
            bench=bench,
            rebalance_dates=bucket_ts["date"],
            horizon_days=int(args.target_horizon_days),
            logger=logger,
        )

    net_cols = [f"top_{a}_{b}_return_net" for (a, b) in buckets]
    out = bucket_ts[["date"] + _bucket_cols(buckets) + net_cols].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    out["benchmark_return"] = out["date"].map(lambda d: float(bench_series.get(d, float("nan"))) if hasattr(bench_series, "get") else float("nan"))

    # Convert to % for readability (same convention as existing reports)
    pct_cols = _bucket_cols(buckets) + net_cols + ["benchmark_return"]
    for c in pct_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce") * 100.0

    # Compute cumulative (compounded) returns
    # Treat missing returns as 0% to avoid breaking the cumprod (benchmark may have NaNs at edges).
    def _cum_pct(series_pct: pd.Series) -> pd.Series:
        r = pd.to_numeric(series_pct, errors="coerce").fillna(0.0) / 100.0
        return (1.0 + r).cumprod() - 1.0

    out["cum_top_1_10_return"] = _cum_pct(out["top_1_10_return"]) * 100.0
    out["cum_top_11_20_return"] = _cum_pct(out["top_11_20_return"]) * 100.0
    out["cum_top_21_30_return"] = _cum_pct(out["top_21_30_return"]) * 100.0
    out["cum_top_1_10_return_net"] = _cum_pct(out["top_1_10_return_net"]) * 100.0
    out["cum_top_11_20_return_net"] = _cum_pct(out["top_11_20_return_net"]) * 100.0
    out["cum_top_21_30_return_net"] = _cum_pct(out["top_21_30_return_net"]) * 100.0
    out["cum_benchmark_return"] = _cum_pct(out["benchmark_return"]) * 100.0

    csv_path = out_dir / "buckets_vs_nasdaq.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("Saved CSV: %s", csv_path)

    # Plot: per-period returns
    plt.figure(figsize=(14, 7))
    for (a, b) in buckets:
        col = f"top_{a}_{b}_return"
        plt.plot(out["date"], out[col], linewidth=1.6, label=f"Top {a}-{b}")
    for (a, b) in buckets:
        col = f"top_{a}_{b}_return_net"
        plt.plot(out["date"], out[col], linewidth=1.2, linestyle=":", label=f"Top {a}-{b} (net {args.cost_bps:g}bp)")
    plt.plot(out["date"], out["benchmark_return"], linewidth=2.0, linestyle="--", color="black", label=f"{bench} (T+{args.target_horizon_days})")
    plt.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
    plt.title(f"T+{args.target_horizon_days} bucket returns vs {bench} (rebalance every {args.target_horizon_days} trading days) — model={model_name} — cost={args.cost_bps:g}bp")
    plt.xlabel("Rebalance date")
    plt.ylabel("Average return over next horizon (%)")
    plt.legend()
    plt.tight_layout()
    png_path = out_dir / "buckets_vs_nasdaq.png"
    plt.savefig(png_path, dpi=160)
    plt.close()
    logger.info("Saved PNG: %s", png_path)

    # Plot: cumulative (compounded) returns
    plt.figure(figsize=(14, 7))
    plt.plot(out["date"], out["cum_top_1_10_return"], linewidth=1.8, label="Top 1-10 (cum gross)")
    plt.plot(out["date"], out["cum_top_11_20_return"], linewidth=1.8, label="Top 11-20 (cum gross)")
    plt.plot(out["date"], out["cum_top_21_30_return"], linewidth=1.8, label="Top 21-30 (cum gross)")
    plt.plot(out["date"], out["cum_top_1_10_return_net"], linewidth=1.2, linestyle=":", label=f"Top 1-10 (cum net {args.cost_bps:g}bp)")
    plt.plot(out["date"], out["cum_top_11_20_return_net"], linewidth=1.2, linestyle=":", label=f"Top 11-20 (cum net {args.cost_bps:g}bp)")
    plt.plot(out["date"], out["cum_top_21_30_return_net"], linewidth=1.2, linestyle=":", label=f"Top 21-30 (cum net {args.cost_bps:g}bp)")
    plt.plot(out["date"], out["cum_benchmark_return"], linewidth=2.2, linestyle="--", color="black", label=f"{bench} (cum)")
    plt.axhline(0.0, color="#999999", linewidth=1.0, linestyle=":")
    plt.title(f"Cumulative return vs {bench} (T+{args.target_horizon_days}, rebalance every {args.target_horizon_days} trading days) — model={model_name} — cost={args.cost_bps:g}bp")
    plt.xlabel("Rebalance date")
    plt.ylabel("Cumulative return (%)")
    plt.legend()
    plt.tight_layout()
    cum_png_path = out_dir / "buckets_vs_nasdaq_cumulative.png"
    plt.savefig(cum_png_path, dpi=160)
    plt.close()
    logger.info("Saved cumulative PNG: %s", cum_png_path)

    # Warn if benchmark missing
    if out["benchmark_return"].isna().all():
        logger.warning("Benchmark '%s' is NaN for all dates (dataset missing and yfinance may be unavailable).", bench)

    # Print quick summary
    logger.info("Mean returns (%%): Top1-10=%.4f, Top11-20=%.4f, Top21-30=%.4f, %s=%.4f",
                out["top_1_10_return"].mean(),
                out["top_11_20_return"].mean(),
                out["top_21_30_return"].mean(),
                bench,
                out["benchmark_return"].mean())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


