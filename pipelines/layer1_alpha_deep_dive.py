#!/usr/bin/env python
"""Layer 1 Alpha Deep Dive — Multi-period backtest + Bucket analysis.

Step 1: Layer1-only (no Minervini, no News) at hold periods 3, 5, 7, 10 days
Step 2: Bucket analysis by dist_to_MA20, dist_to_MA50, dist_to_52w_high,
        recent 5d gain, ATR%, RVOL — find the "sweet spot"

Time-leakage prevention identical to minervini_news_backtest.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_OHLCV_PATH = Path("D:/trade/data/raw_ohlcv/polygon_raw_ohlcv_2021_2026.parquet")
RESULT_DIR = Path("D:/trade/result/layer1_deep_dive")
SPY_CACHE_PATH = Path("D:/trade/result/minervini_news_backtest/spy_daily.csv")

INITIAL_CAPITAL = 100_000.0
COST_BPS = 10
RISK_FREE_RATE = 0.04
START_DATE = "2022-03-01"
END_DATE = "2025-12-31"

# Layer 1 thresholds
PRICE_MAX = 100.0
VOLUME_MIN = 50_000
RVOL_MIN = 1.5
DAILY_RETURN_MIN = 0.02

# Hold periods to test
HOLD_PERIODS = [3, 5, 7, 10]

# Top-K per day (instead of all passers)
TOP_K = 8


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ohlcv(path: Path) -> pd.DataFrame:
    log("  Reading parquet...")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index(["date", "ticker"]).sort_index()
    df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
    log(f"  Loaded {len(df):,} rows, "
        f"{df.index.get_level_values('ticker').nunique()} tickers, "
        f"{df.index.get_level_values('date').nunique()} dates")
    return df


def load_spy() -> pd.DataFrame:
    df = pd.read_csv(SPY_CACHE_PATH, parse_dates=["date"], index_col="date")
    log(f"  SPY from cache: {len(df)} bars")
    return df


# ---------------------------------------------------------------------------
# Compute indicators
# ---------------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ticker", sort=False)

    log("  daily_return...")
    df["daily_return"] = g["Close"].pct_change()

    log("  SMAs (20/50)...")
    df["sma20"] = g["Close"].transform(lambda x: x.rolling(20, min_periods=15).mean())
    df["sma50"] = g["Close"].transform(lambda x: x.rolling(50, min_periods=45).mean())

    log("  RVOL (20d baseline, shift(1) excludes day D)...")
    df["vol_20d_avg"] = g["Volume"].transform(
        lambda x: x.rolling(20, min_periods=15).mean().shift(1)
    )
    df["rvol"] = df["Volume"] / df["vol_20d_avg"]

    log("  52-week high/low...")
    df["high_52w"] = g["Close"].transform(lambda x: x.rolling(252, min_periods=200).max())
    df["low_52w"] = g["Close"].transform(lambda x: x.rolling(252, min_periods=200).min())

    log("  ATR% (14-day)...")
    df["prev_close"] = g["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["prev_close"]).abs()
    tr3 = (df["Low"] - df["prev_close"]).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = g["tr"].transform(lambda x: x.rolling(14, min_periods=10).mean())
    df["atr_pct"] = df["atr14"] / df["Close"] * 100

    log("  5-day cumulative return...")
    df["ret_5d_cum"] = g["Close"].pct_change(5)

    log("  Distance metrics...")
    df["dist_ma20_pct"] = (df["Close"] - df["sma20"]) / df["sma20"] * 100
    df["dist_ma50_pct"] = (df["Close"] - df["sma50"]) / df["sma50"] * 100
    df["dist_52w_high_pct"] = (df["Close"] - df["high_52w"]) / df["high_52w"] * 100

    log("  Dollar volume (20d avg)...")
    df["dollar_vol_20d"] = g.apply(
        lambda x: (x["Close"] * x["Volume"]).rolling(20, min_periods=15).mean()
    ).droplevel(0).sort_index()

    # Forward returns for multiple periods (analysis only, NOT used in signal generation)
    log("  Forward returns (3/5/7/10 day)...")
    df["next_open"] = g["Open"].shift(-1)  # D+1 Open (buy price)
    for h in HOLD_PERIODS:
        # Exit at Open of D+1+h (h trading days after entry)
        df[f"exit_open_{h}d"] = g["Open"].shift(-(1 + h))
        df[f"fwd_ret_{h}d"] = df[f"exit_open_{h}d"] / df["next_open"] - 1

    log(f"  Done: {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Layer 1 filter
# ---------------------------------------------------------------------------
def apply_layer1(df: pd.DataFrame) -> pd.Series:
    return (
        (df["Close"] < PRICE_MAX) &
        (df["Volume"] > VOLUME_MIN) &
        (df["rvol"] > RVOL_MIN) &
        (df["daily_return"] > DAILY_RETURN_MIN)
    )


# ---------------------------------------------------------------------------
# Step 1: Multi-period Layer1 analysis
# ---------------------------------------------------------------------------
def step1_multiperiod(df: pd.DataFrame, mask: pd.Series) -> dict:
    """Compute avg return, win rate, etc. for Layer1 passers at each hold period."""
    results = {}
    passed = df[mask].copy()

    for h in HOLD_PERIODS:
        col = f"fwd_ret_{h}d"
        valid = passed[col].dropna()
        n = len(valid)
        if n == 0:
            continue
        arr = valid.values
        wins = (arr > 0).sum()
        results[h] = {
            "hold_days": h,
            "n_trades": n,
            "avg_ret": arr.mean() * 100,
            "med_ret": np.median(arr) * 100,
            "win_rate": wins / n * 100,
            "std_ret": arr.std() * 100,
            "sharpe_trade": arr.mean() / arr.std() if arr.std() > 0 else 0,
            "best": arr.max() * 100,
            "worst": arr.min() * 100,
            "p90": np.percentile(arr, 90) * 100,
            "p10": np.percentile(arr, 10) * 100,
            "skew": float(pd.Series(arr).skew()),
            "tail_profit_pct": (arr[arr > np.percentile(arr, 90)].sum() / arr.sum() * 100
                                if arr.sum() != 0 else 0),
        }
    return results


# ---------------------------------------------------------------------------
# Step 2: Bucket analysis
# ---------------------------------------------------------------------------
def step2_bucket_analysis(df: pd.DataFrame, mask: pd.Series, hold_col: str) -> dict:
    """Bucket Layer1 passers by various features, compute returns per bucket."""
    passed = df[mask].copy()
    passed = passed[passed[hold_col].notna()]

    bucket_configs = {
        "dist_ma20_pct": {
            "label": "Distance to MA20 (%)",
            "bins": [-999, 0, 3, 6, 9, 12, 20, 999],
            "labels": ["<0%", "0-3%", "3-6%", "6-9%", "9-12%", "12-20%", ">20%"],
        },
        "dist_ma50_pct": {
            "label": "Distance to MA50 (%)",
            "bins": [-999, 0, 5, 10, 15, 25, 999],
            "labels": ["<0%", "0-5%", "5-10%", "10-15%", "15-25%", ">25%"],
        },
        "dist_52w_high_pct": {
            "label": "Distance to 52w High (%)",
            "bins": [-999, -25, -15, -10, -5, 0, 999],
            "labels": ["<-25%", "-25~-15%", "-15~-10%", "-10~-5%", "-5~0%", "At/Above"],
        },
        "ret_5d_cum": {
            "label": "Recent 5-day Return (%)",
            "bins": [-999, 0, 0.05, 0.10, 0.15, 0.25, 999],
            "labels": ["<0%", "0-5%", "5-10%", "10-15%", "15-25%", ">25%"],
            "scale": 100,
        },
        "atr_pct": {
            "label": "ATR% (14d)",
            "bins": [0, 2, 3, 4, 6, 8, 999],
            "labels": ["0-2%", "2-3%", "3-4%", "4-6%", "6-8%", ">8%"],
        },
        "rvol": {
            "label": "RVOL",
            "bins": [0, 2, 3, 4, 6, 10, 999],
            "labels": ["1.5-2x", "2-3x", "3-4x", "4-6x", "6-10x", ">10x"],
        },
        "daily_return": {
            "label": "Day-of Return (%)",
            "bins": [0, 0.03, 0.05, 0.07, 0.10, 0.15, 999],
            "labels": ["2-3%", "3-5%", "5-7%", "7-10%", "10-15%", ">15%"],
        },
    }

    all_buckets = {}
    for feature, cfg in bucket_configs.items():
        col_data = passed[feature]
        if feature == "ret_5d_cum":
            col_data = col_data  # already fraction, bins are in fraction
        bucket_col = pd.cut(col_data, bins=cfg["bins"], labels=cfg["labels"], right=True)

        bucket_stats = []
        for label in cfg["labels"]:
            group = passed[bucket_col == label]
            rets = group[hold_col].dropna()
            n = len(rets)
            if n < 5:
                bucket_stats.append({
                    "bucket": label, "n": n,
                    "avg_ret": float("nan"), "med_ret": float("nan"),
                    "win_rate": float("nan"), "sharpe_t": float("nan"),
                })
                continue
            arr = rets.values
            bucket_stats.append({
                "bucket": label,
                "n": n,
                "avg_ret": arr.mean() * 100,
                "med_ret": np.median(arr) * 100,
                "win_rate": (arr > 0).sum() / n * 100,
                "sharpe_t": arr.mean() / arr.std() if arr.std() > 0 else 0,
            })

        all_buckets[feature] = {
            "label": cfg["label"],
            "stats": bucket_stats,
        }

    return all_buckets


# ---------------------------------------------------------------------------
# TopK ranking analysis
# ---------------------------------------------------------------------------
def step3_topk_analysis(df: pd.DataFrame, mask: pd.Series) -> dict:
    """Rank Layer1 passers by composite score, take top-K per day."""
    passed = df[mask].copy()

    # Simple ranking score: higher RVOL * daily_return (momentum * volume conviction)
    passed["score_rvol_x_ret"] = passed["rvol"] * passed["daily_return"]
    # Alternative: just daily_return
    passed["score_ret"] = passed["daily_return"]
    # Alternative: RVOL only
    passed["score_rvol"] = passed["rvol"]

    results = {}
    for score_col, label in [
        ("score_rvol_x_ret", "RVOL x Return"),
        ("score_ret", "Return Only"),
        ("score_rvol", "RVOL Only"),
    ]:
        period_results = {}
        for h in HOLD_PERIODS:
            fwd_col = f"fwd_ret_{h}d"

            # Top K per day
            topk = passed.groupby(level="date").apply(
                lambda g: g.nlargest(TOP_K, score_col)
            )
            if isinstance(topk.index, pd.MultiIndex) and topk.index.nlevels > 2:
                topk = topk.droplevel(0)

            valid = topk[fwd_col].dropna()
            n = len(valid)
            if n == 0:
                continue
            arr = valid.values
            wins = (arr > 0).sum()

            # Avg trades per day
            days_with_trades = topk.index.get_level_values("date").nunique()

            period_results[h] = {
                "n_trades": n,
                "avg_per_day": n / days_with_trades if days_with_trades > 0 else 0,
                "avg_ret": arr.mean() * 100,
                "med_ret": np.median(arr) * 100,
                "win_rate": wins / n * 100,
                "sharpe_trade": arr.mean() / arr.std() if arr.std() > 0 else 0,
            }

        results[label] = period_results

    return results


# ---------------------------------------------------------------------------
# Backtest engine (simple equal-weight, Layer1-only, TopK per day)
# ---------------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    mask: pd.Series,
    trading_days: list[pd.Timestamp],
    warmup_end: pd.Timestamp,
    hold_days: int,
    top_k: int = TOP_K,
    cost_bps: int = COST_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run actual portfolio backtest with capital management."""
    # Build price lookup
    price_lookup: dict[tuple[str, str], dict] = {}
    for (dt, ticker), row in df[["Open", "Close"]].iterrows():
        ds = dt.strftime("%Y-%m-%d")
        price_lookup[(ds, ticker)] = {"Open": row["Open"], "Close": row["Close"]}

    td_strs = [td.strftime("%Y-%m-%d") for td in trading_days]
    td_idx = {ds: i for i, ds in enumerate(td_strs)}

    # Pre-compute signals: for each date, get top-K layer1 passers ranked by rvol*return
    passed = df[mask].copy()
    passed["score"] = passed["rvol"] * passed["daily_return"]
    signals_by_date: dict[str, list[str]] = {}

    available_dates = set(df.index.get_level_values("date").unique())
    for td in trading_days:
        td_norm = pd.Timestamp(td).normalize()
        if td_norm <= warmup_end or td_norm not in available_dates:
            continue
        ds = td_norm.strftime("%Y-%m-%d")
        try:
            day_passed = passed.loc[td_norm]
            if isinstance(day_passed, pd.Series):
                signals_by_date[ds] = [day_passed.name] if isinstance(day_passed.name, str) else []
            else:
                topk = day_passed.nlargest(top_k, "score")
                signals_by_date[ds] = topk.index.tolist()
        except KeyError:
            continue

    # Run backtest
    open_positions: list[dict] = []
    completed_trades: list[dict] = []
    cash = INITIAL_CAPITAL
    cost_mult = cost_bps / 10_000.0
    daily_equity: list[dict] = []
    open_tickers: set[str] = set()

    for day_str in td_strs:
        # 1. Exits
        still_open = []
        for pos in open_positions:
            if day_str >= pos["exit_date"]:
                bar = price_lookup.get((day_str, pos["ticker"]))
                if bar and bar["Open"] > 0:
                    exit_price = bar["Open"]
                    sell_proceeds = pos["shares"] * exit_price * (1 - cost_mult)
                    pnl = sell_proceeds - pos["dollars"]
                    ret = pnl / pos["dollars"] if pos["dollars"] > 0 else 0
                    completed_trades.append({
                        "ticker": pos["ticker"],
                        "entry_date": pos["entry_date"],
                        "exit_date": day_str,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "return_pct": ret * 100,
                    })
                    cash += sell_proceeds
                    open_tickers.discard(pos["ticker"])
                else:
                    still_open.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. Entries (signals from previous day, buy at today's open)
        idx = td_idx.get(day_str)
        if idx is not None and idx > 0:
            prev_day = td_strs[idx - 1]
            tickers = signals_by_date.get(prev_day, [])

            valid = []
            for t in tickers:
                if t in open_tickers:
                    continue
                bar = price_lookup.get((day_str, t))
                if bar and bar["Open"] > 0:
                    valid.append((t, bar["Open"]))

            if valid and cash > 100:
                per_stock = cash / len(valid)
                if per_stock > 50:
                    exit_idx = idx + hold_days
                    exit_date = td_strs[exit_idx] if exit_idx < len(td_strs) else td_strs[-1]

                    for ticker, open_price in valid:
                        buy_cost = per_stock * (1 + cost_mult)
                        if buy_cost > cash:
                            buy_cost = cash
                        shares = (buy_cost / (1 + cost_mult)) / open_price
                        open_positions.append({
                            "ticker": ticker,
                            "entry_date": day_str,
                            "entry_price": open_price,
                            "shares": shares,
                            "exit_date": exit_date,
                            "dollars": buy_cost,
                        })
                        cash -= buy_cost
                        open_tickers.add(ticker)

        # 3. Mark-to-market
        pos_val = sum(
            pos["shares"] * price_lookup.get((day_str, pos["ticker"]), {}).get("Close", pos["entry_price"])
            for pos in open_positions
        )
        daily_equity.append({"date": day_str, "equity": cash + pos_val, "n_pos": len(open_positions)})

    trades_df = pd.DataFrame(completed_trades)
    equity_df = pd.DataFrame(daily_equity)
    if not equity_df.empty:
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df.set_index("date")
    return trades_df, equity_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(equity_series: pd.Series, label: str) -> dict:
    returns = equity_series.pct_change().dropna()
    n_days = len(returns)
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    years = n_days / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    excess = returns - RISK_FREE_RATE / 252
    sharpe = excess.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    downside = returns[returns < 0]
    sortino = (excess.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0
    cummax = equity_series.cummax()
    dd = (equity_series - cummax) / cummax
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {
        "label": label, "cagr": cagr * 100, "sharpe": sharpe, "sortino": sortino,
        "max_dd": max_dd * 100, "calmar": calmar, "total_ret": total_return * 100,
        "ann_vol": returns.std() * np.sqrt(252) * 100,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_bucket_heatmap(all_buckets: dict, hold_label: str, save_path: Path):
    """Plot bucket analysis as a grid of bar charts."""
    n_features = len(all_buckets)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 3.5 * n_features))
    if n_features == 1:
        axes = [axes]

    for ax, (feature, data) in zip(axes, all_buckets.items()):
        stats = data["stats"]
        labels = [s["bucket"] for s in stats]
        avg_rets = [s["avg_ret"] if not np.isnan(s.get("avg_ret", float("nan"))) else 0 for s in stats]
        counts = [s["n"] for s in stats]
        win_rates = [s["win_rate"] if not np.isnan(s.get("win_rate", float("nan"))) else 0 for s in stats]

        x = np.arange(len(labels))
        colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in avg_rets]
        bars = ax.bar(x, avg_rets, color=colors, alpha=0.8, edgecolor="gray")

        # Add count labels
        for i, (bar, n, wr) in enumerate(zip(bars, counts, win_rates)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"n={n}\nWR={wr:.0f}%", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Avg Forward Return (%)")
        ax.set_title(f"{data['label']} — {hold_label}")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_equity_comparison(
    equity_curves: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    save_path: Path,
):
    """Plot equity curves for all hold periods + SPY."""
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    for (label, eq_df), color in zip(equity_curves.items(), colors):
        if eq_df.empty:
            continue
        ax.plot(eq_df.index, eq_df["equity"], label=label, linewidth=1.5, color=color)

    # SPY benchmark
    spy_df_norm = spy_df.copy()
    spy_df_norm.index = spy_df_norm.index.normalize()
    first_eq = list(equity_curves.values())[0]
    if not first_eq.empty:
        first_date = first_eq.index[0].normalize()
        if first_date in spy_df_norm.index:
            spy_entry = spy_df_norm.loc[first_date, "Open"]
            spy_shares = INITIAL_CAPITAL / spy_entry
            spy_eq = []
            last_val = INITIAL_CAPITAL
            for d in first_eq.index:
                dn = d.normalize()
                if dn in spy_df_norm.index:
                    last_val = spy_shares * spy_df_norm.loc[dn, "Close"]
                spy_eq.append(last_val)
            ax.plot(first_eq.index, spy_eq, label="SPY B&H", linewidth=1.5,
                    color="gray", alpha=0.6, linestyle="--")

    ax.set_title("Layer 1 Only — TopK=8 — Multi-Period Comparison", fontsize=13)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Layer1 Alpha Deep Dive")
    parser.add_argument("--start", type=str, default=START_DATE)
    parser.add_argument("--end", type=str, default=END_DATE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    top_k = args.top_k
    start_date = args.start
    end_date = args.end

    log("=" * 65)
    log("Layer 1 Alpha Deep Dive")
    log(f"Period: {start_date} -> {end_date}, TopK={top_k}")
    log("=" * 65)

    # [1] Load data
    log("\n[1/6] Loading OHLCV...")
    ohlcv = load_ohlcv(RAW_OHLCV_PATH)

    log("\n[2/6] Loading SPY...")
    spy_df = load_spy()

    # [3] Indicators
    log("\n[3/6] Computing indicators...")
    ohlcv = compute_indicators(ohlcv)

    # Warmup
    all_dates = sorted(ohlcv.index.get_level_values("date").unique())
    warmup_end = all_dates[252] if len(all_dates) > 252 else all_dates[-1]
    log(f"  Warmup ends: {warmup_end.strftime('%Y-%m-%d')}")

    # [4] Layer 1 mask
    log("\n[4/6] Applying Layer 1 filter...")
    mask = apply_layer1(ohlcv)
    n_passed = mask.sum()
    n_total = len(mask)
    log(f"  Layer 1 passers: {n_passed:,} / {n_total:,} ({n_passed / n_total * 100:.1f}%)")

    # Filter to post-warmup only for analysis
    post_warmup = ohlcv.index.get_level_values("date") > warmup_end
    mask_pw = mask & post_warmup

    # =====================================================================
    # STEP 1: Multi-period analysis (all passers)
    # =====================================================================
    log("\n[5/6] Step 1: Multi-period analysis (ALL Layer1 passers)...")
    mp_results = step1_multiperiod(ohlcv, mask_pw)

    lines = [
        "=" * 80,
        "STEP 1: Layer1-Only Multi-Period Analysis (ALL passers, no ranking)",
        "=" * 80,
        "",
        f"{'Hold':>6} {'N Trades':>10} {'Avg Ret':>10} {'Med Ret':>10} {'Win%':>8} "
        f"{'Sharpe/T':>10} {'Skew':>8} {'P10':>8} {'P90':>8} {'Best':>8} {'Worst':>8}",
        "-" * 100,
    ]
    for h in HOLD_PERIODS:
        r = mp_results.get(h)
        if not r:
            continue
        lines.append(
            f"{h:>4}d  {r['n_trades']:>10,} {r['avg_ret']:>+9.2f}% {r['med_ret']:>+9.2f}% "
            f"{r['win_rate']:>7.1f}% {r['sharpe_trade']:>9.4f} {r['skew']:>+7.2f} "
            f"{r['p10']:>+7.1f}% {r['p90']:>+7.1f}% {r['best']:>+7.1f}% {r['worst']:>+7.1f}%"
        )

    # STEP 1b: TopK analysis
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"STEP 1b: TopK={top_k} per Day — Ranking Comparison")
    lines.append("=" * 80)

    topk_results = step3_topk_analysis(ohlcv, mask_pw)
    for ranking_label, period_data in topk_results.items():
        lines.append(f"\n  Ranking: {ranking_label}")
        lines.append(f"  {'Hold':>6} {'N Trades':>10} {'Avg/Day':>8} {'Avg Ret':>10} "
                      f"{'Med Ret':>10} {'Win%':>8} {'Sharpe/T':>10}")
        lines.append("  " + "-" * 75)
        for h in HOLD_PERIODS:
            r = period_data.get(h)
            if not r:
                continue
            lines.append(
                f"  {h:>4}d  {r['n_trades']:>10,} {r['avg_per_day']:>7.1f} "
                f"{r['avg_ret']:>+9.2f}% {r['med_ret']:>+9.2f}% "
                f"{r['win_rate']:>7.1f}% {r['sharpe_trade']:>9.4f}"
            )

    step1_text = "\n".join(lines)
    print(step1_text)

    # =====================================================================
    # STEP 2: Bucket analysis
    # =====================================================================
    log("\n[6/6] Step 2: Bucket analysis...")

    # Use the best hold period from step 1 for bucket analysis
    # Also do 5d for comparison
    bucket_lines = [
        "",
        "=" * 80,
        "STEP 2: Bucket Analysis — Where is the Alpha?",
        "=" * 80,
    ]

    for h in [5, 7]:
        hold_col = f"fwd_ret_{h}d"
        buckets = step2_bucket_analysis(ohlcv, mask_pw, hold_col)

        bucket_lines.append(f"\n{'='*60}")
        bucket_lines.append(f"  Hold Period: {h} days")
        bucket_lines.append(f"{'='*60}")

        for feature, data in buckets.items():
            bucket_lines.append(f"\n  {data['label']}:")
            bucket_lines.append(f"  {'Bucket':>15} {'N':>8} {'Avg Ret':>10} {'Med Ret':>10} "
                                 f"{'Win%':>8} {'Sharpe/T':>10}")
            bucket_lines.append("  " + "-" * 65)
            for s in data["stats"]:
                avg = f"{s['avg_ret']:+.2f}%" if not np.isnan(s.get("avg_ret", float("nan"))) else "   N/A"
                med = f"{s['med_ret']:+.2f}%" if not np.isnan(s.get("med_ret", float("nan"))) else "   N/A"
                wr = f"{s['win_rate']:.1f}%" if not np.isnan(s.get("win_rate", float("nan"))) else "  N/A"
                sh = f"{s['sharpe_t']:.4f}" if not np.isnan(s.get("sharpe_t", float("nan"))) else "  N/A"
                bucket_lines.append(
                    f"  {s['bucket']:>15} {s['n']:>8,} {avg:>10} {med:>10} {wr:>8} {sh:>10}"
                )

        # Plot
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        plot_bucket_heatmap(buckets, f"Hold {h}d", RESULT_DIR / f"buckets_{h}d.png")
        log(f"  Saved: buckets_{h}d.png")

    bucket_text = "\n".join(bucket_lines)
    print(bucket_text)

    # =====================================================================
    # Run actual portfolio backtests for each hold period
    # =====================================================================
    log("\nRunning portfolio backtests (TopK={top_k})...")
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = list(schedule.index)

    equity_curves = {}
    bt_lines = [
        "",
        "=" * 80,
        f"PORTFOLIO BACKTEST — TopK={top_k}, Equal-Weight, Cost={COST_BPS}bps",
        "=" * 80,
        "",
        f"{'Hold':>6} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} "
        f"{'Calmar':>8} {'Total':>10} {'Trades':>8}",
        "-" * 75,
    ]

    for h in HOLD_PERIODS:
        trades_df, eq_df = run_backtest(
            ohlcv, mask, trading_days, warmup_end,
            hold_days=h, top_k=top_k, cost_bps=COST_BPS,
        )
        label = f"Hold {h}d"
        equity_curves[label] = eq_df

        if not eq_df.empty:
            m = compute_metrics(eq_df["equity"], label)
            bt_lines.append(
                f"{h:>4}d  {m['cagr']:>+7.2f}% {m['sharpe']:>7.3f} {m['sortino']:>7.3f} "
                f"{m['max_dd']:>7.2f}% {m['calmar']:>7.3f} {m['total_ret']:>+9.2f}% "
                f"{len(trades_df):>8}"
            )

            # Annual breakdown
            if not trades_df.empty:
                trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year
                for year, grp in trades_df.groupby("year"):
                    n = len(grp)
                    wins = (grp["pnl"] > 0).sum()
                    wr = wins / n * 100 if n > 0 else 0
                    avg_r = grp["return_pct"].mean()
                    tot_pnl = grp["pnl"].sum()
                    bt_lines.append(
                        f"       {year}: {n:4d} trades, win={wr:5.1f}%, "
                        f"avg={avg_r:+5.2f}%, pnl=${tot_pnl:+,.0f}"
                    )
                bt_lines.append("")

    # SPY benchmark
    spy_df_norm = spy_df.copy()
    spy_df_norm.index = spy_df_norm.index.normalize()
    first_eq = list(equity_curves.values())[0]
    if not first_eq.empty:
        first_date = first_eq.index[0].normalize()
        if first_date in spy_df_norm.index:
            spy_entry = spy_df_norm.loc[first_date, "Open"]
            spy_shares = INITIAL_CAPITAL / spy_entry
            spy_eq = []
            last_val = INITIAL_CAPITAL
            for d in first_eq.index:
                dn = d.normalize()
                if dn in spy_df_norm.index:
                    last_val = spy_shares * spy_df_norm.loc[dn, "Close"]
                spy_eq.append(last_val)
            spy_series = pd.Series(spy_eq, index=first_eq.index)
            spy_m = compute_metrics(spy_series, "SPY B&H")
            bt_lines.append(
                f" SPY   {spy_m['cagr']:>+7.2f}% {spy_m['sharpe']:>7.3f} {spy_m['sortino']:>7.3f} "
                f"{spy_m['max_dd']:>7.2f}% {spy_m['calmar']:>7.3f} {spy_m['total_ret']:>+9.2f}%"
            )

    bt_text = "\n".join(bt_lines)
    print(bt_text)

    # Plot equity comparison
    plot_equity_comparison(equity_curves, spy_df, RESULT_DIR / "equity_comparison.png")
    log(f"Saved: equity_comparison.png")

    # Save all text
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    full_text = step1_text + "\n" + bucket_text + "\n" + bt_text
    with open(RESULT_DIR / "full_analysis.txt", "w") as f:
        f.write(full_text)
    log(f"Saved: full_analysis.txt")

    log("\nDone!")


if __name__ == "__main__":
    main()
