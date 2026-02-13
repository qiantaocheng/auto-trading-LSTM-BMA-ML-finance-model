#!/usr/bin/env python
"""Layer 1 Robustness Tests — Is this real alpha or backtest illusion?

Test 1: Fine-grained hold period sweep (1–15 days)
Test 3: Cost/slippage stress test (0/10/20/30/50 bps + extreme-stock penalty)
Test 4: Tradability — gap-up filter, skip entries with opening gap > X%
Test 5: Single-name concentration — top ticker profit contribution
Test 7: Regime gating — SPY > SMA200, reduce/stop during bear regime
"""
from __future__ import annotations

import argparse
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
RESULT_DIR = Path("D:/trade/result/layer1_robustness")
SPY_CACHE_PATH = Path("D:/trade/result/minervini_news_backtest/spy_daily.csv")

INITIAL_CAPITAL = 100_000.0
RISK_FREE_RATE = 0.04
START_DATE = "2022-03-01"
END_DATE = "2025-12-31"

PRICE_MAX = 100.0
VOLUME_MIN = 50_000
RVOL_MIN = 1.5
DAILY_RETURN_MIN = 0.02
TOP_K = 8
BEST_HOLD = 7


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading & indicators (reuse from deep dive)
# ---------------------------------------------------------------------------
def load_ohlcv(path: Path) -> pd.DataFrame:
    log("  Reading parquet...")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index(["date", "ticker"]).sort_index()
    df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
    log(f"  {len(df):,} rows, {df.index.get_level_values('ticker').nunique()} tickers")
    return df


def load_spy() -> pd.DataFrame:
    df = pd.read_csv(SPY_CACHE_PATH, parse_dates=["date"], index_col="date")
    df.index = df.index.normalize()
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ticker", sort=False)

    log("  daily_return, SMAs, RVOL, ATR...")
    df["daily_return"] = g["Close"].pct_change()
    df["sma20"] = g["Close"].transform(lambda x: x.rolling(20, min_periods=15).mean())
    df["sma50"] = g["Close"].transform(lambda x: x.rolling(50, min_periods=45).mean())
    df["vol_20d_avg"] = g["Volume"].transform(
        lambda x: x.rolling(20, min_periods=15).mean().shift(1)
    )
    df["rvol"] = df["Volume"] / df["vol_20d_avg"]

    # ATR%
    df["prev_close"] = g["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["prev_close"]).abs()
    tr3 = (df["Low"] - df["prev_close"]).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = g["tr"].transform(lambda x: x.rolling(14, min_periods=10).mean())
    df["atr_pct"] = df["atr14"] / df["Close"] * 100

    # Opening gap (next day): (Open[D+1] - Close[D]) / Close[D]
    df["next_open"] = g["Open"].shift(-1)
    df["opening_gap"] = (df["next_open"] - df["Close"]) / df["Close"]

    # Forward opens for hold period sweep
    log("  Forward open prices (1-15 days)...")
    for h in range(1, 16):
        df[f"exit_open_{h}d"] = g["Open"].shift(-(1 + h))
        df[f"fwd_ret_{h}d"] = df[f"exit_open_{h}d"] / df["next_open"] - 1

    # Ranking score
    df["score"] = df["rvol"] * df["daily_return"]

    log(f"  Done: {len(df):,} rows")
    return df


def apply_layer1(df: pd.DataFrame) -> pd.Series:
    return (
        (df["Close"] < PRICE_MAX) &
        (df["Volume"] > VOLUME_MIN) &
        (df["rvol"] > RVOL_MIN) &
        (df["daily_return"] > DAILY_RETURN_MIN)
    )


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------
def compute_metrics(eq: pd.Series, label: str = "") -> dict:
    rets = eq.pct_change().dropna()
    n = len(rets)
    if n < 10:
        return {"label": label, "cagr": 0, "sharpe": 0, "sortino": 0, "max_dd": 0,
                "calmar": 0, "total": 0, "vol": 0}
    total = eq.iloc[-1] / eq.iloc[0] - 1
    years = n / 252
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    ex = rets - RISK_FREE_RATE / 252
    sharpe = ex.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    ds = rets[rets < 0]
    sortino = ex.mean() / ds.std() * np.sqrt(252) if len(ds) > 0 and ds.std() > 0 else 0
    dd = (eq - eq.cummax()) / eq.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return {"label": label, "cagr": cagr * 100, "sharpe": sharpe, "sortino": sortino,
            "max_dd": max_dd * 100, "calmar": calmar, "total": total * 100,
            "vol": rets.std() * np.sqrt(252) * 100}


# ---------------------------------------------------------------------------
# Unified backtest engine with all test features
# ---------------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    mask: pd.Series,
    trading_days: list[pd.Timestamp],
    warmup_end: pd.Timestamp,
    hold_days: int = 7,
    top_k: int = 8,
    cost_bps: int = 10,
    extreme_cost_mult: float = 1.0,  # extra cost multiplier for ATR%>8 or RVOL>10
    gap_limit: float = 999.0,  # skip entry if opening gap > this (e.g. 0.12 = 12%)
    spy_gate: bool = False,  # only trade when SPY > SMA200
    spy_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest with configurable cost, gap filter, and regime gating."""

    # Price lookup
    price_lookup: dict[tuple[str, str], dict] = {}
    for (dt, ticker), row in df[["Open", "Close"]].iterrows():
        ds = dt.strftime("%Y-%m-%d")
        price_lookup[(ds, ticker)] = {"Open": row["Open"], "Close": row["Close"]}

    # Indicator lookup for cost penalty
    indicator_lookup: dict[tuple[str, str], dict] = {}
    for (dt, ticker), row in df[["atr_pct", "rvol", "opening_gap", "score"]].iterrows():
        ds = dt.strftime("%Y-%m-%d")
        indicator_lookup[(ds, ticker)] = {
            "atr_pct": row["atr_pct"], "rvol": row["rvol"],
            "gap": row["opening_gap"], "score": row["score"],
        }

    td_strs = [td.strftime("%Y-%m-%d") for td in trading_days]
    td_idx = {ds: i for i, ds in enumerate(td_strs)}

    # SPY regime: precompute SPY > SMA200 for each date
    spy_ok: dict[str, bool] = {}
    if spy_gate and spy_df is not None:
        spy_c = spy_df["Close"]
        spy_sma200 = spy_c.rolling(200, min_periods=190).mean()
        for d in spy_c.index:
            ds = d.strftime("%Y-%m-%d")
            spy_ok[ds] = bool(spy_c.loc[d] > spy_sma200.loc[d]) if pd.notna(spy_sma200.loc[d]) else True

    # Pre-compute daily signals
    passed = df[mask].copy()
    available_dates = set(df.index.get_level_values("date").unique())
    signals_by_date: dict[str, list[str]] = {}
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

    # Run
    open_positions: list[dict] = []
    completed_trades: list[dict] = []
    cash = INITIAL_CAPITAL
    base_cost = cost_bps / 10_000.0
    daily_equity: list[dict] = []
    open_tickers: set[str] = set()

    for day_str in td_strs:
        # Exits
        still_open = []
        for pos in open_positions:
            if day_str >= pos["exit_date"]:
                bar = price_lookup.get((day_str, pos["ticker"]))
                if bar and bar["Open"] > 0:
                    # Apply extreme cost penalty on exit too
                    cost = base_cost * pos.get("cost_mult", 1.0)
                    exit_price = bar["Open"]
                    proceeds = pos["shares"] * exit_price * (1 - cost)
                    pnl = proceeds - pos["dollars"]
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
                    cash += proceeds
                    open_tickers.discard(pos["ticker"])
                else:
                    still_open.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # Entries
        idx = td_idx.get(day_str)
        if idx is not None and idx > 0:
            prev_day = td_strs[idx - 1]

            # SPY regime gate
            if spy_gate:
                if not spy_ok.get(prev_day, True):
                    # Skip all entries during bear regime
                    pass
                else:
                    tickers = signals_by_date.get(prev_day, [])
                    valid = []
                    for t in tickers:
                        if t in open_tickers:
                            continue
                        bar = price_lookup.get((day_str, t))
                        if not bar or bar["Open"] <= 0:
                            continue
                        # Gap filter (Test 4)
                        ind = indicator_lookup.get((prev_day, t), {})
                        gap = ind.get("gap", 0) or 0
                        if abs(gap) > gap_limit:
                            continue
                        # Determine cost multiplier
                        cm = 1.0
                        if extreme_cost_mult > 1.0:
                            atr_p = ind.get("atr_pct", 0) or 0
                            rv = ind.get("rvol", 0) or 0
                            if atr_p > 8 or rv > 10:
                                cm = extreme_cost_mult
                        valid.append((t, bar["Open"], cm))

                    if valid and cash > 100:
                        per_stock = cash / len(valid)
                        if per_stock > 50:
                            exit_idx = idx + hold_days
                            exit_date = td_strs[exit_idx] if exit_idx < len(td_strs) else td_strs[-1]
                            for ticker, op, cm in valid:
                                cost = base_cost * cm
                                buy_cost = per_stock * (1 + cost)
                                if buy_cost > cash:
                                    buy_cost = cash
                                shares = (buy_cost / (1 + cost)) / op
                                open_positions.append({
                                    "ticker": ticker, "entry_date": day_str,
                                    "entry_price": op, "shares": shares,
                                    "exit_date": exit_date, "dollars": buy_cost,
                                    "cost_mult": cm,
                                })
                                cash -= buy_cost
                                open_tickers.add(ticker)
            else:
                tickers = signals_by_date.get(prev_day, [])
                valid = []
                for t in tickers:
                    if t in open_tickers:
                        continue
                    bar = price_lookup.get((day_str, t))
                    if not bar or bar["Open"] <= 0:
                        continue
                    ind = indicator_lookup.get((prev_day, t), {})
                    gap = ind.get("gap", 0) or 0
                    if abs(gap) > gap_limit:
                        continue
                    cm = 1.0
                    if extreme_cost_mult > 1.0:
                        atr_p = ind.get("atr_pct", 0) or 0
                        rv = ind.get("rvol", 0) or 0
                        if atr_p > 8 or rv > 10:
                            cm = extreme_cost_mult
                    valid.append((t, bar["Open"], cm))

                if valid and cash > 100:
                    per_stock = cash / len(valid)
                    if per_stock > 50:
                        exit_idx = idx + hold_days
                        exit_date = td_strs[exit_idx] if exit_idx < len(td_strs) else td_strs[-1]
                        for ticker, op, cm in valid:
                            cost = base_cost * cm
                            buy_cost = per_stock * (1 + cost)
                            if buy_cost > cash:
                                buy_cost = cash
                            shares = (buy_cost / (1 + cost)) / op
                            open_positions.append({
                                "ticker": ticker, "entry_date": day_str,
                                "entry_price": op, "shares": shares,
                                "exit_date": exit_date, "dollars": buy_cost,
                                "cost_mult": cm,
                            })
                            cash -= buy_cost
                            open_tickers.add(ticker)

        # MTM
        pos_val = sum(
            pos["shares"] * price_lookup.get((day_str, pos["ticker"]), {}).get("Close", pos["entry_price"])
            for pos in open_positions
        )
        daily_equity.append({"date": day_str, "equity": cash + pos_val, "n_pos": len(open_positions)})

    trades_df = pd.DataFrame(completed_trades)
    eq_df = pd.DataFrame(daily_equity)
    if not eq_df.empty:
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        eq_df = eq_df.set_index("date")
    return trades_df, eq_df


# ---------------------------------------------------------------------------
# Test 1: Hold period sweep 1-15 days
# ---------------------------------------------------------------------------
def test1_hold_sweep(df: pd.DataFrame, mask: pd.Series) -> str:
    log("  Test 1: Hold period sweep 1-15 days...")
    passed = df[mask].copy()

    # TopK per day
    topk = passed.groupby(level="date").apply(lambda g: g.nlargest(TOP_K, "score"))
    if topk.index.nlevels > 2:
        topk = topk.droplevel(0)

    lines = [
        "=" * 90,
        "TEST 1: Hold Period Sweep (1-15 days) — TopK=8, RVOL×Return ranking",
        "=" * 90,
        "",
        f"{'Hold':>5} {'N':>7} {'Avg%':>8} {'Med%':>8} {'Win%':>7} "
        f"{'Sharpe/T':>9} {'Skew':>7} {'P10%':>8} {'P90%':>8} "
        f"{'Top5% PnL Share':>16}",
        "-" * 90,
    ]

    sweep_data = []
    for h in range(1, 16):
        col = f"fwd_ret_{h}d"
        if col not in topk.columns:
            continue
        arr = topk[col].dropna().values
        n = len(arr)
        if n < 50:
            continue
        wins = (arr > 0).sum()
        avg = arr.mean() * 100
        med = np.median(arr) * 100
        wr = wins / n * 100
        sh = arr.mean() / arr.std() if arr.std() > 0 else 0
        sk = float(pd.Series(arr).skew())
        p10 = np.percentile(arr, 10) * 100
        p90 = np.percentile(arr, 90) * 100
        # Top 5% winners contribution
        threshold_95 = np.percentile(arr, 95)
        top5_sum = arr[arr >= threshold_95].sum()
        total_sum = arr.sum()
        top5_share = top5_sum / total_sum * 100 if total_sum != 0 else 0

        sweep_data.append({"hold": h, "avg": avg, "sharpe_t": sh, "win": wr})
        lines.append(
            f"{h:>4}d {n:>7,} {avg:>+7.2f}% {med:>+7.2f}% {wr:>6.1f}% "
            f"{sh:>8.4f} {sk:>+6.1f} {p10:>+7.1f}% {p90:>+7.1f}% "
            f"{top5_share:>14.1f}%"
        )

    # Find plateau
    if sweep_data:
        best = max(sweep_data, key=lambda x: x["sharpe_t"])
        lines.append(f"\n  >>> Best Sharpe/trade at Hold={best['hold']}d "
                      f"(Sharpe/T={best['sharpe_t']:.4f}, Avg={best['avg']:+.2f}%)")
        # Check if 5-9 day range is plateau
        plateau = [d for d in sweep_data if 5 <= d["hold"] <= 9]
        if plateau:
            avg_sh = np.mean([d["sharpe_t"] for d in plateau])
            lines.append(f"  >>> 5-9d plateau avg Sharpe/T = {avg_sh:.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test 3: Cost stress test
# ---------------------------------------------------------------------------
def test3_cost_stress(
    df, mask, trading_days, warmup_end, spy_df
) -> str:
    log("  Test 3: Cost/slippage stress test...")
    scenarios = [
        {"label": "0 bps (frictionless)", "cost": 0, "ext_mult": 1.0},
        {"label": "10 bps (current)", "cost": 10, "ext_mult": 1.0},
        {"label": "20 bps (realistic)", "cost": 20, "ext_mult": 1.0},
        {"label": "30 bps (conservative)", "cost": 30, "ext_mult": 1.0},
        {"label": "50 bps (worst case)", "cost": 50, "ext_mult": 1.0},
        {"label": "20 bps + 1.5x extreme", "cost": 20, "ext_mult": 1.5},
        {"label": "30 bps + 2x extreme", "cost": 30, "ext_mult": 2.0},
    ]

    lines = [
        "",
        "=" * 95,
        "TEST 3: Cost & Slippage Stress Test (Hold=7d, TopK=8)",
        "  'Extreme' = ATR%>8 or RVOL>10x gets extra cost multiplier",
        "=" * 95,
        "",
        f"{'Scenario':>30} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} "
        f"{'MaxDD':>8} {'Total':>10} {'Trades':>7}",
        "-" * 90,
    ]

    for sc in scenarios:
        trades_df, eq_df = run_backtest(
            df, mask, trading_days, warmup_end,
            hold_days=BEST_HOLD, top_k=TOP_K,
            cost_bps=sc["cost"], extreme_cost_mult=sc["ext_mult"],
        )
        if eq_df.empty:
            continue
        m = compute_metrics(eq_df["equity"], sc["label"])
        lines.append(
            f"{sc['label']:>30} {m['cagr']:>+7.2f}% {m['sharpe']:>7.3f} "
            f"{m['sortino']:>7.3f} {m['max_dd']:>7.2f}% {m['total']:>+9.2f}% "
            f"{len(trades_df):>7}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test 4: Gap-up tradability filter
# ---------------------------------------------------------------------------
def test4_gap_filter(
    df, mask, trading_days, warmup_end, spy_df
) -> str:
    log("  Test 4: Gap-up tradability filter...")
    gap_limits = [
        {"label": "No gap limit", "gap": 999.0},
        {"label": "Gap < 20%", "gap": 0.20},
        {"label": "Gap < 15%", "gap": 0.15},
        {"label": "Gap < 12%", "gap": 0.12},
        {"label": "Gap < 10%", "gap": 0.10},
        {"label": "Gap < 8%", "gap": 0.08},
        {"label": "Gap < 5%", "gap": 0.05},
    ]

    lines = [
        "",
        "=" * 95,
        "TEST 4: Gap-Up Tradability Filter (Hold=7d, TopK=8, Cost=20bps)",
        "  Skip entry if next-day opening gap exceeds limit",
        "=" * 95,
        "",
        f"{'Filter':>20} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} "
        f"{'MaxDD':>8} {'Total':>10} {'Trades':>7} {'Skipped':>8}",
        "-" * 90,
    ]

    base_trades = None
    for gl in gap_limits:
        trades_df, eq_df = run_backtest(
            df, mask, trading_days, warmup_end,
            hold_days=BEST_HOLD, top_k=TOP_K,
            cost_bps=20, gap_limit=gl["gap"],
        )
        if base_trades is None:
            base_trades = len(trades_df)
        skipped = base_trades - len(trades_df)
        if eq_df.empty:
            continue
        m = compute_metrics(eq_df["equity"])
        lines.append(
            f"{gl['label']:>20} {m['cagr']:>+7.2f}% {m['sharpe']:>7.3f} "
            f"{m['sortino']:>7.3f} {m['max_dd']:>7.2f}% {m['total']:>+9.2f}% "
            f"{len(trades_df):>7} {skipped:>8}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test 5: Single-name concentration
# ---------------------------------------------------------------------------
def test5_concentration(
    df, mask, trading_days, warmup_end
) -> str:
    log("  Test 5: Single-name concentration...")
    trades_df, _ = run_backtest(
        df, mask, trading_days, warmup_end,
        hold_days=BEST_HOLD, top_k=TOP_K, cost_bps=10,
    )

    if trades_df.empty:
        return "No trades for concentration analysis"

    ticker_pnl = trades_df.groupby("ticker").agg(
        total_pnl=("pnl", "sum"),
        n_trades=("pnl", "count"),
        avg_ret=("return_pct", "mean"),
        max_ret=("return_pct", "max"),
    ).sort_values("total_pnl", ascending=False)

    total_pnl = trades_df["pnl"].sum()
    total_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()

    lines = [
        "",
        "=" * 90,
        "TEST 5: Single-Name Concentration (Hold=7d, TopK=8, Cost=10bps)",
        "=" * 90,
        f"\n  Total trades: {len(trades_df)}, Unique tickers: {len(ticker_pnl)}",
        f"  Total P&L: ${total_pnl:+,.0f}, Total Profit (winners only): ${total_profit:+,.0f}",
        "",
        "  Top 20 Tickers by P&L Contribution:",
        f"  {'Ticker':>8} {'Trades':>7} {'Total PnL':>12} {'% of Total':>10} "
        f"{'Avg Ret%':>9} {'Best Ret%':>10}",
        "  " + "-" * 65,
    ]

    cum_pct = 0
    for i, (ticker, row) in enumerate(ticker_pnl.head(20).iterrows()):
        pct = row["total_pnl"] / total_pnl * 100 if total_pnl != 0 else 0
        cum_pct += pct
        lines.append(
            f"  {ticker:>8} {row['n_trades']:>7} ${row['total_pnl']:>+11,.0f} "
            f"{pct:>+9.1f}% {row['avg_ret']:>+8.2f}% {row['max_ret']:>+9.2f}%"
        )

    # Concentration metrics
    top5_pnl = ticker_pnl.head(5)["total_pnl"].sum()
    top10_pnl = ticker_pnl.head(10)["total_pnl"].sum()
    top20_pnl = ticker_pnl.head(20)["total_pnl"].sum()

    lines.append("")
    lines.append(f"  Top 5 tickers:  ${top5_pnl:+,.0f} = {top5_pnl / total_pnl * 100:.1f}% of total P&L")
    lines.append(f"  Top 10 tickers: ${top10_pnl:+,.0f} = {top10_pnl / total_pnl * 100:.1f}% of total P&L")
    lines.append(f"  Top 20 tickers: ${top20_pnl:+,.0f} = {top20_pnl / total_pnl * 100:.1f}% of total P&L")

    # Single-trade concentration
    trades_sorted = trades_df.sort_values("pnl", ascending=False)
    top5_trades = trades_sorted.head(5)["pnl"].sum()
    top10_trades = trades_sorted.head(10)["pnl"].sum()
    top20_trades = trades_sorted.head(20)["pnl"].sum()

    lines.append("")
    lines.append("  Single-Trade Concentration:")
    lines.append(f"  Top 5 trades:  ${top5_trades:+,.0f} = {top5_trades / total_pnl * 100:.1f}% of total P&L")
    lines.append(f"  Top 10 trades: ${top10_trades:+,.0f} = {top10_trades / total_pnl * 100:.1f}% of total P&L")
    lines.append(f"  Top 20 trades: ${top20_trades:+,.0f} = {top20_trades / total_pnl * 100:.1f}% of total P&L")

    # Annual stability (Test 2 lite)
    trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year
    lines.append("")
    lines.append("  Annual Breakdown + Top-10-Trade Dependence:")
    for year, grp in trades_df.groupby("year"):
        n = len(grp)
        total = grp["pnl"].sum()
        wins = (grp["pnl"] > 0).sum()
        wr = wins / n * 100 if n > 0 else 0
        top10 = grp.nlargest(10, "pnl")["pnl"].sum()
        dep = top10 / total * 100 if total != 0 else 0
        lines.append(
            f"    {year}: {n:4d} trades, PnL=${total:+,.0f}, WR={wr:.1f}%, "
            f"Top10 trades=${top10:+,.0f} ({dep:.0f}% of year's PnL)"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test 7: Regime gating
# ---------------------------------------------------------------------------
def test7_regime_gating(
    df, mask, trading_days, warmup_end, spy_df
) -> str:
    log("  Test 7: Regime gating (SPY > SMA200)...")

    scenarios = [
        {"label": "No gating (baseline)", "gate": False, "cost": 10},
        {"label": "SPY > SMA200 gate", "gate": True, "cost": 10},
        {"label": "SPY gate + 20bps cost", "gate": True, "cost": 20},
        {"label": "SPY gate + 30bps cost", "gate": True, "cost": 30},
        {"label": "No gate + 20bps cost", "gate": False, "cost": 20},
    ]

    lines = [
        "",
        "=" * 100,
        "TEST 7: Regime Gating — SPY > SMA200 (Hold=7d, TopK=8)",
        "  When SPY < SMA200: skip all new entries (stay in cash)",
        "=" * 100,
        "",
        f"{'Scenario':>30} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} "
        f"{'MaxDD':>8} {'Calmar':>8} {'Total':>10} {'Trades':>7}",
        "-" * 100,
    ]

    equity_curves = {}
    for sc in scenarios:
        trades_df, eq_df = run_backtest(
            df, mask, trading_days, warmup_end,
            hold_days=BEST_HOLD, top_k=TOP_K,
            cost_bps=sc["cost"],
            spy_gate=sc["gate"], spy_df=spy_df,
        )
        if eq_df.empty:
            continue
        m = compute_metrics(eq_df["equity"], sc["label"])
        lines.append(
            f"{sc['label']:>30} {m['cagr']:>+7.2f}% {m['sharpe']:>7.3f} "
            f"{m['sortino']:>7.3f} {m['max_dd']:>7.2f}% {m['calmar']:>7.3f} "
            f"{m['total']:>+9.2f}% {len(trades_df):>7}"
        )
        equity_curves[sc["label"]] = eq_df

        # Annual breakdown for gated versions
        if sc["gate"] and not trades_df.empty:
            trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year
            for year, grp in trades_df.groupby("year"):
                n = len(grp)
                total = grp["pnl"].sum()
                wr = (grp["pnl"] > 0).sum() / n * 100 if n > 0 else 0
                lines.append(f"    {year}: {n} trades, PnL=${total:+,.0f}, WR={wr:.1f}%")
            lines.append("")

    # Plot comparison
    if equity_curves:
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]
        for (label, eq_df), color in zip(equity_curves.items(), colors):
            ax.plot(eq_df.index, eq_df["equity"], label=label, linewidth=1.2, color=color)
        ax.set_title("Regime Gating Comparison (Hold=7d, TopK=8)", fontsize=12)
        ax.set_ylabel("Portfolio ($)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        fig.tight_layout()
        fig.savefig(RESULT_DIR / "test7_regime_gating.png", dpi=150)
        plt.close(fig)
        log("  Saved: test7_regime_gating.png")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    args = parser.parse_args()

    log("=" * 65)
    log("Layer 1 Robustness Tests")
    log(f"Period: {args.start} -> {args.end}")
    log("=" * 65)

    log("\n[1/4] Loading data & computing indicators...")
    ohlcv = load_ohlcv(RAW_OHLCV_PATH)
    spy_df = load_spy()
    ohlcv = compute_indicators(ohlcv)

    all_dates = sorted(ohlcv.index.get_level_values("date").unique())
    warmup_end = all_dates[252]
    mask = apply_layer1(ohlcv)
    post_warmup = ohlcv.index.get_level_values("date") > warmup_end
    mask_pw = mask & post_warmup

    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(start_date=args.start, end_date=args.end)
    trading_days = list(schedule.index)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Run all tests
    log("\n[2/4] Running Test 1 (hold sweep) & Test 5 (concentration)...")
    t1 = test1_hold_sweep(ohlcv, mask_pw)
    print(t1)

    t5 = test5_concentration(ohlcv, mask, trading_days, warmup_end)
    print(t5)

    log("\n[3/4] Running Test 3 (cost stress) & Test 4 (gap filter)...")
    t3 = test3_cost_stress(ohlcv, mask, trading_days, warmup_end, spy_df)
    print(t3)

    t4 = test4_gap_filter(ohlcv, mask, trading_days, warmup_end, spy_df)
    print(t4)

    log("\n[4/4] Running Test 7 (regime gating)...")
    t7 = test7_regime_gating(ohlcv, mask, trading_days, warmup_end, spy_df)
    print(t7)

    # Save
    full = t1 + "\n" + t5 + "\n" + t3 + "\n" + t4 + "\n" + t7
    with open(RESULT_DIR / "robustness_results.txt", "w") as f:
        f.write(full)
    log(f"\nSaved: {RESULT_DIR / 'robustness_results.txt'}")
    log("Done!")


if __name__ == "__main__":
    main()
