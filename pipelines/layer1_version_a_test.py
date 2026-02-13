#!/usr/bin/env python
"""Version A Exit Strategy Test — Hard Stop + Staged TP + Time Stop

Compare against C0 (fixed 7d) using daily OHLCV + 15-min intraday validation.

Version A rules:
  - Hard stop at entry - 1.2*ATR14
  - TP1 at +2R: sell 50%, move stop to breakeven
  - TP2 at +3R: sell 25%, keep 25% to expiry
  - Day3 time stop: if <0.5R and no new high → sell 50%
  - Day 7+1: exit all remaining at Open
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_OHLCV_PATH = Path("D:/trade/data/raw_ohlcv/polygon_raw_ohlcv_2021_2026.parquet")
RESULT_DIR = Path("D:/trade/result/layer1_version_a")
INTRADAY_CACHE = RESULT_DIR / "intraday_cache"

INITIAL_CAPITAL = 100_000.0
RISK_FREE_RATE = 0.04
START_DATE = "2022-03-01"
END_DATE = "2025-12-31"

# Layer 1 filter
PRICE_MAX = 100.0
VOLUME_MIN = 50_000
RVOL_MIN = 1.5
DAILY_RETURN_MIN = 0.02
TOP_K = 8
BEST_HOLD = 7

# Version A exit parameters
STOP_ATR_MULT = 1.2     # hard stop distance in ATR
TP1_ATR_MULT = 2.0      # TP1 distance in ATR
TP2_ATR_MULT = 3.0      # TP2 distance in ATR
TP1_SELL_PCT = 0.50      # sell 50% at TP1
TP2_SELL_PCT = 0.50      # sell 50% of remaining at TP2 (=25% of original)
DAY3_CHECK_DAY = 3       # time stop check day
DAY3_MIN_R = 0.5         # minimum R-multiple to keep
DAY3_SELL_PCT = 0.50     # sell 50% of remaining at day3 stop


def log(msg: str = ""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading & indicators (reused from layer1_hmm_tests.py)
# ---------------------------------------------------------------------------
def load_ohlcv(path: Path) -> pd.DataFrame:
    log("  Reading parquet...")
    df = pd.read_parquet(path, columns=["date", "ticker", "Open", "High", "Low", "Close", "Volume"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index(["date", "ticker"]).sort_index()
    log(f"  {len(df):,} rows, {df.index.get_level_values('ticker').nunique()} tickers")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(level="ticker", group_keys=False)
    log("  daily_return, SMAs, RVOL, ATR...")
    df["daily_return"] = g["Close"].pct_change()
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

    # Opening gap
    df["next_open"] = g["Open"].shift(-1)
    df["opening_gap"] = (df["next_open"] - df["Close"]) / df["Close"]

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
# Vectorized lookup building
# ---------------------------------------------------------------------------
def build_lookups(df, mask, trading_days, warmup_end, top_k=8):
    log("  Building lookups (vectorized)...")
    dates = df.index.get_level_values("date")
    tickers = df.index.get_level_values("ticker")
    date_strs = dates.strftime("%Y-%m-%d")
    keys = list(zip(date_strs, tickers))

    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    price_lookup = {k: {"Open": o[i], "High": h[i], "Low": l[i], "Close": c[i]}
                    for i, k in enumerate(keys)}

    atr_pct_vals = df["atr_pct"].values
    gap_vals = df["opening_gap"].values
    indicator_lookup = {k: {"atr_pct": atr_pct_vals[i], "gap": gap_vals[i]}
                        for i, k in enumerate(keys)}

    td_strs = [td.strftime("%Y-%m-%d") for td in trading_days]
    td_idx = {ds: i for i, ds in enumerate(td_strs)}
    warmup_str = warmup_end.strftime("%Y-%m-%d")

    passed = df[mask].copy()
    available_dates = set(df.index.get_level_values("date").unique())
    signals_by_date = {}
    for td in trading_days:
        td_norm = pd.Timestamp(td).normalize()
        if td_norm <= warmup_end or td_norm not in available_dates:
            continue
        ds = td_norm.strftime("%Y-%m-%d")
        try:
            day_passed = passed.loc[td_norm]
            if isinstance(day_passed, pd.Series):
                if isinstance(day_passed.name, str):
                    atr_v = day_passed.get("atr_pct", 5.0)
                    signals_by_date[ds] = [(day_passed.name, atr_v if pd.notna(atr_v) else 5.0)]
            else:
                topk = day_passed.nlargest(top_k, "score")
                signals_by_date[ds] = [
                    (t, topk.loc[t, "atr_pct"] if pd.notna(topk.loc[t, "atr_pct"]) else 5.0)
                    for t in topk.index.tolist()
                ]
        except KeyError:
            continue

    log(f"  Lookups built: {len(price_lookup):,} prices, {len(signals_by_date)} signal days")
    return {
        "price_lookup": price_lookup,
        "indicator_lookup": indicator_lookup,
        "signals_by_date": signals_by_date,
        "td_strs": td_strs,
        "td_idx": td_idx,
        "warmup_str": warmup_str,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eq: pd.Series, label: str = "") -> dict:
    rets = eq.pct_change().dropna()
    n = len(rets)
    if n < 10:
        return {"label": label, "cagr": 0, "sharpe": 0, "sortino": 0, "max_dd": 0,
                "calmar": 0, "total": 0}
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
            "max_dd": max_dd * 100, "calmar": calmar, "total": total * 100}


# ---------------------------------------------------------------------------
# C0 Baseline Backtest (fixed 7d exit)
# ---------------------------------------------------------------------------
def run_c0_backtest(prebuilt, cost_bps=20, gap_limit=0.08):
    """Run C0 baseline: TopK=8, 7d hold, gap<8%, 20bps. Returns trades + equity."""
    pl = prebuilt["price_lookup"]
    il = prebuilt["indicator_lookup"]
    sbd = prebuilt["signals_by_date"]
    td_strs = prebuilt["td_strs"]
    td_idx = prebuilt["td_idx"]
    warmup_str = prebuilt["warmup_str"]

    open_positions = []
    completed_trades = []
    cash = INITIAL_CAPITAL
    base_cost = cost_bps / 10_000.0
    daily_equity = []
    open_tickers = set()

    for day_str in td_strs:
        if day_str <= warmup_str:
            daily_equity.append({"date": day_str, "equity": INITIAL_CAPITAL, "n_pos": 0})
            continue

        # Exit positions at scheduled date
        still_open = []
        for pos in open_positions:
            if day_str >= pos["exit_date"]:
                bar = pl.get((day_str, pos["ticker"]))
                exit_price = bar["Open"] if bar and bar["Open"] > 0 else pos["entry_price"]
                cost = base_cost
                proceeds = pos["shares"] * exit_price * (1 - cost)
                pnl = proceeds - pos["dollars"]
                ret = pnl / pos["dollars"] if pos["dollars"] > 0 else 0
                completed_trades.append({
                    "ticker": pos["ticker"],
                    "signal_date": pos["signal_date"],
                    "entry_date": pos["entry_date"],
                    "entry_price": pos["entry_price"],
                    "exit_date": day_str,
                    "exit_price": exit_price,
                    "shares": pos["shares"],
                    "dollars": pos["dollars"],
                    "pnl": pnl,
                    "return_pct": ret * 100,
                    "entry_atr_pct": pos["entry_atr_pct"],
                })
                cash += proceeds
                open_tickers.discard(pos["ticker"])
            else:
                still_open.append(pos)
        open_positions = still_open

        # Entries
        idx = td_idx.get(day_str)
        if idx is not None and idx > 0:
            prev_day = td_strs[idx - 1]
            signals = sbd.get(prev_day, [])[:TOP_K]

            valid = []
            for t, entry_atr in signals:
                if t in open_tickers:
                    continue
                bar = pl.get((day_str, t))
                if not bar or bar["Open"] <= 0:
                    continue
                ind = il.get((prev_day, t), {})
                gap = ind.get("gap", 0) or 0
                if abs(gap) > gap_limit:
                    continue
                valid.append((t, bar["Open"], entry_atr))

            if valid and cash > 100:
                per_stock = cash / len(valid)
                if per_stock > 50:
                    exit_idx = idx + BEST_HOLD
                    exit_date = td_strs[exit_idx] if exit_idx < len(td_strs) else td_strs[-1]
                    for ticker, op, entry_atr in valid:
                        cost = base_cost
                        buy_cost = per_stock * (1 + cost)
                        if buy_cost > cash:
                            buy_cost = cash
                        shares = (buy_cost / (1 + cost)) / op
                        open_positions.append({
                            "ticker": ticker, "entry_date": day_str,
                            "entry_price": op, "shares": shares,
                            "exit_date": exit_date, "dollars": buy_cost,
                            "entry_atr_pct": entry_atr,
                            "signal_date": prev_day,
                        })
                        cash -= buy_cost
                        open_tickers.add(ticker)

        # MTM
        pos_val = sum(
            pos["shares"] * pl.get((day_str, pos["ticker"]), {}).get("Close", pos["entry_price"])
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
# Version A: Per-trade simulation using daily OHLCV
# ---------------------------------------------------------------------------
def simulate_va_trade(
    ticker, entry_date, entry_price, atr_pct,
    price_lookup, td_strs, td_idx, cost_bps=20,
):
    """Simulate Version A exit rules for one trade using daily OHLCV bars.

    Returns dict with Version A result fields.
    """
    R = entry_price * atr_pct / 100.0  # 1R in dollars
    if R <= 0:
        R = entry_price * 0.05  # fallback 5%

    stop = entry_price - STOP_ATR_MULT * R
    tp1 = entry_price + TP1_ATR_MULT * R
    tp2 = entry_price + TP2_ATR_MULT * R
    cost = cost_bps / 10_000.0

    remaining = 1.0  # fraction of position remaining
    stop_price = stop
    tp1_hit = False
    tp2_hit = False
    day3_stopped = False
    hard_stopped = False
    max_close = entry_price
    partial_exits = []  # (pct_of_original, exit_price, reason, day_num)
    ambiguous_days = 0

    entry_idx = td_idx.get(entry_date, -1)
    if entry_idx < 0:
        return None

    for d in range(1, BEST_HOLD + 1):
        if remaining <= 0.001:
            break
        day_idx = entry_idx + d
        if day_idx >= len(td_strs):
            break
        day_str = td_strs[day_idx]
        bar = price_lookup.get((day_str, ticker))
        if not bar:
            continue

        O, H, L, C = bar["Open"], bar["High"], bar["Low"], bar["Close"]
        if O <= 0:
            continue

        # Track ambiguity: stop and TP both in range
        tp_level = tp1 if not tp1_hit else (tp2 if not tp2_hit else 999999)
        if L <= stop_price and H >= tp_level:
            ambiguous_days += 1

        # 1. Gap-down: Open < stop_price
        if O <= stop_price:
            partial_exits.append((remaining, O, "gap_stop", d))
            hard_stopped = True
            remaining = 0
            break

        # 2. Hard stop: Low ≤ stop_price (conservative: check before TP)
        if L <= stop_price:
            partial_exits.append((remaining, stop_price, "hard_stop", d))
            hard_stopped = True
            remaining = 0
            break

        # 3. TP1: High ≥ tp1 (first time)
        if not tp1_hit and H >= tp1:
            sell_pct = remaining * TP1_SELL_PCT
            partial_exits.append((sell_pct, tp1, "tp1", d))
            remaining -= sell_pct
            tp1_hit = True
            stop_price = entry_price  # move stop to breakeven

        # 4. TP2: High ≥ tp2 (after TP1 hit)
        if tp1_hit and not tp2_hit and H >= tp2:
            sell_pct = remaining * TP2_SELL_PCT
            partial_exits.append((sell_pct, tp2, "tp2", d))
            remaining -= sell_pct
            tp2_hit = True

        # After TP, check breakeven stop (stop moved to entry)
        if tp1_hit and not hard_stopped and remaining > 0.001:
            if L <= stop_price:  # stop_price is now entry_price (breakeven)
                partial_exits.append((remaining, stop_price, "breakeven_stop", d))
                remaining = 0
                break

        # 5. Day3 time stop (at Close)
        max_close = max(max_close, C)
        if d == DAY3_CHECK_DAY and remaining > 0.001:
            r_return = (C - entry_price) / R if R > 0 else 0
            if r_return < DAY3_MIN_R and C <= max_close * 0.999:
                sell_pct = remaining * DAY3_SELL_PCT
                partial_exits.append((sell_pct, C, "day3_stop", d))
                remaining -= sell_pct
                day3_stopped = True

        max_close = max(max_close, C)

    # 6. Expiry: exit remaining at D+hold+1 Open
    if remaining > 0.001:
        exit_idx = entry_idx + BEST_HOLD + 1
        if exit_idx < len(td_strs):
            exit_day = td_strs[exit_idx]
            exit_bar = price_lookup.get((exit_day, ticker))
            exit_p = exit_bar["Open"] if exit_bar and exit_bar["Open"] > 0 else entry_price
        else:
            exit_p = entry_price
        partial_exits.append((remaining, exit_p, "time_expiry", BEST_HOLD + 1))
        remaining = 0

    # Compute P&L
    buy_adj = entry_price * (1 + cost)
    total_sell = sum(pe[0] * pe[1] * (1 - cost) for pe in partial_exits)
    va_return = (total_sell / buy_adj - 1) * 100 if buy_adj > 0 else 0

    # Weighted average exit price
    total_pct = sum(pe[0] for pe in partial_exits)
    wavg_exit = sum(pe[0] * pe[1] for pe in partial_exits) / total_pct if total_pct > 0 else entry_price

    # Primary exit reason
    reasons = set(pe[2] for pe in partial_exits)
    if "hard_stop" in reasons or "gap_stop" in reasons:
        primary = "hard_stop"
    elif "breakeven_stop" in reasons:
        primary = "breakeven_stop"
    elif "tp1" in reasons and "tp2" in reasons:
        primary = "tp1+tp2"
    elif "tp1" in reasons:
        primary = "tp1_only"
    elif "day3_stop" in reasons and len(reasons) == 2:
        primary = "day3_stop"
    else:
        primary = "time_expiry"

    return {
        "va_return_pct": va_return,
        "va_exit_reason": primary,
        "tp1_hit": tp1_hit,
        "tp2_hit": tp2_hit,
        "day3_stopped": day3_stopped,
        "hard_stopped": hard_stopped,
        "n_partial_exits": len(partial_exits),
        "wavg_exit_price": wavg_exit,
        "R_dollar": R,
        "stop_init": stop,
        "tp1_price": tp1,
        "tp2_price": tp2,
        "ambiguous_days": ambiguous_days,
    }


def simulate_all_va_daily(trades_c0, price_lookup, td_strs, td_idx, cost_bps=20):
    """Simulate Version A on all C0 trades using daily bars."""
    results = []
    for _, trade in trades_c0.iterrows():
        va = simulate_va_trade(
            ticker=trade["ticker"],
            entry_date=trade["entry_date"],
            entry_price=trade["entry_price"],
            atr_pct=trade["entry_atr_pct"],
            price_lookup=price_lookup,
            td_strs=td_strs,
            td_idx=td_idx,
            cost_bps=cost_bps,
        )
        if va is None:
            continue
        row = {
            "ticker": trade["ticker"],
            "signal_date": trade["signal_date"],
            "entry_date": trade["entry_date"],
            "entry_price": trade["entry_price"],
            "atr_pct": trade["entry_atr_pct"],
            "c0_exit_price": trade["exit_price"],
            "c0_return_pct": trade["return_pct"],
            "c0_pnl": trade["pnl"],
        }
        row.update(va)
        results.append(row)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Portfolio-level backtest with Version A exits
# ---------------------------------------------------------------------------
def run_va_portfolio_backtest(prebuilt, cost_bps=20, gap_limit=0.08):
    """Portfolio backtest with Version A partial-exit rules."""
    pl = prebuilt["price_lookup"]
    il = prebuilt["indicator_lookup"]
    sbd = prebuilt["signals_by_date"]
    td_strs = prebuilt["td_strs"]
    td_idx = prebuilt["td_idx"]
    warmup_str = prebuilt["warmup_str"]

    open_positions = []
    completed_trades = []
    cash = INITIAL_CAPITAL
    base_cost = cost_bps / 10_000.0
    daily_equity = []
    open_tickers = set()

    for day_str in td_strs:
        if day_str <= warmup_str:
            daily_equity.append({"date": day_str, "equity": INITIAL_CAPITAL, "n_pos": 0})
            continue

        # --- DAILY POSITION MONITORING ---
        still_open = []
        for pos in open_positions:
            ticker = pos["ticker"]
            bar = pl.get((day_str, ticker))
            if not bar or bar["Open"] <= 0:
                still_open.append(pos)
                continue

            O, H, L, C = bar["Open"], bar["High"], bar["Low"], bar["Close"]
            pos["days_held"] = pos.get("days_held", 0) + 1
            pos["max_close"] = max(pos.get("max_close", pos["entry_price"]), C)

            entry_p = pos["entry_price"]
            R = pos["R_dollar"]
            remaining = pos["remaining_pct"]
            stop_price = pos["stop_price"]

            closed_out = False

            # Scheduled expiry check (day 7+1: exit at Open)
            if day_str >= pos["exit_date"]:
                if remaining > 0.001:
                    proceeds = pos["shares"] * remaining * O * (1 - base_cost)
                    pos["total_proceeds"] += proceeds
                    cash += proceeds
                    pos["remaining_pct"] = 0
                    pos["final_reason"] = pos.get("final_reason", "time_expiry")
                closed_out = True

            if not closed_out:
                # 1. Gap-down
                if O <= stop_price and remaining > 0.001:
                    proceeds = pos["shares"] * remaining * O * (1 - base_cost)
                    pos["total_proceeds"] += proceeds
                    cash += proceeds
                    pos["remaining_pct"] = 0
                    pos["final_reason"] = "hard_stop"
                    closed_out = True

                # 2. Hard stop
                if not closed_out and L <= stop_price and remaining > 0.001:
                    proceeds = pos["shares"] * remaining * stop_price * (1 - base_cost)
                    pos["total_proceeds"] += proceeds
                    cash += proceeds
                    pos["remaining_pct"] = 0
                    pos["final_reason"] = "hard_stop"
                    closed_out = True

                # 3. TP1
                if not closed_out and not pos["tp1_hit"] and H >= pos["tp1_price"] and remaining > 0.001:
                    sell_pct = remaining * TP1_SELL_PCT
                    proceeds = pos["shares"] * sell_pct * pos["tp1_price"] * (1 - base_cost)
                    pos["total_proceeds"] += proceeds
                    cash += proceeds
                    pos["remaining_pct"] -= sell_pct
                    remaining = pos["remaining_pct"]
                    pos["tp1_hit"] = True
                    pos["stop_price"] = entry_p  # breakeven

                # 4. TP2
                if not closed_out and pos["tp1_hit"] and not pos["tp2_hit"] and H >= pos["tp2_price"] and remaining > 0.001:
                    sell_pct = remaining * TP2_SELL_PCT
                    proceeds = pos["shares"] * sell_pct * pos["tp2_price"] * (1 - base_cost)
                    pos["total_proceeds"] += proceeds
                    cash += proceeds
                    pos["remaining_pct"] -= sell_pct
                    remaining = pos["remaining_pct"]
                    pos["tp2_hit"] = True

                # Breakeven stop after TP1
                if not closed_out and pos["tp1_hit"] and remaining > 0.001:
                    if L <= pos["stop_price"]:
                        proceeds = pos["shares"] * remaining * pos["stop_price"] * (1 - base_cost)
                        pos["total_proceeds"] += proceeds
                        cash += proceeds
                        pos["remaining_pct"] = 0
                        pos["final_reason"] = "breakeven_stop"
                        closed_out = True

                # 5. Day3 time stop
                if not closed_out and pos["days_held"] == DAY3_CHECK_DAY and remaining > 0.001:
                    r_ret = (C - entry_p) / R if R > 0 else 0
                    if r_ret < DAY3_MIN_R and C <= pos["max_close"] * 0.999:
                        sell_pct = remaining * DAY3_SELL_PCT
                        proceeds = pos["shares"] * sell_pct * C * (1 - base_cost)
                        pos["total_proceeds"] += proceeds
                        cash += proceeds
                        pos["remaining_pct"] -= sell_pct
                        pos["day3_stopped"] = True

            if pos["remaining_pct"] <= 0.001:
                pnl = pos["total_proceeds"] - pos["dollars"]
                ret = pnl / pos["dollars"] if pos["dollars"] > 0 else 0
                completed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": day_str,
                    "entry_price": entry_p,
                    "pnl": pnl,
                    "return_pct": ret * 100,
                    "exit_reason": pos.get("final_reason", "mixed"),
                    "tp1_hit": pos["tp1_hit"],
                    "tp2_hit": pos["tp2_hit"],
                    "day3_stopped": pos.get("day3_stopped", False),
                    "signal_date": pos.get("signal_date", ""),
                })
                open_tickers.discard(ticker)
            else:
                still_open.append(pos)

        open_positions = still_open

        # --- ENTRIES ---
        idx = td_idx.get(day_str)
        if idx is not None and idx > 0:
            prev_day = td_strs[idx - 1]
            signals = sbd.get(prev_day, [])[:TOP_K]

            valid = []
            for t, entry_atr in signals:
                if t in open_tickers:
                    continue
                bar = pl.get((day_str, t))
                if not bar or bar["Open"] <= 0:
                    continue
                ind = il.get((prev_day, t), {})
                gap = ind.get("gap", 0) or 0
                if abs(gap) > gap_limit:
                    continue
                valid.append((t, bar["Open"], entry_atr))

            if valid and cash > 100:
                per_stock = cash / len(valid)
                if per_stock > 50:
                    exit_idx = idx + BEST_HOLD + 1  # +1 because we exit at Open of day after hold
                    exit_date = td_strs[exit_idx] if exit_idx < len(td_strs) else td_strs[-1]
                    for ticker, op, entry_atr in valid:
                        buy_cost = per_stock * (1 + base_cost)
                        if buy_cost > cash:
                            buy_cost = cash
                        shares = (buy_cost / (1 + base_cost)) / op
                        R = op * entry_atr / 100.0
                        if R <= 0:
                            R = op * 0.05
                        open_positions.append({
                            "ticker": ticker, "entry_date": day_str,
                            "entry_price": op, "shares": shares,
                            "exit_date": exit_date, "dollars": buy_cost,
                            "entry_atr_pct": entry_atr,
                            "signal_date": prev_day,
                            "R_dollar": R,
                            "stop_price": op - STOP_ATR_MULT * R,
                            "tp1_price": op + TP1_ATR_MULT * R,
                            "tp2_price": op + TP2_ATR_MULT * R,
                            "tp1_hit": False, "tp2_hit": False,
                            "day3_stopped": False,
                            "remaining_pct": 1.0,
                            "total_proceeds": 0.0,
                            "days_held": 0,
                            "max_close": op,
                            "final_reason": "time_expiry",
                        })
                        cash -= buy_cost
                        open_tickers.add(ticker)

        # MTM
        pos_val = sum(
            pos["shares"] * pos["remaining_pct"] *
            pl.get((day_str, pos["ticker"]), {}).get("Close", pos["entry_price"])
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
# 15-min data fetching from Polygon
# ---------------------------------------------------------------------------
def fetch_15min_bars(ticker, start_date, end_date, api_key):
    """Fetch 15-min bars from Polygon API for one ticker/period. Returns DataFrame."""
    cache_file = INTRADAY_CACHE / f"{ticker}_{start_date}_{end_date}.csv"
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["datetime"])

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/15/minute/"
        f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TraderApp/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return pd.DataFrame()

    results = data.get("results", [])
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        dt = pd.Timestamp(r["t"], unit="ms", tz="US/Eastern").tz_localize(None)
        rows.append({
            "datetime": dt,
            "Open": r["o"], "High": r["h"], "Low": r["l"], "Close": r["c"],
            "Volume": r.get("v", 0),
        })

    df = pd.DataFrame(rows)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df


def fetch_15min_sample(trades_va, api_key, max_trades=200):
    """Fetch 15-min data for a stratified sample of trades."""
    INTRADAY_CACHE.mkdir(parents=True, exist_ok=True)

    # Stratified sample by year
    trades_va = trades_va.copy()
    trades_va["entry_year"] = pd.to_datetime(trades_va["entry_date"]).dt.year

    sample_rows = []
    for year in sorted(trades_va["entry_year"].unique()):
        yr_trades = trades_va[trades_va["entry_year"] == year]
        # Prioritize ambiguous trades
        amb = yr_trades[yr_trades["ambiguous_days"] > 0]
        non_amb = yr_trades[yr_trades["ambiguous_days"] == 0]
        per_year = max_trades // len(trades_va["entry_year"].unique())
        n_amb = min(len(amb), per_year // 2)
        n_rest = min(len(non_amb), per_year - n_amb)
        if n_amb > 0:
            sample_rows.append(amb.sample(n=n_amb, random_state=42))
        if n_rest > 0:
            sample_rows.append(non_amb.sample(n=n_rest, random_state=42))

    if not sample_rows:
        return pd.DataFrame(), {}
    sample = pd.concat(sample_rows).head(max_trades)
    log(f"  Fetching 15-min data for {len(sample)} trades...")

    intraday_data = {}
    fetched = 0
    for _, trade in sample.iterrows():
        ticker = trade["ticker"]
        entry = trade["entry_date"]
        # Need data from entry to entry+hold+2 (extra buffer)
        entry_ts = pd.Timestamp(entry)
        end_ts = entry_ts + pd.Timedelta(days=12)  # calendar days buffer
        end_str = end_ts.strftime("%Y-%m-%d")

        bars = fetch_15min_bars(ticker, entry, end_str, api_key)
        if not bars.empty:
            key = (ticker, entry)
            intraday_data[key] = bars
            fetched += 1

        time.sleep(12.5)  # rate limit: 5 calls/min

        if fetched % 20 == 0:
            log(f"    Fetched {fetched}/{len(sample)}...")

    log(f"  Done: {fetched} successful fetches")
    return sample, intraday_data


# ---------------------------------------------------------------------------
# 15-min Version A simulation
# ---------------------------------------------------------------------------
def simulate_va_trade_15min(trade, bars_15min, td_strs, td_idx, cost_bps=20):
    """Simulate Version A on 15-min bars for one trade."""
    entry_price = trade["entry_price"]
    atr_pct = trade["atr_pct"]
    entry_date = trade["entry_date"]
    ticker = trade["ticker"]

    R = entry_price * atr_pct / 100.0
    if R <= 0:
        R = entry_price * 0.05

    stop = entry_price - STOP_ATR_MULT * R
    tp1 = entry_price + TP1_ATR_MULT * R
    tp2 = entry_price + TP2_ATR_MULT * R
    cost = cost_bps / 10_000.0

    remaining = 1.0
    stop_price = stop
    tp1_hit = False
    tp2_hit = False
    day3_stopped = False
    hard_stopped = False
    partial_exits = []

    entry_idx = td_idx.get(entry_date, -1)
    if entry_idx < 0 or bars_15min.empty:
        return None

    # Group 15-min bars by trading day
    bars_15min = bars_15min.copy()
    bars_15min["trade_date"] = bars_15min["datetime"].dt.normalize().dt.strftime("%Y-%m-%d")

    max_close_15m = entry_price

    for d in range(1, BEST_HOLD + 1):
        if remaining <= 0.001:
            break
        day_idx = entry_idx + d
        if day_idx >= len(td_strs):
            break
        day_str = td_strs[day_idx]

        day_bars = bars_15min[bars_15min["trade_date"] == day_str].sort_values("datetime")
        if day_bars.empty:
            continue

        for _, bar in day_bars.iterrows():
            if remaining <= 0.001:
                break
            O, H, L, C = bar["Open"], bar["High"], bar["Low"], bar["Close"]

            # Gap-down
            if O <= stop_price and remaining > 0.001:
                partial_exits.append((remaining, O, "gap_stop", d))
                hard_stopped = True
                remaining = 0
                break

            # Hard stop
            if L <= stop_price and remaining > 0.001:
                partial_exits.append((remaining, stop_price, "hard_stop", d))
                hard_stopped = True
                remaining = 0
                break

            # TP1
            if not tp1_hit and H >= tp1 and remaining > 0.001:
                sell_pct = remaining * TP1_SELL_PCT
                partial_exits.append((sell_pct, tp1, "tp1", d))
                remaining -= sell_pct
                tp1_hit = True
                stop_price = entry_price

            # TP2
            if tp1_hit and not tp2_hit and H >= tp2 and remaining > 0.001:
                sell_pct = remaining * TP2_SELL_PCT
                partial_exits.append((sell_pct, tp2, "tp2", d))
                remaining -= sell_pct
                tp2_hit = True

            # Breakeven stop
            if tp1_hit and remaining > 0.001 and L <= stop_price:
                partial_exits.append((remaining, stop_price, "breakeven_stop", d))
                remaining = 0
                break

            max_close_15m = max(max_close_15m, C)

        # Day3 time stop (at end of day 3)
        if d == DAY3_CHECK_DAY and remaining > 0.001:
            if not day_bars.empty:
                last_close = day_bars.iloc[-1]["Close"]
                r_ret = (last_close - entry_price) / R if R > 0 else 0
                if r_ret < DAY3_MIN_R and last_close <= max_close_15m * 0.999:
                    sell_pct = remaining * DAY3_SELL_PCT
                    partial_exits.append((sell_pct, last_close, "day3_stop", d))
                    remaining -= sell_pct
                    day3_stopped = True

    # Expiry
    if remaining > 0.001:
        exit_idx = entry_idx + BEST_HOLD + 1
        if exit_idx < len(td_strs):
            exit_day = td_strs[exit_idx]
            exit_bars = bars_15min[bars_15min["trade_date"] == exit_day].sort_values("datetime")
            if not exit_bars.empty:
                exit_p = exit_bars.iloc[0]["Open"]
            else:
                exit_p = entry_price
        else:
            exit_p = entry_price
        partial_exits.append((remaining, exit_p, "time_expiry", BEST_HOLD + 1))

    # P&L
    buy_adj = entry_price * (1 + cost)
    total_sell = sum(pe[0] * pe[1] * (1 - cost) for pe in partial_exits)
    va_return = (total_sell / buy_adj - 1) * 100 if buy_adj > 0 else 0

    reasons = set(pe[2] for pe in partial_exits)
    if "hard_stop" in reasons or "gap_stop" in reasons:
        primary = "hard_stop"
    elif "breakeven_stop" in reasons:
        primary = "breakeven_stop"
    elif "tp1" in reasons and "tp2" in reasons:
        primary = "tp1+tp2"
    elif "tp1" in reasons:
        primary = "tp1_only"
    elif "day3_stop" in reasons:
        primary = "day3_stop"
    else:
        primary = "time_expiry"

    return {
        "va_15min_return_pct": va_return,
        "va_15min_reason": primary,
        "tp1_hit_15m": tp1_hit,
        "tp2_hit_15m": tp2_hit,
    }


# ---------------------------------------------------------------------------
# Comparison & Reporting
# ---------------------------------------------------------------------------
def generate_report(trades_va, c0_eq, va_eq, trades_va_15min=None):
    """Generate full comparison report."""
    lines = []

    # --- Aggregate comparison ---
    mc0 = compute_metrics(c0_eq["equity"], "C0")
    mva = compute_metrics(va_eq["equity"], "Version A")

    lines.append("=" * 110)
    lines.append("VERSION A EXIT STRATEGY TEST: C0 (fixed 7d) vs Version A (stop + staged TP + time stop)")
    lines.append(f"  Stop: entry - {STOP_ATR_MULT}×ATR | TP1: +{TP1_ATR_MULT}R (sell 50%) | "
                 f"TP2: +{TP2_ATR_MULT}R (sell 25%) | Day3 stop (<0.5R, no new high) | 7d expiry")
    lines.append("=" * 110)

    lines.append("")
    lines.append("PORTFOLIO-LEVEL COMPARISON (same entries, different exits):")
    lines.append(f"{'Metric':>25s} {'C0':>12s} {'Version A':>12s} {'Delta':>12s}")
    lines.append("-" * 65)
    for metric, fmt in [("cagr", ".2f"), ("sharpe", ".3f"), ("sortino", ".3f"),
                        ("max_dd", ".2f"), ("calmar", ".3f"), ("total", ".2f")]:
        v0, va = mc0[metric], mva[metric]
        delta = va - v0
        sfx = "%" if metric in ("cagr", "max_dd", "total") else ""
        lines.append(f"{metric.upper():>25s} {v0:>+11{fmt}}{sfx} {va:>+11{fmt}}{sfx} {delta:>+11{fmt}}{sfx}")

    # Trade-level stats
    lines.append("")
    lines.append("TRADE-LEVEL COMPARISON:")
    n = len(trades_va)
    lines.append(f"  Total trades: {n}")
    lines.append(f"  {'':>25s} {'C0':>12s} {'Version A':>12s} {'Delta':>12s}")
    lines.append(f"  {'-'*65}")

    c0_avg = trades_va["c0_return_pct"].mean()
    va_avg = trades_va["va_return_pct"].mean()
    lines.append(f"  {'Avg Return%':>25s} {c0_avg:>+11.2f}% {va_avg:>+11.2f}% {va_avg-c0_avg:>+11.2f}%")

    c0_med = trades_va["c0_return_pct"].median()
    va_med = trades_va["va_return_pct"].median()
    lines.append(f"  {'Median Return%':>25s} {c0_med:>+11.2f}% {va_med:>+11.2f}% {va_med-c0_med:>+11.2f}%")

    c0_wr = (trades_va["c0_return_pct"] > 0).mean() * 100
    va_wr = (trades_va["va_return_pct"] > 0).mean() * 100
    lines.append(f"  {'Win Rate':>25s} {c0_wr:>11.1f}% {va_wr:>11.1f}% {va_wr-c0_wr:>+11.1f}%")

    c0_max = trades_va["c0_return_pct"].max()
    va_max = trades_va["va_return_pct"].max()
    lines.append(f"  {'Max Single Win':>25s} {c0_max:>+11.1f}% {va_max:>+11.1f}%")

    c0_min = trades_va["c0_return_pct"].min()
    va_min = trades_va["va_return_pct"].min()
    lines.append(f"  {'Max Single Loss':>25s} {c0_min:>+11.1f}% {va_min:>+11.1f}%")

    c0_std = trades_va["c0_return_pct"].std()
    va_std = trades_va["va_return_pct"].std()
    lines.append(f"  {'Std Dev':>25s} {c0_std:>11.2f}% {va_std:>11.2f}%")

    # How many trades improved/hurt
    improved = (trades_va["va_return_pct"] > trades_va["c0_return_pct"]).sum()
    hurt = (trades_va["va_return_pct"] < trades_va["c0_return_pct"]).sum()
    same = n - improved - hurt
    lines.append(f"\n  Trades improved by VA: {improved} ({improved/n*100:.1f}%)")
    lines.append(f"  Trades hurt by VA:     {hurt} ({hurt/n*100:.1f}%)")
    lines.append(f"  Trades unchanged:      {same} ({same/n*100:.1f}%)")

    # Exit reason distribution
    lines.append("")
    lines.append("EXIT REASON DISTRIBUTION (Version A):")
    reason_counts = trades_va["va_exit_reason"].value_counts()
    for reason, count in reason_counts.items():
        pct = count / n * 100
        avg_ret = trades_va[trades_va["va_exit_reason"] == reason]["va_return_pct"].mean()
        lines.append(f"  {reason:>20s}: {count:>5d} ({pct:>5.1f}%)  avg_ret={avg_ret:>+.2f}%")

    # TP stats
    tp1_count = trades_va["tp1_hit"].sum()
    tp2_count = trades_va["tp2_hit"].sum()
    stop_count = trades_va["hard_stopped"].sum()
    d3_count = trades_va["day3_stopped"].sum()
    lines.append(f"\n  TP1 triggered: {tp1_count} ({tp1_count/n*100:.1f}%)")
    lines.append(f"  TP2 triggered: {tp2_count} ({tp2_count/n*100:.1f}%)")
    lines.append(f"  Hard stop hit: {stop_count} ({stop_count/n*100:.1f}%)")
    lines.append(f"  Day3 time stop: {d3_count} ({d3_count/n*100:.1f}%)")

    # Annual breakdown
    lines.append("")
    lines.append("ANNUAL BREAKDOWN:")
    trades_va_copy = trades_va.copy()
    trades_va_copy["year"] = pd.to_datetime(trades_va_copy["entry_date"]).dt.year
    for year in sorted(trades_va_copy["year"].unique()):
        yr = trades_va_copy[trades_va_copy["year"] == year]
        c0_pnl = yr["c0_pnl"].sum() if "c0_pnl" in yr.columns else 0
        va_pnl_approx = (yr["va_return_pct"] / 100 * yr["entry_price"]).sum()
        c0_wr_yr = (yr["c0_return_pct"] > 0).mean() * 100
        va_wr_yr = (yr["va_return_pct"] > 0).mean() * 100
        lines.append(
            f"  {year}: N={len(yr):>4d}  C0 avg={yr['c0_return_pct'].mean():>+.2f}% WR={c0_wr_yr:.0f}%  "
            f"VA avg={yr['va_return_pct'].mean():>+.2f}% WR={va_wr_yr:.0f}%"
        )

    # Ambiguous days
    total_amb = trades_va["ambiguous_days"].sum()
    trades_with_amb = (trades_va["ambiguous_days"] > 0).sum()
    lines.append(f"\n  Ambiguous days (stop+TP in range): {total_amb} across {trades_with_amb} trades")

    # 15-min validation
    if trades_va_15min is not None and not trades_va_15min.empty:
        lines.append("")
        lines.append("=" * 80)
        lines.append("15-MIN INTRADAY VALIDATION")
        lines.append("=" * 80)
        matched = trades_va_15min.dropna(subset=["va_return_pct", "va_15min_return_pct"])
        if not matched.empty:
            n_matched = len(matched)
            reason_match = (matched["va_exit_reason"] == matched["va_15min_reason"]).sum()
            pnl_diff = (matched["va_15min_return_pct"] - matched["va_return_pct"]).abs()
            lines.append(f"  Sample size: {n_matched}")
            lines.append(f"  Matching exit reason: {reason_match} ({reason_match/n_matched*100:.1f}%)")
            lines.append(f"  Avg |PnL difference|: {pnl_diff.mean():.3f}%")
            lines.append(f"  Max |PnL difference|: {pnl_diff.max():.3f}%")
            lines.append(f"  Correlation: {matched['va_return_pct'].corr(matched['va_15min_return_pct']):.4f}")

    txt = "\n".join(lines)
    return txt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Version A Exit Strategy Test")
    parser.add_argument("--polygon-key", default="FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1")
    parser.add_argument("--max-15min-trades", type=int, default=200)
    parser.add_argument("--skip-15min", action="store_true", help="Skip 15-min fetch")
    args = parser.parse_args()

    log("=" * 70)
    log("Version A Exit Strategy Test")
    log(f"  Stop: -{STOP_ATR_MULT}×ATR | TP1: +{TP1_ATR_MULT}R sell 50% | "
        f"TP2: +{TP2_ATR_MULT}R sell 25% | Day3 stop | 7d expiry")
    log("=" * 70)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # [1] Load data
    log("\n[1/7] Loading OHLCV data...")
    df = load_ohlcv(RAW_OHLCV_PATH)

    log("[2/7] Computing indicators...")
    df = compute_indicators(df)
    mask = apply_layer1(df)

    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date="2021-01-01", end_date=END_DATE)
    trading_days = list(schedule.index.normalize())
    warmup_end = pd.Timestamp(START_DATE)

    log(f"  Layer 1 passers: {mask.sum():,}")

    # [2] Build lookups
    log("\n[3/7] Building lookups...")
    prebuilt = build_lookups(df, mask, trading_days, warmup_end, TOP_K)

    # [3] C0 baseline
    log("\n[4/7] Running C0 baseline backtest...")
    trades_c0, eq_c0 = run_c0_backtest(prebuilt, cost_bps=20, gap_limit=0.08)
    log(f"  C0 trades: {len(trades_c0)}")
    mc0 = compute_metrics(eq_c0["equity"], "C0")
    log(f"  C0: CAGR={mc0['cagr']:+.2f}%, Sharpe={mc0['sharpe']:.3f}, MaxDD={mc0['max_dd']:+.2f}%")

    # [4] Version A trade-by-trade simulation (daily)
    log("\n[5/7] Simulating Version A on daily OHLCV...")
    trades_va = simulate_all_va_daily(
        trades_c0, prebuilt["price_lookup"],
        prebuilt["td_strs"], prebuilt["td_idx"], cost_bps=20,
    )
    log(f"  VA trades simulated: {len(trades_va)}")
    va_avg = trades_va["va_return_pct"].mean()
    c0_avg = trades_va["c0_return_pct"].mean()
    log(f"  Avg return: C0={c0_avg:+.2f}% → VA={va_avg:+.2f}%")
    log(f"  Hard stopped: {trades_va['hard_stopped'].sum()} | "
        f"TP1 hit: {trades_va['tp1_hit'].sum()} | TP2 hit: {trades_va['tp2_hit'].sum()}")

    # Save trade-by-trade comparison
    trades_va.to_csv(RESULT_DIR / "trades_c0_vs_va.csv", index=False)

    # [5] Version A portfolio backtest
    log("\n[6/7] Running Version A portfolio backtest...")
    trades_va_port, eq_va = run_va_portfolio_backtest(prebuilt, cost_bps=20, gap_limit=0.08)
    mva = compute_metrics(eq_va["equity"], "VA")
    log(f"  VA portfolio: CAGR={mva['cagr']:+.2f}%, Sharpe={mva['sharpe']:.3f}, MaxDD={mva['max_dd']:+.2f}%")

    # [6] 15-min validation
    trades_va_15min = None
    if not args.skip_15min:
        log("\n[7/7] Fetching 15-min intraday data...")
        sample_df, intraday_data = fetch_15min_sample(trades_va, args.polygon_key, args.max_15min_trades)
        if not sample_df.empty and intraday_data:
            log(f"  Simulating Version A on 15-min bars for {len(intraday_data)} trades...")
            results_15min = []
            for _, trade in sample_df.iterrows():
                key = (trade["ticker"], trade["entry_date"])
                if key not in intraday_data:
                    continue
                va15 = simulate_va_trade_15min(
                    trade, intraday_data[key],
                    prebuilt["td_strs"], prebuilt["td_idx"], cost_bps=20,
                )
                if va15:
                    row = trade.to_dict()
                    row.update(va15)
                    results_15min.append(row)
            if results_15min:
                trades_va_15min = pd.DataFrame(results_15min)
                trades_va_15min.to_csv(RESULT_DIR / "trades_15min_validation.csv", index=False)
                log(f"  15-min validated: {len(trades_va_15min)} trades")
    else:
        log("\n[7/7] Skipping 15-min fetch (--skip-15min)")

    # [7] Report
    log("\nGenerating comparison report...")
    report = generate_report(trades_va, eq_c0, eq_va, trades_va_15min)
    print(report)
    (RESULT_DIR / "version_a_results.txt").write_text(report)

    # Equity curve plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(eq_c0.index, eq_c0["equity"].values, label="C0 (fixed 7d)", linewidth=1.5)
    ax.plot(eq_va.index, eq_va["equity"].values, label="Version A", linewidth=1.5)
    ax.set_title("C0 vs Version A — Portfolio Equity")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "equity_c0_vs_va.png", dpi=150)
    plt.close(fig)

    log(f"\nSaved: {RESULT_DIR / 'version_a_results.txt'}")
    log("Done!")


if __name__ == "__main__":
    main()
