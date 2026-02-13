#!/usr/bin/env python
"""Layer 1 HMM Integration Tests — Can HMM reduce drawdowns without killing the right tail?

Test 1: HMM Risk Budgeting (adjust TopK & position size by p_crisis regime)
Test 2: HMM Exit Tightening (shorten holds / add trailing stops in high-crisis)
Test 3: Winner Extension C0/C1/C2 (early exit losers, extend winners — no HMM)
Test 4: HMM Regime Slice (validate HMM identifies bad periods)
Test 5: Walk-Forward with Purge (3-fold parameter robustness)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

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
RESULT_DIR = Path("D:/trade/result/layer1_hmm")
SPY_CACHE_PATH = Path("D:/trade/result/minervini_news_backtest/spy_daily.csv")
SPY_EXTENDED_PATH = RESULT_DIR / "spy_extended.csv"

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

# HMM parameters (matching hmm_bridge.py canonical values)
HMM_N_STATES = 3
HMM_TRAIN_WINDOW = 1000
HMM_RETRAIN_FREQ = 21
HMM_EMA_SPAN = 4
HMM_GAMMA = 2
HMM_CRISIS_ENTER = 0.70
HMM_CRISIS_EXIT = 0.40
HMM_CONFIRM_DAYS = 2
HMM_COOLDOWN_DAYS = 3

POLYGON_API_KEY = ""


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data loading & indicators
# ---------------------------------------------------------------------------
def load_ohlcv(path: Path) -> pd.DataFrame:
    log("  Reading parquet...")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index(["date", "ticker"]).sort_index()
    df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
    log(f"  {len(df):,} rows, {df.index.get_level_values('ticker').nunique()} tickers")
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
# SPY data (extended to 2017 for HMM training window)
# ---------------------------------------------------------------------------
def fetch_spy_polygon(api_key: str, start: str, end: str) -> pd.DataFrame:
    """Fetch SPY daily bars from Polygon API."""
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start}/{end}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "TraderApp/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
        results = data.get("results", [])
    rows = []
    for bar in results:
        ts = bar.get("t", 0)
        dt = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        rows.append({
            "date": dt, "Open": bar.get("o", 0), "High": bar.get("h", 0),
            "Low": bar.get("l", 0), "Close": bar.get("c", 0), "Volume": bar.get("v", 0),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def load_spy_extended(api_key: str) -> pd.DataFrame:
    """Load or create extended SPY cache (2017+). Uses yfinance for historical data."""
    if SPY_EXTENDED_PATH.exists():
        df = pd.read_csv(SPY_EXTENDED_PATH, parse_dates=["date"], index_col="date")
        if len(df) > 1500:
            log(f"  SPY extended from cache: {len(df)} bars")
            return df

    log("  Building extended SPY cache...")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    existing = pd.read_csv(SPY_CACHE_PATH, parse_dates=["date"], index_col="date")
    existing.index = existing.index.normalize()

    # Fetch older data via yfinance (no date restrictions)
    log("  Fetching SPY 2017-2021 via yfinance...")
    import yfinance as yf
    ticker = yf.Ticker("SPY")
    hist = ticker.history(start="2017-01-01", end="2021-02-11", auto_adjust=True)
    older = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    older.index = older.index.normalize().tz_localize(None)
    older.index.name = "date"

    # Combine, deduplicate, sort
    combined = pd.concat([older, existing])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    combined.to_csv(SPY_EXTENDED_PATH)
    log(f"  SPY extended: {len(combined)} bars ({combined.index[0].date()} → {combined.index[-1].date()})")
    return combined


# ---------------------------------------------------------------------------
# HMM regime computation (walk-forward, no lookahead)
# Adapted from etf_rotation_v3_monthly_hmm.py:compute_hmm_regime_series()
# ---------------------------------------------------------------------------
@dataclass
class HmmState:
    p_crisis_history: List[float] = field(default_factory=list)
    crisis_mode: bool = False
    crisis_confirm_days: int = 0
    safe_confirm_days: int = 0
    cooldown_remaining: int = 0


def compute_hmm_series(
    spy_df: pd.DataFrame,
    train_window: int = HMM_TRAIN_WINDOW,
    retrain_freq: int = HMM_RETRAIN_FREQ,
    ema_span: int = HMM_EMA_SPAN,
    crisis_enter: float = HMM_CRISIS_ENTER,
    crisis_exit: float = HMM_CRISIS_EXIT,
) -> pd.DataFrame:
    """Walk-forward HMM p_crisis computation. Causal — no lookahead."""
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    close = spy_df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    vol_10d = log_ret.rolling(10, min_periods=10).std()

    features = pd.DataFrame({
        "log_ret": log_ret, "vol_10d": vol_10d,
    }, index=spy_df.index).dropna()

    n = len(features)
    result = {
        "p_crisis": np.full(n, np.nan),
        "p_crisis_smooth": np.full(n, np.nan),
        "risk_gate": np.ones(n),
        "crisis_mode": np.zeros(n, dtype=bool),
        "hmm_state": ["SAFE"] * n,
    }

    min_train = max(train_window, 100)
    if n < min_train + 10:
        log(f"  Warning: HMM needs {min_train} days, only have {n}")
        return pd.DataFrame(result, index=features.index)

    state = HmmState()
    last_train_idx = -999
    model = None
    scaler = None
    crisis_state_idx = 0
    label_map = {}

    for i in range(min_train, n):
        # Retrain periodically
        if i - last_train_idx >= retrain_freq or model is None:
            train_start = max(0, i - train_window)
            X_raw = features.iloc[train_start:i][["log_ret", "vol_10d"]].values

            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)

            model = GaussianHMM(
                n_components=HMM_N_STATES,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                tol=1e-4,
            )
            try:
                model.fit(X)
            except Exception:
                continue

            # Label states by volatility (ascending: SAFE, MID, CRISIS)
            means_orig = scaler.inverse_transform(model.means_)
            state_order = np.argsort(means_orig[:, 1])
            crisis_state_idx = int(state_order[-1])
            label_map = {
                int(state_order[0]): "SAFE",
                int(state_order[1]): "MID",
                int(state_order[2]): "CRISIS",
            }
            last_train_idx = i

        if model is None or scaler is None:
            continue

        # Single-day prediction (no backward pass for single observation)
        x_now = scaler.transform(features.iloc[i:i+1][["log_ret", "vol_10d"]].values)
        try:
            posteriors = model.predict_proba(x_now)
            p_crisis = float(posteriors[0, crisis_state_idx])
        except Exception:
            p_crisis = 0.0

        # EMA smoothing
        state.p_crisis_history.append(p_crisis)
        if len(state.p_crisis_history) > 30:
            state.p_crisis_history = state.p_crisis_history[-30:]

        ema_s = pd.Series(state.p_crisis_history)
        p_smooth = float(ema_s.ewm(span=ema_span, adjust=False).mean().iloc[-1])

        # Risk gate
        rg = (1.0 - p_smooth) ** HMM_GAMMA
        if rg < 0.05:
            rg = 0.0

        # Hysteresis
        if state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1

        if not state.crisis_mode:
            if p_smooth >= crisis_enter and state.cooldown_remaining == 0:
                state.crisis_confirm_days += 1
                if state.crisis_confirm_days >= HMM_CONFIRM_DAYS:
                    state.crisis_mode = True
                    state.crisis_confirm_days = 0
                    state.safe_confirm_days = 0
            else:
                state.crisis_confirm_days = 0
        else:
            if p_smooth <= crisis_exit:
                state.safe_confirm_days += 1
                if state.safe_confirm_days >= HMM_CONFIRM_DAYS:
                    state.crisis_mode = False
                    state.safe_confirm_days = 0
                    state.cooldown_remaining = HMM_COOLDOWN_DAYS
            else:
                state.safe_confirm_days = 0

        try:
            sp = model.predict(x_now)
            sl = label_map.get(int(sp[0]), "SAFE")
        except Exception:
            sl = "SAFE"

        result["p_crisis"][i] = p_crisis
        result["p_crisis_smooth"][i] = p_smooth
        result["risk_gate"][i] = rg
        result["crisis_mode"][i] = state.crisis_mode
        result["hmm_state"][i] = sl

    df_out = pd.DataFrame(result, index=features.index)
    return df_out


# ---------------------------------------------------------------------------
# Metrics helper
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


def top_n_pnl_share(trades_df: pd.DataFrame, n: int = 10) -> float:
    """Return fraction of total P&L from top N trades."""
    if trades_df.empty or "pnl" not in trades_df.columns:
        return 0.0
    total = trades_df["pnl"].sum()
    if total <= 0:
        return 0.0
    top_n = trades_df.nlargest(n, "pnl")["pnl"].sum()
    return top_n / total * 100


def worst_year_pnl(trades_df: pd.DataFrame) -> tuple[int, float]:
    if trades_df.empty:
        return (0, 0.0)
    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year
    yearly = trades_df.groupby("year")["pnl"].sum()
    worst_yr = yearly.idxmin()
    return (int(worst_yr), float(yearly.loc[worst_yr]))


# ---------------------------------------------------------------------------
# Lookup building — done ONCE, reused across all backtest calls
# ---------------------------------------------------------------------------
def build_lookups(
    df: pd.DataFrame,
    mask: pd.Series,
    trading_days: list[pd.Timestamp],
    warmup_end: pd.Timestamp,
    top_k: int = 8,
) -> dict:
    """Build price/indicator/signal lookups once (vectorized, no iterrows)."""
    log("  Building lookups (vectorized)...")

    dates = df.index.get_level_values("date")
    tickers = df.index.get_level_values("ticker")
    date_strs = dates.strftime("%Y-%m-%d")
    keys = list(zip(date_strs, tickers))

    # Price lookup
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    price_lookup = {k: {"Open": o[i], "High": h[i], "Low": l[i], "Close": c[i]}
                    for i, k in enumerate(keys)}

    # Indicator lookup (only gap is used in backtest)
    atr_vals = df["atr_pct"].values
    gap_vals = df["opening_gap"].values
    indicator_lookup = {k: {"atr_pct": atr_vals[i], "gap": gap_vals[i]}
                        for i, k in enumerate(keys)}

    td_strs = [td.strftime("%Y-%m-%d") for td in trading_days]
    td_idx = {ds: i for i, ds in enumerate(td_strs)}
    warmup_str = warmup_end.strftime("%Y-%m-%d")

    # Pre-compute daily signals
    passed = df[mask].copy()
    available_dates = set(df.index.get_level_values("date").unique())
    signals_by_date: dict[str, list[tuple[str, float]]] = {}
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


def build_hmm_lookup(hmm_series: pd.DataFrame | None) -> dict[str, float]:
    """Build HMM p_crisis lookup from series."""
    hmm_lookup: dict[str, float] = {}
    if hmm_series is not None:
        for dt, row in hmm_series.iterrows():
            ds = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            val = row.get("p_crisis_smooth", 0.0) if isinstance(row, pd.Series) else 0.0
            if pd.notna(val):
                hmm_lookup[ds] = float(val)
    return hmm_lookup


# ---------------------------------------------------------------------------
# Backtest Engine V2 — supports HMM budgeting + dynamic exits
# ---------------------------------------------------------------------------
def run_backtest_v2(
    df: pd.DataFrame,
    mask: pd.Series,
    trading_days: list[pd.Timestamp],
    warmup_end: pd.Timestamp,
    hmm_series: pd.DataFrame | None = None,
    hold_days: int = 7,
    top_k: int = 8,
    cost_bps: int = 20,
    gap_limit: float = 0.08,
    # HMM budgeting
    hmm_budget: bool = False,
    budget_config: dict | None = None,
    # Early exit (C1)
    early_exit: bool = False,
    early_exit_day: int = 3,
    early_exit_min_r: float = 0.5,
    # Winner extension (C2)
    winner_extend: bool = False,
    extend_r_threshold: float = 2.0,
    extend_max_days: int = 15,
    trailing_stop_pct: float = 0.10,
    # HMM exit tightening
    hmm_exit: bool = False,
    risk_off_hold: int = 5,
    risk_off_trailing: float = 0.08,
    # Period restriction (for walk-forward)
    backtest_start: str | None = None,
    backtest_end: str | None = None,
    # Pre-built lookups (pass to avoid rebuilding)
    prebuilt: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extended backtest engine with HMM budgeting, dynamic exits, winner extension."""

    bc = budget_config or {
        "thresh_low": 0.35, "thresh_high": 0.60,
        "low_topk": 8, "mid_topk": 5, "high_topk": 2,
        "low_size": 1.0, "mid_size": 0.6, "high_size": 0.3,
    }

    if prebuilt:
        price_lookup = prebuilt["price_lookup"]
        indicator_lookup = prebuilt["indicator_lookup"]
        signals_by_date = prebuilt["signals_by_date"]
        td_strs = prebuilt["td_strs"]
        td_idx = prebuilt["td_idx"]
        warmup_str = prebuilt["warmup_str"]
    else:
        pb = build_lookups(df, mask, trading_days, warmup_end, top_k)
        price_lookup = pb["price_lookup"]
        indicator_lookup = pb["indicator_lookup"]
        signals_by_date = pb["signals_by_date"]
        td_strs = pb["td_strs"]
        td_idx = pb["td_idx"]
        warmup_str = pb["warmup_str"]

    hmm_lookup = build_hmm_lookup(hmm_series)

    # Determine period
    bt_start = backtest_start or "1900-01-01"
    bt_end = backtest_end or "2099-12-31"

    # Run
    open_positions: list[dict] = []
    completed_trades: list[dict] = []
    cash = INITIAL_CAPITAL
    base_cost = cost_bps / 10_000.0
    daily_equity: list[dict] = []
    open_tickers: set[str] = set()

    for day_str in td_strs:
        if day_str < bt_start or day_str > bt_end:
            continue
        if day_str <= warmup_str:
            daily_equity.append({"date": day_str, "equity": INITIAL_CAPITAL, "n_pos": 0})
            continue

        # --- DAILY POSITION MONITORING (for dynamic exits) ---
        still_open = []
        for pos in open_positions:
            ticker = pos["ticker"]
            bar = price_lookup.get((day_str, ticker))
            if not bar:
                still_open.append(pos)
                continue

            # Update tracking
            pos["days_held"] = pos.get("days_held", 0) + 1
            pos["max_high"] = max(pos.get("max_high", pos["entry_price"]), bar["High"])

            current_close = bar["Close"]
            entry_p = pos["entry_price"]
            r_unit = entry_p * pos.get("entry_atr_pct", 5.0) / 100.0
            r_multiple = (current_close - entry_p) / r_unit if r_unit > 0 else 0

            should_exit = False
            exit_reason = "hold"

            # Check time-based exit
            if day_str >= pos["exit_date"]:
                # On scheduled exit day, check for winner extension
                if winner_extend and not pos.get("extended", False):
                    if r_multiple >= extend_r_threshold:
                        # Extend position
                        cur_idx = td_idx.get(day_str, 0)
                        new_exit_idx = cur_idx + (extend_max_days - hold_days)
                        new_exit = td_strs[new_exit_idx] if new_exit_idx < len(td_strs) else td_strs[-1]
                        pos["exit_date"] = new_exit
                        pos["extended"] = True
                        pos["trailing_stop"] = pos["max_high"] * (1 - trailing_stop_pct)
                        still_open.append(pos)
                        continue
                should_exit = True
                exit_reason = "time"

            # Early exit check (C1): after early_exit_day days
            if early_exit and not should_exit and pos["days_held"] >= early_exit_day:
                new_high = pos["max_high"] > entry_p * 1.005
                if r_multiple < early_exit_min_r and not new_high:
                    should_exit = True
                    exit_reason = "early_stop"

            # Trailing stop for extended positions (C2)
            if pos.get("extended", False) and not should_exit:
                pos["trailing_stop"] = max(
                    pos.get("trailing_stop", 0),
                    pos["max_high"] * (1 - trailing_stop_pct)
                )
                if current_close <= pos["trailing_stop"]:
                    should_exit = True
                    exit_reason = "trailing"

            # HMM exit tightening: if in high-crisis regime
            if hmm_exit and not should_exit:
                p_crisis_today = hmm_lookup.get(day_str, 0.0)
                if p_crisis_today >= bc["thresh_high"]:
                    # Shortened hold
                    if pos["days_held"] >= risk_off_hold:
                        should_exit = True
                        exit_reason = "hmm_early"
                    # Tighter trailing
                    crisis_stop = pos["max_high"] * (1 - risk_off_trailing)
                    if current_close <= crisis_stop:
                        should_exit = True
                        exit_reason = "hmm_trailing"

            if should_exit:
                # Exit at next day's Open (or today's Open if it's the exit_date)
                # For simplicity, exit at today's close price approximation = next day open
                # Actually, use today's bar Open if this is the exit date, else next day Open
                if exit_reason == "time" and bar["Open"] > 0:
                    exit_price = bar["Open"]
                else:
                    # For dynamic exits triggered at Close, exit at next day's Open
                    next_idx = td_idx.get(day_str, 0) + 1
                    if next_idx < len(td_strs):
                        next_bar = price_lookup.get((td_strs[next_idx], ticker))
                        exit_price = next_bar["Open"] if next_bar and next_bar["Open"] > 0 else current_close
                    else:
                        exit_price = current_close

                cost = base_cost * pos.get("cost_mult", 1.0)
                proceeds = pos["shares"] * exit_price * (1 - cost)
                pnl = proceeds - pos["dollars"]
                ret = pnl / pos["dollars"] if pos["dollars"] > 0 else 0
                completed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": day_str,
                    "entry_price": entry_p,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "return_pct": ret * 100,
                    "exit_reason": exit_reason,
                    "days_held": pos["days_held"],
                    "extended": pos.get("extended", False),
                    "signal_date": pos.get("signal_date", ""),
                })
                cash += proceeds
                open_tickers.discard(ticker)
            else:
                still_open.append(pos)
        open_positions = still_open

        # --- ENTRIES ---
        idx = td_idx.get(day_str)
        if idx is not None and idx > 0:
            prev_day = td_strs[idx - 1]
            signals = signals_by_date.get(prev_day, [])

            # HMM budgeting: determine effective TopK and size multiplier
            eff_topk = top_k
            size_mult = 1.0
            if hmm_budget and hmm_lookup:
                p_c = hmm_lookup.get(prev_day, 0.0)
                if p_c >= bc["thresh_high"]:
                    eff_topk = bc["high_topk"]
                    size_mult = bc["high_size"]
                elif p_c >= bc["thresh_low"]:
                    eff_topk = bc["mid_topk"]
                    size_mult = bc["mid_size"]
                else:
                    eff_topk = bc["low_topk"]
                    size_mult = bc["low_size"]

            # Trim signals to effective TopK
            signals = signals[:eff_topk]

            valid = []
            for item in signals:
                t, entry_atr = item
                if t in open_tickers:
                    continue
                bar = price_lookup.get((day_str, t))
                if not bar or bar["Open"] <= 0:
                    continue
                ind = indicator_lookup.get((prev_day, t), {})
                gap = ind.get("gap", 0) or 0
                if abs(gap) > gap_limit:
                    continue
                valid.append((t, bar["Open"], entry_atr))

            if valid and cash > 100:
                alloc = cash * size_mult
                per_stock = alloc / len(valid)
                if per_stock > 50:
                    exit_idx = idx + hold_days
                    exit_date = td_strs[exit_idx] if exit_idx < len(td_strs) else td_strs[-1]

                    # HMM exit: adjust hold for high-crisis
                    actual_hold = hold_days
                    if hmm_exit:
                        p_c = hmm_lookup.get(prev_day, 0.0)
                        if p_c >= bc["thresh_high"]:
                            actual_hold = risk_off_hold
                            exit_idx = idx + actual_hold
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
                            "cost_mult": 1.0, "days_held": 0,
                            "entry_atr_pct": entry_atr,
                            "max_high": op,
                            "extended": False,
                            "trailing_stop": 0.0,
                            "signal_date": prev_day,
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
# Test 1: HMM Risk Budgeting
# ---------------------------------------------------------------------------
def test1_hmm_budgeting(
    df, mask, trading_days, warmup_end, hmm_series, prebuilt=None,
) -> str:
    log("  Test 1: HMM Risk Budgeting...")
    scenarios = [
        ("Baseline (no HMM)", False, None),
        ("Budget-Conservative", True, {
            "thresh_low": 0.35, "thresh_high": 0.60,
            "low_topk": 8, "mid_topk": 5, "high_topk": 2,
            "low_size": 1.0, "mid_size": 0.6, "high_size": 0.3,
        }),
        ("Budget-Moderate", True, {
            "thresh_low": 0.35, "thresh_high": 0.60,
            "low_topk": 8, "mid_topk": 6, "high_topk": 3,
            "low_size": 1.0, "mid_size": 0.8, "high_size": 0.5,
        }),
        ("Budget-Aggressive", True, {
            "thresh_low": 0.35, "thresh_high": 0.60,
            "low_topk": 8, "mid_topk": 4, "high_topk": 1,
            "low_size": 1.0, "mid_size": 0.5, "high_size": 0.2,
        }),
    ]

    lines = [
        "=" * 110,
        "TEST 1: HMM Risk Budgeting (Hold=7d, Gap<8%, Cost=20bps)",
        "  p_crisis thresholds: LOW < 0.35, MID 0.35-0.60, HIGH >= 0.60",
        "=" * 110,
        "",
        f"{'Scenario':>30s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MaxDD':>8s} {'Calmar':>8s} "
        f"{'Total':>10s} {'Trades':>7s} {'Top10%':>8s} {'WorstYr':>12s}",
        "-" * 110,
    ]

    eq_curves = {}
    for name, use_hmm, bc in scenarios:
        trades, eq = run_backtest_v2(
            df, mask, trading_days, warmup_end,
            hmm_series=hmm_series if use_hmm else None,
            hold_days=BEST_HOLD, top_k=TOP_K, cost_bps=20, gap_limit=0.08,
            hmm_budget=use_hmm, budget_config=bc,
            prebuilt=prebuilt,
        )
        m = compute_metrics(eq["equity"], name)
        t10 = top_n_pnl_share(trades, 10)
        wy, wp = worst_year_pnl(trades)
        nt = len(trades)
        lines.append(
            f"{name:>30s} {m['cagr']:>+7.2f}% {m['sharpe']:>7.3f} {m['sortino']:>7.3f} "
            f"{m['max_dd']:>+7.2f}% {m['calmar']:>7.3f} {m['total']:>+9.2f}% "
            f"{nt:>7d} {t10:>7.1f}% {wy}:${wp:>+,.0f}"
        )
        eq_curves[name] = eq["equity"]

    # Plot equity curves
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, eq_s in eq_curves.items():
        ax.plot(eq_s.index, eq_s.values, label=name, linewidth=1.5)
    ax.set_title("Test 1: HMM Risk Budgeting — Equity Curves")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "test1_equity.png", dpi=150)
    plt.close(fig)

    txt = "\n".join(lines)
    (RESULT_DIR / "test1_hmm_budgeting.txt").write_text(txt)
    print(txt)
    return txt


# ---------------------------------------------------------------------------
# Test 2: HMM Exit Tightening Only
# ---------------------------------------------------------------------------
def test2_hmm_exit(
    df, mask, trading_days, warmup_end, hmm_series, prebuilt=None,
) -> str:
    log("  Test 2: HMM Exit Tightening...")
    scenarios = [
        ("Baseline (fixed 7d)", dict(
            hmm_exit=False, early_exit=False, winner_extend=False,
        )),
        ("HMM-Exit (5d+trail@high, extend@low)", dict(
            hmm_exit=True, risk_off_hold=5, risk_off_trailing=0.08,
            early_exit=False,
            winner_extend=True, extend_r_threshold=2.0,
            extend_max_days=12, trailing_stop_pct=0.10,
        )),
    ]

    lines = [
        "=" * 120,
        "TEST 2: HMM Exit Tightening Only (TopK=8, Gap<8%, Cost=20bps — entry unchanged)",
        "  HIGH p_crisis: hold→5d + 8% trailing | LOW p_crisis: winners +2R → extend to 12d",
        "=" * 120,
        "",
        f"{'Scenario':>45s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Calmar':>8s} "
        f"{'Trades':>7s} {'Top5%':>7s} {'Top10%':>8s} {'MaxTrade':>10s} {'WorstYr':>12s}",
        "-" * 120,
    ]

    for name, kw in scenarios:
        trades, eq = run_backtest_v2(
            df, mask, trading_days, warmup_end,
            hmm_series=hmm_series,
            hold_days=BEST_HOLD, top_k=TOP_K, cost_bps=20, gap_limit=0.08,
            prebuilt=prebuilt,
            **kw,
        )
        m = compute_metrics(eq["equity"], name)
        t5 = top_n_pnl_share(trades, 5)
        t10 = top_n_pnl_share(trades, 10)
        max_trade = trades["return_pct"].max() if not trades.empty else 0
        wy, wp = worst_year_pnl(trades)
        nt = len(trades)
        lines.append(
            f"{name:>45s} {m['cagr']:>+7.2f}% {m['sharpe']:>7.3f} {m['max_dd']:>+7.2f}% {m['calmar']:>7.3f} "
            f"{nt:>7d} {t5:>6.1f}% {t10:>7.1f}% {max_trade:>+9.1f}% {wy}:${wp:>+,.0f}"
        )

        # Exit reason breakdown
        if not trades.empty and "exit_reason" in trades.columns:
            reason_counts = trades["exit_reason"].value_counts()
            reason_str = ", ".join(f"{r}={c}" for r, c in reason_counts.items())
            lines.append(f"{'':>45s} Exit reasons: {reason_str}")

    txt = "\n".join(lines)
    (RESULT_DIR / "test2_hmm_exit.txt").write_text(txt)
    print(txt)
    return txt


# ---------------------------------------------------------------------------
# Test 3: Winner Extension C0/C1/C2
# ---------------------------------------------------------------------------
def test3_winner_extension(
    df, mask, trading_days, warmup_end, prebuilt=None,
) -> str:
    log("  Test 3: Winner Extension (C0/C1/C2)...")
    scenarios = [
        ("C0: Fixed 7d exit", dict(
            early_exit=False, winner_extend=False,
        )),
        ("C1: Day3 stop (<0.5R & no new high)", dict(
            early_exit=True, early_exit_day=3, early_exit_min_r=0.5,
            winner_extend=False,
        )),
        ("C2: C1 + extend winners (+2R → 15d trail)", dict(
            early_exit=True, early_exit_day=3, early_exit_min_r=0.5,
            winner_extend=True, extend_r_threshold=2.0,
            extend_max_days=15, trailing_stop_pct=0.10,
        )),
    ]

    lines = [
        "=" * 130,
        "TEST 3: Winner Extension — C0 (baseline) / C1 (early stop) / C2 (early stop + winner extend)",
        "  All: Gap<8%, 20bps, TopK=8, NO HMM",
        "  C1: Day 3 — if return < +0.5R AND no new high → exit",
        "  C2: C1 + day 7 — if return >= +2R → extend to 15d with 10% trailing stop",
        "=" * 130,
        "",
        f"{'Scenario':>45s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Calmar':>8s} "
        f"{'Trades':>7s} {'Top10%':>8s} {'MaxTrade':>10s} {'AvgRet':>8s} {'MedRet':>8s} "
        f"{'WorstYr':>12s}",
        "-" * 130,
    ]

    eq_curves = {}
    all_trades = {}
    for name, kw in scenarios:
        trades, eq = run_backtest_v2(
            df, mask, trading_days, warmup_end,
            hmm_series=None,
            hold_days=BEST_HOLD, top_k=TOP_K, cost_bps=20, gap_limit=0.08,
            prebuilt=prebuilt,
            **kw,
        )
        all_trades[name] = trades
        m = compute_metrics(eq["equity"], name)
        t10 = top_n_pnl_share(trades, 10)
        max_trade = trades["return_pct"].max() if not trades.empty else 0
        avg_ret = trades["return_pct"].mean() if not trades.empty else 0
        med_ret = trades["return_pct"].median() if not trades.empty else 0
        wy, wp = worst_year_pnl(trades)
        nt = len(trades)
        lines.append(
            f"{name:>45s} {m['cagr']:>+7.2f}% {m['sharpe']:>7.3f} {m['max_dd']:>+7.2f}% {m['calmar']:>7.3f} "
            f"{nt:>7d} {t10:>7.1f}% {max_trade:>+9.1f}% {avg_ret:>+7.2f}% {med_ret:>+7.2f}% "
            f"{wy}:${wp:>+,.0f}"
        )

        # Exit reason breakdown
        if not trades.empty and "exit_reason" in trades.columns:
            reason_counts = trades["exit_reason"].value_counts()
            reason_str = ", ".join(f"{r}={c}" for r, c in reason_counts.items())
            lines.append(f"{'':>45s} Exit reasons: {reason_str}")
            # Extended positions stats
            if "extended" in trades.columns:
                ext = trades[trades["extended"] == True]
                if len(ext) > 0:
                    lines.append(
                        f"{'':>45s} Extended: {len(ext)} trades, avg_ret={ext['return_pct'].mean():+.2f}%, "
                        f"avg_days={ext['days_held'].mean():.1f}"
                    )

        eq_curves[name] = eq["equity"]

    # Top-decile analysis (reuse cached trades, no re-run)
    lines.append("")
    lines.append("  Top-Decile Trade Analysis:")
    for name, kw in scenarios:
        trades = all_trades[name]
        if not trades.empty:
            p90 = trades["return_pct"].quantile(0.90)
            top_decile = trades[trades["return_pct"] >= p90]
            lines.append(
                f"    {name:>42s}: top-decile N={len(top_decile)}, "
                f"avg={top_decile['return_pct'].mean():+.2f}%, "
                f"max={top_decile['return_pct'].max():+.2f}%"
            )

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    for name, eq_s in eq_curves.items():
        ax.plot(eq_s.index, eq_s.values, label=name, linewidth=1.5)
    ax.set_title("Test 3: Winner Extension — C0 vs C1 vs C2")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "test3_equity.png", dpi=150)
    plt.close(fig)

    txt = "\n".join(lines)
    (RESULT_DIR / "test3_winner_extension.txt").write_text(txt)
    print(txt)
    return txt


# ---------------------------------------------------------------------------
# Test 4: HMM Regime Slice
# ---------------------------------------------------------------------------
def test4_regime_slice(
    df, mask, trading_days, warmup_end, hmm_series, prebuilt=None,
) -> str:
    log("  Test 4: HMM Regime Slice...")

    # Run baseline backtest to get trades
    trades, eq = run_backtest_v2(
        df, mask, trading_days, warmup_end,
        hmm_series=None,
        hold_days=BEST_HOLD, top_k=TOP_K, cost_bps=20, gap_limit=0.08,
        prebuilt=prebuilt,
    )
    if trades.empty:
        return "No trades"

    # Look up p_crisis for each trade's signal date
    hmm_lookup = {}
    if hmm_series is not None:
        for dt, row in hmm_series.iterrows():
            ds = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            val = row.get("p_crisis_smooth", np.nan) if isinstance(row, pd.Series) else np.nan
            if pd.notna(val):
                hmm_lookup[ds] = float(val)

    trades["p_crisis"] = trades["signal_date"].map(hmm_lookup)
    trades_with_hmm = trades.dropna(subset=["p_crisis"])

    if len(trades_with_hmm) < 20:
        return "Too few trades with HMM data"

    # Split by percentile: Low 30%, Mid 40%, High 30%
    p30 = trades_with_hmm["p_crisis"].quantile(0.30)
    p70 = trades_with_hmm["p_crisis"].quantile(0.70)

    buckets = {
        f"LOW (p<{p30:.3f}, bottom 30%)": trades_with_hmm[trades_with_hmm["p_crisis"] < p30],
        f"MID ({p30:.3f}≤p<{p70:.3f}, middle 40%)": trades_with_hmm[
            (trades_with_hmm["p_crisis"] >= p30) & (trades_with_hmm["p_crisis"] < p70)
        ],
        f"HIGH (p≥{p70:.3f}, top 30%)": trades_with_hmm[trades_with_hmm["p_crisis"] >= p70],
    }

    lines = [
        "=" * 110,
        "TEST 4: HMM Regime Slice — Baseline trades split by p_crisis at signal date",
        f"  Total trades with HMM data: {len(trades_with_hmm)}",
        f"  p_crisis percentiles: P30={p30:.4f}, P70={p70:.4f}",
        "=" * 110,
        "",
        f"{'Bucket':>45s} {'N':>5s} {'Avg%':>8s} {'Med%':>8s} {'Win%':>7s} {'Std%':>8s} "
        f"{'Skew':>7s} {'Kurt':>7s} {'Top5 PnL%':>10s}",
        "-" * 110,
    ]

    for name, subset in buckets.items():
        n = len(subset)
        if n < 5:
            lines.append(f"{name:>45s} {n:>5d}  (too few)")
            continue
        arr = subset["return_pct"].values
        avg = arr.mean()
        med = np.median(arr)
        wr = (arr > 0).sum() / n * 100
        std = arr.std()
        skew = float(pd.Series(arr).skew())
        kurt = float(pd.Series(arr).kurtosis())
        total_pnl = subset["pnl"].sum()
        top5_pnl = subset.nlargest(5, "pnl")["pnl"].sum()
        top5_share = top5_pnl / total_pnl * 100 if total_pnl > 0 else 0

        lines.append(
            f"{name:>45s} {n:>5d} {avg:>+7.2f}% {med:>+7.2f}% {wr:>6.1f}% {std:>7.2f}% "
            f"{skew:>+6.1f} {kurt:>+6.1f} {top5_share:>9.1f}%"
        )

    # Annual breakdown by regime
    lines.append("")
    lines.append("  Annual Breakdown by Regime:")
    for name, subset in buckets.items():
        if len(subset) < 5:
            continue
        subset = subset.copy()
        subset["year"] = pd.to_datetime(subset["entry_date"]).dt.year
        for yr, grp in subset.groupby("year"):
            lines.append(
                f"    {name[:25]:>25s} {yr}: N={len(grp):>3d}, avg={grp['return_pct'].mean():>+6.2f}%, "
                f"WR={((grp['return_pct']>0).sum()/len(grp)*100):>5.1f}%, PnL=${grp['pnl'].sum():>+,.0f}"
            )

    # Plot return distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for idx, (name, subset) in enumerate(buckets.items()):
        ax = axes[idx]
        if len(subset) >= 5:
            ax.hist(subset["return_pct"].clip(-50, 100), bins=40, alpha=0.7, edgecolor="black")
            ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
            ax.axvline(x=subset["return_pct"].mean(), color="blue", linestyle="-", linewidth=2)
        short_name = name.split(",")[0]
        ax.set_title(short_name, fontsize=10)
        ax.set_xlabel("Return (%)")
        if idx == 0:
            ax.set_ylabel("Count")
    fig.suptitle("Test 4: Return Distribution by HMM Regime", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "test4_regime_returns.png", dpi=150)
    plt.close(fig)

    txt = "\n".join(lines)
    (RESULT_DIR / "test4_regime_slice.txt").write_text(txt)
    print(txt)
    return txt


# ---------------------------------------------------------------------------
# Test 5: Walk-Forward with Purge
# ---------------------------------------------------------------------------
def test5_walk_forward(
    df, mask, trading_days, warmup_end, spy_df, prebuilt=None,
) -> str:
    log("  Test 5: Walk-Forward with Purge...")

    PURGE_DAYS = 15  # trading days

    folds = [
        {"name": "Fold 1", "train_start": "2022-03-01", "train_end": "2023-12-31",
         "test_start": "2024-01-22", "test_end": "2024-12-31"},
        {"name": "Fold 2", "train_start": "2022-03-01", "train_end": "2024-12-31",
         "test_start": "2025-01-22", "test_end": "2025-12-31"},
        {"name": "Fold 3", "train_start": "2023-01-01", "train_end": "2024-12-31",
         "test_start": "2025-01-22", "test_end": "2025-12-31"},
    ]

    # Parameter grid
    hmm_thresholds = [
        (0.25, 0.50), (0.35, 0.60), (0.45, 0.70),
    ]
    exit_configs = [
        ("C0", dict(early_exit=False, winner_extend=False)),
        ("C1", dict(early_exit=True, early_exit_day=3, early_exit_min_r=0.5, winner_extend=False)),
        ("C2", dict(early_exit=True, early_exit_day=3, early_exit_min_r=0.5,
                     winner_extend=True, extend_r_threshold=2.0,
                     extend_max_days=15, trailing_stop_pct=0.10)),
    ]

    lines = [
        "=" * 130,
        "TEST 5: Walk-Forward with Purge (15 trading day gap)",
        "  Parameter grid: 3 HMM thresholds × 3 exit strategies = 9 combos per fold",
        "=" * 130,
    ]

    fold_results = []

    for fold in folds:
        lines.append(f"\n--- {fold['name']}: Train {fold['train_start']}→{fold['train_end']}, "
                      f"Test {fold['test_start']}→{fold['test_end']} ---")

        # Compute HMM series for this fold's SPY data
        # Train HMM only on data available at train_end
        spy_up_to_test = spy_df[spy_df.index <= fold["test_end"]]
        hmm_s = compute_hmm_series(spy_up_to_test)

        best_train_sharpe = -999
        best_combo = None
        grid_results = []

        for thresh_low, thresh_high in hmm_thresholds:
            for exit_name, exit_kw in exit_configs:
                combo_name = f"HMM({thresh_low}/{thresh_high})+{exit_name}"

                # --- Train ---
                bc = {
                    "thresh_low": thresh_low, "thresh_high": thresh_high,
                    "low_topk": 8, "mid_topk": 5, "high_topk": 2,
                    "low_size": 1.0, "mid_size": 0.6, "high_size": 0.3,
                }
                trades_train, eq_train = run_backtest_v2(
                    df, mask, trading_days, warmup_end,
                    hmm_series=hmm_s,
                    hold_days=BEST_HOLD, top_k=TOP_K, cost_bps=20, gap_limit=0.08,
                    hmm_budget=True, budget_config=bc,
                    backtest_start=fold["train_start"], backtest_end=fold["train_end"],
                    prebuilt=prebuilt,
                    **exit_kw,
                )
                eq_train_valid = eq_train[eq_train["equity"] != INITIAL_CAPITAL]
                m_train = compute_metrics(eq_train_valid["equity"] if not eq_train_valid.empty else pd.Series([INITIAL_CAPITAL]))

                grid_results.append({
                    "combo": combo_name, "train_sharpe": m_train["sharpe"],
                    "thresh_low": thresh_low, "thresh_high": thresh_high,
                    "exit_name": exit_name, "exit_kw": exit_kw, "bc": bc,
                })

                if m_train["sharpe"] > best_train_sharpe:
                    best_train_sharpe = m_train["sharpe"]
                    best_combo = grid_results[-1]

        if best_combo is None:
            lines.append("  No valid combo found!")
            continue

        # --- Test with best combo ---
        trades_test, eq_test = run_backtest_v2(
            df, mask, trading_days, warmup_end,
            hmm_series=hmm_s,
            hold_days=BEST_HOLD, top_k=TOP_K, cost_bps=20, gap_limit=0.08,
            hmm_budget=True, budget_config=best_combo["bc"],
            backtest_start=fold["test_start"], backtest_end=fold["test_end"],
            prebuilt=prebuilt,
            **best_combo["exit_kw"],
        )
        eq_test_valid = eq_test[eq_test["equity"] != INITIAL_CAPITAL]
        m_test = compute_metrics(eq_test_valid["equity"] if not eq_test_valid.empty else pd.Series([INITIAL_CAPITAL]))

        # Also run baseline on test period (no HMM)
        trades_base, eq_base = run_backtest_v2(
            df, mask, trading_days, warmup_end,
            hmm_series=None,
            hold_days=BEST_HOLD, top_k=TOP_K, cost_bps=20, gap_limit=0.08,
            backtest_start=fold["test_start"], backtest_end=fold["test_end"],
            prebuilt=prebuilt,
        )
        eq_base_valid = eq_base[eq_base["equity"] != INITIAL_CAPITAL]
        m_base = compute_metrics(eq_base_valid["equity"] if not eq_base_valid.empty else pd.Series([INITIAL_CAPITAL]))

        lines.append(f"  Best combo (train): {best_combo['combo']} (Sharpe={best_train_sharpe:.3f})")
        lines.append(f"  Train grid results:")
        for gr in sorted(grid_results, key=lambda x: -x["train_sharpe"]):
            lines.append(f"    {gr['combo']:>35s}: Sharpe={gr['train_sharpe']:>+.3f}")

        lines.append(f"\n  Test Period Results:")
        lines.append(f"    {'Scenario':>35s} {'CAGR':>8s} {'Sharpe':>8s} {'MaxDD':>8s} {'Trades':>7s}")
        lines.append(f"    {'-'*65}")
        lines.append(
            f"    {'Baseline (no HMM, C0)':>35s} {m_base['cagr']:>+7.2f}% {m_base['sharpe']:>7.3f} "
            f"{m_base['max_dd']:>+7.2f}% {len(trades_base):>7d}"
        )
        lines.append(
            f"    {best_combo['combo']:>35s} {m_test['cagr']:>+7.2f}% {m_test['sharpe']:>7.3f} "
            f"{m_test['max_dd']:>+7.2f}% {len(trades_test):>7d}"
        )

        fold_results.append({
            "fold": fold["name"],
            "best_combo": best_combo["combo"],
            "train_sharpe": best_train_sharpe,
            "test_sharpe": m_test["sharpe"],
            "test_cagr": m_test["cagr"],
            "test_maxdd": m_test["max_dd"],
            "baseline_sharpe": m_base["sharpe"],
        })

    # Summary
    lines.append("\n" + "=" * 80)
    lines.append("Walk-Forward Summary:")
    lines.append(f"  {'Fold':>8s} {'Best Combo':>35s} {'TrainSh':>8s} {'TestSh':>8s} {'BaseSh':>8s} {'TestDD':>8s}")
    lines.append(f"  {'-'*75}")
    for fr in fold_results:
        lines.append(
            f"  {fr['fold']:>8s} {fr['best_combo']:>35s} {fr['train_sharpe']:>+7.3f} "
            f"{fr['test_sharpe']:>+7.3f} {fr['baseline_sharpe']:>+7.3f} {fr['test_maxdd']:>+7.2f}%"
        )
    if fold_results:
        worst_test = min(fr["test_sharpe"] for fr in fold_results)
        lines.append(f"\n  Worst fold test Sharpe: {worst_test:+.3f}")
        lines.append(f"  {'PASS' if worst_test > 0 else 'FAIL'}: "
                      f"{'Strategy profitable in all folds' if worst_test > 0 else 'Strategy fails in worst fold'}")

    txt = "\n".join(lines)
    (RESULT_DIR / "test5_walk_forward.txt").write_text(txt)
    print(txt)
    return txt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Layer 1 HMM Integration Tests")
    parser.add_argument("--polygon-key", default="FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--skip-to", type=int, default=0,
                        help="Skip to test N (1-5). Tests before N loaded from saved files.")
    args = parser.parse_args()

    global POLYGON_API_KEY
    POLYGON_API_KEY = args.polygon_key
    skip_to = args.skip_to

    log("=" * 70)
    log("Layer 1 HMM Integration Tests")
    log(f"Period: {args.start} -> {args.end}")
    if skip_to > 0:
        log(f"Skipping to Test {skip_to}")
    log("=" * 70)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # [1/7] Load data
    log("")
    log("[1/7] Loading OHLCV data...")
    df = load_ohlcv(RAW_OHLCV_PATH)

    # [2/7] Compute indicators
    log("[2/7] Computing indicators...")
    df = compute_indicators(df)
    mask = apply_layer1(df)

    # Trading days
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date="2021-01-01", end_date=args.end)
    trading_days = list(schedule.index.normalize())
    warmup_end = pd.Timestamp(args.start)

    log(f"  Layer 1 passers: {mask.sum():,}")
    log(f"  Trading days: {len(trading_days)}")

    # [3/7] Build lookups ONCE (vectorized, no iterrows)
    log("")
    log("[3/7] Building lookups & loading SPY data...")
    prebuilt = build_lookups(df, mask, trading_days, warmup_end, TOP_K)
    spy_df = load_spy_extended(POLYGON_API_KEY)

    # [4/7] Compute HMM p_crisis series
    log("")
    log("[4/7] Computing HMM p_crisis series (walk-forward)...")
    hmm_series = compute_hmm_series(spy_df)
    valid_hmm = hmm_series["p_crisis_smooth"].dropna()
    log(f"  HMM series: {len(valid_hmm)} valid days "
        f"({valid_hmm.index[0].date()} → {valid_hmm.index[-1].date()})")
    log(f"  p_crisis_smooth range: [{valid_hmm.min():.4f}, {valid_hmm.max():.4f}]")
    log(f"  Crisis mode days: {hmm_series['crisis_mode'].sum()}")

    # Save HMM series for reference
    hmm_series.to_csv(RESULT_DIR / "hmm_p_crisis_series.csv")

    all_results = []

    # [5/7] Tests 1 & 2 (HMM-dependent)
    if skip_to <= 1:
        log("")
        log("[5/7] Running Test 1 (HMM Budgeting) & Test 2 (HMM Exit)...")
        all_results.append(test1_hmm_budgeting(df, mask, trading_days, warmup_end, hmm_series, prebuilt))
        log("")
        all_results.append(test2_hmm_exit(df, mask, trading_days, warmup_end, hmm_series, prebuilt))
    else:
        for f in ["test1_hmm_budgeting.txt", "test2_hmm_exit.txt"]:
            p = RESULT_DIR / f
            if p.exists():
                all_results.append(p.read_text())
                log(f"  Loaded saved: {f}")

    # [6/7] Test 3 (Winner Extension — no HMM)
    if skip_to <= 3:
        log("")
        log("[6/7] Running Test 3 (Winner Extension C0/C1/C2)...")
        all_results.append(test3_winner_extension(df, mask, trading_days, warmup_end, prebuilt))
    else:
        p = RESULT_DIR / "test3_winner_extension.txt"
        if p.exists():
            all_results.append(p.read_text())
            log(f"  Loaded saved: test3_winner_extension.txt")

    # [7/7] Tests 4 & 5
    if skip_to <= 4:
        log("")
        log("[7/7] Running Test 4 (Regime Slice) & Test 5 (Walk-Forward)...")
        all_results.append(test4_regime_slice(df, mask, trading_days, warmup_end, hmm_series, prebuilt))
    else:
        p = RESULT_DIR / "test4_regime_slice.txt"
        if p.exists():
            all_results.append(p.read_text())
            log(f"  Loaded saved: test4_regime_slice.txt")

    log("")
    all_results.append(test5_walk_forward(df, mask, trading_days, warmup_end, spy_df, prebuilt))

    # Save combined results
    combined = "\n\n".join(all_results)
    (RESULT_DIR / "robustness_results.txt").write_text(combined)
    log(f"\nSaved: {RESULT_DIR / 'robustness_results.txt'}")
    log("Done!")


if __name__ == "__main__":
    main()
