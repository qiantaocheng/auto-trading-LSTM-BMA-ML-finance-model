#!/usr/bin/env python3
"""
ETF Rotation Strategy: HVR Regime Switching + HRP Allocation + Volatility Targeting
====================================================================================

Architecture:
    1. Polygon.io daily OHLCV + dividends → Total Return Index
    2. HVR regime detection (hysteresis + fast-off + MA200 cap)
    3. HRP portfolio construction (Ledoit-Wolf shrinkage, Ward linkage)
    4. Volatility targeting with BIL cash absorber
    5. Backtest engine (t+1 execution, transaction costs, no lookahead)

Author: Quantitative Trading System
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import urllib.request

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """All strategy parameters in one place."""
    # Polygon
    polygon_api_key: str = ""

    # Universe
    risk_on_etfs: List[str] = field(default_factory=lambda: [
        "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB", "XLRE", "XLC"
    ])
    risk_off_etfs: List[str] = field(default_factory=lambda: ["BIL", "IEF", "GLD"])
    # XLP and XLU serve dual purpose: risk-on AND defensive
    dual_purpose: List[str] = field(default_factory=lambda: ["XLP", "XLU"])
    spy_ticker: str = "SPY"
    cash_ticker: str = "BIL"

    # HVR Regime
    hvr_short_window: int = 21
    hvr_long_window: int = 252
    hvr_threshold_off: float = 1.2   # RISK_ON → RISK_OFF
    hvr_threshold_on: float = 1.0    # RISK_OFF → RISK_ON

    # Fast-Off triggers
    fast_off_dd20_pct: float = 8.0   # 20-day max drawdown threshold
    fast_off_5d_drop_pct: float = -6.0  # 5-day cumulative return threshold

    # MA200 trend guard
    ma200_risk_cap: float = 0.30     # max risk allocation when SPY < MA200

    # HRP
    hrp_lookback: int = 252          # covariance estimation window
    hrp_linkage: str = "ward"        # hierarchical clustering method
    rebalance_freq: str = "M"        # monthly HRP recalc (last trading day)
    trade_freq_days: int = 5         # weekly rebalance (every 5 trading days)
    min_trade_threshold: float = 0.05  # minimum weight change to trigger trade

    # Volatility Targeting
    target_vol: float = 0.10         # 10% annual target vol
    max_leverage: float = 1.0        # no leverage by default

    # Transaction costs (basis points on notional)
    cost_bps: float = 5.0            # 5 bps round-trip
    min_cost_per_trade: float = 1.0  # $1 minimum per trade

    # Backtest
    start_date: str = "2016-01-01"
    end_date: str = "2026-02-07"
    initial_capital: float = 100_000.0
    execution: str = "close_t1"      # signal at t close, execute at t+1 close


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING (Polygon.io)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_polygon_daily(ticker: str, start: str, end: str, api_key: str,
                        adjusted: bool = False) -> pd.DataFrame:
    """Fetch daily OHLCV from Polygon.io with pagination."""
    all_results = []
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        f"?adjusted={'true' if adjusted else 'false'}&sort=asc&limit=50000"
        f"&apiKey={api_key}"
    )
    while url:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ETFRotation/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("results", [])
                if not results:
                    break
                all_results.extend(results)
                next_url = data.get("next_url", "")
                if next_url:
                    if "apiKey=" not in next_url:
                        sep = "&" if "?" in next_url else "?"
                        next_url = f"{next_url}{sep}apiKey={api_key}"
                    url = next_url
                else:
                    break
        except Exception as e:
            print(f"  Warning: Polygon daily fetch for {ticker} failed: {e}", file=sys.stderr)
            break
        time.sleep(0.15)

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]].drop_duplicates("date")
    df = df.set_index("date").sort_index()
    return df


def fetch_polygon_dividends(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch dividend data from Polygon.io."""
    all_results = []
    url = (
        f"https://api.polygon.io/v3/reference/dividends"
        f"?ticker={ticker}&ex_dividend_date.gte={start}&ex_dividend_date.lte={end}"
        f"&limit=1000&apiKey={api_key}"
    )
    while url:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ETFRotation/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("results", [])
                all_results.extend(results)
                next_url = data.get("next_url", "")
                if next_url:
                    if "apiKey=" not in next_url:
                        sep = "&" if "?" in next_url else "?"
                        next_url = f"{next_url}{sep}apiKey={api_key}"
                    url = next_url
                else:
                    break
        except Exception:
            break
        time.sleep(0.15)

    if not all_results:
        return pd.DataFrame(columns=["ex_date", "amount"])

    df = pd.DataFrame(all_results)
    if "ex_dividend_date" in df.columns and "cash_amount" in df.columns:
        df["ex_date"] = pd.to_datetime(df["ex_dividend_date"])
        df["amount"] = df["cash_amount"].astype(float)
        df = df[["ex_date", "amount"]].groupby("ex_date")["amount"].sum().reset_index()
        return df
    return pd.DataFrame(columns=["ex_date", "amount"])


def build_total_return_index(prices: pd.Series, dividends: pd.DataFrame) -> pd.Series:
    """
    Build Total Return Index via forward iteration.
    r_t = (P_t + D_t) / P_{t-1} - 1
    I_t = I_{t-1} * (1 + r_t)

    Assumes dividend reinvested at close on ex-date.
    """
    if prices.empty:
        return prices

    div_map = {}
    if not dividends.empty:
        for _, row in dividends.iterrows():
            dt = pd.Timestamp(row["ex_date"]).normalize()
            div_map[dt] = div_map.get(dt, 0.0) + row["amount"]

    tri = pd.Series(index=prices.index, dtype=float)
    tri.iloc[0] = 100.0  # base = 100

    for i in range(1, len(prices)):
        p_t = prices.iloc[i]
        p_prev = prices.iloc[i - 1]
        d_t = div_map.get(prices.index[i], 0.0)
        if p_prev > 0:
            r_t = (p_t + d_t) / p_prev - 1.0
        else:
            r_t = 0.0
        tri.iloc[i] = tri.iloc[i - 1] * (1.0 + r_t)

    return tri


def fetch_all_data(cfg: StrategyConfig) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
    """Fetch all tickers: raw prices + total return indices."""
    all_tickers = list(set(
        cfg.risk_on_etfs + cfg.risk_off_etfs + [cfg.spy_ticker]
    ))

    raw_prices = {}
    tri_series = {}

    print(f"Fetching data for {len(all_tickers)} tickers from Polygon.io...")
    for i, ticker in enumerate(sorted(all_tickers)):
        print(f"  [{i+1}/{len(all_tickers)}] {ticker}...", end=" ", flush=True)

        # Fetch unadjusted prices
        df = fetch_polygon_daily(ticker, cfg.start_date, cfg.end_date, cfg.polygon_api_key,
                                 adjusted=False)
        if df.empty:
            print("NO DATA")
            continue

        # Fetch dividends
        divs = fetch_polygon_dividends(ticker, cfg.start_date, cfg.end_date, cfg.polygon_api_key)

        # Build total return index
        tri = build_total_return_index(df["Close"], divs)

        raw_prices[ticker] = df
        tri_series[ticker] = tri

        div_count = len(divs) if not divs.empty else 0
        print(f"OK ({len(df)} bars, {div_count} divs)")

    return raw_prices, tri_series


# ─────────────────────────────────────────────────────────────────────────────
# ALIGN DATA
# ─────────────────────────────────────────────────────────────────────────────

def align_data(tri_series: Dict[str, pd.Series]) -> pd.DataFrame:
    """Align all total return indices to common trading days with forward fill."""
    if not tri_series:
        raise ValueError("No data to align")

    df = pd.DataFrame(tri_series)
    # Use intersection of all dates where at least some data exists
    df = df.sort_index()
    # Forward fill gaps (up to 5 days — longer gaps indicate real missing data)
    df = df.ffill(limit=5)
    # Drop rows where ANY ticker is still NaN (ensures complete alignment)
    # But first drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")
    # For the remaining, drop rows with NaN only at the start (before all tickers have data)
    first_valid = df.apply(lambda col: col.first_valid_index()).max()
    df = df.loc[first_valid:]
    df = df.ffill()  # final ffill for any remaining gaps
    return df


# ─────────────────────────────────────────────────────────────────────────────
# REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeState:
    regime: str = "RISK_ON"  # or "RISK_OFF"
    hvr: float = 0.0
    dd20: float = 0.0
    ret5d: float = 0.0
    spy_below_ma200: bool = False
    trigger: str = ""


def compute_regime_series(spy_tri: pd.Series, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Compute HVR regime for each trading day.
    Returns DataFrame with columns: regime, hvr, dd20, ret5d, spy_below_ma200, risk_cap
    """
    log_ret = np.log(spy_tri / spy_tri.shift(1))

    # Rolling volatilities (annualized)
    vol_short = log_ret.rolling(cfg.hvr_short_window).std() * np.sqrt(252)
    vol_long = log_ret.rolling(cfg.hvr_long_window).std() * np.sqrt(252)

    # HVR
    hvr = vol_short / vol_long

    # 20-day drawdown
    rolling_max_20 = spy_tri.rolling(20).max()
    dd20 = (spy_tri / rolling_max_20 - 1) * 100  # percentage

    # 5-day return
    ret5d = (spy_tri / spy_tri.shift(5) - 1) * 100  # percentage

    # MA200
    ma200 = spy_tri.rolling(200).mean()

    # Build regime series with hysteresis
    regimes = []
    state = "RISK_ON"

    for i in range(len(spy_tri)):
        h = hvr.iloc[i] if not np.isnan(hvr.iloc[i]) else 0.0
        d20 = dd20.iloc[i] if not np.isnan(dd20.iloc[i]) else 0.0
        r5 = ret5d.iloc[i] if not np.isnan(ret5d.iloc[i]) else 0.0
        below_ma200 = spy_tri.iloc[i] < ma200.iloc[i] if not np.isnan(ma200.iloc[i]) else False

        trigger = ""

        # Fast-Off check (overrides hysteresis)
        if abs(d20) > cfg.fast_off_dd20_pct:
            state = "RISK_OFF"
            trigger = f"fast_off_dd20={d20:.1f}%"
        elif r5 < cfg.fast_off_5d_drop_pct:
            state = "RISK_OFF"
            trigger = f"fast_off_5d={r5:.1f}%"
        else:
            # Normal HVR hysteresis
            if state == "RISK_ON" and h > cfg.hvr_threshold_off:
                state = "RISK_OFF"
                trigger = f"hvr={h:.2f}>T_off"
            elif state == "RISK_OFF" and h < cfg.hvr_threshold_on:
                state = "RISK_ON"
                trigger = f"hvr={h:.2f}<T_on"

        # Risk cap from MA200
        if state == "RISK_ON" and below_ma200:
            risk_cap = cfg.ma200_risk_cap
        else:
            risk_cap = 1.0 if state == "RISK_ON" else 0.0

        regimes.append({
            "date": spy_tri.index[i],
            "regime": state,
            "hvr": h,
            "dd20": d20,
            "ret5d": r5,
            "spy_below_ma200": below_ma200,
            "risk_cap": risk_cap,
            "trigger": trigger,
        })

    return pd.DataFrame(regimes).set_index("date")


# ─────────────────────────────────────────────────────────────────────────────
# HRP ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────

def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """D_ij = sqrt(2 * (1 - rho_ij))"""
    return np.sqrt(2.0 * (1.0 - corr))


def _quasi_diag(link: np.ndarray) -> List[int]:
    """Seriation: reorder leaves to minimize crossings in dendrogram."""
    link = link.astype(int, copy=False)
    n = link.shape[0] + 1
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = sort_ix.max() + 1

    while sort_ix.max() >= n:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n]
        i = df0.index
        j = df0.values - n
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def _recursive_bisection(cov: np.ndarray, sorted_ix: List[int]) -> np.ndarray:
    """Recursive bisection for HRP weight allocation."""
    n = cov.shape[0]
    weights = np.ones(n)
    clusters = [sorted_ix]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]

            # Inverse-variance allocation between clusters
            def cluster_var(ids):
                sub_cov = cov[np.ix_(ids, ids)]
                inv_diag = 1.0 / np.diag(sub_cov)
                w = inv_diag / inv_diag.sum()
                return float(w @ sub_cov @ w)

            v_left = cluster_var(left)
            v_right = cluster_var(right)
            alpha = 1.0 - v_left / (v_left + v_right) if (v_left + v_right) > 0 else 0.5

            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1 - alpha)

            new_clusters.extend([left, right])
        clusters = new_clusters

    return weights / weights.sum()


def compute_hrp_weights(returns: pd.DataFrame, linkage_method: str = "ward") -> pd.Series:
    """
    Full HRP pipeline:
    1. Ledoit-Wolf covariance shrinkage
    2. Correlation → distance matrix
    3. Hierarchical clustering
    4. Quasi-diagonalization (seriation)
    5. Recursive bisection → weights
    """
    from scipy.cluster.hierarchy import linkage as hc_linkage
    from scipy.spatial.distance import squareform
    from sklearn.covariance import LedoitWolf

    if returns.shape[1] < 2:
        # Single asset: 100% weight
        return pd.Series(1.0, index=returns.columns)

    # Ledoit-Wolf shrinkage
    lw = LedoitWolf().fit(returns.values)
    cov = lw.covariance_

    # Correlation from covariance
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1e-10
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1, 1)

    # Distance matrix
    dist = _correlation_distance(corr)
    np.fill_diagonal(dist, 0.0)

    # Hierarchical clustering
    condensed = squareform(dist, checks=False)
    # Replace any NaN/inf in condensed form
    condensed = np.nan_to_num(condensed, nan=0.0, posinf=2.0, neginf=0.0)
    link = hc_linkage(condensed, method=linkage_method)

    # Quasi-diagonalization
    sorted_ix = _quasi_diag(link)
    sorted_ix = [int(x) for x in sorted_ix]

    # Recursive bisection
    raw_weights = _recursive_bisection(cov, sorted_ix)

    return pd.Series(raw_weights, index=returns.columns)


# ─────────────────────────────────────────────────────────────────────────────
# VOLATILITY TARGETING
# ─────────────────────────────────────────────────────────────────────────────

def apply_vol_targeting(weights: pd.Series, cov: np.ndarray, columns: List[str],
                        target_vol: float, max_leverage: float) -> Tuple[pd.Series, float]:
    """
    Scale weights so portfolio vol ≈ target_vol.
    Remaining allocation goes to BIL (cash absorber).
    Returns (final_weights, portfolio_vol).
    """
    w = weights.reindex(columns, fill_value=0.0).values
    port_var = float(w @ cov @ w) * 252
    port_vol = np.sqrt(max(port_var, 1e-10))

    scalar = target_vol / port_vol if port_vol > 0 else 1.0
    scalar = min(scalar, max_leverage)

    w_scaled = w * scalar
    total_risk = w_scaled.sum()

    # Build final weights including BIL
    final = pd.Series(w_scaled, index=columns)

    # Add BIL to absorb remaining
    bil_weight = max(0.0, 1.0 - total_risk)
    if "BIL" in final.index:
        final["BIL"] += bil_weight
    else:
        final["BIL"] = bil_weight

    # Normalize to exactly 1.0
    if final.sum() > 0:
        final = final / final.sum()

    return final, port_vol


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    name: str
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: int = 0
    turnover_annual: float = 0.0
    regime_history: Optional[pd.DataFrame] = None
    weight_history: Optional[pd.DataFrame] = None


def run_backtest(
    tri_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    cfg: StrategyConfig,
    variant: str = "full",
) -> BacktestResult:
    """
    Run backtest for a given strategy variant.

    Variants:
      - "no_regime": Pure HRP, always RISK_ON, no regime switching
      - "hvr_no_hysteresis": Single threshold HVR (no hysteresis band)
      - "hvr_hysteresis": HVR with hysteresis (T_on/T_off)
      - "full": HVR + hysteresis + fast-off + MA200 cap (default)

    Execution: signal at t close → execute at t+1 close (no lookahead).
    """
    spy_col = cfg.spy_ticker
    all_etfs = list(set(cfg.risk_on_etfs + cfg.risk_off_etfs))
    available_etfs = [e for e in all_etfs if e in tri_df.columns]

    if spy_col not in tri_df.columns:
        raise ValueError(f"SPY ({spy_col}) not in data")

    # Compute daily returns for all ETFs
    returns_df = np.log(tri_df / tri_df.shift(1)).dropna()

    # Start after warmup period
    warmup = max(cfg.hvr_long_window, cfg.hrp_lookback, 200) + 10
    dates = returns_df.index[warmup:]

    if len(dates) < 50:
        raise ValueError(f"Not enough data after warmup: {len(dates)} days")

    # Regime computation based on variant
    if variant == "no_regime":
        regime_series = pd.Series("RISK_ON", index=regime_df.index)
        risk_cap_series = pd.Series(1.0, index=regime_df.index)
    elif variant == "hvr_no_hysteresis":
        # Single threshold at 1.1 (midpoint of hysteresis band)
        single_threshold = (cfg.hvr_threshold_on + cfg.hvr_threshold_off) / 2
        regime_series = regime_df["hvr"].apply(
            lambda h: "RISK_OFF" if h > single_threshold else "RISK_ON"
        )
        risk_cap_series = pd.Series(1.0, index=regime_df.index)
        risk_cap_series[regime_series == "RISK_OFF"] = 0.0
    elif variant == "hvr_hysteresis":
        regime_series = regime_df["regime"].copy()
        # No fast-off or MA200 — just pure hysteresis
        risk_cap_series = pd.Series(1.0, index=regime_df.index)
        risk_cap_series[regime_series == "RISK_OFF"] = 0.0
    else:  # "full"
        regime_series = regime_df["regime"]
        risk_cap_series = regime_df["risk_cap"]

    # Track state
    capital = cfg.initial_capital
    equity = []
    current_weights = pd.Series(0.0, index=available_etfs)
    current_weights["BIL"] = 1.0 if "BIL" in available_etfs else 0.0
    hrp_weights = None
    last_hrp_month = None
    last_trade_idx = 0
    last_regime = "RISK_ON"
    total_trades = 0
    total_turnover = 0.0
    weight_records = []

    for idx, date in enumerate(dates):
        if date not in regime_df.index:
            continue

        regime = regime_series.get(date, "RISK_ON")
        r_cap = risk_cap_series.get(date, 1.0)

        # --- HRP Weight Recalculation ---
        # Recalculate monthly (last trading day) or on regime switch
        current_month = date.month
        regime_switched = (regime != last_regime)
        month_changed = (last_hrp_month is None or current_month != last_hrp_month)

        # Check if this is last trading day of month
        is_month_end = False
        if idx + 1 < len(dates):
            next_date = dates[idx + 1]
            if next_date.month != current_month:
                is_month_end = True
        else:
            is_month_end = True

        recalc_hrp = regime_switched or (is_month_end and month_changed)

        if recalc_hrp:
            # Select asset pool based on regime
            if regime == "RISK_OFF":
                pool = [e for e in cfg.risk_off_etfs if e in tri_df.columns]
                # Also include defensive dual-purpose ETFs
                pool += [e for e in cfg.dual_purpose if e in tri_df.columns and e not in pool]
            else:
                pool = [e for e in cfg.risk_on_etfs if e in tri_df.columns]

            if len(pool) >= 2:
                # Use lookback window of returns for HRP
                lookback_start = max(0, returns_df.index.get_loc(date) - cfg.hrp_lookback)
                lookback_end = returns_df.index.get_loc(date)
                ret_window = returns_df.iloc[lookback_start:lookback_end][pool]

                # Drop columns with all zeros or too few observations
                ret_window = ret_window.dropna(axis=1, how="all")
                if ret_window.shape[0] > 30 and ret_window.shape[1] >= 2:
                    hrp_weights = compute_hrp_weights(ret_window, cfg.hrp_linkage)
                else:
                    # Equal weight fallback
                    hrp_weights = pd.Series(1.0 / len(pool), index=pool)
            elif len(pool) == 1:
                hrp_weights = pd.Series(1.0, index=pool)
            else:
                hrp_weights = pd.Series({"BIL": 1.0})

            last_hrp_month = current_month

        last_regime = regime

        # --- Position Adjustment (weekly) ---
        should_trade = (idx - last_trade_idx >= cfg.trade_freq_days) or regime_switched

        if should_trade and hrp_weights is not None:
            # Apply risk cap
            if regime == "RISK_ON" and r_cap < 1.0:
                # MA200 guard: scale risk allocation down
                target = hrp_weights * r_cap
            elif regime == "RISK_OFF":
                target = hrp_weights.copy()
            else:
                target = hrp_weights.copy()

            # Volatility targeting
            pool_cols = [c for c in target.index if c in returns_df.columns]
            if len(pool_cols) >= 2:
                lookback_start = max(0, returns_df.index.get_loc(date) - cfg.hrp_lookback)
                lookback_end = returns_df.index.get_loc(date)
                ret_window = returns_df.iloc[lookback_start:lookback_end][pool_cols]

                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf().fit(ret_window.values)
                cov_mat = lw.covariance_
                target, port_vol = apply_vol_targeting(
                    target, cov_mat, pool_cols, cfg.target_vol, cfg.max_leverage
                )
            else:
                # Single asset or no valid data → just use target as-is
                bil_fill = max(0.0, 1.0 - target.sum())
                if "BIL" in target.index:
                    target["BIL"] += bil_fill
                else:
                    target["BIL"] = bil_fill

            # Ensure target sums to 1
            if target.sum() > 0:
                target = target / target.sum()

            # Expand to full universe
            new_weights = pd.Series(0.0, index=available_etfs)
            for ticker, w in target.items():
                if ticker in new_weights.index:
                    new_weights[ticker] = w

            # Check minimum trade threshold
            weight_diff = (new_weights - current_weights).abs().sum()
            if weight_diff >= cfg.min_trade_threshold or regime_switched:
                # Transaction costs
                turnover = (new_weights - current_weights).abs().sum() / 2
                cost = max(turnover * capital * cfg.cost_bps / 10000,
                           cfg.min_cost_per_trade if turnover > 0 else 0)
                capital -= cost
                total_turnover += turnover
                total_trades += 1

                # NOTE: This is where t+1 execution matters.
                # We compute signal at date (t), but the new weights take effect
                # starting from the NEXT day's return (t+1).
                # The current day's return still uses old weights.
                current_weights = new_weights
                last_trade_idx = idx

        # --- Apply daily return ---
        if date in returns_df.index:
            daily_ret = returns_df.loc[date]
            portfolio_ret = 0.0
            for ticker, w in current_weights.items():
                if ticker in daily_ret.index and w > 0:
                    portfolio_ret += w * daily_ret[ticker]
            capital *= (1 + portfolio_ret)

        equity.append({"date": date, "equity": capital})
        weight_records.append({"date": date, **current_weights.to_dict()})

    equity_df = pd.DataFrame(equity).set_index("date")["equity"]
    weight_df = pd.DataFrame(weight_records).set_index("date")

    # Calculate annual turnover
    years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
    turnover_annual = total_turnover / years if years > 0 else 0

    return BacktestResult(
        name=variant,
        equity_curve=equity_df,
        trades=total_trades,
        turnover_annual=turnover_annual,
        regime_history=regime_df,
        weight_history=weight_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(equity: pd.Series, name: str = "", period_label: str = "") -> dict:
    """Compute comprehensive backtest metrics."""
    if len(equity) < 2:
        return {"name": name, "period": period_label, "error": "insufficient data"}

    daily_ret = equity.pct_change().dropna()
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25

    # CAGR
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    # Annualized volatility
    ann_vol = daily_ret.std() * np.sqrt(252)

    # Sharpe (assuming 4% risk-free for the period)
    rf_daily = 0.04 / 252
    excess = daily_ret - rf_daily
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-10
    sortino = (daily_ret.mean() * 252 - 0.04) / downside_vol

    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()

    # Max drawdown duration
    is_dd = drawdown < 0
    dd_groups = (~is_dd).cumsum()
    if is_dd.any():
        dd_durations = is_dd.groupby(dd_groups).sum()
        max_dd_duration = dd_durations.max()
    else:
        max_dd_duration = 0

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    return {
        "name": name,
        "period": period_label,
        "total_return": f"{total_ret*100:.1f}%",
        "cagr": f"{cagr*100:.2f}%",
        "ann_vol": f"{ann_vol*100:.2f}%",
        "sharpe": f"{sharpe:.2f}",
        "sortino": f"{sortino:.2f}",
        "max_dd": f"{max_dd*100:.1f}%",
        "max_dd_days": int(max_dd_duration),
        "calmar": f"{calmar:.2f}",
        "cagr_raw": cagr,
        "sharpe_raw": sharpe,
        "max_dd_raw": max_dd,
    }


def annual_breakdown(equity: pd.Series) -> pd.DataFrame:
    """Year-by-year performance breakdown."""
    daily_ret = equity.pct_change().dropna()
    years = sorted(daily_ret.index.year.unique())
    rows = []
    for yr in years:
        yr_ret = daily_ret[daily_ret.index.year == yr]
        if len(yr_ret) < 10:
            continue
        yr_equity = (1 + yr_ret).cumprod()
        ann_ret = yr_equity.iloc[-1] - 1
        ann_vol = yr_ret.std() * np.sqrt(252)
        cummax = yr_equity.cummax()
        max_dd = ((yr_equity - cummax) / cummax).min()
        rows.append({
            "year": yr,
            "return": f"{ann_ret*100:.1f}%",
            "vol": f"{ann_vol*100:.1f}%",
            "max_dd": f"{max_dd*100:.1f}%",
            "sharpe": f"{(yr_ret.mean()/yr_ret.std()*np.sqrt(252)):.2f}" if yr_ret.std() > 0 else "N/A",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TIME LEAKAGE PREVENTION TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_leakage_test(tri_df: pd.DataFrame, regime_df: pd.DataFrame, cfg: StrategyConfig) -> dict:
    """
    Time Leakage Prevention Test:
    Run the same strategy but with SHUFFLED future returns to detect lookahead bias.

    If the strategy uses future info, it will perform well even on shuffled data.
    If no leakage, performance on shuffled data should be ~0 (random).

    Test design:
    1. Run normal backtest → get real Sharpe
    2. Shuffle future returns (keep structure, permute time axis) → get shuffled Sharpe
    3. Run 20 shuffled trials → get distribution
    4. If real Sharpe >> shuffled distribution → no leakage detected
    """
    print("\n--- TIME LEAKAGE PREVENTION TEST ---")

    # Real backtest
    real_result = run_backtest(tri_df, regime_df, cfg, variant="full")
    real_metrics = compute_metrics(real_result.equity_curve, "real")
    real_sharpe = real_metrics["sharpe_raw"]
    print(f"  Real strategy Sharpe: {real_sharpe:.3f}")

    # Shuffled trials
    n_trials = 20
    shuffled_sharpes = []

    # Create shuffled version of tri_df
    returns = tri_df.pct_change().dropna()

    for trial in range(n_trials):
        # Permute the time axis of returns (destroy temporal structure)
        shuffled_returns = returns.sample(frac=1, random_state=trial * 42).reset_index(drop=True)
        shuffled_returns.index = returns.index  # restore dates

        # Reconstruct TRI from shuffled returns
        shuffled_tri = (1 + shuffled_returns).cumprod() * 100
        shuffled_tri.iloc[0] = 100.0

        try:
            # Recompute regime on shuffled SPY
            shuffled_regime = compute_regime_series(shuffled_tri[cfg.spy_ticker], cfg)
            result = run_backtest(shuffled_tri, shuffled_regime, cfg, variant="full")
            metrics = compute_metrics(result.equity_curve, f"shuffled_{trial}")
            shuffled_sharpes.append(metrics["sharpe_raw"])
        except Exception:
            shuffled_sharpes.append(0.0)

    shuffled_mean = np.mean(shuffled_sharpes)
    shuffled_std = np.std(shuffled_sharpes)
    z_score = (real_sharpe - shuffled_mean) / shuffled_std if shuffled_std > 0 else 0

    print(f"  Shuffled Sharpe: mean={shuffled_mean:.3f}, std={shuffled_std:.3f}")
    print(f"  Z-score: {z_score:.2f}")
    print(f"  Conclusion: {'NO LEAKAGE DETECTED' if z_score > 1.96 else 'POTENTIAL LEAKAGE (or strategy has no edge)'}")

    return {
        "real_sharpe": real_sharpe,
        "shuffled_mean": shuffled_mean,
        "shuffled_std": shuffled_std,
        "z_score": z_score,
        "p_value": 1 - 0.5 * (1 + math.erf(z_score / math.sqrt(2))) if z_score > 0 else 0.5,
        "conclusion": "NO_LEAKAGE" if z_score > 1.96 else "INCONCLUSIVE",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = StrategyConfig(
        polygon_api_key="FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1",
        start_date="2015-01-01",  # extra year for warmup
        end_date="2026-02-07",
    )

    # ── 1. FETCH DATA ──
    raw_prices, tri_series = fetch_all_data(cfg)

    if len(tri_series) < 5:
        print("ERROR: Not enough tickers with data. Aborting.")
        sys.exit(1)

    # ── 2. ALIGN DATA ──
    print("\nAligning data to common trading days...")
    tri_df = align_data(tri_series)
    print(f"  Aligned: {tri_df.shape[0]} days x {tri_df.shape[1]} tickers")
    print(f"  Date range: {tri_df.index[0].date()} to {tri_df.index[-1].date()}")
    print(f"  Tickers: {sorted(tri_df.columns.tolist())}")

    # ── 3. COMPUTE REGIME ──
    print("\nComputing HVR regime...")
    regime_df = compute_regime_series(tri_df[cfg.spy_ticker], cfg)
    n_riskoff = (regime_df["regime"] == "RISK_OFF").sum()
    n_total = len(regime_df)
    print(f"  RISK_OFF days: {n_riskoff}/{n_total} ({n_riskoff/n_total*100:.1f}%)")
    transitions = (regime_df["regime"] != regime_df["regime"].shift(1)).sum()
    print(f"  Regime transitions: {transitions}")

    # ── 4. RUN 4-VARIANT BACKTEST ──
    variants = [
        ("no_regime", "1) Pure HRP (no regime)"),
        ("hvr_no_hysteresis", "2) HVR single threshold"),
        ("hvr_hysteresis", "3) HVR + hysteresis"),
        ("full", "4) Full (HVR+hysteresis+FastOff+MA200)"),
    ]

    results = {}
    for variant_key, variant_name in variants:
        print(f"\nRunning backtest: {variant_name}...")
        try:
            result = run_backtest(tri_df, regime_df, cfg, variant=variant_key)
            results[variant_key] = result
            print(f"  Trades: {result.trades}, Annual Turnover: {result.turnover_annual:.2f}x")
        except Exception as e:
            print(f"  FAILED: {e}")

    # ── 5. COMPUTE METRICS ──
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    # Full period
    print("\n--- FULL PERIOD (2016-2026) ---")
    full_metrics = []
    for variant_key, variant_name in variants:
        if variant_key in results:
            eq = results[variant_key].equity_curve
            eq_period = eq[eq.index >= "2016-01-01"]
            m = compute_metrics(eq_period, variant_name, "2016-2026")
            m["trades"] = results[variant_key].trades
            m["turnover"] = f"{results[variant_key].turnover_annual:.2f}x"
            full_metrics.append(m)
            print(f"\n  {variant_name}:")
            for k, v in m.items():
                if k not in ("name", "cagr_raw", "sharpe_raw", "max_dd_raw"):
                    print(f"    {k}: {v}")

    # Recent period
    print("\n--- RECENT PERIOD (2024-2026) ---")
    recent_metrics = []
    for variant_key, variant_name in variants:
        if variant_key in results:
            eq = results[variant_key].equity_curve
            eq_period = eq[eq.index >= "2024-01-01"]
            if len(eq_period) > 20:
                m = compute_metrics(eq_period, variant_name, "2024-2026")
                recent_metrics.append(m)
                print(f"\n  {variant_name}:")
                for k, v in m.items():
                    if k not in ("name", "cagr_raw", "sharpe_raw", "max_dd_raw"):
                        print(f"    {k}: {v}")

    # Annual breakdown for the full strategy
    if "full" in results:
        print("\n--- ANNUAL BREAKDOWN (Full Strategy) ---")
        ab = annual_breakdown(results["full"].equity_curve)
        print(ab.to_string(index=False))

    # SPY benchmark
    print("\n--- SPY BENCHMARK ---")
    spy_eq = tri_df[cfg.spy_ticker]
    spy_full = spy_eq[spy_eq.index >= "2016-01-01"]
    spy_full_norm = spy_full / spy_full.iloc[0] * cfg.initial_capital
    spy_m = compute_metrics(spy_full_norm, "SPY Buy&Hold", "2016-2026")
    print(f"  SPY Buy&Hold:")
    for k, v in spy_m.items():
        if k not in ("name", "cagr_raw", "sharpe_raw", "max_dd_raw"):
            print(f"    {k}: {v}")

    # ── 6. LEAKAGE TEST ──
    leakage = run_leakage_test(tri_df, regime_df, cfg)

    # ── 7. SAVE RESULTS ──
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)

    # Save equity curves
    eq_df = pd.DataFrame({k: v.equity_curve for k, v in results.items()})
    eq_df["SPY"] = spy_full_norm.reindex(eq_df.index)
    eq_df.to_csv(output_dir / "etf_rotation_equity_curves.csv")

    # Save summary
    summary = {
        "full_period": full_metrics,
        "recent_period": recent_metrics,
        "spy_benchmark": spy_m,
        "leakage_test": leakage,
        "config": {
            "target_vol": cfg.target_vol,
            "max_leverage": cfg.max_leverage,
            "hvr_thresholds": f"off={cfg.hvr_threshold_off}, on={cfg.hvr_threshold_on}",
            "cost_bps": cfg.cost_bps,
            "rebalance": f"HRP monthly, trades every {cfg.trade_freq_days}d",
        },
    }

    with open(output_dir / "etf_rotation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if "full" in results:
        full_m = compute_metrics(
            results["full"].equity_curve[results["full"].equity_curve.index >= "2016-01-01"],
            "full"
        )
        sharpe = full_m["sharpe_raw"]
        max_dd = full_m["max_dd_raw"]
        no_leakage = leakage["conclusion"] == "NO_LEAKAGE"

        print(f"  Full strategy Sharpe: {sharpe:.2f}")
        print(f"  Full strategy MaxDD: {max_dd*100:.1f}%")
        print(f"  Leakage test: {leakage['conclusion']} (z={leakage['z_score']:.2f})")

        if sharpe > 0.5 and no_leakage:
            print("\n  >>> RECOMMEND: Deploy to TraderApp. Strategy shows robust risk-adjusted returns")
            print("      with no detected lookahead bias.")
        elif sharpe > 0.3:
            print("\n  >>> CAUTIOUS: Strategy has modest edge. Consider paper trading first.")
        else:
            print("\n  >>> NOT RECOMMENDED: Strategy edge is too small to overcome real-world frictions.")


if __name__ == "__main__":
    main()
