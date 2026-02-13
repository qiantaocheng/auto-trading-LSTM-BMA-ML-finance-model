#!/usr/bin/env python3
"""
Stock Risk Overlay Backtest: T10C-Slim Strategy vs Buy-and-Hold
================================================================

Standalone single-stock risk management overlay derived from T10C-Slim ETF rotation.
Applies the EXACT T10C-Slim risk layers to individual stocks:
  - Vol-target: target_vol(0.12) / stock_blended_vol -> position size
  - MA200 2-level cap: SPY-based (1.0 / 0.60 / 0.30)
  - VIX exposure cap: 0.50 when VIX regime RISK_OFF
  - Asymmetric deadband: 0.02 up / 0.05 down
  - Max step: 15%, min cash: 5%
  - 21-day rebalance cycle
  - 10 bps transaction cost

Tests random 6 small-cap + 6 mid-cap stocks, compares WITH vs WITHOUT strategy.
No local imports — fully standalone.
"""
from __future__ import annotations

import json, sys, random, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ── VIX Regime Detection (inlined from etf_rotation_v6) ─────────────────────

@dataclass
class VixConfig:
    """VIX-based risk overlay parameters."""
    vix_high_threshold: float = 25.0
    vix_low_threshold: float = 20.0
    vix_high_confirm_days: int = 2
    vix_low_confirm_days: int = 5
    theme_budget_normal: float = 0.10
    theme_budget_medium: float = 0.06
    theme_budget_high: float = 0.02
    vix_override_cap: float = 0.30


def compute_vix_regime(vix_close: pd.Series, vcfg: VixConfig) -> pd.DataFrame:
    """
    Compute VIX regime with hysteresis confirmation.
    States: NORMAL / RISK_OFF
    """
    df = pd.DataFrame({"vix_close": vix_close}).fillna(method="ffill")
    regime = []
    theme_budget = []
    current_regime = "NORMAL"
    confirm_counter = 0

    for date, row in df.iterrows():
        vix = row["vix_close"]
        if pd.isna(vix):
            regime.append("NORMAL")
            theme_budget.append(vcfg.theme_budget_normal)
            continue

        if current_regime == "NORMAL":
            if vix >= vcfg.vix_high_threshold:
                confirm_counter += 1
                if confirm_counter >= vcfg.vix_high_confirm_days:
                    current_regime = "RISK_OFF"
                    confirm_counter = 0
            else:
                confirm_counter = 0
        elif current_regime == "RISK_OFF":
            if vix <= vcfg.vix_low_threshold:
                confirm_counter += 1
                if confirm_counter >= vcfg.vix_low_confirm_days:
                    current_regime = "NORMAL"
                    confirm_counter = 0
            else:
                confirm_counter = 0

        if vix < vcfg.vix_low_threshold:
            budget = vcfg.theme_budget_normal
        elif vix < vcfg.vix_high_threshold:
            budget = vcfg.theme_budget_medium
        else:
            budget = vcfg.theme_budget_high

        regime.append(current_regime)
        theme_budget.append(budget)

    df["vix_regime"] = regime
    df["theme_budget"] = theme_budget
    return df


# ── T10C-Slim Parameters (EXACT match to live + backtest) ────────────────────

TARGET_VOL = 0.12
MAX_LEVERAGE = 1.0
VOL_BLEND_SHORT = 20
VOL_BLEND_LONG = 60
VOL_BLEND_ALPHA = 0.7
VOL_FLOOR = 0.08
VOL_CAP = 0.40
MA200_SHALLOW_CAP = 0.60
MA200_DEEP_CAP = 0.30
MA200_DEEP_THRESHOLD = -0.05
MIN_CASH_PCT = 0.05
VIX_EXPOSURE_CAP = 0.50
DEADBAND_UP = 0.02
DEADBAND_DOWN = 0.05
MAX_STEP = 0.15
REBALANCE_FREQ = 21
COST_BPS = 10.0


# ── Helper Functions ──────────────────────────────────────────────────────────

def stock_blended_vol(log_ret: pd.Series, loc: int) -> float:
    """Compute blended vol for a single stock at position loc."""
    if loc < VOL_BLEND_LONG:
        return 0.15  # default fallback

    short_w = log_ret.iloc[max(0, loc - VOL_BLEND_SHORT):loc]
    long_w = log_ret.iloc[max(0, loc - VOL_BLEND_LONG):loc]

    v_short = float(short_w.std() * np.sqrt(252)) if len(short_w) > 5 else 0.15
    v_long = float(long_w.std() * np.sqrt(252)) if len(long_w) > 10 else 0.15

    blended = VOL_BLEND_ALPHA * v_short + (1 - VOL_BLEND_ALPHA) * v_long
    return max(VOL_FLOOR, min(blended, VOL_CAP))


def two_level_risk_cap(spy_price: float, ma200: float) -> float:
    """2-level MA200 risk cap: 1.0 / 0.60 / 0.30."""
    if np.isnan(ma200) or ma200 <= 0:
        return 1.0
    deviation = (spy_price - ma200) / ma200
    if deviation >= 0:
        return 1.0
    elif deviation > MA200_DEEP_THRESHOLD:
        return MA200_SHALLOW_CAP
    else:
        return MA200_DEEP_CAP


def run_single_stock_overlay(
    stock_close: pd.Series,
    spy_close: pd.Series,
    vix_regime: pd.Series,
    initial_capital: float = 100_000,
) -> Tuple[pd.Series, int, dict]:
    """
    Run T10C-Slim risk overlay on a single stock.

    Returns: (equity_curve, trade_count, stats_dict)
    """
    # Align all data to common dates
    common = stock_close.index.intersection(spy_close.index).intersection(vix_regime.index)
    stock = stock_close.loc[common].sort_index()
    spy = spy_close.loc[common].sort_index()
    vix_reg = vix_regime.loc[common].sort_index()

    # Compute log returns for stock
    log_ret = np.log(stock / stock.shift(1)).dropna()

    # SPY MA200
    ma200 = spy.rolling(200).mean()

    warmup = max(200, VOL_BLEND_LONG) + 10
    if len(stock) < warmup + 50:
        return pd.Series(dtype=float), 0, {"error": "insufficient data"}

    dates = stock.index[warmup:]
    capital = initial_capital
    equity = []
    trades = 0
    cur_exp = 0.0  # current exposure (0 to 1)
    last_rb = -999

    risk_cap_days = 0
    vix_cap_days = 0

    for idx, date in enumerate(dates):
        ds = idx - last_rb
        loc = stock.index.get_loc(date)

        sp = float(spy.iloc[loc])
        m2v = float(ma200.iloc[loc])

        # Daily P&L from stock return * current exposure
        if idx > 0 and loc > 0:
            stock_ret = float(stock.iloc[loc] / stock.iloc[loc - 1] - 1)
            daily_pnl = capital * cur_exp * stock_ret
            capital += daily_pnl

        rebal = ds >= REBALANCE_FREQ or idx == 0
        if not rebal:
            equity.append(capital)
            continue

        # ── Rebalance logic (exact T10C-Slim order) ──

        # 1. Vol-target
        lr_loc = log_ret.index.get_loc(date) if date in log_ret.index else None
        if lr_loc is None:
            equity.append(capital)
            continue

        bvol = stock_blended_vol(log_ret, lr_loc)
        te = TARGET_VOL / bvol if bvol > 0 else 1.0
        te = min(te, MAX_LEVERAGE)

        # 2. MA200 2-level cap
        rc = two_level_risk_cap(sp, m2v)
        te = min(te, rc)
        if rc < 1.0:
            risk_cap_days += 1

        # 3. VIX exposure cap (L4)
        if date in vix_reg.index and vix_reg.loc[date] == "RISK_OFF":
            te = min(te, VIX_EXPOSURE_CAP)
            vix_cap_days += 1

        # 4. Min cash
        te = max(0.0, min(1.0 - MIN_CASH_PCT, te))

        # 5. Asymmetric deadband (L6)
        delta = te - cur_exp
        if delta > 0 and delta < DEADBAND_UP:
            te = cur_exp  # suppress small increase
        elif delta < 0 and abs(delta) < DEADBAND_DOWN:
            te = cur_exp  # suppress small decrease
        elif delta > MAX_STEP:
            te = cur_exp + MAX_STEP
        elif delta < -MAX_STEP:
            te = cur_exp - MAX_STEP

        # Apply trade cost
        actual_delta = abs(te - cur_exp)
        if actual_delta > 0.02:
            cost = actual_delta * capital * COST_BPS / 10_000
            capital -= cost
            trades += 1
            last_rb = idx

        cur_exp = te
        equity.append(capital)

    if not equity:
        return pd.Series(dtype=float), 0, {"error": "no equity data"}

    eq_series = pd.Series(equity, index=dates[:len(equity)])

    stats = {
        "trades": trades,
        "risk_cap_days": risk_cap_days,
        "vix_cap_days": vix_cap_days,
    }
    return eq_series, trades, stats


def run_buy_and_hold(
    stock_close: pd.Series,
    spy_close: pd.Series,
    start_date: pd.Timestamp,
    initial_capital: float = 100_000,
) -> pd.Series:
    """Simple buy-and-hold from start_date."""
    stock = stock_close.loc[start_date:].dropna()
    if len(stock) < 2:
        return pd.Series(dtype=float)

    return stock / stock.iloc[0] * initial_capital


def compute_metrics(equity: pd.Series, name: str = "") -> dict:
    """Compute CAGR, MaxDD, Sharpe, Calmar."""
    if len(equity) < 20:
        return {"name": name, "cagr": 0, "maxdd": 0, "sharpe": 0, "calmar": 0, "vol": 0}

    daily_ret = equity.pct_change().dropna()
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25

    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = daily_ret.std() * np.sqrt(252)
    rf_daily = 0.04 / 252
    excess = daily_ret - rf_daily
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    cummax = equity.cummax()
    maxdd = ((equity - cummax) / cummax).min()
    calmar = cagr / abs(maxdd) if abs(maxdd) > 0 else 0

    return {
        "name": name,
        "cagr": float(cagr),
        "maxdd": float(maxdd),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "vol": float(ann_vol),
        "total_ret": float(total_ret),
    }


def yearly_returns(eq: pd.Series) -> Dict[int, dict]:
    """Year-by-year returns."""
    out = {}
    for yr in sorted(eq.index.year.unique()):
        mask = eq.index.year == yr
        if mask.sum() < 10:
            continue
        seg = eq[mask]
        ret = seg.iloc[-1] / seg.iloc[0] - 1
        cmx = seg.cummax()
        dd = ((seg - cmx) / cmx).min()
        out[yr] = {"ret": float(ret), "maxdd": float(dd)}
    return out


def print_separator(char="=", width=120):
    print(char * width)


def main():
    initial_capital = 100_000

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Load SPY + VIX data for macro signals
    # ═══════════════════════════════════════════════════════════════════════
    print_separator()
    print("  Stock Risk Overlay Backtest: T10C-Slim Strategy vs Buy-and-Hold")
    print_separator()

    import yfinance as yf

    print("\nLoading SPY data from yfinance...", flush=True)
    spy_raw = yf.download("SPY", start="2020-01-01", end="2026-02-11",
                          progress=False, auto_adjust=True)
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_close = spy_raw["Close"].sort_index()
    spy_close.index = pd.to_datetime(spy_close.index).normalize()
    print(f"  SPY: {len(spy_close)} bars ({spy_close.index[0].date()} to {spy_close.index[-1].date()})")

    print("Loading VIX data...", flush=True)
    try:
        vix_data = yf.download("^VIX", start="2020-01-01", end="2026-02-11",
                               progress=False, auto_adjust=True)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.get_level_values(0)
        vix_series = vix_data["Close"].sort_index()
        vix_series.index = pd.to_datetime(vix_series.index).normalize()
        print(f"  VIX: {len(vix_series)} bars")
    except Exception as e:
        print(f"  VIX load failed: {e} — using dummy NORMAL regime")
        vix_series = pd.Series(15.0, index=spy_close.index)

    # Compute VIX regime (same as backtest)
    vcfg = VixConfig()
    vix_regime_df = compute_vix_regime(vix_series, vcfg)
    vix_regime = vix_regime_df["vix_regime"]
    print(f"  VIX regime: {(vix_regime == 'RISK_OFF').sum()} RISK_OFF days / {len(vix_regime)} total")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Load stock data + market cap classification
    # ═══════════════════════════════════════════════════════════════════════

    cap_csv = Path(r"D:\trade\data\factor_exports\polygon_factors_all_2021_2026_T5_final_cap200m_15b.market_caps_20260204.csv")
    parquet_path = Path(r"D:\trade\data\factor_exports\polygon_full_features_T5.parquet")

    if not cap_csv.exists():
        print(f"\nERROR: Market cap file not found: {cap_csv}")
        return
    if not parquet_path.exists():
        print(f"\nERROR: Parquet file not found: {parquet_path}")
        return

    caps = pd.read_csv(cap_csv)
    small_caps = caps[(caps["market_cap"] >= 300e6) & (caps["market_cap"] < 2e9)]["ticker"].tolist()
    mid_caps = caps[(caps["market_cap"] >= 2e9) & (caps["market_cap"] < 10e9)]["ticker"].tolist()

    print(f"\nMarket cap classification:")
    print(f"  Small-cap ($300M-$2B): {len(small_caps)} tickers")
    print(f"  Mid-cap ($2B-$10B):    {len(mid_caps)} tickers")

    print(f"\nLoading stock prices from parquet...", flush=True)
    df_all = pd.read_parquet(parquet_path, columns=["Close"])
    print(f"  {len(df_all)} rows loaded")

    available_tickers = set(df_all.index.get_level_values("ticker").unique())
    small_avail = [t for t in small_caps if t in available_tickers]
    mid_avail = [t for t in mid_caps if t in available_tickers]
    print(f"  Small-cap with data: {len(small_avail)}")
    print(f"  Mid-cap with data:   {len(mid_avail)}")

    # Randomly select 6 of each (fixed seed for reproducibility)
    random.seed(42)
    selected_small = random.sample(small_avail, min(6, len(small_avail)))
    selected_mid = random.sample(mid_avail, min(6, len(mid_avail)))

    cap_map = caps.set_index("ticker")["market_cap"].to_dict()

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Run T10C-Slim on each stock vs Buy-and-Hold
    # ═══════════════════════════════════════════════════════════════════════

    def get_stock_close(ticker: str) -> pd.Series:
        """Extract close prices for a single stock from parquet."""
        try:
            stock_data = df_all.loc[(slice(None), ticker), "Close"].droplevel("ticker")
            stock_data = stock_data.sort_index().dropna()
            stock_data.index = pd.to_datetime(stock_data.index).normalize()
            return stock_data
        except Exception:
            return pd.Series(dtype=float)

    def test_stock(ticker: str, label: str) -> Optional[dict]:
        """Run T10C-Slim + buy-and-hold comparison for one stock."""
        stock = get_stock_close(ticker)
        if len(stock) < 300:
            return None

        # Run T10C-Slim
        eq_strat, trades, stats = run_single_stock_overlay(
            stock, spy_close, vix_regime, initial_capital
        )
        if eq_strat.empty:
            return None

        # Buy-and-hold from same start date
        eq_bah = run_buy_and_hold(stock, spy_close, eq_strat.index[0], initial_capital)

        # Align to common dates
        common = eq_strat.index.intersection(eq_bah.index)
        if len(common) < 50:
            return None
        eq_strat = eq_strat.loc[common]
        eq_bah = eq_bah.loc[common]

        m_strat = compute_metrics(eq_strat, f"{ticker} T10C")
        m_bah = compute_metrics(eq_bah, f"{ticker} B&H")

        yr_strat = yearly_returns(eq_strat)
        yr_bah = yearly_returns(eq_bah)

        mcap = cap_map.get(ticker, 0)

        return {
            "ticker": ticker,
            "label": label,
            "mcap": mcap,
            "mcap_str": f"${mcap/1e9:.1f}B" if mcap >= 1e9 else f"${mcap/1e6:.0f}M",
            "strat": m_strat,
            "bah": m_bah,
            "yr_strat": yr_strat,
            "yr_bah": yr_bah,
            "trades": trades,
            "stats": stats,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # PART A: Small-Cap Results
    # ═══════════════════════════════════════════════════════════════════════
    print_separator()
    print("  PART A: SMALL-CAP STOCKS — T10C-Slim vs Buy-and-Hold")
    print_separator()

    small_results = []
    for ticker in selected_small:
        print(f"  Testing {ticker}...", flush=True)
        result = test_stock(ticker, "small-cap")
        if result:
            small_results.append(result)

    if small_results:
        print(f"\n{'Ticker':<8} {'MCap':>8} {'':>6} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Vol':>8} {'Trades':>7}")
        print("-" * 80)
        for r in small_results:
            ms = r["strat"]
            mb = r["bah"]
            print(f"{r['ticker']:<8} {r['mcap_str']:>8} {'T10C':>6} {ms['cagr']:>7.1%} {ms['maxdd']:>7.1%} {ms['sharpe']:>7.2f} {ms['calmar']:>7.2f} {ms['vol']:>7.1%} {r['trades']:>6d}")
            print(f"{'':8} {'':>8} {'B&H':>6} {mb['cagr']:>7.1%} {mb['maxdd']:>7.1%} {mb['sharpe']:>7.2f} {mb['calmar']:>7.2f} {mb['vol']:>7.1%} {'':>6}")

        # Year-by-year for each stock
        print(f"\n{'':─<120}")
        print(f"  Year-by-Year Returns (T10C / B&H)")
        print(f"{'':─<120}")
        all_years = sorted(set().union(*[set(r["yr_strat"]) | set(r["yr_bah"]) for r in small_results]))
        header = f"{'Ticker':<8}"
        for yr in all_years:
            header += f" {yr:>12}"
        print(header)
        print("-" * (8 + 13 * len(all_years)))
        for r in small_results:
            line_s = f"{r['ticker']:<7}S"
            line_b = f"{'':7}B"
            for yr in all_years:
                s_ret = r["yr_strat"].get(yr, {}).get("ret", None)
                b_ret = r["yr_bah"].get(yr, {}).get("ret", None)
                line_s += f" {s_ret:>+11.1%}" if s_ret is not None else f" {'n/a':>11}"
                line_b += f" {b_ret:>+11.1%}" if b_ret is not None else f" {'n/a':>11}"
            print(line_s)
            print(line_b)

        # Summary: average improvement
        avg_cagr_strat = np.mean([r["strat"]["cagr"] for r in small_results])
        avg_cagr_bah = np.mean([r["bah"]["cagr"] for r in small_results])
        avg_dd_strat = np.mean([r["strat"]["maxdd"] for r in small_results])
        avg_dd_bah = np.mean([r["bah"]["maxdd"] for r in small_results])
        avg_sharpe_strat = np.mean([r["strat"]["sharpe"] for r in small_results])
        avg_sharpe_bah = np.mean([r["bah"]["sharpe"] for r in small_results])
        avg_calmar_strat = np.mean([r["strat"]["calmar"] for r in small_results])
        avg_calmar_bah = np.mean([r["bah"]["calmar"] for r in small_results])

        print(f"\n{'Metric':<20} {'T10C-Slim Avg':>14} {'B&H Avg':>14} {'Delta':>14}")
        print("-" * 65)
        print(f"{'CAGR':<20} {avg_cagr_strat:>13.1%} {avg_cagr_bah:>13.1%} {avg_cagr_strat - avg_cagr_bah:>+13.1%}")
        print(f"{'MaxDD':<20} {avg_dd_strat:>13.1%} {avg_dd_bah:>13.1%} {avg_dd_strat - avg_dd_bah:>+13.1%}")
        print(f"{'Sharpe':<20} {avg_sharpe_strat:>13.2f} {avg_sharpe_bah:>13.2f} {avg_sharpe_strat - avg_sharpe_bah:>+13.2f}")
        print(f"{'Calmar':<20} {avg_calmar_strat:>13.2f} {avg_calmar_bah:>13.2f} {avg_calmar_strat - avg_calmar_bah:>+13.2f}")

        wins = sum(1 for r in small_results if r["strat"]["sharpe"] > r["bah"]["sharpe"])
        print(f"\nT10C-Slim beats B&H (Sharpe): {wins}/{len(small_results)} stocks")
        wins_dd = sum(1 for r in small_results if r["strat"]["maxdd"] > r["bah"]["maxdd"])
        print(f"T10C-Slim better MaxDD:       {wins_dd}/{len(small_results)} stocks")

    # ═══════════════════════════════════════════════════════════════════════
    # PART B: Mid-Cap Results
    # ═══════════════════════════════════════════════════════════════════════
    print_separator()
    print("  PART B: MID-CAP STOCKS — T10C-Slim vs Buy-and-Hold")
    print_separator()

    mid_results = []
    for ticker in selected_mid:
        print(f"  Testing {ticker}...", flush=True)
        result = test_stock(ticker, "mid-cap")
        if result:
            mid_results.append(result)

    if mid_results:
        print(f"\n{'Ticker':<8} {'MCap':>8} {'':>6} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8} {'Calmar':>8} {'Vol':>8} {'Trades':>7}")
        print("-" * 80)
        for r in mid_results:
            ms = r["strat"]
            mb = r["bah"]
            print(f"{r['ticker']:<8} {r['mcap_str']:>8} {'T10C':>6} {ms['cagr']:>7.1%} {ms['maxdd']:>7.1%} {ms['sharpe']:>7.2f} {ms['calmar']:>7.2f} {ms['vol']:>7.1%} {r['trades']:>6d}")
            print(f"{'':8} {'':>8} {'B&H':>6} {mb['cagr']:>7.1%} {mb['maxdd']:>7.1%} {mb['sharpe']:>7.2f} {mb['calmar']:>7.2f} {mb['vol']:>7.1%} {'':>6}")

        # Year-by-year
        print(f"\n{'':─<120}")
        print(f"  Year-by-Year Returns (T10C / B&H)")
        print(f"{'':─<120}")
        all_years = sorted(set().union(*[set(r["yr_strat"]) | set(r["yr_bah"]) for r in mid_results]))
        header = f"{'Ticker':<8}"
        for yr in all_years:
            header += f" {yr:>12}"
        print(header)
        print("-" * (8 + 13 * len(all_years)))
        for r in mid_results:
            line_s = f"{r['ticker']:<7}S"
            line_b = f"{'':7}B"
            for yr in all_years:
                s_ret = r["yr_strat"].get(yr, {}).get("ret", None)
                b_ret = r["yr_bah"].get(yr, {}).get("ret", None)
                line_s += f" {s_ret:>+11.1%}" if s_ret is not None else f" {'n/a':>11}"
                line_b += f" {b_ret:>+11.1%}" if b_ret is not None else f" {'n/a':>11}"
            print(line_s)
            print(line_b)

        # Summary
        avg_cagr_strat = np.mean([r["strat"]["cagr"] for r in mid_results])
        avg_cagr_bah = np.mean([r["bah"]["cagr"] for r in mid_results])
        avg_dd_strat = np.mean([r["strat"]["maxdd"] for r in mid_results])
        avg_dd_bah = np.mean([r["bah"]["maxdd"] for r in mid_results])
        avg_sharpe_strat = np.mean([r["strat"]["sharpe"] for r in mid_results])
        avg_sharpe_bah = np.mean([r["bah"]["sharpe"] for r in mid_results])
        avg_calmar_strat = np.mean([r["strat"]["calmar"] for r in mid_results])
        avg_calmar_bah = np.mean([r["bah"]["calmar"] for r in mid_results])

        print(f"\n{'Metric':<20} {'T10C-Slim Avg':>14} {'B&H Avg':>14} {'Delta':>14}")
        print("-" * 65)
        print(f"{'CAGR':<20} {avg_cagr_strat:>13.1%} {avg_cagr_bah:>13.1%} {avg_cagr_strat - avg_cagr_bah:>+13.1%}")
        print(f"{'MaxDD':<20} {avg_dd_strat:>13.1%} {avg_dd_bah:>13.1%} {avg_dd_strat - avg_dd_bah:>+13.1%}")
        print(f"{'Sharpe':<20} {avg_sharpe_strat:>13.2f} {avg_sharpe_bah:>13.2f} {avg_sharpe_strat - avg_sharpe_bah:>+13.2f}")
        print(f"{'Calmar':<20} {avg_calmar_strat:>13.2f} {avg_calmar_bah:>13.2f} {avg_calmar_strat - avg_calmar_bah:>+13.2f}")

        wins = sum(1 for r in mid_results if r["strat"]["sharpe"] > r["bah"]["sharpe"])
        print(f"\nT10C-Slim beats B&H (Sharpe): {wins}/{len(mid_results)} stocks")
        wins_dd = sum(1 for r in mid_results if r["strat"]["maxdd"] > r["bah"]["maxdd"])
        print(f"T10C-Slim better MaxDD:       {wins_dd}/{len(mid_results)} stocks")

    # ═══════════════════════════════════════════════════════════════════════
    # COMBINED SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    all_results = small_results + mid_results
    if all_results:
        print_separator()
        print("  COMBINED SUMMARY: All 12 Stocks")
        print_separator()

        avg_cagr_s = np.mean([r["strat"]["cagr"] for r in all_results])
        avg_cagr_b = np.mean([r["bah"]["cagr"] for r in all_results])
        avg_dd_s = np.mean([r["strat"]["maxdd"] for r in all_results])
        avg_dd_b = np.mean([r["bah"]["maxdd"] for r in all_results])
        avg_sharpe_s = np.mean([r["strat"]["sharpe"] for r in all_results])
        avg_sharpe_b = np.mean([r["bah"]["sharpe"] for r in all_results])
        avg_calmar_s = np.mean([r["strat"]["calmar"] for r in all_results])
        avg_calmar_b = np.mean([r["bah"]["calmar"] for r in all_results])

        print(f"\n{'Metric':<20} {'T10C-Slim Avg':>14} {'B&H Avg':>14} {'Improvement':>14}")
        print("-" * 65)
        print(f"{'CAGR':<20} {avg_cagr_s:>13.1%} {avg_cagr_b:>13.1%} {avg_cagr_s - avg_cagr_b:>+13.1%}")
        print(f"{'MaxDD':<20} {avg_dd_s:>13.1%} {avg_dd_b:>13.1%} {avg_dd_s - avg_dd_b:>+13.1%}")
        print(f"{'Sharpe':<20} {avg_sharpe_s:>13.2f} {avg_sharpe_b:>13.2f} {avg_sharpe_s - avg_sharpe_b:>+13.2f}")
        print(f"{'Calmar':<20} {avg_calmar_s:>13.2f} {avg_calmar_b:>13.2f} {avg_calmar_s - avg_calmar_b:>+13.2f}")

        total_wins_sharpe = sum(1 for r in all_results if r["strat"]["sharpe"] > r["bah"]["sharpe"])
        total_wins_dd = sum(1 for r in all_results if r["strat"]["maxdd"] > r["bah"]["maxdd"])
        total_wins_calmar = sum(1 for r in all_results if r["strat"]["calmar"] > r["bah"]["calmar"])

        print(f"\nWin Rate (T10C > B&H):")
        print(f"  Sharpe:  {total_wins_sharpe}/{len(all_results)} ({total_wins_sharpe/len(all_results)*100:.0f}%)")
        print(f"  MaxDD:   {total_wins_dd}/{len(all_results)} ({total_wins_dd/len(all_results)*100:.0f}%)")
        print(f"  Calmar:  {total_wins_calmar}/{len(all_results)} ({total_wins_calmar/len(all_results)*100:.0f}%)")

        # Key insight
        print(f"\nKey Insight:")
        print(f"  T10C-Slim is a RISK MANAGEMENT overlay. Its primary value is:")
        print(f"  - Reducing MaxDD (drawdown protection via vol-target + MA200 cap + VIX cap)")
        print(f"  - Improving risk-adjusted returns (Sharpe/Calmar) even if raw CAGR is lower")
        print(f"  - The asymmetric deadband prevents whipsaw from frequent rebalancing")


if __name__ == "__main__":
    main()
