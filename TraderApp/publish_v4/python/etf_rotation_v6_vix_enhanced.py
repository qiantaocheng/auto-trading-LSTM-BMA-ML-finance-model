#!/usr/bin/env python3
"""
ETF Rotation V6: VIX-Enhanced Risk Management
==============================================

Enhancements over V5 Portfolio B + P2 2-Level Cap:

1. VIX Risk-Off Veto:
   - VIX >= 25 (2-day confirm) → force MA200_DEEP_CAP (0.30)
   - VIX <= 20 (5-day confirm) → allow normal cap logic
   - Uses ^VIX index directly (yfinance free tier)

2. VIX-Based Theme Leg Budget Control:
   - VIX < 20: COPX+URA allow 10% combined (baseline)
   - 20 <= VIX < 25: reduce to 6% combined
   - VIX >= 25: reduce to 2% combined (or 0%)
   - Excess reallocated to USMV/QUAL proportionally

Strategies compared:
  - P2_baseline: Current best (Portfolio B + 2-Level Cap)
  - P2_vix_veto: P2 + VIX risk-off veto only
  - P2_vix_full: P2 + VIX veto + theme budget control

Benchmark: QQQ, SPY, P2_baseline
Transaction cost: 10 bps
Backtest: 2022-02-24 to 2026-02-09
"""
from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from etf_rotation_strategy import (
    StrategyConfig,
    fetch_polygon_daily,
    fetch_polygon_dividends,
    build_total_return_index,
    compute_regime_series,
    compute_metrics,
    annual_breakdown,
    BacktestResult,
)
from etf_rotation_v3_monthly_hmm import (
    V3Config,
    compute_hmm_regime_series,
)
from etf_rotation_v4_refined import (
    V4Config,
    blended_vol,
    two_level_risk_cap,
    hmm_continuous_weight,
    apply_deadband_and_step,
)


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_B = {
    "QQQ":  0.250, "USMV": 0.250, "QUAL": 0.200,
    "PDBC": 0.150, "DBA":  0.050, "COPX": 0.050, "URA": 0.050,
}

CASH_TICKER = "BIL"
SIGNAL_TICKER = "SPY"
VIX_PROXY = "^VIX"  # VIX index directly from yfinance

THEME_TICKERS = ["COPX", "URA"]  # Theme legs subject to VIX budget control
DEFENSIVE_TICKERS = ["USMV", "QUAL"]  # Receive theme reallocation


# ─────────────────────────────────────────────────────────────────────────────
# VIX CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VixConfig:
    """VIX-based risk overlay parameters."""
    # VIX thresholds
    vix_high_threshold: float = 25.0      # Risk-off veto trigger
    vix_low_threshold: float = 20.0       # Normal mode restore

    # Confirmation periods (in trading days)
    vix_high_confirm_days: int = 2        # 2-day confirm for risk-off
    vix_low_confirm_days: int = 5         # 5-day confirm for normal

    # Theme budget control
    theme_budget_normal: float = 0.10     # VIX < 20: 10% combined
    theme_budget_medium: float = 0.06     # 20 <= VIX < 25: 6% combined
    theme_budget_high: float = 0.02       # VIX >= 25: 2% combined

    # Risk-off override cap
    vix_override_cap: float = 0.30        # Force deep cap when VIX risk-off confirmed


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_portfolio_data(
    api_key: str, start: str, end: str,
) -> Dict[str, pd.Series]:
    """Fetch TRI for portfolio + signal + VIX proxy."""
    all_tickers = (list(PORTFOLIO_B.keys()) +
                   [CASH_TICKER, SIGNAL_TICKER, VIX_PROXY])
    tri_map = {}

    print(f"Fetching data for {len(all_tickers)} tickers...")
    for i, ticker in enumerate(sorted(set(all_tickers))):
        print(f"  [{i+1}/{len(all_tickers)}] {ticker}...", end=" ", flush=True)

        # Special handling for VIX index — use yfinance
        if ticker == "^VIX":
            try:
                import yfinance as yf
                vix = yf.Ticker(ticker)
                df = vix.history(start=start, end=end)
                if df.empty:
                    print("NO DATA")
                    continue
                # VIX has no dividends, use Close as TRI
                # Remove timezone info to match Polygon data (tz-naive)
                vix_series = df["Close"].copy()
                vix_series.index = vix_series.index.tz_localize(None)
                tri_map[ticker] = vix_series
                print(f"OK ({len(df)} bars, 0 divs)")
            except Exception as e:
                print(f"FAILED: {e}")
            continue

        # All other tickers use Polygon
        df = fetch_polygon_daily(ticker, start, end, api_key, adjusted=False)
        if df.empty:
            print("NO DATA")
            continue

        divs = fetch_polygon_dividends(ticker, start, end, api_key)
        tri = build_total_return_index(df["Close"], divs)
        tri_map[ticker] = tri

        div_count = len(divs) if not divs.empty else 0
        print(f"OK ({len(df)} bars, {div_count} divs)")

    return tri_map


def align_portfolio_data(tri_map: Dict[str, pd.Series]) -> pd.DataFrame:
    """Align all TRI to common dates."""
    df = pd.DataFrame(tri_map).sort_index()
    df = df.ffill(limit=5)
    df = df.dropna(axis=1, how="all")

    # Start from when core tickers have data (allow VIX to start later)
    core = [SIGNAL_TICKER] + list(PORTFOLIO_B.keys())
    available_core = [t for t in core if t in df.columns]
    first_valid = df[available_core].apply(lambda col: col.first_valid_index()).max()
    df = df.loc[first_valid:]
    df = df.ffill()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# VIX SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def compute_vix_regime(vix_close: pd.Series, vcfg: VixConfig) -> pd.DataFrame:
    """
    Compute VIX regime with hysteresis confirmation.

    States:
      - NORMAL: VIX below thresholds, standard risk management
      - RISK_OFF: VIX >= high threshold (confirmed), force deep cap

    Returns DataFrame with columns:
      - vix_close: VIX proxy close price
      - vix_regime: NORMAL / RISK_OFF
      - theme_budget: max allowed theme allocation (0.02/0.06/0.10)
    """
    df = pd.DataFrame({"vix_close": vix_close}).fillna(method="ffill")

    # Initialize
    regime = []
    theme_budget = []
    current_regime = "NORMAL"
    confirm_counter = 0

    for date, row in df.iterrows():
        vix = row["vix_close"]

        if pd.isna(vix):
            # No VIX data — default to NORMAL
            regime.append("NORMAL")
            theme_budget.append(vcfg.theme_budget_normal)
            continue

        # State machine with confirmation
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

        # Determine theme budget based on current VIX level
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


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO VOL ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_blended_vol(
    log_ret_df: pd.DataFrame,
    weights: Dict[str, float],
    loc: int,
    v4: V4Config,
) -> float:
    """Blended portfolio vol estimate using weighted returns."""
    available = [t for t in weights if t in log_ret_df.columns]
    if not available:
        return 0.15

    w_arr = np.array([weights.get(t, 0.0) for t in available])
    w_arr = w_arr / w_arr.sum()
    port_lr = (log_ret_df[available].iloc[:loc] * w_arr).sum(axis=1)

    short_w = port_lr.iloc[max(0, len(port_lr) - v4.vol_blend_short):]
    long_w = port_lr.iloc[max(0, len(port_lr) - v4.vol_blend_long):]

    v_short = float(short_w.std() * np.sqrt(252)) if len(short_w) > 5 else 0.15
    v_long = float(long_w.std() * np.sqrt(252)) if len(long_w) > 10 else 0.15

    blended = v4.vol_blend_alpha * v_short + (1 - v4.vol_blend_alpha) * v_long
    return max(v4.vol_floor, min(blended, v4.vol_cap))


# ─────────────────────────────────────────────────────────────────────────────
# THEME BUDGET CONTROL
# ─────────────────────────────────────────────────────────────────────────────

def apply_theme_budget(
    base_weights: Dict[str, float],
    theme_budget: float,
) -> Dict[str, float]:
    """
    Adjust portfolio weights to respect VIX theme budget cap.

    If theme legs exceed budget, scale them down and reallocate excess
    to defensive tickers (USMV/QUAL) proportionally.

    Args:
        base_weights: Initial strategic weights (normalized)
        theme_budget: Max allowed combined weight for theme tickers

    Returns:
        Adjusted weights (normalized to sum to 1.0)
    """
    weights = base_weights.copy()

    # Calculate current theme allocation
    theme_current = sum(weights.get(t, 0.0) for t in THEME_TICKERS)

    if theme_current <= theme_budget:
        # Within budget — no adjustment needed
        return weights

    # Excess to reallocate
    excess = theme_current - theme_budget

    # Scale down theme tickers proportionally to budget
    if theme_current > 0:
        scale = theme_budget / theme_current
        for ticker in THEME_TICKERS:
            if ticker in weights:
                weights[ticker] *= scale

    # Reallocate excess to defensive tickers proportionally
    defensive_current = sum(weights.get(t, 0.0) for t in DEFENSIVE_TICKERS)
    if defensive_current > 0:
        for ticker in DEFENSIVE_TICKERS:
            if ticker in weights:
                weights[ticker] += excess * (weights[ticker] / defensive_current)
    else:
        # Fallback: split equally
        for ticker in DEFENSIVE_TICKERS:
            if ticker in weights:
                weights[ticker] += excess / len(DEFENSIVE_TICKERS)

    # Renormalize
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_portfolio_backtest(
    tri_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    cfg: StrategyConfig,
    v4: V4Config,
    vcfg: VixConfig,
    variant: str,
) -> BacktestResult:
    """
    VIX-enhanced portfolio backtest.

    Variants:
      "p2_baseline" — Original P2 (2-level cap, no VIX)
      "p2_vix_veto" — P2 + VIX risk-off veto (force 0.30 cap)
      "p2_vix_full" — P2 + VIX veto + theme budget control
    """
    use_vix_veto = variant in ("p2_vix_veto", "p2_vix_full")
    use_theme_budget = variant == "p2_vix_full"

    spy = SIGNAL_TICKER
    simple_ret = tri_df.pct_change().fillna(0.0)
    log_ret = np.log(tri_df / tri_df.shift(1)).fillna(0.0)
    ma200 = tri_df[spy].rolling(200).mean()

    # Portfolio tickers
    port_tickers = [t for t in PORTFOLIO_B if t in tri_df.columns]
    port_weights = {t: PORTFOLIO_B[t] for t in port_tickers}
    total_w = sum(port_weights.values())
    if total_w > 0:
        port_weights = {t: w / total_w for t, w in port_weights.items()}

    all_tickers = sorted(set(port_tickers + [CASH_TICKER]))
    available = [t for t in all_tickers if t in tri_df.columns]

    # Warmup
    warmup = max(cfg.hvr_long_window, 252) + 10
    dates = simple_ret.index[warmup:]

    capital = cfg.initial_capital
    equity = []
    current_exposure = 0.0
    current_weights = pd.Series(0.0, index=available)
    if CASH_TICKER in current_weights.index:
        current_weights[CASH_TICKER] = 1.0

    total_trades = 0
    total_turnover = 0.0
    last_rebal_idx = -999

    for idx, date in enumerate(dates):
        days_since_rebal = idx - last_rebal_idx
        should_rebal = (days_since_rebal >= v4.rebalance_freq_days) or (idx == 0)

        if should_rebal:
            loc = tri_df.index.get_loc(date)

            # Vol-target base exposure
            pv = portfolio_blended_vol(log_ret, port_weights, loc, v4)
            scalar = cfg.target_vol / pv if pv > 0 else 1.0
            scalar = min(scalar, cfg.max_leverage)
            target_exposure = scalar

            # 2-level MA200 cap (standard P2)
            spy_price = float(tri_df[spy].iloc[loc])
            ma200_val = float(ma200.iloc[loc])
            risk_cap = two_level_risk_cap(spy_price, ma200_val, v4)
            target_exposure = min(target_exposure, risk_cap)

            # VIX risk-off veto override
            if use_vix_veto and date in vix_df.index:
                vix_regime = vix_df.loc[date, "vix_regime"]
                if vix_regime == "RISK_OFF":
                    # Force deep cap regardless of MA200
                    target_exposure = min(target_exposure, vcfg.vix_override_cap)

            # Cash buffer
            target_exposure = min(target_exposure, 1.0 - v4.min_cash_pct)
            target_exposure = max(0.0, min(1.0, target_exposure))

            # Deadband + step limit
            new_exposure = apply_deadband_and_step(
                current_exposure, target_exposure, days_since_rebal, v4)

            # Base strategic weights
            strategic_weights = port_weights.copy()

            # VIX theme budget control
            if use_theme_budget and date in vix_df.index:
                theme_budget = vix_df.loc[date, "theme_budget"]
                strategic_weights = apply_theme_budget(strategic_weights, theme_budget)

            # Build target weights
            new_weights = pd.Series(0.0, index=available)
            for t, w in strategic_weights.items():
                if t in new_weights.index:
                    new_weights[t] = w * new_exposure
            bil_w = max(0.0, 1.0 - new_weights.sum())
            if CASH_TICKER in new_weights.index:
                new_weights[CASH_TICKER] = bil_w

            # Normalize
            total = new_weights.sum()
            if total > 0:
                new_weights = new_weights / total

            # Transaction
            diff = (new_weights - current_weights).abs().sum()
            if diff > 0.02 or idx == 0:
                turnover = diff / 2
                cost = turnover * capital * v4.cost_bps / 10000
                capital -= cost
                total_turnover += turnover
                total_trades += 1
                current_weights = new_weights
                current_exposure = new_exposure
                last_rebal_idx = idx

        # Daily P&L
        if date in simple_ret.index:
            day_ret = simple_ret.loc[date]
            port_ret = sum(
                current_weights.get(t, 0.0) * day_ret.get(t, 0.0)
                for t in current_weights.index
                if current_weights.get(t, 0.0) > 0
            )
            capital *= (1 + port_ret)

        equity.append({"date": date, "equity": capital})

    eq = pd.DataFrame(equity).set_index("date")["equity"]
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    to_annual = total_turnover / years if years > 0 else 0

    return BacktestResult(
        name=variant,
        equity_curve=eq,
        trades=total_trades,
        turnover_annual=to_annual,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    api_key = "FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1"

    # Focus on recent period where VIX proxy data is reliable
    cfg = StrategyConfig(
        polygon_api_key=api_key,
        start_date="2021-01-01",  # buffer for warmup
        end_date="2026-02-10",
        trade_freq_days=21,
    )
    v4 = V4Config()
    vcfg = VixConfig()

    # Target backtest period
    backtest_start = "2022-02-24"

    print("="*80)
    print("  ETF Rotation V6: VIX-Enhanced Risk Management")
    print("="*80)
    print(f"  Backtest: {backtest_start} to {cfg.end_date}")
    print(f"  Portfolio: {PORTFOLIO_B}")
    print(f"  VIX proxy: {VIX_PROXY}")
    print("="*80)

    # ── 1. FETCH DATA ──
    tri_map = fetch_portfolio_data(api_key, cfg.start_date, cfg.end_date)

    print("\nAligning data...")
    tri_df = align_portfolio_data(tri_map)
    print(f"  {tri_df.shape[0]} days x {tri_df.shape[1]} tickers")
    print(f"  Range: {tri_df.index[0].date()} to {tri_df.index[-1].date()}")

    if SIGNAL_TICKER not in tri_df.columns:
        print(f"ERROR: {SIGNAL_TICKER} not in data")
        sys.exit(1)

    # ── 2. VIX REGIME ──
    print(f"\nComputing VIX regime ({VIX_PROXY})...")
    if VIX_PROXY in tri_df.columns:
        vix_df = compute_vix_regime(tri_df[VIX_PROXY], vcfg)
        n_risk_off = (vix_df["vix_regime"] == "RISK_OFF").sum()
        print(f"  VIX RISK_OFF: {n_risk_off}/{len(vix_df)} ({n_risk_off/len(vix_df)*100:.1f}%)")

        # Theme budget stats
        budget_dist = vix_df["theme_budget"].value_counts().sort_index()
        print(f"  Theme budget distribution:")
        for budget, count in budget_dist.items():
            print(f"    {budget:.2f}: {count} days ({count/len(vix_df)*100:.1f}%)")
    else:
        print(f"  WARNING: {VIX_PROXY} not available — VIX features DISABLED")
        vix_df = pd.DataFrame(index=tri_df.index)
        vix_df["vix_regime"] = "NORMAL"
        vix_df["theme_budget"] = vcfg.theme_budget_normal

    # ── 3. TRIM TO TARGET BACKTEST PERIOD ──
    backtest_start_dt = pd.Timestamp(backtest_start)
    tri_df = tri_df[tri_df.index >= backtest_start_dt]
    vix_df = vix_df[vix_df.index >= backtest_start_dt]

    print(f"\nTrimmed to backtest period:")
    print(f"  {tri_df.shape[0]} days from {tri_df.index[0].date()} to {tri_df.index[-1].date()}")

    # ── 4. RUN STRATEGIES ──
    variants = [
        ("p2_baseline", "P2 Baseline (no VIX)"),
        ("p2_vix_veto", "P2 + VIX Veto"),
        ("p2_vix_full", "P2 + VIX Full"),
    ]

    results = {}
    for vkey, vname in variants:
        print(f"\n  Running {vname}...", end=" ", flush=True)
        try:
            r = run_portfolio_backtest(tri_df, vix_df, cfg, v4, vcfg, vkey)
            results[vkey] = r
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    # ── 5. COMPUTE METRICS ──
    metrics = {}
    for vkey, vname in variants:
        if vkey not in results:
            continue
        eq = results[vkey].equity_curve
        m = compute_metrics(eq, vname)
        m["trades"] = results[vkey].trades
        m["turnover_annual"] = results[vkey].turnover_annual
        metrics[vkey] = m

    # ── 6. BENCHMARKS ──
    earliest = tri_df.index[0]
    benchmarks = {}
    for ticker in ["QQQ", "SPY"]:
        if ticker in tri_df.columns:
            p = tri_df[ticker][tri_df.index >= earliest]
            norm = p / p.iloc[0] * cfg.initial_capital
            benchmarks[ticker] = compute_metrics(norm, f"{ticker} B&H")

    # ── 7. RESULTS TABLE ──
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"  {'Strategy':<22} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>7} {'Calmar':>7} {'Vol':>7} {'TO':>5}")
    print(f"  {'-'*70}")

    for vkey, vname in variants:
        if vkey not in metrics:
            continue
        m = metrics[vkey]
        calmar = m["cagr_raw"] / abs(m["max_dd_raw"]) if abs(m["max_dd_raw"]) > 0 else 0
        print(f"  {vname:<22} {m['sharpe_raw']:>7.2f} {m['cagr']:>8} {m['max_dd']:>7} "
              f"{calmar:>7.2f} {m['ann_vol']:>7} {results[vkey].turnover_annual:>4.1f}x")

    # Benchmarks
    for ticker in ["QQQ", "SPY"]:
        if ticker in benchmarks:
            bm = benchmarks[ticker]
            calmar = bm["cagr_raw"] / abs(bm["max_dd_raw"]) if abs(bm["max_dd_raw"]) > 0 else 0
            print(f"  {ticker + ' B&H':<22} {bm['sharpe_raw']:>7.2f} {bm['cagr']:>8} {bm['max_dd']:>7} "
                  f"{calmar:>7.2f} {bm['ann_vol']:>7}   REF")

    # ── 8. ANNUAL BREAKDOWN ──
    for vkey, vname in variants:
        if vkey not in results:
            continue
        print(f"\n  --- Annual Breakdown: {vname} ---")
        ab = annual_breakdown(results[vkey].equity_curve)
        print("  " + ab.to_string(index=False).replace("\n", "\n  "))

    # ── 9. IMPROVEMENT ANALYSIS ──
    if "p2_baseline" in metrics and "p2_vix_full" in metrics:
        base_sharpe = metrics["p2_baseline"]["sharpe_raw"]
        full_sharpe = metrics["p2_vix_full"]["sharpe_raw"]
        delta_sharpe = full_sharpe - base_sharpe

        base_dd = metrics["p2_baseline"]["max_dd_raw"]
        full_dd = metrics["p2_vix_full"]["max_dd_raw"]
        delta_dd = full_dd - base_dd

        print(f"\n" + "="*80)
        print("VIX ENHANCEMENT IMPACT")
        print("="*80)
        print(f"  Sharpe improvement: {delta_sharpe:+.3f} ({delta_sharpe/base_sharpe*100:+.1f}%)")
        print(f"  MaxDD improvement:  {delta_dd:+.3f} ({delta_dd/base_dd*100:+.1f}%)")
        print(f"  Status: {'BENEFICIAL' if delta_sharpe > 0 else 'NEUTRAL/NEGATIVE'}")

    # ── 10. SAVE ──
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)

    summary = {
        "portfolio": {t: f"{w*100:.1f}%" for t, w in PORTFOLIO_B.items()},
        "vix_proxy": VIX_PROXY,
        "vix_config": {
            "high_threshold": vcfg.vix_high_threshold,
            "low_threshold": vcfg.vix_low_threshold,
            "high_confirm_days": vcfg.vix_high_confirm_days,
            "low_confirm_days": vcfg.vix_low_confirm_days,
            "theme_budgets": {
                "normal": vcfg.theme_budget_normal,
                "medium": vcfg.theme_budget_medium,
                "high": vcfg.theme_budget_high,
            },
        },
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "name"}
                    for k, v in metrics.items()},
        "benchmarks": benchmarks,
        "backtest_period": {"start": backtest_start, "end": cfg.end_date},
    }

    output_file = output_dir / "etf_rotation_v6_vix_enhanced.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
