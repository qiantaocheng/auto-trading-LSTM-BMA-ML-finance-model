#!/usr/bin/env python3
"""
ETF Rotation V5: Multi-Asset Portfolio with Risk Management
=============================================================

Replace SPY-only with an 8-ETF strategic allocation:

  Core Growth (50%):
    QQQ   20%   NASDAQ-100
    VTI   20%   Total US Market
    AVUV  10%   Small Cap Value

  Defensive (35%):
    USMV  20%   US Low Volatility
    QUAL  15%   US Quality Factor

  Structural Commodities (15%):
    COPX   5%   Copper Mining
    URA    5%   Uranium / Nuclear
    DBA    5%   Agriculture

  Cash (variable):
    BIL    0%+  T-Bill cash absorber (risk-off)

Risk engine determines total equity exposure (0–100%).
Equity portion distributed at strategic weights.
Remainder → BIL.

Strategies (top 4 from V4):
  P1) Portfolio baseline         — vol target + binary MA200 cap
  P2) Portfolio + 2-level cap    — vol target + continuous MA200 cap (100/60/30)
  P3) Portfolio + consensus      — vol target + HVR/MA200/HMM vote
  P4) Portfolio + B1 continuous  — vol target + continuous HMM sizing
  P5) Buy & Hold (no risk management, static weights)

Benchmark: QQQ B&H, SPY B&H
Transaction cost: 10 bps
"""
from __future__ import annotations

import json
import sys
import time
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
PORTFOLIO_B_DEF = {
    "QQQ":  0.225, "USMV": 0.275, "QUAL": 0.200,
    "PDBC": 0.150, "DBA":  0.050, "COPX": 0.050, "URA": 0.050,
}
PORTFOLIO_B_TRIM = {
    "QQQ":  0.250, "USMV": 0.250, "QUAL": 0.220,
    "PDBC": 0.150, "DBA":  0.050, "COPX": 0.040, "URA": 0.040,
}
# Active portfolio (set by main)
PORTFOLIO = PORTFOLIO_B

CASH_TICKER = "BIL"
SIGNAL_TICKER = "SPY"  # regime signals still from SPY

ALL_TICKERS = list(PORTFOLIO.keys()) + [CASH_TICKER, SIGNAL_TICKER]


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_portfolio_data(
    api_key: str, start: str, end: str,
) -> Dict[str, pd.Series]:
    """Fetch TRI for all portfolio + signal tickers."""
    tri_map = {}
    tickers = sorted(set(ALL_TICKERS))

    print(f"Fetching data for {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)

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

    # Start from when ALL portfolio tickers have data
    first_valid = df.apply(lambda col: col.first_valid_index()).max()
    df = df.loc[first_valid:]
    df = df.ffill()
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
    """
    Blended portfolio vol estimate.
    Uses weighted returns → rolling vol.
    """
    available = [t for t in weights if t in log_ret_df.columns]
    if not available:
        return 0.15

    # Weighted portfolio returns
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
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_portfolio_backtest(
    tri_df: pd.DataFrame,
    hvr_df: pd.DataFrame,
    hmm_df: pd.DataFrame,
    cfg: StrategyConfig,
    v4: V4Config,
    variant: str,
) -> BacktestResult:
    """
    Multi-asset portfolio backtest.

    Risk engine determines total equity exposure (0–1.0).
    Equity portion split among PORTFOLIO at strategic weights.
    Remainder → BIL.

    Variants:
      "p_baseline"    — vol target + binary MA200 cap
      "p_2level"      — vol target + 2-level MA200 cap
      "p_consensus"   — vol target + HVR/MA200/HMM consensus
      "p_b1_cont"     — vol target + continuous HMM
      "p_buyhold"     — static weights, no risk management
    """
    is_buyhold = variant == "p_buyhold"
    use_2level = variant in ("p_2level", "p_consensus")
    use_consensus = variant == "p_consensus"
    use_hmm = variant == "p_b1_cont"

    spy = SIGNAL_TICKER
    simple_ret = tri_df.pct_change().fillna(0.0)
    log_ret = np.log(tri_df / tri_df.shift(1)).fillna(0.0)
    ma200 = tri_df[spy].rolling(200).mean()

    # Portfolio tickers available in data
    port_tickers = [t for t in PORTFOLIO if t in tri_df.columns]
    port_weights = {t: PORTFOLIO[t] for t in port_tickers}
    # Renormalize in case some tickers missing
    total_w = sum(port_weights.values())
    if total_w > 0:
        port_weights = {t: w / total_w for t, w in port_weights.items()}

    all_tickers = sorted(set(port_tickers + [CASH_TICKER]))
    available = [t for t in all_tickers if t in tri_df.columns]

    # Warmup
    warmup = max(cfg.hvr_long_window, 252) + 10
    if use_hmm or use_consensus:
        hmm_first = hmm_df["risk_gate"].first_valid_index()
        if hmm_first is not None:
            warmup = max(warmup, tri_df.index.get_loc(hmm_first))

    dates = simple_ret.index[warmup:]
    capital = cfg.initial_capital
    equity = []
    current_exposure = 0.0  # total equity exposure (0–1)
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

            if is_buyhold:
                # Static weights, no risk management
                target_exposure = 1.0
            else:
                # Vol-target base exposure
                pv = portfolio_blended_vol(log_ret, port_weights, loc, v4)
                scalar = cfg.target_vol / pv if pv > 0 else 1.0
                scalar = min(scalar, cfg.max_leverage)
                target_exposure = scalar

                if use_hmm:
                    # B1 continuous HMM sizing
                    if date in hmm_df.index:
                        p_smooth = hmm_df.loc[date, "p_crisis_smooth"]
                        if np.isnan(p_smooth):
                            p_smooth = 0.0
                        hmm_w = hmm_continuous_weight(p_smooth, v4)
                        target_exposure *= hmm_w

                        # MA200 gating: don't let HMM over-reduce when SPY > MA200
                        spy_price = float(tri_df[spy].iloc[loc])
                        ma200_val = float(ma200.iloc[loc])
                        if not np.isnan(ma200_val) and spy_price > ma200_val:
                            target_exposure = max(target_exposure, v4.b1_gated_floor * scalar)

                elif use_consensus:
                    # HVR + MA200 + HMM vote
                    if date in hvr_df.index and date in hmm_df.index:
                        hmm_crisis = bool(hmm_df.loc[date, "crisis_mode"])
                        hvr_riskoff = (hvr_df.loc[date, "regime"] == "RISK_OFF")
                        spy_price = float(tri_df[spy].iloc[loc])
                        ma200_val = float(ma200.iloc[loc])
                        below_ma = spy_price < ma200_val if not np.isnan(ma200_val) else False

                        votes_off = int(hmm_crisis) + int(hvr_riskoff) + int(below_ma)
                        if votes_off >= 3:
                            target_exposure = min(target_exposure, v4.ma200_deep_cap)
                        elif votes_off == 2:
                            target_exposure = min(target_exposure, v4.ma200_shallow_cap)

                elif use_2level:
                    # 2-level MA200 cap
                    spy_price = float(tri_df[spy].iloc[loc])
                    ma200_val = float(ma200.iloc[loc])
                    risk_cap = two_level_risk_cap(spy_price, ma200_val, v4)
                    target_exposure = min(target_exposure, risk_cap)

                else:
                    # Baseline: binary MA200
                    if date in hvr_df.index:
                        below_ma200 = bool(hvr_df.loc[date, "spy_below_ma200"])
                        if below_ma200:
                            target_exposure = min(target_exposure, cfg.ma200_risk_cap)

                # Cash buffer
                if not is_buyhold:
                    target_exposure = min(target_exposure, 1.0 - v4.min_cash_pct)

            target_exposure = max(0.0, min(target_exposure, 1.0))

            # Deadband + step limit (not for buy-and-hold)
            if not is_buyhold:
                new_exposure = apply_deadband_and_step(
                    current_exposure, target_exposure, days_since_rebal, v4)
            else:
                new_exposure = target_exposure

            # Build target weights
            new_weights = pd.Series(0.0, index=available)
            for t, w in port_weights.items():
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

def run_one_portfolio(
    label: str,
    portfolio: dict,
    tri_df: pd.DataFrame,
    hvr_df: pd.DataFrame,
    hmm_df: pd.DataFrame,
    cfg: StrategyConfig,
    v4: V4Config,
) -> dict:
    """Run all 5 strategies for one portfolio. Returns dict of metrics."""
    global PORTFOLIO
    PORTFOLIO = portfolio

    port_tickers = [t for t in portfolio if t in tri_df.columns]
    port_str = " | ".join(f"{t} {portfolio[t]*100:.0f}%" for t in portfolio)

    print(f"\n{'='*80}")
    print(f"  {label}: {port_str}")
    print(f"{'='*80}")

    missing = [t for t in portfolio if t not in tri_df.columns]
    if missing:
        print(f"  WARNING: Missing: {missing}")

    variants = [
        ("p_buyhold",   "P0 B&H"),
        ("p_baseline",  "P1 baseline"),
        ("p_2level",    "P2 2-level"),
        ("p_consensus", "P3 consensus"),
        ("p_b1_cont",   "P4 B1-HMM"),
    ]

    results = {}
    for vkey, vname in variants:
        try:
            r = run_portfolio_backtest(tri_df, hvr_df, hmm_df, cfg, v4, vkey)
            results[vkey] = r
        except Exception as e:
            print(f"  {vname} FAILED: {e}")

    # Compute metrics
    metrics = {}
    for vkey, vname in variants:
        if vkey not in results:
            continue
        eq = results[vkey].equity_curve
        m = compute_metrics(eq, vname)
        m["trades"] = results[vkey].trades
        m["turnover_annual"] = results[vkey].turnover_annual
        metrics[vkey] = m

    # Print compact table
    print(f"\n  {'Strategy':<16} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>7} {'Calmar':>7} {'Vol':>7} {'TO':>5}")
    print(f"  {'-'*60}")
    for vkey, vname in variants:
        if vkey not in metrics:
            continue
        m = metrics[vkey]
        calmar = m["cagr_raw"] / abs(m["max_dd_raw"]) if abs(m["max_dd_raw"]) > 0 else 0
        print(f"  {vname:<16} {m['sharpe_raw']:>7.2f} {m['cagr']:>8} {m['max_dd']:>7} {calmar:>7.2f} {m['ann_vol']:>7} {results[vkey].turnover_annual:>4.1f}x")

    # Annual for best two
    ranked = sorted(metrics.items(), key=lambda x: -x[1]["sharpe_raw"])
    for vkey, m in ranked[:2]:
        vname = dict(variants).get(vkey, vkey)
        print(f"\n  --- Annual: {vname} ---")
        ab = annual_breakdown(results[vkey].equity_curve)
        print("  " + ab.to_string(index=False).replace("\n", "\n  "))

    return metrics


def main():
    api_key = "FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1"
    cfg = StrategyConfig(
        polygon_api_key=api_key,
        start_date="2015-01-01",
        end_date="2026-02-10",
        trade_freq_days=21,
    )
    v3cfg = V3Config()
    v4 = V4Config()

    # ── 1. FETCH ALL DATA (union of all 3 portfolios) ──
    all_tickers_needed = sorted(set(
        list(PORTFOLIO_B.keys()) + list(PORTFOLIO_B_DEF.keys()) + list(PORTFOLIO_B_TRIM.keys()) +
        [CASH_TICKER, SIGNAL_TICKER]
    ))

    # Temporarily set ALL_TICKERS for fetch
    global ALL_TICKERS
    ALL_TICKERS = all_tickers_needed

    tri_map = fetch_portfolio_data(api_key, cfg.start_date, cfg.end_date)

    print("\nAligning data...")
    tri_df = align_portfolio_data(tri_map)
    print(f"  {tri_df.shape[0]} days x {tri_df.shape[1]} tickers")
    print(f"  Range: {tri_df.index[0].date()} to {tri_df.index[-1].date()}")

    if SIGNAL_TICKER not in tri_df.columns:
        print(f"ERROR: {SIGNAL_TICKER} not in data"); sys.exit(1)

    # ── 2. REGIMES (shared — both use SPY) ──
    print("\nComputing HVR regime (SPY)...")
    hvr_df = compute_regime_series(tri_df[SIGNAL_TICKER], cfg)
    n_off = (hvr_df["regime"] == "RISK_OFF").sum()
    print(f"  RISK_OFF: {n_off}/{len(hvr_df)} ({n_off/len(hvr_df)*100:.1f}%)")

    print("\nComputing HMM regime (SPY, walk-forward)...")
    hmm_df = compute_hmm_regime_series(tri_df[SIGNAL_TICKER], v3cfg)
    hmm_valid = hmm_df["p_crisis"].notna().sum()
    n_crisis = hmm_df["crisis_mode"].sum()
    print(f"  HMM valid: {hmm_valid}/{len(hmm_df)}, Crisis: {n_crisis}")

    # ── 3. RUN ALL 3 PORTFOLIOS ──
    metrics_b = run_one_portfolio("PORTFOLIO B (baseline)", PORTFOLIO_B,
                                  tri_df, hvr_df, hmm_df, cfg, v4)
    metrics_b_def = run_one_portfolio("PORTFOLIO B-Def (robust defense)", PORTFOLIO_B_DEF,
                                      tri_df, hvr_df, hmm_df, cfg, v4)
    metrics_b_trim = run_one_portfolio("PORTFOLIO B-Trim (theme trim)", PORTFOLIO_B_TRIM,
                                       tri_df, hvr_df, hmm_df, cfg, v4)

    # ── 4. BENCHMARKS ──
    # Use common earliest start
    earliest = tri_df.index[max(cfg.hvr_long_window, 252) + 10]
    hmm_first = hmm_df["risk_gate"].first_valid_index()
    if hmm_first is not None:
        earliest = max(earliest, hmm_first)

    benchmarks = {}
    for ticker in ["QQQ", "SPY"]:
        if ticker in tri_df.columns:
            p = tri_df[ticker][tri_df.index >= earliest]
            norm = p / p.iloc[0] * cfg.initial_capital
            benchmarks[ticker] = compute_metrics(norm, f"{ticker} B&H")

    # ── 5. THREE-WAY COMPARISON ──
    print("\n" + "=" * 80)
    print("THREE-WAY COMPARISON: B vs B-Def vs B-Trim")
    print("=" * 80)

    pb_str = " | ".join(f"{t} {PORTFOLIO_B[t]*100:.1f}%" for t in PORTFOLIO_B)
    pbdef_str = " | ".join(f"{t} {PORTFOLIO_B_DEF[t]*100:.1f}%" for t in PORTFOLIO_B_DEF)
    pbtrim_str = " | ".join(f"{t} {PORTFOLIO_B_TRIM[t]*100:.1f}%" for t in PORTFOLIO_B_TRIM)
    print(f"\n  B:       {pb_str}")
    print(f"  B-Def:   {pbdef_str}")
    print(f"  B-Trim:  {pbtrim_str}")

    strats = ["p_buyhold", "p_baseline", "p_2level", "p_consensus", "p_b1_cont"]
    snames = {"p_buyhold": "P0 B&H", "p_baseline": "P1 base",
              "p_2level": "P2 2lvl", "p_consensus": "P3 cons",
              "p_b1_cont": "P4 HMM"}

    print(f"\n  {'Strat':<10} {'B Sharpe':>8} {'Def Sharpe':>10} {'Trim Sharpe':>11} {'B CAGR':>8} {'Def CAGR':>8} {'Trim CAGR':>8} {'B MaxDD':>8} {'Def DD':>7} {'Trim DD':>7}")
    print(f"  {'-'*93}")

    for sk in strats:
        mb = metrics_b.get(sk, {})
        mbd = metrics_b_def.get(sk, {})
        mbt = metrics_b_trim.get(sk, {})
        if not mb or not mbd or not mbt:
            continue
        sb = mb["sharpe_raw"]; sbd = mbd["sharpe_raw"]; sbt = mbt["sharpe_raw"]
        print(f"  {snames[sk]:<10} {sb:>8.2f} {sbd:>10.2f} {sbt:>11.2f} {mb['cagr']:>8} {mbd['cagr']:>8} {mbt['cagr']:>8} {mb['max_dd']:>8} {mbd['max_dd']:>7} {mbt['max_dd']:>7}")

    # Benchmarks
    for ticker in ["QQQ", "SPY"]:
        if ticker in benchmarks:
            bm = benchmarks[ticker]
            print(f"  {ticker:<10} {bm['sharpe_raw']:>8.2f} {bm['sharpe_raw']:>10.2f} {bm['sharpe_raw']:>11.2f} {bm['cagr']:>8} {bm['cagr']:>8} {bm['cagr']:>8} {bm['max_dd']:>8}          REF")

    # Winners
    best_b = max(metrics_b.values(), key=lambda m: m["sharpe_raw"])
    best_bd = max(metrics_b_def.values(), key=lambda m: m["sharpe_raw"])
    best_bt = max(metrics_b_trim.values(), key=lambda m: m["sharpe_raw"])

    print(f"\n  B best:      Sharpe={best_b['sharpe_raw']:.2f}, CAGR={best_b['cagr']}, MaxDD={best_b['max_dd']}")
    print(f"  B-Def best:  Sharpe={best_bd['sharpe_raw']:.2f}, CAGR={best_bd['cagr']}, MaxDD={best_bd['max_dd']}")
    print(f"  B-Trim best: Sharpe={best_bt['sharpe_raw']:.2f}, CAGR={best_bt['cagr']}, MaxDD={best_bt['max_dd']}")

    sharpes = [("B", best_b["sharpe_raw"]), ("B-Def", best_bd["sharpe_raw"]), ("B-Trim", best_bt["sharpe_raw"])]
    sharpes_sorted = sorted(sharpes, key=lambda x: -x[1])

    print(f"\n  Ranking by Sharpe:")
    for i, (name, sharpe) in enumerate(sharpes_sorted, 1):
        delta = sharpes_sorted[0][1] - sharpe if i > 1 else 0
        print(f"    {i}. {name:<8} Sharpe={sharpe:.2f} {('-' + f'{delta:.2f}' if delta > 0 else '')}".rstrip())

    # ── 6. SAVE ──
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)

    summary = {
        "portfolio_b": {t: f"{w*100:.1f}%" for t, w in PORTFOLIO_B.items()},
        "portfolio_b_def": {t: f"{w*100:.1f}%" for t, w in PORTFOLIO_B_DEF.items()},
        "portfolio_b_trim": {t: f"{w*100:.1f}%" for t, w in PORTFOLIO_B_TRIM.items()},
        "metrics_b": {k: {kk: vv for kk, vv in v.items() if kk != "name"}
                      for k, v in metrics_b.items()},
        "metrics_b_def": {k: {kk: vv for kk, vv in v.items() if kk != "name"}
                          for k, v in metrics_b_def.items()},
        "metrics_b_trim": {k: {kk: vv for kk, vv in v.items() if kk != "name"}
                           for k, v in metrics_b_trim.items()},
        "benchmarks": {k: v for k, v in benchmarks.items()},
        "config": {"cost_bps": v4.cost_bps, "rebalance": f"{v4.rebalance_freq_days}d"},
    }
    with open(output_dir / "etf_rotation_v5_ab_comparison.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
