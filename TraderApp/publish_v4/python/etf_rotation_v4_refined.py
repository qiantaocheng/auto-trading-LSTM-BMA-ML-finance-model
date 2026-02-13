#!/usr/bin/env python3
"""
ETF Rotation V4: Refined A1 / B1 / B2 Improvements
=====================================================

LINE 1 — Strengthen A1 (SPY + HVR risk engine):
  V1) A1-baseline       : current best (monthly rebalance, vol target, MA200 guard)
  V2) A1-deadband       : + exposure deadband (±5%), max step (±15%), min hold (5d)
  V3) A1-blendvol       : + blended vol (0.7*vol20 + 0.3*vol60), vol floor/cap
  V4) A1-2level         : + 2-level risk cap (100/60/30 based on MA200 distance)
  V5) A1-production     : V2 + V3 + V4 combined ("production candidate")
  V6) A1-consensus      : V5 + HVR & MA200 & HMM consensus filter

LINE 2 — Improve B1 (SPY + HMM):
  V7) B1-continuous     : continuous position sizing (p0=0.55, p1=0.75, w_min=0.3, w_max=1.0)
  V8) B1-gated          : continuous + MA200 gating (>MA200 → max drop to 0.6)

LINE 2 — Improve B2 (Soft + HRP + HMM):
  V9) B2-tilt           : HRP × momentum tilt (α=0.5) + candidate hysteresis (6-in/8-out)
  V10) B2-tilt+floor    : V9 + top2 min weight 10%, max weight 35%

Benchmark: QQQ (instead of SPY)
Transaction cost: 10 bps per rebalance (realistic ETF cost)
"""
from __future__ import annotations

import json
import math
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
    fetch_all_data,
    align_data,
    compute_regime_series,
    compute_hrp_weights,
    apply_vol_targeting,
    compute_metrics,
    annual_breakdown,
    BacktestResult,
    fetch_polygon_daily,
    fetch_polygon_dividends,
    build_total_return_index,
)
from etf_rotation_v2_momentum import (
    V2Config,
    compute_momentum,
    precompute_training_panel,
    ridge_predict_from_panel,
    rank_normalize,
)
from etf_rotation_v3_monthly_hmm import (
    V3Config,
    soft_momentum_weights,
    tilt_hrp_by_momentum,
    compute_hmm_regime_series,
)


# ─────────────────────────────────────────────────────────────────────────────
# V4 CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class V4Config:
    """All V4 refinement parameters."""
    # Monthly rebalance
    rebalance_freq_days: int = 21

    # === A1 DEADBAND / RATE LIMIT ===
    exposure_deadband: float = 0.05       # skip rebalance if |Δw| < 5%
    max_step: float = 0.15               # max position change per rebalance ±15%
    min_hold_days: int = 5               # min hold after rebalance before reverse

    # === A1 BLENDED VOL ===
    vol_blend_short: int = 20
    vol_blend_long: int = 60
    vol_blend_alpha: float = 0.7          # 0.7*vol20 + 0.3*vol60
    vol_floor: float = 0.08              # min annualized vol estimate
    vol_cap: float = 0.40                # max annualized vol estimate

    # === A1 2-LEVEL RISK CAP ===
    # SPY > MA200: full exposure (vol-target)
    # SPY < MA200, not deep: cap at 60%
    # SPY significantly below MA200: cap at 30%
    ma200_shallow_cap: float = 0.60      # shallow deviation cap
    ma200_deep_cap: float = 0.30         # deep deviation cap
    ma200_deep_threshold: float = -0.05  # -5% below MA200 = "deep"

    # === A1 CASH BUFFER ===
    min_cash_pct: float = 0.05           # always hold 5% BIL even in risk-on

    # === B1 CONTINUOUS ===
    b1_w_max: float = 1.0
    b1_w_min: float = 0.30
    b1_p0: float = 0.55                  # start reducing at p_riskoff=0.55
    b1_p1: float = 0.75                  # fully reduced at p_riskoff=0.75

    # === B1 GATED — MA200 override ===
    b1_gated_floor: float = 0.60         # when SPY > MA200, HMM can only drop to 60%

    # === B1 ENHANCED HMM HYSTERESIS ===
    b1_hmm_cooldown: int = 5             # 5 trading days cooldown
    b1_hmm_extreme_override: float = 0.85  # extreme crisis bypasses cooldown

    # === B2 TILT ===
    b2_tilt_alpha: float = 0.5           # HRP^α * mom^(1-α)
    b2_candidate_enter_rank: int = 6     # enter candidate set: rank ≤ 6
    b2_candidate_exit_rank: int = 8      # exit candidate set: rank ≥ 8
    b2_soft_temperature: float = 1.0     # soft weight temperature (tighter than v3's 1.5)

    # === B2 FLOOR/CAP ===
    b2_min_weight_top2: float = 0.10     # top 2 sectors min 10% each
    b2_max_weight: float = 0.35          # max 35% per sector

    # === TRANSACTION COST ===
    cost_bps: float = 10.0               # 10 bps (realistic for ETFs)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: BLENDED VOL ESTIMATE
# ─────────────────────────────────────────────────────────────────────────────

def blended_vol(log_ret: pd.Series, loc: int, v4: V4Config) -> float:
    """Blended vol = alpha*vol_short + (1-alpha)*vol_long, clamped to [floor, cap]."""
    short_w = log_ret.iloc[max(0, loc - v4.vol_blend_short):loc]
    long_w = log_ret.iloc[max(0, loc - v4.vol_blend_long):loc]

    v_short = float(short_w.std() * np.sqrt(252)) if len(short_w) > 5 else 0.15
    v_long = float(long_w.std() * np.sqrt(252)) if len(long_w) > 10 else 0.15

    blended = v4.vol_blend_alpha * v_short + (1 - v4.vol_blend_alpha) * v_long
    return max(v4.vol_floor, min(blended, v4.vol_cap))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: 2-LEVEL RISK CAP
# ─────────────────────────────────────────────────────────────────────────────

def two_level_risk_cap(spy_price: float, ma200: float, v4: V4Config) -> float:
    """
    Returns max exposure based on SPY vs MA200 distance.
      SPY > MA200        → 1.0 (full)
      SPY < MA200, >-5%  → 0.60
      SPY < MA200, <-5%  → 0.30
    """
    if np.isnan(ma200) or ma200 <= 0:
        return 1.0
    deviation = (spy_price - ma200) / ma200
    if deviation >= 0:
        return 1.0
    elif deviation > v4.ma200_deep_threshold:
        return v4.ma200_shallow_cap
    else:
        return v4.ma200_deep_cap


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: CONTINUOUS HMM POSITION SIZING
# ─────────────────────────────────────────────────────────────────────────────

def hmm_continuous_weight(p_riskoff: float, v4: V4Config) -> float:
    """
    Continuous position sizing based on HMM p_riskoff.
    w = w_max - (w_max - w_min) * clamp((p_riskoff - p0)/(p1 - p0), 0, 1)
    """
    if np.isnan(p_riskoff):
        return v4.b1_w_max
    t = (p_riskoff - v4.b1_p0) / (v4.b1_p1 - v4.b1_p0)
    t = max(0.0, min(1.0, t))
    return v4.b1_w_max - (v4.b1_w_max - v4.b1_w_min) * t


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: DEADBAND + MAX STEP + MIN HOLD
# ─────────────────────────────────────────────────────────────────────────────

def apply_deadband_and_step(
    current_spy_w: float,
    target_spy_w: float,
    days_since_last_rebal: int,
    v4: V4Config,
) -> float:
    """Apply deadband, max step, and min hold to smooth position changes."""
    delta = target_spy_w - current_spy_w

    # Min hold: skip if too recent
    if days_since_last_rebal < v4.min_hold_days and abs(delta) < 0.30:
        return current_spy_w

    # Deadband: skip small changes
    if abs(delta) < v4.exposure_deadband:
        return current_spy_w

    # Max step: clamp position change
    if delta > v4.max_step:
        return current_spy_w + v4.max_step
    elif delta < -v4.max_step:
        return current_spy_w - v4.max_step

    return target_spy_w


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: B2 CANDIDATE SET HYSTERESIS
# ─────────────────────────────────────────────────────────────────────────────

def update_candidate_set(
    mom_scores: pd.Series,
    prev_candidates: set,
    v4: V4Config,
) -> set:
    """
    Update candidate sector set with hysteresis to prevent churn.
    Enter: rank ≤ 6 (enter_rank)
    Exit: rank ≥ 8 (exit_rank)
    """
    ranked = mom_scores.rank(ascending=False, method="min")
    new_candidates = set()

    for ticker in mom_scores.index:
        rank = ranked[ticker]
        if ticker in prev_candidates:
            # Existing member: keep unless rank drops to exit_rank or worse
            if rank <= v4.b2_candidate_exit_rank:
                new_candidates.add(ticker)
        else:
            # New candidate: enter only if rank ≤ enter_rank
            if rank <= v4.b2_candidate_enter_rank:
                new_candidates.add(ticker)

    # Ensure at least 3 candidates
    if len(new_candidates) < 3:
        top = ranked.nsmallest(3).index
        new_candidates = set(top)

    return new_candidates


def apply_weight_floors_caps(weights: pd.Series, mom_scores: pd.Series, v4: V4Config) -> pd.Series:
    """Apply min weight for top-2 and max weight cap."""
    if len(weights) == 0:
        return weights

    ranked = mom_scores.reindex(weights.index).rank(ascending=False, method="min")
    top2 = ranked.nsmallest(2).index

    w = weights.copy()

    # Apply max cap
    w = w.clip(upper=v4.b2_max_weight)

    # Apply min floor for top 2
    for t in top2:
        if t in w.index and w[t] < v4.b2_min_weight_top2:
            w[t] = v4.b2_min_weight_top2

    # Renormalize
    if w.sum() > 0:
        w = w / w.sum()

    return w


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE — A1 VARIANTS (V1-V6)
# ─────────────────────────────────────────────────────────────────────────────

def run_a1_backtest(
    tri_df: pd.DataFrame,
    hvr_regime_df: pd.DataFrame,
    hmm_regime_df: pd.DataFrame,
    cfg: StrategyConfig,
    v4: V4Config,
    variant: str,
) -> BacktestResult:
    """
    A1 variants — SPY + BIL with various refinements.
    Variants: "a1_baseline", "a1_deadband", "a1_blendvol", "a1_2level",
              "a1_production", "a1_consensus"
    """
    use_deadband = variant in ("a1_deadband", "a1_production", "a1_consensus")
    use_blendvol = variant in ("a1_blendvol", "a1_production", "a1_consensus")
    use_2level = variant in ("a1_2level", "a1_production", "a1_consensus")
    use_consensus = variant == "a1_consensus"
    use_cash_buffer = variant in ("a1_production", "a1_consensus")

    spy = cfg.spy_ticker
    simple_ret = tri_df.pct_change().fillna(0.0)
    log_ret = np.log(tri_df / tri_df.shift(1)).fillna(0.0)
    spy_lr = log_ret[spy]

    # MA200
    ma200 = tri_df[spy].rolling(200).mean()

    warmup = max(cfg.hvr_long_window, 252) + 10
    if use_consensus:
        hmm_first = hmm_regime_df["risk_gate"].first_valid_index()
        if hmm_first is not None:
            warmup = max(warmup, tri_df.index.get_loc(hmm_first))

    dates = simple_ret.index[warmup:]
    capital = cfg.initial_capital
    equity = []
    current_spy_w = 0.0
    total_trades = 0
    total_turnover = 0.0
    last_rebal_idx = -999
    days_since_rebal = 999

    for idx, date in enumerate(dates):
        days_since_rebal = idx - last_rebal_idx

        # Monthly rebalance check
        should_rebal = (days_since_rebal >= v4.rebalance_freq_days) or (idx == 0)

        if should_rebal:
            loc = log_ret.index.get_loc(date)

            # Step 1: Vol estimate
            if use_blendvol:
                rv = blended_vol(spy_lr, loc, v4)
            else:
                window = spy_lr.iloc[max(0, loc - 21):loc]
                rv = float(window.std() * np.sqrt(252)) if len(window) > 5 else 0.15

            # Step 2: Base vol-target weight
            scalar = cfg.target_vol / rv if rv > 0 else 1.0
            scalar = min(scalar, cfg.max_leverage)
            target_w = scalar

            # Step 3: Risk cap
            if use_2level:
                spy_price = float(tri_df[spy].iloc[loc])
                ma200_val = float(ma200.iloc[loc])
                risk_cap = two_level_risk_cap(spy_price, ma200_val, v4)
                target_w = min(target_w, risk_cap)
            else:
                # Original binary MA200 cap
                if date in hvr_regime_df.index:
                    below_ma200 = bool(hvr_regime_df.loc[date, "spy_below_ma200"])
                    if below_ma200:
                        target_w = min(target_w, cfg.ma200_risk_cap)

            # Step 4: Consensus filter (HVR + MA200 + HMM vote)
            if use_consensus and date in hmm_regime_df.index and date in hvr_regime_df.index:
                hmm_crisis = bool(hmm_regime_df.loc[date, "crisis_mode"])
                hvr_riskoff = (hvr_regime_df.loc[date, "regime"] == "RISK_OFF")
                spy_price = float(tri_df[spy].iloc[loc])
                ma200_val = float(ma200.iloc[loc])
                below_ma = spy_price < ma200_val if not np.isnan(ma200_val) else False

                # Count risk-off votes
                votes_off = int(hmm_crisis) + int(hvr_riskoff) + int(below_ma)

                if votes_off >= 3:
                    # All agree risk-off → aggressive de-risk
                    target_w = min(target_w, v4.ma200_deep_cap)
                elif votes_off == 2:
                    # Two agree → moderate de-risk
                    target_w = min(target_w, v4.ma200_shallow_cap)
                # votes_off <= 1: keep vol-target weight (no override)

            # Cash buffer
            if use_cash_buffer:
                target_w = min(target_w, 1.0 - v4.min_cash_pct)

            target_w = max(0.0, min(target_w, 1.0))

            # Step 5: Deadband + max step + min hold
            if use_deadband:
                new_spy_w = apply_deadband_and_step(
                    current_spy_w, target_w, days_since_rebal, v4)
            else:
                new_spy_w = target_w

            # Apply trade
            delta = abs(new_spy_w - current_spy_w)
            if delta > 0.01 or idx == 0:
                bil_w = max(0.0, 1.0 - new_spy_w)
                turnover = delta / 2
                cost = turnover * capital * v4.cost_bps / 10000
                capital -= cost
                total_turnover += turnover
                total_trades += 1
                current_spy_w = new_spy_w
                last_rebal_idx = idx

        # Daily P&L
        if date in simple_ret.index:
            spy_ret = simple_ret.loc[date].get(spy, 0.0)
            bil_ret = simple_ret.loc[date].get("BIL", 0.0)
            bil_w = max(0.0, 1.0 - current_spy_w)
            port_ret = current_spy_w * spy_ret + bil_w * bil_ret
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
# BACKTEST ENGINE — B1 VARIANTS (V7-V8)
# ─────────────────────────────────────────────────────────────────────────────

def run_b1_backtest(
    tri_df: pd.DataFrame,
    hmm_regime_df: pd.DataFrame,
    cfg: StrategyConfig,
    v4: V4Config,
    variant: str,
) -> BacktestResult:
    """
    B1 variants — SPY + HMM with continuous position sizing.
    Variants: "b1_continuous", "b1_gated"
    """
    use_gating = variant == "b1_gated"

    spy = cfg.spy_ticker
    simple_ret = tri_df.pct_change().fillna(0.0)
    log_ret = np.log(tri_df / tri_df.shift(1)).fillna(0.0)
    spy_lr = log_ret[spy]
    ma200 = tri_df[spy].rolling(200).mean()

    # HMM warmup
    hmm_first = hmm_regime_df["risk_gate"].first_valid_index()
    warmup = max(cfg.hvr_long_window, 252) + 10
    if hmm_first is not None:
        warmup = max(warmup, tri_df.index.get_loc(hmm_first))

    dates = simple_ret.index[warmup:]
    capital = cfg.initial_capital
    equity = []
    current_spy_w = 0.0
    total_trades = 0
    total_turnover = 0.0
    last_rebal_idx = -999

    for idx, date in enumerate(dates):
        should_rebal = (idx - last_rebal_idx >= v4.rebalance_freq_days) or (idx == 0)

        if should_rebal and date in hmm_regime_df.index:
            loc = log_ret.index.get_loc(date)

            # Vol estimate (blended)
            rv = blended_vol(spy_lr, loc, v4)

            # Vol-target base weight
            scalar = cfg.target_vol / rv if rv > 0 else 1.0
            scalar = min(scalar, cfg.max_leverage)

            # HMM continuous sizing
            p_smooth = hmm_regime_df.loc[date, "p_crisis_smooth"]
            if np.isnan(p_smooth):
                p_smooth = 0.0
            hmm_w = hmm_continuous_weight(p_smooth, v4)

            target_w = scalar * hmm_w

            # MA200 gating: if SPY > MA200, don't let HMM drop below floor
            if use_gating:
                spy_price = float(tri_df[spy].iloc[loc])
                ma200_val = float(ma200.iloc[loc])
                if not np.isnan(ma200_val) and spy_price > ma200_val:
                    target_w = max(target_w, v4.b1_gated_floor * scalar)

            # Cash buffer
            target_w = min(target_w, 1.0 - v4.min_cash_pct)
            target_w = max(0.0, min(target_w, 1.0))

            # Deadband + step limit
            new_spy_w = apply_deadband_and_step(
                current_spy_w, target_w, idx - last_rebal_idx, v4)

            delta = abs(new_spy_w - current_spy_w)
            if delta > 0.01 or idx == 0:
                turnover = delta / 2
                cost = turnover * capital * v4.cost_bps / 10000
                capital -= cost
                total_turnover += turnover
                total_trades += 1
                current_spy_w = new_spy_w
                last_rebal_idx = idx

        # Daily P&L
        if date in simple_ret.index:
            spy_ret = simple_ret.loc[date].get(spy, 0.0)
            bil_ret = simple_ret.loc[date].get("BIL", 0.0)
            bil_w = max(0.0, 1.0 - current_spy_w)
            port_ret = current_spy_w * spy_ret + bil_w * bil_ret
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
# BACKTEST ENGINE — B2 VARIANTS (V9-V10)
# ─────────────────────────────────────────────────────────────────────────────

def run_b2_backtest(
    tri_df: pd.DataFrame,
    hmm_regime_df: pd.DataFrame,
    cfg: StrategyConfig,
    v2cfg: V2Config,
    v4: V4Config,
    variant: str,
) -> BacktestResult:
    """
    B2 variants — Soft Momentum + HRP tilt + HMM with improvements.
    Variants: "b2_tilt", "b2_tilt_floor"
    """
    use_floor = variant == "b2_tilt_floor"

    sectors = [e for e in cfg.risk_on_etfs if e in tri_df.columns]
    all_etfs = sorted(set(
        sectors +
        [e for e in cfg.risk_off_etfs if e in tri_df.columns] +
        [cfg.spy_ticker]
    ))
    available = [e for e in all_etfs if e in tri_df.columns]

    simple_ret = tri_df.pct_change().fillna(0.0)
    log_ret = np.log(tri_df / tri_df.shift(1)).fillna(0.0)

    # HMM warmup
    hmm_first = hmm_regime_df["risk_gate"].first_valid_index()
    warmup = max(cfg.hvr_long_window, v2cfg.mom_lookback, 252) + 10
    if hmm_first is not None:
        warmup = max(warmup, tri_df.index.get_loc(hmm_first))

    dates = simple_ret.index[warmup:]
    capital = cfg.initial_capital
    equity = []
    current_weights = pd.Series(0.0, index=available)
    if "BIL" in current_weights.index:
        current_weights["BIL"] = 1.0

    total_trades = 0
    total_turnover = 0.0
    last_rebal_idx = -999
    candidate_set = set()

    for idx, date in enumerate(dates):
        if date not in hmm_regime_df.index:
            equity.append({"date": date, "equity": capital})
            continue

        rg = float(hmm_regime_df.loc[date, "risk_gate"])
        crisis = bool(hmm_regime_df.loc[date, "crisis_mode"])
        regime_risk_on = not crisis and rg > 0.1

        date_loc = tri_df.index.get_loc(date)
        should_rebal = (idx - last_rebal_idx >= v4.rebalance_freq_days) or (idx == 0)

        if should_rebal:
            new_weights = pd.Series(0.0, index=available)

            if not regime_risk_on:
                # Risk-off: defensive pool
                pool = [e for e in cfg.risk_off_etfs if e in tri_df.columns]
                pool += [e for e in cfg.dual_purpose if e in tri_df.columns and e not in pool]
                if len(pool) >= 2:
                    loc = log_ret.index.get_loc(date)
                    start = max(0, loc - cfg.hrp_lookback)
                    rw = log_ret.iloc[start:loc][[c for c in pool if c in log_ret.columns]]
                    if rw.shape[0] > 30 and rw.shape[1] >= 2:
                        pw = compute_hrp_weights(rw, cfg.hrp_linkage)
                    else:
                        pw = pd.Series(1.0 / len(pool), index=pool)
                elif len(pool) == 1:
                    pw = pd.Series(1.0, index=pool)
                else:
                    pw = pd.Series({"BIL": 1.0})
                for t, w in pw.items():
                    if t in new_weights.index:
                        new_weights[t] = w
            else:
                # Risk-on: momentum with candidate hysteresis

                # Step 1: Momentum scores
                mom = compute_momentum(tri_df, sectors, date_loc,
                                       v2cfg.mom_lookback, v2cfg.mom_skip)

                # Step 2: Update candidate set with hysteresis
                candidate_set = update_candidate_set(mom, candidate_set, v4)
                candidates = [s for s in candidate_set if s in mom.index]
                if len(candidates) == 0:
                    candidates = list(mom.nlargest(5).index)

                candidate_mom = mom.reindex(candidates).dropna()

                # Step 3: Soft weights within candidate set
                soft_w = soft_momentum_weights(candidate_mom, v4.b2_soft_temperature)

                if len(soft_w) >= 2:
                    # Step 4: HRP weights
                    loc = log_ret.index.get_loc(date)
                    start = max(0, loc - cfg.hrp_lookback)
                    hrp_cols = [c for c in soft_w.index if c in log_ret.columns]
                    rw = log_ret.iloc[start:loc][hrp_cols]

                    if rw.shape[0] > 30 and rw.shape[1] >= 2:
                        hrp_w = compute_hrp_weights(rw, cfg.hrp_linkage)
                    else:
                        hrp_w = pd.Series(1.0 / len(hrp_cols), index=hrp_cols)

                    # Step 5: Power tilt (HRP^α × mom^(1-α))
                    common = hrp_w.index.intersection(soft_w.index)
                    hrp_r = hrp_w.reindex(common, fill_value=0.0).clip(lower=1e-10)
                    soft_r = soft_w.reindex(common, fill_value=0.0).clip(lower=1e-10)
                    alpha = v4.b2_tilt_alpha
                    tilted = (hrp_r ** alpha) * (soft_r ** (1 - alpha))
                    pool_w = tilted / tilted.sum()

                    # Step 6: Floor/cap (if variant requires it)
                    if use_floor:
                        pool_w = apply_weight_floors_caps(pool_w, candidate_mom, v4)
                else:
                    pool_w = soft_w if len(soft_w) > 0 else pd.Series({"BIL": 1.0})

                # Step 7: HMM risk scaling (continuous)
                p_smooth = hmm_regime_df.loc[date, "p_crisis_smooth"]
                if not np.isnan(p_smooth):
                    hmm_w = hmm_continuous_weight(p_smooth, v4)
                else:
                    hmm_w = 1.0
                pool_w = pool_w * hmm_w

                # Step 8: Vol targeting
                pool_cols = [c for c in pool_w.index if c in log_ret.columns]
                if len(pool_cols) >= 2:
                    from sklearn.covariance import LedoitWolf
                    loc = log_ret.index.get_loc(date)
                    start = max(0, loc - cfg.hrp_lookback)
                    rw = log_ret.iloc[start:loc][pool_cols]
                    if rw.shape[0] > 30:
                        lw = LedoitWolf().fit(rw.values)
                        pool_w, _ = apply_vol_targeting(
                            pool_w, lw.covariance_, pool_cols,
                            cfg.target_vol, cfg.max_leverage,
                        )

                for t, w in pool_w.items():
                    if t in new_weights.index:
                        new_weights[t] = w

                # BIL absorbs remainder
                remainder = max(0.0, 1.0 - new_weights.sum())
                if "BIL" in new_weights.index:
                    new_weights["BIL"] += remainder

            # Normalize
            if new_weights.sum() > 0:
                new_weights = new_weights / new_weights.sum()

            # Transaction
            diff = (new_weights - current_weights).abs().sum()
            if diff > 0.03 or idx == 0:
                turnover = diff / 2
                cost = turnover * capital * v4.cost_bps / 10000
                capital -= cost
                total_turnover += turnover
                total_trades += 1
                current_weights = new_weights
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
# QQQ BENCHMARK FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_qqq_benchmark(cfg: StrategyConfig) -> pd.Series:
    """Fetch QQQ total return index."""
    print("\nFetching QQQ benchmark...")
    df = fetch_polygon_daily("QQQ", cfg.start_date, cfg.end_date, cfg.polygon_api_key, adjusted=False)
    if df.empty:
        print("  QQQ data not available!")
        return pd.Series(dtype=float)
    divs = fetch_polygon_dividends("QQQ", cfg.start_date, cfg.end_date, cfg.polygon_api_key)
    tri = build_total_return_index(df["Close"], divs)
    print(f"  QQQ: {len(df)} bars, {len(divs) if not divs.empty else 0} divs")
    return tri


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = StrategyConfig(
        polygon_api_key="FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1",
        start_date="2015-01-01",
        end_date="2026-02-10",
        trade_freq_days=21,
    )
    v2cfg = V2Config(rebalance_freq_days=21)
    v3cfg = V3Config()
    v4 = V4Config()

    # ── 1. FETCH & ALIGN ──
    raw_prices, tri_series = fetch_all_data(cfg)
    if len(tri_series) < 5:
        print("ERROR: insufficient data"); sys.exit(1)

    print("\nAligning data...")
    tri_df = align_data(tri_series)
    print(f"  {tri_df.shape[0]} days x {tri_df.shape[1]} tickers")
    print(f"  Range: {tri_df.index[0].date()} to {tri_df.index[-1].date()}")

    # ── 2. QQQ BENCHMARK ──
    qqq_tri = fetch_qqq_benchmark(cfg)

    # ── 3. HVR REGIME ──
    print("\nComputing HVR regime...")
    hvr_df = compute_regime_series(tri_df[cfg.spy_ticker], cfg)
    n_off = (hvr_df["regime"] == "RISK_OFF").sum()
    print(f"  RISK_OFF: {n_off}/{len(hvr_df)} ({n_off/len(hvr_df)*100:.1f}%)")

    # ── 4. HMM REGIME ──
    print("\nComputing HMM regime (walk-forward)...")
    hmm_df = compute_hmm_regime_series(tri_df[cfg.spy_ticker], v3cfg)
    hmm_valid = hmm_df["p_crisis"].notna().sum()
    n_crisis = hmm_df["crisis_mode"].sum()
    print(f"  HMM valid days: {hmm_valid}/{len(hmm_df)}")
    print(f"  Crisis mode: {n_crisis}/{hmm_valid} ({n_crisis/max(1,hmm_valid)*100:.1f}%)")

    # ── 5. RUN 10 VARIANTS ──
    variants = [
        # A1 line (6 variants)
        ("a1_baseline",   "V1) A1 baseline",         "a1"),
        ("a1_deadband",   "V2) A1 + deadband/step",  "a1"),
        ("a1_blendvol",   "V3) A1 + blended vol",    "a1"),
        ("a1_2level",     "V4) A1 + 2-level cap",    "a1"),
        ("a1_production", "V5) A1 production",        "a1"),
        ("a1_consensus",  "V6) A1 + consensus",       "a1"),
        # B1 line (2 variants)
        ("b1_continuous", "V7) B1 continuous",         "b1"),
        ("b1_gated",      "V8) B1 gated",             "b1"),
        # B2 line (2 variants)
        ("b2_tilt",       "V9) B2 tilt",              "b2"),
        ("b2_tilt_floor", "V10) B2 tilt+floor",       "b2"),
    ]

    results = {}
    for vkey, vname, vtype in variants:
        print(f"\nRunning: {vname}...")
        try:
            if vtype == "a1":
                r = run_a1_backtest(tri_df, hvr_df, hmm_df, cfg, v4, vkey)
            elif vtype == "b1":
                r = run_b1_backtest(tri_df, hmm_df, cfg, v4, vkey)
            elif vtype == "b2":
                r = run_b2_backtest(tri_df, hmm_df, cfg, v2cfg, v4, vkey)
            else:
                continue

            results[vkey] = r
            print(f"  Trades: {r.trades}, Turnover: {r.turnover_annual:.2f}x")
            print(f"  Period: {r.equity_curve.index[0].date()} to {r.equity_curve.index[-1].date()}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    if not results:
        print("\nAll variants failed."); sys.exit(1)

    # ── 6. METRICS ──
    print("\n" + "=" * 80)
    print("ETF ROTATION V4: REFINED A1 / B1 / B2")
    print(f"Transaction cost: {v4.cost_bps} bps per rebalance")
    print("=" * 80)

    all_metrics = []
    for vkey, vname, _ in variants:
        if vkey not in results:
            continue
        eq = results[vkey].equity_curve
        period = f"{eq.index[0].year}-{eq.index[-1].year}"
        m = compute_metrics(eq, vname, period)
        m["trades"] = results[vkey].trades
        m["turnover"] = f"{results[vkey].turnover_annual:.2f}x"
        all_metrics.append(m)
        print(f"\n  {vname} ({period}):")
        for k, v in m.items():
            if k not in ("name", "cagr_raw", "sharpe_raw", "max_dd_raw"):
                print(f"    {k}: {v}")

    # QQQ benchmark
    earliest = min(r.equity_curve.index[0] for r in results.values())
    if not qqq_tri.empty:
        qqq_p = qqq_tri[qqq_tri.index >= earliest]
        qqq_norm = qqq_p / qqq_p.iloc[0] * cfg.initial_capital
        qqq_m = compute_metrics(qqq_norm, "QQQ Buy&Hold",
                                f"{qqq_norm.index[0].year}-{qqq_norm.index[-1].year}")
        print(f"\n  QQQ Buy&Hold:")
        for k, v in qqq_m.items():
            if k not in ("name", "cagr_raw", "sharpe_raw", "max_dd_raw"):
                print(f"    {k}: {v}")
    else:
        qqq_m = {}
        qqq_norm = pd.Series(dtype=float)

    # SPY benchmark (also include for comparison)
    spy_eq = tri_df[cfg.spy_ticker]
    spy_p = spy_eq[spy_eq.index >= earliest]
    spy_norm = spy_p / spy_p.iloc[0] * cfg.initial_capital
    spy_m = compute_metrics(spy_norm, "SPY Buy&Hold",
                            f"{spy_norm.index[0].year}-{spy_norm.index[-1].year}")
    print(f"\n  SPY Buy&Hold:")
    for k, v in spy_m.items():
        if k not in ("name", "cagr_raw", "sharpe_raw", "max_dd_raw"):
            print(f"    {k}: {v}")

    # ── COMMON PERIOD ──
    if len(results) >= 2:
        common_start = max(r.equity_curve.index[0] for r in results.values())
        common_end = min(r.equity_curve.index[-1] for r in results.values())
        print(f"\n--- COMMON PERIOD ({common_start.date()} to {common_end.date()}) ---")

        rows = []
        for vkey, vname, _ in variants:
            if vkey not in results:
                continue
            eq = results[vkey].equity_curve
            ec = eq[(eq.index >= common_start) & (eq.index <= common_end)]
            if len(ec) > 20:
                m = compute_metrics(ec, vname)
                rows.append({
                    "name": vname,
                    "sharpe": m["sharpe_raw"],
                    "cagr": m["cagr"],
                    "max_dd": m["max_dd"],
                    "turnover": f"{results[vkey].turnover_annual:.2f}x",
                })
                print(f"  {vname}: Sharpe={m['sharpe_raw']:.2f}, CAGR={m['cagr']}, MaxDD={m['max_dd']}, Turnover={results[vkey].turnover_annual:.2f}x")

        # QQQ on common period
        if not qqq_norm.empty:
            qqq_c = qqq_norm[(qqq_norm.index >= common_start) & (qqq_norm.index <= common_end)]
            if len(qqq_c) > 20:
                qm = compute_metrics(qqq_c, "QQQ B&H")
                print(f"  QQQ B&H: Sharpe={qm['sharpe_raw']:.2f}, CAGR={qm['cagr']}, MaxDD={qm['max_dd']}")

        spy_c = spy_norm[(spy_norm.index >= common_start) & (spy_norm.index <= common_end)]
        if len(spy_c) > 20:
            sm = compute_metrics(spy_c, "SPY B&H")
            print(f"  SPY B&H: Sharpe={sm['sharpe_raw']:.2f}, CAGR={sm['cagr']}, MaxDD={sm['max_dd']}")

    # ── ANNUAL BREAKDOWNS ──
    for tag, label in [
        ("a1_baseline",   "V1 A1 baseline"),
        ("a1_production", "V5 A1 production"),
        ("a1_consensus",  "V6 A1 consensus"),
        ("b1_continuous", "V7 B1 continuous"),
        ("b1_gated",      "V8 B1 gated"),
        ("b2_tilt",       "V9 B2 tilt"),
        ("b2_tilt_floor", "V10 B2 tilt+floor"),
    ]:
        if tag in results:
            print(f"\n--- ANNUAL: {label} ---")
            ab = annual_breakdown(results[tag].equity_curve)
            print(ab.to_string(index=False))

    # ── 7. SAVE ──
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)

    eq_df = pd.DataFrame({k: v.equity_curve for k, v in results.items()})
    if not qqq_norm.empty:
        eq_df["QQQ"] = qqq_norm.reindex(eq_df.index)
    eq_df["SPY"] = spy_norm.reindex(eq_df.index)
    eq_df.to_csv(output_dir / "etf_rotation_v4_equity_curves.csv")

    summary = {
        "all_metrics": all_metrics,
        "qqq_benchmark": qqq_m if qqq_m else "N/A",
        "spy_benchmark": spy_m,
        "config": {
            "rebalance": f"monthly ({v4.rebalance_freq_days}d)",
            "cost_bps": v4.cost_bps,
            "exposure_deadband": v4.exposure_deadband,
            "max_step": v4.max_step,
            "min_hold_days": v4.min_hold_days,
            "vol_blend": f"{v4.vol_blend_alpha}*vol{v4.vol_blend_short} + {1-v4.vol_blend_alpha}*vol{v4.vol_blend_long}",
            "vol_floor_cap": f"[{v4.vol_floor}, {v4.vol_cap}]",
            "2level_caps": f"full/shallow={v4.ma200_shallow_cap}/deep={v4.ma200_deep_cap} (threshold={v4.ma200_deep_threshold})",
            "b1_continuous": f"p0={v4.b1_p0}, p1={v4.b1_p1}, w_min={v4.b1_w_min}, w_max={v4.b1_w_max}",
            "b1_gated_floor": v4.b1_gated_floor,
            "b2_tilt_alpha": v4.b2_tilt_alpha,
            "b2_candidate_hysteresis": f"enter≤{v4.b2_candidate_enter_rank}, exit≥{v4.b2_candidate_exit_rank}",
            "b2_floor_cap": f"top2≥{v4.b2_min_weight_top2}, max≤{v4.b2_max_weight}",
        },
    }
    with open(output_dir / "etf_rotation_v4_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")

    # ── 8. VERDICT ──
    print("\n" + "=" * 80)
    print("VERDICT (ranked by Sharpe)")
    print("=" * 80)

    ranked = sorted(
        [(k, compute_metrics(v.equity_curve, k)["sharpe_raw"]) for k, v in results.items()],
        key=lambda x: -x[1],
    )
    name_map = {vkey: vname for vkey, vname, _ in variants}
    for vkey, sharpe in ranked:
        vname = name_map.get(vkey, vkey)
        m = compute_metrics(results[vkey].equity_curve, vkey)
        print(f"  {vname}: Sharpe={sharpe:.2f}, CAGR={m['cagr']}, MaxDD={m['max_dd']}, Turnover={results[vkey].turnover_annual:.1f}x")

    if qqq_m:
        print(f"  QQQ B&H: Sharpe={qqq_m.get('sharpe_raw', 0):.2f}, CAGR={qqq_m.get('cagr', 'N/A')}, MaxDD={qqq_m.get('max_dd', 'N/A')}")
    print(f"  SPY B&H: Sharpe={spy_m['sharpe_raw']:.2f}, CAGR={spy_m['cagr']}, MaxDD={spy_m['max_dd']}")

    best_key, best_sharpe = ranked[0]
    best_name = name_map.get(best_key, best_key)
    print(f"\n  BEST: {best_name} (Sharpe={best_sharpe:.2f})")

    # Show improvement over baseline
    baseline_sharpe = next((s for k, s in ranked if k == "a1_baseline"), None)
    if baseline_sharpe is not None and best_key != "a1_baseline":
        delta = best_sharpe - baseline_sharpe
        print(f"  vs A1 baseline: {'+' if delta >= 0 else ''}{delta:.2f} Sharpe")


if __name__ == "__main__":
    main()
