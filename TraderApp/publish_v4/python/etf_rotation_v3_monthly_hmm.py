#!/usr/bin/env python3
"""
ETF Rotation V3: Monthly Rebalance + Soft Weights + HMM
========================================================

Changes from V2:
  1. Monthly rebalance (21 trading days) instead of weekly
  2. Soft weighting: continuous momentum-proportional weights instead of hard Top-K
  3. HMM integration: 3-state GaussianHMM walk-forward for risk gating

8 variants:
  Group A (HVR regime):
    A1) SPY Vol Target + MA200 guard (monthly)
    A2) Soft Momentum + HRP tilt + HVR risk gating (monthly)
    A3) Soft Momentum + Ridge ML fusion + HRP tilt + HVR risk gating (monthly)
    A4) Soft Momentum equal weight + HVR risk gating (monthly)

  Group B (HMM regime):
    B1) SPY Vol Target + HMM risk_gate (monthly)
    B2) Soft Momentum + HRP tilt + HMM risk gating (monthly)
    B3) Soft Momentum + Ridge ML fusion + HRP tilt + HMM risk gating (monthly)
    B4) Soft Momentum equal weight + HMM risk gating (monthly)
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
)
from etf_rotation_v2_momentum import (
    V2Config,
    FEATURE_NAMES,
    compute_momentum,
    precompute_training_panel,
    ridge_predict_from_panel,
    rank_normalize,
)


# ─────────────────────────────────────────────────────────────────────────────
# V3 CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class V3Config:
    # Monthly rebalance
    rebalance_freq_days: int = 21

    # Soft weighting temperature (higher = more concentrated on top momentum)
    soft_weight_temperature: float = 1.5

    # HMM parameters
    hmm_n_states: int = 3
    hmm_train_window: int = 500   # 2yr rolling window (limited by 5yr data)
    hmm_retrain_freq: int = 21    # retrain monthly
    hmm_ema_span: int = 4         # EMA smoothing for p_crisis
    hmm_gamma: int = 2            # risk_gate exponent

    # HMM hysteresis
    hmm_crisis_enter: float = 0.70
    hmm_crisis_exit: float = 0.40
    hmm_crisis_confirm_days: int = 2
    hmm_cooldown_days: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# SOFT WEIGHTING
# ─────────────────────────────────────────────────────────────────────────────

def soft_momentum_weights(scores: pd.Series, temperature: float = 1.5) -> pd.Series:
    """
    Convert momentum scores to continuous soft weights via exponential.

    w_i = exp(z_i * temperature) / sum(exp(z_j * temperature))

    Higher temperature → more concentrated on winners.
    temperature=0 → equal weight.
    """
    valid = scores.dropna()
    if len(valid) == 0:
        return pd.Series(dtype=float)
    if len(valid) == 1:
        return pd.Series(1.0, index=valid.index)

    mu = valid.mean()
    sigma = valid.std()
    if sigma < 1e-10:
        return pd.Series(1.0 / len(valid), index=valid.index)

    z = (valid - mu) / sigma
    exp_w = np.exp(z * temperature)
    return exp_w / exp_w.sum()


def tilt_hrp_by_momentum(hrp_weights: pd.Series, soft_weights: pd.Series) -> pd.Series:
    """
    Tilt HRP weights by momentum soft weights.
    final_w = HRP_w * soft_w / sum(HRP_w * soft_w)

    Preserves risk-parity structure while overweighting momentum winners.
    """
    common = hrp_weights.index.intersection(soft_weights.index)
    if len(common) == 0:
        return hrp_weights

    hrp = hrp_weights.reindex(common, fill_value=0.0)
    soft = soft_weights.reindex(common, fill_value=0.0)

    tilted = hrp * soft
    total = tilted.sum()
    if total < 1e-10:
        return hrp / hrp.sum()
    return tilted / total


# ─────────────────────────────────────────────────────────────────────────────
# HMM REGIME DETECTION (walk-forward for backtest)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HmmState:
    """Persistent HMM state for hysteresis."""
    p_crisis_history: List[float] = field(default_factory=list)
    crisis_mode: bool = False
    crisis_confirm_days: int = 0
    safe_confirm_days: int = 0
    cooldown_remaining: int = 0


def compute_hmm_regime_series(
    spy_tri: pd.Series,
    v3cfg: V3Config,
) -> pd.DataFrame:
    """
    Walk-forward HMM regime detection.

    At each retrain date:
      1. Train 3-state GaussianHMM on SPY {log_return, 10d_vol}
      2. Get p_crisis for each day since last retrain
      3. Apply EMA smoothing + hysteresis
      4. Compute risk_gate = (1 - p_crisis_smooth)^gamma

    Returns DataFrame with: p_crisis, p_crisis_smooth, risk_gate, crisis_mode, hmm_state
    """
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    # Compute features
    log_ret = np.log(spy_tri / spy_tri.shift(1))
    vol_10d = log_ret.rolling(10, min_periods=10).std()

    features = pd.DataFrame({
        "log_ret": log_ret,
        "vol_10d": vol_10d,
    }, index=spy_tri.index).dropna()

    # Initialize output
    n = len(features)
    result_data = {
        "p_crisis": np.full(n, np.nan),
        "p_crisis_smooth": np.full(n, np.nan),
        "risk_gate": np.ones(n),
        "crisis_mode": np.zeros(n, dtype=bool),
        "hmm_state": ["SAFE"] * n,
    }

    min_train = max(v3cfg.hmm_train_window, 100)
    if n < min_train + 10:
        print(f"  Warning: HMM needs {min_train} days, only have {n}")
        result_df = pd.DataFrame(result_data, index=features.index)
        return result_df

    hmm_state = HmmState()
    last_train_idx = -999
    model = None
    scaler = None
    crisis_state_idx = 0

    for i in range(min_train, n):
        # Retrain periodically
        if i - last_train_idx >= v3cfg.hmm_retrain_freq or model is None:
            train_start = max(0, i - v3cfg.hmm_train_window)
            X_raw = features.iloc[train_start:i][["log_ret", "vol_10d"]].values

            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)

            model = GaussianHMM(
                n_components=v3cfg.hmm_n_states,
                covariance_type="full",
                n_iter=100,
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

        # Get p_crisis for current day
        if model is None or scaler is None:
            continue

        x_now = scaler.transform(features.iloc[i:i+1][["log_ret", "vol_10d"]].values)
        try:
            posteriors = model.predict_proba(x_now)
            p_crisis = float(posteriors[0, crisis_state_idx])
        except Exception:
            p_crisis = 0.0

        # EMA smoothing
        hmm_state.p_crisis_history.append(p_crisis)
        if len(hmm_state.p_crisis_history) > 30:
            hmm_state.p_crisis_history = hmm_state.p_crisis_history[-30:]

        ema_series = pd.Series(hmm_state.p_crisis_history)
        p_smooth = float(ema_series.ewm(span=v3cfg.hmm_ema_span, adjust=False).mean().iloc[-1])

        # Risk gate
        rg = (1.0 - p_smooth) ** v3cfg.hmm_gamma
        if rg < 0.05:
            rg = 0.0

        # Hysteresis for crisis mode
        if hmm_state.cooldown_remaining > 0:
            hmm_state.cooldown_remaining -= 1

        if not hmm_state.crisis_mode:
            if p_smooth >= v3cfg.hmm_crisis_enter and hmm_state.cooldown_remaining == 0:
                hmm_state.crisis_confirm_days += 1
                if hmm_state.crisis_confirm_days >= v3cfg.hmm_crisis_confirm_days:
                    hmm_state.crisis_mode = True
                    hmm_state.crisis_confirm_days = 0
                    hmm_state.safe_confirm_days = 0
            else:
                hmm_state.crisis_confirm_days = 0
        else:
            if p_smooth <= v3cfg.hmm_crisis_exit:
                hmm_state.safe_confirm_days += 1
                if hmm_state.safe_confirm_days >= 2:
                    hmm_state.crisis_mode = False
                    hmm_state.safe_confirm_days = 0
                    hmm_state.cooldown_remaining = v3cfg.hmm_cooldown_days
            else:
                hmm_state.safe_confirm_days = 0

        # Get state label
        try:
            state_pred = model.predict(x_now)
            state_label = label_map.get(int(state_pred[0]), "SAFE")
        except Exception:
            state_label = "SAFE"

        result_data["p_crisis"][i] = p_crisis
        result_data["p_crisis_smooth"][i] = p_smooth
        result_data["risk_gate"][i] = rg
        result_data["crisis_mode"][i] = hmm_state.crisis_mode
        result_data["hmm_state"][i] = state_label

    result_df = pd.DataFrame(result_data, index=features.index)
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE (8 variants)
# ─────────────────────────────────────────────────────────────────────────────

def run_v3_backtest(
    tri_df: pd.DataFrame,
    hvr_regime_df: pd.DataFrame,
    hmm_regime_df: pd.DataFrame,
    cfg: StrategyConfig,
    v2cfg: V2Config,
    v3cfg: V3Config,
    variant: str,
    panel: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """
    Unified backtest for 8 variants.

    Variants:
      HVR group: "a1_spy_hvr", "a2_soft_hrp_hvr", "a3_soft_ml_hvr", "a4_soft_eq_hvr"
      HMM group: "b1_spy_hmm", "b2_soft_hrp_hmm", "b3_soft_ml_hmm", "b4_soft_eq_hmm"
    """
    use_hmm = variant.startswith("b")
    regime_df = hmm_regime_df if use_hmm else hvr_regime_df
    is_spy_only = "spy" in variant
    use_hrp = "hrp" in variant
    use_ml = "ml" in variant
    use_equal = "eq" in variant

    sectors = [e for e in cfg.risk_on_etfs if e in tri_df.columns]
    all_etfs = sorted(set(
        sectors +
        [e for e in cfg.risk_off_etfs if e in tri_df.columns] +
        [cfg.spy_ticker]
    ))
    available = [e for e in all_etfs if e in tri_df.columns]

    simple_ret = tri_df.pct_change().fillna(0.0)
    log_ret_df = np.log(tri_df / tri_df.shift(1)).fillna(0.0)

    # Warmup
    warmup = max(cfg.hvr_long_window, v2cfg.mom_lookback, 252) + 10
    if use_hmm:
        # HMM needs extra warmup for training
        hmm_first_valid = hmm_regime_df["risk_gate"].first_valid_index()
        if hmm_first_valid is not None:
            hmm_warmup = tri_df.index.get_loc(hmm_first_valid)
            warmup = max(warmup, hmm_warmup)
    if use_ml:
        warmup = max(warmup, 252 + v2cfg.ml_train_window + v2cfg.ml_label_horizon)

    if warmup >= len(simple_ret):
        raise ValueError(f"Not enough data for {variant}")

    dates = simple_ret.index[warmup:]
    if len(dates) < 30:
        raise ValueError(f"Insufficient dates for {variant}: {len(dates)}")

    capital = cfg.initial_capital
    equity = []
    current_weights = pd.Series(0.0, index=available)
    if "BIL" in current_weights.index:
        current_weights["BIL"] = 1.0

    total_trades = 0
    total_turnover = 0.0
    last_rebal_idx = -999
    last_ml_train_idx = -999
    ml_predictions = None

    for idx, date in enumerate(dates):
        if date not in regime_df.index:
            equity.append({"date": date, "equity": capital})
            continue

        # ── Get regime signal ──
        if use_hmm:
            rg = float(regime_df.loc[date, "risk_gate"])
            crisis = bool(regime_df.loc[date, "crisis_mode"])
            regime_risk_on = not crisis and rg > 0.1
        else:
            regime_val = hvr_regime_df.loc[date, "regime"]
            r_cap = float(hvr_regime_df.loc[date, "risk_cap"])
            regime_risk_on = (regime_val == "RISK_ON")

        date_loc = tri_df.index.get_loc(date)

        # ── Monthly rebalance ──
        should_rebal = (idx - last_rebal_idx >= v3cfg.rebalance_freq_days) or (idx == 0)

        if should_rebal:
            new_weights = pd.Series(0.0, index=available)

            if is_spy_only:
                # ── SPY variants (A1/B1) ──
                spy_lr = log_ret_df[cfg.spy_ticker]
                loc = log_ret_df.index.get_loc(date)
                window = spy_lr.iloc[max(0, loc - 21):loc]
                rv = float(window.std() * np.sqrt(252)) if len(window) > 5 else 0.15
                scalar = cfg.target_vol / rv if rv > 0 else 1.0
                scalar = min(scalar, cfg.max_leverage)

                if use_hmm:
                    spy_w = scalar * rg
                else:
                    below_ma200 = bool(hvr_regime_df.loc[date, "spy_below_ma200"])
                    spy_w = min(scalar, cfg.ma200_risk_cap) if below_ma200 else scalar

                spy_w = max(0.0, min(spy_w, 1.0))
                if cfg.spy_ticker in new_weights.index:
                    new_weights[cfg.spy_ticker] = spy_w
                if "BIL" in new_weights.index:
                    new_weights["BIL"] = max(0, 1.0 - spy_w)

            elif not regime_risk_on:
                # ── Risk-off: defensive pool ──
                pool = [e for e in cfg.risk_off_etfs if e in tri_df.columns]
                pool += [e for e in cfg.dual_purpose if e in tri_df.columns and e not in pool]

                if len(pool) >= 2:
                    loc = log_ret_df.index.get_loc(date)
                    start = max(0, loc - cfg.hrp_lookback)
                    rw = log_ret_df.iloc[start:loc][[c for c in pool if c in log_ret_df.columns]]
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
                # ── Risk-on: soft momentum sector allocation ──

                # Step 1: Momentum scores for all sectors
                mom = compute_momentum(tri_df, sectors, date_loc,
                                       v2cfg.mom_lookback, v2cfg.mom_skip)

                # Step 2: Score fusion with ML (if applicable)
                if use_ml and panel is not None:
                    if idx - last_ml_train_idx >= v2cfg.ml_retrain_freq:
                        ml_predictions = ridge_predict_from_panel(
                            panel, tri_df, log_ret_df, sectors,
                            cfg.spy_ticker, date_loc, v2cfg,
                        )
                        last_ml_train_idx = idx

                    if ml_predictions is not None:
                        mom_rank = rank_normalize(mom)
                        ml_rank = rank_normalize(ml_predictions)
                        common = mom_rank.index.intersection(ml_rank.index)
                        fused = (v2cfg.fusion_mom_weight * mom_rank.reindex(common, fill_value=0.5) +
                                 v2cfg.fusion_ml_weight * ml_rank.reindex(common, fill_value=0.5))
                        selection_scores = fused
                    else:
                        selection_scores = mom
                else:
                    selection_scores = mom

                # Step 3: Soft weights (continuous, not hard Top-K)
                soft_w = soft_momentum_weights(selection_scores, v3cfg.soft_weight_temperature)
                if soft_w.empty:
                    if "BIL" in new_weights.index:
                        new_weights["BIL"] = 1.0
                else:
                    if use_hrp and len(soft_w) >= 2:
                        # HRP + momentum tilt
                        loc = log_ret_df.index.get_loc(date)
                        start = max(0, loc - cfg.hrp_lookback)
                        hrp_cols = [c for c in soft_w.index if c in log_ret_df.columns]
                        rw = log_ret_df.iloc[start:loc][hrp_cols]
                        if rw.shape[0] > 30 and rw.shape[1] >= 2:
                            hrp_w = compute_hrp_weights(rw, cfg.hrp_linkage)
                            pool_w = tilt_hrp_by_momentum(hrp_w, soft_w)
                        else:
                            pool_w = soft_w
                    elif use_equal:
                        pool_w = soft_w
                    else:
                        pool_w = soft_w

                    # Step 4: Risk scaling
                    if use_hmm:
                        pool_w = pool_w * rg
                    else:
                        if r_cap < 1.0:
                            pool_w = pool_w * r_cap

                    # Step 5: Vol targeting
                    pool_cols = [c for c in pool_w.index if c in log_ret_df.columns]
                    if len(pool_cols) >= 2:
                        from sklearn.covariance import LedoitWolf
                        loc = log_ret_df.index.get_loc(date)
                        start = max(0, loc - cfg.hrp_lookback)
                        rw = log_ret_df.iloc[start:loc][pool_cols]
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

            # Transaction costs
            diff = (new_weights - current_weights).abs().sum()
            if diff > cfg.min_trade_threshold or idx == 0:
                turnover = diff / 2
                cost = max(turnover * capital * cfg.cost_bps / 10000,
                           cfg.min_cost_per_trade if turnover > 0 else 0)
                capital -= cost
                total_turnover += turnover
                total_trades += 1
                current_weights = new_weights
                last_rebal_idx = idx

        # ── Daily P&L ──
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
    cfg = StrategyConfig(
        polygon_api_key="FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1",
        start_date="2015-01-01",
        end_date="2026-02-07",
        trade_freq_days=21,
    )
    v2cfg = V2Config(rebalance_freq_days=21)
    v3cfg = V3Config()

    # ── 1. FETCH & ALIGN ──
    raw_prices, tri_series = fetch_all_data(cfg)
    if len(tri_series) < 5:
        print("ERROR: insufficient data"); sys.exit(1)

    print("\nAligning data...")
    tri_df = align_data(tri_series)
    print(f"  {tri_df.shape[0]} days x {tri_df.shape[1]} tickers")
    print(f"  Range: {tri_df.index[0].date()} to {tri_df.index[-1].date()}")

    # ── 2. HVR REGIME ──
    print("\nComputing HVR regime...")
    hvr_df = compute_regime_series(tri_df[cfg.spy_ticker], cfg)
    n_off = (hvr_df["regime"] == "RISK_OFF").sum()
    print(f"  RISK_OFF: {n_off}/{len(hvr_df)} ({n_off/len(hvr_df)*100:.1f}%)")

    # ── 3. HMM REGIME (walk-forward) ──
    print("\nComputing HMM regime (walk-forward)...")
    hmm_df = compute_hmm_regime_series(tri_df[cfg.spy_ticker], v3cfg)
    n_crisis = hmm_df["crisis_mode"].sum()
    hmm_valid = hmm_df["p_crisis"].notna().sum()
    print(f"  HMM valid days: {hmm_valid}/{len(hmm_df)}")
    print(f"  Crisis mode days: {n_crisis}/{hmm_valid} ({n_crisis/max(1,hmm_valid)*100:.1f}%)")
    mean_rg = hmm_df.loc[hmm_df["risk_gate"] < 1.0, "risk_gate"].mean()
    print(f"  Mean risk_gate (when < 1): {mean_rg:.3f}" if not np.isnan(mean_rg) else "  Mean risk_gate: N/A")

    # ── 4. ML PANEL ──
    sectors = [e for e in cfg.risk_on_etfs if e in tri_df.columns]
    print(f"\nPrecomputing ML panel for {len(sectors)} sectors...")
    panel = precompute_training_panel(
        tri_df, sectors, cfg.spy_ticker,
        step=v2cfg.ml_retrain_freq,
        label_horizon=v2cfg.ml_label_horizon,
    )
    print(f"  Panel: {len(panel)} rows")

    # ── 5. RUN 8 VARIANTS ──
    variants = [
        ("a1_spy_hvr",      "A1) SPY VolTarget + MA200 (monthly)"),
        ("a2_soft_hrp_hvr",  "A2) Soft Mom + HRP tilt + HVR (monthly)"),
        ("a3_soft_ml_hvr",   "A3) Soft Mom + ML + HRP tilt + HVR (monthly)"),
        ("a4_soft_eq_hvr",   "A4) Soft Mom EqWt + HVR (monthly)"),
        ("b1_spy_hmm",      "B1) SPY VolTarget + HMM (monthly)"),
        ("b2_soft_hrp_hmm",  "B2) Soft Mom + HRP tilt + HMM (monthly)"),
        ("b3_soft_ml_hmm",   "B3) Soft Mom + ML + HRP tilt + HMM (monthly)"),
        ("b4_soft_eq_hmm",   "B4) Soft Mom EqWt + HMM (monthly)"),
    ]

    results = {}
    for vkey, vname in variants:
        print(f"\nRunning: {vname}...")
        try:
            r = run_v3_backtest(
                tri_df, hvr_df, hmm_df, cfg, v2cfg, v3cfg,
                vkey, panel=panel,
            )
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
    print("ETF ROTATION V3: MONTHLY + SOFT WEIGHTS + HMM")
    print("=" * 80)

    # Per-variant metrics
    all_metrics = []
    for vkey, vname in variants:
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

    # SPY benchmark
    earliest = min(r.equity_curve.index[0] for r in results.values())
    spy_eq = tri_df[cfg.spy_ticker]
    spy_p = spy_eq[spy_eq.index >= earliest]
    spy_norm = spy_p / spy_p.iloc[0] * cfg.initial_capital
    spy_m = compute_metrics(spy_norm, "SPY Buy&Hold",
                            f"{spy_norm.index[0].year}-{spy_norm.index[-1].year}")
    print(f"\n  SPY Buy&Hold:")
    for k, v in spy_m.items():
        if k not in ("name", "cagr_raw", "sharpe_raw", "max_dd_raw"):
            print(f"    {k}: {v}")

    # ── Common period (all 8 variants overlap) ──
    if len(results) >= 2:
        common_start = max(r.equity_curve.index[0] for r in results.values())
        common_end = min(r.equity_curve.index[-1] for r in results.values())
        print(f"\n--- COMMON PERIOD ({common_start.date()} to {common_end.date()}) ---")

        for vkey, vname in variants:
            if vkey not in results:
                continue
            eq = results[vkey].equity_curve
            ec = eq[(eq.index >= common_start) & (eq.index <= common_end)]
            if len(ec) > 20:
                m = compute_metrics(ec, vname)
                print(f"  {vname}: Sharpe={m['sharpe_raw']:.2f}, CAGR={m['cagr']}, MaxDD={m['max_dd']}, Turnover={results[vkey].turnover_annual:.2f}x")

        spy_c = spy_norm[(spy_norm.index >= common_start) & (spy_norm.index <= common_end)]
        if len(spy_c) > 20:
            sc = compute_metrics(spy_c, "SPY")
            print(f"  SPY B&H: Sharpe={sc['sharpe_raw']:.2f}, CAGR={sc['cagr']}, MaxDD={sc['max_dd']}")

    # Annual breakdowns
    for tag, label in [("a2_soft_hrp_hvr", "A2 Soft+HRP+HVR"),
                       ("b2_soft_hrp_hmm", "B2 Soft+HRP+HMM"),
                       ("a1_spy_hvr", "A1 SPY+HVR"),
                       ("b1_spy_hmm", "B1 SPY+HMM")]:
        if tag in results:
            print(f"\n--- ANNUAL: {label} ---")
            ab = annual_breakdown(results[tag].equity_curve)
            print(ab.to_string(index=False))

    # ── 7. SAVE ──
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)

    eq_df = pd.DataFrame({k: v.equity_curve for k, v in results.items()})
    eq_df["SPY"] = spy_norm.reindex(eq_df.index)
    eq_df.to_csv(output_dir / "etf_rotation_v3_equity_curves.csv")

    summary = {
        "all_metrics": all_metrics,
        "spy_benchmark": spy_m,
        "config": {
            "rebalance": f"monthly ({v3cfg.rebalance_freq_days}d)",
            "soft_weight_temp": v3cfg.soft_weight_temperature,
            "hmm_train_window": f"{v3cfg.hmm_train_window}d",
            "hmm_retrain_freq": f"{v3cfg.hmm_retrain_freq}d",
            "hmm_crisis_thresholds": f"enter={v3cfg.hmm_crisis_enter}, exit={v3cfg.hmm_crisis_exit}",
            "momentum": f"12-1 (lookback={v2cfg.mom_lookback}, skip={v2cfg.mom_skip})",
            "ml": f"Ridge walk-forward, train={v2cfg.ml_train_window}d, label=T+{v2cfg.ml_label_horizon}",
            "fusion": f"{v2cfg.fusion_mom_weight}*mom + {v2cfg.fusion_ml_weight}*ml",
            "target_vol": cfg.target_vol,
            "cost_bps": cfg.cost_bps,
        },
    }
    with open(output_dir / "etf_rotation_v3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")

    # ── 8. VERDICT ──
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    ranked = sorted(
        [(k, compute_metrics(v.equity_curve, k)["sharpe_raw"]) for k, v in results.items()],
        key=lambda x: -x[1],
    )
    for vkey, sharpe in ranked:
        vname = dict(variants).get(vkey, vkey)
        m = compute_metrics(results[vkey].equity_curve, vkey)
        print(f"  {vname}: Sharpe={sharpe:.2f}, CAGR={m['cagr']}, MaxDD={m['max_dd']}, Turnover={results[vkey].turnover_annual:.1f}x")

    print(f"  SPY B&H: Sharpe={spy_m['sharpe_raw']:.2f}, CAGR={spy_m['cagr']}, MaxDD={spy_m['max_dd']}")

    best_key, best_sharpe = ranked[0]
    best_name = dict(variants).get(best_key, best_key)
    print(f"\n  Best: {best_name} (Sharpe={best_sharpe:.2f})")

    if best_sharpe > 0.5:
        print("  >>> RECOMMEND: Deployable strategy.")
    elif best_sharpe > 0.3:
        print("  >>> CAUTIOUS: Modest edge. Paper trade first.")
    else:
        print("  >>> NOT RECOMMENDED: Insufficient edge.")


if __name__ == "__main__":
    main()
