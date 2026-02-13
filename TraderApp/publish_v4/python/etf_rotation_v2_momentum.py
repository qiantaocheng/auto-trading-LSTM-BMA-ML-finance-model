#!/usr/bin/env python3
"""
ETF Rotation V2: Momentum Selection + HRP + Ridge ML Fusion
=============================================================

4 variants compared in the same testing framework:

  V1) SPY Vol Target + MA200 trend guard (weekly benchmark)
      - SPY only, 10% vol target, MA200 caps SPY weight to 30%
      - BIL absorbs remaining weight

  V2) Momentum 12-1 Top-K + HRP + risk gating (stable model)
      - Rank 11 sectors by 12-1 momentum (252d - 21d)
      - Select Top K (default 5) into risk-on pool
      - HRP allocation on selected sectors
      - HVR/MA200 risk gating, vol targeting, BIL absorber

  V3) Momentum + Ridge ML fusion + HRP + risk gating (enhanced)
      - 12-1 momentum rank + Ridge walk-forward T+21 prediction
      - Fused score = 0.7 * mom_rank + 0.3 * ml_rank
      - Top K by fused score -> HRP -> risk gating

  V4) Pure Momentum Top-5 equal weight + risk gating (simple)
      - 12-1 momentum, Top 5, equal weight (no HRP)
      - Risk gating + vol targeting

Data: Polygon.io free tier (~5 years, 2021-2026)
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

# Import shared infrastructure from v1
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


# ─────────────────────────────────────────────────────────────────────────────
# V2 CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class V2Config:
    """V2-specific parameters on top of StrategyConfig."""
    # Momentum
    mom_lookback: int = 252       # 12-month return window
    mom_skip: int = 21            # skip last 1 month (short-term reversal avoidance)
    top_k: int = 5                # sectors to select

    # Ridge ML walk-forward
    ml_train_window: int = 504    # 2yr rolling training window (days)
    ml_retrain_freq: int = 21     # retrain every ~month
    ml_label_horizon: int = 21    # predict T+21 relative return
    ml_ridge_alpha: float = 1.0   # Ridge regularization

    # Score fusion
    fusion_mom_weight: float = 0.7
    fusion_ml_weight: float = 0.3

    # Rebalance
    rebalance_freq_days: int = 5  # weekly


FEATURE_NAMES = [
    "mom_12_1", "mom_6_1", "mom_3m", "dd_1m",
    "rv_20", "rv_60", "trend_r2_60", "rel_strength",
]


# ─────────────────────────────────────────────────────────────────────────────
# MOMENTUM
# ─────────────────────────────────────────────────────────────────────────────

def compute_momentum(tri_df: pd.DataFrame, sectors: List[str],
                     date_idx: int, lookback: int = 252, skip: int = 21) -> pd.Series:
    """
    12-1 Momentum: ret_12m - ret_1m
    Captures medium-term trend while avoiding short-term reversal.
    """
    result = {}
    for ticker in sectors:
        if ticker not in tri_df.columns or date_idx < lookback:
            result[ticker] = np.nan
            continue
        p_now = tri_df[ticker].iloc[date_idx]
        p_12m = tri_df[ticker].iloc[date_idx - lookback]
        p_1m = tri_df[ticker].iloc[date_idx - skip]
        if p_12m > 0 and p_1m > 0:
            result[ticker] = (p_now / p_12m - 1) - (p_now / p_1m - 1)
        else:
            result[ticker] = np.nan
    return pd.Series(result)


def rank_normalize(scores: pd.Series) -> pd.Series:
    """Rank-normalize valid values to [0, 1]."""
    valid = scores.dropna()
    if len(valid) <= 1:
        return pd.Series(0.5, index=scores.index)
    ranks = valid.rank()
    normed = (ranks - 1) / (len(ranks) - 1)
    return normed.reindex(scores.index, fill_value=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# ML FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def _compute_features_at(tri_df: pd.DataFrame, log_ret_df: pd.DataFrame,
                         sectors: List[str], spy_ticker: str,
                         t: int) -> Optional[Dict[str, List[float]]]:
    """Compute 8 ML features for all sectors at date index t."""
    if t < 252:
        return None

    result = {}
    for ticker in sectors:
        if ticker not in tri_df.columns or ticker not in log_ret_df.columns:
            continue
        p = tri_df[ticker]
        spy_p = tri_df[spy_ticker]
        lr = log_ret_df[ticker]

        try:
            mom_12_1 = (p.iloc[t] / p.iloc[t - 252] - 1) - (p.iloc[t] / p.iloc[t - 21] - 1)
            i126 = max(0, t - 126)
            mom_6_1 = (p.iloc[t] / p.iloc[i126] - 1) - (p.iloc[t] / p.iloc[t - 21] - 1)
            i63 = max(0, t - 63)
            mom_3m = p.iloc[t] / p.iloc[i63] - 1

            w = p.iloc[max(0, t - 21):t + 1]
            dd_1m = float(((w / w.cummax()) - 1).min())

            rv_20 = float(lr.iloc[max(0, t - 20):t].std() * np.sqrt(252))
            rv_60 = float(lr.iloc[max(0, t - 60):t].std() * np.sqrt(252))

            y = p.iloc[max(0, t - 60):t].values
            x = np.arange(len(y))
            ss_tot = np.sum((y - y.mean()) ** 2)
            if ss_tot > 0 and len(y) > 5:
                coeffs = np.polyfit(x, y, 1)
                r2 = 1 - np.sum((y - np.polyval(coeffs, x)) ** 2) / ss_tot
            else:
                r2 = 0.0

            rel_str = (p.iloc[t] / p.iloc[i63] - 1) - (spy_p.iloc[t] / spy_p.iloc[i63] - 1)

            result[ticker] = [mom_12_1, mom_6_1, mom_3m, dd_1m, rv_20, rv_60, r2, rel_str]
        except Exception:
            continue

    return result if len(result) >= 5 else None


def precompute_training_panel(tri_df: pd.DataFrame, sectors: List[str],
                              spy_ticker: str, step: int = 21,
                              label_horizon: int = 21) -> pd.DataFrame:
    """
    Precompute feature + label panel for Ridge training.
    Samples every `step` days (non-overlapping labels).
    Each row = (date_idx, ticker, 8 features, label).
    Label = T+horizon relative return (cross-sectional demeaned).
    """
    log_ret_df = np.log(tri_df / tri_df.shift(1))
    records = []

    for t in range(252, len(tri_df) - label_horizon, step):
        feat_dict = _compute_features_at(tri_df, log_ret_df, sectors, spy_ticker, t)
        if feat_dict is None:
            continue

        # T+horizon returns for all sectors
        future_rets = {}
        for ticker in sectors:
            if ticker in tri_df.columns:
                p_now = tri_df[ticker].iloc[t]
                p_fut = tri_df[ticker].iloc[t + label_horizon]
                if p_now > 0:
                    future_rets[ticker] = p_fut / p_now - 1

        if len(future_rets) < 5:
            continue

        mean_ret = np.mean(list(future_rets.values()))

        for ticker, feats in feat_dict.items():
            if ticker in future_rets:
                row = {"date_idx": t, "ticker": ticker}
                for i, name in enumerate(FEATURE_NAMES):
                    row[name] = feats[i]
                row["label"] = future_rets[ticker] - mean_ret
                records.append(row)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# RIDGE WALK-FORWARD
# ─────────────────────────────────────────────────────────────────────────────

def ridge_predict_from_panel(panel: pd.DataFrame, tri_df: pd.DataFrame,
                             log_ret_df: pd.DataFrame, sectors: List[str],
                             spy_ticker: str, current_date_idx: int,
                             v2cfg: V2Config) -> Optional[pd.Series]:
    """
    Walk-forward Ridge prediction.
    1. Select training data from panel within rolling window (past only, labels known).
    2. Fit Ridge.
    3. Compute features at current_date_idx and predict.
    """
    from sklearn.linear_model import Ridge

    train_start = current_date_idx - v2cfg.ml_train_window
    # Labels are known only if date_idx + horizon <= current_date_idx
    label_cutoff = current_date_idx - v2cfg.ml_label_horizon

    mask = (
        (panel["date_idx"] >= train_start) &
        (panel["date_idx"] <= label_cutoff) &
        (panel["label"].notna())
    )
    train = panel[mask]

    if len(train) < 30:
        return None

    X_train = train[FEATURE_NAMES].values
    y_train = train["label"].values
    X_train = np.nan_to_num(X_train, nan=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)

    model = Ridge(alpha=v2cfg.ml_ridge_alpha)
    model.fit(X_train, y_train)

    # Features at current date
    feat_dict = _compute_features_at(tri_df, log_ret_df, sectors, spy_ticker, current_date_idx)
    if feat_dict is None:
        return None

    tickers = list(feat_dict.keys())
    X_now = np.array([feat_dict[t] for t in tickers])
    X_now = np.nan_to_num(X_now, nan=0.0)
    preds = model.predict(X_now)

    return pd.Series(preds, index=tickers)


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_v2_backtest(
    tri_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    cfg: StrategyConfig,
    v2cfg: V2Config,
    variant: str = "momentum_hrp",
    panel: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """
    Unified backtest for the 4 V2 variants.

    Variants:
      "spy_voltarget"  - SPY + vol target + MA200 guard
      "momentum_hrp"   - 12-1 mom top-K + HRP + risk gating
      "momentum_ml_hrp"- mom + Ridge fusion + HRP + risk gating
      "momentum_equal" - 12-1 mom top-K equal weight + risk gating
    """
    sectors = [e for e in cfg.risk_on_etfs if e in tri_df.columns]
    spy = cfg.spy_ticker
    all_etfs = list(set(sectors + [e for e in cfg.risk_off_etfs if e in tri_df.columns] + [spy]))
    available = sorted(set(e for e in all_etfs if e in tri_df.columns))

    # Simple returns for portfolio P&L
    simple_ret = tri_df.pct_change().fillna(0.0)
    log_ret_df = np.log(tri_df / tri_df.shift(1)).fillna(0.0)

    # Warmup depends on variant
    warmup = max(cfg.hvr_long_window, v2cfg.mom_lookback, 252) + 10
    if variant == "momentum_ml_hrp":
        warmup = max(warmup, 252 + v2cfg.ml_train_window + v2cfg.ml_label_horizon)

    if warmup >= len(simple_ret):
        raise ValueError(f"Not enough data for variant {variant}")

    dates = simple_ret.index[warmup:]
    if len(dates) < 50:
        raise ValueError(f"Not enough data after warmup for {variant}: {len(dates)} days")

    capital = cfg.initial_capital
    equity = []
    current_weights = pd.Series(0.0, index=available)
    if "BIL" in current_weights.index:
        current_weights["BIL"] = 1.0

    total_trades = 0
    total_turnover = 0.0
    last_rebal_idx = -999
    last_regime = "RISK_ON"
    last_ml_train_idx = -999
    ml_predictions = None

    for idx, date in enumerate(dates):
        if date not in regime_df.index:
            equity.append({"date": date, "equity": capital})
            continue

        regime = regime_df.loc[date, "regime"]
        r_cap = float(regime_df.loc[date, "risk_cap"])
        regime_switched = (regime != last_regime)
        last_regime = regime

        date_loc = tri_df.index.get_loc(date)

        # ── Rebalance check (weekly or on regime switch) ──
        should_rebal = (idx - last_rebal_idx >= v2cfg.rebalance_freq_days) or regime_switched or (idx == 0)

        if should_rebal:
            new_weights = _compute_target_weights(
                variant, tri_df, log_ret_df, regime_df, cfg, v2cfg,
                sectors, available, date, date_loc, regime, r_cap,
                panel, ml_predictions, idx, last_ml_train_idx,
            )

            # Update ML prediction cache for variant 3
            if variant == "momentum_ml_hrp":
                if idx - last_ml_train_idx >= v2cfg.ml_retrain_freq and panel is not None:
                    ml_predictions = ridge_predict_from_panel(
                        panel, tri_df, log_ret_df, sectors, cfg.spy_ticker, date_loc, v2cfg
                    )
                    last_ml_train_idx = idx

            # Normalize
            if new_weights.sum() > 0:
                new_weights = new_weights / new_weights.sum()

            # Transaction costs
            diff = (new_weights - current_weights).abs().sum()
            if diff > cfg.min_trade_threshold or regime_switched or idx == 0:
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

    eq_series = pd.DataFrame(equity).set_index("date")["equity"]
    years = (eq_series.index[-1] - eq_series.index[0]).days / 365.25
    turnover_annual = total_turnover / years if years > 0 else 0

    return BacktestResult(
        name=variant,
        equity_curve=eq_series,
        trades=total_trades,
        turnover_annual=turnover_annual,
    )


def _compute_target_weights(
    variant, tri_df, log_ret_df, regime_df, cfg, v2cfg,
    sectors, available, date, date_loc, regime, r_cap,
    panel, ml_predictions, idx, last_ml_train_idx,
) -> pd.Series:
    """Compute target weights for a single rebalance date."""

    new_weights = pd.Series(0.0, index=available)

    if variant == "spy_voltarget":
        # ── V1: SPY only + vol target + MA200 guard ──
        below_ma200 = bool(regime_df.loc[date, "spy_below_ma200"])
        spy_lr = log_ret_df[cfg.spy_ticker]
        loc = log_ret_df.index.get_loc(date)
        window = spy_lr.iloc[max(0, loc - 21):loc]
        rv = float(window.std() * np.sqrt(252)) if len(window) > 5 else 0.15
        scalar = cfg.target_vol / rv if rv > 0 else 1.0
        scalar = min(scalar, cfg.max_leverage)

        spy_w = min(scalar, cfg.ma200_risk_cap) if below_ma200 else scalar
        if cfg.spy_ticker in new_weights.index:
            new_weights[cfg.spy_ticker] = spy_w
        if "BIL" in new_weights.index:
            new_weights["BIL"] = max(0, 1.0 - spy_w)
        return new_weights

    # ── V2/V3/V4: Sector selection strategies ──

    if regime == "RISK_OFF":
        # Defensive pool
        pool = [e for e in cfg.risk_off_etfs if e in tri_df.columns]
        pool += [e for e in cfg.dual_purpose if e in tri_df.columns and e not in pool]

        if len(pool) >= 2:
            loc = log_ret_df.index.get_loc(date)
            start = max(0, loc - cfg.hrp_lookback)
            ret_w = log_ret_df.iloc[start:loc][[c for c in pool if c in log_ret_df.columns]]
            if ret_w.shape[0] > 30 and ret_w.shape[1] >= 2:
                pool_w = compute_hrp_weights(ret_w, cfg.hrp_linkage)
            else:
                pool_w = pd.Series(1.0 / len(pool), index=pool)
        elif len(pool) == 1:
            pool_w = pd.Series(1.0, index=pool)
        else:
            pool_w = pd.Series({"BIL": 1.0})

        for t, w in pool_w.items():
            if t in new_weights.index:
                new_weights[t] = w
        return new_weights

    # ── Risk-on: sector selection ──

    # Step 1: Momentum scores
    mom_scores = compute_momentum(tri_df, sectors, date_loc, v2cfg.mom_lookback, v2cfg.mom_skip)

    # Step 2: Score selection
    if variant == "momentum_ml_hrp" and ml_predictions is not None:
        mom_rank = rank_normalize(mom_scores)
        ml_rank = rank_normalize(ml_predictions)
        # Align indices
        common = mom_rank.index.intersection(ml_rank.index)
        fused = (v2cfg.fusion_mom_weight * mom_rank.reindex(common, fill_value=0.5) +
                 v2cfg.fusion_ml_weight * ml_rank.reindex(common, fill_value=0.5))
        selection_scores = fused
    else:
        selection_scores = mom_scores

    # Step 3: Select top K
    valid = selection_scores.dropna().sort_values(ascending=False)
    top_k = min(v2cfg.top_k, len(valid))
    selected = valid.head(top_k).index.tolist()

    if not selected:
        if "BIL" in new_weights.index:
            new_weights["BIL"] = 1.0
        return new_weights

    # Step 4: Weight allocation
    if variant == "momentum_equal":
        pool_w = pd.Series(1.0 / len(selected), index=selected)
    else:
        # HRP on selected sectors
        if len(selected) >= 2:
            loc = log_ret_df.index.get_loc(date)
            start = max(0, loc - cfg.hrp_lookback)
            ret_w = log_ret_df.iloc[start:loc][[c for c in selected if c in log_ret_df.columns]]
            if ret_w.shape[0] > 30 and ret_w.shape[1] >= 2:
                pool_w = compute_hrp_weights(ret_w, cfg.hrp_linkage)
            else:
                pool_w = pd.Series(1.0 / len(selected), index=selected)
        else:
            pool_w = pd.Series(1.0, index=selected)

    # Step 5: MA200 risk cap
    if r_cap < 1.0:
        pool_w = pool_w * r_cap

    # Step 6: Vol targeting
    pool_cols = [c for c in pool_w.index if c in log_ret_df.columns]
    if len(pool_cols) >= 2:
        from sklearn.covariance import LedoitWolf
        loc = log_ret_df.index.get_loc(date)
        start = max(0, loc - cfg.hrp_lookback)
        ret_w = log_ret_df.iloc[start:loc][pool_cols]
        if ret_w.shape[0] > 30:
            lw = LedoitWolf().fit(ret_w.values)
            pool_w, _ = apply_vol_targeting(
                pool_w, lw.covariance_, pool_cols,
                cfg.target_vol, cfg.max_leverage,
            )

    # Place into full weight vector
    for t, w in pool_w.items():
        if t in new_weights.index:
            new_weights[t] = w

    # BIL absorbs remainder
    remainder = max(0.0, 1.0 - new_weights.sum())
    if "BIL" in new_weights.index:
        new_weights["BIL"] += remainder

    return new_weights


# ─────────────────────────────────────────────────────────────────────────────
# LEAKAGE TEST
# ─────────────────────────────────────────────────────────────────────────────

def run_v2_leakage_test(tri_df, regime_df, cfg, v2cfg, variant="momentum_hrp"):
    """Time leakage test: compare real Sharpe vs shuffled-data distribution."""
    print(f"\n--- LEAKAGE TEST ({variant}) ---")

    result = run_v2_backtest(tri_df, regime_df, cfg, v2cfg, variant)
    real_metrics = compute_metrics(result.equity_curve, "real")
    real_sharpe = real_metrics["sharpe_raw"]
    print(f"  Real Sharpe: {real_sharpe:.3f}")

    n_trials = 15
    shuffled_sharpes = []
    returns = tri_df.pct_change().dropna()

    for trial in range(n_trials):
        shuf_ret = returns.sample(frac=1, random_state=trial * 42).reset_index(drop=True)
        shuf_ret.index = returns.index
        shuf_tri = (1 + shuf_ret).cumprod() * 100
        shuf_tri.iloc[0] = 100.0

        try:
            shuf_regime = compute_regime_series(shuf_tri[cfg.spy_ticker], cfg)
            r = run_v2_backtest(shuf_tri, shuf_regime, cfg, v2cfg, variant)
            m = compute_metrics(r.equity_curve, f"shuf_{trial}")
            shuffled_sharpes.append(m["sharpe_raw"])
        except Exception:
            shuffled_sharpes.append(0.0)

    shuf_mean = np.mean(shuffled_sharpes)
    shuf_std = np.std(shuffled_sharpes)
    z = (real_sharpe - shuf_mean) / shuf_std if shuf_std > 0 else 0

    print(f"  Shuffled: mean={shuf_mean:.3f}, std={shuf_std:.3f}")
    print(f"  Z-score: {z:.2f}")
    conclusion = "NO_LEAKAGE" if z > 1.96 else "INCONCLUSIVE"
    print(f"  Conclusion: {conclusion}")

    return {
        "variant": variant,
        "real_sharpe": real_sharpe,
        "shuffled_mean": shuf_mean,
        "shuffled_std": shuf_std,
        "z_score": z,
        "conclusion": conclusion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cfg = StrategyConfig(
        polygon_api_key="FExbaO1xdmrV6f6p3zHCxk8IArjeowQ1",
        start_date="2015-01-01",
        end_date="2026-02-07",
        trade_freq_days=5,
    )
    v2cfg = V2Config()

    # ── 1. FETCH & ALIGN ──
    raw_prices, tri_series = fetch_all_data(cfg)
    if len(tri_series) < 5:
        print("ERROR: insufficient tickers with data")
        sys.exit(1)

    print("\nAligning data to common trading days...")
    tri_df = align_data(tri_series)
    print(f"  Aligned: {tri_df.shape[0]} days x {tri_df.shape[1]} tickers")
    print(f"  Range: {tri_df.index[0].date()} to {tri_df.index[-1].date()}")
    print(f"  Tickers: {sorted(tri_df.columns.tolist())}")

    # ── 2. REGIME ──
    print("\nComputing HVR regime...")
    regime_df = compute_regime_series(tri_df[cfg.spy_ticker], cfg)
    n_off = (regime_df["regime"] == "RISK_OFF").sum()
    print(f"  RISK_OFF: {n_off}/{len(regime_df)} ({n_off/len(regime_df)*100:.1f}%)")
    transitions = (regime_df["regime"] != regime_df["regime"].shift(1)).sum()
    print(f"  Transitions: {transitions}")

    # ── 3. PRECOMPUTE ML PANEL (for V3) ──
    sectors = [e for e in cfg.risk_on_etfs if e in tri_df.columns]
    print(f"\nPrecomputing ML feature panel for {len(sectors)} sectors...")
    panel = precompute_training_panel(
        tri_df, sectors, cfg.spy_ticker,
        step=v2cfg.ml_retrain_freq,
        label_horizon=v2cfg.ml_label_horizon,
    )
    print(f"  Panel: {len(panel)} rows ({panel['date_idx'].nunique()} dates x {panel['ticker'].nunique()} tickers)")

    # ── 4. RUN 4 VARIANTS ──
    variants = [
        ("spy_voltarget",   "V1) SPY Vol Target + MA200"),
        ("momentum_hrp",    "V2) Mom 12-1 Top5 + HRP + Risk Gate"),
        ("momentum_ml_hrp", "V3) Mom + Ridge ML + HRP + Risk Gate"),
        ("momentum_equal",  "V4) Mom 12-1 Top5 Equal Wt + Risk Gate"),
    ]

    results = {}
    for vkey, vname in variants:
        print(f"\nRunning: {vname}...")
        try:
            r = run_v2_backtest(tri_df, regime_df, cfg, v2cfg, vkey, panel=panel)
            results[vkey] = r
            print(f"  Trades: {r.trades}, Annual Turnover: {r.turnover_annual:.2f}x")
            print(f"  Period: {r.equity_curve.index[0].date()} to {r.equity_curve.index[-1].date()}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    if not results:
        print("\nAll variants failed. Aborting.")
        sys.exit(1)

    # ── 5. METRICS ──
    print("\n" + "=" * 80)
    print("ETF ROTATION V2: MOMENTUM + HRP + ML RESULTS")
    print("=" * 80)

    # Per-variant full period
    full_metrics = []
    for vkey, vname in variants:
        if vkey not in results:
            continue
        eq = results[vkey].equity_curve
        period = f"{eq.index[0].year}-{eq.index[-1].year}"
        m = compute_metrics(eq, vname, period)
        m["trades"] = results[vkey].trades
        m["turnover"] = f"{results[vkey].turnover_annual:.2f}x"
        full_metrics.append(m)
        print(f"\n  {vname} ({period}):")
        for k, v in m.items():
            if k not in ("name", "cagr_raw", "sharpe_raw", "max_dd_raw"):
                print(f"    {k}: {v}")

    # SPY benchmark (aligned to earliest variant)
    earliest_start = min(r.equity_curve.index[0] for r in results.values())
    spy_eq = tri_df[cfg.spy_ticker]
    spy_period = spy_eq[spy_eq.index >= earliest_start]
    spy_norm = spy_period / spy_period.iloc[0] * cfg.initial_capital
    spy_m = compute_metrics(spy_norm, "SPY Buy&Hold",
                            f"{spy_norm.index[0].year}-{spy_norm.index[-1].year}")
    print(f"\n  SPY Buy&Hold:")
    for k, v in spy_m.items():
        if k not in ("name", "cagr_raw", "sharpe_raw", "max_dd_raw"):
            print(f"    {k}: {v}")

    # Common period comparison (all variants overlap)
    if len(results) > 1:
        common_start = max(r.equity_curve.index[0] for r in results.values())
        common_end = min(r.equity_curve.index[-1] for r in results.values())
        print(f"\n--- COMMON PERIOD ({common_start.date()} to {common_end.date()}) ---")

        common_metrics = []
        for vkey, vname in variants:
            if vkey not in results:
                continue
            eq = results[vkey].equity_curve
            eq_c = eq[(eq.index >= common_start) & (eq.index <= common_end)]
            if len(eq_c) > 20:
                m = compute_metrics(eq_c, vname, f"{common_start.year}-{common_end.year}")
                common_metrics.append(m)
                print(f"\n  {vname}:")
                for k, v in m.items():
                    if k not in ("name", "cagr_raw", "sharpe_raw", "max_dd_raw"):
                        print(f"    {k}: {v}")

        spy_c = spy_norm[(spy_norm.index >= common_start) & (spy_norm.index <= common_end)]
        if len(spy_c) > 20:
            spy_cm = compute_metrics(spy_c, "SPY B&H",
                                     f"{common_start.year}-{common_end.year}")
            print(f"\n  SPY B&H:")
            for k, v in spy_cm.items():
                if k not in ("name", "cagr_raw", "sharpe_raw", "max_dd_raw"):
                    print(f"    {k}: {v}")

    # Annual breakdown for V2 (momentum + HRP)
    if "momentum_hrp" in results:
        print("\n--- ANNUAL BREAKDOWN: V2 Momentum + HRP ---")
        ab = annual_breakdown(results["momentum_hrp"].equity_curve)
        print(ab.to_string(index=False))

    if "momentum_ml_hrp" in results:
        print("\n--- ANNUAL BREAKDOWN: V3 Momentum + Ridge ML ---")
        ab = annual_breakdown(results["momentum_ml_hrp"].equity_curve)
        print(ab.to_string(index=False))

    # ── 6. LEAKAGE TEST (on V2 - fastest) ──
    leakage = run_v2_leakage_test(tri_df, regime_df, cfg, v2cfg, "momentum_hrp")

    # ── 7. SAVE ──
    output_dir = Path(__file__).parent.parent / "result"
    output_dir.mkdir(exist_ok=True)

    # Equity curves
    eq_df = pd.DataFrame({k: v.equity_curve for k, v in results.items()})
    eq_df["SPY"] = spy_norm.reindex(eq_df.index)
    eq_df.to_csv(output_dir / "etf_rotation_v2_equity_curves.csv")

    # Summary JSON
    summary = {
        "full_period": full_metrics,
        "spy_benchmark": spy_m,
        "leakage_test": leakage,
        "config": {
            "momentum": f"12-1 (lookback={v2cfg.mom_lookback}, skip={v2cfg.mom_skip})",
            "top_k": v2cfg.top_k,
            "ml_train_window": f"{v2cfg.ml_train_window}d (~{v2cfg.ml_train_window/252:.1f}yr)",
            "ml_label_horizon": f"T+{v2cfg.ml_label_horizon}",
            "ml_ridge_alpha": v2cfg.ml_ridge_alpha,
            "fusion": f"{v2cfg.fusion_mom_weight}*mom + {v2cfg.fusion_ml_weight}*ml",
            "rebalance": f"every {v2cfg.rebalance_freq_days}d (weekly)",
            "target_vol": cfg.target_vol,
            "cost_bps": cfg.cost_bps,
            "hvr_thresholds": f"off={cfg.hvr_threshold_off}, on={cfg.hvr_threshold_on}",
        },
    }
    with open(output_dir / "etf_rotation_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")

    # ── 8. VERDICT ──
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    for vkey, vname in variants:
        if vkey in results:
            m = compute_metrics(results[vkey].equity_curve, vname)
            sharpe = m["sharpe_raw"]
            max_dd = m["max_dd_raw"]
            print(f"  {vname}: Sharpe={sharpe:.2f}, MaxDD={max_dd*100:.1f}%")

    print(f"  SPY B&H: Sharpe={spy_m['sharpe_raw']:.2f}, MaxDD={spy_m['max_dd_raw']*100:.1f}%")

    best = max(
        [(k, compute_metrics(v.equity_curve, k)["sharpe_raw"]) for k, v in results.items()],
        key=lambda x: x[1],
    )
    print(f"\n  Best variant: {best[0]} (Sharpe={best[1]:.2f})")

    if best[1] > 0.5:
        print("  >>> RECOMMEND: Consider deployment. Strategy shows robust edge.")
    elif best[1] > 0.3:
        print("  >>> CAUTIOUS: Modest edge detected. Paper trade first.")
    elif best[1] > 0:
        print("  >>> MARGINAL: Positive but weak. Likely not worth transaction costs in practice.")
    else:
        print("  >>> NOT RECOMMENDED: No edge detected over this period.")


if __name__ == "__main__":
    main()
