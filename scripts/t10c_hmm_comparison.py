#!/usr/bin/env python3
"""T10C-Slim: V7 vs HMM Combo vs Buy&Hold — Walkforward Comparison.

Strategies:
  A: SPY Buy & Hold
  B: ETF Portfolio Buy & Hold (risk-on weights, no risk mgmt)
  C: T10C-Slim Current (vol-target + MA200 cap + VIX cap + deadband + 21d rebalance)
  D: T10C-Slim + HMM Combo (vol-target + HMM cap + deadband + 21d rebalance)

HMM is walkforward: train on expanding window, retrain every 21d, no data leakage.
Test starts Year 2 (2023+), Year 1 = HMM training + MA200 warmup.

Usage:
    python scripts/t10c_hmm_comparison.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ═══════════════════════════════════════════════════════════════
# T10C-Slim Config (matches production exactly)
# ═══════════════════════════════════════════════════════════════

PORTFOLIO_RISK_ON = {
    'SMH': 0.25, 'USMV': 0.25, 'QUAL': 0.20,
    'PDBC': 0.15, 'COPX': 0.05, 'URA': 0.05, 'DBA': 0.05,
}
PORTFOLIO_RISK_OFF = {
    'USMV': 0.25, 'QUAL': 0.20, 'GDX': 0.20,
    'PDBC': 0.15, 'DBA': 0.10, 'COPX': 0.05, 'URA': 0.05,
}

# Vol target
TARGET_VOL = 0.12
VOL_BLEND_SHORT = 20
VOL_BLEND_LONG = 60
VOL_BLEND_ALPHA = 0.7
VOL_FLOOR = 0.08
VOL_CAP = 0.40

# MA200 cap (P2 2-level)
MA200_SHALLOW_CAP = 0.60
MA200_DEEP_CAP = 0.30
MA200_DEEP_THRESHOLD = -0.05

# VIX trigger
VIX_ENABLE_THRESH = 25.0
VIX_DISABLE_THRESH = 20.0
VIX_ENABLE_CONFIRM = 2
VIX_DISABLE_CONFIRM = 5
VIX_EXPOSURE_CAP = 0.50

# Theme budget
THEME_TICKERS = ['COPX', 'URA']
DEFENSIVE_TICKERS = ['USMV', 'QUAL']
THEME_BUDGET_NORMAL = 0.10
THEME_BUDGET_MEDIUM = 0.06
THEME_BUDGET_HIGH = 0.02

# Deadband & rebalance
DEADBAND_UP = 0.02
DEADBAND_DOWN = 0.05
MAX_STEP = 0.15
REBALANCE_FREQ = 21
COST_BPS = 10.0
MIN_CASH = 0.05

# HMM config
HMM_MIN_TRAIN = 252
HMM_RETRAIN_FREQ = 21
HMM_EMA_SPAN = 4
HMM_GAMMA = 2
HMM_FLOOR = 0.05
HMM_MID_WEIGHT = 0.5
HMM_N_STATES = 3


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_data(cache_path: Path = Path('data/etf_rotation_prices.parquet')) -> pd.DataFrame:
    """Load ETF + SPY + VIX daily close prices. Cache to parquet."""
    if cache_path.exists():
        print(f"  Loading cached: {cache_path}")
        return pd.read_parquet(cache_path)

    import yfinance as yf
    tickers = list(set(
        list(PORTFOLIO_RISK_ON.keys()) + list(PORTFOLIO_RISK_OFF.keys())
        + ['SPY', '^VIX']
    ))
    print(f"  Downloading {tickers} from yfinance...")
    data = yf.download(tickers, start='2021-01-01', end='2026-12-31', auto_adjust=False)
    close = data['Close'].copy()
    close.columns = [c.replace('^', '') for c in close.columns]  # ^VIX -> VIX
    close.index.name = 'date'
    close = close.dropna(subset=['SPY'])  # drop days with no SPY close (e.g. today before market close)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(cache_path)
    print(f"  Cached to {cache_path}")
    return close


# ═══════════════════════════════════════════════════════════════
# Vol-Target
# ═══════════════════════════════════════════════════════════════

def compute_portfolio_vol(close_df: pd.DataFrame, weights: Dict[str, float],
                          t: int) -> float:
    """Blended portfolio vol at day t using static weights."""
    available = [tk for tk in weights if tk in close_df.columns]
    w = np.array([weights[tk] for tk in available])
    w = w / w.sum()

    log_ret = np.log(close_df[available] / close_df[available].shift(1))

    # Portfolio log return
    port_lr = (log_ret.iloc[:t + 1] * w).sum(axis=1).dropna()

    if len(port_lr) < VOL_BLEND_LONG + 5:
        return 0.15  # fallback

    short_w = port_lr.iloc[-VOL_BLEND_SHORT:]
    long_w = port_lr.iloc[-VOL_BLEND_LONG:]

    v_short = float(short_w.std() * np.sqrt(252)) if len(short_w) > 5 else 0.15
    v_long = float(long_w.std() * np.sqrt(252)) if len(long_w) > 10 else 0.15

    blended = VOL_BLEND_ALPHA * v_short + (1 - VOL_BLEND_ALPHA) * v_long
    return max(VOL_FLOOR, min(blended, VOL_CAP))


def vol_target_exposure(bvol: float) -> float:
    return min(TARGET_VOL / bvol if bvol > 0 else 1.0, 1.0)


# ═══════════════════════════════════════════════════════════════
# MA200 Cap
# ═══════════════════════════════════════════════════════════════

def ma200_cap(spy_price: float, ma200: float) -> float:
    if np.isnan(ma200) or ma200 <= 0:
        return 1.0
    dev = (spy_price - ma200) / ma200
    if dev >= 0:
        return 1.0
    elif dev > MA200_DEEP_THRESHOLD:
        return MA200_SHALLOW_CAP
    else:
        return MA200_DEEP_CAP


# ═══════════════════════════════════════════════════════════════
# VIX Trigger State Machine
# ═══════════════════════════════════════════════════════════════

class VixTrigger:
    def __init__(self):
        self.mode = 'baseline'
        self.enable_count = 0
        self.disable_count = 0

    def update(self, spy_price: float, ma200: float, vix: float) -> bool:
        """Update VIX trigger. Returns True if vix_active."""
        if np.isnan(vix) or np.isnan(ma200):
            return self.mode == 'vix_active'

        spy_below = spy_price < ma200
        spy_above = spy_price > ma200

        if self.mode == 'baseline':
            if spy_below and vix >= VIX_ENABLE_THRESH:
                self.enable_count += 1
                if self.enable_count >= VIX_ENABLE_CONFIRM:
                    self.mode = 'vix_active'
                    self.enable_count = 0
            else:
                self.enable_count = 0
        else:  # vix_active
            if spy_above and vix <= VIX_DISABLE_THRESH:
                self.disable_count += 1
                if self.disable_count >= VIX_DISABLE_CONFIRM:
                    self.mode = 'baseline'
                    self.disable_count = 0
            else:
                self.disable_count = 0

        return self.mode == 'vix_active'


def theme_budget(vix: float) -> float:
    if vix < VIX_DISABLE_THRESH:
        return THEME_BUDGET_NORMAL
    elif vix < VIX_ENABLE_THRESH:
        return THEME_BUDGET_MEDIUM
    else:
        return THEME_BUDGET_HIGH


def apply_theme_budget(weights: Dict, budget: float) -> Dict:
    w = weights.copy()
    theme_sum = sum(w.get(t, 0) for t in THEME_TICKERS)
    if theme_sum <= budget:
        return w
    excess = theme_sum - budget
    if theme_sum > 0:
        scale = budget / theme_sum
        for t in THEME_TICKERS:
            if t in w:
                w[t] *= scale
    def_sum = sum(w.get(t, 0) for t in DEFENSIVE_TICKERS)
    if def_sum > 0:
        for t in DEFENSIVE_TICKERS:
            if t in w:
                w[t] += excess * (w[t] / def_sum)
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}
    return w


# ═══════════════════════════════════════════════════════════════
# HMM Walkforward
# ═══════════════════════════════════════════════════════════════

def prepare_hmm_features(spy_close: pd.Series, vix_close: pd.Series,
                          qqq_close: pd.Series = None,
                          tlt_close: pd.Series = None) -> pd.DataFrame:
    """Build 4 HMM observables from raw price data (no pre-computed z-scores)."""
    logret = np.log(spy_close / spy_close.shift(1))
    vol_10d = logret.rolling(10, min_periods=10).std()

    # VIX z-score (252d rolling)
    vix_mean = vix_close.rolling(252, min_periods=60).mean()
    vix_std = vix_close.rolling(252, min_periods=60).std()
    vix_z = (vix_close - vix_mean) / vix_std.replace(0, np.nan)

    # Risk premium proxy: just use VIX level z-score as 4th feature
    # (simpler than QQQ/TLT ratio which needs extra data)
    # Use SPY momentum as proxy for risk appetite
    spy_mom_20 = spy_close.pct_change(20)
    spy_mom_z_mean = spy_mom_20.rolling(252, min_periods=60).mean()
    spy_mom_z_std = spy_mom_20.rolling(252, min_periods=60).std()
    spy_mom_z = (spy_mom_20 - spy_mom_z_mean) / spy_mom_z_std.replace(0, np.nan)

    features = pd.DataFrame({
        'logret': logret,
        'vol_10d': vol_10d,
        'vix_z': vix_z,
        'mom_z': spy_mom_z,
    }, index=spy_close.index).dropna()

    return features


def fit_hmm(X_raw: np.ndarray, seed: int = 42):
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    model = GaussianHMM(n_components=HMM_N_STATES, covariance_type="full",
                         n_iter=300, random_state=seed, tol=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)

    means_orig = scaler.inverse_transform(model.means_)
    mean_vols = means_orig[:, 1]
    order = np.argsort(mean_vols)
    crisis_idx = int(order[2])
    mid_idx = int(order[1])
    return model, scaler, crisis_idx, mid_idx


# ═══════════════════════════════════════════════════════════════
# Deadband
# ═══════════════════════════════════════════════════════════════

def apply_deadband(target_exp: float, current_exp: float) -> float:
    delta = target_exp - current_exp
    if 0 < delta < DEADBAND_UP:
        return current_exp
    elif delta < 0 and abs(delta) < DEADBAND_DOWN:
        return current_exp
    elif delta > MAX_STEP:
        return current_exp + MAX_STEP
    elif delta < -MAX_STEP:
        return current_exp - MAX_STEP
    return target_exp


# ═══════════════════════════════════════════════════════════════
# Portfolio Return Computation
# ═══════════════════════════════════════════════════════════════

def portfolio_daily_return(close_df: pd.DataFrame, weights: Dict[str, float],
                            exposure: float, t: int) -> float:
    """Compute weighted portfolio return at day t, scaled by exposure."""
    if t < 1:
        return 0.0
    ret = 0.0
    for ticker, w in weights.items():
        if ticker in close_df.columns:
            p0 = close_df[ticker].iloc[t - 1]
            p1 = close_df[ticker].iloc[t]
            if p0 > 0 and not np.isnan(p1):
                ret += w * (p1 / p0 - 1)
    return ret * exposure


# ═══════════════════════════════════════════════════════════════
# Main Simulation
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('results/t10c_hmm_comparison'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("T10C-Slim: V7 vs HMM Combo — Walkforward Comparison")
    print("=" * 70)

    # ── Load data ──
    close_df = load_data()
    dates = close_df.index
    n = len(dates)
    spy = close_df['SPY']
    vix = close_df['VIX']
    spy_ma200 = spy.rolling(200, min_periods=200).mean()

    print(f"  Dates: {dates[0].date()} to {dates[-1].date()} ({n} days)")

    # ── HMM features (from raw prices, no pre-computed z-scores) ──
    hmm_feat = prepare_hmm_features(spy, vix)
    hmm_dates_set = set(hmm_feat.index)
    hmm_X_all = hmm_feat.values
    hmm_date_to_idx = {d: i for i, d in enumerate(hmm_feat.index)}

    # ── Determine test start ──
    # Need: MA200 (200d) + HMM train (252d) + vol (60d)
    # MA200 warmup ends ~200 trading days in. HMM needs 252 days.
    # Start test at max(200, 252) = day 252 from start of HMM features
    # HMM features start ~272 days into close_df (252d for z-score rolling)
    # So total warmup from close_df start: ~272 + 252 = ~524 days

    # Find first date where we have: MA200 valid + at least 252 HMM features
    test_start_idx = None
    for t in range(n):
        d = dates[t]
        if np.isnan(spy_ma200.iloc[t]):
            continue
        if d in hmm_date_to_idx:
            hmm_idx = hmm_date_to_idx[d]
            if hmm_idx >= HMM_MIN_TRAIN:
                test_start_idx = t
                break

    if test_start_idx is None:
        print("ERROR: Not enough data for test start")
        return 1

    print(f"  Test start: day {test_start_idx} = {dates[test_start_idx].date()}")
    print(f"  Test days: {n - test_start_idx}")

    # ── Strategy state ──
    # Strategy A: SPY B&H
    a_equity = [1.0]

    # Strategy B: ETF B&H (risk-on portfolio, no rebalance after initial)
    b_equity = [1.0]

    # Strategy C: T10C-Slim V7 (current production)
    c_equity = [1.0]
    c_exposure = 0.0
    c_vix_trigger = VixTrigger()
    c_rebal_counter = 0
    c_trades = 0

    # Strategy D: T10C-Slim + HMM Combo
    d_equity = [1.0]
    d_exposure = 0.0
    d_rebal_counter = 0
    d_trades = 0
    d_hmm_model = None
    d_hmm_scaler = None
    d_hmm_crisis_idx = None
    d_hmm_mid_idx = None
    d_hmm_last_train = -1
    d_hmm_n_retrains = 0
    d_p_risk_smooth = 0.0
    d_alpha = 2.0 / (HMM_EMA_SPAN + 1)

    # Active portfolio for C and D (simplified: always risk-on since we don't have
    # the V7 state machine for portfolio switching — that's a separate C# service)
    # For fair comparison, both C and D use risk-on portfolio
    active_weights = PORTFOLIO_RISK_ON

    records = []

    for t in range(test_start_idx, n):
        d = dates[t]
        spy_price = spy.iloc[t]
        ma200_val = spy_ma200.iloc[t]
        vix_val = vix.iloc[t] if not np.isnan(vix.iloc[t]) else 20.0

        # ── Strategy A: SPY B&H ──
        if t > test_start_idx:
            spy_ret = spy.iloc[t] / spy.iloc[t - 1] - 1
            a_equity.append(a_equity[-1] * (1 + spy_ret))
        else:
            a_equity.append(1.0)

        # ── Strategy B: ETF Portfolio B&H ──
        if t > test_start_idx:
            b_ret = portfolio_daily_return(close_df, active_weights, 1.0, t)
            b_equity.append(b_equity[-1] * (1 + b_ret))
        else:
            b_equity.append(1.0)

        # ── Shared: vol-target ──
        bvol = compute_portfolio_vol(close_df, active_weights, t)
        vt_exp = vol_target_exposure(bvol)

        # ════════════════════════════════════════════════
        # Strategy C: T10C-Slim V7 (current production)
        # ════════════════════════════════════════════════
        c_rebal_counter += 1
        is_rebal_c = (c_rebal_counter >= REBALANCE_FREQ) or (t == test_start_idx)

        if is_rebal_c:
            # L1: vol-target
            c_target = vt_exp

            # L2: MA200 cap
            c_risk_cap = ma200_cap(spy_price, ma200_val)
            c_target = min(c_target, c_risk_cap)

            # L3: VIX cap
            c_vix_active = c_vix_trigger.update(spy_price, ma200_val, vix_val)
            if c_vix_active:
                c_target = min(c_target, VIX_EXPOSURE_CAP)

            # L4: min cash
            c_target = min(c_target, 1.0 - MIN_CASH)

            # L5: deadband
            c_new_exp = apply_deadband(c_target, c_exposure)

            # Trade cost
            delta_c = abs(c_new_exp - c_exposure)
            if delta_c > 0.02:
                cost = delta_c * COST_BPS / 10_000
                c_equity[-1] *= (1 - cost)
                c_trades += 1

            c_exposure = c_new_exp
            c_rebal_counter = 0
        else:
            # Still update VIX trigger daily (matches production)
            c_vix_trigger.update(spy_price, ma200_val, vix_val)

        # Compute C return
        if t > test_start_idx:
            c_ret = portfolio_daily_return(close_df, active_weights, c_exposure, t)
            c_equity.append(c_equity[-1] * (1 + c_ret))
        else:
            c_equity.append(c_equity[-1])

        # ════════════════════════════════════════════════
        # Strategy D: T10C-Slim + HMM Combo
        # ════════════════════════════════════════════════
        d_rebal_counter += 1
        is_rebal_d = (d_rebal_counter >= REBALANCE_FREQ) or (t == test_start_idx)

        # HMM update (daily, regardless of rebalance)
        hmm_p_risk = 0.0
        if d in hmm_date_to_idx:
            hmm_idx = hmm_date_to_idx[d]

            # Retrain check
            days_since = hmm_idx - d_hmm_last_train
            if d_hmm_model is None or days_since >= HMM_RETRAIN_FREQ:
                train_X = hmm_X_all[:hmm_idx]  # all data BEFORE today
                if len(train_X) >= HMM_MIN_TRAIN:
                    try:
                        d_hmm_model, d_hmm_scaler, d_hmm_crisis_idx, d_hmm_mid_idx = \
                            fit_hmm(train_X, seed=42 + d_hmm_n_retrains)
                        d_hmm_last_train = hmm_idx
                        d_hmm_n_retrains += 1
                    except Exception:
                        pass

            # Predict
            if d_hmm_model is not None:
                history_X = hmm_X_all[:hmm_idx + 1]
                try:
                    X_scaled = d_hmm_scaler.transform(history_X)
                    posteriors = d_hmm_model.predict_proba(X_scaled)
                    p_crisis = float(posteriors[-1, d_hmm_crisis_idx])
                    p_mid = float(posteriors[-1, d_hmm_mid_idx])
                    hmm_p_risk = p_crisis + HMM_MID_WEIGHT * p_mid
                except Exception:
                    hmm_p_risk = 0.0

        # EMA smooth
        if t == test_start_idx:
            d_p_risk_smooth = hmm_p_risk
        else:
            d_p_risk_smooth = d_alpha * hmm_p_risk + (1 - d_alpha) * d_p_risk_smooth

        # HMM exposure cap
        hmm_cap = max(HMM_FLOOR, (1 - d_p_risk_smooth) ** HMM_GAMMA)

        if is_rebal_d:
            # L1: vol-target
            d_target = vt_exp

            # L2: HMM combo cap (replaces MA200 + VIX)
            d_target = min(d_target, hmm_cap)

            # L3: (deleted — HMM covers VIX info)

            # L4: min cash
            d_target = min(d_target, 1.0 - MIN_CASH)

            # L5: deadband
            d_new_exp = apply_deadband(d_target, d_exposure)

            # Trade cost
            delta_d = abs(d_new_exp - d_exposure)
            if delta_d > 0.02:
                cost = delta_d * COST_BPS / 10_000
                d_equity[-1] *= (1 - cost)
                d_trades += 1

            d_exposure = d_new_exp
            d_rebal_counter = 0

        # Compute D return
        if t > test_start_idx:
            d_ret = portfolio_daily_return(close_df, active_weights, d_exposure, t)
            d_equity.append(d_equity[-1] * (1 + d_ret))
        else:
            d_equity.append(d_equity[-1])

        # Record
        records.append({
            'date': d,
            'spy': spy_price,
            'vix': vix_val,
            'ma200': ma200_val,
            'bvol': round(bvol, 4),
            'vt_exp': round(vt_exp, 4),
            'c_exposure': round(c_exposure, 4),
            'c_vix_active': c_vix_trigger.mode == 'vix_active',
            'd_exposure': round(d_exposure, 4),
            'hmm_p_risk': round(hmm_p_risk, 4),
            'hmm_p_smooth': round(d_p_risk_smooth, 4),
            'hmm_cap': round(hmm_cap, 4),
            'a_equity': a_equity[-1],
            'b_equity': b_equity[-1],
            'c_equity': c_equity[-1],
            'd_equity': d_equity[-1],
        })

    # Drop the first entry (initialization)
    a_equity = a_equity[1:]
    b_equity = b_equity[1:]
    c_equity = c_equity[1:]
    d_equity = d_equity[1:]

    result_df = pd.DataFrame(records)

    # ── Stats ──
    def _stats(equity_arr, label):
        eq = np.array(equity_arr)
        rets = eq[1:] / eq[:-1] - 1
        if len(rets) < 2:
            return {}
        n_d = len(rets)
        mean_r = float(np.mean(rets))
        std_r = float(np.std(rets, ddof=1))
        sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0
        peak = np.maximum.accumulate(eq)
        max_dd = float(((eq - peak) / peak).min())
        n_years = n_d / 252.0
        cagr = float(eq[-1] ** (1.0 / n_years) - 1) if eq[-1] > 0 and n_years > 0 else 0
        calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0
        return {
            'label': label, 'sharpe': round(sharpe, 3), 'cagr': round(cagr, 4),
            'max_dd': round(max_dd, 4), 'calmar': round(calmar, 3),
            'total_return': round(float(eq[-1] - 1), 4),
        }

    stats_a = _stats(a_equity, 'A: SPY B&H')
    stats_b = _stats(b_equity, 'B: ETF Portfolio B&H')
    stats_c = _stats(c_equity, 'C: T10C-Slim V7 (current)')
    stats_d = _stats(d_equity, 'D: T10C-Slim + HMM Combo')

    print(f"\n{'='*70}")
    print(f" RESULTS — Test: {result_df['date'].iloc[0].date()} to {result_df['date'].iloc[-1].date()}")
    print(f" ({len(result_df)} days, C trades={c_trades}, D trades={d_trades})")
    print(f"{'='*70}")

    print(f"\n  {'Strategy':<30s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s} {'Calmar':>8s} {'TotRet':>8s}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for s in [stats_a, stats_b, stats_c, stats_d]:
        print(f"  {s['label']:<30s} {s['sharpe']:7.3f} {s['cagr']:8.2%} "
              f"{s['max_dd']:8.2%} {s['calmar']:8.3f} {s['total_return']:8.2%}")

    # ── Yearly breakdown ──
    print(f"\n--- Yearly Breakdown ---")
    result_df['year'] = pd.to_datetime(result_df['date']).dt.year

    for yr, grp in result_df.groupby('year'):
        print(f"\n  {yr}:")
        for col, label in [('a_equity', 'SPY B&H'), ('b_equity', 'ETF B&H'),
                             ('c_equity', 'V7 Current'), ('d_equity', 'HMM Combo')]:
            eq = grp[col].values
            if len(eq) < 2:
                continue
            yr_ret = eq[-1] / eq[0] - 1
            yr_eq = eq / eq[0]
            yr_dd = float(((yr_eq - np.maximum.accumulate(yr_eq)) / np.maximum.accumulate(yr_eq)).min())
            print(f"    {label:<15s}: ret={yr_ret:+7.2%}, DD={yr_dd:+6.2%}")

    # ── Exposure comparison ──
    print(f"\n--- Avg Exposure ---")
    print(f"  V7 Current:  {result_df['c_exposure'].mean():.1%}")
    print(f"  HMM Combo:   {result_df['d_exposure'].mean():.1%}")

    # ── Drawdown episodes ──
    print(f"\n--- Top 3 Drawdown Episodes (V7 vs HMM) ---")
    for col, label in [('c_equity', 'V7'), ('d_equity', 'HMM')]:
        eq = result_df[col].values
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        # Find episodes
        in_dd = False
        episodes = []
        for i in range(len(dd)):
            if dd[i] < -0.02 and not in_dd:
                in_dd = True
                start = i
            elif dd[i] >= 0 and in_dd:
                in_dd = False
                episodes.append((start, i, float(dd[start:i].min())))
        if in_dd:
            episodes.append((start, len(dd) - 1, float(dd[start:].min())))
        episodes.sort(key=lambda x: x[2])
        print(f"\n  {label}:")
        for ep in episodes[:3]:
            s, e, mdd = ep
            print(f"    {result_df['date'].iloc[s].date()} to {result_df['date'].iloc[e].date()}: "
                  f"DD={mdd:.1%}, dur={e-s}d")

    # ── Save ──
    result_df.to_csv(args.output_dir / 'comparison_detail.csv', index=False)
    summary = {
        'test_period': f"{result_df['date'].iloc[0].date()} to {result_df['date'].iloc[-1].date()}",
        'n_days': len(result_df),
        'strategies': {
            'A_SPY_BH': stats_a,
            'B_ETF_BH': stats_b,
            'C_V7_Current': stats_c,
            'D_HMM_Combo': stats_d,
        },
        'c_trades': c_trades,
        'd_trades': d_trades,
        'c_avg_exposure': round(float(result_df['c_exposure'].mean()), 3),
        'd_avg_exposure': round(float(result_df['d_exposure'].mean()), 3),
    }

    def _safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, pd.Timestamp): return str(obj.date())
        raise TypeError(f"Not serializable: {type(obj)}")

    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, default=_safe), encoding='utf-8')
    print(f"\n  Results saved to {args.output_dir}/")
    print(f"  HMM retrains: {d_hmm_n_retrains}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
