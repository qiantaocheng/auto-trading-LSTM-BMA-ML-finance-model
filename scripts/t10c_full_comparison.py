#!/usr/bin/env python3
"""T10C-Slim Full Comparison: V7 vs HMM System 1&2 vs HMM System 3.

Strategies (7 total):
  A: SPY Buy & Hold
  B: ETF Portfolio Buy & Hold (risk-on weights)
  C: T10C-Slim V7 Current (vol-target + MA200 cap + VIX cap + deadband + 21d rebalance)
  D: ETF + Sys1&2 (HMM cap replaces MA200+VIX) — exposure = min(vol_target, (1-p_risk)^2)
  E: ETF + Sys3 (HMM context signal — portfolio switch + theme budget, p>0.9 slight reduction)
  F: SPY + Sys1&2 (HMM cap on SPY single stock)
  G: SPY + Sys3 (HMM context on SPY — p>0.9 slight reduction only)

System 3 logic:
  p_risk < 0.5 -> normal (risk-on portfolio, no HMM intervention)
  p_risk 0.5-0.9 -> risk-off portfolio (GDX swap) + VIX theme budget control
  p_risk > 0.9 -> risk-off + theme budget + liquidity reduction (0.85 cap)

HMM is walkforward: expanding window, retrain every 21d, no data leakage.

Usage:
    python scripts/t10c_full_comparison.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ===================================================================
# T10C-Slim Config (matches production)
# ===================================================================

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

# MA200 cap
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

# System 3 thresholds
SYS3_SWITCH_THRESHOLD = 0.50   # p_risk > 0.5 -> risk-off + theme budget
SYS3_REDUCE_THRESHOLD = 0.90   # p_risk > 0.9 -> additional exposure reduction
SYS3_REDUCE_CAP = 0.85         # exposure cap when p_risk > 0.9


# ===================================================================
# Data Loading
# ===================================================================

def load_data(cache_path: Path = Path('data/etf_rotation_prices.parquet')) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    import yfinance as yf
    tickers = list(set(
        list(PORTFOLIO_RISK_ON.keys()) + list(PORTFOLIO_RISK_OFF.keys())
        + ['SPY', '^VIX']
    ))
    print(f"  Downloading {tickers} from yfinance...")
    data = yf.download(tickers, start='2021-01-01', end='2026-12-31', auto_adjust=False)
    close = data['Close'].copy()
    close.columns = [c.replace('^', '') for c in close.columns]
    close.index.name = 'date'
    close = close.dropna(subset=['SPY'])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(cache_path)
    return close


# ===================================================================
# Shared Helpers
# ===================================================================

def compute_portfolio_vol(close_df: pd.DataFrame, weights: Dict[str, float],
                          t: int) -> float:
    available = [tk for tk in weights if tk in close_df.columns]
    w = np.array([weights[tk] for tk in available])
    w = w / w.sum()
    log_ret = np.log(close_df[available] / close_df[available].shift(1))
    port_lr = (log_ret.iloc[:t + 1] * w).sum(axis=1).dropna()
    if len(port_lr) < VOL_BLEND_LONG + 5:
        return 0.15
    v_short = float(port_lr.iloc[-VOL_BLEND_SHORT:].std() * np.sqrt(252))
    v_long = float(port_lr.iloc[-VOL_BLEND_LONG:].std() * np.sqrt(252))
    blended = VOL_BLEND_ALPHA * v_short + (1 - VOL_BLEND_ALPHA) * v_long
    return max(VOL_FLOOR, min(blended, VOL_CAP))


def compute_spy_vol(spy: pd.Series, t: int) -> float:
    log_ret = np.log(spy / spy.shift(1))
    lr = log_ret.iloc[:t + 1].dropna()
    if len(lr) < VOL_BLEND_LONG + 5:
        return 0.15
    v_short = float(lr.iloc[-VOL_BLEND_SHORT:].std() * np.sqrt(252))
    v_long = float(lr.iloc[-VOL_BLEND_LONG:].std() * np.sqrt(252))
    blended = VOL_BLEND_ALPHA * v_short + (1 - VOL_BLEND_ALPHA) * v_long
    return max(VOL_FLOOR, min(blended, VOL_CAP))


def vol_target_exposure(bvol: float) -> float:
    return min(TARGET_VOL / bvol if bvol > 0 else 1.0, 1.0)


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


class VixTrigger:
    def __init__(self):
        self.mode = 'baseline'
        self.enable_count = 0
        self.disable_count = 0

    def update(self, spy_price: float, ma200: float, vix: float) -> bool:
        if np.isnan(vix) or np.isnan(ma200):
            return self.mode == 'vix_active'
        if self.mode == 'baseline':
            if spy_price < ma200 and vix >= VIX_ENABLE_THRESH:
                self.enable_count += 1
                if self.enable_count >= VIX_ENABLE_CONFIRM:
                    self.mode = 'vix_active'
                    self.enable_count = 0
            else:
                self.enable_count = 0
        else:
            if spy_price > ma200 and vix <= VIX_DISABLE_THRESH:
                self.disable_count += 1
                if self.disable_count >= VIX_DISABLE_CONFIRM:
                    self.mode = 'baseline'
                    self.disable_count = 0
            else:
                self.disable_count = 0
        return self.mode == 'vix_active'


def theme_budget_for_vix(vix: float) -> float:
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


def portfolio_daily_return(close_df: pd.DataFrame, weights: Dict[str, float],
                            exposure: float, t: int) -> float:
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


# ===================================================================
# HMM Walkforward
# ===================================================================

def prepare_hmm_features(spy_close: pd.Series, vix_close: pd.Series) -> pd.DataFrame:
    logret = np.log(spy_close / spy_close.shift(1))
    vol_10d = logret.rolling(10, min_periods=10).std()
    vix_mean = vix_close.rolling(252, min_periods=60).mean()
    vix_std = vix_close.rolling(252, min_periods=60).std()
    vix_z = (vix_close - vix_mean) / vix_std.replace(0, np.nan)
    spy_mom_20 = spy_close.pct_change(20, fill_method=None)
    spy_mom_z_mean = spy_mom_20.rolling(252, min_periods=60).mean()
    spy_mom_z_std = spy_mom_20.rolling(252, min_periods=60).std()
    spy_mom_z = (spy_mom_20 - spy_mom_z_mean) / spy_mom_z_std.replace(0, np.nan)
    features = pd.DataFrame({
        'logret': logret, 'vol_10d': vol_10d,
        'vix_z': vix_z, 'mom_z': spy_mom_z,
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


# ===================================================================
# Stats
# ===================================================================

def _stats(equity_arr, label):
    eq = np.array(equity_arr)
    rets = eq[1:] / eq[:-1] - 1
    if len(rets) < 2:
        return {'label': label, 'sharpe': 0, 'cagr': 0, 'max_dd': 0, 'calmar': 0, 'total_return': 0}
    n_d = len(rets)
    mean_r = float(np.nanmean(rets))
    std_r = float(np.nanstd(rets, ddof=1))
    sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.nanmin((eq - peak) / peak))
    n_years = n_d / 252.0
    cagr = float(eq[-1] ** (1.0 / n_years) - 1) if eq[-1] > 0 and n_years > 0 else 0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0
    return {
        'label': label, 'sharpe': round(sharpe, 3), 'cagr': round(cagr, 4),
        'max_dd': round(max_dd, 4), 'calmar': round(calmar, 3),
        'total_return': round(float(eq[-1] - 1), 4),
    }


def yearly_stats(result_df, col, label):
    """Print yearly breakdown for one equity column."""
    rows = []
    for yr, grp in result_df.groupby('year'):
        eq = grp[col].values
        if len(eq) < 2:
            continue
        yr_ret = eq[-1] / eq[0] - 1
        yr_eq = eq / eq[0]
        pk = np.maximum.accumulate(yr_eq)
        yr_dd = float(((yr_eq - pk) / pk).min())
        rows.append((yr, yr_ret, yr_dd))
    return rows


# ===================================================================
# Main
# ===================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('results/t10c_full_comparison'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("T10C Full Comparison: V7 vs HMM Sys1&2 vs HMM Sys3 (ETF + SPY)")
    print("=" * 80)

    close_df = load_data()
    dates = close_df.index
    n = len(dates)
    spy = close_df['SPY']
    vix = close_df['VIX']
    spy_ma200 = spy.rolling(200, min_periods=200).mean()

    print(f"  Dates: {dates[0].date()} to {dates[-1].date()} ({n} days)")

    hmm_feat = prepare_hmm_features(spy, vix)
    hmm_X_all = hmm_feat.values
    hmm_date_to_idx = {d: i for i, d in enumerate(hmm_feat.index)}

    # Determine test start
    test_start_idx = None
    for t in range(n):
        d = dates[t]
        if np.isnan(spy_ma200.iloc[t]):
            continue
        if d in hmm_date_to_idx:
            if hmm_date_to_idx[d] >= HMM_MIN_TRAIN:
                test_start_idx = t
                break

    if test_start_idx is None:
        print("ERROR: Not enough data")
        return

    print(f"  Test start: day {test_start_idx} = {dates[test_start_idx].date()}")
    print(f"  Test days: {n - test_start_idx}")

    # ── Strategy states ──
    # A: SPY B&H
    a_eq = [1.0]
    # B: ETF B&H
    b_eq = [1.0]
    # C: V7 Current (ETF)
    c_eq = [1.0]; c_exp = 0.0; c_vix = VixTrigger(); c_rebal = 0; c_trades = 0
    # D: ETF + Sys1&2 (HMM cap)
    d_eq = [1.0]; d_exp = 0.0; d_rebal = 0; d_trades = 0
    # E: ETF + Sys3 (HMM context)
    e_eq = [1.0]; e_exp = 0.0; e_rebal = 0; e_trades = 0
    # F: SPY + Sys1&2 (HMM cap on SPY)
    f_eq = [1.0]; f_exp = 0.0; f_rebal = 0; f_trades = 0
    # G: SPY + Sys3 (HMM context on SPY)
    g_eq = [1.0]; g_exp = 0.0; g_rebal = 0; g_trades = 0

    # Shared HMM state (one model for all strategies)
    hmm_model = None
    hmm_scaler = None
    hmm_crisis_idx = None
    hmm_mid_idx = None
    hmm_last_train = -1
    hmm_n_retrains = 0
    p_risk_smooth = 0.0
    alpha = 2.0 / (HMM_EMA_SPAN + 1)

    records = []

    for t in range(test_start_idx, n):
        d = dates[t]
        spy_price = spy.iloc[t]
        ma200_val = spy_ma200.iloc[t]
        vix_val = vix.iloc[t] if not np.isnan(vix.iloc[t]) else 20.0
        is_first = (t == test_start_idx)

        # ── HMM update (shared across D, E, F, G) ──
        hmm_p_risk = 0.0
        if d in hmm_date_to_idx:
            hmm_idx = hmm_date_to_idx[d]
            days_since = hmm_idx - hmm_last_train
            if hmm_model is None or days_since >= HMM_RETRAIN_FREQ:
                train_X = hmm_X_all[:hmm_idx]
                if len(train_X) >= HMM_MIN_TRAIN:
                    try:
                        hmm_model, hmm_scaler, hmm_crisis_idx, hmm_mid_idx = \
                            fit_hmm(train_X, seed=42 + hmm_n_retrains)
                        hmm_last_train = hmm_idx
                        hmm_n_retrains += 1
                    except Exception:
                        pass
            if hmm_model is not None:
                try:
                    X_scaled = hmm_scaler.transform(hmm_X_all[:hmm_idx + 1])
                    posteriors = hmm_model.predict_proba(X_scaled)
                    p_crisis = float(posteriors[-1, hmm_crisis_idx])
                    p_mid = float(posteriors[-1, hmm_mid_idx])
                    hmm_p_risk = p_crisis + HMM_MID_WEIGHT * p_mid
                except Exception:
                    hmm_p_risk = 0.0

        if is_first:
            p_risk_smooth = hmm_p_risk
        else:
            p_risk_smooth = alpha * hmm_p_risk + (1 - alpha) * p_risk_smooth

        hmm_cap = max(HMM_FLOOR, (1 - p_risk_smooth) ** HMM_GAMMA)

        # ── A: SPY B&H ──
        if not is_first:
            spy_ret = spy.iloc[t] / spy.iloc[t - 1] - 1
            a_eq.append(a_eq[-1] * (1 + spy_ret))
        else:
            a_eq.append(1.0)

        # ── B: ETF Portfolio B&H ──
        if not is_first:
            b_ret = portfolio_daily_return(close_df, PORTFOLIO_RISK_ON, 1.0, t)
            b_eq.append(b_eq[-1] * (1 + b_ret))
        else:
            b_eq.append(1.0)

        # ── Shared: portfolio vol & vol-target ──
        bvol = compute_portfolio_vol(close_df, PORTFOLIO_RISK_ON, t)
        vt_exp = vol_target_exposure(bvol)

        # SPY vol for F, G
        spy_vol = compute_spy_vol(spy, t)
        spy_vt_exp = vol_target_exposure(spy_vol)

        # ═══════════════════════════════════════════════════
        # C: T10C-Slim V7 Current
        # ═══════════════════════════════════════════════════
        c_rebal += 1
        is_rebal_c = (c_rebal >= REBALANCE_FREQ) or is_first
        if is_rebal_c:
            ct = min(vt_exp, ma200_cap(spy_price, ma200_val))
            c_vix_active = c_vix.update(spy_price, ma200_val, vix_val)
            if c_vix_active:
                ct = min(ct, VIX_EXPOSURE_CAP)
            ct = min(ct, 1.0 - MIN_CASH)
            c_new = apply_deadband(ct, c_exp)
            delta = abs(c_new - c_exp)
            if delta > 0.02:
                c_eq[-1] *= (1 - delta * COST_BPS / 10_000)
                c_trades += 1
            c_exp = c_new
            c_rebal = 0
        else:
            c_vix.update(spy_price, ma200_val, vix_val)

        if not is_first:
            c_eq.append(c_eq[-1] * (1 + portfolio_daily_return(close_df, PORTFOLIO_RISK_ON, c_exp, t)))
        else:
            c_eq.append(c_eq[-1])

        # ═══════════════════════════════════════════════════
        # D: ETF + Sys1&2 (HMM cap replaces MA200+VIX)
        # ═══════════════════════════════════════════════════
        d_rebal += 1
        is_rebal_d = (d_rebal >= REBALANCE_FREQ) or is_first
        if is_rebal_d:
            dt = min(vt_exp, hmm_cap)
            dt = min(dt, 1.0 - MIN_CASH)
            d_new = apply_deadband(dt, d_exp)
            delta = abs(d_new - d_exp)
            if delta > 0.02:
                d_eq[-1] *= (1 - delta * COST_BPS / 10_000)
                d_trades += 1
            d_exp = d_new
            d_rebal = 0

        # D always uses risk-on portfolio (HMM doesn't switch portfolio)
        if not is_first:
            d_eq.append(d_eq[-1] * (1 + portfolio_daily_return(close_df, PORTFOLIO_RISK_ON, d_exp, t)))
        else:
            d_eq.append(d_eq[-1])

        # ═══════════════════════════════════════════════════
        # E: ETF + Sys3 (HMM context signal)
        # ═══════════════════════════════════════════════════
        e_rebal += 1
        is_rebal_e = (e_rebal >= REBALANCE_FREQ) or is_first

        # Sys3: HMM drives portfolio selection + theme budget
        if p_risk_smooth >= SYS3_SWITCH_THRESHOLD:
            e_portfolio = PORTFOLIO_RISK_OFF  # risk-off (GDX swap)
        else:
            e_portfolio = PORTFOLIO_RISK_ON

        if is_rebal_e:
            # L1: vol-target (uses active portfolio's vol)
            e_bvol = compute_portfolio_vol(close_df, e_portfolio, t)
            et = vol_target_exposure(e_bvol)

            # L2: MA200 cap (keep — Sys3 does NOT remove this)
            et = min(et, ma200_cap(spy_price, ma200_val))

            # L3: HMM context — no VIX hard cap, but theme budget when p_risk > 0.5
            # (theme budget already handled via portfolio switch above)
            # Additional: if p_risk very high, slight liquidity reduction
            if p_risk_smooth >= SYS3_REDUCE_THRESHOLD:
                et = min(et, SYS3_REDUCE_CAP)

            # L4: min cash
            et = min(et, 1.0 - MIN_CASH)

            # L5: deadband
            e_new = apply_deadband(et, e_exp)
            delta = abs(e_new - e_exp)
            if delta > 0.02:
                e_eq[-1] *= (1 - delta * COST_BPS / 10_000)
                e_trades += 1
            e_exp = e_new
            e_rebal = 0

        # Apply theme budget when HMM says risk elevated
        if p_risk_smooth >= SYS3_SWITCH_THRESHOLD:
            e_weights = apply_theme_budget(e_portfolio, theme_budget_for_vix(vix_val))
        else:
            e_weights = e_portfolio

        if not is_first:
            e_eq.append(e_eq[-1] * (1 + portfolio_daily_return(close_df, e_weights, e_exp, t)))
        else:
            e_eq.append(e_eq[-1])

        # ═══════════════════════════════════════════════════
        # F: SPY + Sys1&2 (HMM cap on SPY)
        # ═══════════════════════════════════════════════════
        f_rebal += 1
        is_rebal_f = (f_rebal >= REBALANCE_FREQ) or is_first
        if is_rebal_f:
            ft = min(spy_vt_exp, hmm_cap)
            ft = min(ft, 1.0 - MIN_CASH)
            f_new = apply_deadband(ft, f_exp)
            delta = abs(f_new - f_exp)
            if delta > 0.02:
                f_eq[-1] *= (1 - delta * COST_BPS / 10_000)
                f_trades += 1
            f_exp = f_new
            f_rebal = 0

        if not is_first:
            spy_ret = spy.iloc[t] / spy.iloc[t - 1] - 1
            f_eq.append(f_eq[-1] * (1 + spy_ret * f_exp))
        else:
            f_eq.append(f_eq[-1])

        # ═══════════════════════════════════════════════════
        # G: SPY + Sys3 (HMM context on SPY)
        # ═══════════════════════════════════════════════════
        g_rebal += 1
        is_rebal_g = (g_rebal >= REBALANCE_FREQ) or is_first
        if is_rebal_g:
            # Vol-target + MA200 cap (keep)
            gt = min(spy_vt_exp, ma200_cap(spy_price, ma200_val))

            # Sys3 on SPY: p_risk > 0.9 -> liquidity reduction
            if p_risk_smooth >= SYS3_REDUCE_THRESHOLD:
                gt = min(gt, SYS3_REDUCE_CAP)

            gt = min(gt, 1.0 - MIN_CASH)
            g_new = apply_deadband(gt, g_exp)
            delta = abs(g_new - g_exp)
            if delta > 0.02:
                g_eq[-1] *= (1 - delta * COST_BPS / 10_000)
                g_trades += 1
            g_exp = g_new
            g_rebal = 0

        if not is_first:
            spy_ret = spy.iloc[t] / spy.iloc[t - 1] - 1
            g_eq.append(g_eq[-1] * (1 + spy_ret * g_exp))
        else:
            g_eq.append(g_eq[-1])

        # Record
        records.append({
            'date': d, 'spy': spy_price, 'vix': vix_val, 'ma200': ma200_val,
            'p_risk_raw': round(hmm_p_risk, 4),
            'p_risk_smooth': round(p_risk_smooth, 4),
            'hmm_cap': round(hmm_cap, 4),
            'c_exp': round(c_exp, 4), 'd_exp': round(d_exp, 4),
            'e_exp': round(e_exp, 4), 'e_portfolio': 'risk_off' if p_risk_smooth >= SYS3_SWITCH_THRESHOLD else 'risk_on',
            'f_exp': round(f_exp, 4), 'g_exp': round(g_exp, 4),
            'a_eq': a_eq[-1], 'b_eq': b_eq[-1],
            'c_eq': c_eq[-1], 'd_eq': d_eq[-1],
            'e_eq': e_eq[-1], 'f_eq': f_eq[-1], 'g_eq': g_eq[-1],
        })

    # Drop init
    all_eqs = {
        'A': a_eq[1:], 'B': b_eq[1:], 'C': c_eq[1:],
        'D': d_eq[1:], 'E': e_eq[1:], 'F': f_eq[1:], 'G': g_eq[1:],
    }

    result_df = pd.DataFrame(records)
    result_df['year'] = pd.to_datetime(result_df['date']).dt.year

    # ── Print results ──
    labels = {
        'A': 'SPY B&H',
        'B': 'ETF Portfolio B&H',
        'C': 'V7 Current (ETF)',
        'D': 'ETF+Sys1&2 (HMM cap)',
        'E': 'ETF+Sys3 (HMM context)',
        'F': 'SPY+Sys1&2 (HMM cap)',
        'G': 'SPY+Sys3 (HMM context)',
    }
    eq_cols = {'A': 'a_eq', 'B': 'b_eq', 'C': 'c_eq', 'D': 'd_eq',
               'E': 'e_eq', 'F': 'f_eq', 'G': 'g_eq'}
    exp_cols = {'C': 'c_exp', 'D': 'd_exp', 'E': 'e_exp', 'F': 'f_exp', 'G': 'g_exp'}
    trades_map = {'C': c_trades, 'D': d_trades, 'E': e_trades, 'F': f_trades, 'G': g_trades}

    all_stats = {}
    for key in 'ABCDEFG':
        all_stats[key] = _stats(all_eqs[key], f'{key}: {labels[key]}')

    print(f"\n{'='*80}")
    print(f" RESULTS -- Test: {result_df['date'].iloc[0].date()} to {result_df['date'].iloc[-1].date()}")
    print(f" ({len(result_df)} days, HMM retrains: {hmm_n_retrains})")
    print(f"{'='*80}")

    # Summary table
    print(f"\n  {'Strategy':<28s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s} {'Calmar':>8s} {'TotRet':>8s} {'Trades':>6s}")
    print(f"  {'-'*28} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for key in 'ABCDEFG':
        s = all_stats[key]
        tr = trades_map.get(key, '-')
        print(f"  {s['label']:<28s} {s['sharpe']:7.3f} {s['cagr']:8.2%} "
              f"{s['max_dd']:8.2%} {s['calmar']:8.3f} {s['total_return']:8.2%} {str(tr):>6s}")

    # Average exposure
    print(f"\n--- Average Exposure ---")
    for key in 'CDEFG':
        avg_exp = result_df[exp_cols[key]].mean()
        print(f"  {labels[key]:28s}: {avg_exp:.1%}")

    # Sys3 portfolio switch stats
    n_risk_off = (result_df['e_portfolio'] == 'risk_off').sum()
    print(f"\n  Sys3 portfolio switch: risk-off {n_risk_off}/{len(result_df)} days ({n_risk_off/len(result_df)*100:.1f}%)")

    # Yearly breakdown
    print(f"\n--- Yearly Breakdown ---")
    for yr, grp in result_df.groupby('year'):
        print(f"\n  {yr}:")
        for key in 'ABCDEFG':
            col = eq_cols[key]
            eq = grp[col].values
            if len(eq) < 2:
                continue
            yr_ret = eq[-1] / eq[0] - 1
            yr_eq = eq / eq[0]
            pk = np.maximum.accumulate(yr_eq)
            yr_dd = float(((yr_eq - pk) / pk).min())
            tag = labels[key]
            print(f"    {tag:<28s}: ret={yr_ret:+7.2%}, DD={yr_dd:+6.2%}")

    # HMM p_risk distribution
    print(f"\n--- HMM p_risk_smooth Distribution ---")
    p = result_df['p_risk_smooth']
    for pct in [10, 25, 50, 75, 90]:
        print(f"  P{pct:02d}: {p.quantile(pct/100):.3f}")
    print(f"  Days p_risk > 0.50: {(p > 0.50).sum()} ({(p > 0.50).mean()*100:.1f}%)")
    print(f"  Days p_risk > 0.90: {(p > 0.90).sum()} ({(p > 0.90).mean()*100:.1f}%)")

    # Save
    result_df.to_csv(args.output_dir / 'detail.csv', index=False)
    with open(args.output_dir / 'summary.json', 'w') as f:
        json.dump({k: v for k, v in all_stats.items()}, f, indent=2)
    print(f"\n  Saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
