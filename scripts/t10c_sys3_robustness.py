#!/usr/bin/env python3
"""T10C Sys3 Robustness Test — Original vs A vs B.

Tests:
  1. Full-period detailed metrics (Sharpe, Sortino, Calmar, tail risk, win rate, etc.)
  2. Yearly breakdown with all metrics
  3. Monthly return table
  4. Rolling 12-month Sharpe
  5. Parameter sensitivity (vary p_risk thresholds, caps)
  6. Rolling start-date robustness (shift test start by quarters)
  7. Drawdown episode analysis
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
from typing import Dict
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Config ──
PORTFOLIO_RISK_ON = {
    'SMH': 0.25, 'USMV': 0.25, 'QUAL': 0.20,
    'PDBC': 0.15, 'COPX': 0.05, 'URA': 0.05, 'DBA': 0.05,
}
PORTFOLIO_RISK_OFF = {
    'USMV': 0.25, 'QUAL': 0.20, 'GDX': 0.20,
    'PDBC': 0.15, 'DBA': 0.10, 'COPX': 0.05, 'URA': 0.05,
}
TARGET_VOL = 0.12
VOL_BLEND_SHORT = 20; VOL_BLEND_LONG = 60; VOL_BLEND_ALPHA = 0.7
VOL_FLOOR = 0.08; VOL_CAP = 0.40
MA200_SHALLOW_CAP = 0.60; MA200_DEEP_CAP = 0.30; MA200_DEEP_THRESHOLD = -0.05
VIX_ENABLE_THRESH = 25.0; VIX_DISABLE_THRESH = 20.0
VIX_ENABLE_CONFIRM = 2; VIX_DISABLE_CONFIRM = 5; VIX_EXPOSURE_CAP = 0.50
THEME_TICKERS = ['COPX', 'URA']; DEFENSIVE_TICKERS = ['USMV', 'QUAL']
THEME_BUDGET_NORMAL = 0.10; THEME_BUDGET_MEDIUM = 0.06; THEME_BUDGET_HIGH = 0.02
DEADBAND_UP = 0.02; DEADBAND_DOWN = 0.05; MAX_STEP = 0.15
REBALANCE_FREQ = 21; COST_BPS = 10.0; MIN_CASH = 0.05
HMM_MIN_TRAIN = 252; HMM_RETRAIN_FREQ = 21; HMM_EMA_SPAN = 4
HMM_GAMMA = 2; HMM_FLOOR = 0.05; HMM_MID_WEIGHT = 0.5; HMM_N_STATES = 3


# ── Helpers ──

def load_data(cache_path=Path('data/etf_rotation_prices.parquet')):
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    import yfinance as yf
    tickers = list(set(list(PORTFOLIO_RISK_ON) + list(PORTFOLIO_RISK_OFF) + ['SPY', '^VIX']))
    data = yf.download(tickers, start='2021-01-01', end='2026-12-31', auto_adjust=False)
    close = data['Close'].copy()
    close.columns = [c.replace('^', '') for c in close.columns]
    close.index.name = 'date'
    close = close.dropna(subset=['SPY'])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    close.to_parquet(cache_path)
    return close


def compute_portfolio_vol(close_df, weights, t):
    available = [tk for tk in weights if tk in close_df.columns]
    w = np.array([weights[tk] for tk in available]); w = w / w.sum()
    log_ret = np.log(close_df[available] / close_df[available].shift(1))
    port_lr = (log_ret.iloc[:t+1] * w).sum(axis=1).dropna()
    if len(port_lr) < VOL_BLEND_LONG + 5: return 0.15
    v_s = float(port_lr.iloc[-VOL_BLEND_SHORT:].std() * np.sqrt(252))
    v_l = float(port_lr.iloc[-VOL_BLEND_LONG:].std() * np.sqrt(252))
    return max(VOL_FLOOR, min(VOL_BLEND_ALPHA * v_s + (1-VOL_BLEND_ALPHA) * v_l, VOL_CAP))


def vt_exp(bvol):
    return min(TARGET_VOL / bvol if bvol > 0 else 1.0, 1.0)


def ma200_cap_val(sp, ma):
    if np.isnan(ma) or ma <= 0: return 1.0
    dev = (sp - ma) / ma
    if dev >= 0: return 1.0
    elif dev > MA200_DEEP_THRESHOLD: return MA200_SHALLOW_CAP
    else: return MA200_DEEP_CAP


class VixTrigger:
    def __init__(self):
        self.mode = 'baseline'; self.ec = 0; self.dc = 0
    def update(self, sp, ma, vx):
        if np.isnan(vx) or np.isnan(ma): return self.mode == 'vix_active'
        if self.mode == 'baseline':
            if sp < ma and vx >= VIX_ENABLE_THRESH:
                self.ec += 1
                if self.ec >= VIX_ENABLE_CONFIRM: self.mode = 'vix_active'; self.ec = 0
            else: self.ec = 0
        else:
            if sp > ma and vx <= VIX_DISABLE_THRESH:
                self.dc += 1
                if self.dc >= VIX_DISABLE_CONFIRM: self.mode = 'baseline'; self.dc = 0
            else: self.dc = 0
        return self.mode == 'vix_active'


def theme_budget_for_vix(vix):
    if vix < VIX_DISABLE_THRESH: return THEME_BUDGET_NORMAL
    elif vix < VIX_ENABLE_THRESH: return THEME_BUDGET_MEDIUM
    else: return THEME_BUDGET_HIGH


def apply_theme_budget(weights, budget):
    w = weights.copy()
    ts = sum(w.get(t, 0) for t in THEME_TICKERS)
    if ts <= budget: return w
    excess = ts - budget
    if ts > 0:
        sc = budget / ts
        for t in THEME_TICKERS:
            if t in w: w[t] *= sc
    ds = sum(w.get(t, 0) for t in DEFENSIVE_TICKERS)
    if ds > 0:
        for t in DEFENSIVE_TICKERS:
            if t in w: w[t] += excess * (w[t] / ds)
    total = sum(w.values())
    if total > 0: w = {k: v/total for k, v in w.items()}
    return w


def apply_deadband(target, current):
    delta = target - current
    if 0 < delta < DEADBAND_UP: return current
    elif delta < 0 and abs(delta) < DEADBAND_DOWN: return current
    elif delta > MAX_STEP: return current + MAX_STEP
    elif delta < -MAX_STEP: return current - MAX_STEP
    return target


def port_ret(close_df, weights, exposure, t):
    if t < 1: return 0.0
    ret = 0.0
    for tk, w in weights.items():
        if tk in close_df.columns:
            p0 = close_df[tk].iloc[t-1]; p1 = close_df[tk].iloc[t]
            if p0 > 0 and not np.isnan(p1): ret += w * (p1/p0 - 1)
    return ret * exposure


def prepare_hmm_features(spy_close, vix_close):
    logret = np.log(spy_close / spy_close.shift(1))
    vol_10d = logret.rolling(10, min_periods=10).std()
    vix_mean = vix_close.rolling(252, min_periods=60).mean()
    vix_std = vix_close.rolling(252, min_periods=60).std()
    vix_z = (vix_close - vix_mean) / vix_std.replace(0, np.nan)
    spy_mom = spy_close.pct_change(20, fill_method=None)
    m_mean = spy_mom.rolling(252, min_periods=60).mean()
    m_std = spy_mom.rolling(252, min_periods=60).std()
    mom_z = (spy_mom - m_mean) / m_std.replace(0, np.nan)
    return pd.DataFrame({'logret': logret, 'vol_10d': vol_10d,
                         'vix_z': vix_z, 'mom_z': mom_z},
                        index=spy_close.index).dropna()


def fit_hmm(X_raw, seed=42):
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(); X = scaler.fit_transform(X_raw)
    model = GaussianHMM(n_components=HMM_N_STATES, covariance_type="full",
                         n_iter=300, random_state=seed, tol=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); model.fit(X)
    means_orig = scaler.inverse_transform(model.means_)
    order = np.argsort(means_orig[:, 1])
    return model, scaler, int(order[2]), int(order[1])


# ── Simulation Engine ──

def run_simulation(close_df, dates, spy, vix, spy_ma200, hmm_X_all, hmm_d2i,
                   test_start_idx, sys3_switch=0.50, sys3_reduce=0.90,
                   sys3_reduce_cap=0.85, double_bearish_cap=0.85):
    """Run all strategies and return daily records DataFrame."""
    n = len(dates)
    alpha = 2.0 / (HMM_EMA_SPAN + 1)

    # States
    a_eq = [1.0]; b_eq = [1.0]
    c_eq = [1.0]; c_exp = 0.0; c_vix = VixTrigger(); c_rb = 0; c_tr = 0
    e0_eq = [1.0]; e0_exp = 0.0; e0_rb = 0; e0_tr = 0
    ea_eq = [1.0]; ea_exp = 0.0; ea_rb = 0; ea_tr = 0
    eb_eq = [1.0]; eb_exp = 0.0; eb_rb = 0; eb_tr = 0

    hmm_model = None; hmm_scaler = None; hmm_ci = None; hmm_mi = None
    hmm_lt = -1; hmm_nr = 0; p_smooth = 0.0

    records = []

    for t in range(test_start_idx, n):
        d = dates[t]; sp = spy.iloc[t]; ma = spy_ma200.iloc[t]
        vx = vix.iloc[t] if not np.isnan(vix.iloc[t]) else 20.0
        first = (t == test_start_idx)
        spy_below = sp < ma if not np.isnan(ma) else False

        # HMM
        p_risk = 0.0
        if d in hmm_d2i:
            hi = hmm_d2i[d]
            if hmm_model is None or (hi - hmm_lt) >= HMM_RETRAIN_FREQ:
                tX = hmm_X_all[:hi]
                if len(tX) >= HMM_MIN_TRAIN:
                    try:
                        hmm_model, hmm_scaler, hmm_ci, hmm_mi = fit_hmm(tX, 42+hmm_nr)
                        hmm_lt = hi; hmm_nr += 1
                    except: pass
            if hmm_model is not None:
                try:
                    Xs = hmm_scaler.transform(hmm_X_all[:hi+1])
                    post = hmm_model.predict_proba(Xs)
                    p_risk = float(post[-1, hmm_ci]) + HMM_MID_WEIGHT * float(post[-1, hmm_mi])
                except: p_risk = 0.0

        if first: p_smooth = p_risk
        else: p_smooth = alpha * p_risk + (1-alpha) * p_smooth

        hmm_off = (p_smooth >= sys3_switch)
        hmm_extreme = (p_smooth >= sys3_reduce)

        if hmm_off:
            s3_port = PORTFOLIO_RISK_OFF
            s3_w = apply_theme_budget(PORTFOLIO_RISK_OFF, theme_budget_for_vix(vx))
        else:
            s3_port = PORTFOLIO_RISK_ON; s3_w = PORTFOLIO_RISK_ON

        # A
        if not first: a_eq.append(a_eq[-1] * (1 + spy.iloc[t]/spy.iloc[t-1] - 1))
        else: a_eq.append(1.0)

        # B
        if not first: b_eq.append(b_eq[-1] * (1 + port_ret(close_df, PORTFOLIO_RISK_ON, 1.0, t)))
        else: b_eq.append(1.0)

        bvol = compute_portfolio_vol(close_df, PORTFOLIO_RISK_ON, t)
        vte = vt_exp(bvol)

        # C: V7
        c_rb += 1
        if c_rb >= REBALANCE_FREQ or first:
            ct = min(vte, ma200_cap_val(sp, ma))
            if c_vix.update(sp, ma, vx): ct = min(ct, VIX_EXPOSURE_CAP)
            ct = min(ct, 1-MIN_CASH)
            cn = apply_deadband(ct, c_exp)
            dl = abs(cn - c_exp)
            if dl > 0.02: c_eq[-1] *= (1-dl*COST_BPS/10000); c_tr += 1
            c_exp = cn; c_rb = 0
        else: c_vix.update(sp, ma, vx)
        if not first: c_eq.append(c_eq[-1] * (1+port_ret(close_df, PORTFOLIO_RISK_ON, c_exp, t)))
        else: c_eq.append(c_eq[-1])

        # E0: Sys3 original
        e0_rb += 1
        if e0_rb >= REBALANCE_FREQ or first:
            e0_bv = compute_portfolio_vol(close_df, s3_port, t)
            e0t = vt_exp(e0_bv)
            e0t = min(e0t, ma200_cap_val(sp, ma))
            if hmm_extreme: e0t = min(e0t, sys3_reduce_cap)
            e0t = min(e0t, 1-MIN_CASH)
            e0n = apply_deadband(e0t, e0_exp)
            dl = abs(e0n - e0_exp)
            if dl > 0.02: e0_eq[-1] *= (1-dl*COST_BPS/10000); e0_tr += 1
            e0_exp = e0n; e0_rb = 0
        if not first: e0_eq.append(e0_eq[-1] * (1+port_ret(close_df, s3_w, e0_exp, t)))
        else: e0_eq.append(e0_eq[-1])

        # EA: Sys3-A (no MA200)
        ea_rb += 1
        if ea_rb >= REBALANCE_FREQ or first:
            ea_bv = compute_portfolio_vol(close_df, s3_port, t)
            eat = vt_exp(ea_bv)
            if hmm_extreme: eat = min(eat, sys3_reduce_cap)
            eat = min(eat, 1-MIN_CASH)
            ean = apply_deadband(eat, ea_exp)
            dl = abs(ean - ea_exp)
            if dl > 0.02: ea_eq[-1] *= (1-dl*COST_BPS/10000); ea_tr += 1
            ea_exp = ean; ea_rb = 0
        if not first: ea_eq.append(ea_eq[-1] * (1+port_ret(close_df, s3_w, ea_exp, t)))
        else: ea_eq.append(ea_eq[-1])

        # EB: Sys3-B (double-bearish)
        eb_rb += 1
        if eb_rb >= REBALANCE_FREQ or first:
            eb_bv = compute_portfolio_vol(close_df, s3_port, t)
            ebt = vt_exp(eb_bv)
            if hmm_off and spy_below: ebt = min(ebt, double_bearish_cap)
            if hmm_extreme: ebt = min(ebt, sys3_reduce_cap)
            ebt = min(ebt, 1-MIN_CASH)
            ebn = apply_deadband(ebt, eb_exp)
            dl = abs(ebn - eb_exp)
            if dl > 0.02: eb_eq[-1] *= (1-dl*COST_BPS/10000); eb_tr += 1
            eb_exp = ebn; eb_rb = 0
        if not first: eb_eq.append(eb_eq[-1] * (1+port_ret(close_df, s3_w, eb_exp, t)))
        else: eb_eq.append(eb_eq[-1])

        records.append({
            'date': d, 'spy': sp, 'vix': vx, 'p_smooth': round(p_smooth, 4),
            'hmm_off': hmm_off, 'spy_below': spy_below,
            'c_exp': c_exp, 'e0_exp': e0_exp, 'ea_exp': ea_exp, 'eb_exp': eb_exp,
            'a_eq': a_eq[-1], 'b_eq': b_eq[-1],
            'c_eq': c_eq[-1], 'e0_eq': e0_eq[-1],
            'ea_eq': ea_eq[-1], 'eb_eq': eb_eq[-1],
        })

    return pd.DataFrame(records), {'C': c_tr, 'E0': e0_tr, 'EA': ea_tr, 'EB': eb_tr}


# ── Detailed Metrics ──

def full_metrics(eq_arr, label=''):
    eq = np.array(eq_arr)
    rets = eq[1:] / eq[:-1] - 1
    n_d = len(rets)
    if n_d < 10:
        return {}
    ny = n_d / 252.0
    # Basic
    mean_r = np.nanmean(rets)
    std_r = np.nanstd(rets, ddof=1)
    sharpe = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0
    # Sortino
    neg_rets = rets[rets < 0]
    downside_std = np.sqrt(np.nanmean(neg_rets**2)) if len(neg_rets) > 0 else 1e-10
    sortino = mean_r / downside_std * np.sqrt(252) if downside_std > 0 else 0
    # CAGR
    cagr = eq[-1] ** (1.0/ny) - 1 if eq[-1] > 0 and ny > 0 else 0
    # Drawdown
    pk = np.maximum.accumulate(eq)
    dd = (eq - pk) / pk
    max_dd = float(np.nanmin(dd))
    # Calmar
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0
    # Win rate
    win_rate = np.nanmean(rets > 0)
    # Profit factor
    gains = rets[rets > 0].sum() if (rets > 0).any() else 0
    losses = abs(rets[rets < 0].sum()) if (rets < 0).any() else 1e-10
    profit_factor = gains / losses if losses > 0 else 999
    # Tail risk
    var_5 = np.nanpercentile(rets, 5)
    cvar_5 = np.nanmean(rets[rets <= var_5]) if (rets <= var_5).any() else var_5
    # Best/worst
    best_day = np.nanmax(rets)
    worst_day = np.nanmin(rets)
    best_month_rets = pd.Series(rets, index=range(len(rets)))
    # Avg DD duration
    in_dd = dd < 0
    dd_starts = np.diff(in_dd.astype(int))
    # Skewness, kurtosis
    from scipy.stats import skew, kurtosis
    sk = skew(rets[~np.isnan(rets)])
    ku = kurtosis(rets[~np.isnan(rets)])
    # Annualized vol
    ann_vol = std_r * np.sqrt(252)
    # Total return
    total_ret = eq[-1] / eq[0] - 1

    return {
        'label': label,
        'total_return': round(total_ret, 4),
        'cagr': round(cagr, 4),
        'ann_vol': round(ann_vol, 4),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'calmar': round(calmar, 3),
        'max_dd': round(max_dd, 4),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 3),
        'var_5pct': round(var_5, 5),
        'cvar_5pct': round(cvar_5, 5),
        'best_day': round(best_day, 4),
        'worst_day': round(worst_day, 4),
        'skewness': round(sk, 3),
        'kurtosis': round(ku, 3),
        'n_days': n_d,
        'n_years': round(ny, 2),
    }


def yearly_metrics(df, eq_col, exp_col=None):
    """Compute per-year metrics."""
    rows = []
    df = df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year
    for yr, grp in df.groupby('year'):
        eq = grp[eq_col].values
        if len(eq) < 5: continue
        rets = eq[1:] / eq[:-1] - 1
        n_d = len(rets)
        yr_ret = eq[-1] / eq[0] - 1
        yr_eq = eq / eq[0]
        pk = np.maximum.accumulate(yr_eq)
        yr_dd = float(((yr_eq - pk) / pk).min())
        std_r = np.nanstd(rets, ddof=1)
        ann_vol = std_r * np.sqrt(252) if std_r > 0 else 0
        sharpe = (np.nanmean(rets) / std_r * np.sqrt(252)) if std_r > 0 else 0
        neg_r = rets[rets < 0]
        ds = np.sqrt(np.nanmean(neg_r**2)) if len(neg_r) > 0 else 1e-10
        sortino = np.nanmean(rets) / ds * np.sqrt(252) if ds > 0 else 0
        calmar = yr_ret / abs(yr_dd) if yr_dd < 0 else 0
        win_rate = np.nanmean(rets > 0)
        avg_exp = grp[exp_col].mean() if exp_col else 1.0
        rows.append({
            'year': yr, 'return': round(yr_ret, 4), 'max_dd': round(yr_dd, 4),
            'ann_vol': round(ann_vol, 4), 'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3), 'calmar': round(calmar, 3),
            'win_rate': round(win_rate, 3), 'avg_exp': round(avg_exp, 3),
            'n_days': n_d,
        })
    return rows


def monthly_returns(df, eq_col):
    """Monthly return matrix."""
    df = df.copy()
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    rows = []
    for m, grp in df.groupby('month'):
        eq = grp[eq_col].values
        if len(eq) < 2: continue
        ret = eq[-1] / eq[0] - 1
        rows.append({'month': str(m), 'return': ret})
    return rows


# ── Main ──

def main():
    out_dir = Path('results/t10c_sys3_robustness')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("T10C Sys3 Robustness: Original vs A vs B — Comprehensive Test")
    print("=" * 90)

    close_df = load_data()
    dates = close_df.index; n = len(dates)
    spy = close_df['SPY']; vix = close_df['VIX']
    spy_ma200 = spy.rolling(200, min_periods=200).mean()

    hmm_feat = prepare_hmm_features(spy, vix)
    hmm_X_all = hmm_feat.values
    hmm_d2i = {d: i for i, d in enumerate(hmm_feat.index)}

    test_start = None
    for t in range(n):
        d = dates[t]
        if np.isnan(spy_ma200.iloc[t]): continue
        if d in hmm_d2i and hmm_d2i[d] >= HMM_MIN_TRAIN:
            test_start = t; break

    print(f"  Data: {dates[0].date()} to {dates[-1].date()} ({n} days)")
    print(f"  Test: {dates[test_start].date()} to {dates[-1].date()} ({n-test_start} days)")

    # ═══════════════════════════════════════════════════════════
    # TEST 1: Full-period detailed metrics
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(" TEST 1: Full-Period Detailed Metrics")
    print(f"{'='*90}")

    df, trades = run_simulation(close_df, dates, spy, vix, spy_ma200,
                                 hmm_X_all, hmm_d2i, test_start)

    strats = {
        'A': ('a_eq', None, 'SPY B&H'),
        'B': ('b_eq', None, 'ETF B&H'),
        'C': ('c_eq', 'c_exp', 'V7 Current'),
        'E0': ('e0_eq', 'e0_exp', 'Sys3 Original'),
        'EA': ('ea_eq', 'ea_exp', 'Sys3-A (no MA200)'),
        'EB': ('eb_eq', 'eb_exp', 'Sys3-B (dbl-bear)'),
    }

    all_metrics = {}
    for key, (eq_col, exp_col, label) in strats.items():
        eq_arr = df[eq_col].values
        m = full_metrics(eq_arr, label)
        m['trades'] = trades.get(key, 0)
        m['avg_exposure'] = round(df[exp_col].mean(), 3) if exp_col else 1.0
        all_metrics[key] = m

    # Print comparison table
    metric_names = [
        ('total_return', 'Total Return', '{:+.2%}'),
        ('cagr', 'CAGR', '{:+.2%}'),
        ('ann_vol', 'Ann. Vol', '{:.2%}'),
        ('sharpe', 'Sharpe', '{:.3f}'),
        ('sortino', 'Sortino', '{:.3f}'),
        ('max_dd', 'Max DD', '{:.2%}'),
        ('calmar', 'Calmar', '{:.3f}'),
        ('win_rate', 'Win Rate', '{:.1%}'),
        ('profit_factor', 'Profit Factor', '{:.3f}'),
        ('var_5pct', 'VaR 5%', '{:.3%}'),
        ('cvar_5pct', 'CVaR 5%', '{:.3%}'),
        ('best_day', 'Best Day', '{:+.2%}'),
        ('worst_day', 'Worst Day', '{:+.2%}'),
        ('skewness', 'Skewness', '{:.3f}'),
        ('kurtosis', 'Kurtosis', '{:.3f}'),
        ('avg_exposure', 'Avg Exposure', '{:.1%}'),
        ('trades', 'Trades', '{:d}'),
    ]

    keys_order = ['A', 'B', 'C', 'E0', 'EA', 'EB']
    header = f"  {'Metric':<16s}"
    for k in keys_order:
        header += f" {strats[k][2]:>14s}"
    print(header)
    print(f"  {'-'*16}" + f" {'-'*14}" * len(keys_order))

    for mkey, mname, fmt in metric_names:
        row = f"  {mname:<16s}"
        for k in keys_order:
            val = all_metrics[k].get(mkey, 0)
            if mkey == 'trades':
                row += f" {val:>14d}"
            else:
                row += f" {fmt.format(val):>14s}"
        print(row)

    # ═══════════════════════════════════════════════════════════
    # TEST 2: Yearly breakdown with all metrics
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(" TEST 2: Yearly Breakdown")
    print(f"{'='*90}")

    for key in ['A', 'B', 'C', 'E0', 'EA', 'EB']:
        eq_col, exp_col, label = strats[key]
        ym = yearly_metrics(df, eq_col, exp_col)
        print(f"\n  --- {label} ---")
        print(f"  {'Year':>6s} {'Return':>8s} {'MaxDD':>8s} {'Vol':>7s} {'Sharpe':>7s} "
              f"{'Sortino':>8s} {'Calmar':>8s} {'WinR':>6s} {'Exp':>6s}")
        for r in ym:
            print(f"  {r['year']:>6d} {r['return']:+8.2%} {r['max_dd']:+8.2%} "
                  f"{r['ann_vol']:7.2%} {r['sharpe']:7.3f} {r['sortino']:8.3f} "
                  f"{r['calmar']:8.3f} {r['win_rate']:6.1%} {r['avg_exp']:6.1%}")

    # ═══════════════════════════════════════════════════════════
    # TEST 3: Monthly returns comparison
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(" TEST 3: Monthly Returns")
    print(f"{'='*90}")

    monthly_data = {}
    for key in ['C', 'E0', 'EA', 'EB']:
        eq_col = strats[key][0]
        monthly_data[key] = {r['month']: r['return'] for r in monthly_returns(df, eq_col)}

    months = sorted(set().union(*[set(v.keys()) for v in monthly_data.values()]))
    print(f"  {'Month':>8s} {'V7':>8s} {'Sys3Orig':>8s} {'Sys3-A':>8s} {'Sys3-B':>8s}  {'Best':>8s}")
    for m in months:
        vals = {k: monthly_data[k].get(m, 0) for k in ['C', 'E0', 'EA', 'EB']}
        best = max(vals, key=vals.get)
        best_label = {'C': 'V7', 'E0': 'Orig', 'EA': 'A', 'EB': 'B'}[best]
        print(f"  {m:>8s} {vals['C']:+8.2%} {vals['E0']:+8.2%} "
              f"{vals['EA']:+8.2%} {vals['EB']:+8.2%}  {best_label:>8s}")

    # Win count
    wins = {k: 0 for k in ['C', 'E0', 'EA', 'EB']}
    for m in months:
        vals = {k: monthly_data[k].get(m, 0) for k in ['C', 'E0', 'EA', 'EB']}
        best = max(vals, key=vals.get)
        wins[best] += 1
    print(f"\n  Monthly wins: V7={wins['C']}, Orig={wins['E0']}, A={wins['EA']}, B={wins['EB']} (of {len(months)})")

    # ═══════════════════════════════════════════════════════════
    # TEST 4: Rolling 12-month Sharpe
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(" TEST 4: Rolling 12-Month Sharpe (sampled quarterly)")
    print(f"{'='*90}")

    window = 252
    sample_every = 63  # quarterly
    print(f"  {'End Date':>12s} {'V7':>8s} {'Sys3Orig':>8s} {'Sys3-A':>8s} {'Sys3-B':>8s}")
    for i in range(window, len(df), sample_every):
        end_d = df['date'].iloc[i]
        row = f"  {str(end_d.date()):>12s}"
        for key in ['C', 'E0', 'EA', 'EB']:
            eq = df[strats[key][0]].iloc[i-window:i+1].values
            rets = eq[1:]/eq[:-1] - 1
            sr = np.nanmean(rets)/np.nanstd(rets, ddof=1)*np.sqrt(252) if np.nanstd(rets, ddof=1) > 0 else 0
            row += f" {sr:8.3f}"
        print(row)

    # ═══════════════════════════════════════════════════════════
    # TEST 5: Parameter Sensitivity
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(" TEST 5: Parameter Sensitivity (Sys3 Original)")
    print(f"{'='*90}")

    param_configs = [
        # (switch_thresh, reduce_thresh, reduce_cap, label)
        (0.40, 0.85, 0.85, 'switch=0.40 reduce=0.85'),
        (0.45, 0.85, 0.85, 'switch=0.45 reduce=0.85'),
        (0.50, 0.90, 0.85, 'switch=0.50 reduce=0.90 [DEFAULT]'),
        (0.55, 0.90, 0.85, 'switch=0.55 reduce=0.90'),
        (0.60, 0.90, 0.85, 'switch=0.60 reduce=0.90'),
        (0.50, 0.80, 0.85, 'switch=0.50 reduce=0.80'),
        (0.50, 0.90, 0.75, 'switch=0.50 reduce_cap=0.75'),
        (0.50, 0.90, 0.95, 'switch=0.50 reduce_cap=0.95'),
    ]

    print(f"  {'Config':<32s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s} {'Calmar':>8s} {'AvgExp':>7s}")
    print(f"  {'-'*32} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for sw, rd, rc, lbl in param_configs:
        df_p, _ = run_simulation(close_df, dates, spy, vix, spy_ma200,
                                  hmm_X_all, hmm_d2i, test_start,
                                  sys3_switch=sw, sys3_reduce=rd, sys3_reduce_cap=rc)
        m = full_metrics(df_p['e0_eq'].values)
        avg_e = df_p['e0_exp'].mean()
        print(f"  {lbl:<32s} {m['sharpe']:7.3f} {m['cagr']:8.2%} "
              f"{m['max_dd']:8.2%} {m['calmar']:8.3f} {avg_e:7.1%}")

    # Also test Sys3-B with different double-bearish caps
    print(f"\n  --- Sys3-B double-bearish cap sensitivity ---")
    print(f"  {'DB Cap':>8s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s} {'Calmar':>8s} {'AvgExp':>7s}")
    for db_cap in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        df_p, _ = run_simulation(close_df, dates, spy, vix, spy_ma200,
                                  hmm_X_all, hmm_d2i, test_start,
                                  double_bearish_cap=db_cap)
        m = full_metrics(df_p['eb_eq'].values)
        avg_e = df_p['eb_exp'].mean()
        print(f"  {db_cap:8.2f} {m['sharpe']:7.3f} {m['cagr']:8.2%} "
              f"{m['max_dd']:8.2%} {m['calmar']:8.3f} {avg_e:7.1%}")

    # ═══════════════════════════════════════════════════════════
    # TEST 6: Rolling Start-Date Robustness
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(" TEST 6: Rolling Start-Date Robustness (shift test start by quarters)")
    print(f"{'='*90}")

    print(f"  {'Start':>12s} {'Days':>5s}  "
          f"{'V7 Sh':>6s} {'V7 Cal':>7s}  "
          f"{'Orig Sh':>7s} {'Orig Cal':>8s}  "
          f"{'A Sh':>6s} {'A Cal':>7s}  "
          f"{'B Sh':>6s} {'B Cal':>7s}  {'Winner':>8s}")

    quarter_shift = 63
    start_idx = test_start
    n_windows = 0
    orig_wins = 0

    while start_idx + 252 < n:  # need at least 1 year of test data
        df_w, _ = run_simulation(close_df, dates, spy, vix, spy_ma200,
                                  hmm_X_all, hmm_d2i, start_idx)
        mc = full_metrics(df_w['c_eq'].values)
        me0 = full_metrics(df_w['e0_eq'].values)
        mea = full_metrics(df_w['ea_eq'].values)
        meb = full_metrics(df_w['eb_eq'].values)

        calmars = {'V7': mc['calmar'], 'Orig': me0['calmar'],
                   'A': mea['calmar'], 'B': meb['calmar']}
        winner = max(calmars, key=calmars.get)
        if winner == 'Orig': orig_wins += 1
        n_windows += 1

        sd = dates[start_idx].date()
        nd = len(df_w)
        print(f"  {str(sd):>12s} {nd:5d}  "
              f"{mc['sharpe']:6.3f} {mc['calmar']:7.3f}  "
              f"{me0['sharpe']:7.3f} {me0['calmar']:8.3f}  "
              f"{mea['sharpe']:6.3f} {mea['calmar']:7.3f}  "
              f"{meb['sharpe']:6.3f} {meb['calmar']:7.3f}  {winner:>8s}")

        start_idx += quarter_shift

    print(f"\n  Original wins {orig_wins}/{n_windows} windows on Calmar")

    # ═══════════════════════════════════════════════════════════
    # TEST 7: Drawdown Episodes
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print(" TEST 7: Drawdown Episodes (> 3%)")
    print(f"{'='*90}")

    for key in ['C', 'E0', 'EA', 'EB']:
        eq_col, _, label = strats[key]
        eq = df[eq_col].values
        pk = np.maximum.accumulate(eq)
        dd = (eq - pk) / pk

        episodes = []
        in_dd = False
        for i in range(len(dd)):
            if dd[i] < -0.03 and not in_dd:
                in_dd = True; start = i
            elif dd[i] >= 0 and in_dd:
                in_dd = False
                episodes.append((start, i, float(dd[start:i].min())))
        if in_dd:
            episodes.append((start, len(dd)-1, float(dd[start:].min())))

        episodes.sort(key=lambda x: x[2])
        print(f"\n  --- {label} ---")
        print(f"  {'Start':>12s} {'End':>12s} {'MaxDD':>8s} {'Dur':>5s} {'AvgExp':>7s} {'AvgVIX':>7s}")
        for s, e, mdd in episodes[:5]:
            dur = e - s
            avg_exp = df[strats[key][1]].iloc[s:e+1].mean() if strats[key][1] else 1.0
            avg_vix = df['vix'].iloc[s:e+1].mean()
            sd = df['date'].iloc[s].date()
            ed = df['date'].iloc[e].date()
            print(f"  {str(sd):>12s} {str(ed):>12s} {mdd:+8.2%} {dur:5d} {avg_exp:7.1%} {avg_vix:7.1f}")

    # Save all results
    summary = {
        'full_metrics': all_metrics,
        'test_period': f"{df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}",
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    df.to_csv(out_dir / 'detail.csv', index=False)
    print(f"\n  All results saved to {out_dir}/")


if __name__ == '__main__':
    main()
