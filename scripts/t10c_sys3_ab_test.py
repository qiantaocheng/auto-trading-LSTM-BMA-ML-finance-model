#!/usr/bin/env python3
"""T10C-Slim Sys3 A vs B test.

Strategies:
  A: SPY B&H (baseline)
  B: ETF B&H (baseline)
  C: V7 Current (production)
  E0: Sys3 Original (MA200 cap kept as-is — may conflict with HMM)
  EA: Sys3-A (delete MA200 cap, fully trust HMM)
  EB: Sys3-B (double-bearish cap: HMM risk-off AND SPY<MA200 -> cap 0.85)

Sys3-B final architecture:
  L1: HMM portfolio switch (p_risk < 0.5 risk-on, >= 0.5 risk-off)
  L2: Vol-target (unchanged)
  L3: Double-bearish cap (HMM risk-off AND SPY < MA200 -> cap 0.85)
  L4: Liquidity cap (p_risk >= 0.9 -> cap 0.85, stacks with L3 -> min 0.72)
  L5: Min cash 5%
  L6: Deadband + 21d rebalance
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

SYS3_SWITCH = 0.50; SYS3_REDUCE = 0.90; SYS3_REDUCE_CAP = 0.85
DOUBLE_BEARISH_CAP = 0.85  # L3 for option B


# ── Helpers (same as full comparison) ──

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
    if len(port_lr) < VOL_BLEND_LONG + 5:
        return 0.15
    v_s = float(port_lr.iloc[-VOL_BLEND_SHORT:].std() * np.sqrt(252))
    v_l = float(port_lr.iloc[-VOL_BLEND_LONG:].std() * np.sqrt(252))
    return max(VOL_FLOOR, min(VOL_BLEND_ALPHA * v_s + (1-VOL_BLEND_ALPHA) * v_l, VOL_CAP))


def vt_exp(bvol):
    return min(TARGET_VOL / bvol if bvol > 0 else 1.0, 1.0)


def ma200_cap_val(spy_price, ma200):
    if np.isnan(ma200) or ma200 <= 0: return 1.0
    dev = (spy_price - ma200) / ma200
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


def _stats(eq_arr, label):
    eq = np.array(eq_arr); rets = eq[1:]/eq[:-1] - 1
    if len(rets) < 2:
        return {'label': label, 'sharpe': 0, 'cagr': 0, 'max_dd': 0, 'calmar': 0, 'total_return': 0}
    n_d = len(rets)
    mr = float(np.nanmean(rets)); sr = float(np.nanstd(rets, ddof=1))
    sharpe = float(mr/sr*np.sqrt(252)) if sr > 0 else 0
    pk = np.maximum.accumulate(eq); mdd = float(np.nanmin((eq-pk)/pk))
    ny = n_d/252.0
    cagr = float(eq[-1]**(1.0/ny) - 1) if eq[-1] > 0 and ny > 0 else 0
    calmar = float(cagr/abs(mdd)) if mdd < 0 else 0
    return {'label': label, 'sharpe': round(sharpe, 3), 'cagr': round(cagr, 4),
            'max_dd': round(mdd, 4), 'calmar': round(calmar, 3),
            'total_return': round(float(eq[-1]-1), 4)}


# ── Main ──

def main():
    out_dir = Path('results/t10c_sys3_ab')
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("T10C Sys3: Option A vs Option B vs Original")
    print("=" * 80)

    close_df = load_data()
    dates = close_df.index; n = len(dates)
    spy = close_df['SPY']; vix = close_df['VIX']
    spy_ma200 = spy.rolling(200, min_periods=200).mean()
    print(f"  Dates: {dates[0].date()} to {dates[-1].date()} ({n} days)")

    hmm_feat = prepare_hmm_features(spy, vix)
    hmm_X_all = hmm_feat.values
    hmm_d2i = {d: i for i, d in enumerate(hmm_feat.index)}

    test_start = None
    for t in range(n):
        d = dates[t]
        if np.isnan(spy_ma200.iloc[t]): continue
        if d in hmm_d2i and hmm_d2i[d] >= HMM_MIN_TRAIN:
            test_start = t; break
    print(f"  Test start: {dates[test_start].date()}, test days: {n - test_start}")

    # Strategy states
    a_eq = [1.0]  # SPY B&H
    b_eq = [1.0]  # ETF B&H
    # C: V7
    c_eq = [1.0]; c_exp = 0.0; c_vix = VixTrigger(); c_rb = 0; c_tr = 0
    # E0: Sys3 original (MA200 cap kept)
    e0_eq = [1.0]; e0_exp = 0.0; e0_rb = 0; e0_tr = 0
    # EA: Sys3-A (no MA200 cap)
    ea_eq = [1.0]; ea_exp = 0.0; ea_rb = 0; ea_tr = 0
    # EB: Sys3-B (double-bearish cap)
    eb_eq = [1.0]; eb_exp = 0.0; eb_rb = 0; eb_tr = 0

    # Shared HMM
    hmm_model = None; hmm_scaler = None; hmm_ci = None; hmm_mi = None
    hmm_lt = -1; hmm_nr = 0; p_smooth = 0.0
    alpha = 2.0 / (HMM_EMA_SPAN + 1)

    records = []

    for t in range(test_start, n):
        d = dates[t]
        sp = spy.iloc[t]; ma = spy_ma200.iloc[t]
        vx = vix.iloc[t] if not np.isnan(vix.iloc[t]) else 20.0
        first = (t == test_start)
        spy_below_ma200 = sp < ma if not np.isnan(ma) else False

        # ── HMM update ──
        p_risk = 0.0
        if d in hmm_d2i:
            hi = hmm_d2i[d]
            if hmm_model is None or (hi - hmm_lt) >= HMM_RETRAIN_FREQ:
                tX = hmm_X_all[:hi]
                if len(tX) >= HMM_MIN_TRAIN:
                    try:
                        hmm_model, hmm_scaler, hmm_ci, hmm_mi = fit_hmm(tX, 42 + hmm_nr)
                        hmm_lt = hi; hmm_nr += 1
                    except Exception: pass
            if hmm_model is not None:
                try:
                    Xs = hmm_scaler.transform(hmm_X_all[:hi+1])
                    post = hmm_model.predict_proba(Xs)
                    p_risk = float(post[-1, hmm_ci]) + HMM_MID_WEIGHT * float(post[-1, hmm_mi])
                except Exception: p_risk = 0.0

        if first: p_smooth = p_risk
        else: p_smooth = alpha * p_risk + (1-alpha) * p_smooth

        hmm_risk_off = (p_smooth >= SYS3_SWITCH)
        hmm_extreme = (p_smooth >= SYS3_REDUCE)

        # Portfolio selection based on HMM
        if hmm_risk_off:
            sys3_portfolio = PORTFOLIO_RISK_OFF
            sys3_weights = apply_theme_budget(PORTFOLIO_RISK_OFF, theme_budget_for_vix(vx))
        else:
            sys3_portfolio = PORTFOLIO_RISK_ON
            sys3_weights = PORTFOLIO_RISK_ON

        # ── A: SPY B&H ──
        if not first:
            a_eq.append(a_eq[-1] * (1 + spy.iloc[t]/spy.iloc[t-1] - 1))
        else:
            a_eq.append(1.0)

        # ── B: ETF B&H ──
        if not first:
            b_eq.append(b_eq[-1] * (1 + port_ret(close_df, PORTFOLIO_RISK_ON, 1.0, t)))
        else:
            b_eq.append(1.0)

        # ── Shared vol ──
        bvol_on = compute_portfolio_vol(close_df, PORTFOLIO_RISK_ON, t)
        vt_on = vt_exp(bvol_on)

        # ════════════════════════════════════════
        # C: V7 Current
        # ════════════════════════════════════════
        c_rb += 1
        if c_rb >= REBALANCE_FREQ or first:
            ct = min(vt_on, ma200_cap_val(sp, ma))
            if c_vix.update(sp, ma, vx): ct = min(ct, VIX_EXPOSURE_CAP)
            ct = min(ct, 1-MIN_CASH)
            cn = apply_deadband(ct, c_exp)
            dl = abs(cn - c_exp)
            if dl > 0.02: c_eq[-1] *= (1 - dl*COST_BPS/10000); c_tr += 1
            c_exp = cn; c_rb = 0
        else:
            c_vix.update(sp, ma, vx)
        if not first:
            c_eq.append(c_eq[-1] * (1 + port_ret(close_df, PORTFOLIO_RISK_ON, c_exp, t)))
        else:
            c_eq.append(c_eq[-1])

        # ════════════════════════════════════════
        # E0: Sys3 Original (MA200 cap kept as-is)
        # ════════════════════════════════════════
        e0_rb += 1
        if e0_rb >= REBALANCE_FREQ or first:
            e0_bvol = compute_portfolio_vol(close_df, sys3_portfolio, t)
            e0t = vt_exp(e0_bvol)
            e0t = min(e0t, ma200_cap_val(sp, ma))  # MA200 cap (original - potential conflict)
            if hmm_extreme: e0t = min(e0t, SYS3_REDUCE_CAP)
            e0t = min(e0t, 1-MIN_CASH)
            e0n = apply_deadband(e0t, e0_exp)
            dl = abs(e0n - e0_exp)
            if dl > 0.02: e0_eq[-1] *= (1 - dl*COST_BPS/10000); e0_tr += 1
            e0_exp = e0n; e0_rb = 0
        if not first:
            e0_eq.append(e0_eq[-1] * (1 + port_ret(close_df, sys3_weights, e0_exp, t)))
        else:
            e0_eq.append(e0_eq[-1])

        # ════════════════════════════════════════
        # EA: Sys3-A (delete MA200 cap, trust HMM)
        # ════════════════════════════════════════
        ea_rb += 1
        if ea_rb >= REBALANCE_FREQ or first:
            ea_bvol = compute_portfolio_vol(close_df, sys3_portfolio, t)
            eat = vt_exp(ea_bvol)
            # NO MA200 cap — fully trust HMM
            if hmm_extreme: eat = min(eat, SYS3_REDUCE_CAP)
            eat = min(eat, 1-MIN_CASH)
            ean = apply_deadband(eat, ea_exp)
            dl = abs(ean - ea_exp)
            if dl > 0.02: ea_eq[-1] *= (1 - dl*COST_BPS/10000); ea_tr += 1
            ea_exp = ean; ea_rb = 0
        if not first:
            ea_eq.append(ea_eq[-1] * (1 + port_ret(close_df, sys3_weights, ea_exp, t)))
        else:
            ea_eq.append(ea_eq[-1])

        # ════════════════════════════════════════
        # EB: Sys3-B (double-bearish cap)
        # ════════════════════════════════════════
        eb_rb += 1
        if eb_rb >= REBALANCE_FREQ or first:
            eb_bvol = compute_portfolio_vol(close_df, sys3_portfolio, t)
            ebt = vt_exp(eb_bvol)
            # L3: Double-bearish — only cap when BOTH HMM risk-off AND SPY < MA200
            if hmm_risk_off and spy_below_ma200:
                ebt = min(ebt, DOUBLE_BEARISH_CAP)
            # L4: Liquidity cap at extreme p_risk (stacks with L3)
            if hmm_extreme:
                ebt = min(ebt, SYS3_REDUCE_CAP)
            ebt = min(ebt, 1-MIN_CASH)
            ebn = apply_deadband(ebt, eb_exp)
            dl = abs(ebn - eb_exp)
            if dl > 0.02: eb_eq[-1] *= (1 - dl*COST_BPS/10000); eb_tr += 1
            eb_exp = ebn; eb_rb = 0
        if not first:
            eb_eq.append(eb_eq[-1] * (1 + port_ret(close_df, sys3_weights, eb_exp, t)))
        else:
            eb_eq.append(eb_eq[-1])

        records.append({
            'date': d, 'spy': sp, 'vix': vx, 'ma200': ma,
            'p_risk_smooth': round(p_smooth, 4),
            'hmm_risk_off': hmm_risk_off, 'spy_below_ma200': spy_below_ma200,
            'double_bearish': hmm_risk_off and spy_below_ma200,
            'c_exp': round(c_exp, 4),
            'e0_exp': round(e0_exp, 4), 'ea_exp': round(ea_exp, 4), 'eb_exp': round(eb_exp, 4),
            'a_eq': a_eq[-1], 'b_eq': b_eq[-1],
            'c_eq': c_eq[-1], 'e0_eq': e0_eq[-1],
            'ea_eq': ea_eq[-1], 'eb_eq': eb_eq[-1],
        })

    # Stats
    eqs = {'A': a_eq[1:], 'B': b_eq[1:], 'C': c_eq[1:],
           'E0': e0_eq[1:], 'EA': ea_eq[1:], 'EB': eb_eq[1:]}
    labels = {
        'A': 'SPY B&H', 'B': 'ETF B&H', 'C': 'V7 Current',
        'E0': 'Sys3 Original', 'EA': 'Sys3-A (no MA200)',
        'EB': 'Sys3-B (dbl-bear)',
    }
    eq_cols = {'A': 'a_eq', 'B': 'b_eq', 'C': 'c_eq',
               'E0': 'e0_eq', 'EA': 'ea_eq', 'EB': 'eb_eq'}
    exp_cols = {'C': 'c_exp', 'E0': 'e0_exp', 'EA': 'ea_exp', 'EB': 'eb_exp'}
    tr_map = {'C': c_tr, 'E0': e0_tr, 'EA': ea_tr, 'EB': eb_tr}

    stats = {k: _stats(eqs[k], f'{labels[k]}') for k in eqs}

    df = pd.DataFrame(records)
    df['year'] = pd.to_datetime(df['date']).dt.year

    print(f"\n{'='*80}")
    print(f" Test: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()} ({len(df)} days)")
    print(f" HMM retrains: {hmm_nr}")
    print(f"{'='*80}")

    print(f"\n  {'Strategy':<22s} {'Sharpe':>7s} {'CAGR':>8s} {'MaxDD':>8s} {'Calmar':>8s} {'TotRet':>8s} {'Tr':>4s}")
    print(f"  {'-'*22} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")
    for k in ['A', 'B', 'C', 'E0', 'EA', 'EB']:
        s = stats[k]
        tr = str(tr_map.get(k, '-'))
        print(f"  {s['label']:<22s} {s['sharpe']:7.3f} {s['cagr']:8.2%} "
              f"{s['max_dd']:8.2%} {s['calmar']:8.3f} {s['total_return']:8.2%} {tr:>4s}")

    # Avg exposure
    print(f"\n--- Avg Exposure ---")
    for k in ['C', 'E0', 'EA', 'EB']:
        print(f"  {labels[k]:<22s}: {df[exp_cols[k]].mean():.1%}")

    # Double-bearish stats
    n_db = df['double_bearish'].sum()
    n_hmm_off = df['hmm_risk_off'].sum()
    n_spy_below = df['spy_below_ma200'].sum()
    print(f"\n--- Signal Overlap ---")
    print(f"  HMM risk-off days:     {n_hmm_off:4d} ({n_hmm_off/len(df)*100:.1f}%)")
    print(f"  SPY < MA200 days:      {n_spy_below:4d} ({n_spy_below/len(df)*100:.1f}%)")
    print(f"  Double-bearish days:   {n_db:4d} ({n_db/len(df)*100:.1f}%)")
    print(f"  HMM-only risk-off:     {n_hmm_off - n_db:4d} (HMM sees risk, MA200 doesn't)")
    print(f"  MA200-only bearish:    {n_spy_below - n_db:4d} (MA200 bearish, HMM says safe)")

    # Yearly
    print(f"\n--- Yearly Breakdown ---")
    for yr, grp in df.groupby('year'):
        print(f"\n  {yr}:")
        for k in ['A', 'B', 'C', 'E0', 'EA', 'EB']:
            eq = grp[eq_cols[k]].values
            if len(eq) < 2: continue
            yr_ret = eq[-1]/eq[0] - 1
            yr_eq = eq/eq[0]; pk = np.maximum.accumulate(yr_eq)
            yr_dd = float(((yr_eq - pk)/pk).min())
            print(f"    {labels[k]:<22s}: ret={yr_ret:+7.2%}, DD={yr_dd:+6.2%}")

    # Conflict analysis: days where HMM and MA200 disagree
    print(f"\n--- Conflict Analysis ---")
    # Case 1: HMM risk-on but SPY < MA200 (MA200 bearish, HMM says safe)
    c1 = df[(~df['hmm_risk_off']) & df['spy_below_ma200']]
    if len(c1) > 0:
        print(f"  HMM risk-on + SPY<MA200: {len(c1)} days")
        # What happened on these days across strategies?
        for k in ['E0', 'EA', 'EB']:
            avg_exp = c1[exp_cols[k]].mean()
            print(f"    {labels[k]:22s} avg_exp={avg_exp:.1%}")

    # Case 2: HMM risk-off but SPY > MA200 (HMM sees risk, MA200 doesn't)
    c2 = df[df['hmm_risk_off'] & (~df['spy_below_ma200'])]
    if len(c2) > 0:
        print(f"  HMM risk-off + SPY>MA200: {len(c2)} days")
        for k in ['E0', 'EA', 'EB']:
            avg_exp = c2[exp_cols[k]].mean()
            print(f"    {labels[k]:22s} avg_exp={avg_exp:.1%}")

    # Save
    df.to_csv(out_dir / 'detail.csv', index=False)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved to {out_dir}/")


if __name__ == '__main__':
    main()
