#!/usr/bin/env python3
"""Detailed analysis of the combo HMM sizing strategy."""
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# Load data
df = pd.read_parquet('data/regime_features.parquet')

# Prepare features
close = df['close']
logret = np.log(close / close.shift(1))
vol_10d = logret.rolling(10, min_periods=10).std()
features = pd.DataFrame({
    'logret': logret, 'vol_10d': vol_10d,
    'vix_z': df['vix_z_score'], 'risk_prem_z': df['qqq_tlt_ratio_z'],
}, index=df.index).dropna()

# Train 3-state HMM
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(features.values)
model = GaussianHMM(n_components=3, covariance_type='full', n_iter=300, random_state=42, tol=1e-5)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    model.fit(X)

means_orig = scaler.inverse_transform(model.means_)
mean_vols = means_orig[:, 1]
order = np.argsort(mean_vols)
label_map = {int(order[0]): 'SAFE', int(order[1]): 'MID', int(order[2]): 'CRISIS'}
crisis_idx = int(order[2])
mid_idx = int(order[1])
safe_idx = int(order[0])

posteriors = model.predict_proba(X)
states = model.predict(X)
labels = [label_map[s] for s in states]

hmm_dates = features.index
p_crisis = posteriors[:, crisis_idx]
p_mid = posteriors[:, mid_idx]
p_safe = posteriors[:, safe_idx]

# Combo: crisis + 0.5*mid
p_risk = p_crisis + 0.5 * p_mid
p_risk_smooth = pd.Series(p_risk, index=hmm_dates).ewm(span=4, adjust=False).mean().values
combo_exposure = np.clip((1 - p_risk_smooth) ** 2, 0.05, 1.0)

# SPY returns
spy_ret = close.reindex(hmm_dates).pct_change().fillna(0).values
combo_ret = spy_ret * combo_exposure
bh_equity = np.cumprod(1 + spy_ret)
combo_equity = np.cumprod(1 + combo_ret)

# Build detailed dataframe
detail = pd.DataFrame({
    'date': hmm_dates,
    'spy_close': close.reindex(hmm_dates).values,
    'spy_ret': spy_ret,
    'hmm_state': labels,
    'p_safe': p_safe,
    'p_mid': p_mid,
    'p_crisis': p_crisis,
    'p_risk_raw': p_risk,
    'p_risk_smooth': p_risk_smooth,
    'combo_exposure': combo_exposure,
    'combo_ret': combo_ret,
    'bh_equity': bh_equity,
    'combo_equity': combo_equity,
    'vix': df['vix_close'].reindex(hmm_dates).values,
})

# === OUTPUT ===
print('=' * 70)
print('COMBO STRATEGY DETAILED ANALYSIS')
print('  Formula: p_risk = p_crisis + 0.5 * p_mid')
print('  Smoothing: EMA(span=4)')
print('  Exposure: clip((1 - p_risk_smooth)^2, 0.05, 1.0)')
print('=' * 70)

# --- Exposure distribution ---
print('\n--- Exposure Distribution ---')
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
labels_bin = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
              '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
cuts = pd.cut(combo_exposure, bins=bins, labels=labels_bin)
dist = cuts.value_counts().sort_index()
for b, cnt in dist.items():
    pct = cnt / len(combo_exposure) * 100
    print(f'  {b:10s}: {cnt:4d} days ({pct:5.1f}%)')

print(f'\n  Percentiles:')
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f'    P{p:02d}: {np.percentile(combo_exposure, p):.1%}')

# --- Drawdown episodes ---
print('\n--- Top 5 Drawdown Episodes (Combo vs B&H) ---')
peak_c = np.maximum.accumulate(combo_equity)
dd_c = (combo_equity - peak_c) / peak_c
peak_b = np.maximum.accumulate(bh_equity)
dd_b = (bh_equity - peak_b) / peak_b

in_dd = False
episodes = []
for i in range(len(dd_c)):
    if dd_c[i] < -0.02 and not in_dd:
        in_dd = True
        start = i
    elif dd_c[i] >= 0 and in_dd:
        in_dd = False
        episodes.append((start, i, float(dd_c[start:i].min())))
if in_dd:
    episodes.append((start, len(dd_c) - 1, float(dd_c[start:].min())))

episodes.sort(key=lambda x: x[2])
for ep in episodes[:5]:
    s, e, mdd = ep
    avg_exp = float(np.mean(combo_exposure[s:e + 1]))
    avg_vix = float(np.mean(detail['vix'].iloc[s:e + 1]))
    bh_dd_ep = float(dd_b[s:e + 1].min())
    print(f'  {detail["date"].iloc[s].date()} to {detail["date"].iloc[e].date()}: '
          f'combo DD={mdd:.1%}, SPY DD={bh_dd_ep:.1%}, '
          f'avg_exp={avg_exp:.1%}, avg_VIX={avg_vix:.1f}, dur={e - s}d')

# --- Yearly breakdown ---
print('\n--- Yearly Breakdown ---')
detail['year'] = pd.to_datetime(detail['date']).dt.year
for yr, grp in detail.groupby('year'):
    c_ret = (1 + grp['combo_ret']).prod() - 1
    b_ret = (1 + grp['spy_ret']).prod() - 1
    c_eq = np.cumprod(1 + grp['combo_ret'].values)
    b_eq = np.cumprod(1 + grp['spy_ret'].values)
    c_dd = float(((c_eq - np.maximum.accumulate(c_eq)) / np.maximum.accumulate(c_eq)).min())
    b_dd = float(((b_eq - np.maximum.accumulate(b_eq)) / np.maximum.accumulate(b_eq)).min())
    avg_exp = grp['combo_exposure'].mean()
    n_low = (grp['combo_exposure'] < 0.50).sum()
    print(f'  {yr}: Combo={c_ret:+7.2%} (DD {c_dd:+6.2%}), '
          f'B&H={b_ret:+7.2%} (DD {b_dd:+6.2%}), '
          f'AvgExp={avg_exp:.1%}, Days<50%={n_low}')

# --- Monthly returns comparison ---
print('\n--- Monthly Returns (Combo vs Buy&Hold) ---')
monthly = detail.copy()
monthly['month'] = pd.to_datetime(monthly['date']).dt.to_period('M')
mret = monthly.groupby('month').agg(
    combo=('combo_ret', lambda x: (1 + x).prod() - 1),
    bh=('spy_ret', lambda x: (1 + x).prod() - 1),
    avg_exp=('combo_exposure', 'mean'),
    avg_vix=('vix', 'mean'),
).reset_index()
print(f'  {"Month":>8s} {"Combo":>8s} {"B&H":>8s} {"Excess":>8s} {"AvgExp":>7s} {"VIX":>5s}')
for _, row in mret.iterrows():
    excess = row['combo'] - row['bh']
    print(f'  {str(row["month"]):>8s} {row["combo"]:+8.2%} {row["bh"]:+8.2%} '
          f'{excess:+8.2%} {row["avg_exp"]:7.1%} {row["avg_vix"]:5.1f}')

# --- Low exposure periods ---
print('\n--- Periods with Exposure < 30% ---')
low_exp = detail[detail['combo_exposure'] < 0.30].copy()
if len(low_exp) > 0:
    low_exp['gap'] = (low_exp['date'].diff() > pd.Timedelta(days=5)).cumsum()
    for gid, grp in low_exp.groupby('gap'):
        start = grp['date'].iloc[0].date()
        end = grp['date'].iloc[-1].date()
        avg_exp = grp['combo_exposure'].mean()
        avg_vix = grp['vix'].mean()
        spy_ret_period = (1 + grp['spy_ret']).prod() - 1
        print(f'  {start} to {end}: {len(grp)}d, avg_exp={avg_exp:.1%}, '
              f'avg_VIX={avg_vix:.1f}, SPY={spy_ret_period:+.2%}')
else:
    print('  No days with exposure < 30%')

# --- Key state transitions at market events ---
print('\n--- HMM State at Major Market Events ---')
events = {
    '2022-06-16': 'June 2022 selloff bottom',
    '2022-09-30': 'Sep 2022 bear low',
    '2022-10-13': 'Oct 2022 CPI/bottom',
    '2023-03-13': 'SVB bank crisis',
    '2023-10-27': '10Y peak / Oct correction',
    '2024-04-19': 'April 2024 pullback',
    '2024-08-05': 'Aug 2024 yen carry unwind',
    '2024-12-18': 'Dec 2024 Fed hawkish',
}
for d_str, evt in events.items():
    d = pd.Timestamp(d_str)
    if d in detail['date'].values:
        row = detail[detail['date'] == d].iloc[0]
    else:
        nearest = detail.iloc[(pd.to_datetime(detail['date']) - d).abs().argsort()[:1]]
        row = nearest.iloc[0]
        d_str = str(row['date'].date())
    print(f'  {d_str} ({evt}):')
    print(f'    State={row["hmm_state"]}, p_safe={row["p_safe"]:.2f}, '
          f'p_mid={row["p_mid"]:.2f}, p_crisis={row["p_crisis"]:.2f}')
    print(f'    p_risk_smooth={row["p_risk_smooth"]:.2f}, '
          f'exposure={row["combo_exposure"]:.1%}, VIX={row["vix"]:.1f}')

# --- P&L Attribution ---
print('\n--- P&L Attribution (days when combo differs from B&H) ---')
# When exposure < 0.9 (actively reducing)
reducing = detail[detail['combo_exposure'] < 0.90].copy()
normal = detail[detail['combo_exposure'] >= 0.90].copy()
print(f'  Reducing days (exposure<90%): {len(reducing)} ({len(reducing)/len(detail)*100:.1f}%)')
print(f'  Normal days (exposure>=90%):  {len(normal)} ({len(normal)/len(detail)*100:.1f}%)')
if len(reducing) > 0:
    red_spy = float(reducing['spy_ret'].sum())
    red_combo = float(reducing['combo_ret'].sum())
    red_saved = red_spy - red_combo
    print(f'  On reducing days:')
    print(f'    SPY cumulative return:   {red_spy:+.2%}')
    print(f'    Combo cumulative return: {red_combo:+.2%}')
    print(f'    Return saved/lost:       {red_saved:+.2%}')
    # Days where SPY was negative
    red_neg = reducing[reducing['spy_ret'] < 0]
    red_pos = reducing[reducing['spy_ret'] >= 0]
    print(f'  Of reducing days: {len(red_neg)} down-days, {len(red_pos)} up-days')
    if len(red_neg) > 0:
        print(f'    Down-day SPY sum:   {float(red_neg["spy_ret"].sum()):+.2%}')
        print(f'    Down-day Combo sum: {float(red_neg["combo_ret"].sum()):+.2%}')
        print(f'    Saved on down days: {float(red_neg["spy_ret"].sum()) - float(red_neg["combo_ret"].sum()):+.2%}')
    if len(red_pos) > 0:
        print(f'    Up-day SPY sum:     {float(red_pos["spy_ret"].sum()):+.2%}')
        print(f'    Up-day Combo sum:   {float(red_pos["combo_ret"].sum()):+.2%}')
        print(f'    Missed on up days:  {float(red_pos["spy_ret"].sum()) - float(red_pos["combo_ret"].sum()):+.2%}')

# Save
detail.to_csv('results/regime_investigation/combo_detail.csv', index=False)
print(f'\nDetailed CSV saved to results/regime_investigation/combo_detail.csv')
