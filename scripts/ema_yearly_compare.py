#!/usr/bin/env python3
"""Compare Raw vs EMA variants by year, with 2025 quarterly breakdown (tariff impact)."""

import pickle
import numpy as np
import pandas as pd
import yfinance as yf

# Load cached predictions
cache = pickle.load(open('data/factor_exports/_wf_preds_cache.pkl', 'rb'))
preds_cat = cache['preds']
dates_cat = cache['dates']
tickers_cat = cache['tickers']
unique_dates = np.sort(np.unique(dates_cat))


def apply_ema_per_ticker(raw, dates, tickers, span):
    alpha = 2.0 / (span + 1)
    sm = raw.copy()
    ema = {}
    for d in np.sort(np.unique(dates)):
        mask = dates == d
        for idx in np.where(mask)[0]:
            tk = tickers[idx]
            r = raw[idx]
            if tk in ema:
                v = alpha * r + (1 - alpha) * ema[tk]
            else:
                v = r
            sm[idx] = v
            ema[tk] = v
    return sm


def build_topk(use_preds, K=10):
    holdings = {}
    for d in unique_dates:
        mask = dates_cat == d
        dp = use_preds[mask]
        dt = tickers_cat[mask]
        top_idx = np.argsort(-dp)[:K]
        holdings[d] = list(dt[top_idx])
    return holdings


# Precompute EMA variants
print('Computing EMA variants...')
ema2 = apply_ema_per_ticker(preds_cat, dates_cat, tickers_cat, 2)
ema4 = apply_ema_per_ticker(preds_cat, dates_cat, tickers_cat, 4)
ema7 = apply_ema_per_ticker(preds_cat, dates_cat, tickers_cat, 7)
ema10 = apply_ema_per_ticker(preds_cat, dates_cat, tickers_cat, 10)

print('Building top-10 for each variant...')
variants = {
    'Raw': build_topk(preds_cat),
    'EMA(2)': build_topk(ema2),
    'EMA(4)': build_topk(ema4),
    'EMA(7)': build_topk(ema7),
    'EMA(10)': build_topk(ema10),
}

# Collect all unique tickers
all_tickers = set()
for v in variants.values():
    for tks in v.values():
        all_tickers.update(tks)
all_tickers.add('SPY')
print(f'Downloading {len(all_tickers)} tickers...')

prices_raw = yf.download(list(all_tickers), start='2021-12-01', end='2026-01-01',
                          auto_adjust=True, progress=False)['Close']
prices = prices_raw.ffill().bfill()
trading_days = prices.index

# SPY MA200
spy = prices['SPY'].dropna()
spy_ma200 = spy.rolling(200).mean()


def simulate_period(holdings_dict, day_list, sl_pct=0.02):
    """Simulate a period and return (final_equity, max_drawdown, n_stops)."""
    if len(day_list) < 2:
        return 1.0, 0.0, 0

    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    cur_h = []
    entry_p = {}
    n_stops = 0
    cost_bps = 0.001

    # Find initial holdings
    for rd in sorted(unique_dates):
        ts = pd.Timestamp(rd)
        if ts <= day_list[0] and rd in holdings_dict:
            cur_h = holdings_dict[rd]
    for tk in cur_h:
        if tk in prices.columns and day_list[0] in prices.index:
            entry_p[tk] = prices.loc[day_list[0], tk]

    # Rebalance dates (every 5th unique prediction date in the period)
    period_start = day_list[0]
    period_end = day_list[-1]
    rebal_dates = [d for d in unique_dates
                   if pd.Timestamp(d) >= period_start and pd.Timestamp(d) <= period_end]
    rebal_set = set()
    for i, d in enumerate(rebal_dates):
        if i % 5 == 0:
            rebal_set.add(pd.Timestamp(d))

    for i in range(1, len(day_list)):
        day = day_list[i]
        prev = day_list[i - 1]

        # MA200 cap
        ma200_cap = 1.0
        if day in spy.index and day in spy_ma200.index:
            s, m = spy.loc[day], spy_ma200.loc[day]
            if pd.notna(s) and pd.notna(m) and m > 0:
                if s < m * 0.95:
                    ma200_cap = 0.30
                elif s < m:
                    ma200_cap = 0.60

        if len(cur_h) == 0:
            continue

        rets = []
        stopped = []
        for tk in cur_h:
            if tk not in prices.columns:
                rets.append(0.0)
                continue
            pn = prices.loc[day, tk] if day in prices.index else np.nan
            pp = prices.loc[prev, tk] if prev in prices.index else np.nan
            if pd.isna(pn) or pd.isna(pp) or pp <= 0:
                rets.append(0.0)
                continue
            r = pn / pp - 1
            if tk in entry_p and entry_p[tk] > 0:
                dd = (pn / entry_p[tk]) - 1.0
                if dd <= -sl_pct:
                    stop_price = entry_p[tk] * (1 - sl_pct)
                    r = stop_price / pp - 1
                    stopped.append(tk)
            rets.append(r)

        if rets:
            eq *= (1 + np.mean(rets) * ma200_cap)

        peak = max(peak, eq)
        dd = (eq / peak) - 1.0
        max_dd = min(max_dd, dd)

        for tk in stopped:
            cur_h = [t for t in cur_h if t != tk]
            entry_p.pop(tk, None)
            n_stops += 1

        # Rebalance
        np_day = np.datetime64(pd.Timestamp(day))
        if np_day in holdings_dict and pd.Timestamp(day) in rebal_set:
            new_h = holdings_dict[np_day]
            old_set = set(cur_h)
            new_set = set(new_h)
            trades = len(old_set - new_set) + len(new_set - old_set)
            eq *= (1 - cost_bps * trades / max(1, len(new_set)))
            cur_h = new_h
            entry_p = {}
            for tk in cur_h:
                if tk in prices.columns and day in prices.index:
                    entry_p[tk] = prices.loc[day, tk]

    return eq, max_dd, n_stops


# === YEARLY COMPARISON ===
years = [2022, 2023, 2024, 2025]
print()
print('=' * 100)
print('  YEARLY COMPARISON: Raw vs EMA variants  (K=10, MA200=ON, SL=2%, 5d rebal, 10bps)')
print('=' * 100)

header = f'{"Config":<12s}'
for y in years:
    header += f'  {"Return":>8s}  {"MaxDD":>7s}'
header += f'  {"Total":>9s}'
print(header)
print('-' * 100)

for name, hdict in variants.items():
    line = f'{name:<12s}'
    total_eq = 1.0
    for y in years:
        y_days = [d for d in trading_days if d.year == y]
        eq, mdd, ns = simulate_period(hdict, y_days)
        total_eq *= eq
        ret = (eq - 1) * 100
        line += f'  {ret:>+7.1f}%  {mdd*100:>6.1f}%'
    total_ret = (total_eq - 1) * 100
    line += f'  {total_ret:>+8.1f}%'
    print(line)

# === 2022 MONTHLY BREAKDOWN (Bear market) ===
print()
print('=' * 100)
print('  2022 BI-MONTHLY BREAKDOWN (Bear Market)')
print('=' * 100)
periods_2022 = [
    (1, 2, 'Jan-Feb'), (3, 4, 'Mar-Apr'), (5, 6, 'May-Jun'),
    (7, 8, 'Jul-Aug'), (9, 10, 'Sep-Oct'), (11, 12, 'Nov-Dec')
]
header = f'{"Config":<12s}'
for _, _, lbl in periods_2022:
    header += f'  {lbl:>8s}'
print(header)
print('-' * 100)

for name, hdict in variants.items():
    line = f'{name:<12s}'
    for m_start, m_end, _ in periods_2022:
        p_days = [d for d in trading_days if d.year == 2022 and m_start <= d.month <= m_end]
        if not p_days:
            line += f'  {"N/A":>8s}'
            continue
        eq, _, _ = simulate_period(hdict, p_days)
        ret = (eq - 1) * 100
        line += f'  {ret:>+7.1f}%'
    print(line)

# === 2025 QUARTERLY BREAKDOWN (Tariff Impact) ===
print()
print('=' * 100)
print('  2025 QUARTERLY BREAKDOWN (Tariff Impact)')
print('  Q1: Jan-Mar | Q2: Apr-Jun (tariff shock) | Q3: Jul-Sep | Q4: Oct-Dec')
print('=' * 100)
quarters = [
    (1, 3, 'Q1 Jan-Mar'), (4, 6, 'Q2 Apr-Jun'),
    (7, 9, 'Q3 Jul-Sep'), (10, 12, 'Q4 Oct-Dec')
]
header = f'{"Config":<12s}'
for _, _, lbl in quarters:
    header += f'  {lbl:>12s}'
print(header)
print('-' * 100)

for name, hdict in variants.items():
    line = f'{name:<12s}'
    for m_start, m_end, _ in quarters:
        q_days = [d for d in trading_days if d.year == 2025 and m_start <= d.month <= m_end]
        if not q_days:
            line += f'  {"N/A":>12s}'
            continue
        eq, mdd, ns = simulate_period(hdict, q_days)
        ret = (eq - 1) * 100
        line += f'  {ret:>+11.1f}%'
    print(line)

# === STABILITY: Overlap rate during stress periods ===
print()
print('=' * 100)
print('  STABILITY: Adjacent Rebal Overlap Rate (higher = more stable)')
print('=' * 100)

def overlap_rate(hdict, year):
    year_dates = [d for d in unique_dates if pd.Timestamp(d).year == year]
    rebal = [d for i, d in enumerate(year_dates) if i % 5 == 0]
    if len(rebal) < 2:
        return np.nan
    overlaps = []
    for i in range(1, len(rebal)):
        prev_set = set(hdict.get(rebal[i-1], []))
        cur_set = set(hdict.get(rebal[i], []))
        if len(cur_set) > 0:
            overlaps.append(len(prev_set & cur_set) / len(cur_set))
    return np.mean(overlaps) * 100 if overlaps else np.nan

header = f'{"Config":<12s}'
for y in years:
    header += f'  {y:>8d}'
print(header)
print('-' * 60)

for name, hdict in variants.items():
    line = f'{name:<12s}'
    for y in years:
        ovl = overlap_rate(hdict, y)
        line += f'  {ovl:>7.1f}%'
    print(line)

print()
print('=' * 100)
print('  RECOMMENDATION')
print('=' * 100)
