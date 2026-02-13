#!/usr/bin/env python3
"""Walkforward HMM Combo Sizing — No Data Leakage.

Train HMM on expanding window, retrain every RETRAIN_FREQ days.
Year 1 (2022) = training only. Testing starts Year 2 (2023+).

Usage:
    python scripts/hmm_combo_walkforward.py
    python scripts/hmm_combo_walkforward.py --retrain-freq 63 --min-train 252
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Config ──
MIN_TRAIN_DAYS = 252       # minimum training window (1 year)
RETRAIN_FREQ = 21          # retrain every N trading days
EMA_SPAN = 4               # smoothing span for p_risk
GAMMA = 2                  # exposure exponent
FLOOR = 0.05               # minimum exposure
MID_WEIGHT = 0.5           # weight of MID state in p_risk
N_STATES = 3               # HMM states


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 4 HMM observables from regime_features data."""
    close = df['close']
    logret = np.log(close / close.shift(1))
    vol_10d = logret.rolling(10, min_periods=10).std()
    return pd.DataFrame({
        'logret': logret,
        'vol_10d': vol_10d,
        'vix_z': df['vix_z_score'],
        'risk_prem_z': df['qqq_tlt_ratio_z'],
    }, index=df.index).dropna()


def fit_hmm(X_raw: np.ndarray, seed: int = 42):
    """Train 3-state GaussianHMM, return (model, scaler, label_map, crisis_idx, mid_idx).

    Labels by vol ascending: SAFE < MID < CRISIS.
    """
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=300,
        random_state=seed,
        tol=1e-5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X)

    means_orig = scaler.inverse_transform(model.means_)
    mean_vols = means_orig[:, 1]  # vol_10d column
    order = np.argsort(mean_vols)  # ascending
    label_map = {int(order[0]): 'SAFE', int(order[1]): 'MID', int(order[2]): 'CRISIS'}
    crisis_idx = int(order[2])
    mid_idx = int(order[1])

    return model, scaler, label_map, crisis_idx, mid_idx


def predict_day(model, scaler, X_history_raw: np.ndarray,
                crisis_idx: int, mid_idx: int) -> Tuple[float, float, float, str]:
    """Score all history up to current day, return (p_crisis, p_mid, p_safe, label) for last day.

    IMPORTANT: predict_proba on FULL history (Viterbi uses all past observations).
    """
    X = scaler.transform(X_history_raw)
    posteriors = model.predict_proba(X)
    states = model.predict(X)

    p_crisis = float(posteriors[-1, crisis_idx])
    p_mid = float(posteriors[-1, mid_idx])
    p_safe = 1.0 - p_crisis - p_mid

    label_names = {0: 'S0', 1: 'S1', 2: 'S2'}
    # Override with vol-sorted labels
    mean_vols = scaler.inverse_transform(model.means_)[:, 1]
    order = np.argsort(mean_vols)
    names = {int(order[0]): 'SAFE', int(order[1]): 'MID', int(order[2]): 'CRISIS'}
    label = names.get(int(states[-1]), 'UNK')

    return p_crisis, p_mid, p_safe, label


def compute_stats(rets: np.ndarray) -> Dict:
    """Compute Sharpe, CAGR, MaxDD, Calmar from daily returns."""
    if len(rets) < 2:
        return {}
    cum = np.cumprod(1 + rets)
    n = len(rets)
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1))
    sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
    peak = np.maximum.accumulate(cum)
    max_dd = float(((cum - peak) / peak).min())
    n_years = n / 252.0
    cagr = float(cum[-1] ** (1.0 / n_years) - 1) if cum[-1] > 0 and n_years > 0 else 0.0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    return {
        'sharpe': round(sharpe, 3), 'cagr': round(cagr, 4),
        'max_dd': round(max_dd, 4), 'calmar': round(calmar, 3),
        'total_return': round(float(cum[-1] - 1), 4),
        'n_days': n,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Walkforward HMM Combo Sizing')
    parser.add_argument('--data-file', type=Path,
                        default=Path('data/regime_features.parquet'))
    parser.add_argument('--min-train', type=int, default=MIN_TRAIN_DAYS)
    parser.add_argument('--retrain-freq', type=int, default=RETRAIN_FREQ)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/hmm_walkforward'))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Walkforward HMM Combo Sizing — No Data Leakage")
    print(f"  min_train={args.min_train}d, retrain_freq={args.retrain_freq}d")
    print(f"  combo: p_risk = p_crisis + {MID_WEIGHT}*p_mid")
    print(f"  exposure = clip((1-EMA(p_risk,{EMA_SPAN}))^{GAMMA}, {FLOOR}, 1.0)")
    print("=" * 70)

    # ── Load & prepare ──
    df = pd.read_parquet(args.data_file)
    features = prepare_features(df)
    spy_close = df['close'].reindex(features.index)
    spy_ret = spy_close.pct_change().fillna(0).values
    dates = features.index
    X_all = features.values
    n_total = len(dates)

    print(f"\n  Data: {dates[0].date()} to {dates[-1].date()} ({n_total} days)")
    print(f"  Training: first {args.min_train} days ({dates[0].date()} to {dates[args.min_train-1].date()})")
    print(f"  Testing:  day {args.min_train} onwards ({dates[args.min_train].date()} to {dates[-1].date()})")
    print(f"  Test days: {n_total - args.min_train}")

    # ── Walkforward loop ──
    model = None
    scaler = None
    crisis_idx = None
    mid_idx = None
    label_map = None
    last_train_day = -1
    n_retrains = 0
    p_risk_smooth = 0.0
    alpha = 2.0 / (EMA_SPAN + 1)

    records = []

    for t in range(args.min_train, n_total):
        # Retrain check
        days_since_train = t - last_train_day
        if model is None or days_since_train >= args.retrain_freq:
            # Train on all data up to (but not including) day t
            train_X = X_all[:t]
            try:
                model, scaler, label_map, crisis_idx, mid_idx = fit_hmm(
                    train_X, seed=42 + n_retrains)
                last_train_day = t
                n_retrains += 1
                if n_retrains <= 5 or n_retrains % 10 == 0:
                    print(f"  Retrain #{n_retrains} at day {t} ({dates[t].date()}), "
                          f"train_size={len(train_X)}")
            except Exception as e:
                print(f"  HMM fit failed at day {t}: {e}")
                # Use previous model
                if model is None:
                    continue

        # Predict using all history up to and including day t
        # (HMM forward-backward uses full sequence for posteriors)
        history_X = X_all[:t + 1]
        try:
            p_crisis, p_mid, p_safe, label = predict_day(
                model, scaler, history_X, crisis_idx, mid_idx)
        except Exception:
            p_crisis, p_mid, p_safe, label = 0.0, 0.0, 1.0, 'UNK'

        # Combo risk
        p_risk = p_crisis + MID_WEIGHT * p_mid

        # EMA smoothing (online)
        if t == args.min_train:
            p_risk_smooth = p_risk  # initialize
        else:
            p_risk_smooth = alpha * p_risk + (1 - alpha) * p_risk_smooth

        # Exposure
        exposure = max(FLOOR, (1 - p_risk_smooth) ** GAMMA)
        exposure = min(1.0, exposure)

        # Portfolio return
        combo_ret = spy_ret[t] * exposure

        records.append({
            'date': dates[t],
            'spy_close': spy_close.iloc[t],
            'spy_ret': spy_ret[t],
            'hmm_state': label,
            'p_safe': round(p_safe, 4),
            'p_mid': round(p_mid, 4),
            'p_crisis': round(p_crisis, 4),
            'p_risk': round(p_risk, 4),
            'p_risk_smooth': round(p_risk_smooth, 4),
            'exposure': round(exposure, 4),
            'combo_ret': combo_ret,
            'retrain': 1 if t == last_train_day else 0,
        })

    print(f"\n  Total retrains: {n_retrains}")

    # ── Build results ──
    result_df = pd.DataFrame(records)
    result_df['bh_equity'] = np.cumprod(1 + result_df['spy_ret'].values)
    result_df['combo_equity'] = np.cumprod(1 + result_df['combo_ret'].values)

    # ── Overall stats ──
    bh_stats = compute_stats(result_df['spy_ret'].values)
    combo_stats = compute_stats(result_df['combo_ret'].values)

    print(f"\n{'='*60}")
    print(" WALKFORWARD RESULTS (out-of-sample)")
    print(f"  Test period: {result_df['date'].iloc[0].date()} to {result_df['date'].iloc[-1].date()}")
    print(f"  Test days: {len(result_df)}")
    print(f"{'='*60}")

    print(f"\n  {'Metric':<20s} {'Combo':>10s} {'B&H SPY':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    for key in ['sharpe', 'cagr', 'max_dd', 'calmar', 'total_return']:
        cv = combo_stats.get(key, 0)
        bv = bh_stats.get(key, 0)
        if key in ('cagr', 'max_dd', 'total_return'):
            print(f"  {key:<20s} {cv:>10.2%} {bv:>10.2%}")
        else:
            print(f"  {key:<20s} {cv:>10.3f} {bv:>10.3f}")

    avg_exp = result_df['exposure'].mean()
    print(f"  {'avg_exposure':<20s} {avg_exp:>10.1%} {'100.0%':>10s}")

    # ── Yearly breakdown ──
    print(f"\n--- Yearly Breakdown ---")
    result_df['year'] = pd.to_datetime(result_df['date']).dt.year
    print(f"  {'Year':<6s} {'Combo':>8s} {'B&H':>8s} {'C_DD':>8s} {'B_DD':>8s} "
          f"{'AvgExp':>7s} {'Days<50%':>9s}")
    for yr, grp in result_df.groupby('year'):
        c_ret = (1 + grp['combo_ret']).prod() - 1
        b_ret = (1 + grp['spy_ret']).prod() - 1
        c_eq = np.cumprod(1 + grp['combo_ret'].values)
        b_eq = np.cumprod(1 + grp['spy_ret'].values)
        c_dd = float(((c_eq - np.maximum.accumulate(c_eq)) / np.maximum.accumulate(c_eq)).min())
        b_dd = float(((b_eq - np.maximum.accumulate(b_eq)) / np.maximum.accumulate(b_eq)).min())
        ae = grp['exposure'].mean()
        low = (grp['exposure'] < 0.50).sum()
        print(f"  {yr:<6d} {c_ret:+8.2%} {b_ret:+8.2%} {c_dd:+8.2%} {b_dd:+8.2%} "
              f"{ae:7.1%} {low:9d}")

    # ── Exposure distribution ──
    print(f"\n--- Exposure Distribution ---")
    bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01]
    labels_bin = ['0-10%', '10-20%', '20-30%', '30-50%', '50-70%', '70-90%', '90-100%']
    cuts = pd.cut(result_df['exposure'].values, bins=bins, labels=labels_bin)
    for b in labels_bin:
        cnt = (cuts == b).sum()
        pct = cnt / len(result_df) * 100
        print(f"  {b:10s}: {cnt:4d} days ({pct:5.1f}%)")

    # ── Drawdown episodes ──
    print(f"\n--- Top 5 Drawdown Episodes ---")
    combo_eq = result_df['combo_equity'].values
    bh_eq = result_df['bh_equity'].values
    peak_c = np.maximum.accumulate(combo_eq)
    dd_c = (combo_eq - peak_c) / peak_c
    peak_b = np.maximum.accumulate(bh_eq)
    dd_b = (bh_eq - peak_b) / peak_b

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
        avg_exp = float(np.mean(result_df['exposure'].iloc[s:e + 1]))
        bh_dd_ep = float(dd_b[s:e + 1].min())
        print(f"  {result_df['date'].iloc[s].date()} to {result_df['date'].iloc[e].date()}: "
              f"combo={mdd:.1%}, SPY={bh_dd_ep:.1%}, "
              f"avg_exp={avg_exp:.1%}, dur={e - s}d")

    # ── Key events ──
    print(f"\n--- HMM at Key Events ---")
    events = {
        '2023-03-13': 'SVB crisis',
        '2023-10-27': '10Y peak',
        '2024-04-19': 'Apr pullback',
        '2024-08-05': 'Yen carry unwind',
        '2024-12-18': 'Fed hawkish',
        '2025-03-10': '2025 Mar selloff',
        '2025-04-07': '2025 Apr tariff',
    }
    for d_str, evt in events.items():
        d = pd.Timestamp(d_str)
        mask = (result_df['date'] - d).abs()
        idx = mask.idxmin()
        row = result_df.loc[idx]
        print(f"  {str(row['date'].date()):>10s} ({evt}): "
              f"state={row['hmm_state']}, p_risk={row['p_risk_smooth']:.2f}, "
              f"exp={row['exposure']:.1%}, VIX=n/a")

    # ── Monthly returns ──
    print(f"\n--- Monthly Returns ---")
    result_df['month'] = pd.to_datetime(result_df['date']).dt.to_period('M')
    mret = result_df.groupby('month').agg(
        combo=('combo_ret', lambda x: (1 + x).prod() - 1),
        bh=('spy_ret', lambda x: (1 + x).prod() - 1),
        avg_exp=('exposure', 'mean'),
    ).reset_index()
    print(f"  {'Month':>8s} {'Combo':>8s} {'B&H':>8s} {'Excess':>8s} {'AvgExp':>7s}")
    for _, row in mret.iterrows():
        excess = row['combo'] - row['bh']
        print(f"  {str(row['month']):>8s} {row['combo']:+8.2%} {row['bh']:+8.2%} "
              f"{excess:+8.2%} {row['avg_exp']:7.1%}")

    # ── P&L attribution ──
    print(f"\n--- P&L Attribution ---")
    reducing = result_df[result_df['exposure'] < 0.90]
    normal = result_df[result_df['exposure'] >= 0.90]
    print(f"  Reducing days (exp<90%): {len(reducing)} ({len(reducing)/len(result_df)*100:.1f}%)")
    print(f"  Normal days (exp>=90%):  {len(normal)} ({len(normal)/len(result_df)*100:.1f}%)")
    if len(reducing) > 0:
        neg = reducing[reducing['spy_ret'] < 0]
        pos = reducing[reducing['spy_ret'] >= 0]
        saved = float(neg['spy_ret'].sum() - neg['combo_ret'].sum())
        missed = float(pos['spy_ret'].sum() - pos['combo_ret'].sum())
        print(f"  Down-day savings:  {saved:+.2%} ({len(neg)} days)")
        print(f"  Up-day missed:     {missed:+.2%} ({len(pos)} days)")
        print(f"  Net simple return: {saved - missed:+.2%}")

    # ── Save ──
    result_df.to_csv(args.output_dir / 'walkforward_detail.csv', index=False)

    summary = {
        'config': {
            'min_train': args.min_train,
            'retrain_freq': args.retrain_freq,
            'ema_span': EMA_SPAN,
            'gamma': GAMMA,
            'floor': FLOOR,
            'mid_weight': MID_WEIGHT,
            'n_states': N_STATES,
        },
        'test_period': f"{result_df['date'].iloc[0].date()} to {result_df['date'].iloc[-1].date()}",
        'n_test_days': len(result_df),
        'n_retrains': n_retrains,
        'combo': combo_stats,
        'buy_hold': bh_stats,
        'avg_exposure': round(avg_exp, 3),
    }

    def _safe(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, pd.Timestamp): return str(obj.date())
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, default=_safe), encoding='utf-8')

    print(f"\n  Results saved to {args.output_dir}/")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
