#!/usr/bin/env python3
"""
LambdaRank Walk-Forward: MA200 Method Comparison
NO time leakage — expanding-window WF cached predictions.

Compares 3 MA200 implementations at SL=3.5%:
  1. No MA200 (baseline)
  2. Daily return multiplier (backtest method) — reduces exposure every day
  3. Buy-time sizing cap (TraderApp live method) — only caps new buys at rebalance

Settings: K=10, equal weight, 5d rebalance, 10bps cost, SL=3.5%
"""

import sys, warnings, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

FEATURES = [
    'volume_price_corr_3d', 'rsi_14', 'reversal_3d', 'momentum_10d',
    'liquid_momentum_10d', 'sharpe_momentum_5d', 'price_ma20_deviation',
    'avg_trade_size', 'trend_r2_20', 'dollar_vol_20', 'ret_skew_20d',
    'reversal_5d', 'near_52w_high', 'atr_pct_14', 'amihud_20',
]

BEST_PARAMS = {
    'learning_rate': 0.04, 'num_leaves': 11, 'max_depth': 3,
    'min_data_in_leaf': 350, 'lambda_l2': 120, 'feature_fraction': 1.0,
    'bagging_fraction': 0.70, 'bagging_freq': 1, 'min_gain_to_split': 0.30,
    'lambdarank_truncation_level': 25, 'sigmoid': 1.1, 'label_gain_power': 2.1,
}

COMMON = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [10, 20],
    'n_quantiles': 64, 'early_stopping_rounds': 50,
}

MA200_DEEP_THR = 0.95
MA200_SHALLOW = 0.60
MA200_DEEP = 0.30
TOP_K = 10
REBALANCE_DAYS = 5
COST_BPS = 10
STOP_LOSS_PCT = 0.035

import lightgbm as lgb


def build_quantile_labels(y, dates, n_quantiles):
    labels = np.zeros(len(y), dtype=np.int32)
    for d in np.unique(dates):
        mask = dates == d
        if np.sum(mask) <= 1:
            continue
        values = y[mask]
        ranks = stats.rankdata(values, method='average')
        quantiles = np.floor(ranks / (len(values) + 1) * n_quantiles).astype(np.int32)
        labels[mask] = np.clip(quantiles, 0, n_quantiles - 1)
    return labels


def group_counts(dates):
    return [int(np.sum(dates == d)) for d in np.unique(dates)]


def purged_cv_splits(dates, n_splits, gap, embargo):
    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)
    fold_size = max(1, n_dates // n_splits)
    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = n_dates if fold == n_splits - 1 else (fold + 1) * fold_size
        val_dates = unique_dates[val_start:val_end]
        train_end = max(0, val_start - gap)
        embargo_start = min(n_dates, val_end + embargo)
        train_dates = np.concatenate((unique_dates[:train_end], unique_dates[embargo_start:]))
        train_mask = np.isin(dates, train_dates)
        val_mask = np.isin(dates, val_dates)
        if np.sum(train_mask) < 100 or np.sum(val_mask) < 50:
            continue
        yield np.where(train_mask)[0], np.where(val_mask)[0]


def train_lambdarank(train_df, feature_cols, params, cv_splits, gap, embargo,
                     n_boost_round, seed):
    X = train_df[feature_cols].fillna(0.0).to_numpy()
    y = train_df['target'].to_numpy()
    dates = train_df.index.get_level_values('date').to_numpy()
    labels = build_quantile_labels(y, dates, COMMON['n_quantiles'])
    lgb_params = {
        'objective': COMMON['objective'], 'metric': COMMON['metric'],
        'ndcg_eval_at': COMMON['ndcg_eval_at'],
        'learning_rate': params['learning_rate'], 'num_leaves': params['num_leaves'],
        'max_depth': params['max_depth'], 'min_data_in_leaf': params['min_data_in_leaf'],
        'lambda_l1': 0.0, 'lambda_l2': params['lambda_l2'],
        'feature_fraction': params['feature_fraction'],
        'bagging_fraction': params['bagging_fraction'],
        'bagging_freq': params['bagging_freq'],
        'min_gain_to_split': params['min_gain_to_split'],
        'lambdarank_truncation_level': params['lambdarank_truncation_level'],
        'sigmoid': params['sigmoid'],
        'verbose': -1, 'force_row_wise': True,
        'seed': seed, 'bagging_seed': seed,
        'feature_fraction_seed': seed, 'data_random_seed': seed,
        'deterministic': True,
    }
    label_gain = [(i / (COMMON['n_quantiles'] - 1)) ** params['label_gain_power']
                  * (COMMON['n_quantiles'] - 1) for i in range(COMMON['n_quantiles'])]
    lgb_params['label_gain'] = label_gain
    rounds_list = []
    for train_idx, val_idx in purged_cv_splits(dates, cv_splits, gap, embargo):
        train_set = lgb.Dataset(X[train_idx], label=labels[train_idx],
                                group=group_counts(dates[train_idx]))
        val_set = lgb.Dataset(X[val_idx], label=labels[val_idx],
                              group=group_counts(dates[val_idx]))
        cbs = [lgb.log_evaluation(0)]
        if COMMON['early_stopping_rounds']:
            cbs.append(lgb.early_stopping(COMMON['early_stopping_rounds']))
        bst = lgb.train(lgb_params, train_set, num_boost_round=n_boost_round,
                        valid_sets=[val_set], valid_names=['val'], callbacks=cbs)
        br = bst.best_iteration if bst.best_iteration > 0 else n_boost_round
        rounds_list.append(br)
    best_round = int(np.mean(rounds_list)) if rounds_list else n_boost_round
    best_round = max(1, best_round)
    full_set = lgb.Dataset(X, label=labels, group=group_counts(dates))
    model = lgb.train(lgb_params, full_set, num_boost_round=best_round,
                      callbacks=[lgb.log_evaluation(0)])
    return model, best_round


def calc_metrics(eq_arr, rebal_days=5):
    if len(eq_arr) < rebal_days + 1:
        return {'CAGR': 0, 'Vol': 0, 'Sharpe': 0, 'MaxDD': 0, 'Calmar': 0}
    idx = np.arange(0, len(eq_arr), rebal_days)
    if idx[-1] != len(eq_arr) - 1:
        idx = np.append(idx, len(eq_arr) - 1)
    sampled = eq_arr[idx]
    period_rets = sampled[1:] / sampled[:-1] - 1
    n_days = len(eq_arr) - 1
    years = n_days / 252
    if years <= 0 or len(period_rets) < 2:
        return {'CAGR': 0, 'Vol': 0, 'Sharpe': 0, 'MaxDD': 0, 'Calmar': 0}
    cagr = (eq_arr[-1] / eq_arr[0]) ** (1.0 / years) - 1
    periods_per_year = 252 / rebal_days
    vol = float(np.std(period_rets, ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(np.mean(period_rets) / np.std(period_rets, ddof=1) * np.sqrt(periods_per_year)) if vol > 0 else 0
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    maxdd = float(dd.min())
    calmar = cagr / abs(maxdd) if maxdd < 0 else 0
    return {'CAGR': cagr, 'Vol': vol, 'Sharpe': sharpe, 'MaxDD': maxdd, 'Calmar': calmar}


def get_ma200_exposure(dt, spy_px, spy_ma200):
    """Get MA200-based exposure multiplier for a given date."""
    if dt in spy_px.index and dt in spy_ma200.index:
        sp = float(spy_px.loc[dt])
        ma = float(spy_ma200.loc[dt])
        if not np.isnan(sp) and not np.isnan(ma) and ma > 0:
            if sp < ma * MA200_DEEP_THR:
                return MA200_DEEP
            elif sp < ma:
                return MA200_SHALLOW
    return 1.0


def simulate_no_ma200(oos_trading_dates, topk_by_date, px, spy_px, spy_ma200,
                      top_k, cost_bps, rebal_days, stop_loss_pct):
    """Method 1: No MA200 overlay at all."""
    pred_dates_ts = [pd.Timestamp(d) for d in sorted(topk_by_date.keys())]
    cap = 1.0
    eq = [cap]
    holdings = []
    entry_prices = {}
    rc = rebal_days
    n_stops = 0

    for i in range(1, len(oos_trading_dates)):
        dt = oos_trading_dates[i]
        prev_dt = oos_trading_dates[i - 1]

        port_ret = 0.0
        stopped = []
        if holdings:
            w = 1.0 / len(holdings)
            for tk in holdings:
                if tk in px.columns and dt in px.index and prev_dt in px.index:
                    p_now = px.loc[dt, tk]
                    p_prev = px.loc[prev_dt, tk]
                    if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                        tk_ret = p_now / p_prev - 1
                        if stop_loss_pct and tk in entry_prices and entry_prices[tk] > 0:
                            dd = (p_now / entry_prices[tk]) - 1.0
                            if dd <= -stop_loss_pct:
                                tk_ret = entry_prices[tk] * (1 - stop_loss_pct) / p_prev - 1
                                stopped.append(tk)
                        port_ret += w * tk_ret

        if stopped:
            n_stops += len(stopped)
            cost_pct = len(stopped) / max(len(holdings), 1) * cost_bps / 10_000
            cap -= cost_pct * cap
            holdings = [tk for tk in holdings if tk not in stopped]
            for tk in stopped:
                entry_prices.pop(tk, None)

        cap *= (1 + port_ret)

        rc += 1
        if rc >= rebal_days:
            rc = 0
            best_pred_dt = None
            for pd_ts in reversed(pred_dates_ts):
                if pd_ts <= dt:
                    best_pred_dt = pd_ts
                    break
            if best_pred_dt is not None:
                for key in topk_by_date:
                    if pd.Timestamp(key) == best_pred_dt:
                        new_h = [tk for tk, sc in topk_by_date[key][:top_k] if tk in px.columns]
                        if new_h:
                            turnover = len(set(new_h) - set(holdings)) / max(len(new_h), 1)
                            cap -= turnover * cost_bps / 10_000 * cap
                            for tk in new_h:
                                if tk not in holdings and dt in px.index:
                                    p = px.loc[dt, tk]
                                    if not np.isnan(p):
                                        entry_prices[tk] = float(p)
                            for tk in holdings:
                                if tk not in new_h:
                                    entry_prices.pop(tk, None)
                            holdings = new_h
                        break
        eq.append(cap)
    return np.array(eq), n_stops


def simulate_daily_multiplier(oos_trading_dates, topk_by_date, px, spy_px, spy_ma200,
                              top_k, cost_bps, rebal_days, stop_loss_pct):
    """Method 2: MA200 as daily return multiplier (backtest method).
    Every day: cap *= (1 + exposure * port_ret)
    Exposure = 0.30/0.60/1.0 based on SPY vs MA200.
    """
    pred_dates_ts = [pd.Timestamp(d) for d in sorted(topk_by_date.keys())]
    cap = 1.0
    eq = [cap]
    holdings = []
    entry_prices = {}
    rc = rebal_days
    n_stops = 0

    for i in range(1, len(oos_trading_dates)):
        dt = oos_trading_dates[i]
        prev_dt = oos_trading_dates[i - 1]

        port_ret = 0.0
        stopped = []
        if holdings:
            w = 1.0 / len(holdings)
            for tk in holdings:
                if tk in px.columns and dt in px.index and prev_dt in px.index:
                    p_now = px.loc[dt, tk]
                    p_prev = px.loc[prev_dt, tk]
                    if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                        tk_ret = p_now / p_prev - 1
                        if stop_loss_pct and tk in entry_prices and entry_prices[tk] > 0:
                            dd = (p_now / entry_prices[tk]) - 1.0
                            if dd <= -stop_loss_pct:
                                tk_ret = entry_prices[tk] * (1 - stop_loss_pct) / p_prev - 1
                                stopped.append(tk)
                        port_ret += w * tk_ret

        if stopped:
            n_stops += len(stopped)
            cost_pct = len(stopped) / max(len(holdings), 1) * cost_bps / 10_000
            cap -= cost_pct * cap
            holdings = [tk for tk in holdings if tk not in stopped]
            for tk in stopped:
                entry_prices.pop(tk, None)

        # DAILY RETURN MULTIPLIER — this is the key difference
        exp = get_ma200_exposure(dt, spy_px, spy_ma200)
        cap *= (1 + exp * port_ret)

        rc += 1
        if rc >= rebal_days:
            rc = 0
            best_pred_dt = None
            for pd_ts in reversed(pred_dates_ts):
                if pd_ts <= dt:
                    best_pred_dt = pd_ts
                    break
            if best_pred_dt is not None:
                for key in topk_by_date:
                    if pd.Timestamp(key) == best_pred_dt:
                        new_h = [tk for tk, sc in topk_by_date[key][:top_k] if tk in px.columns]
                        if new_h:
                            turnover = len(set(new_h) - set(holdings)) / max(len(new_h), 1)
                            cap -= turnover * cost_bps / 10_000 * cap
                            for tk in new_h:
                                if tk not in holdings and dt in px.index:
                                    p = px.loc[dt, tk]
                                    if not np.isnan(p):
                                        entry_prices[tk] = float(p)
                            for tk in holdings:
                                if tk not in new_h:
                                    entry_prices.pop(tk, None)
                            holdings = new_h
                        break
        eq.append(cap)
    return np.array(eq), n_stops


def simulate_buytime_cap(oos_trading_dates, topk_by_date, px, spy_px, spy_ma200,
                         top_k, cost_bps, rebal_days, stop_loss_pct):
    """Method 3: MA200 as buy-time position sizing cap (TraderApp live method).
    At rebalance: each position gets budget * ma200_cap / price shares.
    Between rebalances: positions run at full size, no daily adjustment.
    The "uninvested" fraction (1 - cap) stays as cash earning 0%.
    """
    pred_dates_ts = [pd.Timestamp(d) for d in sorted(topk_by_date.keys())]
    cap = 1.0
    eq = [cap]
    holdings = []
    entry_prices = {}
    invested_frac = 1.0  # fraction of capital actually in stocks (rest is cash)
    rc = rebal_days
    n_stops = 0

    for i in range(1, len(oos_trading_dates)):
        dt = oos_trading_dates[i]
        prev_dt = oos_trading_dates[i - 1]

        port_ret = 0.0
        stopped = []
        if holdings:
            w = 1.0 / len(holdings)
            for tk in holdings:
                if tk in px.columns and dt in px.index and prev_dt in px.index:
                    p_now = px.loc[dt, tk]
                    p_prev = px.loc[prev_dt, tk]
                    if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                        tk_ret = p_now / p_prev - 1
                        if stop_loss_pct and tk in entry_prices and entry_prices[tk] > 0:
                            dd = (p_now / entry_prices[tk]) - 1.0
                            if dd <= -stop_loss_pct:
                                tk_ret = entry_prices[tk] * (1 - stop_loss_pct) / p_prev - 1
                                stopped.append(tk)
                        port_ret += w * tk_ret

        if stopped:
            n_stops += len(stopped)
            cost_pct = len(stopped) / max(len(holdings), 1) * cost_bps / 10_000
            cap -= cost_pct * cap
            holdings = [tk for tk in holdings if tk not in stopped]
            for tk in stopped:
                entry_prices.pop(tk, None)

        # BUY-TIME CAP: only invested_frac of capital participates in returns
        # cash portion (1 - invested_frac) earns 0%
        cap *= (1 + invested_frac * port_ret)

        rc += 1
        if rc >= rebal_days:
            rc = 0
            best_pred_dt = None
            for pd_ts in reversed(pred_dates_ts):
                if pd_ts <= dt:
                    best_pred_dt = pd_ts
                    break
            if best_pred_dt is not None:
                for key in topk_by_date:
                    if pd.Timestamp(key) == best_pred_dt:
                        new_h = [tk for tk, sc in topk_by_date[key][:top_k] if tk in px.columns]
                        if new_h:
                            turnover = len(set(new_h) - set(holdings)) / max(len(new_h), 1)
                            cap -= turnover * cost_bps / 10_000 * cap

                            # AT REBALANCE: determine sizing cap from MA200
                            ma200_cap = get_ma200_exposure(dt, spy_px, spy_ma200)
                            invested_frac = ma200_cap  # e.g., 0.30, 0.60, or 1.0

                            for tk in new_h:
                                if tk not in holdings and dt in px.index:
                                    p = px.loc[dt, tk]
                                    if not np.isnan(p):
                                        entry_prices[tk] = float(p)
                            for tk in holdings:
                                if tk not in new_h:
                                    entry_prices.pop(tk, None)
                            holdings = new_h
                        break
        eq.append(cap)
    return np.array(eq), n_stops


def main():
    sep = "=" * 110

    print(f"\n{sep}")
    print("  LambdaRank WF: MA200 METHOD Comparison")
    print(f"  Settings: K={TOP_K}, equal weight, {REBALANCE_DAYS}d rebal, {COST_BPS}bps cost, SL={STOP_LOSS_PCT*100:.1f}%")
    print(f"  Method 1: No MA200 (baseline)")
    print(f"  Method 2: MA200 daily return multiplier (backtest method)")
    print(f"  Method 3: MA200 buy-time sizing cap (TraderApp live method)")
    print(f"  Walk-forward: expanding window, 252d init, 63d step — NO time leakage")
    print(sep)

    # ─── 1. Load data ───
    data_file = Path('data/factor_exports/polygon_full_features_T5.parquet')
    print(f"\n[1/5] Loading data: {data_file}")
    df = pd.read_parquet(data_file)
    if isinstance(df.index, pd.MultiIndex) and {'date', 'ticker'}.issubset(df.index.names):
        df = df.sort_index()
    elif {'date', 'ticker'}.issubset(df.columns):
        df = df.set_index(['date', 'ticker']).sort_index()
    if 'target' in df.columns:
        df['target'] = df['target'].clip(-0.55, 0.55)
    dates_all = df.index.get_level_values('date').unique().sort_values()
    print(f"  {len(df)} rows, {len(dates_all)} dates ({dates_all[0].date()} .. {dates_all[-1].date()})")

    # ─── 2. WF predictions ───
    cache_file = Path('data/factor_exports/_wf_preds_cache.pkl')
    if cache_file.exists():
        print(f"\n[2/5] Loading cached WF predictions from {cache_file}")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        preds_cat = cache['preds']
        dates_cat = cache['dates']
        tickers_cat = cache['tickers']
        print(f"  Loaded {len(preds_cat)} predictions")
    else:
        print("  ERROR: No WF prediction cache found. Run lambdarank_ma200_stoploss_sweep.py first.")
        sys.exit(1)

    # ─── 3. Build top-K ───
    print(f"\n[3/5] Building top-K holdings...")
    unique_oos_dates = np.sort(np.unique(dates_cat))
    print(f"  OOS: {pd.Timestamp(unique_oos_dates[0]).date()} .. {pd.Timestamp(unique_oos_dates[-1]).date()} ({len(unique_oos_dates)} days)")

    topk_by_date = {}
    for d in unique_oos_dates:
        mask = dates_cat == d
        dp = preds_cat[mask]
        dt_tickers = tickers_cat[mask]
        if len(dp) < 20:
            continue
        order = np.argsort(-dp)[:20]
        topk_by_date[d] = [(dt_tickers[o], float(dp[o])) for o in order]

    # ─── 4. Download prices ───
    print(f"\n[4/5] Downloading prices...")
    all_tickers_set = set()
    for entries in topk_by_date.values():
        for tk, sc in entries:
            all_tickers_set.add(tk)
    all_tickers_set.add('SPY')
    tickers_list = sorted(all_tickers_set)
    print(f"  Downloading {len(tickers_list)} tickers from yfinance...")

    import yfinance as yf
    start_dt = pd.Timestamp(unique_oos_dates[0]) - pd.Timedelta(days=300)
    end_dt = pd.Timestamp(unique_oos_dates[-1]) + pd.Timedelta(days=5)
    px = yf.download(tickers_list, start=start_dt.strftime('%Y-%m-%d'),
                     end=end_dt.strftime('%Y-%m-%d'), progress=False)['Close']
    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers_list[0])
    px = px.ffill()
    print(f"  Price data: {len(px)} days, {px.shape[1]} tickers")

    spy_px = px['SPY'] if 'SPY' in px.columns else pd.Series(dtype=float)
    spy_ma200 = spy_px.rolling(200).mean()

    oos_trading_dates = sorted(px.index[px.index >= pd.Timestamp(unique_oos_dates[0])])

    # ─── Count MA200 regime days ───
    n_deep = 0
    n_shallow = 0
    n_above = 0
    for dt in oos_trading_dates:
        e = get_ma200_exposure(dt, spy_px, spy_ma200)
        if e == MA200_DEEP:
            n_deep += 1
        elif e == MA200_SHALLOW:
            n_shallow += 1
        else:
            n_above += 1
    total = len(oos_trading_dates)
    print(f"\n  MA200 regime breakdown:")
    print(f"    SPY >= MA200:          {n_above:4d} days ({n_above/total:5.1%})")
    print(f"    SPY < MA200 (exp=0.6): {n_shallow:4d} days ({n_shallow/total:5.1%})")
    print(f"    SPY < 0.95*MA200 (0.3):{n_deep:4d} days ({n_deep/total:5.1%})")

    # ─── 5. Run 3 methods ───
    print(f"\n[5/5] Running 3 MA200 methods with SL={STOP_LOSS_PCT*100:.1f}%...")

    methods = [
        ("No MA200", simulate_no_ma200),
        ("Daily multiplier", simulate_daily_multiplier),
        ("Buy-time cap", simulate_buytime_cap),
    ]

    results = {}
    for name, sim_fn in methods:
        eq, n_stops = sim_fn(oos_trading_dates, topk_by_date, px, spy_px, spy_ma200,
                             TOP_K, COST_BPS, REBALANCE_DAYS, STOP_LOSS_PCT)
        m = calc_metrics(eq)
        m['Stops'] = n_stops
        results[name] = {'eq': eq, 'metrics': m}
        print(f"    {name:20s}  Sharpe={m['Sharpe']:6.3f}  CAGR={m['CAGR']:+8.2%}  "
              f"MaxDD={m['MaxDD']:7.1%}  Calmar={m['Calmar']:7.3f}  Vol={m['Vol']:5.1%}  Stops={n_stops}")

    # ═══════════════════════════════════════
    #  FULL PERIOD
    # ═══════════════════════════════════════
    print(f"\n{sep}")
    print(f"  FULL PERIOD RESULTS — SL={STOP_LOSS_PCT*100:.1f}%")
    print(sep)
    print(f"  {'Method':20s}  {'Sharpe':>7s}  {'CAGR':>10s}  {'MaxDD':>7s}  {'Calmar':>8s}  {'Vol':>6s}  {'Stops':>5s}")
    print(f"  {'-'*72}")
    for name, res in results.items():
        m = res['metrics']
        print(f"  {name:20s}  {m['Sharpe']:7.3f}  {m['CAGR']:+9.2%}  {m['MaxDD']:6.1%}  "
              f"{m['Calmar']:8.3f}  {m['Vol']:5.1%}  {m['Stops']:5d}")

    # Delta vs baseline
    base = results["No MA200"]['metrics']
    print(f"\n  Delta vs No MA200:")
    for name in ["Daily multiplier", "Buy-time cap"]:
        m = results[name]['metrics']
        print(f"    {name:20s}: Sharpe {m['Sharpe']-base['Sharpe']:+.3f}, "
              f"CAGR {m['CAGR']-base['CAGR']:+.2%}, MaxDD {m['MaxDD']-base['MaxDD']:+.1%}")

    # ═══════════════════════════════════════
    #  YEARLY BREAKDOWN
    # ═══════════════════════════════════════
    years = sorted(set(d.year for d in oos_trading_dates))

    print(f"\n{sep}")
    print(f"  YEARLY SHARPE")
    print(sep)
    header = f"  {'Method':20s}"
    for yr in years:
        header += f"  {yr:>8d}"
    print(header)
    print(f"  {'-'*(20 + 10*len(years))}")
    for name, res in results.items():
        eq_arr = res['eq']
        line = f"  {name:20s}"
        for yr in years:
            yr_idx = [j for j, d in enumerate(oos_trading_dates) if d.year == yr]
            if len(yr_idx) < 2:
                line += f"  {'N/A':>8s}"
                continue
            yr_eq = eq_arr[yr_idx[0]:yr_idx[-1]+1]
            yr_eq = yr_eq / yr_eq[0]
            yr_m = calc_metrics(yr_eq)
            line += f"  {yr_m['Sharpe']:8.3f}"
        print(line)

    print(f"\n{sep}")
    print(f"  YEARLY CAGR")
    print(sep)
    print(header)
    print(f"  {'-'*(20 + 10*len(years))}")
    for name, res in results.items():
        eq_arr = res['eq']
        line = f"  {name:20s}"
        for yr in years:
            yr_idx = [j for j, d in enumerate(oos_trading_dates) if d.year == yr]
            if len(yr_idx) < 2:
                line += f"  {'N/A':>8s}"
                continue
            yr_eq = eq_arr[yr_idx[0]:yr_idx[-1]+1]
            yr_eq = yr_eq / yr_eq[0]
            yr_m = calc_metrics(yr_eq)
            line += f"  {yr_m['CAGR']:+7.1%}"
        print(line)

    print(f"\n{sep}")
    print(f"  YEARLY MaxDD")
    print(sep)
    print(header)
    print(f"  {'-'*(20 + 10*len(years))}")
    for name, res in results.items():
        eq_arr = res['eq']
        line = f"  {name:20s}"
        for yr in years:
            yr_idx = [j for j, d in enumerate(oos_trading_dates) if d.year == yr]
            if len(yr_idx) < 2:
                line += f"  {'N/A':>8s}"
                continue
            yr_eq = eq_arr[yr_idx[0]:yr_idx[-1]+1]
            peak = np.maximum.accumulate(yr_eq)
            dd = (yr_eq - peak) / peak
            line += f"  {dd.min():7.1%}"
        print(line)

    # ═══════════════════════════════════════
    #  6-FOLD WALKFORWARD ROBUSTNESS
    # ═══════════════════════════════════════
    print(f"\n{sep}")
    print(f"  6-FOLD WALKFORWARD ROBUSTNESS")
    print(sep)
    n_folds = 6
    n_dates_oos = len(oos_trading_dates)
    fold_size = n_dates_oos // n_folds
    folds = []
    for f in range(n_folds):
        f_start = f * fold_size
        f_end = n_dates_oos if f == n_folds - 1 else (f + 1) * fold_size
        folds.append((f_start, f_end))
        d_s = oos_trading_dates[f_start]
        d_e = oos_trading_dates[min(f_end, n_dates_oos) - 1]
        print(f"  Fold {f+1}: {pd.Timestamp(d_s).strftime('%Y-%m-%d')} .. "
              f"{pd.Timestamp(d_e).strftime('%Y-%m-%d')} ({f_end - f_start} days)")

    # Sharpe by fold
    print(f"\n  Sharpe by Fold:")
    header_f = f"  {'Method':20s}"
    for f in range(n_folds):
        header_f += f"  {'F'+str(f+1):>8s}"
    header_f += f"  {'Mean':>8s}  {'StdDev':>8s}  {'Min':>8s}"
    print(header_f)
    print(f"  {'-'*(20 + 10*(n_folds+3))}")

    for name, res in results.items():
        eq_arr = res['eq']
        fold_sharpes = []
        line = f"  {name:20s}"
        for f_start, f_end in folds:
            fold_eq = eq_arr[f_start:f_end]
            fold_eq = fold_eq / fold_eq[0]
            fm = calc_metrics(fold_eq)
            fold_sharpes.append(fm['Sharpe'])
            line += f"  {fm['Sharpe']:8.3f}"
        mean_s = np.mean(fold_sharpes)
        std_s = np.std(fold_sharpes, ddof=1)
        min_s = np.min(fold_sharpes)
        line += f"  {mean_s:8.3f}  {std_s:8.3f}  {min_s:8.3f}"
        results[name]['fold_sharpes'] = fold_sharpes
        print(line)

    # MaxDD by fold
    print(f"\n  MaxDD by Fold:")
    header_f2 = f"  {'Method':20s}"
    for f in range(n_folds):
        header_f2 += f"  {'F'+str(f+1):>8s}"
    header_f2 += f"  {'Worst':>8s}"
    print(header_f2)
    print(f"  {'-'*(20 + 10*(n_folds+1))}")

    for name, res in results.items():
        eq_arr = res['eq']
        fold_mdds = []
        line = f"  {name:20s}"
        for f_start, f_end in folds:
            fold_eq = eq_arr[f_start:f_end]
            peak = np.maximum.accumulate(fold_eq)
            dd = (fold_eq - peak) / peak
            mdd = float(dd.min())
            fold_mdds.append(mdd)
            line += f"  {mdd:7.1%}"
        line += f"  {min(fold_mdds):7.1%}"
        print(line)

    # Fold wins
    ref_sharpes = results["No MA200"]['fold_sharpes']
    print(f"\n  Folds beating No MA200:")
    for name in ["Daily multiplier", "Buy-time cap"]:
        wins = sum(1 for a, b in zip(results[name]['fold_sharpes'], ref_sharpes) if a > b)
        print(f"    {name:20s}: {wins}/{n_folds} folds")

    # Head-to-head
    dm_sharpes = results["Daily multiplier"]['fold_sharpes']
    bt_sharpes = results["Buy-time cap"]['fold_sharpes']
    dm_wins = sum(1 for a, b in zip(dm_sharpes, bt_sharpes) if a > b)
    print(f"\n  Daily multiplier vs Buy-time cap: {dm_wins}/{n_folds} folds")

    # ═══════════════════════════════════════
    #  RECOMMENDATION
    # ═══════════════════════════════════════
    print(f"\n{sep}")
    print(f"  RECOMMENDATION")
    print(sep)

    best_name = max(results, key=lambda n: results[n]['metrics']['Sharpe'])
    best_wf = max(results, key=lambda n: np.mean(results[n].get('fold_sharpes', [0])))
    safest = min(results, key=lambda n: abs(results[n]['metrics']['MaxDD']))

    bm = results[best_name]['metrics']
    print(f"  Best Sharpe:    {best_name:20s}  Sharpe={bm['Sharpe']:.3f}")
    bm = results[best_wf]['metrics']
    print(f"  Best WF Mean:   {best_wf:20s}  Mean={np.mean(results[best_wf]['fold_sharpes']):.3f}")
    bm = results[safest]['metrics']
    print(f"  Lowest MaxDD:   {safest:20s}  MaxDD={bm['MaxDD']:.1%}")

    print(f"\n{sep}")
    print(f"  DONE")
    print(sep)


if __name__ == '__main__':
    main()
