#!/usr/bin/env python3
"""
LambdaRank Walk-Forward: MA200 × Stop-Loss Sweep
NO time leakage — uses expanding-window WF cached predictions.

Grid:
  MA200:     ON / OFF
  Stop Loss: None, 1%, 2%, 3%, 5%, 7%, 10%

Each config tested with:
  - Full period metrics (Sharpe, CAGR, MaxDD, Calmar, Vol)
  - Yearly breakdown
  - 6-fold walkforward robustness (each ~6 months)

Settings: K=10, equal weight, 5d rebalance, 10bps cost
"""

import sys, warnings, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ─── LambdaRank pipeline components ───
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

# ─── Overlay constants ───
MA200_DEEP_THR = 0.95
MA200_SHALLOW = 0.60
MA200_DEEP = 0.30
TOP_K = 10
REBALANCE_DAYS = 5
COST_BPS = 10

import lightgbm as lgb


# ─── LambdaRank training ───
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


# ─── Metrics ───
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


# ─── Portfolio simulation ───
def simulate_portfolio(oos_trading_dates, topk_by_date, px, spy_px, spy_ma200,
                       use_ma200, top_k, cost_bps, rebal_days, stop_loss_pct):
    """
    Equal-weight top-K simulation with optional MA200 overlay and stop loss.
    MA200 overlay: daily return multiplier (same as backtest).
    """
    pred_dates_ts = [pd.Timestamp(d) for d in sorted(topk_by_date.keys())]

    def _get_exposure(dt):
        if not use_ma200:
            return 1.0
        if dt in spy_px.index and dt in spy_ma200.index:
            sp = float(spy_px.loc[dt])
            ma = float(spy_ma200.loc[dt])
            if not np.isnan(sp) and not np.isnan(ma) and ma > 0:
                if sp < ma * MA200_DEEP_THR:
                    return MA200_DEEP
                elif sp < ma:
                    return MA200_SHALLOW
        return 1.0

    cap = 1.0
    eq = [cap]
    holdings = []
    entry_prices = {}
    rc = rebal_days   # trigger first rebalance immediately
    n_stops = 0

    for i in range(1, len(oos_trading_dates)):
        dt = oos_trading_dates[i]
        prev_dt = oos_trading_dates[i - 1]

        # ── Daily portfolio return (equal weight) ──
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

                        # Stop-loss check
                        if (stop_loss_pct is not None and tk in entry_prices
                                and entry_prices[tk] > 0):
                            drawdown = (p_now / entry_prices[tk]) - 1.0
                            if drawdown <= -stop_loss_pct:
                                stop_price = entry_prices[tk] * (1 - stop_loss_pct)
                                tk_ret = stop_price / p_prev - 1
                                stopped.append(tk)

                        port_ret += w * tk_ret

        # Remove stopped stocks
        if stopped:
            n_stops += len(stopped)
            stopped_frac = len(stopped) / max(len(holdings), 1)
            cost_pct = stopped_frac * cost_bps / 10_000
            cap -= cost_pct * cap
            holdings = [tk for tk in holdings if tk not in stopped]
            for tk in stopped:
                entry_prices.pop(tk, None)

        # Apply MA200 exposure as daily return multiplier
        exp = _get_exposure(dt)
        cap *= (1 + exp * port_ret)

        # ── Rebalance ──
        rc += 1
        if rc >= rebal_days:
            rc = 0
            best_pred_dt = None
            for pd_ts in reversed(pred_dates_ts):
                if pd_ts <= dt:
                    best_pred_dt = pd_ts
                    break
            if best_pred_dt is not None:
                for key in topk_by_date.keys():
                    if pd.Timestamp(key) == best_pred_dt:
                        new_holdings = [tk for tk, sc in topk_by_date[key][:top_k]
                                        if tk in px.columns]
                        if new_holdings:
                            turnover = len(set(new_holdings) - set(holdings)) / max(len(new_holdings), 1)
                            cost_pct = turnover * cost_bps / 10_000
                            cap -= cost_pct * cap

                            for tk in new_holdings:
                                if tk not in holdings and dt in px.index:
                                    p_entry = px.loc[dt, tk]
                                    if not np.isnan(p_entry):
                                        entry_prices[tk] = float(p_entry)
                            for tk in holdings:
                                if tk not in new_holdings:
                                    entry_prices.pop(tk, None)

                            holdings = new_holdings
                        break

        eq.append(cap)

    return np.array(eq), n_stops


def main():
    sep = "=" * 110

    # ─── Config grid ───
    MA200_OPTIONS = [False, True]
    SL_OPTIONS = [None, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

    configs = []
    for ma in MA200_OPTIONS:
        for sl in SL_OPTIONS:
            ma_lbl = "MA200" if ma else "NoMA"
            sl_lbl = f"SL={sl*100:.0f}%" if sl is not None else "NoSL"
            name = f"{ma_lbl} + {sl_lbl}"
            configs.append({'name': name, 'ma200': ma, 'sl': sl})

    print(f"\n{sep}")
    print("  LambdaRank WF: MA200 x Stop-Loss Sweep")
    print(f"  Settings: K={TOP_K}, equal weight, {REBALANCE_DAYS}d rebal, {COST_BPS}bps cost")
    print(f"  Grid: {len(configs)} configs (MA200 ON/OFF x SL None/1/2/3/5/7/10%)")
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
    print(f"  {len(df)} rows, {len(dates_all)} dates "
          f"({dates_all[0].date()} .. {dates_all[-1].date()})")

    # ─── 2. Walk-forward predictions (from cache) ───
    INIT_DAYS = 252
    STEP_DAYS = 63
    HORIZON = 5
    CV_SPLITS = 5
    N_BOOST = 800
    SEED = 0

    cache_file = Path('data/factor_exports/_wf_preds_cache.pkl')

    if cache_file.exists():
        print(f"\n[2/5] Loading cached WF predictions from {cache_file}")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        preds_cat = cache['preds']
        targets_cat = cache['targets']
        dates_cat = cache['dates']
        tickers_cat = cache['tickers']
        print(f"  Loaded {len(preds_cat)} predictions")
    else:
        print(f"\n[2/5] Walk-forward LambdaRank (init={INIT_DAYS}d, step={STEP_DAYS}d)...")
        n_dates = len(dates_all)
        all_preds, all_targets, all_dates_out, all_tickers_out = [], [], [], []
        cursor = INIT_DAYS
        fold_num = 0
        while cursor < n_dates:
            fold_num += 1
            test_end = min(cursor + STEP_DAYS, n_dates)
            train_end_idx = max(0, cursor - HORIZON)
            train_dates_sel = dates_all[:train_end_idx]
            test_dates_sel = dates_all[cursor:test_end]
            if len(train_dates_sel) < 100 or len(test_dates_sel) == 0:
                cursor = test_end
                continue
            train_df = df.loc[(train_dates_sel, slice(None)), :]
            test_df = df.loc[(test_dates_sel, slice(None)), :]
            print(f"  [Fold {fold_num}] train: {train_dates_sel[0].date()}..{train_dates_sel[-1].date()} "
                  f"({len(train_dates_sel)}d) | test: {test_dates_sel[0].date()}..{test_dates_sel[-1].date()} "
                  f"({len(test_dates_sel)}d)")
            model, rounds = train_lambdarank(
                train_df, FEATURES, BEST_PARAMS,
                cv_splits=CV_SPLITS, gap=HORIZON, embargo=HORIZON,
                n_boost_round=N_BOOST, seed=SEED)
            X_test = test_df[FEATURES].fillna(0.0).to_numpy()
            preds = model.predict(X_test)
            all_preds.append(preds)
            all_targets.append(test_df['target'].to_numpy())
            all_dates_out.append(test_df.index.get_level_values('date').to_numpy())
            all_tickers_out.append(test_df.index.get_level_values('ticker').to_numpy())
            cursor = test_end
        preds_cat = np.concatenate(all_preds)
        targets_cat = np.concatenate(all_targets)
        dates_cat = np.concatenate(all_dates_out)
        tickers_cat = np.concatenate(all_tickers_out)
        with open(cache_file, 'wb') as f:
            pickle.dump({'preds': preds_cat, 'targets': targets_cat,
                         'dates': dates_cat, 'tickers': tickers_cat}, f)
        print(f"  Saved WF predictions to {cache_file}")

    # ─── 3. Build top-K by date ───
    print(f"\n[3/5] Building top-K holdings...")
    unique_oos_dates = np.sort(np.unique(dates_cat))
    print(f"  OOS period: {pd.Timestamp(unique_oos_dates[0]).date()} .. "
          f"{pd.Timestamp(unique_oos_dates[-1]).date()} ({len(unique_oos_dates)} days)")

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

    # Trading dates
    oos_trading_dates = sorted(px.index[px.index >= pd.Timestamp(unique_oos_dates[0])])

    # ─── 5. Run all configs ───
    print(f"\n[5/5] Running {len(configs)} configs...")
    results = {}

    for cfg in configs:
        eq, n_stops = simulate_portfolio(
            oos_trading_dates, topk_by_date, px, spy_px, spy_ma200,
            use_ma200=cfg['ma200'], top_k=TOP_K, cost_bps=COST_BPS,
            rebal_days=REBALANCE_DAYS, stop_loss_pct=cfg['sl'])
        m = calc_metrics(eq)
        m['Stops'] = n_stops
        results[cfg['name']] = {'eq': eq, 'metrics': m, 'cfg': cfg}
        print(f"    {cfg['name']:22s}  Sharpe={m['Sharpe']:6.3f}  CAGR={m['CAGR']:+8.2%}  "
              f"MaxDD={m['MaxDD']:7.1%}  Calmar={m['Calmar']:7.3f}  Stops={n_stops}")

    # ═══════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════
    print(f"\n{sep}")
    print(f"  MA200 x STOP-LOSS SWEEP — FULL PERIOD")
    print(sep)
    print(f"  {'Config':22s}  {'Sharpe':>7s}  {'CAGR':>10s}  {'MaxDD':>7s}  {'Calmar':>8s}  {'Vol':>6s}  {'Stops':>5s}")
    print(f"  {'-'*70}")
    for name, res in results.items():
        m = res['metrics']
        print(f"  {name:22s}  {m['Sharpe']:7.3f}  {m['CAGR']:+9.2%}  {m['MaxDD']:6.1%}  "
              f"{m['Calmar']:8.3f}  {m['Vol']:5.1%}  {m['Stops']:5d}")

    # ─── Pivot tables: one for each metric ───
    print(f"\n{sep}")
    print(f"  PIVOT: Sharpe (rows=SL, cols=MA200)")
    print(sep)
    sl_labels = ['NoSL', 'SL=1%', 'SL=2%', 'SL=3%', 'SL=5%', 'SL=7%', 'SL=10%']
    ma_labels = ['NoMA', 'MA200']
    print(f"  {'StopLoss':>10s}  {'NoMA200':>10s}  {'MA200':>10s}  {'Delta':>10s}")
    print(f"  {'-'*45}")
    for sl_lbl in sl_labels:
        no_ma_name = f"NoMA + {sl_lbl}"
        ma_name = f"MA200 + {sl_lbl}"
        s_no = results[no_ma_name]['metrics']['Sharpe']
        s_ma = results[ma_name]['metrics']['Sharpe']
        print(f"  {sl_lbl:>10s}  {s_no:10.3f}  {s_ma:10.3f}  {s_ma - s_no:+10.3f}")

    print(f"\n{sep}")
    print(f"  PIVOT: MaxDD (rows=SL, cols=MA200)")
    print(sep)
    print(f"  {'StopLoss':>10s}  {'NoMA200':>10s}  {'MA200':>10s}  {'Delta':>10s}")
    print(f"  {'-'*45}")
    for sl_lbl in sl_labels:
        no_ma_name = f"NoMA + {sl_lbl}"
        ma_name = f"MA200 + {sl_lbl}"
        d_no = results[no_ma_name]['metrics']['MaxDD']
        d_ma = results[ma_name]['metrics']['MaxDD']
        print(f"  {sl_lbl:>10s}  {d_no:9.1%}  {d_ma:9.1%}  {d_ma - d_no:+9.1%}")

    print(f"\n{sep}")
    print(f"  PIVOT: Calmar (rows=SL, cols=MA200)")
    print(sep)
    print(f"  {'StopLoss':>10s}  {'NoMA200':>10s}  {'MA200':>10s}  {'Delta':>10s}")
    print(f"  {'-'*45}")
    for sl_lbl in sl_labels:
        no_ma_name = f"NoMA + {sl_lbl}"
        ma_name = f"MA200 + {sl_lbl}"
        c_no = results[no_ma_name]['metrics']['Calmar']
        c_ma = results[ma_name]['metrics']['Calmar']
        print(f"  {sl_lbl:>10s}  {c_no:10.3f}  {c_ma:10.3f}  {c_ma - c_no:+10.3f}")

    # ─── Yearly breakdown ───
    print(f"\n{sep}")
    print(f"  YEARLY SHARPE BREAKDOWN")
    print(sep)
    years = sorted(set(d.year for d in oos_trading_dates))
    header = f"  {'Config':22s}"
    for yr in years:
        header += f"  {yr:>8d}"
    print(header)
    print(f"  {'-'*(22 + 10*len(years))}")

    for name, res in results.items():
        eq_arr = res['eq']
        line = f"  {name:22s}"
        for yr in years:
            yr_idx = [j for j, d in enumerate(oos_trading_dates) if d.year == yr]
            if len(yr_idx) < 2:
                line += f"  {'N/A':>8s}"
                continue
            yr_eq = eq_arr[yr_idx[0]:yr_idx[-1]+1]
            yr_eq = yr_eq / yr_eq[0]  # normalize to 1.0 at start of year
            yr_m = calc_metrics(yr_eq)
            line += f"  {yr_m['Sharpe']:8.3f}"
        print(line)

    # ─── Yearly MaxDD ───
    print(f"\n{sep}")
    print(f"  YEARLY MaxDD BREAKDOWN")
    print(sep)
    print(header)
    print(f"  {'-'*(22 + 10*len(years))}")

    for name, res in results.items():
        eq_arr = res['eq']
        line = f"  {name:22s}"
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

    # ─── 6-fold walkforward robustness ───
    print(f"\n{sep}")
    print(f"  6-FOLD WALKFORWARD ROBUSTNESS (each fold ~6 months)")
    print(sep)
    n_folds = 6
    n_dates_oos = len(oos_trading_dates)
    fold_size = n_dates_oos // n_folds
    folds = []
    for f in range(n_folds):
        f_start = f * fold_size
        f_end = n_dates_oos if f == n_folds - 1 else (f + 1) * fold_size
        folds.append((f_start, f_end))
        d_start = oos_trading_dates[f_start]
        d_end = oos_trading_dates[min(f_end, n_dates_oos) - 1]
        print(f"  Fold {f+1}: {pd.Timestamp(d_start).strftime('%Y-%m-%d')} .. "
              f"{pd.Timestamp(d_end).strftime('%Y-%m-%d')} ({f_end - f_start} days)")

    print(f"\n  Sharpe by Fold:")
    header = f"  {'Config':22s}"
    for f in range(n_folds):
        header += f"  {'F'+str(f+1):>8s}"
    header += f"  {'Mean':>8s}  {'StdDev':>8s}"
    print(header)
    print(f"  {'-'*(22 + 10*(n_folds+2))}")

    # Reference: best no-overlay config for "wins" counting
    ref_name = "NoMA + NoSL"

    for name, res in results.items():
        eq_arr = res['eq']
        fold_sharpes = []
        line = f"  {name:22s}"
        for f_start, f_end in folds:
            fold_eq = eq_arr[f_start:f_end]
            fold_eq = fold_eq / fold_eq[0]
            fm = calc_metrics(fold_eq)
            fold_sharpes.append(fm['Sharpe'])
            line += f"  {fm['Sharpe']:8.3f}"
        mean_s = np.mean(fold_sharpes)
        std_s = np.std(fold_sharpes, ddof=1) if len(fold_sharpes) > 1 else 0
        line += f"  {mean_s:8.3f}  {std_s:8.3f}"
        results[name]['fold_sharpes'] = fold_sharpes
        print(line)

    # Fold wins vs baseline (NoMA + NoSL)
    if ref_name in results:
        ref_sharpes = results[ref_name]['fold_sharpes']
        print(f"\n  Folds beating {ref_name}:")
        for name, res in results.items():
            if name == ref_name:
                continue
            wins = sum(1 for a, b in zip(res['fold_sharpes'], ref_sharpes) if a > b)
            print(f"    {name:22s}: {wins}/{n_folds} folds")

    # ─── Summary / recommendation ───
    print(f"\n{sep}")
    print(f"  SUMMARY & RECOMMENDATION")
    print(sep)
    best_sharpe_name = max(results, key=lambda n: results[n]['metrics']['Sharpe'])
    best_calmar_name = max(results, key=lambda n: results[n]['metrics']['Calmar'])
    best_fold_name = max(results, key=lambda n: np.mean(results[n].get('fold_sharpes', [0])))

    bm = results[best_sharpe_name]['metrics']
    print(f"  Best Sharpe:  {best_sharpe_name:22s}  Sharpe={bm['Sharpe']:.3f}  CAGR={bm['CAGR']:+.2%}  MaxDD={bm['MaxDD']:.1%}")
    bm = results[best_calmar_name]['metrics']
    print(f"  Best Calmar:  {best_calmar_name:22s}  Calmar={bm['Calmar']:.3f}  CAGR={bm['CAGR']:+.2%}  MaxDD={bm['MaxDD']:.1%}")
    bm_f = results[best_fold_name]
    print(f"  Best WF Mean: {best_fold_name:22s}  Mean Sharpe={np.mean(bm_f['fold_sharpes']):.3f}")

    # Compare MA200 ON vs OFF at best SL
    print(f"\n  MA200 impact at each SL level:")
    for sl_lbl in sl_labels:
        no_name = f"NoMA + {sl_lbl}"
        ma_name = f"MA200 + {sl_lbl}"
        s_diff = results[ma_name]['metrics']['Sharpe'] - results[no_name]['metrics']['Sharpe']
        d_diff = results[ma_name]['metrics']['MaxDD'] - results[no_name]['metrics']['MaxDD']
        print(f"    {sl_lbl:>6s}: Sharpe {s_diff:+.3f}, MaxDD {d_diff:+.1%}")

    print(f"\n{sep}")
    print(f"  DONE")
    print(sep)


if __name__ == '__main__':
    main()
