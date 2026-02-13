#!/usr/bin/env python3
"""
Direct Prediction Test v2: 80/20 + Walk-Forward with MA200 & Stop-Loss.
Uses Close prices from parquet for portfolio sim — only downloads SPY (1 ticker).

Target: Close(d+6)/Close(d+1) - 1  (5-day forward return with T+1 lag)
Models: LambdaRank, ElasticNet, XGBoost-Reg, CatBoost-Reg
Portfolio: K=10, EW, 5d rebal, MA200 daily mult, SL=3.4%, 10bps cost
"""

import sys, warnings, time
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

N_QUANTILES = 64
MA200_DEEP_THR = 0.95
MA200_SHALLOW = 0.60
MA200_DEEP = 0.30
TOP_K = 10
REBALANCE_DAYS = 5
COST_BPS = 10
STOP_LOSS_PCT = 0.034

INIT_DAYS = 252
STEP_DAYS = 63
HORIZON = 5
SEED = 0


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
    ppyr = 252 / rebal_days
    vol = float(np.std(period_rets, ddof=1) * np.sqrt(ppyr))
    sharpe = float(np.mean(period_rets) / np.std(period_rets, ddof=1) * np.sqrt(ppyr)) if vol > 0 else 0
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    maxdd = float(dd.min())
    calmar = cagr / abs(maxdd) if maxdd < 0 else 0
    return {'CAGR': cagr, 'Vol': vol, 'Sharpe': sharpe, 'MaxDD': maxdd, 'Calmar': calmar}


def calc_ic(preds, targets, dates):
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if np.sum(mask) < 20:
            continue
        p, t = preds[mask], targets[mask]
        if np.std(p) == 0 or np.std(t) == 0:
            continue
        ic, _ = stats.spearmanr(p, t)
        if not np.isnan(ic):
            ics.append(ic)
    return np.mean(ics) if ics else 0.0


def calc_ndcg_at_k(preds, targets, dates, k=10):
    ndcgs = []
    for d in np.unique(dates):
        mask = dates == d
        if np.sum(mask) < k:
            continue
        p, t = preds[mask], targets[mask]
        topk_idx = np.argsort(-p)[:k]
        ideal_idx = np.argsort(-t)[:k]
        dcg = sum(t[topk_idx[i]] / np.log2(i + 2) for i in range(k))
        idcg = sum(t[ideal_idx[i]] / np.log2(i + 2) for i in range(k))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return np.mean(ndcgs) if ndcgs else 0.0


# ─── Portfolio simulation with MA200 + stop-loss ───
def simulate_portfolio(oos_trading_dates, topk_by_date, px, spy_px, spy_ma200):
    pred_dates_ts = sorted(topk_by_date.keys())

    def _get_exp(dt):
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
    rc = REBALANCE_DAYS
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
                        if STOP_LOSS_PCT and tk in entry_prices and entry_prices[tk] > 0:
                            dd_val = (p_now / entry_prices[tk]) - 1.0
                            if dd_val <= -STOP_LOSS_PCT:
                                tk_ret = entry_prices[tk] * (1 - STOP_LOSS_PCT) / p_prev - 1
                                stopped.append(tk)
                        port_ret += w * tk_ret

        if stopped:
            n_stops += len(stopped)
            cost_pct = len(stopped) / max(len(holdings), 1) * COST_BPS / 10_000
            cap -= cost_pct * cap
            holdings = [tk for tk in holdings if tk not in stopped]
            for tk in stopped:
                entry_prices.pop(tk, None)

        exp = _get_exp(dt)
        cap *= (1 + exp * port_ret)

        rc += 1
        if rc >= REBALANCE_DAYS:
            rc = 0
            best_pred_dt = None
            for pd_ts in reversed(pred_dates_ts):
                if pd_ts <= dt:
                    best_pred_dt = pd_ts
                    break
            if best_pred_dt is not None:
                new_h = [tk for tk, sc in topk_by_date[best_pred_dt][:TOP_K] if tk in px.columns]
                if new_h:
                    turnover = len(set(new_h) - set(holdings)) / max(len(new_h), 1)
                    cap -= turnover * COST_BPS / 10_000 * cap
                    for tk in new_h:
                        if tk not in holdings and dt in px.index:
                            p = px.loc[dt, tk]
                            if not np.isnan(p):
                                entry_prices[tk] = float(p)
                    for tk in holdings:
                        if tk not in new_h:
                            entry_prices.pop(tk, None)
                    holdings = new_h
        eq.append(cap)
    return np.array(eq), n_stops


# ─── Model trainers ───
def build_quantile_labels(y, dates, n_q):
    labels = np.zeros(len(y), dtype=np.int32)
    for d in np.unique(dates):
        mask = dates == d
        if np.sum(mask) <= 1:
            continue
        values = y[mask]
        ranks = stats.rankdata(values, method='average')
        quantiles = np.floor(ranks / (len(values) + 1) * n_q).astype(np.int32)
        labels[mask] = np.clip(quantiles, 0, n_q - 1)
    return labels


def group_counts(dates):
    unique, counts = np.unique(dates, return_counts=True)
    return counts.tolist()


def train_lambdarank(X_train, y_train, dates_train, X_test):
    import lightgbm as lgb
    labels = build_quantile_labels(y_train, dates_train, N_QUANTILES)
    lgb_params = {
        'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [10, 20],
        'learning_rate': 0.04, 'num_leaves': 11, 'max_depth': 3,
        'min_data_in_leaf': 350, 'lambda_l1': 0.0, 'lambda_l2': 120,
        'feature_fraction': 1.0, 'bagging_fraction': 0.70, 'bagging_freq': 1,
        'min_gain_to_split': 0.30, 'lambdarank_truncation_level': 25,
        'sigmoid': 1.1, 'verbose': -1, 'force_row_wise': True,
        'seed': SEED, 'bagging_seed': SEED, 'deterministic': True,
    }
    label_gain = [(i / (N_QUANTILES - 1)) ** 2.1 * (N_QUANTILES - 1) for i in range(N_QUANTILES)]
    lgb_params['label_gain'] = label_gain
    groups = group_counts(dates_train)
    train_set = lgb.Dataset(X_train, label=labels, group=groups)
    model = lgb.train(lgb_params, train_set, num_boost_round=500, callbacks=[lgb.log_evaluation(0)])
    return model.predict(X_test)


def train_elasticnet(X_train, y_train, dates_train, X_test):
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    Xt = scaler.transform(X_test)
    model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000, random_state=SEED)
    model.fit(Xs, y_train)
    return model.predict(Xt)


def train_xgboost_reg(X_train, y_train, dates_train, X_test):
    import xgboost as xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse',
        'eta': 0.04, 'max_depth': 3, 'min_child_weight': 350,
        'subsample': 0.70, 'colsample_bytree': 1.0, 'lambda': 120,
        'gamma': 0.30, 'seed': SEED, 'verbosity': 0,
    }
    model = xgb.train(params, dtrain, num_boost_round=500)
    return model.predict(dtest)


def train_catboost_reg(X_train, y_train, dates_train, X_test):
    from catboost import CatBoost, Pool
    train_pool = Pool(X_train, label=y_train)
    test_pool = Pool(X_test)
    params = {
        'loss_function': 'RMSE', 'iterations': 500, 'learning_rate': 0.04,
        'depth': 3, 'l2_leaf_reg': 120, 'random_seed': SEED,
        'verbose': 0, 'thread_count': -1,
    }
    model = CatBoost(params)
    model.fit(train_pool)
    return model.predict(test_pool)


ALL_TRAINERS = {
    'LambdaRank': train_lambdarank,
    'ElasticNet': train_elasticnet,
    'XGBoost-Reg': train_xgboost_reg,
    'CatBoost-Reg': train_catboost_reg,
}


def run_model_predictions(df, dates_all, model_names, mode, init_days=252, step_days=63, horizon=5):
    """Run either 80/20 or walk-forward, return per-model OOS predictions + topk."""
    if mode == '8020':
        n = len(dates_all)
        split_idx = int(n * 0.8)
        train_dates = dates_all[:split_idx - horizon]
        test_dates = dates_all[split_idx:]
        folds = [(train_dates, test_dates)]
        print(f"\n  Train: {len(train_dates)}d ({train_dates[0].date()}..{train_dates[-1].date()})")
        print(f"  Test:  {len(test_dates)}d ({test_dates[0].date()}..{test_dates[-1].date()})")
    else:
        n = len(dates_all)
        folds = []
        cursor = init_days
        while cursor < n:
            test_end = min(cursor + step_days, n)
            train_end_idx = max(0, cursor - horizon)
            folds.append((dates_all[:train_end_idx], dates_all[cursor:test_end]))
            cursor = test_end
        folds = [(tr, te) for tr, te in folds if len(tr) >= 50 and len(te) > 0]
        print(f"\n  {len(folds)} WF folds, {init_days}d init, {step_days}d step, {horizon}d gap")

    model_preds = {name: {'preds': [], 'dates': [], 'tickers': [], 'targets': []} for name in model_names}

    for fold_num, (train_dates_sel, test_dates_sel) in enumerate(folds, 1):
        train_df = df.loc[(train_dates_sel, slice(None)), :]
        test_df = df.loc[(test_dates_sel, slice(None)), :]

        X_train = train_df[FEATURES].fillna(0.0).to_numpy()
        y_train = train_df['target'].to_numpy()
        dates_train = train_df.index.get_level_values('date').to_numpy()

        X_test = test_df[FEATURES].fillna(0.0).to_numpy()
        y_test = test_df['target'].to_numpy()
        dates_test = test_df.index.get_level_values('date').to_numpy()
        tickers_test = test_df.index.get_level_values('ticker').to_numpy()

        print(f"    [Fold {fold_num}/{len(folds)}] train: {len(train_dates_sel)}d, "
              f"{len(train_df):,}rows | test: {len(test_dates_sel)}d, {len(test_df):,}rows",
              end='', flush=True)

        for name in model_names:
            t0 = time.time()
            try:
                preds = ALL_TRAINERS[name](X_train, y_train, dates_train, X_test)
                elapsed = time.time() - t0
                model_preds[name]['preds'].append(preds)
                model_preds[name]['dates'].append(dates_test)
                model_preds[name]['tickers'].append(tickers_test)
                model_preds[name]['targets'].append(y_test)
                print(f"  {name[:6]}={elapsed:.0f}s", end='', flush=True)
            except Exception as e:
                print(f"  {name[:6]}=FAIL", end='', flush=True)
                model_preds[name]['preds'].append(np.zeros(len(X_test)))
                model_preds[name]['dates'].append(dates_test)
                model_preds[name]['tickers'].append(tickers_test)
                model_preds[name]['targets'].append(y_test)
        print(flush=True)

    # Concatenate
    for name in model_names:
        mp = model_preds[name]
        mp['preds_cat'] = np.concatenate(mp['preds'])
        mp['dates_cat'] = np.concatenate(mp['dates'])
        mp['tickers_cat'] = np.concatenate(mp['tickers'])
        mp['targets_cat'] = np.concatenate(mp['targets'])
    return model_preds


def build_topk(model_preds, model_names, top_n=20):
    """Build topk_by_date dict for each model."""
    topk_all = {}
    for name in model_names:
        mp = model_preds[name]
        unique_dates = np.sort(np.unique(mp['dates_cat']))
        topk = {}
        for d in unique_dates:
            mask = mp['dates_cat'] == d
            dp = mp['preds_cat'][mask]
            dt_tickers = mp['tickers_cat'][mask]
            if len(dp) < 20:
                continue
            order = np.argsort(-dp)[:top_n]
            topk[d] = [(dt_tickers[o], float(dp[o])) for o in order]
        topk_all[name] = topk
    return topk_all


def evaluate_all(model_preds, model_names, topk_all, px, spy_px, spy_ma200, oos_dates, label):
    """Full evaluation: IC, NDCG, portfolio sim with MA200 + SL."""
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)

    # Ranking metrics
    print(f"\n  {'Model':18s}  {'IC':>7s} {'NDCG@10':>8s}  {'Sharpe':>7s} {'CAGR':>10s} {'MaxDD':>7s} "
          f"{'Calmar':>8s} {'Vol':>6s} {'Stops':>6s}")
    print(f"  {'-'*100}")

    results = {}
    for name in model_names:
        mp = model_preds[name]
        ic = calc_ic(mp['preds_cat'], mp['targets_cat'], mp['dates_cat'])
        ndcg = calc_ndcg_at_k(mp['preds_cat'], mp['targets_cat'], mp['dates_cat'], k=TOP_K)

        topk = topk_all[name]
        eq, n_stops = simulate_portfolio(oos_dates, topk, px, spy_px, spy_ma200)
        m = calc_metrics(eq)

        results[name] = {'IC': ic, 'NDCG@10': ndcg, 'eq': eq, 'stops': n_stops, **m}

        print(f"  {name:18s}  {ic:7.4f} {ndcg:8.4f}  {m['Sharpe']:7.3f} {m['CAGR']:+9.2%} "
              f"{m['MaxDD']:6.1%} {m['Calmar']:8.3f} {m['Vol']:5.1%} {n_stops:6d}")

    # 6-fold robustness
    print(f"\n  6-Fold Sharpe Robustness:")
    print(f"  {'Model':18s}  {'F1':>7s} {'F2':>7s} {'F3':>7s} {'F4':>7s} {'F5':>7s} {'F6':>7s}  {'Mean':>7s} {'Min':>7s}")
    print(f"  {'-'*85}")
    for name in model_names:
        eq = results[name]['eq']
        n = len(eq)
        fold_size = n // 6
        fold_sharpes = []
        line = f"  {name:18s}"
        for fi in range(6):
            s = fi * fold_size
            e = (fi + 1) * fold_size if fi < 5 else n
            fold_eq = eq[s:e] / eq[s]
            fm = calc_metrics(fold_eq)
            fold_sharpes.append(fm['Sharpe'])
            line += f"  {fm['Sharpe']:7.3f}"
        line += f"  {np.mean(fold_sharpes):7.3f} {np.min(fold_sharpes):7.3f}"
        print(line)

    return results


def main():
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  DIRECT PREDICTION TEST v2 — MA200 + Stop-Loss")
    print(f"  Data: parquet Close prices (NO yfinance for stocks, only SPY)")
    print(f"  Target: Close(d+6)/Close(d+1) - 1 (T+1 lag, 5-day holding)")
    print(f"  Models: LambdaRank, ElasticNet, XGBoost-Reg, CatBoost-Reg")
    print(f"  Portfolio: K={TOP_K}, EW, {REBALANCE_DAYS}d rebal, MA200 daily mult, "
          f"SL={STOP_LOSS_PCT*100:.1f}%, {COST_BPS}bps")
    print(sep)

    # Load data
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
    print(f"  {len(df):,} rows, {len(dates_all)} dates ({dates_all[0].date()} .. {dates_all[-1].date()})")

    # Build price matrix from parquet Close
    print(f"\n[2/5] Building price matrix from parquet Close column...")
    px = df['Close'].unstack('ticker')
    px = px.ffill()
    print(f"  Price matrix: {px.shape[0]} dates x {px.shape[1]} tickers")

    # Download SPY only
    print(f"  Downloading SPY for MA200...")
    import yfinance as yf
    start_dt = dates_all[0] - pd.Timedelta(days=300)
    end_dt = dates_all[-1] + pd.Timedelta(days=5)
    spy_data = yf.download('SPY', start=start_dt.strftime('%Y-%m-%d'),
                           end=end_dt.strftime('%Y-%m-%d'), progress=False)
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_px = spy_data[('Close', 'SPY')]
    else:
        spy_px = spy_data['Close']
    spy_px = spy_px.ffill()
    spy_ma200 = spy_px.rolling(200).mean()
    print(f"  SPY: {len(spy_px)} days ({spy_px.index[0].date()} .. {spy_px.index[-1].date()})")

    model_names = list(ALL_TRAINERS.keys())

    # ═══════════════════════════════════════
    #  80/20 SPLIT
    # ═══════════════════════════════════════
    print(f"\n{'='*120}")
    print(f"  [3/5] 80/20 CHRONOLOGICAL SPLIT")
    print(f"{'='*120}")

    preds_8020 = run_model_predictions(df, dates_all, model_names, mode='8020')
    topk_8020 = build_topk(preds_8020, model_names)

    # OOS dates for 80/20
    split_idx = int(len(dates_all) * 0.8)
    oos_dates_8020 = sorted(px.index[px.index >= dates_all[split_idx]])

    results_8020 = evaluate_all(preds_8020, model_names, topk_8020, px, spy_px, spy_ma200,
                                oos_dates_8020, "80/20 RESULTS — MA200 + SL=3.4%")

    # ═══════════════════════════════════════
    #  WALK-FORWARD
    # ═══════════════════════════════════════
    print(f"\n{'='*120}")
    print(f"  [4/5] WALK-FORWARD (expanding window, {INIT_DAYS}d init, {STEP_DAYS}d step, {HORIZON}d gap)")
    print(f"{'='*120}")

    preds_wf = run_model_predictions(df, dates_all, model_names, mode='wf')
    topk_wf = build_topk(preds_wf, model_names)

    # OOS dates for WF: starts from INIT_DAYS
    first_oos_date = dates_all[INIT_DAYS]
    oos_dates_wf = sorted(px.index[px.index >= first_oos_date])

    results_wf = evaluate_all(preds_wf, model_names, topk_wf, px, spy_px, spy_ma200,
                              oos_dates_wf, "WALK-FORWARD RESULTS — MA200 + SL=3.4%")

    # ═══════════════════════════════════════
    #  HEAD-TO-HEAD
    # ═══════════════════════════════════════
    print(f"\n{'='*120}")
    print(f"  [5/5] HEAD-TO-HEAD COMPARISON: 80/20 vs Walk-Forward")
    print(f"{'='*120}")

    print(f"\n  {'Model':18s}  {'— 80/20 (MA200+SL) —':^40s}  {'— Walk-Forward (MA200+SL) —':^40s}  {'dSharpe':>8s}")
    print(f"  {'':18s}  {'IC':>7s} {'NDCG':>7s} {'Sharpe':>7s} {'CAGR':>9s} {'MaxDD':>7s} {'Stops':>6s}"
          f"  {'IC':>7s} {'NDCG':>7s} {'Sharpe':>7s} {'CAGR':>9s} {'MaxDD':>7s} {'Stops':>6s}  {'':>8s}")
    print(f"  {'-'*125}")

    for name in model_names:
        r1 = results_8020[name]
        r2 = results_wf[name]
        d = r2['Sharpe'] - r1['Sharpe']
        print(f"  {name:18s}  {r1['IC']:7.4f} {r1['NDCG@10']:7.4f} {r1['Sharpe']:7.3f} "
              f"{r1['CAGR']:+8.2%} {r1['MaxDD']:6.1%} {r1['stops']:6d}"
              f"  {r2['IC']:7.4f} {r2['NDCG@10']:7.4f} {r2['Sharpe']:7.3f} "
              f"{r2['CAGR']:+8.2%} {r2['MaxDD']:6.1%} {r2['stops']:6d}  {d:+7.3f}")

    # Ranking
    print(f"\n  RANKING (by Walk-Forward Sharpe with MA200+SL):")
    ranked = sorted(model_names, key=lambda n: results_wf[n]['Sharpe'], reverse=True)
    for i, name in enumerate(ranked, 1):
        rw = results_wf[name]
        r8 = results_8020[name]
        print(f"  #{i}  {name:18s}  WF={rw['Sharpe']:.3f} (CAGR={rw['CAGR']:+.0%}, MaxDD={rw['MaxDD']:.1%})"
              f"  |  80/20={r8['Sharpe']:.3f} (CAGR={r8['CAGR']:+.0%})")

    print(f"\n{sep}")
    print(f"  DONE")
    print(sep)


if __name__ == '__main__':
    main()
