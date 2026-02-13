#!/usr/bin/env python3
"""
Walk-Forward Model Comparison: LambdaRank vs ElasticNet vs XGBoost vs CatBoost
Same features, same target, same WF methodology — NO time leakage.

Models:
  1. LambdaRank (LightGBM lambdarank) — existing cached preds
  2. ElasticNet (sklearn) — regression, rank by predicted return
  3. XGBoost (rank:pairwise) — learning-to-rank
  4. CatBoost (YetiRank) — learning-to-rank

Settings: K=10, equal weight, 5d rebalance, MA200 daily multiplier, SL=3.4%, 10bps cost
Walk-forward: expanding window, 252d init, 63d step, 5d gap
"""

import sys, warnings, pickle, time
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

# LambdaRank params (for reference / LightGBM lambdarank)
LR_PARAMS = {
    'learning_rate': 0.04, 'num_leaves': 11, 'max_depth': 3,
    'min_data_in_leaf': 350, 'lambda_l2': 120, 'feature_fraction': 1.0,
    'bagging_fraction': 0.70, 'bagging_freq': 1, 'min_gain_to_split': 0.30,
    'lambdarank_truncation_level': 25, 'sigmoid': 1.1, 'label_gain_power': 2.1,
}

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


# ─── Portfolio simulation ───
def simulate_portfolio(oos_trading_dates, topk_by_date, px, spy_px, spy_ma200):
    pred_dates_ts = [pd.Timestamp(d) for d in sorted(topk_by_date.keys())]

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
                            dd = (p_now / entry_prices[tk]) - 1.0
                            if dd <= -STOP_LOSS_PCT:
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
                for key in topk_by_date:
                    if pd.Timestamp(key) == best_pred_dt:
                        new_h = [tk for tk, sc in topk_by_date[key][:TOP_K] if tk in px.columns]
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
                        break
        eq.append(cap)
    return np.array(eq), n_stops


# ═══════════════════════════════════════════════════════
# MODEL TRAINERS — each returns predictions for test set
# ═══════════════════════════════════════════════════════

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
    return [int(np.sum(dates == d)) for d in np.unique(dates)]


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


def train_xgboost_rank(X_train, y_train, dates_train, X_test):
    import xgboost as xgb

    # Build group sizes for ranking
    unique_dates = np.unique(dates_train)
    groups = [int(np.sum(dates_train == d)) for d in unique_dates]

    # Use quantile labels like LambdaRank for fair comparison
    labels = build_quantile_labels(y_train, dates_train, N_QUANTILES)

    dtrain = xgb.DMatrix(X_train, label=labels)
    dtrain.set_group(groups)
    dtest = xgb.DMatrix(X_test)

    params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'ndcg@10',
        'eta': 0.08,
        'max_depth': 3,
        'min_child_weight': 350,
        'subsample': 0.70,
        'colsample_bytree': 1.0,
        'lambda': 120,
        'gamma': 0.30,
        'seed': SEED,
        'verbosity': 0,
        'nthread': -1,
    }
    model = xgb.train(params, dtrain, num_boost_round=250)
    return model.predict(dtest)


def train_catboost_rank(X_train, y_train, dates_train, X_test):
    from catboost import CatBoost, Pool

    # Build group IDs for ranking — aggressive subsample to speed up YetiRank
    unique_dates = np.unique(dates_train)
    # Use last 40% of training dates
    cutoff = int(len(unique_dates) * 0.6)
    recent_dates = unique_dates[cutoff:]
    recent_mask = np.isin(dates_train, recent_dates)
    X_sub = X_train[recent_mask]
    y_sub = y_train[recent_mask]
    dates_sub = dates_train[recent_mask]

    # Further subsample: max 200 stocks per date to reduce pairwise explosion
    rng = np.random.RandomState(SEED)
    keep_idx = []
    for d in np.unique(dates_sub):
        d_mask = np.where(dates_sub == d)[0]
        if len(d_mask) > 200:
            keep_idx.append(rng.choice(d_mask, 200, replace=False))
        else:
            keep_idx.append(d_mask)
    keep_idx = np.concatenate(keep_idx)
    keep_idx.sort()
    X_sub = X_sub[keep_idx]
    y_sub = y_sub[keep_idx]
    dates_sub = dates_sub[keep_idx]

    sub_unique = np.unique(dates_sub)
    date_to_group = {d: i for i, d in enumerate(sub_unique)}
    group_ids = np.array([date_to_group[d] for d in dates_sub])
    labels = build_quantile_labels(y_sub, dates_sub, N_QUANTILES)

    train_pool = Pool(X_sub, label=labels, group_id=group_ids)
    test_pool = Pool(X_test)

    params = {
        'loss_function': 'YetiRank',
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 3,
        'l2_leaf_reg': 120,
        'random_seed': SEED,
        'verbose': 0,
        'thread_count': -1,
    }
    model = CatBoost(params)
    model.fit(train_pool)
    return model.predict(test_pool)


# Also test regression variants
def train_xgboost_reg(X_train, y_train, dates_train, X_test):
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.04,
        'max_depth': 3,
        'min_child_weight': 350,
        'subsample': 0.70,
        'colsample_bytree': 1.0,
        'lambda': 120,
        'gamma': 0.30,
        'seed': SEED,
        'verbosity': 0,
    }
    model = xgb.train(params, dtrain, num_boost_round=500)
    return model.predict(dtest)


def train_catboost_reg(X_train, y_train, dates_train, X_test):
    from catboost import CatBoost, Pool

    train_pool = Pool(X_train, label=y_train)
    test_pool = Pool(X_test)

    params = {
        'loss_function': 'RMSE',
        'iterations': 500,
        'learning_rate': 0.04,
        'depth': 3,
        'l2_leaf_reg': 120,
        'random_seed': SEED,
        'verbose': 0,
        'thread_count': -1,
    }
    model = CatBoost(params)
    model.fit(train_pool)
    return model.predict(test_pool)


MODEL_TRAINERS = {
    'LambdaRank': train_lambdarank,
    'ElasticNet': train_elasticnet,
    'XGBoost-Rank': train_xgboost_rank,
    'XGBoost-Reg': train_xgboost_reg,
    'CatBoost-Rank': train_catboost_rank,
    'CatBoost-Reg': train_catboost_reg,
}


def main():
    sep = "=" * 110

    print(f"\n{sep}")
    print("  Walk-Forward MODEL COMPARISON")
    print(f"  Models: {', '.join(MODEL_TRAINERS.keys())}")
    print(f"  Settings: K={TOP_K}, EW, {REBALANCE_DAYS}d rebal, MA200 daily mult, SL={STOP_LOSS_PCT*100:.1f}%, {COST_BPS}bps")
    print(f"  WF: expanding window, {INIT_DAYS}d init, {STEP_DAYS}d step, {HORIZON}d gap — NO leakage")
    print(sep)

    # ─── 1. Load data ───
    data_file = Path('data/factor_exports/polygon_full_features_T5.parquet')
    print(f"\n[1/4] Loading data: {data_file}")
    df = pd.read_parquet(data_file)
    if isinstance(df.index, pd.MultiIndex) and {'date', 'ticker'}.issubset(df.index.names):
        df = df.sort_index()
    elif {'date', 'ticker'}.issubset(df.columns):
        df = df.set_index(['date', 'ticker']).sort_index()
    if 'target' in df.columns:
        df['target'] = df['target'].clip(-0.55, 0.55)

    dates_all = df.index.get_level_values('date').unique().sort_values()
    print(f"  {len(df)} rows, {len(dates_all)} dates ({dates_all[0].date()} .. {dates_all[-1].date()})")

    # ─── 2. Walk-forward for all models ───
    print(f"\n[2/4] Walk-forward training ({len(MODEL_TRAINERS)} models)...")

    n_dates = len(dates_all)
    # Pre-compute fold boundaries
    folds = []
    cursor = INIT_DAYS
    while cursor < n_dates:
        test_end = min(cursor + STEP_DAYS, n_dates)
        train_end_idx = max(0, cursor - HORIZON)
        train_dates_sel = dates_all[:train_end_idx]
        test_dates_sel = dates_all[cursor:test_end]
        if len(train_dates_sel) >= 100 and len(test_dates_sel) > 0:
            folds.append((train_dates_sel, test_dates_sel))
        cursor = test_end

    print(f"  {len(folds)} WF folds")

    # Store predictions per model
    model_preds = {name: {'preds': [], 'dates': [], 'tickers': []} for name in MODEL_TRAINERS}

    for fold_num, (train_dates_sel, test_dates_sel) in enumerate(folds, 1):
        train_df = df.loc[(train_dates_sel, slice(None)), :]
        test_df = df.loc[(test_dates_sel, slice(None)), :]

        X_train = train_df[FEATURES].fillna(0.0).to_numpy()
        y_train = train_df['target'].to_numpy()
        dates_train = train_df.index.get_level_values('date').to_numpy()

        X_test = test_df[FEATURES].fillna(0.0).to_numpy()
        test_dates = test_df.index.get_level_values('date').to_numpy()
        test_tickers = test_df.index.get_level_values('ticker').to_numpy()

        print(f"  [Fold {fold_num}/{len(folds)}] train: {train_dates_sel[0].date()}..{train_dates_sel[-1].date()} "
              f"({len(train_dates_sel)}d) | test: {test_dates_sel[0].date()}..{test_dates_sel[-1].date()} "
              f"({len(test_dates_sel)}d, {len(test_df)}rows)", end='')

        for model_name, trainer in MODEL_TRAINERS.items():
            t0 = time.time()
            try:
                preds = trainer(X_train, y_train, dates_train, X_test)
                elapsed = time.time() - t0
                model_preds[model_name]['preds'].append(preds)
                model_preds[model_name]['dates'].append(test_dates)
                model_preds[model_name]['tickers'].append(test_tickers)
                print(f"  {model_name[:6]}={elapsed:.0f}s", end='')
            except Exception as e:
                print(f"  {model_name[:6]}=FAIL({e})", end='')
                # Fill with zeros
                model_preds[model_name]['preds'].append(np.zeros(len(X_test)))
                model_preds[model_name]['dates'].append(test_dates)
                model_preds[model_name]['tickers'].append(test_tickers)
        print()

    # Concatenate predictions
    for name in MODEL_TRAINERS:
        mp = model_preds[name]
        mp['preds_cat'] = np.concatenate(mp['preds'])
        mp['dates_cat'] = np.concatenate(mp['dates'])
        mp['tickers_cat'] = np.concatenate(mp['tickers'])
        print(f"  {name}: {len(mp['preds_cat'])} OOS predictions")

    # ─── 3. Build top-K and download prices ───
    print(f"\n[3/4] Building top-K and downloading prices...")

    # Use first model's dates as reference (all identical)
    ref = model_preds[list(MODEL_TRAINERS.keys())[0]]
    unique_oos_dates = np.sort(np.unique(ref['dates_cat']))
    print(f"  OOS: {pd.Timestamp(unique_oos_dates[0]).date()} .. {pd.Timestamp(unique_oos_dates[-1]).date()} ({len(unique_oos_dates)} days)")

    # Build topk_by_date for each model
    model_topk = {}
    all_tickers_set = set()
    for name in MODEL_TRAINERS:
        mp = model_preds[name]
        topk = {}
        for d in unique_oos_dates:
            mask = mp['dates_cat'] == d
            dp = mp['preds_cat'][mask]
            dt_tickers = mp['tickers_cat'][mask]
            if len(dp) < 20:
                continue
            order = np.argsort(-dp)[:20]
            topk[d] = [(dt_tickers[o], float(dp[o])) for o in order]
            for o in order:
                all_tickers_set.add(dt_tickers[o])
        model_topk[name] = topk

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

    # ─── 4. Run portfolio simulations ───
    print(f"\n[4/4] Running portfolio simulations...")

    results = {}
    for name in MODEL_TRAINERS:
        eq, n_stops = simulate_portfolio(oos_trading_dates, model_topk[name], px, spy_px, spy_ma200)
        m = calc_metrics(eq)
        m['Stops'] = n_stops
        results[name] = {'eq': eq, 'metrics': m}
        print(f"  {name:18s}  Sharpe={m['Sharpe']:6.3f}  CAGR={m['CAGR']:+8.2%}  "
              f"MaxDD={m['MaxDD']:7.1%}  Calmar={m['Calmar']:7.3f}  Stops={n_stops}")

    # ═══════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════
    print(f"\n{sep}")
    print(f"  FULL PERIOD RESULTS — K={TOP_K}, MA200 daily mult, SL={STOP_LOSS_PCT*100:.1f}%")
    print(sep)
    print(f"  {'Model':18s}  {'Sharpe':>7s}  {'CAGR':>10s}  {'MaxDD':>7s}  {'Calmar':>8s}  {'Vol':>6s}  {'Stops':>5s}")
    print(f"  {'-'*72}")
    for name, res in results.items():
        m = res['metrics']
        print(f"  {name:18s}  {m['Sharpe']:7.3f}  {m['CAGR']:+9.2%}  {m['MaxDD']:6.1%}  "
              f"{m['Calmar']:8.3f}  {m['Vol']:5.1%}  {m['Stops']:5d}")

    # Yearly Sharpe
    years = sorted(set(d.year for d in oos_trading_dates))
    print(f"\n{sep}")
    print(f"  YEARLY SHARPE")
    print(sep)
    header = f"  {'Model':18s}"
    for yr in years:
        header += f"  {yr:>8d}"
    print(header)
    print(f"  {'-'*(18 + 10*len(years))}")
    for name, res in results.items():
        eq_arr = res['eq']
        line = f"  {name:18s}"
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

    # Yearly MaxDD
    print(f"\n{sep}")
    print(f"  YEARLY MaxDD")
    print(sep)
    print(header)
    print(f"  {'-'*(18 + 10*len(years))}")
    for name, res in results.items():
        eq_arr = res['eq']
        line = f"  {name:18s}"
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

    # 6-fold WF robustness
    print(f"\n{sep}")
    print(f"  6-FOLD WALKFORWARD ROBUSTNESS")
    print(sep)
    n_folds = 6
    n_dates_oos = len(oos_trading_dates)
    fold_size = n_dates_oos // n_folds
    wf_folds = []
    for f in range(n_folds):
        f_start = f * fold_size
        f_end = n_dates_oos if f == n_folds - 1 else (f + 1) * fold_size
        wf_folds.append((f_start, f_end))
        d_s = oos_trading_dates[f_start]
        d_e = oos_trading_dates[min(f_end, n_dates_oos) - 1]
        print(f"  Fold {f+1}: {pd.Timestamp(d_s).strftime('%Y-%m-%d')} .. "
              f"{pd.Timestamp(d_e).strftime('%Y-%m-%d')} ({f_end - f_start} days)")

    print(f"\n  Sharpe by Fold:")
    header_f = f"  {'Model':18s}"
    for f in range(n_folds):
        header_f += f"  {'F'+str(f+1):>8s}"
    header_f += f"  {'Mean':>8s}  {'StdDev':>8s}  {'Min':>8s}"
    print(header_f)
    print(f"  {'-'*(18 + 10*(n_folds+3))}")

    for name, res in results.items():
        eq_arr = res['eq']
        fold_sharpes = []
        line = f"  {name:18s}"
        for f_start, f_end in wf_folds:
            fold_eq = eq_arr[f_start:f_end]
            fold_eq = fold_eq / fold_eq[0]
            fm = calc_metrics(fold_eq)
            fold_sharpes.append(fm['Sharpe'])
            line += f"  {fm['Sharpe']:8.3f}"
        mean_s = np.mean(fold_sharpes)
        std_s = np.std(fold_sharpes, ddof=1) if len(fold_sharpes) > 1 else 0
        min_s = np.min(fold_sharpes)
        line += f"  {mean_s:8.3f}  {std_s:8.3f}  {min_s:8.3f}"
        results[name]['fold_sharpes'] = fold_sharpes
        print(line)

    # MaxDD by fold
    print(f"\n  MaxDD by Fold:")
    header_f2 = f"  {'Model':18s}"
    for f in range(n_folds):
        header_f2 += f"  {'F'+str(f+1):>8s}"
    header_f2 += f"  {'Worst':>8s}"
    print(header_f2)
    print(f"  {'-'*(18 + 10*(n_folds+1))}")

    for name, res in results.items():
        eq_arr = res['eq']
        line = f"  {name:18s}"
        worst_mdd = 0
        for f_start, f_end in wf_folds:
            fold_eq = eq_arr[f_start:f_end]
            peak = np.maximum.accumulate(fold_eq)
            dd = (fold_eq - peak) / peak
            mdd = float(dd.min())
            if mdd < worst_mdd:
                worst_mdd = mdd
            line += f"  {mdd:7.1%}"
        line += f"  {worst_mdd:7.1%}"
        print(line)

    # Fold wins vs LambdaRank
    ref_name = 'LambdaRank'
    if ref_name in results:
        ref_sharpes = results[ref_name]['fold_sharpes']
        print(f"\n  Folds beating {ref_name}:")
        for name in MODEL_TRAINERS:
            if name == ref_name:
                continue
            wins = sum(1 for a, b in zip(results[name]['fold_sharpes'], ref_sharpes) if a > b)
            print(f"    {name:18s}: {wins}/{n_folds} folds")

    # Ranking correlation between models
    print(f"\n{sep}")
    print(f"  TOP-10 OVERLAP BETWEEN MODELS (avg % shared tickers)")
    print(sep)
    model_names = list(MODEL_TRAINERS.keys())
    print(f"  {'':18s}", end='')
    for n2 in model_names:
        print(f"  {n2[:8]:>8s}", end='')
    print()
    print(f"  {'-'*(18 + 10*len(model_names))}")

    for n1 in model_names:
        line = f"  {n1:18s}"
        for n2 in model_names:
            if n1 == n2:
                line += f"  {'100%':>8s}"
            else:
                overlaps = []
                for d in unique_oos_dates:
                    if d in model_topk[n1] and d in model_topk[n2]:
                        t1 = set(tk for tk, _ in model_topk[n1][d][:TOP_K])
                        t2 = set(tk for tk, _ in model_topk[n2][d][:TOP_K])
                        overlaps.append(len(t1 & t2) / TOP_K)
                avg_overlap = np.mean(overlaps) if overlaps else 0
                line += f"  {avg_overlap:7.0%}"
        print(line)

    # Summary
    print(f"\n{sep}")
    print(f"  SUMMARY")
    print(sep)
    best_sharpe = max(results, key=lambda n: results[n]['metrics']['Sharpe'])
    best_calmar = max(results, key=lambda n: results[n]['metrics']['Calmar'])
    best_wf = max(results, key=lambda n: np.mean(results[n].get('fold_sharpes', [0])))
    lowest_mdd = min(results, key=lambda n: abs(results[n]['metrics']['MaxDD']))

    bm = results[best_sharpe]['metrics']
    print(f"  Best Sharpe:  {best_sharpe:18s}  {bm['Sharpe']:.3f}")
    bm = results[best_calmar]['metrics']
    print(f"  Best Calmar:  {best_calmar:18s}  {bm['Calmar']:.3f}")
    print(f"  Best WF Mean: {best_wf:18s}  {np.mean(results[best_wf]['fold_sharpes']):.3f}")
    bm = results[lowest_mdd]['metrics']
    print(f"  Lowest MaxDD: {lowest_mdd:18s}  {bm['MaxDD']:.1%}")

    print(f"\n{sep}")
    print(f"  DONE")
    print(sep)


if __name__ == '__main__':
    main()
