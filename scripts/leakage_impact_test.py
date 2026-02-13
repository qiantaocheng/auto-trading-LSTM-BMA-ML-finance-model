#!/usr/bin/env python3
"""
Leakage Impact Test: How much do the results change when we fix:
  1. The 1-day boundary overlap in WF gap (HORIZON=5 → 6)
  2. Overlapping target returns in training (every date → every 6th date)

Tests LambdaRank (the main model) with 3 configurations:
  A. CURRENT:       HORIZON=5 gap, all training dates (original, has 1-day leak)
  B. FIXED_GAP:     HORIZON=6 gap, all training dates (clean boundary)
  C. NON_OVERLAP:   HORIZON=6 gap, every 6th training date (independent labels)

Target formula: Close(d+6)/Close(d+1) - 1  (T+1 execution lag, 5-day holding)
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


# ─── LambdaRank trainer ───
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


def build_wf_folds(dates_all, horizon, non_overlap_step=None):
    """Build walk-forward folds.

    Args:
        dates_all: sorted array of all dates
        horizon: gap size (number of dates to skip between train end and test start)
        non_overlap_step: if set, subsample training dates every N dates (for non-overlapping targets)

    Returns:
        list of (train_dates, test_dates) tuples
    """
    n_dates = len(dates_all)
    folds = []
    cursor = INIT_DAYS
    while cursor < n_dates:
        test_end = min(cursor + STEP_DAYS, n_dates)
        train_end_idx = max(0, cursor - horizon)
        train_dates_sel = dates_all[:train_end_idx]
        test_dates_sel = dates_all[cursor:test_end]

        if non_overlap_step is not None and non_overlap_step > 1:
            # Subsample training dates for non-overlapping targets
            # Keep every Nth date to avoid overlapping return windows
            train_dates_sel = train_dates_sel[::non_overlap_step]

        if len(train_dates_sel) >= 50 and len(test_dates_sel) > 0:
            folds.append((train_dates_sel, test_dates_sel))
        cursor = test_end
    return folds


def run_variant(variant_name, df, dates_all, horizon, non_overlap_step, model_names):
    """Run walk-forward for specified models with given settings."""
    folds = build_wf_folds(dates_all, horizon, non_overlap_step)
    print(f"\n  [{variant_name}] horizon={horizon}, non_overlap_step={non_overlap_step}, {len(folds)} folds")

    if len(folds) == 0:
        print(f"    No folds! Skipping.")
        return {}

    # Show first fold details
    train_d, test_d = folds[0]
    print(f"    Fold 1: train={len(train_d)}d ({train_d[0].date()}..{train_d[-1].date()}) | "
          f"test={len(test_d)}d ({test_d[0].date()}..{test_d[-1].date()})")

    model_preds = {name: {'preds': [], 'dates': [], 'tickers': []} for name in model_names}

    for fold_num, (train_dates_sel, test_dates_sel) in enumerate(folds, 1):
        train_df = df.loc[df.index.get_level_values('date').isin(train_dates_sel)]
        test_df = df.loc[(test_dates_sel, slice(None)), :]

        X_train = train_df[FEATURES].fillna(0.0).to_numpy()
        y_train = train_df['target'].to_numpy()
        dates_train = train_df.index.get_level_values('date').to_numpy()

        X_test = test_df[FEATURES].fillna(0.0).to_numpy()
        test_dates = test_df.index.get_level_values('date').to_numpy()
        test_tickers = test_df.index.get_level_values('ticker').to_numpy()

        print(f"    [Fold {fold_num}/{len(folds)}] train: {len(train_dates_sel)}d, "
              f"{len(train_df)}rows | test: {len(test_dates_sel)}d, {len(test_df)}rows", end='', flush=True)

        for model_name in model_names:
            trainer = ALL_TRAINERS[model_name]
            t0 = time.time()
            try:
                preds = trainer(X_train, y_train, dates_train, X_test)
                elapsed = time.time() - t0
                model_preds[model_name]['preds'].append(preds)
                model_preds[model_name]['dates'].append(test_dates)
                model_preds[model_name]['tickers'].append(test_tickers)
                print(f"  {model_name[:6]}={elapsed:.0f}s", end='', flush=True)
            except Exception as e:
                print(f"  {model_name[:6]}=FAIL({e})", end='', flush=True)
                model_preds[model_name]['preds'].append(np.zeros(len(X_test)))
                model_preds[model_name]['dates'].append(test_dates)
                model_preds[model_name]['tickers'].append(test_tickers)
        print(flush=True)

    # Concatenate
    for name in model_names:
        mp = model_preds[name]
        mp['preds_cat'] = np.concatenate(mp['preds'])
        mp['dates_cat'] = np.concatenate(mp['dates'])
        mp['tickers_cat'] = np.concatenate(mp['tickers'])

    return model_preds


def main():
    sep = "=" * 110

    print(f"\n{sep}")
    print("  LEAKAGE IMPACT TEST")
    print(f"  Target: Close(d+6)/Close(d+1) - 1  (T+1 lag, 5-day holding)")
    print(f"  Models: LambdaRank, ElasticNet, XGBoost-Reg, CatBoost-Reg")
    print(f"  Settings: K={TOP_K}, EW, {REBALANCE_DAYS}d rebal, MA200 daily mult, SL={STOP_LOSS_PCT*100:.1f}%, {COST_BPS}bps")
    print(sep)

    # Variants to test
    VARIANTS = {
        'A_CURRENT':     {'horizon': 5, 'non_overlap_step': None,
                          'desc': 'HORIZON=5 gap, all dates (original, 1-day boundary leak)'},
        'B_FIXED_GAP':   {'horizon': 6, 'non_overlap_step': None,
                          'desc': 'HORIZON=6 gap, all dates (clean boundary)'},
        'C_NON_OVERLAP': {'horizon': 6, 'non_overlap_step': 6,
                          'desc': 'HORIZON=6 gap, every 6th date (independent labels, ~1/6 training data)'},
    }

    model_names = list(ALL_TRAINERS.keys())

    # Load data
    data_file = Path('data/factor_exports/polygon_full_features_T5.parquet')
    print(f"\n[1/3] Loading data: {data_file}")
    df = pd.read_parquet(data_file)
    if isinstance(df.index, pd.MultiIndex) and {'date', 'ticker'}.issubset(df.index.names):
        df = df.sort_index()
    elif {'date', 'ticker'}.issubset(df.columns):
        df = df.set_index(['date', 'ticker']).sort_index()
    if 'target' in df.columns:
        df['target'] = df['target'].clip(-0.55, 0.55)

    dates_all = df.index.get_level_values('date').unique().sort_values()
    print(f"  {len(df)} rows, {len(dates_all)} dates ({dates_all[0].date()} .. {dates_all[-1].date()})")

    # Run all variants
    print(f"\n[2/3] Running {len(VARIANTS)} variants x {len(model_names)} models...")

    all_results = {}  # variant -> model -> model_preds
    for var_name, var_cfg in VARIANTS.items():
        print(f"\n{'─'*80}")
        print(f"  VARIANT {var_name}: {var_cfg['desc']}")
        print(f"{'─'*80}")
        model_preds = run_variant(
            var_name, df, dates_all,
            horizon=var_cfg['horizon'],
            non_overlap_step=var_cfg['non_overlap_step'],
            model_names=model_names,
        )
        all_results[var_name] = model_preds

    # Build top-K and download prices
    print(f"\n[3/3] Building top-K and downloading prices...")

    # Collect all unique OOS dates and tickers across all variants
    all_oos_dates = set()
    all_tickers_set = set()

    variant_topk = {}  # variant -> model -> topk
    for var_name, model_preds in all_results.items():
        var_topk = {}
        for name in model_names:
            mp = model_preds[name]
            unique_dates = np.sort(np.unique(mp['dates_cat']))
            topk = {}
            for d in unique_dates:
                all_oos_dates.add(d)
                mask = mp['dates_cat'] == d
                dp = mp['preds_cat'][mask]
                dt_tickers = mp['tickers_cat'][mask]
                if len(dp) < 20:
                    continue
                order = np.argsort(-dp)[:20]
                topk[d] = [(dt_tickers[o], float(dp[o])) for o in order]
                for o in order:
                    all_tickers_set.add(dt_tickers[o])
            var_topk[name] = topk
        variant_topk[var_name] = var_topk

    all_tickers_set.add('SPY')
    tickers_list = sorted(all_tickers_set)
    unique_oos_dates = np.sort(list(all_oos_dates))
    print(f"  OOS: {pd.Timestamp(unique_oos_dates[0]).date()} .. {pd.Timestamp(unique_oos_dates[-1]).date()} ({len(unique_oos_dates)} days)")
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

    # Run portfolio simulations for all variants
    print(f"\n{'='*110}")
    print(f"  PORTFOLIO SIMULATION RESULTS")
    print(f"{'='*110}")

    all_metrics = {}  # (variant, model) -> metrics
    for var_name in VARIANTS:
        print(f"\n  --- {var_name}: {VARIANTS[var_name]['desc']} ---")
        for name in model_names:
            topk = variant_topk[var_name][name]
            eq, n_stops = simulate_portfolio(oos_trading_dates, topk, px, spy_px, spy_ma200)
            m = calc_metrics(eq)
            m['Stops'] = n_stops
            all_metrics[(var_name, name)] = {'metrics': m, 'eq': eq}
            print(f"    {name:18s}  Sharpe={m['Sharpe']:7.3f}  CAGR={m['CAGR']:+9.2%}  "
                  f"MaxDD={m['MaxDD']:7.1%}  Calmar={m['Calmar']:8.3f}  Vol={m['Vol']:5.1%}  Stops={m['Stops']}")

    # ═══════════════════════════════════════
    #  COMPARISON TABLE
    # ═══════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  SIDE-BY-SIDE COMPARISON — Does leakage fix change results?")
    print(f"{'='*110}")

    for name in model_names:
        print(f"\n  {name}:")
        print(f"    {'Variant':20s}  {'Sharpe':>7s}  {'CAGR':>10s}  {'MaxDD':>7s}  {'Calmar':>8s}  {'Vol':>6s}")
        print(f"    {'-'*65}")
        base_sharpe = None
        for var_name in VARIANTS:
            m = all_metrics[(var_name, name)]['metrics']
            delta = ''
            if base_sharpe is None:
                base_sharpe = m['Sharpe']
            else:
                d = m['Sharpe'] - base_sharpe
                delta = f"  (Sharpe delta={d:+.3f})"
            print(f"    {var_name:20s}  {m['Sharpe']:7.3f}  {m['CAGR']:+9.2%}  {m['MaxDD']:6.1%}  "
                  f"{m['Calmar']:8.3f}  {m['Vol']:5.1%}{delta}")

    # 6-fold robustness for each variant x model
    print(f"\n{'='*110}")
    print(f"  6-FOLD SHARPE ROBUSTNESS")
    print(f"{'='*110}")

    n_oos = len(oos_trading_dates)
    fold_size = n_oos // 6
    fold_ranges = []
    for fi in range(6):
        s = fi * fold_size
        e = (fi + 1) * fold_size if fi < 5 else n_oos
        fold_ranges.append((s, e))

    for name in model_names:
        print(f"\n  {name}:")
        header = f"    {'Variant':20s}"
        for fi in range(6):
            header += f"  {'F'+str(fi+1):>7s}"
        header += f"  {'Mean':>7s}  {'Min':>7s}"
        print(header)
        print(f"    {'-'*90}")

        for var_name in VARIANTS:
            eq_arr = all_metrics[(var_name, name)]['eq']
            line = f"    {var_name:20s}"
            fold_sharpes = []
            for s, e in fold_ranges:
                if e > len(eq_arr) or s >= len(eq_arr):
                    line += f"  {'N/A':>7s}"
                    continue
                fold_eq = eq_arr[s:e]
                fold_eq = fold_eq / fold_eq[0]
                fm = calc_metrics(fold_eq)
                fold_sharpes.append(fm['Sharpe'])
                line += f"  {fm['Sharpe']:7.3f}"
            if fold_sharpes:
                line += f"  {np.mean(fold_sharpes):7.3f}  {np.min(fold_sharpes):7.3f}"
            print(line)

    # Top-K overlap: how similar are picks between variants?
    print(f"\n{'='*110}")
    print(f"  TOP-10 OVERLAP BETWEEN VARIANTS (for LambdaRank)")
    print(f"{'='*110}")

    lr_name = 'LambdaRank'
    var_names = list(VARIANTS.keys())
    print(f"  {'':20s}", end='')
    for v2 in var_names:
        print(f"  {v2[:12]:>12s}", end='')
    print()
    print(f"  {'-'*(20 + 14*len(var_names))}")

    for v1 in var_names:
        line = f"  {v1:20s}"
        for v2 in var_names:
            if v1 == v2:
                line += f"  {'100%':>12s}"
            else:
                topk1 = variant_topk[v1][lr_name]
                topk2 = variant_topk[v2][lr_name]
                common_dates = set(topk1.keys()) & set(topk2.keys())
                overlaps = []
                for d in common_dates:
                    t1 = set(tk for tk, _ in topk1[d][:TOP_K])
                    t2 = set(tk for tk, _ in topk2[d][:TOP_K])
                    overlaps.append(len(t1 & t2) / TOP_K)
                avg = np.mean(overlaps) if overlaps else 0
                line += f"  {avg:11.0%}"
        print(line)

    # Summary
    print(f"\n{'='*110}")
    print(f"  VERDICT")
    print(f"{'='*110}")

    for name in model_names:
        m_cur = all_metrics[('A_CURRENT', name)]['metrics']
        m_fix = all_metrics[('B_FIXED_GAP', name)]['metrics']
        m_nov = all_metrics[('C_NON_OVERLAP', name)]['metrics']
        gap_delta = m_fix['Sharpe'] - m_cur['Sharpe']
        nov_delta = m_nov['Sharpe'] - m_cur['Sharpe']
        print(f"  {name:18s}: Current Sharpe={m_cur['Sharpe']:.3f} | "
              f"FixGap delta={gap_delta:+.3f} | NonOverlap delta={nov_delta:+.3f}")

    print(f"\n{'='*110}")
    print(f"  DONE")
    print(f"{'='*110}")


if __name__ == '__main__':
    main()
