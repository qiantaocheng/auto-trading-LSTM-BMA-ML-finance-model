#!/usr/bin/env python3
"""
Direct Prediction Test: 80/20 split AND Walk-Forward for all models.
Uses target column directly from parquet — NO yfinance download needed.

Target: Close(d+6)/Close(d+1) - 1  (5-day forward return with T+1 lag)

Models: LambdaRank, ElasticNet, XGBoost-Reg, CatBoost-Reg
Evaluation: IC, NDCG@10, Top-K mean return, compounded portfolio Sharpe/CAGR/MaxDD
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
TOP_K = 10
REBALANCE_DAYS = 5
COST_BPS = 10

INIT_DAYS = 252
STEP_DAYS = 63
HORIZON = 5
SEED = 0


# ─── Ranking metrics ───
def calc_ic(preds, targets, dates):
    """Per-date Spearman rank IC, then mean."""
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
    return np.mean(ics) if ics else 0.0, ics


def calc_ndcg_at_k(preds, targets, dates, k=10):
    """Per-date NDCG@K, then mean."""
    ndcgs = []
    for d in np.unique(dates):
        mask = dates == d
        if np.sum(mask) < k:
            continue
        p, t = preds[mask], targets[mask]
        # predicted top-K
        topk_idx = np.argsort(-p)[:k]
        # ideal top-K
        ideal_idx = np.argsort(-t)[:k]
        # DCG
        dcg = sum(t[topk_idx[i]] / np.log2(i + 2) for i in range(k))
        idcg = sum(t[ideal_idx[i]] / np.log2(i + 2) for i in range(k))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return np.mean(ndcgs) if ndcgs else 0.0


def calc_topk_returns(preds, targets, dates, k=10):
    """Per-date top-K EW mean return, and bottom-K for spread."""
    top_rets, bot_rets = [], []
    for d in np.unique(dates):
        mask = dates == d
        if np.sum(mask) < 2 * k:
            continue
        p, t = preds[mask], targets[mask]
        order = np.argsort(-p)
        top_rets.append(np.mean(t[order[:k]]))
        bot_rets.append(np.mean(t[order[-k:]]))
    return np.array(top_rets), np.array(bot_rets)


def compound_returns(period_rets, periods_per_year=252/5, cost_bps=10):
    """Compound period returns into equity curve with transaction cost."""
    cost_frac = cost_bps / 10_000
    eq = [1.0]
    for r in period_rets:
        eq.append(eq[-1] * (1 + r - cost_frac))
    return np.array(eq)


def calc_metrics(eq_arr, periods_per_year=252/5):
    """Sharpe, CAGR, MaxDD, Calmar from equity curve."""
    if len(eq_arr) < 3:
        return {'Sharpe': 0, 'CAGR': 0, 'MaxDD': 0, 'Calmar': 0, 'Vol': 0}
    period_rets = eq_arr[1:] / eq_arr[:-1] - 1
    n_periods = len(period_rets)
    years = n_periods / periods_per_year
    if years <= 0:
        return {'Sharpe': 0, 'CAGR': 0, 'MaxDD': 0, 'Calmar': 0, 'Vol': 0}
    cagr = (eq_arr[-1] / eq_arr[0]) ** (1.0 / years) - 1
    vol = float(np.std(period_rets, ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(np.mean(period_rets) / np.std(period_rets, ddof=1) * np.sqrt(periods_per_year)) if vol > 0 else 0
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    maxdd = float(dd.min())
    calmar = cagr / abs(maxdd) if maxdd < 0 else 0
    return {'Sharpe': sharpe, 'CAGR': cagr, 'MaxDD': maxdd, 'Calmar': calmar, 'Vol': vol}


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


# ─── Evaluation ───
def evaluate_predictions(preds, targets, dates, label):
    """Full evaluation: IC, NDCG, top-K returns, portfolio metrics."""
    ic_mean, ics = calc_ic(preds, targets, dates)
    ndcg = calc_ndcg_at_k(preds, targets, dates, k=TOP_K)
    top_rets, bot_rets = calc_topk_returns(preds, targets, dates, k=TOP_K)

    # Compound top-K returns (each is a 5-day period return)
    eq = compound_returns(top_rets, periods_per_year=252/REBALANCE_DAYS, cost_bps=COST_BPS)
    m = calc_metrics(eq, periods_per_year=252/REBALANCE_DAYS)

    # Long-short spread
    spread_rets = top_rets - bot_rets
    eq_ls = compound_returns(spread_rets, periods_per_year=252/REBALANCE_DAYS, cost_bps=COST_BPS*2)
    m_ls = calc_metrics(eq_ls, periods_per_year=252/REBALANCE_DAYS)

    return {
        'IC': ic_mean,
        'IC_std': np.std(ics) if ics else 0,
        'NDCG@10': ndcg,
        'Top10_mean_5d': np.mean(top_rets) if len(top_rets) else 0,
        'Bot10_mean_5d': np.mean(bot_rets) if len(bot_rets) else 0,
        'Spread_5d': np.mean(spread_rets) if len(spread_rets) else 0,
        'Top10_Sharpe': m['Sharpe'],
        'Top10_CAGR': m['CAGR'],
        'Top10_MaxDD': m['MaxDD'],
        'Top10_Calmar': m['Calmar'],
        'Top10_Vol': m['Vol'],
        'LS_Sharpe': m_ls['Sharpe'],
        'LS_CAGR': m_ls['CAGR'],
        'eq_top': eq,
        'eq_ls': eq_ls,
        'n_periods': len(top_rets),
        'ics': ics,
    }


def print_results_table(results_dict, model_names, label):
    """Print a formatted comparison table."""
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)

    # Header
    print(f"\n  {'Model':18s}  {'IC':>7s} {'IC_std':>7s} {'NDCG@10':>8s} "
          f"{'Top10_5d':>9s} {'Bot10_5d':>9s} {'Spread':>8s} "
          f"{'Sharpe':>7s} {'CAGR':>9s} {'MaxDD':>7s} {'Calmar':>8s} {'Vol':>6s} "
          f"{'LS_Shrp':>8s}")
    print(f"  {'-'*116}")

    for name in model_names:
        r = results_dict[name]
        print(f"  {name:18s}  {r['IC']:7.4f} {r['IC_std']:7.4f} {r['NDCG@10']:8.4f} "
              f"{r['Top10_mean_5d']:+8.3%} {r['Bot10_mean_5d']:+8.3%} {r['Spread_5d']:+7.3%} "
              f"{r['Top10_Sharpe']:7.3f} {r['Top10_CAGR']:+8.2%} {r['Top10_MaxDD']:6.1%} "
              f"{r['Top10_Calmar']:8.3f} {r['Top10_Vol']:5.1%} "
              f"{r['LS_Sharpe']:8.3f}")


def print_6fold_robustness(results_dict, model_names, label):
    """6-fold sub-period Sharpe robustness."""
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  6-FOLD SHARPE ROBUSTNESS — {label}")
    print(sep)

    for name in model_names:
        r = results_dict[name]
        eq = r['eq_top']
        n = len(eq)
        fold_size = n // 6
        header = f"  {name:18s}"
        fold_sharpes = []
        for fi in range(6):
            s = fi * fold_size
            e = (fi + 1) * fold_size if fi < 5 else n
            if s >= n or e > n:
                header += f"  {'N/A':>7s}"
                continue
            fold_eq = eq[s:e]
            fold_eq = fold_eq / fold_eq[0]
            fm = calc_metrics(fold_eq, periods_per_year=252/REBALANCE_DAYS)
            fold_sharpes.append(fm['Sharpe'])
            header += f"  {fm['Sharpe']:7.3f}"
        if fold_sharpes:
            header += f"  | Mean={np.mean(fold_sharpes):.3f}  Min={np.min(fold_sharpes):.3f}"
        print(header)


# ─── Run 80/20 ───
def run_8020(df, dates_all, model_names):
    """Train on first 80% of dates, test on last 20%."""
    n_dates = len(dates_all)
    split_idx = int(n_dates * 0.8)
    train_dates = dates_all[:split_idx]
    test_dates = dates_all[split_idx:]

    print(f"\n  Train: {len(train_dates)} dates ({train_dates[0].date()} .. {train_dates[-1].date()})")
    print(f"  Test:  {len(test_dates)} dates ({test_dates[0].date()} .. {test_dates[-1].date()})")
    print(f"  Gap:   {HORIZON} dates between train end and test start")

    # Apply gap: remove last HORIZON dates from train
    train_dates_gapped = train_dates[:len(train_dates) - HORIZON]
    print(f"  Train (after gap): {len(train_dates_gapped)} dates "
          f"({train_dates_gapped[0].date()} .. {train_dates_gapped[-1].date()})")

    train_df = df.loc[(train_dates_gapped, slice(None)), :]
    test_df = df.loc[(test_dates, slice(None)), :]

    X_train = train_df[FEATURES].fillna(0.0).to_numpy()
    y_train = train_df['target'].to_numpy()
    dates_train = train_df.index.get_level_values('date').to_numpy()

    X_test = test_df[FEATURES].fillna(0.0).to_numpy()
    y_test = test_df['target'].to_numpy()
    dates_test = test_df.index.get_level_values('date').to_numpy()

    print(f"  Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    results = {}
    for name in model_names:
        trainer = ALL_TRAINERS[name]
        t0 = time.time()
        try:
            preds = trainer(X_train, y_train, dates_train, X_test)
            elapsed = time.time() - t0
            r = evaluate_predictions(preds, y_test, dates_test, name)
            results[name] = r
            print(f"  {name:18s} done ({elapsed:.0f}s) — IC={r['IC']:.4f}, "
                  f"Sharpe={r['Top10_Sharpe']:.3f}, periods={r['n_periods']}")
        except Exception as e:
            print(f"  {name:18s} FAILED: {e}")
            results[name] = {k: 0 for k in ['IC', 'IC_std', 'NDCG@10', 'Top10_mean_5d',
                             'Bot10_mean_5d', 'Spread_5d', 'Top10_Sharpe', 'Top10_CAGR',
                             'Top10_MaxDD', 'Top10_Calmar', 'Top10_Vol', 'LS_Sharpe', 'LS_CAGR',
                             'n_periods']}
            results[name]['eq_top'] = np.array([1.0])
            results[name]['eq_ls'] = np.array([1.0])
            results[name]['ics'] = []

    return results


# ─── Run Walk-Forward ───
def run_walkforward(df, dates_all, model_names):
    """Expanding window walk-forward."""
    n_dates = len(dates_all)
    folds = []
    cursor = INIT_DAYS
    while cursor < n_dates:
        test_end = min(cursor + STEP_DAYS, n_dates)
        train_end_idx = max(0, cursor - HORIZON)
        train_dates_sel = dates_all[:train_end_idx]
        test_dates_sel = dates_all[cursor:test_end]
        if len(train_dates_sel) >= 50 and len(test_dates_sel) > 0:
            folds.append((train_dates_sel, test_dates_sel))
        cursor = test_end

    print(f"\n  {len(folds)} WF folds, {INIT_DAYS}d init, {STEP_DAYS}d step, {HORIZON}d gap")

    # Show first/last fold
    tr0, te0 = folds[0]
    trL, teL = folds[-1]
    print(f"  Fold 1:  train={len(tr0)}d ({tr0[0].date()}..{tr0[-1].date()}) | "
          f"test={len(te0)}d ({te0[0].date()}..{te0[-1].date()})")
    print(f"  Fold {len(folds)}: train={len(trL)}d ({trL[0].date()}..{trL[-1].date()}) | "
          f"test={len(teL)}d ({teL[0].date()}..{teL[-1].date()})")

    model_preds = {name: {'preds': [], 'dates': [], 'targets': []} for name in model_names}

    for fold_num, (train_dates_sel, test_dates_sel) in enumerate(folds, 1):
        train_df = df.loc[(train_dates_sel, slice(None)), :]
        test_df = df.loc[(test_dates_sel, slice(None)), :]

        X_train = train_df[FEATURES].fillna(0.0).to_numpy()
        y_train = train_df['target'].to_numpy()
        dates_train = train_df.index.get_level_values('date').to_numpy()

        X_test = test_df[FEATURES].fillna(0.0).to_numpy()
        y_test = test_df['target'].to_numpy()
        dates_test = test_df.index.get_level_values('date').to_numpy()

        print(f"    [Fold {fold_num}/{len(folds)}] train: {len(train_dates_sel)}d, "
              f"{len(train_df):,}rows | test: {len(test_dates_sel)}d, {len(test_df):,}rows",
              end='', flush=True)

        for name in model_names:
            trainer = ALL_TRAINERS[name]
            t0 = time.time()
            try:
                preds = trainer(X_train, y_train, dates_train, X_test)
                elapsed = time.time() - t0
                model_preds[name]['preds'].append(preds)
                model_preds[name]['dates'].append(dates_test)
                model_preds[name]['targets'].append(y_test)
                print(f"  {name[:6]}={elapsed:.0f}s", end='', flush=True)
            except Exception as e:
                print(f"  {name[:6]}=FAIL", end='', flush=True)
                model_preds[name]['preds'].append(np.zeros(len(X_test)))
                model_preds[name]['dates'].append(dates_test)
                model_preds[name]['targets'].append(y_test)
        print(flush=True)

    # Concatenate and evaluate
    results = {}
    for name in model_names:
        mp = model_preds[name]
        all_preds = np.concatenate(mp['preds'])
        all_dates = np.concatenate(mp['dates'])
        all_targets = np.concatenate(mp['targets'])
        r = evaluate_predictions(all_preds, all_targets, all_dates, name)
        results[name] = r

    return results


def main():
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  DIRECT PREDICTION TEST — 80/20 & Walk-Forward")
    print(f"  Data: polygon_full_features_T5.parquet (NO yfinance download)")
    print(f"  Target: Close(d+6)/Close(d+1) - 1 (T+1 lag, 5-day holding)")
    print(f"  Models: LambdaRank, ElasticNet, XGBoost-Reg, CatBoost-Reg")
    print(f"  Eval: IC, NDCG@10, Top-{TOP_K} EW return, compounded Sharpe/CAGR/MaxDD")
    print(f"  Settings: K={TOP_K}, {REBALANCE_DAYS}d rebal, {COST_BPS}bps cost, gap={HORIZON}d")
    print(sep)

    # Load data
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
    print(f"  {len(df):,} rows, {len(dates_all)} dates ({dates_all[0].date()} .. {dates_all[-1].date()})")
    print(f"  ~{len(df)//len(dates_all)} stocks/date")

    model_names = list(ALL_TRAINERS.keys())

    # ═══════════════════════════════════════
    #  80/20 SPLIT
    # ═══════════════════════════════════════
    print(f"\n{'='*120}")
    print(f"  [2/4] 80/20 CHRONOLOGICAL SPLIT")
    print(f"{'='*120}")

    results_8020 = run_8020(df, dates_all, model_names)
    print_results_table(results_8020, model_names, "80/20 SPLIT RESULTS — Top-10 EW Long Portfolio")
    print_6fold_robustness(results_8020, model_names, "80/20 Split")

    # ═══════════════════════════════════════
    #  WALK-FORWARD
    # ═══════════════════════════════════════
    print(f"\n{'='*120}")
    print(f"  [3/4] WALK-FORWARD (expanding window, {INIT_DAYS}d init, {STEP_DAYS}d step, {HORIZON}d gap)")
    print(f"{'='*120}")

    results_wf = run_walkforward(df, dates_all, model_names)
    print_results_table(results_wf, model_names, "WALK-FORWARD RESULTS — Top-10 EW Long Portfolio")
    print_6fold_robustness(results_wf, model_names, "Walk-Forward")

    # ═══════════════════════════════════════
    #  HEAD-TO-HEAD COMPARISON
    # ═══════════════════════════════════════
    print(f"\n{'='*120}")
    print(f"  [4/4] HEAD-TO-HEAD: 80/20 vs Walk-Forward")
    print(f"{'='*120}")

    print(f"\n  {'Model':18s}  {'— 80/20 —':^35s}  {'— Walk-Forward —':^35s}  {'— Delta —':^15s}")
    print(f"  {'':18s}  {'IC':>7s} {'NDCG':>7s} {'Sharpe':>7s} {'CAGR':>9s} {'MaxDD':>7s}"
          f"  {'IC':>7s} {'NDCG':>7s} {'Sharpe':>7s} {'CAGR':>9s} {'MaxDD':>7s}"
          f"  {'dSharpe':>8s}")
    print(f"  {'-'*115}")

    for name in model_names:
        r1 = results_8020[name]
        r2 = results_wf[name]
        d_sharpe = r2['Top10_Sharpe'] - r1['Top10_Sharpe']
        print(f"  {name:18s}  {r1['IC']:7.4f} {r1['NDCG@10']:7.4f} {r1['Top10_Sharpe']:7.3f} "
              f"{r1['Top10_CAGR']:+8.2%} {r1['Top10_MaxDD']:6.1%}"
              f"  {r2['IC']:7.4f} {r2['NDCG@10']:7.4f} {r2['Top10_Sharpe']:7.3f} "
              f"{r2['Top10_CAGR']:+8.2%} {r2['Top10_MaxDD']:6.1%}"
              f"  {d_sharpe:+7.3f}")

    # Long-short comparison
    print(f"\n  Long-Short Spread:")
    print(f"  {'Model':18s}  {'— 80/20 —':^25s}  {'— Walk-Forward —':^25s}")
    print(f"  {'':18s}  {'Spread/5d':>10s} {'LS_Sharpe':>10s}  {'Spread/5d':>10s} {'LS_Sharpe':>10s}")
    print(f"  {'-'*80}")

    for name in model_names:
        r1 = results_8020[name]
        r2 = results_wf[name]
        print(f"  {name:18s}  {r1['Spread_5d']:+9.3%} {r1['LS_Sharpe']:10.3f}"
              f"  {r2['Spread_5d']:+9.3%} {r2['LS_Sharpe']:10.3f}")

    # Winner summary
    print(f"\n  {'='*80}")
    print(f"  RANKING SUMMARY (by Walk-Forward Sharpe)")
    print(f"  {'='*80}")
    ranked = sorted(model_names, key=lambda n: results_wf[n]['Top10_Sharpe'], reverse=True)
    for i, name in enumerate(ranked, 1):
        rw = results_wf[name]
        r8 = results_8020[name]
        print(f"  #{i}  {name:18s}  WF_Sharpe={rw['Top10_Sharpe']:.3f}  WF_IC={rw['IC']:.4f}"
              f"  |  80/20_Sharpe={r8['Top10_Sharpe']:.3f}  80/20_IC={r8['IC']:.4f}")

    print(f"\n{'='*120}")
    print(f"  DONE")
    print(f"{'='*120}")


if __name__ == '__main__':
    main()
