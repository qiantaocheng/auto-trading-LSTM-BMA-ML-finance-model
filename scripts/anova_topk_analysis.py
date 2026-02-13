#!/usr/bin/env python3
"""
ANOVA Analysis of Top-10 Portfolio Returns by Model and Time Period.

Analyzes:
1. Within-top-10 variance: dispersion among the 10 stocks per rebalance date
2. Between-period variance: how top-10 mean return varies across time
3. One-way ANOVA: are model mean returns significantly different?
4. Two-way ANOVA: model x time-period interaction
5. Pairwise model comparisons (Tukey HSD)

Uses walk-forward predictions, parquet data only.
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
INIT_DAYS = 252
STEP_DAYS = 63
HORIZON = 5
SEED = 0


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


def main():
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  ANOVA ANALYSIS: Top-10 Returns by Model and Time Period")
    print(f"  Walk-Forward, parquet data only")
    print(sep)

    # Load
    data_file = Path('data/factor_exports/polygon_full_features_T5.parquet')
    print(f"\n[1/3] Loading data: {data_file}")
    df = pd.read_parquet(data_file)
    if isinstance(df.index, pd.MultiIndex) and {'date', 'ticker'}.issubset(df.index.names):
        df = df.sort_index()
    elif {'date', 'ticker'}.issubset(df.columns):
        df = df.set_index(['date', 'ticker']).sort_index()
    df['target'] = df['target'].clip(-0.55, 0.55)
    dates_all = df.index.get_level_values('date').unique().sort_values()
    print(f"  {len(df):,} rows, {len(dates_all)} dates")

    # Build WF folds
    n = len(dates_all)
    folds = []
    cursor = INIT_DAYS
    while cursor < n:
        test_end = min(cursor + STEP_DAYS, n)
        train_end_idx = max(0, cursor - HORIZON)
        tr = dates_all[:train_end_idx]
        te = dates_all[cursor:test_end]
        if len(tr) >= 50 and len(te) > 0:
            folds.append((tr, te))
        cursor = test_end
    print(f"  {len(folds)} WF folds")

    model_names = list(ALL_TRAINERS.keys())

    # Run WF and collect per-date top-K individual stock returns
    print(f"\n[2/3] Walk-forward training + collecting top-{TOP_K} stock returns...")

    # model -> list of dicts {date, mean_ret, individual_rets, tickers, var_within}
    model_data = {name: [] for name in model_names}

    for fold_num, (train_dates, test_dates) in enumerate(folds, 1):
        train_df = df.loc[(train_dates, slice(None)), :]
        test_df = df.loc[(test_dates, slice(None)), :]

        X_train = train_df[FEATURES].fillna(0.0).to_numpy()
        y_train = train_df['target'].to_numpy()
        dates_train = train_df.index.get_level_values('date').to_numpy()

        X_test = test_df[FEATURES].fillna(0.0).to_numpy()
        y_test = test_df['target'].to_numpy()
        dates_test = test_df.index.get_level_values('date').to_numpy()
        tickers_test = test_df.index.get_level_values('ticker').to_numpy()

        print(f"  [Fold {fold_num}/{len(folds)}] train: {len(train_dates)}d, {len(train_df):,}rows | "
              f"test: {len(test_dates)}d", end='', flush=True)

        for name in model_names:
            t0 = time.time()
            preds = ALL_TRAINERS[name](X_train, y_train, dates_train, X_test)
            elapsed = time.time() - t0

            # Per-date top-K analysis
            for d in np.unique(dates_test):
                mask = dates_test == d
                if np.sum(mask) < 2 * TOP_K:
                    continue
                p = preds[mask]
                t = y_test[mask]
                tk = tickers_test[mask]
                order = np.argsort(-p)
                top_idx = order[:TOP_K]
                bot_idx = order[-TOP_K:]

                top_rets = t[top_idx]
                bot_rets = t[bot_idx]

                model_data[name].append({
                    'date': d,
                    'fold': fold_num,
                    'mean_top': np.mean(top_rets),
                    'mean_bot': np.mean(bot_rets),
                    'spread': np.mean(top_rets) - np.mean(bot_rets),
                    'var_within_top': np.var(top_rets, ddof=1),
                    'std_within_top': np.std(top_rets, ddof=1),
                    'top_rets': top_rets.tolist(),
                    'top_tickers': tk[top_idx].tolist(),
                })

            print(f"  {name[:6]}={elapsed:.0f}s", end='', flush=True)
        print(flush=True)

    # ═══════════════════════════════════════
    #  ANALYSIS
    # ═══════════════════════════════════════
    print(f"\n[3/3] ANOVA Analysis...")

    # Convert to DataFrames
    model_dfs = {}
    for name in model_names:
        model_dfs[name] = pd.DataFrame(model_data[name])

    n_dates = len(model_dfs[model_names[0]])

    # Assign time periods (6 equal blocks)
    all_dates_sorted = sorted(model_dfs[model_names[0]]['date'].unique())
    n_td = len(all_dates_sorted)
    period_size = n_td // 6
    date_to_period = {}
    for i, d in enumerate(all_dates_sorted):
        date_to_period[d] = min(i // period_size, 5)  # 0-5
    period_labels = ['P1_early', 'P2', 'P3', 'P4', 'P5', 'P6_late']

    for name in model_names:
        model_dfs[name]['period'] = model_dfs[name]['date'].map(date_to_period)
        model_dfs[name]['period_label'] = model_dfs[name]['period'].map(lambda p: period_labels[p])

    # ─── 1. Within-top-10 variance by model ───
    print(f"\n{sep}")
    print(f"  1. WITHIN-TOP-10 VARIANCE (dispersion among 10 stocks per date)")
    print(sep)
    print(f"\n  {'Model':18s}  {'Mean(Var)':>10s} {'Mean(Std)':>10s} {'Median(Std)':>12s} "
          f"{'Std_range':>10s} {'CoeffVar':>10s}")
    print(f"  {'-'*75}")

    for name in model_names:
        mdf = model_dfs[name]
        mean_var = mdf['var_within_top'].mean()
        mean_std = mdf['std_within_top'].mean()
        median_std = mdf['std_within_top'].median()
        # Coefficient of variation of the top-10 mean return
        cv = mdf['std_within_top'].mean() / abs(mdf['mean_top'].mean()) if mdf['mean_top'].mean() != 0 else 0
        std_range = f"{mdf['std_within_top'].quantile(0.25):.4f}-{mdf['std_within_top'].quantile(0.75):.4f}"
        print(f"  {name:18s}  {mean_var:10.6f} {mean_std:10.4f} {median_std:12.4f} "
              f"{std_range:>10s} {cv:10.2f}")

    # Within-variance by period
    print(f"\n  Within-top-10 Std by Period:")
    print(f"  {'Model':18s}", end='')
    for pl in period_labels:
        print(f"  {pl:>10s}", end='')
    print()
    print(f"  {'-'*82}")
    for name in model_names:
        mdf = model_dfs[name]
        line = f"  {name:18s}"
        for p in range(6):
            sub = mdf[mdf['period'] == p]
            line += f"  {sub['std_within_top'].mean():10.4f}"
        print(line)

    # ─── 2. Between-period variance ───
    print(f"\n{sep}")
    print(f"  2. BETWEEN-PERIOD VARIANCE (how top-10 mean return varies over time)")
    print(sep)

    print(f"\n  Mean top-10 return by period (5-day forward return):")
    print(f"  {'Model':18s}", end='')
    for pl in period_labels:
        print(f"  {pl:>10s}", end='')
    print(f"  {'Overall':>10s} {'Btw-Var':>10s}")
    print(f"  {'-'*100}")

    for name in model_names:
        mdf = model_dfs[name]
        line = f"  {name:18s}"
        period_means = []
        for p in range(6):
            sub = mdf[mdf['period'] == p]
            pm = sub['mean_top'].mean()
            period_means.append(pm)
            line += f"  {pm:+9.3%}"
        overall = mdf['mean_top'].mean()
        btw_var = np.var(period_means, ddof=1)
        line += f"  {overall:+9.3%} {btw_var:10.6f}"
        print(line)

    # Spread by period
    print(f"\n  Long-short spread by period:")
    print(f"  {'Model':18s}", end='')
    for pl in period_labels:
        print(f"  {pl:>10s}", end='')
    print(f"  {'Overall':>10s}")
    print(f"  {'-'*90}")

    for name in model_names:
        mdf = model_dfs[name]
        line = f"  {name:18s}"
        for p in range(6):
            sub = mdf[mdf['period'] == p]
            line += f"  {sub['spread'].mean():+9.3%}"
        line += f"  {mdf['spread'].mean():+9.3%}"
        print(line)

    # ─── 3. One-way ANOVA: Model effect on top-10 mean return ───
    print(f"\n{sep}")
    print(f"  3. ONE-WAY ANOVA: Does model choice significantly affect top-10 returns?")
    print(sep)

    groups = [model_dfs[name]['mean_top'].values for name in model_names]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\n  F-statistic: {f_stat:.4f}")
    print(f"  p-value:     {p_val:.6f}")
    print(f"  Significant: {'YES (p < 0.05)' if p_val < 0.05 else 'NO (p >= 0.05)'}")

    # Group means and std
    print(f"\n  Per-model summary (top-10 5d mean return):")
    print(f"  {'Model':18s}  {'N':>6s} {'Mean':>10s} {'Std':>10s} {'Median':>10s} {'Skew':>8s} {'Kurt':>8s}")
    print(f"  {'-'*75}")
    for name in model_names:
        arr = model_dfs[name]['mean_top'].values
        print(f"  {name:18s}  {len(arr):6d} {np.mean(arr):+9.4%} {np.std(arr):10.4f} "
              f"{np.median(arr):+9.4%} {stats.skew(arr):8.3f} {stats.kurtosis(arr):8.3f}")

    # ─── 4. Pairwise comparisons ───
    print(f"\n{sep}")
    print(f"  4. PAIRWISE COMPARISONS (Welch t-test, Bonferroni corrected)")
    print(sep)

    n_comparisons = len(model_names) * (len(model_names) - 1) // 2
    print(f"\n  {'Pair':40s}  {'t-stat':>8s} {'raw_p':>10s} {'adj_p':>10s} {'Sig':>5s} {'Cohen_d':>8s}")
    print(f"  {'-'*85}")

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            n1, n2 = model_names[i], model_names[j]
            a1 = model_dfs[n1]['mean_top'].values
            a2 = model_dfs[n2]['mean_top'].values
            t_stat, p_raw = stats.ttest_ind(a1, a2, equal_var=False)
            p_adj = min(p_raw * n_comparisons, 1.0)  # Bonferroni
            pooled_std = np.sqrt((np.var(a1, ddof=1) + np.var(a2, ddof=1)) / 2)
            cohens_d = (np.mean(a1) - np.mean(a2)) / pooled_std if pooled_std > 0 else 0
            sig = '*' if p_adj < 0.05 else ''
            print(f"  {n1+' vs '+n2:40s}  {t_stat:8.3f} {p_raw:10.6f} {p_adj:10.6f} {sig:>5s} {cohens_d:+8.4f}")

    # ─── 5. Two-way ANOVA: Model x Period interaction ───
    print(f"\n{sep}")
    print(f"  5. TWO-WAY ANOVA: Model x Time-Period interaction")
    print(sep)

    # Build combined DataFrame
    all_rows = []
    for name in model_names:
        mdf = model_dfs[name]
        for _, row in mdf.iterrows():
            all_rows.append({
                'model': name,
                'period': row['period'],
                'mean_top': row['mean_top'],
                'spread': row['spread'],
                'var_within': row['var_within_top'],
            })
    combined = pd.DataFrame(all_rows)

    # Manual two-way ANOVA (Type I SS)
    grand_mean = combined['mean_top'].mean()
    n_total = len(combined)

    # SS Total
    ss_total = np.sum((combined['mean_top'] - grand_mean) ** 2)

    # SS Model (between models)
    ss_model = 0
    for name in model_names:
        sub = combined[combined['model'] == name]
        ss_model += len(sub) * (sub['mean_top'].mean() - grand_mean) ** 2

    # SS Period (between periods)
    ss_period = 0
    for p in range(6):
        sub = combined[combined['period'] == p]
        ss_period += len(sub) * (sub['mean_top'].mean() - grand_mean) ** 2

    # SS Interaction
    ss_interaction = 0
    for name in model_names:
        for p in range(6):
            sub = combined[(combined['model'] == name) & (combined['period'] == p)]
            if len(sub) == 0:
                continue
            model_mean = combined[combined['model'] == name]['mean_top'].mean()
            period_mean = combined[combined['period'] == p]['mean_top'].mean()
            cell_mean = sub['mean_top'].mean()
            ss_interaction += len(sub) * (cell_mean - model_mean - period_mean + grand_mean) ** 2

    # SS Error
    ss_error = ss_total - ss_model - ss_period - ss_interaction

    # Degrees of freedom
    df_model = len(model_names) - 1
    df_period = 5
    df_interaction = df_model * df_period
    df_error = n_total - len(model_names) * 6

    # Mean squares
    ms_model = ss_model / df_model if df_model > 0 else 0
    ms_period = ss_period / df_period if df_period > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0

    # F-statistics
    f_model = ms_model / ms_error if ms_error > 0 else 0
    f_period = ms_period / ms_error if ms_error > 0 else 0
    f_interaction = ms_interaction / ms_error if ms_error > 0 else 0

    # p-values
    p_model = 1 - stats.f.cdf(f_model, df_model, df_error) if df_error > 0 else 1
    p_period = 1 - stats.f.cdf(f_period, df_period, df_error) if df_error > 0 else 1
    p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_error) if df_error > 0 else 1

    # Eta-squared (effect size)
    eta2_model = ss_model / ss_total if ss_total > 0 else 0
    eta2_period = ss_period / ss_total if ss_total > 0 else 0
    eta2_interaction = ss_interaction / ss_total if ss_total > 0 else 0

    print(f"\n  {'Source':18s}  {'SS':>12s} {'df':>5s} {'MS':>12s} {'F':>10s} {'p-value':>10s} {'eta²':>8s} {'Sig':>5s}")
    print(f"  {'-'*90}")
    print(f"  {'Model':18s}  {ss_model:12.6f} {df_model:5d} {ms_model:12.8f} {f_model:10.4f} {p_model:10.6f} {eta2_model:8.4f} "
          f"{'***' if p_model < 0.001 else '**' if p_model < 0.01 else '*' if p_model < 0.05 else '':>5s}")
    print(f"  {'Period':18s}  {ss_period:12.6f} {df_period:5d} {ms_period:12.8f} {f_period:10.4f} {p_period:10.6f} {eta2_period:8.4f} "
          f"{'***' if p_period < 0.001 else '**' if p_period < 0.01 else '*' if p_period < 0.05 else '':>5s}")
    print(f"  {'Model x Period':18s}  {ss_interaction:12.6f} {df_interaction:5d} {ms_interaction:12.8f} {f_interaction:10.4f} {p_interaction:10.6f} {eta2_interaction:8.4f} "
          f"{'***' if p_interaction < 0.001 else '**' if p_interaction < 0.01 else '*' if p_interaction < 0.05 else '':>5s}")
    print(f"  {'Error':18s}  {ss_error:12.6f} {df_error:5d} {ms_error:12.8f}")
    print(f"  {'Total':18s}  {ss_total:12.6f} {n_total-1:5d}")

    # ─── 6. Variance decomposition ───
    print(f"\n{sep}")
    print(f"  6. VARIANCE DECOMPOSITION (% of total variance)")
    print(sep)

    # For each model, decompose variance into: within-top10, between-date, residual
    print(f"\n  Total variance of top-10 individual stock returns, decomposed:")
    print(f"  {'Model':18s}  {'Total_Var':>10s} {'Within-10':>10s} {'Between-Date':>13s} {'%Within':>8s} {'%Between':>9s}")
    print(f"  {'-'*75}")

    for name in model_names:
        mdf = model_dfs[name]
        # Collect all individual returns
        all_ind_rets = []
        for _, row in mdf.iterrows():
            all_ind_rets.extend(row['top_rets'])
        all_ind_rets = np.array(all_ind_rets)
        total_var = np.var(all_ind_rets, ddof=1)

        # Within-group (average within-date variance)
        within_var = mdf['var_within_top'].mean()

        # Between-group (variance of date means)
        between_var = np.var(mdf['mean_top'].values, ddof=1)

        pct_within = within_var / total_var * 100 if total_var > 0 else 0
        pct_between = between_var / total_var * 100 if total_var > 0 else 0

        print(f"  {name:18s}  {total_var:10.6f} {within_var:10.6f} {between_var:13.6f} "
              f"{pct_within:7.1f}% {pct_between:8.1f}%")

    # ─── 7. Top-10 overlap between models ───
    print(f"\n{sep}")
    print(f"  7. TOP-10 OVERLAP BETWEEN MODELS (avg % common stocks per date)")
    print(sep)

    # Build ticker sets per date per model
    model_topk_sets = {}
    for name in model_names:
        mdf = model_dfs[name]
        sets = {}
        for _, row in mdf.iterrows():
            sets[row['date']] = set(row['top_tickers'])
        model_topk_sets[name] = sets

    print(f"\n  {'':18s}", end='')
    for n2 in model_names:
        print(f"  {n2[:12]:>12s}", end='')
    print()
    print(f"  {'-'*(18 + 14*len(model_names))}")

    for n1 in model_names:
        line = f"  {n1:18s}"
        for n2 in model_names:
            if n1 == n2:
                line += f"  {'100%':>12s}"
            else:
                s1 = model_topk_sets[n1]
                s2 = model_topk_sets[n2]
                common_dates = set(s1.keys()) & set(s2.keys())
                overlaps = [len(s1[d] & s2[d]) / TOP_K for d in common_dates if d in s1 and d in s2]
                avg_overlap = np.mean(overlaps) if overlaps else 0
                line += f"  {avg_overlap:11.0%}"
        print(line)

    # ─── 8. Kruskal-Wallis (non-parametric) ───
    print(f"\n{sep}")
    print(f"  8. KRUSKAL-WALLIS TEST (non-parametric alternative to one-way ANOVA)")
    print(sep)

    h_stat, p_kw = stats.kruskal(*groups)
    print(f"\n  H-statistic: {h_stat:.4f}")
    print(f"  p-value:     {p_kw:.6f}")
    print(f"  Significant: {'YES' if p_kw < 0.05 else 'NO'}")

    # Also test on spread
    spread_groups = [model_dfs[name]['spread'].values for name in model_names]
    h_spread, p_spread = stats.kruskal(*spread_groups)
    print(f"\n  On long-short spread:")
    print(f"  H-statistic: {h_spread:.4f}")
    print(f"  p-value:     {p_spread:.6f}")
    print(f"  Significant: {'YES' if p_spread < 0.05 else 'NO'}")

    # ─── 9. Summary ───
    print(f"\n{sep}")
    print(f"  SUMMARY")
    print(sep)
    print(f"""
  Key findings:
  - One-way ANOVA (model effect on top-10 return): F={f_stat:.3f}, p={p_val:.6f}
    {'=> Models produce SIGNIFICANTLY different top-10 returns' if p_val < 0.05 else '=> No significant difference between models'}
  - Two-way ANOVA:
    Model effect:      eta²={eta2_model:.4f} ({eta2_model*100:.2f}% of variance), p={p_model:.6f}
    Period effect:     eta²={eta2_period:.4f} ({eta2_period*100:.2f}% of variance), p={p_period:.6f}
    Interaction:       eta²={eta2_interaction:.4f} ({eta2_interaction*100:.2f}% of variance), p={p_interaction:.6f}
  - Time period explains {'MORE' if eta2_period > eta2_model else 'LESS'} variance than model choice
  - Kruskal-Wallis (non-parametric): H={h_stat:.3f}, p={p_kw:.6f}
""")

    print(f"{sep}")
    print(f"  DONE")
    print(sep)


if __name__ == '__main__':
    main()
