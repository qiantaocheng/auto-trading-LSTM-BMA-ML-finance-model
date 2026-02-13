#!/usr/bin/env python3
"""
LambdaRank Walk-Forward: Conviction-Weighted Top-10 Test

Tests whether allocating more capital to higher-ranked stocks improves performance
compared to equal-weight top 10.

Weight schemes tested:
  1. Equal weight (baseline):     w = [10%, 10%, ..., 10%]
  2. Linear rank decay:           w_i ∝ (K+1-rank_i) → rank1=18.2%, rank10=1.8%
  3. Sqrt rank decay:             w_i ∝ sqrt(K+1-rank_i) → smoother tilt
  4. Log rank decay:              w_i ∝ log(K+2-rank_i) → very gentle tilt
  5. Concentrated top-5 (50/50):  top5=12% each, bot5=8% each → mild tilt
  6. Score-proportional:           w_i ∝ softmax(pred_score * temperature)

Each tested with 6 walkforward folds, MA200 overlay, 2% SL, 5d rebalance.
"""

import sys, warnings, pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ─── LambdaRank pipeline components (same as main pipeline) ───

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


# ─── Portfolio metrics (same as main pipeline) ───
def calc_metrics_5d(eq_arr, rebal_days=5):
    if len(eq_arr) < rebal_days + 1:
        return {'CAGR': 0, 'Vol': 0, 'Sharpe': 0}
    idx = np.arange(0, len(eq_arr), rebal_days)
    if idx[-1] != len(eq_arr) - 1:
        idx = np.append(idx, len(eq_arr) - 1)
    sampled = eq_arr[idx]
    period_rets = sampled[1:] / sampled[:-1] - 1
    n_days = len(eq_arr) - 1
    years = n_days / 252
    if years <= 0 or len(period_rets) < 2:
        return {'CAGR': 0, 'Vol': 0, 'Sharpe': 0}
    cagr = (eq_arr[-1] / eq_arr[0]) ** (1.0 / years) - 1
    periods_per_year = 252 / rebal_days
    vol = float(np.std(period_rets, ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(np.mean(period_rets) / np.std(period_rets, ddof=1) * np.sqrt(periods_per_year)) if vol > 0 else 0
    return {'CAGR': cagr, 'Vol': vol, 'Sharpe': sharpe}


def calc_maxdd_5d(eq_arr, rebal_days=5):
    idx = np.arange(0, len(eq_arr), rebal_days)
    if idx[-1] != len(eq_arr) - 1:
        idx = np.append(idx, len(eq_arr) - 1)
    sampled = eq_arr[idx]
    peak = np.maximum.accumulate(sampled)
    dd = (sampled - peak) / peak
    return float(dd.min())


# ═══════════════════════════════════════════════════════
# CONVICTION-WEIGHTED PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════

def compute_conviction_weights(n_stocks, scheme='equal', scores=None, temperature=1.0):
    """
    Compute portfolio weights for top-K stocks based on their rank.

    Args:
        n_stocks: number of stocks in portfolio (K)
        scheme: weighting scheme name
        scores: raw prediction scores (for score-proportional schemes)
        temperature: softmax temperature (higher = more equal)

    Returns:
        weights array of length n_stocks, sums to 1.0
    """
    if n_stocks == 0:
        return np.array([])

    ranks = np.arange(1, n_stocks + 1)  # 1=best, K=worst

    if scheme == 'equal':
        w = np.ones(n_stocks)

    elif scheme == 'linear':
        # rank1 gets K points, rank10 gets 1 point
        w = (n_stocks + 1 - ranks).astype(float)

    elif scheme == 'sqrt':
        w = np.sqrt(n_stocks + 1 - ranks).astype(float)

    elif scheme == 'log':
        w = np.log(n_stocks + 2 - ranks).astype(float)

    elif scheme == 'top5_tilt':
        # Top 5 get 12% each (60%), bottom 5 get 8% each (40%)
        w = np.where(ranks <= 5, 1.2, 0.8).astype(float)

    elif scheme == 'score_soft':
        # Softmax of prediction scores with temperature
        if scores is not None and len(scores) == n_stocks:
            s = np.array(scores, dtype=float)
            s = (s - s.mean()) / max(s.std(), 1e-8)  # normalize
            s = s / max(temperature, 0.01)
            exp_s = np.exp(s - s.max())
            w = exp_s
        else:
            w = np.ones(n_stocks)

    elif scheme == 'score_soft_warm':
        # Higher temperature = more equal; lower = more concentrated
        if scores is not None and len(scores) == n_stocks:
            s = np.array(scores, dtype=float)
            s = (s - s.mean()) / max(s.std(), 1e-8)
            s = s / max(temperature, 0.01)
            exp_s = np.exp(s - s.max())
            w = exp_s
        else:
            w = np.ones(n_stocks)

    else:
        w = np.ones(n_stocks)

    w = w / w.sum()
    return w


def simulate_portfolio_weighted(oos_trading_dates, topk_scores_by_date, px, spy_px,
                                spy_ma200, use_ma200, top_k, cost_bps, rebal_days,
                                stop_loss_pct, weight_scheme='equal', temperature=1.0):
    """
    Conviction-weighted portfolio simulation.

    topk_scores_by_date: {date: [(ticker, score), ...]} sorted by score descending
    """
    pred_dates_ts = [pd.Timestamp(d) for d in sorted(topk_scores_by_date.keys())]

    def _get_exposure(dt):
        exp = 1.0
        if use_ma200 and dt in spy_px.index and dt in spy_ma200.index:
            sp = float(spy_px.loc[dt])
            ma = float(spy_ma200.loc[dt])
            if not np.isnan(sp) and not np.isnan(ma) and ma > 0:
                if sp < ma * MA200_DEEP_THR:
                    exp = min(exp, MA200_DEEP)
                elif sp < ma:
                    exp = min(exp, MA200_SHALLOW)
        return exp

    cap_ovl = 1.0
    cap_raw = 1.0
    eq_ovl = [cap_ovl]
    eq_raw = [cap_raw]
    eq_dates = [oos_trading_dates[0]]
    drets_ovl = []
    drets_raw = []
    holdings = []          # list of tickers currently held
    weights = {}           # {ticker: weight}  (sums to 1.0)
    entry_prices = {}
    rc = rebal_days        # trigger first rebalance immediately
    n_stops = 0

    for i in range(1, len(oos_trading_dates)):
        dt = oos_trading_dates[i]
        prev_dt = oos_trading_dates[i - 1]

        # ── Compute daily portfolio return (conviction-weighted) ──
        port_ret = 0.0
        stopped = []
        if holdings:
            for tk in holdings:
                if tk in px.columns and dt in px.index and prev_dt in px.index:
                    p_now = px.loc[dt, tk]
                    p_prev = px.loc[prev_dt, tk]
                    if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                        tk_ret = p_now / p_prev - 1
                        w = weights.get(tk, 1.0 / len(holdings))

                        # Stop-loss check
                        if (stop_loss_pct is not None and tk in entry_prices
                                and entry_prices[tk] > 0):
                            drawdown_from_entry = (p_now / entry_prices[tk]) - 1.0
                            if drawdown_from_entry <= -stop_loss_pct:
                                stop_price = entry_prices[tk] * (1 - stop_loss_pct)
                                tk_ret = stop_price / p_prev - 1
                                stopped.append(tk)

                        port_ret += w * tk_ret

        # Remove stopped stocks
        if stopped:
            n_stops += len(stopped)
            # Redistribute stopped weight proportionally to remaining stocks
            stopped_weight = sum(weights.get(tk, 0) for tk in stopped)
            cost_pct = stopped_weight * cost_bps / 10_000
            cap_ovl -= cost_pct * cap_ovl
            cap_raw -= cost_pct * cap_raw
            holdings = [tk for tk in holdings if tk not in stopped]
            for tk in stopped:
                entry_prices.pop(tk, None)
                weights.pop(tk, None)
            # Renormalize remaining weights
            if holdings:
                total_w = sum(weights.get(tk, 0) for tk in holdings)
                if total_w > 0:
                    for tk in holdings:
                        weights[tk] = weights.get(tk, 0) / total_w

        # Raw
        cap_raw *= (1 + port_ret)
        drets_raw.append(port_ret)

        # Overlayed
        exp = _get_exposure(dt) if use_ma200 else 1.0
        cap_ovl *= (1 + exp * port_ret)
        drets_ovl.append(exp * port_ret)

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
                for key in topk_scores_by_date.keys():
                    if pd.Timestamp(key) == best_pred_dt:
                        # Get top-K tickers with scores
                        all_entries = topk_scores_by_date[key][:top_k]
                        new_holdings = [tk for tk, sc in all_entries if tk in px.columns]
                        new_scores = [sc for tk, sc in all_entries if tk in px.columns]

                        if new_holdings:
                            # Compute conviction weights
                            conv_weights = compute_conviction_weights(
                                len(new_holdings), scheme=weight_scheme,
                                scores=new_scores, temperature=temperature)

                            # Transaction cost based on turnover
                            turnover = len(set(new_holdings) - set(holdings)) / max(len(new_holdings), 1)
                            cost_pct = turnover * cost_bps / 10_000
                            cap_ovl -= cost_pct * cap_ovl
                            cap_raw -= cost_pct * cap_raw

                            # Record entry prices for NEW stocks
                            for tk in new_holdings:
                                if tk not in holdings and dt in px.index:
                                    p_entry = px.loc[dt, tk]
                                    if not np.isnan(p_entry):
                                        entry_prices[tk] = float(p_entry)
                            # Remove entry prices for stocks no longer held
                            for tk in holdings:
                                if tk not in new_holdings:
                                    entry_prices.pop(tk, None)

                            # Update holdings and weights
                            holdings = new_holdings
                            weights = {tk: float(w) for tk, w in zip(new_holdings, conv_weights)}
                        break

        eq_ovl.append(cap_ovl)
        eq_raw.append(cap_raw)
        eq_dates.append(dt)

    sim_stats = {'n_stops': n_stops}
    return (np.array(eq_ovl), np.array(eq_raw), eq_dates,
            np.array(drets_ovl), np.array(drets_raw), sim_stats)


def main():
    sep = "=" * 110
    print(f"\n{sep}")
    print("  LambdaRank WF: CONVICTION-WEIGHTED Top-10 Test")
    print(f"  Settings: K=10, MA200=ON, SL=2%, 5d rebal, {COST_BPS}bps cost")
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

    # ─── 2. Walk-forward (load from cache) ───
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
                  f"({len(train_dates_sel)}d) | "
                  f"test: {test_dates_sel[0].date()}..{test_dates_sel[-1].date()} "
                  f"({len(test_dates_sel)}d, {len(test_df)}rows)")

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

    # ─── 3. Build top-K with scores ───
    print(f"\n[3/5] Building top-K holdings with scores...")
    unique_oos_dates = np.sort(np.unique(dates_cat))
    print(f"  OOS period: {pd.Timestamp(unique_oos_dates[0]).date()} .. "
          f"{pd.Timestamp(unique_oos_dates[-1]).date()} ({len(unique_oos_dates)} days)")

    # Build {date: [(ticker, score), ...]} sorted by score descending
    topk_scores_by_date = {}
    for d in unique_oos_dates:
        mask = dates_cat == d
        dp = preds_cat[mask]
        dt_tickers = tickers_cat[mask]
        if len(dp) < 20:
            continue
        order = np.argsort(-dp)[:20]
        topk_scores_by_date[d] = [(dt_tickers[o], float(dp[o])) for o in order]

    # ─── 4. Download prices & compute regimes ───
    print(f"\n[4/5] Downloading prices...")
    all_top_tickers = set()
    for entries in topk_scores_by_date.values():
        all_top_tickers.update(tk for tk, sc in entries)

    import yfinance as yf
    ticker_list = sorted(all_top_tickers)
    if 'SPY' not in ticker_list:
        ticker_list.append('SPY')

    print(f"  Downloading {len(ticker_list)} tickers from yfinance...")
    start_date = str(pd.Timestamp(unique_oos_dates[0]).date() - pd.Timedelta(days=300))
    end_date = str(pd.Timestamp(unique_oos_dates[-1]).date() + pd.Timedelta(days=30))
    px = yf.download(ticker_list, start=start_date, end=end_date,
                     auto_adjust=True, progress=False)['Close']
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(-1)
    px = px.ffill()
    print(f"  Price data: {len(px)} days, {len(px.columns)} tickers")

    spy_px = px['SPY'].dropna()
    spy_ma200 = spy_px.rolling(200).mean()

    oos_trading_dates = px.index[px.index >= pd.Timestamp(unique_oos_dates[0])]
    oos_trading_dates = oos_trading_dates[oos_trading_dates <= pd.Timestamp(unique_oos_dates[-1])]

    # ─── 5. Test all weight schemes ───
    K = 10
    RB = REBALANCE_DAYS
    SL = 0.02

    print(f"\n[5/5] Testing conviction weight schemes...")
    print(f"  Settings: K={K}, MA200=ON, SL={SL*100:.0f}%, {RB}d rebal, {COST_BPS}bps")

    # Define all test configurations
    test_configs = [
        ('Equal (baseline)', 'equal', 1.0),
        ('Linear decay', 'linear', 1.0),
        ('Sqrt decay', 'sqrt', 1.0),
        ('Log decay', 'log', 1.0),
        ('Top5 tilt (60/40)', 'top5_tilt', 1.0),
        ('Score softmax T=0.5', 'score_soft', 0.5),
        ('Score softmax T=1.0', 'score_soft', 1.0),
        ('Score softmax T=2.0', 'score_soft', 2.0),
    ]

    # Show weight distribution for each scheme
    print(f"\n  Weight Distributions (rank 1..10):")
    print(f"  {'Scheme':<22}  {'R1':>5} {'R2':>5} {'R3':>5} {'R4':>5} {'R5':>5} "
          f"{'R6':>5} {'R7':>5} {'R8':>5} {'R9':>5} {'R10':>5}  {'Max/Min':>8}")
    print("  " + "-" * 90)

    # Pick a sample date for score-based weights
    sample_d = list(topk_scores_by_date.keys())[len(topk_scores_by_date)//2]
    sample_scores = [sc for tk, sc in topk_scores_by_date[sample_d][:K]]

    for label, scheme, temp in test_configs:
        if 'score' in scheme:
            w = compute_conviction_weights(K, scheme=scheme, scores=sample_scores, temperature=temp)
        else:
            w = compute_conviction_weights(K, scheme=scheme)
        ratio = w.max() / w.min() if w.min() > 0 else float('inf')
        print(f"  {label:<22}", end='')
        for wi in w:
            print(f"  {wi:>4.1%}", end='')
        print(f"  {ratio:>7.1f}x")

    # Run simulations
    results = []

    print(f"\n  {'='*110}")
    print(f"  CONVICTION WEIGHT TEST — FULL PERIOD")
    print(f"  {'='*110}")
    header = (f"  {'Config':<22} {'Sharpe':>7} {'CAGR':>9} "
              f"{'MaxDD':>7} {'Calmar':>8} {'Vol':>6} "
              f"{'Stops':>5}")
    divider = "  " + "-" * 70
    print(header)
    print(divider)

    for label, scheme, temp in test_configs:
        eq_ovl, eq_raw, eq_dt, dr_ovl, dr_raw, sim_stats = simulate_portfolio_weighted(
            oos_trading_dates, topk_scores_by_date, px, spy_px, spy_ma200,
            use_ma200=True, top_k=K, cost_bps=COST_BPS, rebal_days=RB,
            stop_loss_pct=SL, weight_scheme=scheme, temperature=temp)

        m = calc_metrics_5d(eq_ovl, rebal_days=RB)
        maxdd = calc_maxdd_5d(eq_raw, rebal_days=RB)
        calmar = m['CAGR'] / abs(maxdd) if abs(maxdd) > 0 else 0

        row = {
            'label': label, 'scheme': scheme, 'temp': temp,
            'CAGR': m['CAGR'], 'MaxDD': maxdd, 'Sharpe': m['Sharpe'],
            'Calmar': calmar, 'Vol': m['Vol'], 'n_stops': sim_stats['n_stops'],
            'eq_ovl': eq_ovl, 'eq_raw': eq_raw, 'eq_dt': eq_dt,
        }
        results.append(row)

        print(f"  {label:<22} {m['Sharpe']:>7.3f} {m['CAGR']:>+9.2%} "
              f"{maxdd:>+7.1%} {calmar:>8.3f} {m['Vol']:>5.1%} "
              f"{sim_stats['n_stops']:>5}")

    # ─── YEARLY BREAKDOWN ───
    print(f"\n  {'='*110}")
    print(f"  YEARLY BREAKDOWN")
    print(f"  {'='*110}")

    # Determine all years
    all_years = sorted(set(d.year for r in results for d in r['eq_dt']))

    # Yearly Sharpe table
    print(f"\n  Sharpe by Year:")
    print(f"  {'Config':<22}", end='')
    for yr in all_years:
        print(f"  {yr:>8d}", end='')
    print()
    print("  " + "-" * (22 + 10 * len(all_years)))

    for r in results:
        line = f"  {r['label']:<22}"
        eq_dt = r['eq_dt']
        eq_ovl = r['eq_ovl']
        for yr in all_years:
            yr_idx = np.array([j for j, d in enumerate(eq_dt) if d.year == yr])
            if len(yr_idx) < 10:
                line += f"  {'N/A':>8}"
            else:
                yr_m = calc_metrics_5d(eq_ovl[yr_idx], rebal_days=RB)
                line += f"  {yr_m['Sharpe']:>8.3f}"
        print(line)

    # Yearly CAGR + MaxDD table
    print(f"\n  CAGR / MaxDD by Year:")
    print(f"  {'Config':<22}", end='')
    for yr in all_years:
        print(f"  {str(yr):>14s}", end='')
    print()
    print(f"  {'':22}", end='')
    for yr in all_years:
        print(f"  {'CAGR':>7s} {'MDD':>6s}", end='')
    print()
    print("  " + "-" * (22 + 16 * len(all_years)))

    for r in results:
        line = f"  {r['label']:<22}"
        eq_dt = r['eq_dt']
        eq_ovl = r['eq_ovl']
        eq_raw = r['eq_raw']
        for yr in all_years:
            yr_idx = np.array([j for j, d in enumerate(eq_dt) if d.year == yr])
            if len(yr_idx) < 10:
                line += f"  {'N/A':>7s} {'N/A':>6s}"
            else:
                yr_m = calc_metrics_5d(eq_ovl[yr_idx], rebal_days=RB)
                yr_mdd = calc_maxdd_5d(eq_raw[yr_idx], rebal_days=RB)
                line += f"  {yr_m['CAGR']:>+6.1%} {yr_mdd:>+5.1%}"
        print(line)

    # ─── 6-FOLD WALKFORWARD ROBUSTNESS ───
    print(f"\n  {'='*110}")
    print(f"  6-FOLD WALKFORWARD ROBUSTNESS (each fold ~6 months)")
    print(f"  {'='*110}")

    # Split OOS dates into 6 roughly equal folds
    oos_dt_arr = np.array(oos_trading_dates)
    n_total = len(oos_dt_arr)
    fold_size = n_total // 6
    folds = []
    for f in range(6):
        start_idx = f * fold_size
        end_idx = (f + 1) * fold_size if f < 5 else n_total
        fold_dates = oos_dt_arr[start_idx:end_idx]
        folds.append(fold_dates)
        print(f"  Fold {f+1}: {pd.Timestamp(fold_dates[0]).strftime('%Y-%m-%d')} .. {pd.Timestamp(fold_dates[-1]).strftime('%Y-%m-%d')} "
              f"({len(fold_dates)} days)")

    # For each config, run each fold and report Sharpe
    print(f"\n  Sharpe by Fold:")
    print(f"  {'Config':<22}", end='')
    for f in range(6):
        print(f"  {'F'+str(f+1):>7}", end='')
    print(f"  {'Mean':>7}  {'StdDev':>7}  {'Wins':>5}")
    print("  " + "-" * 92)

    # Track which config wins each fold
    fold_sharpes = {}  # {config_label: [sharpe_f1, ..., sharpe_f6]}

    for r_idx, (label, scheme, temp) in enumerate(test_configs):
        fold_sh = []
        for fold_dates in folds:
            eq_ovl, eq_raw, eq_dt, dr_ovl, dr_raw, ss = simulate_portfolio_weighted(
                fold_dates, topk_scores_by_date, px, spy_px, spy_ma200,
                use_ma200=True, top_k=K, cost_bps=COST_BPS, rebal_days=RB,
                stop_loss_pct=SL, weight_scheme=scheme, temperature=temp)
            m = calc_metrics_5d(eq_ovl, rebal_days=RB)
            fold_sh.append(m['Sharpe'])

        fold_sharpes[label] = fold_sh
        n_wins = sum(1 for f_s in fold_sh if f_s > fold_sharpes.get('Equal (baseline)', [0]*6)[min(len(fold_sh)-1, 5)])
        mean_sh = np.mean(fold_sh)
        std_sh = np.std(fold_sh, ddof=1) if len(fold_sh) > 1 else 0

        line = f"  {label:<22}"
        for s in fold_sh:
            line += f"  {s:>7.3f}"
        line += f"  {mean_sh:>7.3f}  {std_sh:>7.3f}"
        print(line)

    # Print wins vs baseline
    baseline_sh = fold_sharpes.get('Equal (baseline)', [0]*6)
    print(f"\n  Folds beating Equal-Weight baseline:")
    for label in fold_sharpes:
        if label == 'Equal (baseline)':
            continue
        wins = sum(1 for a, b in zip(fold_sharpes[label], baseline_sh) if a > b)
        print(f"    {label:<22}: {wins}/6 folds")

    # ─── SUMMARY ───
    print(f"\n  {'='*110}")
    print(f"  SUMMARY")
    print(f"  {'='*110}")

    best = max(results, key=lambda r: r['Sharpe'])
    baseline = results[0]  # Equal weight

    print(f"\n  Baseline (Equal Weight): Sharpe={baseline['Sharpe']:.3f}  CAGR={baseline['CAGR']:+.2%}  "
          f"MaxDD={baseline['MaxDD']:+.1%}  Calmar={baseline['Calmar']:.3f}")
    print(f"  Best conviction:        Sharpe={best['Sharpe']:.3f}  CAGR={best['CAGR']:+.2%}  "
          f"MaxDD={best['MaxDD']:+.1%}  Calmar={best['Calmar']:.3f}  [{best['label']}]")

    delta_sharpe = best['Sharpe'] - baseline['Sharpe']
    delta_cagr = best['CAGR'] - baseline['CAGR']
    print(f"\n  Delta: Sharpe {delta_sharpe:+.3f}  CAGR {delta_cagr:+.2%}")

    if delta_sharpe > 0.1:
        print(f"\n  >>> RECOMMENDATION: Use '{best['label']}' — meaningful improvement over equal weight")
    elif delta_sharpe > 0:
        print(f"\n  >>> RECOMMENDATION: Marginal improvement — consider '{best['label']}' but equal weight is fine")
    else:
        print(f"\n  >>> RECOMMENDATION: Equal weight is optimal — conviction weighting does not help")

    print(f"\n{sep}")
    print("  DONE")
    print(sep)


if __name__ == '__main__':
    main()
