#!/usr/bin/env python3
"""
LambdaRank Walk-Forward → Top-K Selection → MA200+HMM Overlay Backtest (v2)

Changes from v1:
  - INIT_DAYS=252 to include 2022 bear market in OOS
  - Full-universe Spearman Rank IC (not top-K only)
  - Overlayed returns for Sharpe/CAGR/Calmar; NON-overlayed MaxDD
  - Yearly breakdown table
  - Rolling 252d Sharpe
  - Top-K sensitivity (5, 10, 20)
  - First-half / second-half robustness split
"""

import sys, warnings
from pathlib import Path
from typing import Dict, List

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

    # CV to get best_round (average across folds)
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

    # Retrain on full data
    full_set = lgb.Dataset(X, label=labels, group=group_counts(dates))
    model = lgb.train(lgb_params, full_set, num_boost_round=best_round,
                      callbacks=[lgb.log_evaluation(0)])
    return model, best_round


# ─── HMM p_risk ───
def compute_hmm_prisk(spy_close):
    log_ret = np.log(spy_close / spy_close.shift(1)).dropna()
    p_risk = pd.Series(0.0, index=spy_close.index)

    try:
        from hmmlearn.hmm import GaussianHMM
        HMM_WINDOW, HMM_RETRAIN, HMM_EMA = 500, 21, 10
        X_full = log_ret.values.reshape(-1, 1)
        model, crisis_idx = None, 0

        for i in range(HMM_WINDOW, len(log_ret)):
            if model is None or (i - HMM_WINDOW) % HMM_RETRAIN == 0:
                X_train = X_full[max(0, i - HMM_WINDOW):i]
                try:
                    m = GaussianHMM(n_components=2, covariance_type='full',
                                    n_iter=100, random_state=42)
                    m.fit(X_train)
                    crisis_idx = int(np.argmax(m.covars_.flatten()))
                    model = m
                except:
                    continue
            if model is not None and not np.isnan(X_full[i, 0]):
                try:
                    probs = model.predict_proba(X_full[i:i + 1])
                    p_risk.loc[log_ret.index[i]] = probs[0, crisis_idx]
                except:
                    pass

        p_risk = p_risk.ewm(span=HMM_EMA).mean()
        print("    HMM: Using hmmlearn GaussianHMM (2-state)")
    except ImportError:
        vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
        vol_120 = log_ret.rolling(120).std() * np.sqrt(252)
        vol_ratio = (vol_20 / vol_120).fillna(1.0)
        raw = 1.0 / (1.0 + np.exp(-(vol_ratio - 1.3) * 4))
        p_risk = raw.ewm(span=10).mean().reindex(spy_close.index).fillna(0)
        print("    HMM: Using vol-ratio proxy (hmmlearn not available)")

    return p_risk


# ─── Portfolio metrics ───
def calc_metrics_5d(eq_arr, rebal_days=5):
    """Compute CAGR, Vol, Sharpe from non-overlapping rebalance-period returns.

    Samples equity every `rebal_days` to get independent return observations.
    Sharpe = mean(period_ret) / std(period_ret) * sqrt(periods_per_year)
    """
    if len(eq_arr) < rebal_days + 1:
        return {'CAGR': 0, 'Vol': 0, 'Sharpe': 0}
    # Sample every rebal_days (non-overlapping)
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
    periods_per_year = 252 / rebal_days  # e.g. 50.4 for 5-day
    vol = float(np.std(period_rets, ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(np.mean(period_rets) / np.std(period_rets, ddof=1) * np.sqrt(periods_per_year)) if vol > 0 else 0
    return {'CAGR': cagr, 'Vol': vol, 'Sharpe': sharpe}


def calc_maxdd_5d(eq_arr, rebal_days=5):
    """MaxDD from equity array sampled at non-overlapping rebalance points."""
    idx = np.arange(0, len(eq_arr), rebal_days)
    if idx[-1] != len(eq_arr) - 1:
        idx = np.append(idx, len(eq_arr) - 1)
    sampled = eq_arr[idx]
    peak = np.maximum.accumulate(sampled)
    dd = (sampled - peak) / peak
    return float(dd.min())


# ─── Portfolio simulation engine ───
def simulate_portfolio(oos_trading_dates, top_k_by_date, px, spy_px, spy_ma200,
                       p_risk, use_hmm, use_ma200, top_k, cost_bps, rebal_days,
                       stop_loss_pct=None):
    """
    Returns (overlayed_equity, raw_equity, dates_list, daily_rets_overlay, daily_rets_raw, stats).
    Both curves share same holdings/rebalance. Overlay only affects position sizing.
    stop_loss_pct: if set (e.g. 0.10 for 10%), sells a stock when it drops this % from entry price.
    """
    pred_dates_ts = [pd.Timestamp(d) for d in sorted(top_k_by_date.keys())]

    def _get_exposure(dt):
        exp = 1.0
        if use_hmm and dt in p_risk.index:
            pr = float(p_risk.loc[dt])
            if pr >= 0.9:
                exp *= 0.50
            elif pr >= 0.5:
                exp *= 0.75
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
    entry_prices = {}      # {ticker: entry_price} for stop-loss tracking
    rc = rebal_days        # trigger first rebalance immediately
    n_stops = 0            # count stop-loss triggers

    for i in range(1, len(oos_trading_dates)):
        dt = oos_trading_dates[i]
        prev_dt = oos_trading_dates[i - 1]

        # ── Compute daily portfolio return (stop-loss capped) ──
        port_ret = 0.0
        stopped = []
        if holdings:
            daily_rets = []
            n_held = len(holdings)
            for tk in holdings:
                if tk in px.columns and dt in px.index and prev_dt in px.index:
                    p_now = px.loc[dt, tk]
                    p_prev = px.loc[prev_dt, tk]
                    if not np.isnan(p_now) and not np.isnan(p_prev) and p_prev > 0:
                        tk_ret = p_now / p_prev - 1
                        # Stop-loss: cap at stop level (intraday execution)
                        if (stop_loss_pct is not None and tk in entry_prices
                                and entry_prices[tk] > 0):
                            drawdown_from_entry = (p_now / entry_prices[tk]) - 1.0
                            if drawdown_from_entry <= -stop_loss_pct:
                                # Cap return: stock sold intraday at stop price
                                stop_price = entry_prices[tk] * (1 - stop_loss_pct)
                                tk_ret = stop_price / p_prev - 1
                                stopped.append(tk)
                        daily_rets.append(tk_ret)
            if daily_rets:
                port_ret = np.mean(daily_rets)

        # Remove stopped stocks AFTER return is computed
        if stopped:
            n_stops += len(stopped)
            cost_pct = len(stopped) / max(len(holdings), 1) * cost_bps / 10_000
            cap_ovl -= cost_pct * cap_ovl
            cap_raw -= cost_pct * cap_raw
            holdings = [tk for tk in holdings if tk not in stopped]
            for tk in stopped:
                entry_prices.pop(tk, None)

        # Raw (B&H)
        cap_raw *= (1 + port_ret)
        drets_raw.append(port_ret)

        # Overlayed
        exp = _get_exposure(dt) if (use_hmm or use_ma200) else 1.0
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
                for key in top_k_by_date.keys():
                    if pd.Timestamp(key) == best_pred_dt:
                        new_holdings = [tk for tk in top_k_by_date[key][:top_k]
                                        if tk in px.columns]
                        if new_holdings:
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
                            holdings = new_holdings
                        break

        eq_ovl.append(cap_ovl)
        eq_raw.append(cap_raw)
        eq_dates.append(dt)

    sim_stats = {'n_stops': n_stops}
    return (np.array(eq_ovl), np.array(eq_raw), eq_dates,
            np.array(drets_ovl), np.array(drets_raw), sim_stats)


def main():
    sep = "=" * 100
    print(f"\n{sep}")
    print("  LambdaRank WF Top-K + MA200/HMM Overlay Backtest  (v2: 2022+, robust)")
    print(sep)

    # ─── 1. Load data ───
    data_file = Path('data/factor_exports/polygon_full_features_T5.parquet')
    print(f"\n[1/7] Loading data: {data_file}")
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

    # ─── 2. Walk-forward ───
    INIT_DAYS = 252     # 1 year init → OOS starts ~2022-02
    STEP_DAYS = 63      # ~3 months
    HORIZON = 5
    CV_SPLITS = 5
    N_BOOST = 800
    SEED = 0

    # Check for cached WF predictions
    import pickle
    cache_file = Path('data/factor_exports/_wf_preds_cache.pkl')

    if cache_file.exists():
        print(f"\n[2/7] Loading cached WF predictions from {cache_file}")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        preds_cat = cache['preds']
        targets_cat = cache['targets']
        dates_cat = cache['dates']
        tickers_cat = cache['tickers']
        print(f"  Loaded {len(preds_cat)} predictions")
    else:
        print(f"\n[2/7] Walk-forward LambdaRank (init={INIT_DAYS}d, step={STEP_DAYS}d)...")
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

        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump({'preds': preds_cat, 'targets': targets_cat,
                         'dates': dates_cat, 'tickers': tickers_cat}, f)
        print(f"  Saved WF predictions to {cache_file}")

    # ─── 3. Full-universe ranking metrics ───
    print(f"\n[3/7] Full-universe OOS ranking metrics...")
    unique_oos_dates = np.sort(np.unique(dates_cat))
    print(f"  OOS period: {pd.Timestamp(unique_oos_dates[0]).date()} .. "
          f"{pd.Timestamp(unique_oos_dates[-1]).date()} ({len(unique_oos_dates)} days)")

    # Full-universe Spearman Rank IC + top-K return
    rank_ic_vals = []
    pearson_ic_vals = []
    topk_rets = {k: [] for k in [5, 10, 20]}
    btm_rets = {k: [] for k in [5, 10, 20]}
    ic_by_year = {}

    for d in unique_oos_dates:
        mask = dates_cat == d
        dp, dt = preds_cat[mask], targets_cat[mask]
        if len(dt) < 30:
            continue

        # Full-universe Spearman Rank IC
        ric = float(stats.spearmanr(dp, dt).correlation)
        if not np.isnan(ric):
            rank_ic_vals.append(ric)
            yr = pd.Timestamp(d).year
            ic_by_year.setdefault(yr, []).append(ric)

        pic = float(np.corrcoef(dp, dt)[0, 1])
        if not np.isnan(pic):
            pearson_ic_vals.append(pic)

        # Top-K and Bottom-K returns
        order = np.argsort(-dp)
        for k in [5, 10, 20]:
            if len(dt) >= k:
                topk_rets[k].append(dt[order[:k]].mean())
                btm_rets[k].append(dt[order[-k:]].mean())

    ric_arr = np.array(rank_ic_vals)
    pic_arr = np.array(pearson_ic_vals)

    print(f"\n  Full-Universe Ranking Quality:")
    print(f"    Spearman Rank IC:  mean={ric_arr.mean():.4f}  std={ric_arr.std():.4f}  "
          f"ICIR={ric_arr.mean()/ric_arr.std():.3f}  t={ric_arr.mean()/ric_arr.std()*np.sqrt(len(ric_arr)):.2f}")
    print(f"    Pearson IC:        mean={pic_arr.mean():.4f}  std={pic_arr.std():.4f}  "
          f"ICIR={pic_arr.mean()/pic_arr.std():.3f}")
    print(f"    IC>0 ratio:        {(ric_arr>0).mean():.1%}")

    print(f"\n  Rank IC by Year:")
    for yr in sorted(ic_by_year.keys()):
        arr = np.array(ic_by_year[yr])
        print(f"    {yr}: mean={arr.mean():.4f}  ICIR={arr.mean()/arr.std():.3f}  "
              f"IC>0={float((arr>0).mean()):.1%}  n={len(arr)}")

    print(f"\n  Top-K / Bottom-K Spread (5d forward returns):")
    for k in [5, 10, 20]:
        t = np.array(topk_rets[k])
        b = np.array(btm_rets[k])
        spread = t - b
        ann_sharpe = float(spread.mean() / spread.std() * np.sqrt(252 / HORIZON)) if spread.std() > 0 else 0
        print(f"    K={k:>2}:  Top={t.mean():+.4f}  Bot={b.mean():+.4f}  "
              f"Spread={spread.mean():+.4f}  SpreadSharpe={ann_sharpe:.3f}  WR={float((t>0).mean()):.1%}")

    # ─── 4. Download prices & compute regimes ───
    print(f"\n[4/7] Building top-K holdings & downloading price data...")

    # Extract top-20 tickers per OOS date (max K we test)
    topk_by_date = {}
    for d in unique_oos_dates:
        mask = dates_cat == d
        dp = preds_cat[mask]
        dt_tickers = tickers_cat[mask]
        if len(dp) < 20:
            continue
        order = np.argsort(-dp)[:20]  # store top-20, slice later
        topk_by_date[d] = list(dt_tickers[order])

    all_top_tickers = set()
    for tks in topk_by_date.values():
        all_top_tickers.update(tks)
    print(f"  Unique tickers in top-20 across OOS: {len(all_top_tickers)}")

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

    print(f"  Computing HMM p_risk from SPY...")
    p_risk = compute_hmm_prisk(spy_px)

    prisk_bt = p_risk.loc[p_risk.index >= pd.Timestamp(unique_oos_dates[0])]
    print(f"    p_risk stats (OOS): mean={prisk_bt.mean():.3f} "
          f">0.5={float((prisk_bt>0.5).mean()):.1%} >0.9={float((prisk_bt>0.9).mean()):.1%}")

    oos_trading_dates = px.index[px.index >= pd.Timestamp(unique_oos_dates[0])]
    oos_trading_dates = oos_trading_dates[oos_trading_dates <= pd.Timestamp(unique_oos_dates[-1])]

    # ─── 5. EMA SMOOTHING GRID SEARCH ───
    # Fixed: K=10, MA200 overlay, 2% SL, 5d rebalance
    RB = REBALANCE_DAYS  # 5
    SL = 0.02
    K = 10

    print(f"\n[5/6] EMA Smoothing Grid Search")
    print(f"  Fixed: K={K}, MA200=ON, SL={SL*100:.0f}%, {RB}d rebal, {COST_BPS}bps")
    print(f"  5d non-overlapping Sharpe/Vol/MaxDD")

    # ── Helper: apply per-ticker EMA to predictions ──
    def apply_ema_per_ticker(raw_preds, dates_arr, tickers_arr, span):
        """EMA smooth predictions per ticker over time. alpha = 2/(span+1)."""
        if span <= 1:
            return raw_preds.copy()
        alpha = 2.0 / (span + 1)
        smoothed = raw_preds.copy()
        ticker_ema = {}  # {ticker: last_ema_value}
        for d in np.sort(np.unique(dates_arr)):
            mask = dates_arr == d
            idxs = np.where(mask)[0]
            for idx in idxs:
                tk = tickers_arr[idx]
                raw = raw_preds[idx]
                if tk in ticker_ema:
                    new_val = alpha * raw + (1 - alpha) * ticker_ema[tk]
                else:
                    new_val = raw
                smoothed[idx] = new_val
                ticker_ema[tk] = new_val
        return smoothed

    # ── Helper: blend raw + smoothed ──
    def blend_preds(raw_preds, smoothed_preds, beta):
        """final = beta*raw + (1-beta)*smoothed"""
        return beta * raw_preds + (1 - beta) * smoothed_preds

    # ── Helper: build top-K from predictions ──
    def build_topk(preds_arr, dates_arr, tickers_arr, unique_dates, top_n=20):
        topk = {}
        for d in unique_dates:
            mask = dates_arr == d
            dp = preds_arr[mask]
            dt_tk = tickers_arr[mask]
            if len(dp) < top_n:
                continue
            order = np.argsort(-dp)[:top_n]
            topk[d] = list(dt_tk[order])
        return topk

    # ── Helper: compute stability metrics ──
    def compute_stability(topk_dict, k=10):
        sorted_d = sorted(topk_dict.keys())
        overlaps, turnovers = [], []
        for i in range(1, len(sorted_d)):
            prev = set(topk_dict[sorted_d[i-1]][:k])
            curr = set(topk_dict[sorted_d[i]][:k])
            overlaps.append(len(prev & curr) / k)
            turnovers.append(len(curr - prev))
        return (np.mean(overlaps) if overlaps else 0,
                np.mean(turnovers) if turnovers else 0)

    # ── Helper: run one test config and return full metrics ──
    def run_test(label, use_preds, use_dates, use_tickers, use_unique_dates):
        tkbd = build_topk(use_preds, use_dates, use_tickers, use_unique_dates)
        # Need to collect all tickers for price download — but we already have px
        eq_ovl, eq_raw, eq_dt, dr_ovl, dr_raw, sim_stats = simulate_portfolio(
            oos_trading_dates, tkbd, px, spy_px, spy_ma200, p_risk,
            use_hmm=False, use_ma200=True, top_k=K,
            cost_bps=COST_BPS, rebal_days=RB, stop_loss_pct=SL)

        m = calc_metrics_5d(eq_ovl, rebal_days=RB)
        maxdd = calc_maxdd_5d(eq_raw, rebal_days=RB)
        calmar = m['CAGR'] / abs(maxdd) if abs(maxdd) > 0 else 0

        # Stability
        overlap_rate, avg_turnover = compute_stability(tkbd, k=K)

        # Rebalance trade count (approximate: turnover * n_rebalances)
        n_rebalances = len(eq_dt) // RB
        total_rebal_trades = int(avg_turnover * n_rebalances)

        # Bull/bear Sharpe split (using MA200)
        bull_rets, bear_rets = [], []
        for j, dt in enumerate(eq_dt[1:]):
            if dt in spy_px.index and dt in spy_ma200.index:
                sp = float(spy_px.loc[dt])
                ma = float(spy_ma200.loc[dt])
                if not np.isnan(sp) and not np.isnan(ma) and ma > 0:
                    if sp >= ma:
                        bull_rets.append(dr_ovl[j])
                    else:
                        bear_rets.append(dr_ovl[j])

        def _sharpe_from_daily(rets):
            if len(rets) < 10:
                return 0.0
            arr = np.array(rets)
            # Sample every RB days for non-overlapping
            sampled = arr[::RB]
            if len(sampled) < 2:
                return 0.0
            ppyr = 252 / RB
            return float(np.mean(sampled) / np.std(sampled, ddof=1) * np.sqrt(ppyr)) if np.std(sampled, ddof=1) > 0 else 0.0

        bull_sharpe = _sharpe_from_daily(bull_rets)
        bear_sharpe = _sharpe_from_daily(bear_rets)

        return {
            'label': label,
            'CAGR': m['CAGR'], 'MaxDD': maxdd, 'Sharpe': m['Sharpe'],
            'Calmar': calmar, 'Vol': m['Vol'],
            'overlap': overlap_rate, 'avg_turnover': avg_turnover,
            'n_stops': sim_stats['n_stops'], 'rebal_trades': total_rebal_trades,
            'bull_sharpe': bull_sharpe, 'bear_sharpe': bear_sharpe,
        }

    # ── Print table row ──
    def print_row(r):
        print(f"  {r['label']:<22} {r['Sharpe']:>7.3f} {r['CAGR']:>+9.2%} "
              f"{r['MaxDD']:>+7.1%} {r['Calmar']:>8.3f} "
              f"{r['overlap']:>6.1%} {r['avg_turnover']:>5.1f} "
              f"{r['n_stops']:>5} {r['rebal_trades']:>5} "
              f"{r['bull_sharpe']:>7.3f} {r['bear_sharpe']:>7.3f}")

    header = (f"  {'Config':<22} {'Sharpe':>7} {'CAGR':>9} "
              f"{'MaxDD':>7} {'Calmar':>8} "
              f"{'Ovlp%':>6} {'Turn':>5} "
              f"{'Stops':>5} {'Rebal':>5} "
              f"{'BullSh':>7} {'BearSh':>7}")
    divider = "  " + "-" * 102

    # ═══════════════════════════════════════════════════════
    # PART A: EMA Span Sweep (raw baseline + 6 spans)
    # ═══════════════════════════════════════════════════════
    print(f"\n  {'='*102}")
    print(f"  PART A: EMA SPAN SWEEP  (find best smoothing window)")
    print(f"  {'='*102}")
    print(header)
    print(divider)

    # Baseline: raw predictions (span=1)
    raw_result = run_test("Raw (no EMA)", preds_cat, dates_cat, tickers_cat, unique_oos_dates)
    print_row(raw_result)

    span_results = {'Raw': raw_result}
    best_span = 1
    best_sharpe = raw_result['Sharpe']

    for span in [2, 3, 4, 5, 7, 10]:
        alpha = 2.0 / (span + 1)
        ema_preds = apply_ema_per_ticker(preds_cat, dates_cat, tickers_cat, span)
        label = f"EMA({span}), a={alpha:.2f}"
        r = run_test(label, ema_preds, dates_cat, tickers_cat, unique_oos_dates)
        print_row(r)
        span_results[f"EMA({span})"] = r
        if r['Sharpe'] > best_sharpe:
            best_sharpe = r['Sharpe']
            best_span = span

    print(f"\n  >>> Part A winner: {'Raw' if best_span == 1 else f'EMA({best_span})'} "
          f"(Sharpe={best_sharpe:.3f})")

    # ═══════════════════════════════════════════════════════
    # PART B: Beta Mixing (raw vs smoothed)
    # ═══════════════════════════════════════════════════════
    if best_span > 1:
        ema_winner = apply_ema_per_ticker(preds_cat, dates_cat, tickers_cat, best_span)
    else:
        # If raw won, try span=3 as the smooth component for blending test
        best_span_b = 3
        ema_winner = apply_ema_per_ticker(preds_cat, dates_cat, tickers_cat, best_span_b)
        print(f"\n  (Raw won Part A; using EMA({best_span_b}) as smooth component for Part B)")
        best_span = best_span_b

    print(f"\n  {'='*102}")
    print(f"  PART B: BETA MIXING  (beta*raw + (1-beta)*EMA({best_span}))")
    print(f"  {'='*102}")
    print(header)
    print(divider)

    best_beta = 1.0
    best_beta_sharpe = raw_result['Sharpe']
    beta_results = {}

    for beta in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        blended = blend_preds(preds_cat, ema_winner, beta)
        label = f"beta={beta:.1f}"
        if beta == 1.0:
            label += " (pure raw)"
        elif beta == 0.0:
            label += f" (pure EMA({best_span}))"
        r = run_test(label, blended, dates_cat, tickers_cat, unique_oos_dates)
        print_row(r)
        beta_results[beta] = r
        if r['Sharpe'] > best_beta_sharpe:
            best_beta_sharpe = r['Sharpe']
            best_beta = beta

    print(f"\n  >>> Part B winner: beta={best_beta:.1f} (Sharpe={best_beta_sharpe:.3f})")
    print(f"  >>> Optimal pair: (span={best_span}, beta={best_beta:.1f})")

    # ═══════════════════════════════════════════════════════
    # PART C: YEARLY + QUARTERLY BREAKDOWN for ALL VARIANTS
    # ═══════════════════════════════════════════════════════
    print(f"\n  {'='*102}")
    print(f"  PART C: YEARLY BREAKDOWN — ALL VARIANTS  (2022 bear + 2025 tariff focus)")
    print(f"  {'='*102}")

    # Build all variant prediction arrays
    all_variants = [("Raw", preds_cat)]
    for sp in [2, 4, 7, 10]:
        ema_p = apply_ema_per_ticker(preds_cat, dates_cat, tickers_cat, sp)
        all_variants.append((f"EMA({sp})", ema_p))

    # Run full sim for each variant and collect yearly + quarterly equity
    variant_yearly = {}  # {name: {year: {CAGR, MaxDD, Sharpe, Calmar}}}
    variant_quarterly = {}  # {name: {(year,q): return_pct}}

    for vname, vpreds in all_variants:
        tkbd = build_topk(vpreds, dates_cat, tickers_cat, unique_oos_dates)
        eq_ovl, eq_raw, eq_dt, dr_ovl, dr_raw, sim_stats_v = simulate_portfolio(
            oos_trading_dates, tkbd, px, spy_px, spy_ma200, p_risk,
            use_hmm=False, use_ma200=True, top_k=K,
            cost_bps=COST_BPS, rebal_days=RB, stop_loss_pct=SL)

        years_in = sorted(set(d.year for d in eq_dt))
        variant_yearly[vname] = {}
        variant_quarterly[vname] = {}

        for yr in years_in:
            yr_mask = np.array([d.year == yr for d in eq_dt])
            yr_idx = np.where(yr_mask)[0]
            if len(yr_idx) < 10:
                continue
            yr_m = calc_metrics_5d(eq_ovl[yr_idx], rebal_days=RB)
            yr_maxdd = calc_maxdd_5d(eq_raw[yr_idx], rebal_days=RB)
            yr_cal = yr_m['CAGR'] / abs(yr_maxdd) if abs(yr_maxdd) > 0 else 0
            variant_yearly[vname][yr] = {
                'CAGR': yr_m['CAGR'], 'MaxDD': yr_maxdd,
                'Sharpe': yr_m['Sharpe'], 'Calmar': yr_cal
            }

            # Quarterly breakdown for 2022 and 2025
            if yr in (2022, 2025):
                for q, (m_start, m_end) in enumerate([(1,3),(4,6),(7,9),(10,12)], 1):
                    q_mask = np.array([d.year == yr and m_start <= d.month <= m_end for d in eq_dt])
                    q_idx = np.where(q_mask)[0]
                    if len(q_idx) < 5:
                        variant_quarterly[vname][(yr, q)] = float('nan')
                        continue
                    q_eq = eq_ovl[q_idx]
                    q_ret = (q_eq[-1] / q_eq[0]) - 1.0
                    variant_quarterly[vname][(yr, q)] = q_ret

    # Print yearly table
    years_all = sorted(set(yr for d in variant_yearly.values() for yr in d))
    print(f"\n  {'Config':<12}", end='')
    for yr in years_all:
        print(f"  {yr:>8s}      ", end='')
    print()
    print(f"  {'':12}", end='')
    for yr in years_all:
        print(f"  {'CAGR':>8s} {'MDD':>6s}", end='')
    print()
    print("  " + "-" * (12 + 16 * len(years_all)))

    for vname, _ in all_variants:
        line = f"  {vname:<12}"
        for yr in years_all:
            d = variant_yearly[vname].get(yr)
            if d:
                line += f"  {d['CAGR']:>+7.1%} {d['MaxDD']:>+5.1%}"
            else:
                line += f"  {'N/A':>8} {'N/A':>6}"
        print(line)

    # Print yearly Sharpe table
    print(f"\n  Sharpe by year:")
    print(f"  {'Config':<12}", end='')
    for yr in years_all:
        print(f"  {yr:>8d}", end='')
    print()
    print("  " + "-" * (12 + 10 * len(years_all)))

    for vname, _ in all_variants:
        line = f"  {vname:<12}"
        for yr in years_all:
            d = variant_yearly[vname].get(yr)
            if d:
                line += f"  {d['Sharpe']:>8.3f}"
            else:
                line += f"  {'N/A':>8}"
        print(line)

    # Print 2022 quarterly breakdown
    print(f"\n  2022 Quarterly (Bear Market):")
    print(f"  {'Config':<12}  {'Q1':>8}  {'Q2':>8}  {'Q3':>8}  {'Q4':>8}")
    print("  " + "-" * 52)
    for vname, _ in all_variants:
        line = f"  {vname:<12}"
        for q in range(1, 5):
            val = variant_quarterly[vname].get((2022, q), float('nan'))
            if np.isnan(val):
                line += f"  {'N/A':>8}"
            else:
                line += f"  {val:>+7.1%}"
        print(line)

    # Print 2025 quarterly breakdown
    print(f"\n  2025 Quarterly (Tariff Shock):")
    print(f"  {'Config':<12}  {'Q1':>8}  {'Q2 tariff':>10}  {'Q3':>8}  {'Q4':>8}")
    print("  " + "-" * 56)
    for vname, _ in all_variants:
        line = f"  {vname:<12}"
        for q in range(1, 5):
            val = variant_quarterly[vname].get((2025, q), float('nan'))
            if np.isnan(val):
                line += f"  {'N/A':>10}" if q == 2 else f"  {'N/A':>8}"
            else:
                if q == 2:
                    line += f"  {val:>+9.1%}"
                else:
                    line += f"  {val:>+7.1%}"
        print(line)

    # ─── 6. Summary ───
    print(f"\n[6/6] Final Summary")
    print(f"\n  Optimal config: K={K}, MA200=ON, SL={SL*100:.0f}%, "
          f"EMA span={best_span}, beta={best_beta:.1f}")
    w = beta_results.get(best_beta, raw_result)
    print(f"  Sharpe={w['Sharpe']:.3f}  CAGR={w['CAGR']:+.2%}  MaxDD={w['MaxDD']:+.1%}  "
          f"Calmar={w['Calmar']:.3f}")
    print(f"  Overlap={w['overlap']:.1%}  AvgTurnover={w['avg_turnover']:.1f}/rebal  "
          f"Stops={w['n_stops']}  BullSharpe={w['bull_sharpe']:.3f}  BearSharpe={w['bear_sharpe']:.3f}")

    print(f"\n{sep}")
    print("  DONE")
    print(sep)


if __name__ == '__main__':
    main()
