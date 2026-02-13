#!/usr/bin/env python3
"""
LambdaRank + Regime Fusion — 6-Test Walk-Forward Backtest
==========================================================

Tests:
  T0: Baseline (15 stock-level features only)
  T1: Full Regime (+ 7 regime features)
  T2: Regime-conditioned evaluation (T1 model, split by bull/bear)
  T3: Minimal Regime (+ 2 features: regime_vix, regime_spy_ma200_dev)
  T4: Regime-Gated portfolio (best model + SPY>MA200 gate)
  T5: Interaction features (stock_beta × VIX, stock_mom × SPY_above_MA)

Uses identical walk-forward, purged CV, and evaluation as the production
walkforward_lambdarank.py pipeline.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── constants from production pipeline ────────────────────────────

FEATURES_BASE = [
    "volume_price_corr_3d", "rsi_14", "reversal_3d", "momentum_10d",
    "liquid_momentum_10d", "sharpe_momentum_5d", "price_ma20_deviation",
    "avg_trade_size", "trend_r2_20", "dollar_vol_20", "ret_skew_20d",
    "reversal_5d", "near_52w_high", "atr_pct_14", "amihud_20",
]

FEATURES_REGIME_FULL = [
    "regime_vix", "regime_vix_20d_chg", "regime_spy_ma200_dev",
    "regime_spy_above_ma", "regime_hvr", "regime_spy_mom_1m",
    "regime_spy_dd_20d",
]

FEATURES_REGIME_MINIMAL = [
    "regime_vix", "regime_spy_ma200_dev",
]

FEATURES_INTERACTION = [
    "interact_beta_x_vix", "interact_mom_x_bull",
]

BEST_PARAMS = {
    "learning_rate": 0.04,
    "num_leaves": 11,
    "max_depth": 3,
    "min_data_in_leaf": 350,
    "lambda_l2": 120,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.70,
    "bagging_freq": 1,
    "min_gain_to_split": 0.30,
    "lambdarank_truncation_level": 25,
    "sigmoid": 1.1,
    "label_gain_power": 2.1,
}
N_QUANTILES = 64
HORIZON = 5
MIN_TRAIN_DAYS = 252
TEST_WINDOW = 63
STEP_SIZE = 63
CV_SPLITS = 5
N_BOOST = 800
SEED = 0


# ── helpers (identical to production) ────────────────────────────

def ensure_multiindex(df):
    if isinstance(df.index, pd.MultiIndex) and {"date", "ticker"}.issubset(df.index.names):
        return df.sort_index()
    if {"date", "ticker"}.issubset(df.columns):
        return df.set_index(["date", "ticker"]).sort_index()
    raise ValueError("Need date/ticker columns or MultiIndex")


def build_quantile_labels(y, dates):
    labels = np.zeros(len(y), dtype=np.int32)
    for d in np.unique(dates):
        mask = dates == d
        if np.sum(mask) <= 1:
            continue
        vals = y[mask]
        ranks = sp_stats.rankdata(vals, method="average")
        q = np.floor(ranks / (len(vals) + 1) * N_QUANTILES).astype(np.int32)
        labels[mask] = np.clip(q, 0, N_QUANTILES - 1)
    return labels


def group_counts(dates):
    return [int(np.sum(dates == d)) for d in np.unique(dates)]


def purged_cv_splits(dates, n_splits, gap, embargo):
    unique = np.unique(dates)
    n = len(unique)
    fold = max(1, n // n_splits)
    for i in range(n_splits):
        vs = i * fold
        ve = n if i == n_splits - 1 else (i + 1) * fold
        te = max(0, vs - gap)
        es = min(n, ve + embargo)
        td = np.concatenate((unique[:te], unique[es:]))
        vd = unique[vs:ve]
        tm = np.isin(dates, td)
        vm = np.isin(dates, vd)
        if np.sum(tm) < 100 or np.sum(vm) < 50:
            continue
        yield np.where(tm)[0], np.where(vm)[0]


def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = cum / running_max - 1
    return float(dd.min())


# ── training ─────────────────────────────────────────────────────

def train_model(train_df, features):
    X = train_df[features].fillna(0.0).to_numpy()
    y = train_df["target"].to_numpy()
    dates = train_df.index.get_level_values("date").to_numpy()
    labels = build_quantile_labels(y, dates)

    label_gain = [
        (i / (N_QUANTILES - 1)) ** BEST_PARAMS["label_gain_power"] * (N_QUANTILES - 1)
        for i in range(N_QUANTILES)
    ]
    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 20],
        "learning_rate": BEST_PARAMS["learning_rate"],
        "num_leaves": BEST_PARAMS["num_leaves"],
        "max_depth": BEST_PARAMS["max_depth"],
        "min_data_in_leaf": BEST_PARAMS["min_data_in_leaf"],
        "lambda_l1": 0.0,
        "lambda_l2": BEST_PARAMS["lambda_l2"],
        "feature_fraction": BEST_PARAMS["feature_fraction"],
        "bagging_fraction": BEST_PARAMS["bagging_fraction"],
        "bagging_freq": BEST_PARAMS["bagging_freq"],
        "min_gain_to_split": BEST_PARAMS["min_gain_to_split"],
        "lambdarank_truncation_level": BEST_PARAMS["lambdarank_truncation_level"],
        "sigmoid": BEST_PARAMS["sigmoid"],
        "label_gain": label_gain,
        "verbose": -1,
        "force_row_wise": True,
        "seed": SEED,
        "bagging_seed": SEED,
        "feature_fraction_seed": SEED,
        "data_random_seed": SEED,
        "deterministic": True,
    }

    best_rounds = []
    for ti, vi in purged_cv_splits(dates, CV_SPLITS, HORIZON, HORIZON):
        ts = lgb.Dataset(X[ti], label=labels[ti], group=group_counts(dates[ti]))
        vs = lgb.Dataset(X[vi], label=labels[vi], group=group_counts(dates[vi]))
        bst = lgb.train(lgb_params, ts, num_boost_round=N_BOOST,
                        valid_sets=[vs],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        best_rounds.append(bst.best_iteration or N_BOOST)

    final_rounds = int(np.mean(best_rounds)) if best_rounds else N_BOOST
    full = lgb.Dataset(X, label=labels, group=group_counts(dates))
    model = lgb.train(lgb_params, full, num_boost_round=final_rounds)
    return model, final_rounds


# ── walk-forward ─────────────────────────────────────────────────

def walk_forward(df, features, label=""):
    all_dates = df.index.get_level_values("date").unique().sort_values()
    n_dates = len(all_dates)

    first_test_idx = MIN_TRAIN_DAYS + HORIZON
    if first_test_idx >= n_dates:
        raise ValueError("Not enough data")

    results = []
    step = 0
    test_start_idx = first_test_idx

    while test_start_idx < n_dates:
        test_end_idx = min(test_start_idx + TEST_WINDOW, n_dates)
        train_end_idx = test_start_idx - HORIZON

        train_dates = all_dates[:train_end_idx]
        test_dates_slice = all_dates[test_start_idx:test_end_idx]

        train_df = df.loc[(train_dates, slice(None)), :]
        test_df = df.loc[(test_dates_slice, slice(None)), :]

        print(f"    Step {step}: train {train_dates[0].date()}→{train_dates[-1].date()} "
              f"({len(train_dates)}d) | test {test_dates_slice[0].date()}"
              f"→{test_dates_slice[-1].date()} ({len(test_dates_slice)}d)",
              end="", flush=True)

        model, rounds = train_model(train_df, features)
        print(f"  [{rounds}r]", flush=True)

        X_test = test_df[features].fillna(0.0).to_numpy()
        preds = model.predict(X_test)

        res = test_df[["target"]].copy()
        res["pred"] = preds
        res["wf_step"] = step

        # Store feature importance from last model
        if step == 0:
            fi = dict(zip(features, model.feature_importance(importance_type="gain")))

        results.append(res)
        step += 1
        test_start_idx += STEP_SIZE

    pred_df = pd.concat(results)

    # Get feature importance from the last trained model
    fi_last = dict(zip(features, model.feature_importance(importance_type="gain")))

    return pred_df, fi_last


# ── evaluation ───────────────────────────────────────────────────

def evaluate(pred_df, label="", date_mask=None):
    """Compute all metrics. If date_mask given, filter to those dates only."""
    dates = pred_df.index.get_level_values("date")
    unique_dates = np.sort(dates.unique())

    if date_mask is not None:
        unique_dates = unique_dates[np.isin(unique_dates, date_mask)]

    overlap_top10 = []
    overlap_top10_20 = []
    overlap_bot10 = []
    non_overlap_top10 = []
    non_overlap_top10_20 = []
    non_overlap_dates = unique_dates[::max(1, HORIZON)]

    for d in unique_dates:
        try:
            day = pred_df.loc[d]
        except KeyError:
            continue
        if len(day) < 20:
            continue
        order = np.argsort(-day["pred"].values)
        tgts = day["target"].values

        r_top10 = float(tgts[order[:10]].mean())
        r_top10_20 = float(tgts[order[10:20]].mean())
        r_bot10 = float(tgts[order[-10:]].mean())

        overlap_top10.append(r_top10)
        overlap_top10_20.append(r_top10_20)
        overlap_bot10.append(r_bot10)

        if d in non_overlap_dates:
            non_overlap_top10.append(r_top10)
            non_overlap_top10_20.append(r_top10_20)

    ot10 = np.array(overlap_top10)
    ot1020 = np.array(overlap_top10_20)
    ob10 = np.array(overlap_bot10)
    no10 = np.array(non_overlap_top10)
    no1020 = np.array(non_overlap_top10_20)

    def sharpe(r, freq=1):
        if len(r) < 2 or np.std(r, ddof=1) == 0:
            return np.nan
        return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252.0 / freq))

    def safe_mean(a):
        return float(np.mean(a)) if len(a) > 0 else np.nan

    # NDCG@10 (approximate: fraction of top-10 by pred that are in actual top-10)
    ndcg_vals = []
    for d in unique_dates:
        try:
            day = pred_df.loc[d]
        except KeyError:
            continue
        if len(day) < 20:
            continue
        pred_order = np.argsort(-day["pred"].values)
        true_order = np.argsort(-day["target"].values)
        pred_top10 = set(pred_order[:10])
        true_top10 = set(true_order[:10])
        ndcg_vals.append(len(pred_top10 & true_top10) / 10.0)

    # IC
    ic_vals = []
    for d in unique_dates:
        try:
            day = pred_df.loc[d]
        except KeyError:
            continue
        if len(day) < 20:
            continue
        ic = float(sp_stats.spearmanr(day["pred"].values, day["target"].values).statistic)
        if np.isfinite(ic):
            ic_vals.append(ic)

    # t-test
    if len(ot10) >= 3:
        t_stat, p_val = sp_stats.ttest_1samp(ot10, 0)
    else:
        t_stat, p_val = np.nan, np.nan

    results = {
        "n_dates": len(unique_dates),
        "overlap_top10_mean": safe_mean(ot10),
        "overlap_top10_median": float(np.median(ot10)) if len(ot10) > 0 else np.nan,
        "overlap_top10_winrate": float(np.mean(ot10 > 0)) if len(ot10) > 0 else np.nan,
        "overlap_top10_sharpe": sharpe(ot10, 1),
        "overlap_bot10_mean": safe_mean(ob10),
        "long_short_spread": safe_mean(ot10) - safe_mean(ob10),
        "nonoverlap_top10_mean": safe_mean(no10),
        "nonoverlap_top10_sharpe": sharpe(no10, HORIZON),
        "nonoverlap_top10_maxdd": max_drawdown(no10) if len(no10) > 0 else np.nan,
        "nonoverlap_top10_20_mean": safe_mean(no1020),
        "nonoverlap_top10_20_sharpe": sharpe(no1020, HORIZON),
        "ndcg10_approx": safe_mean(np.array(ndcg_vals)),
        "IC_mean": safe_mean(np.array(ic_vals)),
        "IC_std": float(np.std(ic_vals)) if len(ic_vals) > 1 else np.nan,
        "IC_IR": (safe_mean(np.array(ic_vals)) / float(np.std(ic_vals))
                  if len(ic_vals) > 1 and np.std(ic_vals) > 0 else np.nan),
        "tstat": float(t_stat),
        "pvalue": float(p_val),
    }
    return results


# ── regime feature engineering ───────────────────────────────────

def add_regime_features(df, spy_close, vix_close):
    """Add 7 regime features. SPY/VIX are pd.Series indexed by date."""
    print("  Computing regime features...", flush=True)
    dates = df.index.get_level_values("date").unique().sort_values()

    spy_ma200 = spy_close.rolling(200, min_periods=200).mean()
    spy_ret = spy_close.pct_change()
    spy_std_20 = spy_ret.rolling(20).std()
    spy_std_60 = spy_ret.rolling(60).std()

    regime_data = {}
    for d in dates:
        if d not in spy_close.index or d not in vix_close.index:
            regime_data[d] = {k: 0.0 for k in FEATURES_REGIME_FULL}
            continue

        spy_c = float(spy_close.loc[:d].iloc[-1])
        vix_c = float(vix_close.loc[:d].iloc[-1])
        ma200_val = float(spy_ma200.loc[:d].iloc[-1]) if d in spy_ma200.index else spy_c

        # VIX 20d change
        vix_hist = vix_close.loc[:d]
        if len(vix_hist) >= 20:
            vix_20d_chg = float(vix_hist.iloc[-1] / vix_hist.iloc[-20] - 1.0)
        else:
            vix_20d_chg = 0.0

        # SPY MA200 deviation
        spy_ma200_dev = (spy_c - ma200_val) / ma200_val if ma200_val > 0 else 0.0

        # SPY above MA200
        spy_above = 1.0 if spy_c > ma200_val else 0.0

        # HVR (20d vol / 60d vol)
        v20 = float(spy_std_20.loc[:d].iloc[-1]) if d in spy_std_20.index else 0.01
        v60 = float(spy_std_60.loc[:d].iloc[-1]) if d in spy_std_60.index else 0.01
        hvr = v20 / v60 if v60 > 0 else 1.0

        # SPY 1m momentum
        spy_hist = spy_close.loc[:d]
        if len(spy_hist) >= 21:
            spy_mom_1m = float(spy_hist.iloc[-1] / spy_hist.iloc[-21] - 1.0)
        else:
            spy_mom_1m = 0.0

        # SPY 20d max drawdown
        if len(spy_hist) >= 20:
            spy_20d = spy_hist.iloc[-20:]
            spy_dd_20d = float(spy_20d.min() / spy_20d.max() - 1.0)
        else:
            spy_dd_20d = 0.0

        regime_data[d] = {
            "regime_vix": vix_c,
            "regime_vix_20d_chg": vix_20d_chg,
            "regime_spy_ma200_dev": spy_ma200_dev,
            "regime_spy_above_ma": spy_above,
            "regime_hvr": hvr,
            "regime_spy_mom_1m": spy_mom_1m,
            "regime_spy_dd_20d": spy_dd_20d,
        }

    regime_df = pd.DataFrame.from_dict(regime_data, orient="index")
    regime_df.index.name = "date"

    # Join: each ticker on the same date gets the same regime features
    df = df.join(regime_df, on="date")
    print(f"  Added {len(FEATURES_REGIME_FULL)} regime features to {len(dates)} dates")
    return df


def add_interaction_features(df):
    """Add interaction features for Test 5."""
    # stock_beta proxy: use momentum_10d as beta-like exposure measure
    # (high recent momentum ≈ high market sensitivity in short term)
    # interact_beta_x_vix: stocks with high recent momentum + high VIX = dangerous
    df["interact_beta_x_vix"] = df["momentum_10d"].fillna(0) * df["regime_vix"].fillna(20)

    # interact_mom_x_bull: momentum should work better in bull regime
    df["interact_mom_x_bull"] = (df["liquid_momentum_10d"].fillna(0) *
                                  df["regime_spy_above_ma"].fillna(0.5))
    return df


# ── regime-gated portfolio simulation (Test 4) ──────────────────

def regime_gated_backtest(pred_df, spy_close, spy_ma200):
    """
    Simulate: buy top-10 only when SPY > MA200, else cash.
    Returns dict with portfolio-level metrics.
    """
    dates = pred_df.index.get_level_values("date").unique().sort_values()
    non_overlap_dates = dates[::max(1, HORIZON)]

    gated_returns = []
    ungated_returns = []
    gated_actions = {"trade": 0, "cash": 0}

    for d in non_overlap_dates:
        try:
            day = pred_df.loc[d]
        except KeyError:
            continue
        if len(day) < 20:
            continue

        order = np.argsort(-day["pred"].values)
        r_top10 = float(day["target"].values[order[:10]].mean())
        ungated_returns.append(r_top10)

        # Check SPY vs MA200 on this date
        if d in spy_close.index and d in spy_ma200.index:
            spy_val = float(spy_close.loc[d])
            ma_val = float(spy_ma200.loc[d])
            is_bull = spy_val > ma_val
        else:
            is_bull = True  # default to invested if no data

        if is_bull:
            gated_returns.append(r_top10)
            gated_actions["trade"] += 1
        else:
            gated_returns.append(0.0)  # cash
            gated_actions["cash"] += 1

    gated = np.array(gated_returns)
    ungated = np.array(ungated_returns)

    def sharpe(r, freq=HORIZON):
        if len(r) < 2 or np.std(r, ddof=1) == 0:
            return np.nan
        return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252.0 / freq))

    def cum_ret(r):
        return float(np.prod(1 + r) - 1)

    return {
        "ungated_cum_ret": cum_ret(ungated),
        "ungated_sharpe": sharpe(ungated),
        "ungated_maxdd": max_drawdown(ungated),
        "gated_cum_ret": cum_ret(gated),
        "gated_sharpe": sharpe(gated),
        "gated_maxdd": max_drawdown(gated),
        "n_trade": gated_actions["trade"],
        "n_cash": gated_actions["cash"],
        "pct_invested": gated_actions["trade"] / max(1, sum(gated_actions.values())),
    }


# ── pretty-print helpers ─────────────────────────────────────────

def print_metrics(m, label):
    print(f"  {label}")
    print(f"    NDCG@10 (approx):  {m['ndcg10_approx']:.4f}")
    print(f"    Top-10 mean ret:   {m['overlap_top10_mean']:.6f}")
    print(f"    Bot-10 mean ret:   {m['overlap_bot10_mean']:.6f}")
    print(f"    L/S spread:        {m['long_short_spread']:.6f}")
    print(f"    Top-10 Sharpe(OL): {m['overlap_top10_sharpe']:.3f}")
    print(f"    Top-10 Sharpe(NO): {m['nonoverlap_top10_sharpe']:.3f}")
    print(f"    Top-10 MaxDD(NO):  {m['nonoverlap_top10_maxdd']:.4f}")
    print(f"    Win rate:          {m['overlap_top10_winrate']:.4f}")
    print(f"    IC mean:           {m['IC_mean']:.4f}  std={m['IC_std']:.4f}  IR={m['IC_IR']:.4f}")
    sig = "***" if m["pvalue"] < 0.001 else "**" if m["pvalue"] < 0.01 else "*" if m["pvalue"] < 0.05 else ""
    print(f"    t-stat:            {m['tstat']:.3f}  p={m['pvalue']:.2e} {sig}")
    print(f"    N dates:           {m['n_dates']}")


def print_feature_importance(fi, label, top_n=20):
    print(f"\n  Feature Importance ({label}, top {top_n}):")
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    for i, (f, v) in enumerate(sorted_fi[:top_n]):
        regime_tag = " [REGIME]" if f.startswith("regime_") else ""
        inter_tag = " [INTERACT]" if f.startswith("interact_") else ""
        print(f"    {i+1:>2}. {f:<30} {v:>10.1f}{regime_tag}{inter_tag}")


# ── main ─────────────────────────────────────────────────────────

def main():
    np.random.seed(SEED)
    feature_file = Path("D:/trade/data/factor_exports/polygon_full_features_T5.parquet")
    hmm_file = Path("D:/trade/results/walkforward_lambdarank/hmm_crisis_log.csv")
    output_dir = Path("D:/trade/results/lambdarank_regime_fusion")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("  LambdaRank + Regime Fusion — 6-Test Walk-Forward Backtest")
    print("=" * 120)

    # ── Load data ──
    print("\n[1] Loading data...")
    df = pd.read_parquet(feature_file)
    df = ensure_multiindex(df)
    n_nan = df["target"].isna().sum()
    if n_nan > 0:
        df = df.dropna(subset=["target"])
    all_dates = df.index.get_level_values("date").unique().sort_values()
    print(f"  Features: {len(all_dates)} dates, {len(df)} rows, {len(FEATURES_BASE)} base features")

    # ── Fetch SPY + VIX via yfinance ──
    print("\n[2] Fetching SPY + VIX (yfinance)...")
    import yfinance as yf
    spy_df = yf.download("SPY", start="2020-01-01", end="2026-02-11", progress=False, auto_adjust=True)
    vix_df = yf.download("^VIX", start="2020-01-01", end="2026-02-11", progress=False, auto_adjust=True)

    spy_close = spy_df["Close"].squeeze()
    vix_close = vix_df["Close"].squeeze()
    spy_close.index = spy_close.index.tz_localize(None)
    vix_close.index = vix_close.index.tz_localize(None)

    # Normalize to match feature data dates (which have 05:00 offset)
    spy_close.index = spy_close.index.normalize()
    vix_close.index = vix_close.index.normalize()

    # Also normalize feature dates
    df.index = df.index.set_levels(
        [df.index.levels[0].normalize(), df.index.levels[1]], level=[0, 1]
    )

    spy_ma200 = spy_close.rolling(200, min_periods=200).mean()
    print(f"  SPY: {len(spy_close)} bars, VIX: {len(vix_close)} bars")

    # ── Add regime features ──
    print("\n[3] Engineering regime features...")
    df = add_regime_features(df, spy_close, vix_close)
    df = add_interaction_features(df)

    # Identify bull/bear dates for Test 2
    bull_dates = []
    bear_dates = []
    for d in all_dates:
        d_norm = d.normalize()
        if d_norm in spy_close.index and d_norm in spy_ma200.index:
            if not np.isnan(spy_ma200.loc[d_norm]):
                if spy_close.loc[d_norm] > spy_ma200.loc[d_norm]:
                    bull_dates.append(d_norm)
                else:
                    bear_dates.append(d_norm)
    bull_dates = np.array(bull_dates)
    bear_dates = np.array(bear_dates)
    print(f"  Bull dates: {len(bull_dates)}, Bear dates: {len(bear_dates)}")

    # ── Load HMM crisis log ──
    hmm = pd.read_csv(hmm_file, parse_dates=["date"])
    hmm["date"] = hmm["date"].dt.normalize()
    hmm_crisis_dates = set(hmm[hmm["crisis_mode"] == True]["date"].values)
    print(f"  HMM crisis days: {len(hmm_crisis_dates)}")

    # ══════════════════════════════════════════════════════════════
    # TEST 0: BASELINE (15 stock-level features only)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("  TEST 0: BASELINE (15 stock features)")
    print(f"{'='*120}")
    pred_t0, fi_t0 = walk_forward(df, FEATURES_BASE, label="T0")
    m_t0 = evaluate(pred_t0, "T0")
    print_metrics(m_t0, "T0: Baseline")
    print_feature_importance(fi_t0, "T0")

    # ══════════════════════════════════════════════════════════════
    # TEST 1: FULL REGIME FEATURES (+7)
    # ══════════════════════════════════════════════════════════════
    features_t1 = FEATURES_BASE + FEATURES_REGIME_FULL
    print(f"\n{'='*120}")
    print(f"  TEST 1: FULL REGIME ({len(features_t1)} features)")
    print(f"{'='*120}")
    pred_t1, fi_t1 = walk_forward(df, features_t1, label="T1")
    m_t1 = evaluate(pred_t1, "T1")
    print_metrics(m_t1, "T1: Full Regime")
    print_feature_importance(fi_t1, "T1")

    # ══════════════════════════════════════════════════════════════
    # TEST 2: REGIME-CONDITIONED EVAL (T1 model, split bull/bear)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("  TEST 2: REGIME-CONDITIONED EVALUATION")
    print(f"{'='*120}")

    # Use T0 and T1 predictions, evaluate on bull-only and bear-only
    m_t0_bull = evaluate(pred_t0, "T0-Bull", date_mask=bull_dates)
    m_t0_bear = evaluate(pred_t0, "T0-Bear", date_mask=bear_dates)
    m_t1_bull = evaluate(pred_t1, "T1-Bull", date_mask=bull_dates)
    m_t1_bear = evaluate(pred_t1, "T1-Bear", date_mask=bear_dates)

    print_metrics(m_t0_bull, "T0 Baseline — BULL segment")
    print_metrics(m_t0_bear, "T0 Baseline — BEAR segment")
    print_metrics(m_t1_bull, "T1 Full Regime — BULL segment")
    print_metrics(m_t1_bear, "T1 Full Regime — BEAR segment")

    # ══════════════════════════════════════════════════════════════
    # TEST 3: MINIMAL REGIME (only VIX + SPY_MA200_DEV)
    # ══════════════════════════════════════════════════════════════
    features_t3 = FEATURES_BASE + FEATURES_REGIME_MINIMAL
    print(f"\n{'='*120}")
    print(f"  TEST 3: MINIMAL REGIME ({len(features_t3)} features)")
    print(f"{'='*120}")
    pred_t3, fi_t3 = walk_forward(df, features_t3, label="T3")
    m_t3 = evaluate(pred_t3, "T3")
    print_metrics(m_t3, "T3: Minimal Regime")
    print_feature_importance(fi_t3, "T3")

    # ══════════════════════════════════════════════════════════════
    # TEST 4: REGIME-GATED PORTFOLIO
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("  TEST 4: REGIME-GATED PORTFOLIO (SPY > MA200 → long, else cash)")
    print(f"{'='*120}")

    # Use the better model between T0 and T1
    best_model_label = "T0" if m_t0["nonoverlap_top10_sharpe"] >= m_t1["nonoverlap_top10_sharpe"] else "T1"
    best_pred = pred_t0 if best_model_label == "T0" else pred_t1
    print(f"  Using {best_model_label} predictions (better non-overlap Sharpe)")

    gated = regime_gated_backtest(best_pred, spy_close, spy_ma200)

    print(f"\n  Ungated (always long):")
    print(f"    Cum return:  {gated['ungated_cum_ret']:.2%}")
    print(f"    Sharpe:      {gated['ungated_sharpe']:.3f}")
    print(f"    MaxDD:       {gated['ungated_maxdd']:.4f}")
    print(f"\n  Gated (SPY > MA200 only):")
    print(f"    Cum return:  {gated['gated_cum_ret']:.2%}")
    print(f"    Sharpe:      {gated['gated_sharpe']:.3f}")
    print(f"    MaxDD:       {gated['gated_maxdd']:.4f}")
    print(f"    Invested:    {gated['pct_invested']:.1%} ({gated['n_trade']} trade, {gated['n_cash']} cash)")

    # ══════════════════════════════════════════════════════════════
    # TEST 5: INTERACTION FEATURES
    # ══════════════════════════════════════════════════════════════
    features_t5 = FEATURES_BASE + FEATURES_INTERACTION
    print(f"\n{'='*120}")
    print(f"  TEST 5: INTERACTION FEATURES ({len(features_t5)} features)")
    print(f"{'='*120}")
    pred_t5, fi_t5 = walk_forward(df, features_t5, label="T5")
    m_t5 = evaluate(pred_t5, "T5")
    print_metrics(m_t5, "T5: Interaction Features")
    print_feature_importance(fi_t5, "T5")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("  SUMMARY COMPARISON TABLE")
    print(f"{'='*120}")

    all_tests = [
        ("T0: Baseline", m_t0),
        ("T1: Full Regime", m_t1),
        ("T2: T0-Bull", m_t0_bull),
        ("T2: T0-Bear", m_t0_bear),
        ("T2: T1-Bull", m_t1_bull),
        ("T2: T1-Bear", m_t1_bear),
        ("T3: Minimal", m_t3),
        ("T5: Interact", m_t5),
    ]

    print(f"\n  {'Test':<20} {'NDCG@10':>8} {'Top10':>9} {'Bot10':>9} {'L/S':>9} "
          f"{'Shp(OL)':>8} {'Shp(NO)':>8} {'MaxDD':>8} {'IC':>7} {'p-val':>10}")
    print(f"  {'-'*105}")

    for name, m in all_tests:
        print(f"  {name:<20} {m['ndcg10_approx']:>8.4f} "
              f"{m['overlap_top10_mean']:>+8.5f} {m['overlap_bot10_mean']:>+8.5f} "
              f"{m['long_short_spread']:>+8.5f} "
              f"{m['overlap_top10_sharpe']:>8.3f} "
              f"{m['nonoverlap_top10_sharpe']:>8.3f} "
              f"{m['nonoverlap_top10_maxdd']:>8.4f} "
              f"{m['IC_mean']:>7.4f} "
              f"{m['pvalue']:>10.2e}")

    # Delta vs baseline
    print(f"\n  {'Test':<20} {'dNDCG':>8} {'dTop10':>9} {'dL/S':>9} {'dShp(NO)':>9} {'dIC':>7}")
    print(f"  {'-'*65}")
    for name, m in all_tests:
        if name == "T0: Baseline":
            continue
        d_ndcg = m["ndcg10_approx"] - m_t0["ndcg10_approx"]
        d_top10 = m["overlap_top10_mean"] - m_t0["overlap_top10_mean"]
        d_ls = m["long_short_spread"] - m_t0["long_short_spread"]
        d_shp = m["nonoverlap_top10_sharpe"] - m_t0["nonoverlap_top10_sharpe"]
        d_ic = m["IC_mean"] - m_t0["IC_mean"]
        print(f"  {name:<20} {d_ndcg:>+8.4f} {d_top10:>+8.5f} {d_ls:>+8.5f} "
              f"{d_shp:>+8.3f} {d_ic:>+7.4f}")

    # ══════════════════════════════════════════════════════════════
    # VERDICT
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("  VERDICT")
    print(f"{'='*120}")

    # T1 vs T0
    d_ndcg_t1 = (m_t1["ndcg10_approx"] - m_t0["ndcg10_approx"]) / m_t0["ndcg10_approx"] * 100
    print(f"\n  T1 NDCG vs T0: {d_ndcg_t1:+.2f}% "
          f"({'> 2% threshold' if abs(d_ndcg_t1) > 2 else '< 2% threshold'})")

    # T2: bull/bear consistency
    t1_bull_better = m_t1_bull["overlap_top10_mean"] >= m_t0_bull["overlap_top10_mean"]
    t1_bear_better = m_t1_bear["overlap_top10_mean"] >= m_t0_bear["overlap_top10_mean"]
    print(f"  T2 Bull: T1 {'>=':'<'} T0 → {'PASS' if t1_bull_better else 'FAIL'}")
    print(f"  T2 Bear: T1 {'>=':'<'} T0 → {'PASS' if t1_bear_better else 'FAIL'}")

    # T3 vs T1
    t3_close_t1 = abs(m_t3["ndcg10_approx"] - m_t1["ndcg10_approx"]) < 0.005
    t3_better_t0 = m_t3["ndcg10_approx"] > m_t0["ndcg10_approx"]
    print(f"  T3 ~ T1 and T3 > T0? {t3_close_t1 and t3_better_t0} → "
          f"{'Minimal 2-feature is sufficient' if t3_close_t1 and t3_better_t0 else 'Full set needed or regime features not useful'}")

    # T4: regime gate value
    gate_sharpe_gain = gated["gated_sharpe"] - gated["ungated_sharpe"]
    gate_dd_gain = gated["gated_maxdd"] - gated["ungated_maxdd"]
    print(f"  T4 Gate: Sharpe {gated['ungated_sharpe']:.3f} → {gated['gated_sharpe']:.3f} "
          f"({gate_sharpe_gain:+.3f}), MaxDD {gated['ungated_maxdd']:.3f} → {gated['gated_maxdd']:.3f} "
          f"({gate_dd_gain:+.3f})")

    # T5 interaction importance
    inter_fi = {k: v for k, v in fi_t5.items() if k.startswith("interact_")}
    all_fi_sorted = sorted(fi_t5.values(), reverse=True)
    top20_threshold = all_fi_sorted[min(19, len(all_fi_sorted)-1)] if len(all_fi_sorted) >= 20 else 0
    inter_in_top20 = sum(1 for v in inter_fi.values() if v >= top20_threshold)
    print(f"  T5 Interaction features in top-20 importance: {inter_in_top20}/2")

    # Final recommendation
    print(f"\n  RECOMMENDATION:")
    if d_ndcg_t1 > 2 and t1_bull_better and t1_bear_better:
        print(f"  → Method A WORKS: use regime features in production")
        if t3_close_t1:
            print(f"  → Minimal 2-feature version sufficient (regime_vix + regime_spy_ma200_dev)")
        else:
            print(f"  → Use full 7-feature version")
    elif gate_sharpe_gain > 0.1:
        print(f"  → Regime features don't improve ranking, but regime GATE improves portfolio")
        print(f"  → Keep current model + add MA200 gate at portfolio level")
    else:
        print(f"  → Regime fusion has limited value for this model")

    # Save results
    all_results = {
        "T0": m_t0, "T1": m_t1,
        "T2_T0_bull": m_t0_bull, "T2_T0_bear": m_t0_bear,
        "T2_T1_bull": m_t1_bull, "T2_T1_bear": m_t1_bear,
        "T3": m_t3, "T4_gated": gated, "T5": m_t5,
        "fi_t0": fi_t0, "fi_t1": fi_t1, "fi_t3": fi_t3, "fi_t5": fi_t5,
    }
    with open(output_dir / "regime_fusion_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_dir / 'regime_fusion_results.json'}")


if __name__ == "__main__":
    main()
