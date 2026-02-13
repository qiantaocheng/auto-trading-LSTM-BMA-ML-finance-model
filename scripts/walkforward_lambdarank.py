#!/usr/bin/env python3
"""
Walk-Forward LambdaRank Pipeline — Leakage-Free Design
=======================================================
Expanding-window walk-forward:  at each step, train ONLY on dates
strictly before  (test_start − purge_gap), predict on test window,
step forward.

Leakage prevention checklist
-----------------------------
1. Features are backward-looking rolling indicators (RSI, momentum …).
   No centering / future-scaled normalisation.
2. Target = T5 forward return  →  5-day purge gap between train-end
   and test-start so the last training target never overlaps test dates.
3. Quantile labels built per-date cross-section (no global info).
4. Model retrained fresh at every walk-forward step.

Outputs
-------
Per-window metrics + aggregate:
  Bucket 0-10  (top 10)   — mean, median, win-rate, Sharpe, maxDD
  Bucket 10-20 (rank 11-20) — same
  Top-10 Sharpe ratio (annualised)
  Overlap (daily) vs Non-overlap (every horizon_days)
  Statistical significance for overlap (t-test + bootstrap 95% CI)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── features & hyper-params (mirror existing pipeline) ───────────
FEATURES = [
    "volume_price_corr_3d", "rsi_14", "reversal_3d", "momentum_10d",
    "liquid_momentum_10d", "sharpe_momentum_5d", "price_ma20_deviation",
    "avg_trade_size", "trend_r2_20", "dollar_vol_20", "ret_skew_20d",
    "reversal_5d", "near_52w_high", "atr_pct_14", "amihud_20",
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


# ── helpers ──────────────────────────────────────────────────────
def ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex) and {"date", "ticker"}.issubset(df.index.names):
        return df.sort_index()
    if {"date", "ticker"}.issubset(df.columns):
        return df.set_index(["date", "ticker"]).sort_index()
    raise ValueError("Need date/ticker columns or MultiIndex")


def build_quantile_labels(y: np.ndarray, dates: np.ndarray) -> np.ndarray:
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


def group_counts(dates: np.ndarray) -> List[int]:
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


# ── training ─────────────────────────────────────────────────────
def train_model(train_df: pd.DataFrame, features: List[str],
                horizon: int, cv_splits: int, n_boost: int,
                seed: int) -> Tuple[lgb.Booster, int]:
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
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
    }

    best_rounds: List[int] = []
    for ti, vi in purged_cv_splits(dates, cv_splits, horizon, horizon):
        ts = lgb.Dataset(X[ti], label=labels[ti], group=group_counts(dates[ti]))
        vs = lgb.Dataset(X[vi], label=labels[vi], group=group_counts(dates[vi]))
        bst = lgb.train(lgb_params, ts, num_boost_round=n_boost,
                        valid_sets=[vs],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        best_rounds.append(bst.best_iteration or n_boost)

    final_rounds = int(np.mean(best_rounds)) if best_rounds else n_boost
    full = lgb.Dataset(X, label=labels, group=group_counts(dates))
    model = lgb.train(lgb_params, full, num_boost_round=final_rounds)
    return model, final_rounds


# ── walk-forward engine ──────────────────────────────────────────
def walk_forward(df: pd.DataFrame,
                 features: List[str],
                 horizon: int,
                 min_train_days: int,
                 test_window: int,
                 step_size: int,
                 cv_splits: int,
                 n_boost: int,
                 seed: int) -> pd.DataFrame:
    """
    Expanding-window walk-forward.

    Timeline for each step:
        |---- training data ----|-- purge --|---- test window ----|
        start          train_end   +horizon   test_start  test_end

    train_end = test_start - horizon - 1   (purge gap = horizon days)
    """
    all_dates = df.index.get_level_values("date").unique().sort_values()
    n_dates = len(all_dates)
    print(f"Total dates: {n_dates}  ({all_dates[0].date()} → {all_dates[-1].date()})")
    print(f"Min train: {min_train_days}d, test window: {test_window}d, "
          f"step: {step_size}d, purge: {horizon}d\n")

    # first test window starts after min_train_days + purge
    first_test_idx = min_train_days + horizon
    if first_test_idx >= n_dates:
        raise ValueError("Not enough data for even one walk-forward step")

    results: List[pd.DataFrame] = []
    step = 0
    test_start_idx = first_test_idx

    while test_start_idx < n_dates:
        test_end_idx = min(test_start_idx + test_window, n_dates)
        train_end_idx = test_start_idx - horizon  # exclusive; purge gap

        train_dates = all_dates[:train_end_idx]
        test_dates_slice = all_dates[test_start_idx:test_end_idx]

        train_df = df.loc[(train_dates, slice(None)), :]
        test_df = df.loc[(test_dates_slice, slice(None)), :]

        n_train = len(train_dates)
        n_test = len(test_dates_slice)
        print(f"Step {step}: train {train_dates[0].date()}→{train_dates[-1].date()} "
              f"({n_train}d) | test {test_dates_slice[0].date()}→{test_dates_slice[-1].date()} "
              f"({n_test}d)")

        # ----- LEAKAGE CHECK -----
        # The last target in training spans [train_end - 1 .. train_end - 1 + horizon).
        # test_start = train_end + horizon, so there is zero overlap.  ✓
        last_train_target_end = train_dates[-1]  # target covers +5 bdays from here
        assert test_dates_slice[0] > last_train_target_end, \
            f"Leakage! test_start {test_dates_slice[0]} <= last_train {last_train_target_end}"

        model, rounds = train_model(train_df, features, horizon, cv_splits, n_boost, seed)
        print(f"  → trained {rounds} rounds")

        X_test = test_df[features].fillna(0.0).to_numpy()
        preds = model.predict(X_test)

        res = test_df[["target"]].copy()
        res["pred"] = preds
        res["wf_step"] = step
        results.append(res)

        step += 1
        test_start_idx += step_size

    return pd.concat(results)


# ── bucket analysis ──────────────────────────────────────────────
def max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown from a series of period returns."""
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    dd = cum / running_max - 1
    return float(dd.min()) if len(dd) > 0 else 0.0


def bucket_stats(returns: np.ndarray, freq_days: int, label: str,
                 compute_dd: bool = True) -> Dict:
    """Compute stats for a bucket's return series."""
    if len(returns) == 0:
        return {f"{label}_mean": np.nan, f"{label}_median": np.nan,
                f"{label}_winrate": np.nan, f"{label}_sharpe": np.nan,
                f"{label}_maxdd": np.nan, f"{label}_n": 0}
    mean = float(np.mean(returns))
    median = float(np.median(returns))
    wr = float(np.mean(returns > 0))
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else np.nan
    sharpe = float(mean / std * np.sqrt(252.0 / freq_days)) if std > 0 else np.nan
    # MaxDD only meaningful for non-overlap (actual rebalance cadence)
    mdd = max_drawdown(returns) if compute_dd else np.nan
    return {
        f"{label}_mean": mean,
        f"{label}_median": median,
        f"{label}_winrate": wr,
        f"{label}_sharpe": sharpe,
        f"{label}_maxdd": mdd,
        f"{label}_n": len(returns),
    }


def statistical_significance(returns: np.ndarray, n_bootstrap: int = 10000,
                              label: str = "") -> Dict:
    """T-test vs zero + bootstrap 95% CI on mean."""
    out = {}
    if len(returns) < 3:
        out[f"{label}_tstat"] = np.nan
        out[f"{label}_pvalue"] = np.nan
        out[f"{label}_ci95_lo"] = np.nan
        out[f"{label}_ci95_hi"] = np.nan
        return out

    t, p = sp_stats.ttest_1samp(returns, 0)
    out[f"{label}_tstat"] = float(t)
    out[f"{label}_pvalue"] = float(p)

    # bootstrap CI
    rng = np.random.RandomState(42)
    boot_means = np.array([
        np.mean(rng.choice(returns, size=len(returns), replace=True))
        for _ in range(n_bootstrap)
    ])
    out[f"{label}_ci95_lo"] = float(np.percentile(boot_means, 2.5))
    out[f"{label}_ci95_hi"] = float(np.percentile(boot_means, 97.5))
    return out


def analyse_predictions(pred_df: pd.DataFrame, horizon: int) -> Dict:
    """
    From walk-forward predictions, compute all bucket metrics.

    Buckets (by daily cross-sectional rank):
      top_0_10   : rank 1-10   (top 10)
      top_10_20  : rank 11-20
    """
    dates = pred_df.index.get_level_values("date")
    unique_dates = np.sort(dates.unique())

    # per-date bucket returns
    overlap_top10: List[float] = []
    overlap_top10_20: List[float] = []
    non_overlap_top10: List[float] = []
    non_overlap_top10_20: List[float] = []

    non_overlap_dates = unique_dates[::max(1, horizon)]

    for d in unique_dates:
        day = pred_df.loc[d]
        if len(day) < 20:
            continue
        order = np.argsort(-day["pred"].values)
        tgts = day["target"].values

        r_top10 = float(tgts[order[:10]].mean())
        r_top10_20 = float(tgts[order[10:20]].mean())

        overlap_top10.append(r_top10)
        overlap_top10_20.append(r_top10_20)

        if d in non_overlap_dates:
            non_overlap_top10.append(r_top10)
            non_overlap_top10_20.append(r_top10_20)

    overlap_top10 = np.array(overlap_top10)
    overlap_top10_20 = np.array(overlap_top10_20)
    non_overlap_top10 = np.array(non_overlap_top10)
    non_overlap_top10_20 = np.array(non_overlap_top10_20)

    results = {}

    # ── overlap stats (NO maxDD — overlapping returns, DD meaningless) ─
    results.update(bucket_stats(overlap_top10, 1, "overlap_top0_10", compute_dd=False))
    results.update(bucket_stats(overlap_top10_20, 1, "overlap_top10_20", compute_dd=False))

    # ── non-overlap stats (maxDD valid — matches actual rebalance) ───
    results.update(bucket_stats(non_overlap_top10, horizon, "nonoverlap_top0_10", compute_dd=True))
    results.update(bucket_stats(non_overlap_top10_20, horizon, "nonoverlap_top10_20", compute_dd=True))

    # ── top-10 Sharpe (overlap, annualised) ──────────────
    results["top10_sharpe_overlap"] = results["overlap_top0_10_sharpe"]
    results["top10_sharpe_nonoverlap"] = results["nonoverlap_top0_10_sharpe"]

    # ── statistical significance for overlap ─────────────
    results.update(statistical_significance(overlap_top10, label="overlap_top0_10"))
    results.update(statistical_significance(overlap_top10_20, label="overlap_top10_20"))

    # ── IC: top-10 only (correlation within the stocks we actually trade) ─
    ic_top10_vals = []
    # also keep full-universe IC for reference
    ic_full_vals = []
    for d in unique_dates:
        day = pred_df.loc[d]
        if len(day) < 20:
            continue
        preds_d = day["pred"].values
        tgts_d = day["target"].values
        order = np.argsort(-preds_d)
        # top-10 IC: rank correlation between pred and target within top 10
        top10_preds = preds_d[order[:10]]
        top10_tgts = tgts_d[order[:10]]
        ic10 = float(sp_stats.spearmanr(top10_preds, top10_tgts).statistic)
        if np.isfinite(ic10):
            ic_top10_vals.append(ic10)
        # full-universe IC for reference
        ic_full = float(np.corrcoef(preds_d, tgts_d)[0, 1])
        if np.isfinite(ic_full):
            ic_full_vals.append(ic_full)

    results["IC_top10_mean"] = float(np.nanmean(ic_top10_vals)) if ic_top10_vals else np.nan
    results["IC_top10_std"] = float(np.nanstd(ic_top10_vals)) if ic_top10_vals else np.nan
    results["IC_top10_IR"] = (
        results["IC_top10_mean"] / results["IC_top10_std"]
        if results["IC_top10_std"] > 0 else np.nan
    )
    results["IC_full_mean"] = float(np.nanmean(ic_full_vals)) if ic_full_vals else np.nan
    results["IC_full_std"] = float(np.nanstd(ic_full_vals)) if ic_full_vals else np.nan

    return results


# ── leakage audit on features ────────────────────────────────────
def audit_feature_leakage(feature_df: pd.DataFrame, ohlcv_df: pd.DataFrame,
                           sample_date: pd.Timestamp, sample_ticker: str,
                           features: List[str]) -> Dict:
    """
    Spot-check that features at (date, ticker) only use data ≤ date.
    Recompute a few indicators from raw OHLCV and compare.
    """
    report = {"date": str(sample_date), "ticker": sample_ticker, "checks": {}}

    # get stored feature values
    try:
        stored = feature_df.loc[(sample_date, sample_ticker)]
    except KeyError:
        report["checks"]["error"] = "sample not found in feature_df"
        return report

    # get raw OHLCV up to sample_date (inclusive)
    ohlcv = ohlcv_df[(ohlcv_df["ticker"] == sample_ticker) &
                      (ohlcv_df["date"] <= sample_date)].sort_values("date")
    if len(ohlcv) < 20:
        report["checks"]["error"] = f"only {len(ohlcv)} OHLCV rows"
        return report

    close = ohlcv["Close"].values

    # RSI-14 from scratch
    deltas = np.diff(close[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss > 0:
        rs = avg_gain / avg_loss
        rsi_recomputed = 100 - 100 / (1 + rs)
    else:
        rsi_recomputed = 100.0
    report["checks"]["rsi_14"] = {
        "stored": float(stored.get("rsi_14", np.nan)),
        "recomputed": round(rsi_recomputed, 4),
        "note": "Simple SMA-RSI (approximate); small diff OK, big diff = leakage"
    }

    # momentum_10d
    if len(close) >= 11:
        mom = close[-1] / close[-11] - 1
        report["checks"]["momentum_10d"] = {
            "stored": float(stored.get("momentum_10d", np.nan)),
            "recomputed": round(mom, 6),
        }

    # reversal_3d
    if len(close) >= 4:
        rev3 = close[-1] / close[-4] - 1
        report["checks"]["reversal_3d"] = {
            "stored": float(stored.get("reversal_3d", np.nan)),
            "recomputed": round(rev3, 6),
        }

    return report


# ── per-window detail table ──────────────────────────────────────
def per_window_table(pred_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return per walk-forward-step statistics."""
    rows = []
    for step in sorted(pred_df["wf_step"].unique()):
        chunk = pred_df[pred_df["wf_step"] == step]
        dates = chunk.index.get_level_values("date").unique().sort_values()
        n_dates = len(dates)
        top10_rets = []
        top10_20_rets = []
        for d in dates:
            day = chunk.loc[d]
            if len(day) < 20:
                continue
            order = np.argsort(-day["pred"].values)
            tgts = day["target"].values
            top10_rets.append(float(tgts[order[:10]].mean()))
            top10_20_rets.append(float(tgts[order[10:20]].mean()))
        top10_rets = np.array(top10_rets)
        top10_20_rets = np.array(top10_20_rets)
        rows.append({
            "step": step,
            "test_start": str(dates[0].date()),
            "test_end": str(dates[-1].date()),
            "n_test_days": n_dates,
            "top0_10_mean": float(np.mean(top10_rets)) if len(top10_rets) else np.nan,
            "top0_10_median": float(np.median(top10_rets)) if len(top10_rets) else np.nan,
            "top0_10_wr": float(np.mean(top10_rets > 0)) if len(top10_rets) else np.nan,
            "top0_10_sharpe": (
                float(np.mean(top10_rets) / np.std(top10_rets, ddof=1) * np.sqrt(252))
                if len(top10_rets) > 1 and np.std(top10_rets, ddof=1) > 0 else np.nan
            ),
            "top0_10_maxdd": max_drawdown(top10_rets) if len(top10_rets) else np.nan,
            "top10_20_mean": float(np.mean(top10_20_rets)) if len(top10_20_rets) else np.nan,
            "top10_20_median": float(np.median(top10_20_rets)) if len(top10_20_rets) else np.nan,
            "top10_20_wr": float(np.mean(top10_20_rets > 0)) if len(top10_20_rets) else np.nan,
            "top10_20_sharpe": (
                float(np.mean(top10_20_rets) / np.std(top10_20_rets, ddof=1) * np.sqrt(252))
                if len(top10_20_rets) > 1 and np.std(top10_20_rets, ddof=1) > 0 else np.nan
            ),
            "top10_20_maxdd": max_drawdown(top10_20_rets) if len(top10_20_rets) else np.nan,
        })
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Walk-Forward LambdaRank (leakage-free)")
    p.add_argument("--feature-file", type=Path,
                   default=Path("data/factor_exports/polygon_full_features_T5.parquet"))
    p.add_argument("--ohlcv-file", type=Path,
                   default=Path("data/raw_ohlcv/polygon_raw_ohlcv_2021_2026.parquet"))
    p.add_argument("--horizon", type=int, default=5,
                   help="T5 forward-return horizon (purge gap)")
    p.add_argument("--min-train-days", type=int, default=252,
                   help="Minimum training window (trading days)")
    p.add_argument("--test-window", type=int, default=63,
                   help="Test window size (trading days ≈ 3 months)")
    p.add_argument("--step-size", type=int, default=63,
                   help="Step forward size (trading days)")
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--n-boost", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/walkforward_lambdarank"))
    p.add_argument("--audit-leakage", action="store_true",
                   help="Run feature-leakage spot-check against raw OHLCV")
    p.add_argument("--features", nargs="+", default=FEATURES)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ────────────────────────────────────────
    print("Loading feature data …")
    df = pd.read_parquet(args.feature_file)
    df = ensure_multiindex(df)

    # validate no NaN targets
    n_nan_target = df["target"].isna().sum()
    if n_nan_target > 0:
        print(f"  Dropping {n_nan_target} rows with NaN target")
        df = df.dropna(subset=["target"])

    all_dates = df.index.get_level_values("date").unique().sort_values()
    print(f"  {len(all_dates)} dates, {len(df)} rows")

    # ── optional leakage audit ───────────────────────────
    if args.audit_leakage:
        print("\n=== Feature Leakage Audit ===")
        ohlcv = pd.read_parquet(args.ohlcv_file)
        # pick 3 random (date, ticker) samples from the middle of the dataset
        mid_dates = all_dates[len(all_dates)//3 : 2*len(all_dates)//3]
        rng = np.random.RandomState(42)
        for _ in range(3):
            d = rng.choice(mid_dates)
            tickers = df.loc[d].index.get_level_values("ticker")
            t = rng.choice(tickers)
            report = audit_feature_leakage(df, ohlcv, d, t, args.features)
            print(json.dumps(report, indent=2, default=str))
        print()

    # ── walk-forward ─────────────────────────────────────
    print("=== Walk-Forward Execution ===\n")
    pred_df = walk_forward(
        df, args.features, args.horizon,
        args.min_train_days, args.test_window, args.step_size,
        args.cv_splits, args.n_boost, args.seed,
    )
    print(f"\nTotal predictions: {len(pred_df)}")

    # ── aggregate analysis ───────────────────────────────
    print("\n=== Aggregate Metrics ===\n")
    metrics = analyse_predictions(pred_df, args.horizon)

    # pretty-print
    section_order = [
        ("OVERLAP — Bucket 0-10 (Top 10)", "overlap_top0_10"),
        ("OVERLAP — Bucket 10-20", "overlap_top10_20"),
        ("NON-OVERLAP — Bucket 0-10 (Top 10)", "nonoverlap_top0_10"),
        ("NON-OVERLAP — Bucket 10-20", "nonoverlap_top10_20"),
    ]
    for title, prefix in section_order:
        print(f"  {title}")
        print(f"    Mean:      {metrics.get(f'{prefix}_mean', float('nan')):.6f}")
        print(f"    Median:    {metrics.get(f'{prefix}_median', float('nan')):.6f}")
        print(f"    Win Rate:  {metrics.get(f'{prefix}_winrate', float('nan')):.4f}")
        print(f"    Sharpe:    {metrics.get(f'{prefix}_sharpe', float('nan')):.4f}")
        mdd = metrics.get(f'{prefix}_maxdd', float('nan'))
        if np.isfinite(mdd):
            print(f"    Max DD:    {mdd:.4f}")
        else:
            print(f"    Max DD:    N/A (overlap DD meaningless)")
        print(f"    N:         {metrics.get(f'{prefix}_n', 0)}")
        print()

    print("  TOP-10 SHARPE")
    print(f"    Overlap:     {metrics.get('top10_sharpe_overlap', float('nan')):.4f}")
    print(f"    Non-Overlap: {metrics.get('top10_sharpe_nonoverlap', float('nan')):.4f}")
    print()

    print("  STATISTICAL SIGNIFICANCE (overlap)")
    for prefix in ["overlap_top0_10", "overlap_top10_20"]:
        label = "Top 0-10" if "0_10" in prefix else "Top 10-20"
        t = metrics.get(f"{prefix}_tstat", np.nan)
        p = metrics.get(f"{prefix}_pvalue", np.nan)
        lo = metrics.get(f"{prefix}_ci95_lo", np.nan)
        hi = metrics.get(f"{prefix}_ci95_hi", np.nan)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {label}:  t={t:.3f}  p={p:.6f} {sig}  "
              f"95%CI=[{lo:.6f}, {hi:.6f}]")
    print()

    print("  IC (Top-10 Spearman — within the stocks we trade)")
    print(f"    mean={metrics['IC_top10_mean']:.4f}  std={metrics['IC_top10_std']:.4f}  "
          f"IR={metrics['IC_top10_IR']:.4f}")
    print(f"  IC (Full Universe — reference only)")
    print(f"    mean={metrics['IC_full_mean']:.4f}  std={metrics['IC_full_std']:.4f}")
    print()

    # ── per-window detail ────────────────────────────────
    print("=== Per Walk-Forward Window ===\n")
    window_df = per_window_table(pred_df, args.horizon)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(window_df.to_string(index=False))
    print()

    # ── save ─────────────────────────────────────────────
    out = args.output_dir
    with open(out / "aggregate_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    window_df.to_csv(out / "per_window_metrics.csv", index=False)
    pred_df.to_parquet(out / "walk_forward_predictions.parquet")
    print(f"Saved to {out}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
