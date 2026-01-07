#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze "high-rank big losers": stocks that your model ranks highly but realized very bad forward returns.

Goal:
  Find common feature patterns among these failure cases to design filters / risk rules.

Inputs:
  - allfac data directory (MultiIndex date,ticker) with factor columns + 'target' (T+5 return) or 'ret_fwd_5d'
  - predictions parquet/csv with columns: date,ticker,prediction,actual

Outputs:
  - CSV: per-feature stats (loser vs winner z-score means, diff, effect size, t-stat)
  - CSV: per-feature loser-rate by feature quantile (optional diagnostic)

Typical usage:
  python scripts/analyze_high_rank_losers.py ^
    --data-dir data/factor_exports/allfac ^
    --predictions results/full_bucket_10x150_<SID>/ridge_stacking_predictions_*.parquet ^
    --top-n 30 --worst-m 10 --best-m 10 ^
    --out-dir results/high_rank_loser_analysis_<SID>
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EXCLUDE_COLS = {"target", "Close", "ret_fwd_5d"}


def _standardize_multiindex(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure clean MultiIndex(date, ticker). Mirrors logic in comprehensive_model_backtest.py."""
    if data is None or len(data) == 0:
        return data

    if isinstance(data.index, pd.MultiIndex):
        level_names = [name.lower() if isinstance(name, str) else "" for name in data.index.names]
        if "date" in level_names and ("ticker" in level_names or "symbol" in level_names):
            date_level = level_names.index("date")
            other_level = 1 - date_level
            dates = pd.to_datetime(data.index.get_level_values(date_level)).tz_localize(None).normalize()
            tickers = data.index.get_level_values(other_level).astype(str).str.upper().str.strip()
        else:
            df_reset = data.reset_index()
            if "date" not in df_reset.columns or not any(col in df_reset.columns for col in ["ticker", "symbol"]):
                raise ValueError("MultiIndex missing date/ticker information; cannot standardize.")
            df_reset["date"] = pd.to_datetime(df_reset["date"]).dt.tz_localize(None).dt.normalize()
            ticker_col = "ticker" if "ticker" in df_reset.columns else "symbol"
            df_reset["ticker"] = df_reset[ticker_col].astype(str).str.upper().str.strip()
            data = df_reset.drop(columns=[c for c in ["symbol"] if c in df_reset.columns])
            dates = df_reset["date"]
            tickers = df_reset["ticker"]
    else:
        if {"date", "ticker"}.issubset(data.columns) or {"date", "symbol"}.issubset(data.columns):
            df_reset = data.reset_index(drop=True)
            df_reset["date"] = pd.to_datetime(df_reset["date"]).dt.tz_localize(None).dt.normalize()
            ticker_col = "ticker" if "ticker" in df_reset.columns else "symbol"
            df_reset["ticker"] = df_reset[ticker_col].astype(str).str.upper().str.strip()
            data = df_reset
            dates = df_reset["date"]
            tickers = df_reset["ticker"]
        else:
            raise ValueError("Data missing date/ticker columns; cannot build MultiIndex.")

    standardized = data.copy()
    standardized.index = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
    standardized = standardized[~standardized.index.duplicated(keep="last")]
    standardized = standardized.sort_index()
    return standardized


def load_allfac(data_dir: str, data_file: Optional[str] = None) -> pd.DataFrame:
    if data_file:
        df = pd.read_parquet(data_file)
        return _standardize_multiindex(df)

    # allow passing a single parquet path as data_dir
    if data_dir.lower().endswith(".parquet") and os.path.exists(data_dir):
        df = pd.read_parquet(data_dir)
        return _standardize_multiindex(df)

    manifest_path = os.path.join(data_dir, "manifest.parquet")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.parquet not found under {data_dir}")

    manifest = pd.read_parquet(manifest_path)
    batches: List[pd.DataFrame] = []
    for _, row in manifest.iterrows():
        batch_id = int(row["batch_id"])
        batch_file = os.path.join(data_dir, f"polygon_factors_batch_{batch_id:04d}.parquet")
        if os.path.exists(batch_file):
            batches.append(pd.read_parquet(batch_file))
    if not batches:
        raise ValueError("No batch files loaded from manifest")
    df = pd.concat(batches, axis=0)
    return _standardize_multiindex(df)


def load_predictions(path_or_glob: str) -> pd.DataFrame:
    paths = sorted(glob.glob(path_or_glob)) if any(ch in path_or_glob for ch in "*?[]") else [path_or_glob]
    if not paths or not os.path.exists(paths[0]):
        raise FileNotFoundError(f"Predictions file not found: {path_or_glob}")
    path = paths[-1]
    if path.lower().endswith(".parquet"):
        pred = pd.read_parquet(path)
    else:
        pred = pd.read_csv(path)
    for c in ["date", "ticker", "prediction"]:
        if c not in pred.columns:
            raise ValueError(f"Predictions missing required column '{c}' in {path}")
    pred = pred.copy()
    pred["date"] = pd.to_datetime(pred["date"]).dt.tz_localize(None).dt.normalize()
    pred["ticker"] = pred["ticker"].astype(str).str.upper().str.strip()
    if "actual" in pred.columns:
        pred["actual"] = pd.to_numeric(pred["actual"], errors="coerce")
    return pred


def _pick_label_column(allfac: pd.DataFrame) -> str:
    if "target" in allfac.columns:
        return "target"
    if "ret_fwd_5d" in allfac.columns:
        return "ret_fwd_5d"
    raise ValueError("allfac data missing both 'target' and 'ret_fwd_5d' columns (need forward returns).")


def _zscore_by_date(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Cross-sectional z-score per date for comparability across time."""
    z = df[feature_cols].copy()
    z = z.replace([np.inf, -np.inf], np.nan)
    # z-score within each date
    grouped = z.groupby(level="date", sort=False)
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0, np.nan)
    z = (z - means) / stds
    return z


@dataclass(frozen=True)
class SliceConfig:
    top_n: int
    worst_m: int
    best_m: int
    min_universe: int = 500


def build_event_slices(
    predictions: pd.DataFrame,
    cfg: SliceConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each date:
      - take Top-N by prediction
      - among those, mark bottom 'worst_m' by actual as losers
      - and top 'best_m' by actual as winners (comparison group)
    Returns:
      losers_df, winners_df with columns: date,ticker,prediction,actual
    """
    rows_l: List[pd.DataFrame] = []
    rows_w: List[pd.DataFrame] = []

    for d, g in predictions.groupby("date", sort=True):
        g = g.dropna(subset=["prediction"]).copy()
        if "actual" in g.columns:
            g = g.dropna(subset=["actual"])
        if len(g) < cfg.min_universe:
            continue

        g = g.sort_values("prediction", ascending=False)
        topn = g.head(min(cfg.top_n, len(g))).copy()
        if len(topn) < max(cfg.worst_m, cfg.best_m, 5):
            continue

        # losers = worst actual inside the topn
        topn_sorted_by_actual = topn.sort_values("actual", ascending=True)
        losers = topn_sorted_by_actual.head(min(cfg.worst_m, len(topn_sorted_by_actual))).copy()

        # winners = best actual inside the topn
        winners = topn_sorted_by_actual.tail(min(cfg.best_m, len(topn_sorted_by_actual))).copy()

        rows_l.append(losers)
        rows_w.append(winners)

    if not rows_l or not rows_w:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(rows_l, ignore_index=True), pd.concat(rows_w, ignore_index=True)


def compute_feature_stats(
    allfac: pd.DataFrame,
    losers: pd.DataFrame,
    winners: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Compare feature z-scores (by date) between losers and winners.
    Produces per-feature:
      - loser_mean_z, winner_mean_z, diff
      - pooled std (effect size)
      - approximate t-stat
      - missing rates
    """
    """
    Memory-safe streaming implementation:
      For each date:
        - compute universe mean/std for each feature
        - z-score losers/winners for that date
        - update per-feature aggregates (sum/sumsq/count)

    This avoids materializing a full z-scored allfac matrix (which is huge).
    """
    # Group event lists by date for fast lookup
    losers_by_date = losers.groupby("date")["ticker"].apply(lambda s: [str(x).upper().strip() for x in s]).to_dict()
    winners_by_date = winners.groupby("date")["ticker"].apply(lambda s: [str(x).upper().strip() for x in s]).to_dict()
    dates = sorted(set(losers_by_date.keys()).intersection(winners_by_date.keys()))
    if not dates:
        return pd.DataFrame()

    nF = len(feature_cols)
    # z-score aggregates
    sum_l = np.zeros(nF, dtype=float)
    sumsq_l = np.zeros(nF, dtype=float)
    cnt_l = np.zeros(nF, dtype=float)
    sum_w = np.zeros(nF, dtype=float)
    sumsq_w = np.zeros(nF, dtype=float)
    cnt_w = np.zeros(nF, dtype=float)
    # raw-missing aggregates (on raw feature values)
    miss_l = np.zeros(nF, dtype=float)
    miss_w = np.zeros(nF, dtype=float)
    rows_l = np.zeros(nF, dtype=float)
    rows_w = np.zeros(nF, dtype=float)

    for d in dates:
        try:
            universe = allfac.xs(d, level="date", drop_level=True)
        except KeyError:
            continue
        if universe is None or len(universe) == 0:
            continue

        # Ensure numeric for the whole universe cross-section once per date
        u = universe[feature_cols].replace([np.inf, -np.inf], np.nan)
        # Note: applying to_numeric across many cols is expensive but still far smaller than full allfac z-matrix
        u = u.apply(pd.to_numeric, errors="coerce")

        U = u.to_numpy(dtype=float, copy=False)
        mu = np.nanmean(U, axis=0)
        sigma = np.nanstd(U, axis=0, ddof=1)
        sigma = np.where((sigma > 0) & np.isfinite(sigma), sigma, np.nan)

        # losers
        lt = losers_by_date.get(d, [])
        if lt:
            xl = u.reindex(lt).to_numpy(dtype=float, copy=False)
            zl = (xl - mu) / sigma
            valid = np.isfinite(zl)
            cnt_l += valid.sum(axis=0)
            sum_l += np.where(valid, zl, 0.0).sum(axis=0)
            sumsq_l += np.where(valid, zl * zl, 0.0).sum(axis=0)
            # missingness on raw (after numeric coercion)
            miss_l += np.isnan(xl).sum(axis=0)
            rows_l += xl.shape[0]

        # winners
        wt = winners_by_date.get(d, [])
        if wt:
            xw = u.reindex(wt).to_numpy(dtype=float, copy=False)
            zw = (xw - mu) / sigma
            valid = np.isfinite(zw)
            cnt_w += valid.sum(axis=0)
            sum_w += np.where(valid, zw, 0.0).sum(axis=0)
            sumsq_w += np.where(valid, zw * zw, 0.0).sum(axis=0)
            miss_w += np.isnan(xw).sum(axis=0)
            rows_w += xw.shape[0]

    # Build per-feature stats
    out_rows: List[Dict[str, float]] = []
    for i, col in enumerate(feature_cols):
        nl = float(cnt_l[i])
        nw = float(cnt_w[i])
        if nl < 50 or nw < 50:
            continue

        mu_l = float(sum_l[i] / nl)
        mu_w = float(sum_w[i] / nw)
        diff = mu_l - mu_w

        # population var estimate from sums; convert to sample-ish var with ddof=1 when possible
        var_l = float(max(0.0, (sumsq_l[i] / nl) - mu_l * mu_l))
        var_w = float(max(0.0, (sumsq_w[i] / nw) - mu_w * mu_w))

        # pooled std (Cohen's d)
        pooled = np.sqrt(((nl - 1.0) * var_l + (nw - 1.0) * var_w) / max(1.0, (nl + nw - 2.0)))
        effect = float(diff / pooled) if pooled and np.isfinite(pooled) and pooled > 0 else float("nan")

        se = np.sqrt(var_l / nl + var_w / nw)
        tstat = float(diff / se) if se and np.isfinite(se) and se > 0 else float("nan")

        # missing rates (raw, after coercion)
        denom_l = float(rows_l[i]) if rows_l[i] > 0 else float("nan")
        denom_w = float(rows_w[i]) if rows_w[i] > 0 else float("nan")
        miss_rate_l = float(miss_l[i] / denom_l) if np.isfinite(denom_l) and denom_l > 0 else float("nan")
        miss_rate_w = float(miss_w[i] / denom_w) if np.isfinite(denom_w) and denom_w > 0 else float("nan")

        out_rows.append(
            {
                "feature": col,
                "loser_mean_z": mu_l,
                "winner_mean_z": mu_w,
                "diff_loser_minus_winner_z": diff,
                "effect_size_d": effect,
                "t_stat": tstat,
                "n_loser_rows": int(nl),
                "n_winner_rows": int(nw),
                "missing_rate_loser_raw": miss_rate_l,
                "missing_rate_winner_raw": miss_rate_w,
            }
        )

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    out = out.sort_values("effect_size_d", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return out


def feature_quantile_loser_rate(
    allfac: pd.DataFrame,
    losers: pd.DataFrame,
    winners: pd.DataFrame,
    feature_cols: List[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Diagnostic: for each feature, compute loser-rate by feature quantile (using z-scores).
    loser_rate(bin) = losers_in_bin / (losers_in_bin + winners_in_bin)
    """
    key_l = pd.MultiIndex.from_arrays([losers["date"], losers["ticker"]], names=["date", "ticker"])
    key_w = pd.MultiIndex.from_arrays([winners["date"], winners["ticker"]], names=["date", "ticker"])
    z_all = _zscore_by_date(allfac, feature_cols)
    z_l = z_all.reindex(key_l)
    z_w = z_all.reindex(key_w)

    rows: List[Dict[str, object]] = []
    for col in feature_cols:
        xl = pd.to_numeric(z_l[col], errors="coerce").dropna()
        xw = pd.to_numeric(z_w[col], errors="coerce").dropna()
        if len(xl) < 100 or len(xw) < 100:
            continue

        # build bins on combined distribution so bins are comparable
        x = pd.concat([xl.rename("x"), xw.rename("x")], axis=0)
        try:
            bins = pd.qcut(x, q=n_bins, duplicates="drop")
        except Exception:
            continue
        # reindex bin labels back
        bins_l = bins.reindex(xl.index)
        bins_w = bins.reindex(xw.index)

        for b in bins.cat.categories:
            l_cnt = int((bins_l == b).sum())
            w_cnt = int((bins_w == b).sum())
            denom = l_cnt + w_cnt
            rows.append(
                {
                    "feature": col,
                    "bin": str(b),
                    "loser_cnt": l_cnt,
                    "winner_cnt": w_cnt,
                    "loser_rate": (l_cnt / denom) if denom > 0 else float("nan"),
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data/factor_exports/factors")
    ap.add_argument("--data-file", type=str, default=None, help="Optional single parquet (instead of data-dir batches)")
    ap.add_argument("--predictions", type=str, required=True, help="Predictions parquet/csv path or glob")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--worst-m", type=int, default=10, help="Within Top-N, how many worst-actual names to treat as losers")
    ap.add_argument("--best-m", type=int, default=10, help="Within Top-N, how many best-actual names to treat as winners")
    ap.add_argument("--min-universe", type=int, default=500)
    ap.add_argument("--max-features", type=int, default=300, help="Optional cap for speed; keeps first N feature cols")
    ap.add_argument("--quantile-diagnostic", action="store_true", help="Write loser-rate by feature quantile (slower)")
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    preds = load_predictions(args.predictions)

    # Prefer using 'actual' from predictions if present; otherwise join from allfac later
    allfac = load_allfac(args.data_dir, data_file=args.data_file)
    label_col = _pick_label_column(allfac)

    if "actual" not in preds.columns or preds["actual"].isna().all():
        # join label from allfac
        key = pd.MultiIndex.from_arrays([preds["date"], preds["ticker"]], names=["date", "ticker"])
        actual = allfac.reindex(key)[label_col].reset_index(drop=True)
        preds["actual"] = pd.to_numeric(actual, errors="coerce")

    cfg = SliceConfig(
        top_n=int(args.top_n),
        worst_m=int(args.worst_m),
        best_m=int(args.best_m),
        min_universe=int(args.min_universe),
    )
    losers, winners = build_event_slices(preds, cfg)
    if losers.empty or winners.empty:
        raise RuntimeError("No loser/winner slices constructed. Check min-universe/top-n/worst-m/best-m and data coverage.")

    # feature columns
    feature_cols = [c for c in allfac.columns if c not in EXCLUDE_COLS]
    if args.max_features and len(feature_cols) > int(args.max_features):
        feature_cols = feature_cols[: int(args.max_features)]

    stats = compute_feature_stats(allfac, losers, winners, feature_cols)
    stats_path = os.path.join(args.out_dir, "feature_stats_loser_vs_winner.csv")
    stats.to_csv(stats_path, index=False)

    meta = {
        "predictions": args.predictions,
        "data_dir": args.data_dir,
        "data_file": args.data_file,
        "label_col": label_col,
        "top_n": cfg.top_n,
        "worst_m": cfg.worst_m,
        "best_m": cfg.best_m,
        "min_universe": cfg.min_universe,
        "n_loser_rows": int(len(losers)),
        "n_winner_rows": int(len(winners)),
        "n_features_used": int(len(feature_cols)),
    }
    with open(os.path.join(args.out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, ensure_ascii=False, indent=2)

    # optional diagnostic
    if bool(args.quantile_diagnostic):
        diag = feature_quantile_loser_rate(allfac, losers, winners, feature_cols, n_bins=10)
        diag_path = os.path.join(args.out_dir, "feature_quantile_loser_rate.csv")
        diag.to_csv(diag_path, index=False)

    # write the actual loser/winner event lists
    losers_path = os.path.join(args.out_dir, "loser_events.csv")
    winners_path = os.path.join(args.out_dir, "winner_events.csv")
    losers.to_csv(losers_path, index=False)
    winners.to_csv(winners_path, index=False)

    print(f"OK: wrote {stats_path}")
    print(f"OK: wrote {losers_path}")
    print(f"OK: wrote {winners_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


