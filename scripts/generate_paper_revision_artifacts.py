#!/usr/bin/env python3
"""
Generate artifacts requested by a reviewer for the Equity Ranking paper:
- Ridge meta-learner weights plot (from snapshot ridge_model.pkl)
- Feature lists table (from best_features_per_model.json if present)
- Long-only performance summaries (from performance_report CSV)
- Year-by-year stats (from per-period long-only returns series)
- Distribution stats (skew/kurt)
- FF5 regression (optional; downloads Ken French factors via pandas_datareader if available)

Outputs a folder containing:
- artifacts_index.json (manifest of outputs)
- png plots + csv tables
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_performance_report(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Model" not in df.columns:
        # some reports use lowercase
        if "model" in df.columns:
            df = df.rename(columns={"model": "Model"})
    return df


def _find_latest_performance_report(run_dir: Path) -> Path:
    cands = sorted(run_dir.glob("performance_report_*.csv"))
    if not cands:
        raise FileNotFoundError(f"No performance_report_*.csv under {run_dir}")
    return cands[-1]


def _extract_ridge_meta_weights(snapshot_id: str) -> Tuple[pd.DataFrame, Optional[Path]]:
    """
    Attempts to load ridge_model.pkl from cache/model_snapshots/YYYYMMDD/<snapshot_id>/ridge_model.pkl
    Returns a dataframe of coefficients (feature -> weight).
    """
    snap_root = Path("cache") / "model_snapshots"
    ridge_path = None
    for day_dir in sorted(snap_root.glob("*")):
        p = day_dir / snapshot_id / "ridge_model.pkl"
        if p.exists():
            ridge_path = p
            break
    if ridge_path is None:
        return pd.DataFrame(), None

    # ridge_model.pkl may be saved via joblib (compressed) or pickle.
    model = None
    joblib_mod = _try_import("joblib")
    if joblib_mod is not None:
        try:
            model = joblib_mod.load(ridge_path)
        except Exception:
            model = None
    if model is None:
        import pickle
        try:
            model = pickle.loads(ridge_path.read_bytes())
        except Exception:
            return pd.DataFrame(), ridge_path

    coefs = getattr(model, "coef_", None)
    if coefs is None:
        return pd.DataFrame(), ridge_path

    # attempt to recover feature names
    feat_names = None
    for attr in ("feature_names_in_", "base_cols", "feature_names"):
        v = getattr(model, attr, None)
        if v is not None:
            try:
                feat_names = list(v)
                break
            except Exception:
                pass
    if feat_names is None:
        feat_names = [f"f{i}" for i in range(len(coefs))]

    # Heuristic mapping for the common 3-input ridge stacker used in this repo.
    # If the model didn't persist feature names, label them with the known base columns.
    if all(str(n).startswith("f") for n in feat_names):
        # Heuristic mapping based on common RidgeStacker inputs in this repo.
        if len(feat_names) == 3:
            feat_names = ["pred_catboost", "pred_elastic", "pred_xgb"]
        elif len(feat_names) == 4:
            feat_names = ["pred_catboost", "pred_elastic", "pred_xgb", "pred_lambdarank"]

    w = pd.DataFrame({"feature": feat_names, "weight": np.asarray(coefs).reshape(-1)})
    w["abs_weight"] = w["weight"].abs()
    w = w.sort_values("abs_weight", ascending=False).reset_index(drop=True)
    return w, ridge_path


def _plot_bar(df: pd.DataFrame, x: str, y: str, title: str, out_png: Path, top_k: int = 20):
    mpl = _try_import("matplotlib")
    if mpl is None:
        return
    import matplotlib.pyplot as plt

    d = df.copy().head(top_k)
    if d.empty:
        return
    plt.figure(figsize=(10, max(3, 0.35 * len(d) + 1)))
    plt.barh(d[x][::-1], d[y][::-1])
    plt.title(title)
    plt.xlabel(y)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _find_weekly_returns(run_dir: Path, model: str) -> Optional[Path]:
    cands = sorted(run_dir.glob(f"{model}_weekly_returns_*.csv"))
    return cands[-1] if cands else None


def _yearly_stats(returns: pd.Series) -> pd.DataFrame:
    # returns are per-period (e.g., T+10) simple returns (fraction)
    idx = pd.to_datetime(returns.index)
    df = pd.DataFrame({"date": idx, "ret": pd.to_numeric(returns.values, errors="coerce")}).dropna()
    df["year"] = df["date"].dt.year
    rows = []
    for y, g in df.groupby("year"):
        r = g["ret"].astype(float)
        cum = float((1.0 + r).prod() - 1.0)
        rows.append(
            {
                "year": int(y),
                "n_periods": int(len(r)),
                "mean_period_ret": float(r.mean()),
                "std_period_ret": float(r.std()),
                "cum_ret": cum,
                "win_rate": float((r > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def _dist_stats(returns: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if s.empty:
        return {}
    # use pandas moments (no scipy dependency)
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurt()),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _ff5_regression(
    dates: pd.DatetimeIndex,
    period_returns: pd.Series,
    horizon_days: int,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Optional: regress period excess returns on Ken French daily 5 factors aggregated to period.
    Returns (coef_table, fitted_df).
    """
    pdr = _try_import("pandas_datareader")
    sm = _try_import("statsmodels.api")
    if pdr is None or sm is None:
        return pd.DataFrame(), None

    import pandas_datareader.data as web
    import statsmodels.api as sm_api

    start = pd.to_datetime(dates.min()).date()
    end = pd.to_datetime(dates.max()).date()

    # Try to fetch daily 5 factors (2x3). pandas_datareader returns dict-like with DataFrame at [0]
    ff = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start, end)[0].copy()
    ff.index = pd.to_datetime(ff.index)
    ff = ff.sort_index()

    # Convert percent to fraction
    for c in ff.columns:
        ff[c] = pd.to_numeric(ff[c], errors="coerce") / 100.0

    # Aggregate daily factors to our holding periods: from date (inclusive) to next horizon_days trading rows in ff.
    rows = []
    for d in pd.to_datetime(dates).tz_localize(None):
        if d not in ff.index:
            # nearest next available
            loc = ff.index.searchsorted(d)
            if loc >= len(ff.index):
                continue
            d0 = ff.index[loc]
        else:
            d0 = d
        loc0 = ff.index.get_loc(d0)
        loc1 = min(loc0 + int(horizon_days), len(ff.index) - 1)
        seg = ff.iloc[loc0:loc1 + 1]
        if seg.empty:
            continue
        # Compound each factor series (approx)
        agg = {}
        for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]:
            if c in seg.columns:
                agg[c] = float((1.0 + seg[c]).prod() - 1.0)
        rows.append({"date": d, **agg})

    ff_period = pd.DataFrame(rows).dropna()
    if ff_period.empty:
        return pd.DataFrame(), None

    r = pd.DataFrame({"date": pd.to_datetime(dates).tz_localize(None), "ret": pd.to_numeric(period_returns.values, errors="coerce")})
    merged = r.merge(ff_period, on="date", how="inner").dropna()
    if merged.empty:
        return pd.DataFrame(), None

    y = merged["ret"] - merged.get("RF", 0.0)
    X = merged[[c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if c in merged.columns]].copy()
    X = sm_api.add_constant(X)
    model = sm_api.OLS(y, X).fit()

    coef = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "t": model.tvalues.values,
            "p": model.pvalues.values,
        }
    )
    coef.loc[coef["term"] == "const", "term"] = "alpha"
    fitted = merged.assign(excess=y, fitted=model.fittedvalues)
    return coef, fitted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output artifacts directory")
    ap.add_argument("--run-dir", required=True, help="Backtest run directory that contains performance_report_*.csv and *_weekly_returns_*.csv")
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--horizon-days", type=int, default=10)
    ap.add_argument("--top-model", type=str, default="ridge_stacking", help="Which model to use for FF regression / distribution stats (default ridge_stacking)")
    ap.add_argument("--features-json", type=str, default="results/t10_optimized_all_models/best_features_per_model.json")
    args = ap.parse_args()

    outdir = _ensure_dir(Path(args.outdir))
    run_dir = Path(args.run_dir)
    snapshot_id = str(args.snapshot_id)
    horizon = int(args.horizon_days)

    index: Dict[str, str] = {}

    # Ridge meta weights
    w, ridge_path = _extract_ridge_meta_weights(snapshot_id)
    if not w.empty:
        w_csv = outdir / "ridge_meta_weights.csv"
        w.to_csv(w_csv, index=False, encoding="utf-8")
        index["ridge_meta_weights_csv"] = str(w_csv).replace("\\", "/")
        if ridge_path:
            index["ridge_model_path"] = str(ridge_path).replace("\\", "/")
        w_png = outdir / "ridge_meta_weights_top20.png"
        _plot_bar(w, x="feature", y="weight", title="Ridge meta-learner weights (Top 20 by |weight|)", out_png=w_png, top_k=20)
        if w_png.exists():
            index["ridge_meta_weights_png"] = str(w_png).replace("\\", "/")

    # Feature list (best_features_per_model.json)
    fpath = Path(args.features_json)
    if fpath.exists():
        try:
            feat = json.loads(fpath.read_text(encoding="utf-8"))
            rows = []
            for model, feats in feat.items():
                if isinstance(feats, list):
                    for i, f in enumerate(feats, 1):
                        rows.append({"model": str(model), "rank": i, "feature": str(f)})
            feat_df = pd.DataFrame(rows)
            if not feat_df.empty:
                feat_csv = outdir / "feature_list_best_per_model.csv"
                feat_df.to_csv(feat_csv, index=False, encoding="utf-8")
                index["feature_list_csv"] = str(feat_csv).replace("\\", "/")
        except Exception:
            pass

    # Performance report copy
    perf_path = _find_latest_performance_report(run_dir)
    perf = _load_performance_report(perf_path)
    perf_out = outdir / "performance_report.csv"
    perf.to_csv(perf_out, index=False, encoding="utf-8")
    index["performance_report_csv"] = str(perf_out).replace("\\", "/")
    index["performance_report_source_csv"] = str(perf_path).replace("\\", "/")

    # Long-only distribution + yearly stats (all models found in performance report)
    models_in_report: List[str] = []
    if "Model" in perf.columns:
        models_in_report = [str(m) for m in perf["Model"].dropna().astype(str).tolist()]
    elif "model" in perf.columns:
        models_in_report = [str(m) for m in perf["model"].dropna().astype(str).tolist()]
    models_in_report = sorted(set(models_in_report))

    yearly_rows: List[pd.DataFrame] = []
    dist_rows: List[Dict[str, float]] = []
    for m in models_in_report:
        wk = _find_weekly_returns(run_dir, m)
        if not wk or not wk.exists():
            continue
        wk_df = pd.read_csv(wk)
        if "date" not in wk_df.columns or "top_return_net" not in wk_df.columns:
            continue
        wk_df["date"] = pd.to_datetime(wk_df["date"])
        wk_df = wk_df.sort_values("date")
        r = wk_df.set_index("date")["top_return_net"].astype(float)

        dist = _dist_stats(r)
        if dist:
            dist_rows.append({"model": m, **dist})
        yr = _yearly_stats(r)
        if not yr.empty:
            yr["model"] = m
            yearly_rows.append(yr)

    if dist_rows:
        dist_df = pd.DataFrame(dist_rows).sort_values("model")
        dist_csv = outdir / "dist_stats_all_models.csv"
        dist_df.to_csv(dist_csv, index=False, encoding="utf-8")
        index["dist_stats_all_models_csv"] = str(dist_csv).replace("\\", "/")

    if yearly_rows:
        yr_all = pd.concat(yearly_rows, ignore_index=True)
        yr_csv = outdir / "yearly_stats_all_models.csv"
        yr_all.to_csv(yr_csv, index=False, encoding="utf-8")
        index["yearly_stats_all_models_csv"] = str(yr_csv).replace("\\", "/")

    # Optional FF5 regression only for top_model (can be slow / dependency on pandas_datareader)
    model = str(args.top_model)
    wk = _find_weekly_returns(run_dir, model)
    if wk and wk.exists():
        wk_df = pd.read_csv(wk)
        if "date" in wk_df.columns and "top_return_net" in wk_df.columns:
            wk_df["date"] = pd.to_datetime(wk_df["date"])
            wk_df = wk_df.sort_values("date")
            r = wk_df.set_index("date")["top_return_net"].astype(float)
            coef, fitted = _ff5_regression(r.index, r, horizon_days=horizon)
            if not coef.empty:
                coef_csv = outdir / f"{model}_ff5_regression.csv"
                coef.to_csv(coef_csv, index=False, encoding="utf-8")
                index["ff5_regression_csv"] = str(coef_csv).replace("\\", "/")
            if fitted is not None and not fitted.empty:
                fitted_csv = outdir / f"{model}_ff5_regression_fitted.csv"
                fitted.to_csv(fitted_csv, index=False, encoding="utf-8")
                index["ff5_regression_fitted_csv"] = str(fitted_csv).replace("\\", "/")

    (outdir / "artifacts_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








