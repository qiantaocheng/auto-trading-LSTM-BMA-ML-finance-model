#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a NASDAQ-only Top-K portfolio curve from saved ridge_stacking predictions, with optional feature filter.

What it does:
  - Load predictions parquet/csv: columns date,ticker,prediction,actual
  - Restrict universe to tickers in a provided tickers file (e.g., NASDAQ listed)
  - Each week: sort by prediction, optionally apply evidence-based feature filter (z-score on same-day allfac),
    then pick Top-K and compute realized weekly return = mean(actual) (percent units).
  - Output equity curve CSV and print the latest week's Top-K tickers.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional

import numpy as np
import pandas as pd


def load_tickers_file(path: str) -> List[str]:
    tickers: List[str] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [p for token in s.split(",") for p in token.split()]
            for p in parts:
                t = str(p).strip().strip("'\"").upper()
                if t:
                    tickers.append(t)
    return list(dict.fromkeys(tickers))


def latest(path_or_glob: str) -> str:
    paths = sorted(glob.glob(path_or_glob)) if any(ch in path_or_glob for ch in "*?[]") else [path_or_glob]
    if not paths:
        raise FileNotFoundError(f"No files match: {path_or_glob}")
    p = paths[-1]
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def load_predictions(path_or_glob: str) -> pd.DataFrame:
    p = latest(path_or_glob)
    if p.lower().endswith(".parquet"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    for c in ["date", "ticker", "prediction", "actual"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {p}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    return df


def load_allfac_all(data_dir: str) -> pd.DataFrame:
    """Load full allfac dataset (manifest + batches) to allow same-day z-scores for filtering."""
    manifest_path = os.path.join(data_dir, "manifest.parquet")
    if not os.path.exists(manifest_path):
        # allow passing a single parquet file
        if data_dir.lower().endswith(".parquet") and os.path.exists(data_dir):
            df = pd.read_parquet(data_dir)
            return _standardize_multiindex(df)
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_parquet(manifest_path)
    parts = []
    for _, row in manifest.iterrows():
        bid = int(row["batch_id"])
        pf = os.path.join(data_dir, f"polygon_factors_batch_{bid:04d}.parquet")
        if os.path.exists(pf):
            parts.append(pd.read_parquet(pf))
    if not parts:
        raise RuntimeError("No allfac batch files loaded.")
    df = pd.concat(parts, axis=0)
    return _standardize_multiindex(df)


def _standardize_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        # try to normalize names/values
        names = [n.lower() if isinstance(n, str) else "" for n in df.index.names]
        if "date" in names and ("ticker" in names or "symbol" in names):
            date_level = names.index("date")
            other_level = 1 - date_level
            dates = pd.to_datetime(df.index.get_level_values(date_level)).tz_localize(None).normalize()
            tickers = df.index.get_level_values(other_level).astype(str).str.upper().str.strip()
        else:
            r = df.reset_index()
            if "date" not in r.columns:
                raise ValueError("Cannot find date in MultiIndex")
            ticker_col = "ticker" if "ticker" in r.columns else ("symbol" if "symbol" in r.columns else None)
            if not ticker_col:
                raise ValueError("Cannot find ticker in MultiIndex")
            dates = pd.to_datetime(r["date"]).dt.tz_localize(None).dt.normalize()
            tickers = r[ticker_col].astype(str).str.upper().str.strip()
            df = r.drop(columns=[c for c in ["symbol"] if c in r.columns])
    else:
        raise ValueError("allfac must be MultiIndex(date,ticker)")
    out = df.copy()
    out.index = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def zscore_cross_section(day: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    x = day[cols].copy().replace([np.inf, -np.inf], np.nan)
    mu = x.mean(axis=0, skipna=True)
    sd = x.std(axis=0, skipna=True).replace(0, np.nan)
    return (x - mu) / sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=str, required=True, help="ridge_stacking_predictions_*.parquet (path or glob)")
    ap.add_argument("--tickers-file", type=str, required=True, help="NASDAQ ticker list file")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--save-picks", type=str, default=None, help="Optional CSV of weekly picks")
    ap.add_argument("--apply-feature-filter", action="store_true")
    ap.add_argument("--allfac-dir", type=str, default="data/factor_exports/factors", help="Needed only if --apply-feature-filter (feature z-scores)")
    ap.add_argument("--vol-z-max", type=float, default=2.0)
    ap.add_argument("--near-high-z-min", type=float, default=-1.5)
    ap.add_argument("--vol-feature", type=str, default="hist_vol_40d")
    ap.add_argument("--near-high-feature", type=str, default="near_52w_high")
    args = ap.parse_args()

    top_k = int(args.top_k)
    tickers = set(load_tickers_file(args.tickers_file))
    preds = load_predictions(args.predictions).dropna(subset=["prediction", "actual"])

    # NASDAQ-only restriction
    preds = preds[preds["ticker"].isin(tickers)].copy()
    if preds.empty:
        raise RuntimeError("No predictions left after NASDAQ ticker filter.")

    allfac = None
    if bool(args.apply_feature_filter):
        allfac = load_allfac_all(args.allfac_dir)

    rows = []
    picks_rows = []
    for d, g in preds.groupby("date", sort=True):
        g = g.sort_values("prediction", ascending=False)
        if len(g) < top_k:
            continue

        if allfac is not None:
            try:
                day = allfac.xs(pd.to_datetime(d), level="date", drop_level=True)
            except KeyError:
                day = None

            if day is not None and (args.vol_feature in day.columns) and (args.near_high_feature in day.columns):
                z = zscore_cross_section(day, [args.vol_feature, args.near_high_feature])

                def passes(t: str) -> bool:
                    if t not in z.index:
                        return False
                    vz = z.at[t, args.vol_feature]
                    nh = z.at[t, args.near_high_feature]
                    if pd.notna(vz) and pd.notna(nh):
                        if (float(vz) > float(args.vol_z_max)) and (float(nh) < float(args.near_high_z_min)):
                            return False
                    return True

                selected = []
                for t in g["ticker"].tolist():
                    if passes(t):
                        selected.append(t)
                        if len(selected) >= top_k:
                            break
                g2 = g[g["ticker"].isin(selected)].head(top_k)
            else:
                g2 = g.head(top_k)
        else:
            g2 = g.head(top_k)

        if len(g2) < top_k:
            continue

        weekly_actual = float(g2["actual"].mean())  # percent units
        weekly_pred = float(g2["prediction"].mean())
        rows.append(
            {
                "date": pd.to_datetime(d),
                "top_k": top_k,
                "weekly_expected_pred": weekly_pred,
                "weekly_return_pct": weekly_actual,
            }
        )
        picks_rows.append(
            {
                "date": pd.to_datetime(d),
                "tickers": ",".join(g2["ticker"].tolist()),
                "mean_pred": weekly_pred,
                "mean_actual_pct": weekly_actual,
            }
        )

    curve = pd.DataFrame(rows).sort_values("date")
    if curve.empty:
        raise RuntimeError("No weekly rows produced.")
    curve["weekly_return_decimal"] = curve["weekly_return_pct"] / 100.0
    curve["cum_nav"] = (1.0 + curve["weekly_return_decimal"]).cumprod()
    curve["cum_return_pct"] = (curve["cum_nav"] - 1.0) * 100.0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    curve.to_csv(args.output, index=False)
    if args.save_picks:
        os.makedirs(os.path.dirname(args.save_picks) or ".", exist_ok=True)
        pd.DataFrame(picks_rows).sort_values("date").to_csv(args.save_picks, index=False)

    last = pd.DataFrame(picks_rows).sort_values("date").iloc[-1]
    print(f"WROTE {args.output}")
    if args.save_picks:
        print(f"WROTE {args.save_picks}")
    print(f"LATEST_DATE {pd.to_datetime(last['date']).date()} TOP{top_k} {last['tickers']}")
    print(f"FINAL_NAV {float(curve['cum_nav'].iloc[-1]):.6f} FINAL_RETURN {float(curve['cum_return_pct'].iloc[-1]):.3f}% weeks={len(curve)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


