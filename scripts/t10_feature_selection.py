#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T+10 Feature Selection on MultiIndex-style factor dataset.

Reads a parquet containing columns: date, ticker, <features...>, target
or a shard directory containing `factors_batch_*.parquet` (MultiIndex ok).

Outputs:
- feature_ic_summary.csv : mean RankIC, std, t-stat, win-rate, coverage
- feature_corr.csv       : feature-feature Pearson correlation matrix (sampled)
- selected_features.txt  : greedy-selected feature list (de-correlated)
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


EXCLUDE = {"date", "ticker", "Close", "target"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", type=str, default="data/factor_exports/factors/factors_all.parquet")
    p.add_argument("--data-dir", type=str, default=None, help="Optional: directory with factors_batch_*.parquet shards")
    p.add_argument("--output-dir", type=str, default="results/t10_feature_selection")
    p.add_argument("--date-start", type=str, default=None)
    p.add_argument("--date-end", type=str, default=None)
    p.add_argument("--max-dates", type=int, default=260, help="Sample at most N dates for IC computation (uniform).")
    p.add_argument("--corr-max", type=float, default=0.90, help="Max abs correlation allowed between selected features.")
    p.add_argument("--corr-sample-rows", type=int, default=300_000, help="Rows to sample for correlation matrix.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _read_dataset(args: argparse.Namespace, cols: Optional[List[str]] = None) -> pd.DataFrame:
    if args.data_dir:
        d = Path(args.data_dir)
        files = sorted(d.glob("factors_batch_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No factors_batch_*.parquet under {d}")
        frames = []
        for f in files:
            df = pd.read_parquet(f, columns=cols)
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            frames.append(df)
        out = pd.concat(frames, axis=0, ignore_index=True)
        return out

    df = pd.read_parquet(args.data_file, columns=cols)
    # If MultiIndex, reset
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    return df


def _uniform_sample_dates(dates: pd.Series, max_dates: int) -> List[pd.Timestamp]:
    uniq = pd.Series(pd.to_datetime(dates).dt.normalize().unique()).sort_values()
    uniq = [pd.Timestamp(x) for x in uniq.tolist()]
    if max_dates is None or max_dates <= 0 or len(uniq) <= max_dates:
        return uniq
    # uniform pick
    idx = np.linspace(0, len(uniq) - 1, max_dates).round().astype(int)
    picked = [uniq[i] for i in idx]
    return picked


def _rank_ic_by_date(df: pd.DataFrame, features: List[str], max_dates: int) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str)

    dates = _uniform_sample_dates(df["date"], max_dates=max_dates)
    logger.info(f"IC dates used: {len(dates)}")

    ic_rows = []
    g = df.groupby("date", sort=True)
    for d in dates:
        if d not in g.indices:
            continue
        day = df.loc[g.indices[d], :]
        if len(day) < 50:
            continue
        y = pd.to_numeric(day["target"], errors="coerce")
        if y.notna().sum() < 50:
            continue
        xs = day[features].apply(pd.to_numeric, errors="coerce")
        ic = xs.corrwith(y, method="spearman")
        ic_rows.append(ic.rename(d))

    if not ic_rows:
        raise RuntimeError("No IC rows computed (check target coverage / date filters).")
    ic_df = pd.concat(ic_rows, axis=1).T  # index=date, cols=features
    ic_df.index.name = "date"
    return ic_df


def _summarize_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    n = ic_df.notna().sum(axis=0).astype(int)
    mean = ic_df.mean(axis=0)
    std = ic_df.std(axis=0)
    win = (ic_df > 0).mean(axis=0)
    t = mean / (std / np.sqrt(n.clip(lower=1)))
    out = pd.DataFrame(
        {
            "n_dates": n,
            "mean_rank_ic": mean,
            "std_rank_ic": std,
            "t_stat": t.replace([np.inf, -np.inf], np.nan),
            "win_rate": win,
            "abs_mean_rank_ic": mean.abs(),
        }
    ).sort_values("abs_mean_rank_ic", ascending=False)
    return out


def _corr_matrix_sample(df: pd.DataFrame, features: List[str], sample_rows: int) -> pd.DataFrame:
    if sample_rows and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=42)
    x = df[features].apply(pd.to_numeric, errors="coerce")
    return x.corr()


def _greedy_select(summary: pd.DataFrame, corr: pd.DataFrame, corr_max: float) -> List[str]:
    ordered = summary.index.tolist()
    selected: List[str] = []
    for f in ordered:
        ok = True
        for s in selected:
            c = corr.loc[f, s] if (f in corr.index and s in corr.columns) else np.nan
            if pd.notna(c) and abs(float(c)) >= corr_max:
                ok = False
                break
        if ok:
            selected.append(f)
    return selected


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # First read only columns to discover feature set
    base = _read_dataset(args, cols=None)
    if "target" not in base.columns:
        raise ValueError("Dataset must contain 'target' column (T+10).")
    if "date" not in base.columns or "ticker" not in base.columns:
        raise ValueError("Dataset must contain 'date' and 'ticker' columns.")

    # Date filters
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    if args.date_start:
        base = base[base["date"] >= pd.to_datetime(args.date_start).normalize()]
    if args.date_end:
        base = base[base["date"] <= pd.to_datetime(args.date_end).normalize()]

    features = [c for c in base.columns if c not in EXCLUDE]
    features = [c for c in features if c not in ("date", "ticker")]
    # Keep only numeric-ish feature cols (exclude accidental objects)
    features = [c for c in features if c not in {"date", "ticker"}]

    logger.info(f"Rows: {len(base):,} | Unique tickers: {base['ticker'].nunique()} | Unique dates: {base['date'].nunique()}")
    logger.info(f"Candidate features: {features}")

    # RankIC
    ic_df = _rank_ic_by_date(base, features, max_dates=int(args.max_dates))
    summary = _summarize_ic(ic_df)
    summary_path = out_dir / "feature_ic_summary.csv"
    summary.to_csv(summary_path, index=True)
    logger.info(f"WROTE {summary_path}")

    # Corr matrix (sampled)
    corr = _corr_matrix_sample(base, features, sample_rows=int(args.corr_sample_rows))
    corr_path = out_dir / "feature_corr.csv"
    corr.to_csv(corr_path, index=True)
    logger.info(f"WROTE {corr_path}")

    # Greedy selection
    selected = _greedy_select(summary, corr, corr_max=float(args.corr_max))
    sel_path = out_dir / "selected_features.txt"
    sel_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    logger.info(f"WROTE {sel_path} (n={len(selected)})")

    # Print quick top
    topk = min(10, len(summary))
    logger.info("TOP FEATURES (by abs mean RankIC):")
    logger.info("\n" + summary.head(topk).to_string())
    logger.info(f"SELECTED ({len(selected)}): {selected}")


if __name__ == "__main__":
    main()


