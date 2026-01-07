#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add RSRS (rolling regression beta: High ~ Low) to an existing MultiIndex factor dataset.

RSRS definition:
  beta_t = slope( High ~ Low ) over rolling window N per ticker.
We compute it efficiently as:
  beta = Cov(Low, High) / Var(Low)

Supports:
- Single parquet file (MultiIndex: date,ticker)
- Directory with manifest.parquet pointing to shard files (each MultiIndex)

Typical usage:
  python scripts/add_rsrs_to_multiindex_factors.py \
    --input data/factor_exports/factors \
    --output data/factor_exports/factors_rsrs \
    --window 18
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Input parquet file OR directory containing manifest.parquet")
    p.add_argument("--output", type=str, required=True, help="Output parquet file OR output directory")
    p.add_argument("--window", type=int, default=18)
    p.add_argument("--feature-name", type=str, default="rsrs_beta_18")
    p.add_argument("--high-col", type=str, default="High")
    p.add_argument("--low-col", type=str, default="Low")
    p.add_argument("--fillna", type=float, default=0.0)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("Expected MultiIndex index (date, ticker)")
    names = [(n or "") for n in df.index.names]
    names = [
        "date" if str(n).lower() == "date" else ("ticker" if str(n).lower() in ("ticker", "symbol") else n)
        for n in names
    ]
    df.index.names = names
    if "date" in df.index.names and "ticker" in df.index.names:
        if df.index.names[:2] != ["date", "ticker"]:
            df = df.reorder_levels(["date", "ticker"]).sort_index()
    return df


def compute_rsrs_beta(df: pd.DataFrame, window: int, high_col: str, low_col: str, feature_name: str, fillna: float) -> pd.Series:
    df = _ensure_multiindex(df)
    if high_col not in df.columns or low_col not in df.columns:
        raise ValueError(
            f"RSRS requires columns {low_col!r} and {high_col!r}. "
            f"Missing: {[c for c in (low_col, high_col) if c not in df.columns]}"
        )

    eps = 1e-10
    low = df[low_col].astype(float).replace([np.inf, -np.inf], np.nan)
    high = df[high_col].astype(float).replace([np.inf, -np.inf], np.nan)

    g = low.groupby(level="ticker")
    ex = g.transform(lambda s: s.rolling(window, min_periods=window).mean())
    ey = high.groupby(level="ticker").transform(lambda s: s.rolling(window, min_periods=window).mean())

    exy = (low * high).groupby(level="ticker").transform(lambda s: s.rolling(window, min_periods=window).mean())
    ex2 = (low * low).groupby(level="ticker").transform(lambda s: s.rolling(window, min_periods=window).mean())

    cov_xy = exy - ex * ey
    var_x = ex2 - ex * ex
    beta = (cov_xy / (var_x + eps)).replace([np.inf, -np.inf], np.nan).fillna(fillna)
    beta.name = feature_name
    return beta


def process_file(in_file: Path, out_file: Path, *, window: int, high_col: str, low_col: str, feature_name: str, fillna: float) -> None:
    logger.info(f"Reading {in_file}")
    df = pd.read_parquet(in_file)
    df = _ensure_multiindex(df)

    logger.info(f"Computing {feature_name} window={window} ...")
    beta = compute_rsrs_beta(df, window=window, high_col=high_col, low_col=low_col, feature_name=feature_name, fillna=fillna)

    df[feature_name] = beta.astype(float)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {out_file}")
    df.to_parquet(out_file)


def process_dir(in_dir: Path, out_dir: Path, *, window: int, high_col: str, low_col: str, feature_name: str, fillna: float) -> None:
    manifest_path = in_dir / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.parquet not found under {in_dir}")

    manifest = pd.read_parquet(manifest_path)
    if "file" not in manifest.columns:
        raise ValueError("manifest.parquet missing 'file' column")

    out_dir.mkdir(parents=True, exist_ok=True)
    updated_rows = []

    for _, r in manifest.iterrows():
        f = Path(str(r["file"]))
        if not f.is_absolute():
            # treat as relative to input dir (and also allow relative-to-repo)
            cand1 = (in_dir / f).resolve()
            if cand1.exists():
                f = cand1
            else:
                f = Path(str(r["file"])).resolve()

        out_file = out_dir / Path(f).name
        process_file(f, out_file, window=window, high_col=high_col, low_col=low_col, feature_name=feature_name, fillna=fillna)

        nr = dict(r)
        nr["file"] = str(out_file)
        updated_rows.append(nr)

    out_manifest = pd.DataFrame(updated_rows)
    out_manifest_path = out_dir / "manifest.parquet"
    out_manifest.to_parquet(out_manifest_path, index=False)
    logger.info(f"Wrote updated manifest: {out_manifest_path}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s - %(message)s")

    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.is_dir():
        process_dir(
            in_path,
            out_path,
            window=int(args.window),
            high_col=str(args.high_col),
            low_col=str(args.low_col),
            feature_name=str(args.feature_name),
            fillna=float(args.fillna),
        )
    else:
        process_file(
            in_path,
            out_path,
            window=int(args.window),
            high_col=str(args.high_col),
            low_col=str(args.low_col),
            feature_name=str(args.feature_name),
            fillna=float(args.fillna),
        )


if __name__ == "__main__":
    main()


