#!/usr/bin/env python3
"""
Audit a MultiIndex factor dataset for common "future-known data" / backtest validity issues.

This does NOT prove your dataset is point-in-time, but it catches common failure modes:
- Target column accidentally standardized/capped/fillna'ed (so it is no longer a real return)
- Columns with forward-looking names ("fwd", "future", "lead", ...)
- Suspiciously high correlation with target (often indicates leakage or target contamination)

Usage:
  python scripts/audit_factor_dataset_pit.py --data-file data/factor_exports/factors/factors_all.parquet --outdir result/factor_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
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

    # try to build MultiIndex from columns
    if {"date", "ticker"}.issubset(df.columns):
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
        out["ticker"] = out["ticker"].astype(str)
        return out.set_index(["date", "ticker"]).sort_index()

    raise ValueError("Expected MultiIndex(date,ticker) or columns date,ticker")


def _find_target_col(df: pd.DataFrame) -> str:
    for c in ("ret_fwd_10d", "target", "ret_fwd", "y"):
        if c in df.columns:
            return c
    raise ValueError("Could not find target column (expected one of ret_fwd_10d/target/ret_fwd/y).")


def _robust_corr(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").astype(float)
    b = pd.to_numeric(y, errors="coerce").astype(float)
    m = a.notna() & b.notna()
    if m.sum() < 200:
        return float("nan")
    return float(np.corrcoef(a[m].values, b[m].values)[0, 1])


def audit(data_file: str, outdir: str, sample_rows: int = 1_000_000, seed: int = 42) -> dict:
    df = pd.read_parquet(data_file)
    df = _ensure_multiindex(df)

    target_col = _find_target_col(df)
    report: dict = {}
    report["data_file"] = str(data_file)
    report["rows"] = int(len(df))
    report["cols"] = [str(c) for c in df.columns]
    report["target_col"] = target_col
    report["date_min"] = str(pd.to_datetime(df.index.get_level_values("date")).min())
    report["date_max"] = str(pd.to_datetime(df.index.get_level_values("date")).max())
    report["tickers"] = int(pd.Index(df.index.get_level_values("ticker")).nunique())

    # --- Target sanity checks ---
    y = pd.to_numeric(df[target_col], errors="coerce").astype(float)
    y_q = y.quantile([0.001, 0.01, 0.5, 0.99, 0.999]).to_dict()
    report["target_quantiles"] = {str(k): float(v) if np.isfinite(v) else None for k, v in y_q.items()}
    report["target_abs_gt_1_frac"] = float((y.abs() > 1.0).mean())  # raw 10d returns should almost never exceed 100%

    # Detect "target standardized per date" fingerprint: mean~0 and std~1 for most dates
    by_date = pd.DataFrame({"y": y}).groupby(level="date")["y"]
    mu = by_date.mean()
    sd = by_date.std(ddof=0)
    good_mean = (mu.abs() < 0.05).mean()
    good_std = ((sd > 0.8) & (sd < 1.2)).mean()
    report["target_per_date_mean_abs_lt_0p05_frac"] = float(good_mean)
    report["target_per_date_std_in_0p8_1p2_frac"] = float(good_std)
    report["target_looks_standardized_per_date"] = bool(good_mean > 0.8 and good_std > 0.8)

    # --- Forward-looking name checks ---
    forward_tokens = ("fwd", "forward", "future", "lead", "next", "t+")
    suspect_name_cols = [c for c in df.columns if any(tok in str(c).lower() for tok in forward_tokens) and str(c) != target_col]
    report["suspect_forward_name_cols"] = suspect_name_cols[:200]

    # --- Correlation scan (sampled) ---
    numeric_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    if sample_rows and len(df) > sample_rows:
        samp = df.sample(n=int(sample_rows), random_state=int(seed))
    else:
        samp = df
    y_s = pd.to_numeric(samp[target_col], errors="coerce").astype(float)
    corrs = []
    for c in numeric_cols:
        r = _robust_corr(samp[c], y_s)
        if np.isfinite(r):
            corrs.append((str(c), float(r)))
    corrs.sort(key=lambda t: abs(t[1]), reverse=True)
    report["top_abs_corr_with_target"] = [{"col": c, "corr": r} for c, r in corrs[:50]]

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(report["top_abs_corr_with_target"]).to_csv(out / "top_corr_with_target.csv", index=False)
    pd.DataFrame({"suspect_forward_name_cols": report["suspect_forward_name_cols"]}).to_csv(
        out / "suspect_forward_name_cols.csv", index=False
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", type=str, default="data/factor_exports/factors/factors_all.parquet")
    p.add_argument("--outdir", type=str, default="result/factor_audit")
    p.add_argument("--sample-rows", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rep = audit(args.data_file, args.outdir, sample_rows=int(args.sample_rows), seed=int(args.seed))
    print(f"WROTE {args.outdir}")
    print(f"target_col={rep['target_col']}")
    print(f"target_looks_standardized_per_date={rep['target_looks_standardized_per_date']}")


if __name__ == "__main__":
    main()



