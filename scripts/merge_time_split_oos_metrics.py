#!/usr/bin/env python3
"""
Merge predictive + backtest metrics from time-split `report_df.csv` into `oos_metrics.csv/json`.

Why:
- `scripts/time_split_80_20_oos_eval.py` writes `oos_metrics.csv/json` (Top-N vs QQQ).
- `report_df.csv` (from ComprehensiveModelBacktest) already contains IC/RankIC/MSE/MAE/R2 and
  Top-30 backtest summaries (avg_top_return, avg_top_return_net, turnover, etc.) on the SAME test window.

This script makes `oos_metrics.csv/json` the single citation point for both:
- OOS Top-N vs QQQ metrics (percent units), AND
- predictive metrics (IC/RankIC/MSE/MAE/R2) + Top-30 backtest summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Path to a time-split run dir that contains report_df.csv and oos_metrics.csv (e.g., results/.../run_YYYYMMDD_HHMMSS)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    report_path = run_dir / "report_df.csv"
    oos_csv_path = run_dir / "oos_metrics.csv"
    oos_json_path = run_dir / "oos_metrics.json"

    if not report_path.exists():
        raise FileNotFoundError(f"NOT FOUND: {report_path}")
    if not oos_csv_path.exists():
        raise FileNotFoundError(f"NOT FOUND: {oos_csv_path}")

    rep = pd.read_csv(report_path)
    oos = pd.read_csv(oos_csv_path)
    if oos.empty:
        raise RuntimeError(f"Empty file: {oos_csv_path}")

    model = str(oos.loc[0, "model"]) if "model" in oos.columns else None
    if not model:
        raise RuntimeError(f"oos_metrics.csv missing 'model' column: {oos_csv_path}")

    if "Model" not in rep.columns:
        raise RuntimeError(f"report_df.csv missing 'Model' column: {report_path}")

    rr = rep.loc[rep["Model"].astype(str) == model].head(1)
    if rr.empty:
        raise RuntimeError(f"Could not find model={model!r} row in {report_path}")

    row = rr.iloc[0].to_dict()
    add_keys = [
        # predictive
        "IC",
        "IC_pvalue",
        "Rank_IC",
        "Rank_IC_pvalue",
        "MSE",
        "MAE",
        "R2",
        # backtest summary (Top-30)
        "avg_top_return",
        "avg_top_return_net",
        "avg_top_turnover",
        "avg_top_cost",
        "win_rate",
        "long_short_sharpe",
        "cost_bps",
    ]

    for k in add_keys:
        if k in row:
            oos.loc[0, k] = row.get(k)

    # Normalize numeric-ish fields where possible
    for k in add_keys:
        if k in oos.columns:
            oos.loc[0, k] = _safe_float(oos.loc[0, k])

    oos.to_csv(oos_csv_path, index=False, encoding="utf-8")
    oos_json_path.write_text(json.dumps(oos.loc[0].to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"UPDATED {oos_csv_path}")
    print(f"UPDATED {oos_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



