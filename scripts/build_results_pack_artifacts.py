#!/usr/bin/env python3
"""
Build Results Pack artifacts (inventory + run summary) from existing repo outputs.

Hard rules supported:
- No fabricated numbers: summary is derived only from existing CSVs in an output folder.
- Robust to empty scans: inventory CSV will still be written with headers.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_glob(pattern: str) -> Optional[str]:
    matches = glob.glob(pattern)
    return matches[0] if matches else None


@dataclass
class ResultsPackSummary:
    run_dir: str
    generated_at_utc: str
    data_file: Optional[str]
    tickers_file_applied: Optional[str]
    rebalance_mode: Optional[str]
    target_horizon_days: Optional[int]
    cost_bps: Optional[float]
    evaluation: Dict[str, Any]
    primary_objective: str
    best_model_by_primary_objective: Dict[str, Any]
    source_files: Dict[str, Optional[str]]


def build_run_summary(
    out_run_dir: str,
    data_file: Optional[str],
    tickers_file_applied: Optional[str],
    rebalance_mode: Optional[str],
    target_horizon_days: Optional[int],
    primary_objective_col: str,
) -> ResultsPackSummary:
    perf_csv = _first_glob(os.path.join(out_run_dir, "performance_report_*.csv"))
    if not perf_csv:
        raise FileNotFoundError(
            f"NOT FOUND: {out_run_dir} missing performance_report_*.csv (expected output of scripts/comprehensive_model_backtest.py)"
        )

    df = pd.read_csv(perf_csv)
    if primary_objective_col not in df.columns:
        raise KeyError(
            f"NOT FOUND: column {primary_objective_col!r} in {perf_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    # Best model on objective
    best_idx = df[primary_objective_col].astype(float).idxmax()
    best_row = df.loc[best_idx].to_dict()

    # Attempt to locate ridge_stacking weekly file to infer evaluation date range / count
    weekly_csv = _first_glob(os.path.join(out_run_dir, "ridge_stacking_weekly_returns_*.csv"))
    evaluation: Dict[str, Any] = {}
    if weekly_csv and os.path.exists(weekly_csv):
        weekly = pd.read_csv(weekly_csv)
        if "date" in weekly.columns:
            weekly["date"] = pd.to_datetime(weekly["date"])
            evaluation["n_rebalance_periods"] = int(len(weekly))
            evaluation["start_date"] = str(weekly["date"].min().date())
            evaluation["end_date"] = str(weekly["date"].max().date())
        if "n_stocks" in weekly.columns:
            evaluation["avg_n_stocks_per_date"] = float(pd.to_numeric(weekly["n_stocks"], errors="coerce").mean())
    else:
        weekly_csv = None

    cost_bps = None
    if "cost_bps" in df.columns and len(df) > 0:
        try:
            cost_bps = float(df["cost_bps"].iloc[0])
        except Exception:
            cost_bps = None

    summary = ResultsPackSummary(
        run_dir=out_run_dir.replace("\\", "/"),
        generated_at_utc=_utc_now_iso(),
        data_file=data_file,
        tickers_file_applied=tickers_file_applied,
        rebalance_mode=rebalance_mode,
        target_horizon_days=target_horizon_days,
        cost_bps=cost_bps,
        evaluation=evaluation,
        primary_objective=f"{primary_objective_col} (from performance_report_*.csv)",
        best_model_by_primary_objective={
            "Model": best_row.get("Model"),
            primary_objective_col: best_row.get(primary_objective_col),
            "avg_top_return": best_row.get("avg_top_return"),
            "Rank_IC": best_row.get("Rank_IC"),
            "IC": best_row.get("IC"),
            "long_short_sharpe": best_row.get("long_short_sharpe"),
            "avg_top_turnover": best_row.get("avg_top_turnover"),
            "avg_top_cost": best_row.get("avg_top_cost"),
        },
        source_files={
            "performance_report_csv": perf_csv.replace("\\", "/"),
            "ridge_stacking_weekly_returns_csv": weekly_csv.replace("\\", "/") if weekly_csv else None,
            "buckets_vs_nasdaq_csv": os.path.join(out_run_dir, "buckets_vs_nasdaq.csv").replace("\\", "/")
            if os.path.exists(os.path.join(out_run_dir, "buckets_vs_nasdaq.csv"))
            else None,
            "per_model_topN_vs_benchmark_csv": os.path.join(out_run_dir, "per_model_topN_vs_benchmark.csv").replace("\\", "/")
            if os.path.exists(os.path.join(out_run_dir, "per_model_topN_vs_benchmark.csv"))
            else None,
        },
    )
    return summary


def build_results_inventory(results_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not os.path.isdir(results_dir):
        # return empty but with headers
        return pd.DataFrame(
            columns=[
                "run_dir",
                "has_performance_report",
                "has_snapshot_id",
                "has_sweep_results",
                "has_predictions_parquet",
                "models_in_report",
            ]
        )

    for name in sorted(os.listdir(results_dir)):
        p = os.path.join(results_dir, name)
        if not os.path.isdir(p):
            continue

        perf_reports = glob.glob(os.path.join(p, "performance_report*.csv"))
        snapshot_txts = glob.glob(os.path.join(p, "**", "snapshot_id.txt"), recursive=True)
        sweep_results = glob.glob(os.path.join(p, "**", "sweep_results.csv"), recursive=True)
        pred_parquets = glob.glob(os.path.join(p, "**", "*predictions*.parquet"), recursive=True)

        has_any = bool(perf_reports or snapshot_txts or sweep_results or pred_parquets)
        if not has_any:
            continue

        models_in_report = None
        if perf_reports:
            try:
                t = pd.read_csv(perf_reports[0])
                if "Model" in t.columns:
                    models_in_report = ",".join(list(t["Model"].astype(str).values))
            except Exception:
                models_in_report = None

        rows.append(
            {
                "run_dir": p.replace("\\", "/"),
                "has_performance_report": bool(perf_reports),
                "has_snapshot_id": bool(snapshot_txts),
                "has_sweep_results": bool(sweep_results),
                "has_predictions_parquet": bool(pred_parquets),
                "models_in_report": models_in_report,
            }
        )

    inv = pd.DataFrame(rows)
    if inv.empty:
        # Ensure stable schema (prevents KeyError on sort_values)
        inv = pd.DataFrame(
            columns=[
                "run_dir",
                "has_performance_report",
                "has_snapshot_id",
                "has_sweep_results",
                "has_predictions_parquet",
                "models_in_report",
            ]
        )
    else:
        inv = inv.sort_values("run_dir")
    return inv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-run-dir", required=True, help="A specific results/<run>/ folder (e.g., results/paper_costs_nocat_20260106_135011)")
    ap.add_argument("--results-dir", default="results", help="Top-level results directory to inventory (default: results)")
    ap.add_argument("--inventory-out", default=os.path.join("docs", "results_pack_inventory.csv"))
    ap.add_argument("--summary-out", default=None, help="Where to write results_pack_summary.json (default: <out-run-dir>/results_pack_summary.json)")
    ap.add_argument("--data-file", default=None, help="Data file path used for the run (for summary metadata only)")
    ap.add_argument("--tickers-file-applied", default=None, help="Tickers file applied (if any) (for summary metadata only)")
    ap.add_argument("--rebalance-mode", default=None, help="rebalance mode used (for summary metadata only)")
    ap.add_argument("--target-horizon-days", type=int, default=None, help="target horizon days (for summary metadata only)")
    ap.add_argument("--primary-objective-col", default="avg_top_return_net", help="Column in performance_report used as the primary objective")
    args = ap.parse_args()

    out_run_dir = args.out_run_dir
    summary_out = args.summary_out or os.path.join(out_run_dir, "results_pack_summary.json")

    os.makedirs(os.path.dirname(args.inventory_out), exist_ok=True)
    os.makedirs(out_run_dir, exist_ok=True)

    inv = build_results_inventory(args.results_dir)
    inv.to_csv(args.inventory_out, index=False)
    print(f"WROTE {args.inventory_out} rows={len(inv)}")

    summary = build_run_summary(
        out_run_dir=out_run_dir,
        data_file=args.data_file,
        tickers_file_applied=args.tickers_file_applied,
        rebalance_mode=args.rebalance_mode,
        target_horizon_days=args.target_horizon_days,
        primary_objective_col=args.primary_objective_col,
    )
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
    print(f"WROTE {summary_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



