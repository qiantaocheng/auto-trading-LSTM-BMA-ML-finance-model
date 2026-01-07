#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel-by-model feature combo grid search (T+10)

User request:
- "stop the code, and modify the algorithm make every model run at same time for each combination"

This script:
- Iterates feature subsets (from stage1_ranked_combos.csv top-N)
- For each subset, runs ALL models in parallel (processes), each doing:
  train_single_model.py  -> snapshot id
  comprehensive_model_backtest.py -> avg_top_return

Outputs:
- results/<out>/combo_results.csv (one row per (combo, model))
- results/<out>/combo_summary.csv (one row per combo with per-model metrics)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-file", type=str, required=True, help="stage1_ranked_combos.csv")
    p.add_argument("--top-combos", type=int, default=200)
    p.add_argument("--models", nargs="+", default=["elastic_net", "xgboost", "catboost", "lambdarank"])
    p.add_argument("--data-file", type=str, default="data/factor_exports/factors/factors_all.parquet")
    p.add_argument("--train-data", type=str, default=None,
                   help="Training data path. Can be a directory of MultiIndex shards (recommended) to reduce memory. "
                        "If omitted, uses --data-file.")
    p.add_argument("--max-weeks", type=int, default=260)
    p.add_argument("--output-dir", type=str, default="results/t10_feature_combo_grid_parallel_models")
    p.add_argument("--max-parallel-models", type=int, default=4)
    p.add_argument("--base-config", type=str, default="bma_models/unified_config.yaml")
    p.add_argument("--rebalance-mode", type=str, default="horizon", choices=["horizon", "weekly"],
                   help="Backtest rebalance frequency. Use 'horizon' to avoid overlap vs target horizon.")
    p.add_argument("--target-horizon-days", type=int, default=10,
                   help="Target horizon in trading days (used when rebalance-mode=horizon).")
    p.add_argument("--backtest-start-date", type=str, default=None, help="First rebalance date (YYYY-MM-DD) to include in evaluation.")
    p.add_argument("--backtest-end-date", type=str, default=None, help="Last rebalance date (YYYY-MM-DD) to include in evaluation.")
    p.add_argument("--allow-insample-backtest", action="store_true", help="Allow backtests to run even when no out-of-sample data exists (not recommended).")
    return p.parse_args()


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-4000:] or proc.stdout[-4000:] or "subprocess failed")


def _latest_performance_report(out_dir: Path) -> Path:
    files = sorted(out_dir.glob("performance_report_*.csv"))
    if not files:
        raise FileNotFoundError(f"No performance_report_*.csv under {out_dir}")
    return files[-1]


def _extract_avg_top_return(report_csv: Path, model: str) -> float:
    df = pd.read_csv(report_csv)
    if df.empty:
        raise RuntimeError("empty performance report")
    if "avg_top_return" not in df.columns:
        raise RuntimeError(f"avg_top_return missing in {report_csv}")

    # Prefer matching the requested model row (some snapshots may still contain multiple rows).
    if "Model" in df.columns:
        m = str(model).strip().lower()
        mask = df["Model"].astype(str).str.strip().str.lower() == m
        if mask.any():
            return float(df.loc[mask, "avg_top_return"].iloc[0])

    # Fallback: single-row report
    if len(df) == 1:
        return float(df.loc[0, "avg_top_return"])
    raise RuntimeError(f"Cannot find model={model} row in {report_csv} (rows={len(df)})")


def _worker_train_backtest(
    model: str,
    feature_list: List[str],
    data_file: str,
    train_data: str,
    base_config: str,
    max_weeks: int,
    work_dir: str,
    rebalance_mode: str,
    target_horizon_days: int,
    backtest_start_date: Optional[str],
    backtest_end_date: Optional[str],
    allow_insample_backtest: bool,
) -> Dict[str, Any]:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_id = work / "snapshot_id.txt"

    # Train
    train_cmd = [
        os.environ.get("PYTHON", "python"),
        "scripts/train_single_model.py",
        "--model",
        model,
        "--params",
        "{}",
        "--data-file",
        train_data,
        "--base-config",
        base_config,
        "--snapshot-dir",
        "cache/grid_search_snapshots",
        "--output-file",
        str(out_id),
        "--feature-list",
        json.dumps(feature_list),
    ]
    _run(train_cmd)
    sid = out_id.read_text(encoding="utf-8").strip()
    if not sid:
        raise RuntimeError("missing snapshot id after training")

    # Backtest (isolated output dir)
    bt_out = work / "backtest"
    bt_out.mkdir(parents=True, exist_ok=True)
    bt_cmd = [
        os.environ.get("PYTHON", "python"),
        "scripts/comprehensive_model_backtest.py",
        "--data-file",
        data_file,
        "--snapshot-id",
        sid,
        "--max-weeks",
        str(max_weeks),
        "--rebalance-mode",
        str(rebalance_mode),
        "--target-horizon-days",
        str(int(target_horizon_days)),
        "--output-dir",
        str(bt_out),
    ]
    if backtest_start_date:
        bt_cmd.extend(["--start-date", backtest_start_date])
    if backtest_end_date:
        bt_cmd.extend(["--end-date", backtest_end_date])
    if allow_insample_backtest:
        bt_cmd.append("--allow-insample")
    _run(bt_cmd)
    report_csv = _latest_performance_report(bt_out)
    score = _extract_avg_top_return(report_csv, model=model)

    return {
        "model": model,
        "snapshot_id": sid,
        "avg_top_return": score,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1 = pd.read_csv(args.stage1_file)
    stage1 = stage1.dropna(subset=["features"]).head(int(args.top_combos))
    combos: List[List[str]] = [json.loads(s) for s in stage1["features"].tolist()]

    results_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for ci, feat_list in enumerate(combos, start=1):
        combo_id = f"combo_{ci:04d}"
        logger.info(f"=== {combo_id} k={len(feat_list)} ===")
        combo_dir = out_dir / "runs" / combo_id
        combo_dir.mkdir(parents=True, exist_ok=True)

        # parallel by model for this feature set
        futures = []
        with ProcessPoolExecutor(max_workers=int(args.max_parallel_models)) as ex:
            for model in args.models:
                work_dir = str(combo_dir / model)
                futures.append(
                    ex.submit(
                        _worker_train_backtest,
                        model=model,
                        feature_list=feat_list,
                        data_file=args.data_file,
                        train_data=str(args.train_data or args.data_file),
                        base_config=args.base_config,
                        max_weeks=int(args.max_weeks),
                        work_dir=work_dir,
                        rebalance_mode=str(args.rebalance_mode),
                        target_horizon_days=int(args.target_horizon_days),
                        backtest_start_date=args.backtest_start_date,
                        backtest_end_date=args.backtest_end_date,
                        allow_insample_backtest=bool(args.allow_insample_backtest),
                    )
                )

            combo_summary: Dict[str, Any] = {"combo": combo_id, "k": len(feat_list), "features": json.dumps(feat_list)}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    r["combo"] = combo_id
                    r["k"] = len(feat_list)
                    r["features"] = json.dumps(feat_list)
                    results_rows.append(r)
                    combo_summary[f"{r['model']}_avg_top_return"] = r["avg_top_return"]
                    logger.info(f"{combo_id} {r['model']}: avg_top_return={r['avg_top_return']:.4f}")
                except Exception as e:
                    err = str(e)
                    results_rows.append(
                        {"combo": combo_id, "k": len(feat_list), "features": json.dumps(feat_list), "model": "?", "avg_top_return": float("nan"), "error": err}
                    )
                    logger.warning(f"{combo_id} model failed: {err}")

            summary_rows.append(combo_summary)

        # incremental save
        pd.DataFrame(results_rows).to_csv(out_dir / "combo_results.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(out_dir / "combo_summary.csv", index=False)

    logger.info(f"DONE. WROTE {out_dir / 'combo_results.csv'} and {out_dir / 'combo_summary.csv'}")


if __name__ == "__main__":
    main()


