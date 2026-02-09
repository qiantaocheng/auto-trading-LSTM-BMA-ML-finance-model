#!/usr/bin/env python3
"""Run EMA grid search for the LambdaRank-only pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

TRADE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA = TRADE_DIR / "data" / "factor_exports" / "polygon_full_features_T5.parquet"
DEFAULT_OUTPUT = TRADE_DIR / "results" / "lambdarank_ema_grid"

EMA_GRID = [
    {"id": "ema_n2_a080", "window": 2, "beta": 0.80, "description": "微调降噪：基本看今天，昨天分数微调", "weights": "[0.83, 0.17]"},
    {"id": "ema_n2_a066", "window": 2, "beta": 0.66, "description": "快速平滑", "weights": "[0.75, 0.25]"},
    {"id": "ema_n2_a050", "window": 2, "beta": 0.50, "description": "平衡平滑 (2:1)", "weights": "[0.67, 0.33]"},
    {"id": "ema_n2_a033", "window": 2, "beta": 0.33, "description": "迟钝平滑", "weights": "[0.60, 0.40]"},
    {"id": "ema_n3_a080", "window": 3, "beta": 0.80, "description": "短窗急衰减", "weights": "[0.81, 0.16, 0.03]"},
    {"id": "ema_n3_a066", "window": 3, "beta": 0.66, "description": "3日偏向平滑", "weights": "[0.69, 0.23, 0.08]"},
    {"id": "ema_n3_a050", "window": 3, "beta": 0.50, "description": "经典 3 日 EMA", "weights": "[0.57, 0.29, 0.14]"},
    {"id": "ema_n3_a033", "window": 3, "beta": 0.33, "description": "3日迟钝平滑", "weights": "[0.47, 0.32, 0.21]"},
    {"id": "ema_n4_a080", "window": 4, "beta": 0.80, "description": "长窗急衰减", "weights": "[0.80, 0.16, 0.03, 0.01]"},
    {"id": "ema_n4_a066", "window": 4, "beta": 0.66, "description": "4日快速平滑", "weights": "[0.67, 0.23, 0.07, 0.03]"},
    {"id": "ema_n4_a050", "window": 4, "beta": 0.50, "description": "经典 4 日 EMA", "weights": "[0.53, 0.27, 0.13, 0.07]"},
    {"id": "ema_n4_a033", "window": 4, "beta": 0.33, "description": "长窗重度平滑", "weights": "[0.41, 0.28, 0.19, 0.12]"},
]

METRIC_KEYS = [
    "top_1_10_mean",
    "top_1_10_median",
    "top_1_10_wr",
    "top_5_15_mean",
    "top_10_20_mean",
    "overlap_top_1_10_mean",
    "overlap_top_10_20_mean",
    "IC_mean",
    "NDCG_10",
    "NDCG_20",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LambdaRank EMA grid search")
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--time-fraction", type=float, default=1.0)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boost-round", type=int, default=800)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip combos whose metrics.json already exists")
    parser.add_argument("--combo-ids", nargs="*", default=None,
                        help="Subset of combo ids to run (default: all)")
    return parser.parse_args()


def _filter_grid(ids: Optional[List[str]]) -> List[dict]:
    if not ids:
        return EMA_GRID
    selected = []
    for config in EMA_GRID:
        if config["id"] in ids:
            selected.append(config)
    return selected


def run_combo(args: argparse.Namespace, combo: dict, run_root: Path) -> dict:
    combo_dir = run_root / combo["id"]
    combo_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = combo_dir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as fh:
            metrics = json.load(fh)
        return _summarize(combo, metrics, combo_dir, skipped=True)

    cmd = [
        sys.executable, str(TRADE_DIR / "scripts" / "lambdarank_only_pipeline.py"),
        "--data-file", str(args.data_file),
        "--time-fraction", str(args.time_fraction),
        "--split", str(args.split),
        "--horizon-days", str(args.horizon_days),
        "--seed", str(args.seed),
        "--n-boost-round", str(args.n_boost_round),
        "--ema-length", str(combo["window"]),
        "--ema-beta", str(combo["beta"]),
        "--ema-min-days", str(max(2, min(combo["window"], args.horizon_days))),
        "--output-dir", str(combo_dir),
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(TRADE_DIR))
    result = subprocess.run(cmd, cwd=str(TRADE_DIR), env=env,
                            capture_output=True, text=True)
    if result.returncode != 0:
        (combo_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
        (combo_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
        return {
            "combo_id": combo["id"],
            "window": combo["window"],
            "beta": combo["beta"],
            "description": combo["description"],
            "weights": combo["weights"],
            "status": f"failed ({result.returncode})",
            "metrics_path": str(metrics_path),
        }

    with metrics_path.open(encoding="utf-8") as fh:
        metrics = json.load(fh)
    return _summarize(combo, metrics, combo_dir, skipped=False)


def _summarize(combo: dict, metrics: dict, combo_dir: Path, skipped: bool) -> dict:
    row = {
        "combo_id": combo["id"],
        "window": combo["window"],
        "beta": combo["beta"],
        "description": combo["description"],
        "weights": combo["weights"],
        "status": "skipped" if skipped else "ok",
        "metrics_path": str(combo_dir / "metrics.json"),
    }
    for key in METRIC_KEYS:
        row[key] = metrics.get(key)
    row["top_1_10_wr"] = metrics.get("top_1_10_wr")
    row["overlap_top_1_10_wr"] = metrics.get("overlap_top_1_10_wr")
    row["spread"] = metrics.get("spread")
    return row


def main() -> None:
    args = _parse_args()
    if not args.data_file.exists():
        raise FileNotFoundError(f"data-file not found: {args.data_file}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)

    combos = _filter_grid(args.combo_ids)
    results = []
    for combo in combos:
        print(f"=== Running {combo['id']} (N={combo['window']}, beta={combo['beta']}) ===")
        summary = run_combo(args, combo, run_root)
        results.append(summary)
        status = summary.get("status", "ok")
        print(f"--> {combo['id']} status: {status}")

    df = pd.DataFrame(results)
    summary_file = run_root / "ema_grid_results.csv"
    df.to_csv(summary_file, index=False, encoding="utf-8")
    print(f"Saved EMA grid summary: {summary_file}")
    if not df.empty:
        display_cols = ["combo_id", "window", "beta", "top_1_10_mean", "top_1_10_wr", "IC_mean", "NDCG_10", "status"]
        print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
