#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sweep RidgeStacker input combinations (base prediction columns) and backtest each.

Goal:
  Find the best Ridge stacking input combination among:
    pred_elastic, pred_xgb, pred_catboost, pred_lambdarank

Method:
  1) Train once to obtain first-layer models + OOF stacker_data (cached at model._last_stacker_data)
  2) For each combination of base_cols (size 2..4):
     - Fit a RidgeStacker on the same stacker_data
     - Save a snapshot that reuses the same base models + new ridge stacker
     - Run ComprehensiveModelBacktest and record ridge_stacking avg_top_return

Outputs:
  - output_dir/sweep_results.csv
  - output_dir/best_row.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


PRED_COLS = ("pred_elastic", "pred_xgb", "pred_catboost", "pred_lambdarank")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", type=str, default="data/factor_exports/factors", help="Directory with factors_batch_*.parquet shards (recommended).")
    p.add_argument("--data-dir", type=str, default="data/factor_exports/factors")
    p.add_argument("--data-file", type=str, default="data/factor_exports/factors/factors_all.parquet")
    p.add_argument("--rebalance-mode", type=str, default="horizon", choices=["horizon", "weekly"])
    p.add_argument("--target-horizon-days", type=int, default=10)
    p.add_argument("--max-weeks", type=int, default=260)
    p.add_argument("--min-cols", type=int, default=2, choices=[2, 3, 4])
    p.add_argument("--max-cols", type=int, default=4, choices=[2, 3, 4])
    p.add_argument("--output-dir", type=str, default="results/t10_optimized_all_models/ridge_input_sweep")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _load_ridge_alpha_default() -> float:
    # Best effort: read from unified_config.yaml (if PyYAML available); else default 100.0
    try:
        import yaml  # type: ignore

        cfg_path = Path("bma_models/unified_config.yaml")
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            alpha = (
                cfg.get("models", {})
                .get("ridge_stacker", {})
                .get("alpha", None)
            )
            if alpha is not None:
                return float(alpha)
    except Exception:
        pass
    return 100.0


def _extract_models_from_training_results(training_results: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # model_registry.save_model_snapshot expects training_results['models'] dict
    models = training_results.get("models") or training_results.get("traditional_models", {}).get("models") or {}
    if not isinstance(models, dict) or not models:
        raise ValueError("Training results did not contain models (models/traditional_models.models empty)")

    feature_names_by_model = (
        training_results.get("feature_names_by_model")
        or training_results.get("traditional_models", {}).get("feature_names_by_model")
        or {}
    )
    if not isinstance(feature_names_by_model, dict):
        feature_names_by_model = {}

    return models, feature_names_by_model


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("ridge_input_sweep")

    # Ensure repo modules are importable when running as a script on Windows
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "bma_models"))
    sys.path.insert(0, str(project_root / "scripts"))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training once to obtain stacker_data and base models...")
    from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

    model = UltraEnhancedQuantitativeModel()
    training_results = model.train_from_document(training_data_path=str(Path(args.train_data)), top_n=50)
    if not training_results.get("success", False):
        raise RuntimeError(f"Training failed: {training_results.get('error')}")

    stacker_data = getattr(model, "_last_stacker_data", None)
    if stacker_data is None or not isinstance(stacker_data, pd.DataFrame) or stacker_data.empty:
        raise RuntimeError("Missing model._last_stacker_data; cannot sweep Ridge inputs.")

    # Ensure required pred columns exist in stacker_data
    missing_pred_cols = [c for c in PRED_COLS if c not in stacker_data.columns]
    if missing_pred_cols:
        raise RuntimeError(f"stacker_data missing required pred columns: {missing_pred_cols}. cols={list(stacker_data.columns)[:20]}")

    # Base models for snapshot export:
    # training_results may not expose models in a stable location, but the training pipeline auto-saves a snapshot.
    # Load base models from that snapshot to guarantee we can export per-combo Ridge stackers.
    snapshot_id = (
        getattr(model, "active_snapshot_id", None)
        or training_results.get("snapshot_id")
        or training_results.get("snapshot_used")
    )
    if not snapshot_id:
        raise RuntimeError("Training succeeded but no snapshot_id found (active_snapshot_id / training_results['snapshot_id']).")

    from bma_models.model_registry import load_models_from_snapshot
    loaded = load_models_from_snapshot(str(snapshot_id))
    base_models = loaded.get("models") or {}
    if not isinstance(base_models, dict) or not base_models:
        raise RuntimeError(f"Snapshot {snapshot_id} did not contain base models under loaded['models'].")

    formatted_models = {
        "elastic_net": {"model": base_models.get("elastic_net")},
        "xgboost": {"model": base_models.get("xgboost")},
        "catboost": {"model": base_models.get("catboost")},
    }
    # Some snapshots store LambdaRank stacker separately
    lambda_model = loaded.get("lambda_rank_stacker") or getattr(model, "lambda_rank_stacker", None)
    lambda_pct = loaded.get("lambda_percentile_transformer") or getattr(model, "lambda_percentile_transformer", None)
    # Preserve feature-name metadata if available, but it's optional for snapshot export.
    feature_names_by_model = (
        training_results.get("feature_names_by_model")
        or training_results.get("traditional_models", {}).get("feature_names_by_model")
        or {}
    )
    if not isinstance(feature_names_by_model, dict):
        feature_names_by_model = {}

    ridge_alpha = _load_ridge_alpha_default()
    logger.info("Ridge alpha default: %s", ridge_alpha)

    # Generate combinations
    min_k = int(args.min_cols)
    max_k = int(args.max_cols)
    combos: List[Tuple[str, ...]] = []
    for k in range(min_k, max_k + 1):
        combos.extend(list(itertools.combinations(PRED_COLS, k)))

    logger.info("Sweeping %d ridge base_cols combos (%d..%d)", len(combos), min_k, max_k)

    from bma_models.ridge_stacker import RidgeStacker
    from bma_models.model_registry import save_model_snapshot
    from scripts.comprehensive_model_backtest import ComprehensiveModelBacktest

    rows: List[Dict[str, Any]] = []
    for i, base_cols in enumerate(combos, 1):
        tag = f"ridge_basecols_{'-'.join(base_cols)}_{ts}"
        logger.info("[%d/%d] Fitting RidgeStacker base_cols=%s", i, len(combos), base_cols)

        ridge = RidgeStacker(base_cols=tuple(base_cols), alpha=float(ridge_alpha))
        ridge.fit(stacker_data, max_train_to_today=True)

        snap_training_results = {"models": formatted_models, "feature_names_by_model": feature_names_by_model}
        snapshot_id = save_model_snapshot(
            training_results=snap_training_results,
            ridge_stacker=ridge,
            lambda_rank_stacker=lambda_model,
            rank_aware_blender=None,
            lambda_percentile_transformer=lambda_pct,
            tag=tag,
        )

        bt = ComprehensiveModelBacktest(
            data_dir=str(Path(args.data_dir)),
            snapshot_id=str(snapshot_id),
            data_file=str(Path(args.data_file)),
        )
        bt._rebalance_mode = args.rebalance_mode
        bt._target_horizon_days = int(args.target_horizon_days)
        _, report_df, _ = bt.run_backtest(max_weeks=int(args.max_weeks))

        ridge_row = None
        if not report_df.empty and "Model" in report_df.columns:
            mask = report_df["Model"].astype(str).str.lower().str.contains("ridge")
            if mask.any():
                ridge_row = report_df[mask].iloc[0].to_dict()

        avg_top = float(ridge_row.get("avg_top_return")) if ridge_row and "avg_top_return" in ridge_row else float("nan")

        rows.append(
            {
                "base_cols": json.dumps(list(base_cols)),
                "k": len(base_cols),
                "snapshot_id": snapshot_id,
                "avg_top_return": avg_top,
            }
        )
        logger.info("   -> avg_top_return=%s snapshot_id=%s", avg_top, snapshot_id)

    out_csv = run_dir / "sweep_results.csv"
    df = pd.DataFrame(rows).sort_values("avg_top_return", ascending=False)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("Saved sweep results: %s", out_csv)

    best = df.iloc[0].to_dict() if len(df) else {}
    (run_dir / "best_row.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    logger.info("Best combo: %s", best)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


