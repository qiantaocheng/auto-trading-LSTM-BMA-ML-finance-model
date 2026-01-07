#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a full BMA Ultra snapshot (all base models + Ridge stacking) using the
best-per-model features and then run a non-overlap backtest.

This is meant to validate the Ridge stacking ensemble expected return with the
selected best feature sets.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-data", type=str, default="data/factor_exports/factors", help="Directory with factors_batch_*.parquet shards (recommended).")
    p.add_argument("--data-file", type=str, default="data/factor_exports/factors/factors_all.parquet", help="Factor file used for backtest loading.")
    p.add_argument("--data-dir", type=str, default="data/factor_exports/factors", help="Factor directory used for backtest loading.")
    p.add_argument("--max-weeks", type=int, default=260)
    p.add_argument("--rebalance-mode", type=str, default="horizon", choices=["horizon", "weekly"])
    p.add_argument("--target-horizon-days", type=int, default=10)
    p.add_argument("--output-dir", type=str, default="results/t10_optimized_all_models/ridge_stacking_test")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("ridge_stacking_test")

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    train_data = Path(args.train_data)
    if not train_data.exists():
        raise FileNotFoundError(f"--train-data not found: {train_data}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training full BMA Ultra snapshot (4 base models + Ridge stacking)...")
    logger.info("Train data: %s", train_data)
    logger.info("Best features file: results/t10_optimized_all_models/best_features_per_model.json")

    from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

    model = UltraEnhancedQuantitativeModel()
    training_results = model.train_from_document(training_data_path=str(train_data), top_n=50)
    if not training_results.get("success", False):
        raise RuntimeError(f"Training failed: {training_results.get('error')}")

    # Robust snapshot_id detection:
    # - preferred: model.active_snapshot_id (when set)
    # - fallback: training_results['snapshot_id'] or ['snapshot_used'] (some callers attach it)
    # - final fallback: latest manifest from model_registry (validated by being "recent")
    snapshot_id = getattr(model, "active_snapshot_id", None) or training_results.get("snapshot_id") or training_results.get("snapshot_used")
    if not snapshot_id:
        # If training didn't give us a snapshot_id, force-save a snapshot here.
        # This is critical for evaluating Ridge Stacking in the backtest engine (which loads from snapshot).
        try:
            from bma_models.model_registry import save_model_snapshot
            import pandas as pd

            tag = f"ridge_best_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            snapshot_payload = dict(training_results)
            if 'traditional_models' not in snapshot_payload:
                snapshot_payload['traditional_models'] = {}
            if isinstance(snapshot_payload['traditional_models'], dict) and 'models' not in snapshot_payload['traditional_models']:
                snapshot_payload['traditional_models']['models'] = {}

            snapshot_id = save_model_snapshot(
                training_results=snapshot_payload,
                ridge_stacker=getattr(model, "ridge_stacker", None),
                lambda_rank_stacker=getattr(model, "lambda_rank_stacker", None),
                rank_aware_blender=None,
                lambda_percentile_transformer=getattr(model, "lambda_percentile_transformer", None),
                tag=tag,
            )
            logger.info("✅ Forced snapshot save: snapshot_id=%s (tag=%s)", snapshot_id, tag)
            try:
                model.active_snapshot_id = snapshot_id
            except Exception:
                pass
        except Exception as e:
            raise RuntimeError(f"No snapshot_id produced by training and forced snapshot save failed: {e}") from e

    if not snapshot_id:
        raise RuntimeError("No snapshot_id available after training + forced snapshot save")

    (run_dir / "snapshot_id.txt").write_text(str(snapshot_id), encoding="utf-8")
    logger.info("✅ Training done. snapshot_id=%s", snapshot_id)

    # Backtest via the same engine used by grid search
    logger.info("Running comprehensive backtest (rebalance_mode=%s, horizon=%s days)...", args.rebalance_mode, args.target_horizon_days)
    from scripts.comprehensive_model_backtest import ComprehensiveModelBacktest

    backtest = ComprehensiveModelBacktest(
        data_dir=str(Path(args.data_dir)),
        snapshot_id=str(snapshot_id),
        data_file=str(Path(args.data_file)),
    )
    backtest._rebalance_mode = args.rebalance_mode
    backtest._target_horizon_days = int(args.target_horizon_days)

    _, report_df, _ = backtest.run_backtest(max_weeks=int(args.max_weeks))
    report_path = run_dir / "performance_report.csv"
    report_df.to_csv(report_path, index=False, encoding="utf-8")
    logger.info("📄 Saved report: %s", report_path)

    # Print key headline: Ridge stacking expected return
    if not report_df.empty and "Model" in report_df.columns and "avg_top_return" in report_df.columns:
        ridge_rows = report_df[report_df["Model"].astype(str).str.lower().str.contains("ridge")]
        if len(ridge_rows) > 0:
            v = float(ridge_rows.iloc[0]["avg_top_return"])
            logger.info("🏁 Ridge Stacking avg_top_return = %.6f", v)
            (run_dir / "ridge_avg_top_return.txt").write_text(f"{v:.10f}\n", encoding="utf-8")
        else:
            logger.warning("No 'Ridge' row found in report; available models: %s", list(report_df["Model"].astype(str).unique()))

    logger.info("✅ Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


