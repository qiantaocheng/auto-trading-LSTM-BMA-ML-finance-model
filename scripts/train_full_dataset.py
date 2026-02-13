#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train models with complete dataset (no time split) and create snapshot for production use.
This script trains all models using the full available dataset and saves the snapshot ID
as the default for app.py predictions.
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from datetime import datetime
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train models with complete dataset for production")
    p.add_argument(
        "--train-data",
        type=str,
        default=r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet",
        help="Path to training data file (parquet) or directory with factors_batch_*.parquet shards"
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Top N features to use (default: 50)"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/full_dataset_training",
        help="Output directory for training logs and snapshot ID"
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("train_full_dataset")

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "bma_models"))

    train_data = Path(args.train_data)
    if not train_data.exists():
        raise FileNotFoundError(f"--train-data not found: {train_data}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Training BMA Ultra models with COMPLETE DATASET (for production)")
    logger.info("Note: This trains on all available data. 80/20 time-split evaluation comes next.")
    logger.info("=" * 80)
    logger.info("Train data: %s", train_data)
    logger.info("Top N features: %d", args.top_n)
    logger.info("Output directory: %s", run_dir)

    from bma_models._bma_ultra_enhanced import UltraEnhancedQuantitativeModel

    # Initialize model
    model = UltraEnhancedQuantitativeModel()
    
    # Train with complete dataset (no start_date/end_date filtering)
    logger.info("Starting training with full dataset...")
    
    # Set explicit snapshot tag before training
    import pandas as pd
    explicit_tag = f"FINAL_V2_FULL_DATASET_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info("Using explicit snapshot tag: %s", explicit_tag)
    
    # Try to set tag via environment variable or model attribute if available
    os.environ['BMA_SNAPSHOT_TAG'] = explicit_tag
    
    training_results = model.train_from_document(
        training_data_path=str(train_data),
        top_n=args.top_n
    )
    
    if not training_results.get("success", False):
        error_msg = training_results.get("error", "Unknown error")
        logger.error("Training failed: %s", error_msg)
        raise RuntimeError(f"Training failed: {error_msg}")

    logger.info("?Training completed successfully")

    # Always save a new snapshot with explicit tag for production use
    logger.info("=" * 80)
    logger.info("Saving snapshot with explicit tag for production use...")
    logger.info("=" * 80)
    
    import pandas as pd
    from bma_models.model_registry import save_model_snapshot
    
    # Create explicit tag with FINAL_V2 prefix
    explicit_tag = f"FINAL_V2_FULL_DATASET_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info("Explicit snapshot tag: %s", explicit_tag)
    
    # Prepare snapshot payload - NO FALLBACK, strict validation only
    #  FIX: train_from_document'training_results'?
    # _run_training_phase: {'training_results': actual_training_results, ...}
    # training_results['training_results']?
    
    # Extract actual training results; handle nested structures from train_from_document
    if 'training_results' in training_results and isinstance(training_results['training_results'], dict):
        actual_training_results = training_results['training_results']
        logger.info('Detected nested training_results structure; using inner payload')
    else:
        actual_training_results = training_results
        logger.info('Using top-level training_results payload')

    snapshot_payload = dict(actual_training_results)

    # Validate presence of traditional_models -> models
    if 'traditional_models' not in snapshot_payload:
        logger.error("training_results missing 'traditional_models' key")
        logger.error(f"Available keys: {list(snapshot_payload.keys())}")
        if 'training_results' in training_results:
            nested_keys = list(training_results['training_results'].keys()) if isinstance(training_results.get('training_results'), dict) else 'N/A'
            logger.error(f"Nested training_results keys: {nested_keys}")
        raise ValueError("training_results must include 'traditional_models'")

    trad_models = snapshot_payload['traditional_models']
    if not isinstance(trad_models, dict):
        logger.error(f"traditional_models must be a dict, got {type(trad_models)}")
        raise ValueError("training_results['traditional_models'] must be a dict")

    if 'models' not in trad_models:
        logger.error("traditional_models missing 'models' key")
        logger.error(f"traditional_models keys: {list(trad_models.keys())}")
        raise ValueError("training_results['traditional_models'] must include 'models'")

    models = trad_models['models']
    if not models:
        logger.error("traditional_models['models'] is empty")
        raise ValueError("training_results['traditional_models']['models'] cannot be empty")

    if not isinstance(models, dict):
        logger.error(f"traditional_models['models'] must be a dict, got {type(models)}")
        raise ValueError("training_results['traditional_models']['models'] must be a dict")

    logger.info(f"Validated traditional_models['models'] entries: {list(models.keys())}")

    # Save snapshot with explicit tag (will create a new snapshot even if one already exists)
    #  FIX: ridge_stackermeta_ranker_stacker
    meta_ranker = getattr(model, "meta_ranker_stacker", None)
    
    snapshot_id = save_model_snapshot(
        training_results=snapshot_payload,
        meta_ranker_stacker=meta_ranker,  #  meta_ranker_stacker
        lambda_rank_stacker=getattr(model, "lambda_rank_stacker", None),
        rank_aware_blender=None,
        lambda_percentile_transformer=getattr(model, "lambda_percentile_transformer", None),
        tag=explicit_tag,
    )
    logger.info("?Saved snapshot with explicit tag: snapshot_id=%s, tag=%s", snapshot_id, explicit_tag)
    
    # Update model's active_snapshot_id
    try:
        model.active_snapshot_id = snapshot_id
    except Exception:
        pass

    if not snapshot_id:
        raise RuntimeError("Failed to save snapshot with explicit tag")

    # Save snapshot ID to multiple locations
    snapshot_file = run_dir / "snapshot_id.txt"
    snapshot_file.write_text(str(snapshot_id), encoding="utf-8")
    logger.info("?Saved snapshot ID to: %s", snapshot_file)

    # Update latest_snapshot_id.txt in project root
    latest_snapshot_file = project_root / "latest_snapshot_id.txt"
    latest_snapshot_file.write_text(str(snapshot_id), encoding="utf-8")
    logger.info("?Updated latest_snapshot_id.txt: %s", snapshot_id)

    # Log training summary
    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info("Snapshot ID: %s", snapshot_id)
    logger.info("Training success: %s", training_results.get("success", False))
    
    # Log model information
    if 'traditional_models' in training_results:
        models_info = training_results.get('traditional_models', {})
        if isinstance(models_info, dict) and 'models' in models_info:
            model_names = list(models_info['models'].keys())
            logger.info("Trained models: %s", ', '.join(model_names))
    
    # Log MetaRankerStacker if available
    if hasattr(model, "meta_ranker_stacker") and model.meta_ranker_stacker is not None:
        logger.info("?MetaRankerStacker trained successfully")
    
    logger.info("=" * 80)
    logger.info("?Training complete! Snapshot ready for production use.")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


