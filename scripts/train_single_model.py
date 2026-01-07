#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Model Training Entry Point for Grid Search

This script provides a CLI interface to train the BMA model with specific parameter overrides.
It's designed to be called by the grid search orchestration script.

Usage:
    python scripts/train_single_model.py \
        --model elastic_net \
        --params '{"alpha": 1e-5, "l1_ratio": 0.001}' \
        --data-file data/factor_exports/factors/factors_all.parquet \
        --snapshot-dir cache/grid_search_snapshots \
        --output-file output_snapshot_id.txt
"""

import sys
import json
import argparse
import logging
from pathlib import Path
import tempfile
import yaml
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# DO NOT import model modules here - they will be imported after setting environment variables

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_temp_config(base_config_path: Path, model_name: str, params: dict) -> Path:
    """
    Create a temporary config file with parameter overrides.

    Args:
        base_config_path: Path to the base unified_config.yaml
        model_name: Model to override (elastic_net, xgboost, catboost, lambdarank, ridge)
        params: Dictionary of parameters to override

    Returns:
        Path to the temporary config file
    """
    # Load base config
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    # Ensure training.base_models structure exists for base models
    training_cfg = config.setdefault('training', {})
    base_models_cfg = training_cfg.setdefault('base_models', {})

    # Apply parameter overrides based on model
    if model_name in ('elastic_net', 'xgboost', 'catboost', 'lambdarank'):
        model_cfg = base_models_cfg.setdefault(model_name, {})
        model_cfg.update(params)
        logger.info(f"{model_name} parameters updated under training.base_models: {params}")

    elif model_name == 'ridge':
        ridge_cfg = training_cfg.setdefault('ridge_stacker', {})
        ridge_cfg.update(params)
        logger.info(f"Ridge parameters updated under training.ridge_stacker: {params}")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Create temporary config file
    temp_config = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.yaml',
        delete=False,
        encoding='utf-8'
    )
    yaml.dump(config, temp_config, allow_unicode=True)
    temp_config.close()

    logger.info(f"Temporary config created: {temp_config.name}")
    return Path(temp_config.name)


def apply_lambda_params_override(lambda_config: dict, override_params: dict) -> dict:
    """
    Apply parameter overrides to LambdaRank configuration.

    Args:
        lambda_config: Base LambdaRank config dict
        override_params: Parameters to override

    Returns:
        Updated config dict
    """
    # Map common parameter names
    param_mapping = {
        'num_boost_round': 'num_boost_round',
        'learning_rate': 'lgb_params.learning_rate',
        'num_leaves': 'lgb_params.num_leaves',
        'max_depth': 'lgb_params.max_depth',
        'lambda_l2': 'lgb_params.lambda_l2'
    }

    for param_key, param_value in override_params.items():
        if param_key in param_mapping:
            target_key = param_mapping[param_key]
            if '.' in target_key:
                # Nested parameter (e.g., lgb_params.learning_rate)
                outer_key, inner_key = target_key.split('.')
                if outer_key not in lambda_config:
                    lambda_config[outer_key] = {}
                lambda_config[outer_key][inner_key] = param_value
            else:
                lambda_config[target_key] = param_value

    return lambda_config


def train_model_with_params(
    data_file: str,
    model_name: str,
    params: dict,
    base_config_path: str,
    snapshot_dir: str,
    feature_list: str | None = None,
) -> str:
    """
    Train the BMA model with specified parameters.

    Args:
        data_file: Path to training data file
        model_name: Model to configure
        params: Parameter overrides
        base_config_path: Path to base config
        snapshot_dir: Directory to save snapshots

    Returns:
        snapshot_id (str)
    """
    logger.info("="*80)
    logger.info(f"Training {model_name.upper()} with parameters:")
    for k, v in params.items():
        logger.info(f"  {k}: {v}")
    logger.info("="*80)

    # Create temporary config with overrides
    temp_config_path = create_temp_config(
        Path(base_config_path),
        model_name,
        params
    )

    try:
        # Optional feature subset for this run.
        # Prefer per-model overrides so tuning one model doesn't force the same whitelist on all models.
        # NOTE: "[]" is meaningful (compulsory-only when enforced by model code).
        if feature_list is not None:
            os.environ["BMA_FEATURE_OVERRIDES"] = json.dumps({model_name: json.loads(feature_list)})
            logger.info(f"[FEATURE] Using per-model overrides for {model_name}: {feature_list}")

        # Speed optimization for tuning: train only the requested first-layer model.
        # (Ridge training requires multiple base predictions; do not enable for ridge.)
        if model_name in {"elastic_net", "xgboost", "catboost", "lambdarank"}:
            os.environ["BMA_TRAIN_ONLY_MODEL"] = model_name
        else:
            os.environ.pop("BMA_TRAIN_ONLY_MODEL", None)

        # Set environment variable to use temporary config
        # This must be done BEFORE importing the model module
        os.environ['BMA_TEMP_CONFIG_PATH'] = str(temp_config_path)
        os.environ['BMA_GRID_SEARCH_MODE'] = '1'

        # Now import the model module - CONFIG will be initialized with the temp config
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

        # Instantiate model
        model = UltraEnhancedQuantitativeModel()

        # Special handling for LambdaRank parameters
        if model_name == 'lambdarank' and hasattr(model, 'lambda_rank_stacker'):
            # Apply parameters directly to LambdaRank if it exists
            if model.lambda_rank_stacker is not None:
                # Parameters will be applied during training via config
                pass

        # Train from document
        logger.info(f"Loading training data from: {data_file}")
        training_results = model.train_from_document(
            training_data_path=data_file,
            top_n=10
        )

        if not training_results.get('success'):
            raise RuntimeError(f"Training failed: {training_results.get('error')}")

        # Get the snapshot_id that was automatically saved
        if hasattr(model, 'active_snapshot_id') and model.active_snapshot_id:
            snapshot_id = model.active_snapshot_id
            logger.info(f"✅ Training completed successfully")
            logger.info(f"✅ Snapshot ID: {snapshot_id}")
            return snapshot_id
        else:
            raise RuntimeError("Snapshot ID not found after training")

    finally:
        # Cleanup temporary config
        try:
            if temp_config_path.exists():
                temp_config_path.unlink()
                logger.info(f"Temporary config cleaned up: {temp_config_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp config: {e}")

        # Clear environment variables
        os.environ.pop('BMA_TEMP_CONFIG_PATH', None)
        os.environ.pop('BMA_GRID_SEARCH_MODE', None)
        os.environ.pop('BMA_FEATURE_WHITELIST', None)
        os.environ.pop('BMA_FEATURE_OVERRIDES', None)
        os.environ.pop('BMA_TRAIN_ONLY_MODEL', None)


def main():
    parser = argparse.ArgumentParser(
        description='Train BMA model with specific parameter configuration'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['elastic_net', 'xgboost', 'catboost', 'lambdarank', 'ridge'],
        help='Model to train'
    )
    parser.add_argument(
        '--params',
        type=str,
        required=True,
        help='JSON string of parameters to override (e.g., \'{"alpha": 1e-5}\')'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to training data file (MultiIndex parquet/csv)'
    )
    parser.add_argument(
        '--base-config',
        type=str,
        default='bma_models/unified_config.yaml',
        help='Path to base config file'
    )
    parser.add_argument(
        '--snapshot-dir',
        type=str,
        default='cache/grid_search_snapshots',
        help='Directory to save model snapshots'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='File to write the snapshot_id'
    )
    parser.add_argument(
        '--feature-list',
        type=str,
        default=None,
        help='JSON array of feature names to whitelist (compulsory factors are always included)'
    )

    args = parser.parse_args()

    # Parse parameters
    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse params JSON: {e}")
        sys.exit(1)

    # Validate data file exists
    if not Path(args.data_file).exists():
        logger.error(f"Data file not found: {args.data_file}")
        sys.exit(1)

    # Validate base config exists
    if not Path(args.base_config).exists():
        logger.error(f"Base config not found: {args.base_config}")
        sys.exit(1)

    try:
        # Train model
        snapshot_id = train_model_with_params(
            data_file=args.data_file,
            model_name=args.model,
            params=params,
            base_config_path=args.base_config,
            snapshot_dir=args.snapshot_dir,
            feature_list=args.feature_list
        )

        # Write snapshot_id to output file
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(snapshot_id)

        logger.info(f"Snapshot ID written to: {args.output_file}")
        logger.info("="*80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()

