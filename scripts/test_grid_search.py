#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Grid Search - Small Scale Validation
==========================================

Test the grid search infrastructure with a small subset of parameters.
This validates the complete pipeline before running the full grid search.

Test Configuration:
- ElasticNet: 2×2 = 4 combinations
- alpha: [1e-6, 1e-5]
- l1_ratio: [0.001, 0.01]

Usage:
    python scripts/test_grid_search.py \
        --data-file data/factor_exports/factors/factors_all.parquet \
        --output-dir results/test_grid_search
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import grid search module
from scripts.full_grid_search import grid_search_model, PARAM_GRIDS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_param_grid():
    """
    Create a small test parameter grid for validation.
    """
    test_grid = {
        'elastic_net': {
            'alpha': [1e-6, 1e-5],
            'l1_ratio': [0.001, 0.01]
        }
    }
    return test_grid


def main():
    parser = argparse.ArgumentParser(
        description='Test grid search with small ElasticNet subset'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to training data file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/factor_exports/factors',
        help='Data directory for backtest'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--base-config',
        type=str,
        default='bma_models/unified_config.yaml',
        help='Base configuration file'
    )
    parser.add_argument(
        '--snapshot-dir',
        type=str,
        default='cache/test_grid_search_snapshots',
        help='Directory to save model snapshots'
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.data_file).exists():
        logger.error(f"Data file not found: {args.data_file}")
        sys.exit(1)

    if not Path(args.data_dir).exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    if not Path(args.base_config).exists():
        logger.error(f"Base config not found: {args.base_config}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create snapshot directory
    Path(args.snapshot_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("🧪 GRID SEARCH TEST - Small Scale Validation")
    logger.info("=" * 80)
    logger.info("Testing with ElasticNet 2x2 = 4 combinations")
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    # Temporarily override PARAM_GRIDS with test grid
    test_grid = create_test_param_grid()
    original_grids = PARAM_GRIDS.copy()
    PARAM_GRIDS.clear()
    PARAM_GRIDS.update(test_grid)

    try:
        # Run grid search for ElasticNet
        results_df = grid_search_model(
            model_name='elastic_net',
            data_file=args.data_file,
            data_dir=args.data_dir,
            base_config=args.base_config,
            snapshot_dir=args.snapshot_dir,
            output_dir=str(output_dir)
        )

        logger.info("=" * 80)
        logger.info("✅ TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\nResults Summary:")
        logger.info(f"Total runs: {len(results_df)}")
        logger.info(f"Successful: {results_df['train_success'].sum()}")
        logger.info(f"Best Top20% Return: {results_df['top20_avg_return'].max():.4f}%")
        logger.info("=" * 80)

        # Display top results
        logger.info("\nTop Results:")
        print(results_df[['combination_id', 'alpha', 'l1_ratio', 'top20_avg_return']].head())

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # Restore original parameter grids
        PARAM_GRIDS.clear()
        PARAM_GRIDS.update(original_grids)


if __name__ == '__main__':
    main()

