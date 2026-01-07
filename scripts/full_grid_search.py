#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Grid Search with Backtest Integration
===========================================

对5个模型进行完整网格搜索，每次训练后运行回测，以 Top20% 平均收益作为评估指标。

Models:
1. ElasticNet: alpha × l1_ratio (7×7 = 49 combinations)
2. XGBoost: n_estimators × max_depth × learning_rate × min_child_weight (5×5×5×5 = 625 combinations)
3. CatBoost: iterations × depth × learning_rate × subsample (5×5×5×5 = 625 combinations)
4. LambdaRank: num_boost_round × learning_rate × num_leaves × max_depth × lambda_l2 (5^5 = 3,125 combinations)
5. Ridge: alpha (7 combinations)

Total: 4,431 training+backtest runs

Usage:
    python scripts/full_grid_search.py \
        --data-file data/factor_exports/factors/factors_all.parquet \
        --output-dir results/grid_search_20231205 \
        --models elastic_net xgboost catboost lambdarank ridge
"""

import sys
import os
import json
import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import itertools

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from scripts.comprehensive_model_backtest import ComprehensiveModelBacktest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PARAMETER GRIDS DEFINITION
# ============================================================================

PARAM_GRIDS = {
    'elastic_net': {
        'alpha': [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
        'l1_ratio': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    },
    'xgboost': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'min_child_weight': [1, 3, 5, 7, 10]
    },
    'catboost': {
        'iterations': [1000, 2000, 3000, 4000, 5000],
        'depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
    },
    'lambdarank': {
        'num_boost_round': [50, 100, 200, 300, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'num_leaves': [127, 191, 255, 319, 383],
        'max_depth': [6, 7, 8, 9, 10],
        'lambda_l2': [1.0, 5.0, 10.0, 20.0, 50.0]
    },
    'ridge': {
        # Ridge stacking experiments:
        # - alpha: L2 regularization strength
        # - fit_intercept: whether to fit intercept (usually False since preds are standardized-ish)
        # - base_cols: which first-layer prediction columns feed Ridge
        'alpha': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
        'fit_intercept': [False],
        'base_cols': [
            ['pred_catboost', 'pred_elastic', 'pred_xgb'],
            ['pred_catboost', 'pred_xgb'],
            ['pred_xgb', 'pred_elastic'],
            # LambdaRank-inclusive variants (post-fix retest)
            ['pred_lambdarank'],
            ['pred_xgb', 'pred_lambdarank'],
            ['pred_elastic', 'pred_lambdarank'],
            ['pred_catboost', 'pred_lambdarank'],
            ['pred_xgb', 'pred_elastic', 'pred_lambdarank'],
            ['pred_catboost', 'pred_xgb', 'pred_lambdarank'],
            ['pred_catboost', 'pred_elastic', 'pred_lambdarank'],
            ['pred_catboost', 'pred_elastic', 'pred_xgb', 'pred_lambdarank'],
        ],
    }
}


def get_param_combinations(model_name: str) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations for a specific model.

    Args:
        model_name: Model name

    Returns:
        List of parameter dictionaries
    """
    if model_name not in PARAM_GRIDS:
        raise ValueError(f"Unknown model: {model_name}")

    param_grid = PARAM_GRIDS[model_name]
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    # Generate Cartesian product
    combinations = list(itertools.product(*param_values))

    # Convert to list of dicts
    param_dicts = [
        dict(zip(param_names, combo))
        for combo in combinations
    ]

    return param_dicts


def run_single_training(
    model_name: str,
    params: Dict[str, Any],
    data_file: str,
    base_config: str,
    snapshot_dir: str,
    train_script: str = "scripts/train_single_model.py",
    feature_list: List[str] | None = None,
) -> Tuple[str, bool]:
    """
    Run a single training job with specified parameters.

    Args:
        model_name: Model to train
        params: Parameter dictionary
        data_file: Training data file path
        base_config: Base configuration file path
        snapshot_dir: Directory to save snapshots
        train_script: Path to training script

    Returns:
        (snapshot_id, success)
    """
    # Create temporary file for snapshot_id output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        snapshot_file = f.name

    try:
        # Build command
        params_json = json.dumps(params)
        cmd = [
            sys.executable,
            train_script,
            '--model', model_name,
            '--params', params_json,
            '--data-file', data_file,
            '--base-config', base_config,
            '--snapshot-dir', snapshot_dir,
            '--output-file', snapshot_file
        ]
        # NOTE: empty list [] is meaningful ("compulsory-only" when enforced by model code)
        if feature_list is not None:
            cmd.extend(['--feature-list', json.dumps(feature_list)])

        logger.info(f"Running command: {' '.join(cmd)}")

        # Run training subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hours timeout per training
        )

        if result.returncode != 0:
            logger.error(f"Training failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return None, False

        # Read snapshot_id from output file
        if not Path(snapshot_file).exists():
            logger.error("Snapshot file not created")
            return None, False

        with open(snapshot_file, 'r') as f:
            snapshot_id = f.read().strip()

        if not snapshot_id:
            logger.error("Empty snapshot_id")
            return None, False

        logger.info(f"✅ Training completed successfully, snapshot_id: {snapshot_id}")
        return snapshot_id, True

    except subprocess.TimeoutExpired:
        logger.error("Training timeout (2 hours)")
        return None, False
    except Exception as e:
        logger.error(f"Training exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, False
    finally:
        # Cleanup temp file
        try:
            if Path(snapshot_file).exists():
                Path(snapshot_file).unlink()
        except:
            pass


def _select_model_top_return(report_df: pd.DataFrame, model_name: str) -> float:
    """
    Extract the Top20% average return for the target model row.

    We avoid averaging across models to ensure each grid-search combination
    is scored on its own model performance.
    """
    if report_df is None or report_df.empty:
        return np.nan

    # Normalize names for robust matching
    name_col = report_df['Model'].astype(str).str.lower()
    target = model_name.lower()

    # Map legacy naming used in report to the grid-search model name
    aliases = {
        'ridge': ['ridge', 'ridge_stacking'],
        'lambdarank': ['lambdarank', 'lambda_rank'],
    }
    candidates = aliases.get(target, [target])

    mask = name_col.isin(candidates)
    if mask.any() and 'avg_top_return' in report_df.columns:
        return report_df.loc[mask, 'avg_top_return'].mean()

    # Fallback: if specific row missing, return overall average but warn
    if 'avg_top_return' in report_df.columns:
        logger.warning(f"Target model '{model_name}' not found in report; falling back to overall average")
        return report_df['avg_top_return'].mean()

    logger.error("avg_top_return not found in report")
    return np.nan


def run_backtest_for_snapshot(
    model_name: str,
    snapshot_id: str,
    data_dir: str,
    feature_list: List[str] | None = None,
    max_weeks: int | None = None,
    data_file: str | None = None,
    rebalance_mode: str = 'horizon',
    target_horizon_days: int = 10,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    allow_insample_backtest: bool = False,
) -> Tuple[float, Dict[str, Any], bool]:
    """
    Run backtest for a specific snapshot and extract Top20% average return.

    Args:
        snapshot_id: Snapshot ID to backtest
        data_dir: Data directory for backtest

    Returns:
        (top20_avg_return, full_metrics, success)
    """
    # Ensure feature whitelist is available to the backtest process
    prev_feat_env = os.environ.get("BMA_FEATURE_WHITELIST")
    # NOTE: empty list [] is meaningful ("compulsory-only" when enforced by model code)
    if feature_list is not None:
        os.environ["BMA_FEATURE_WHITELIST"] = json.dumps(feature_list)
    elif "BMA_FEATURE_WHITELIST" in os.environ:
        os.environ.pop("BMA_FEATURE_WHITELIST")

    try:
        logger.info(f"Running backtest for snapshot: {snapshot_id} (model: {model_name})")

        # Initialize backtest with specific snapshot
        backtest = ComprehensiveModelBacktest(
            data_dir=data_dir,
            snapshot_id=snapshot_id,
            data_file=data_file,
            start_date=start_date,
            end_date=end_date,
            allow_insample_backtest=allow_insample_backtest,
        )
        backtest._rebalance_mode = str(rebalance_mode)
        backtest._target_horizon_days = int(target_horizon_days)

        # Run backtest
        all_results, report_df, weekly_details_dict = backtest.run_backtest(max_weeks=max_weeks)

        if report_df.empty:
            logger.error("Backtest returned empty report")
            return np.nan, {}, False

        # Extract the Top20% return for the target model only
        top20_avg_return = _select_model_top_return(report_df, model_name)
        if np.isnan(top20_avg_return):
            return np.nan, {}, False

        # Collect full metrics
        full_metrics = report_df.to_dict(orient='records')

        logger.info(f"✅ Backtest completed, Top20% Avg Return ({model_name}): {top20_avg_return:.4f}%")
        return top20_avg_return, full_metrics, True

    except Exception as e:
        logger.error(f"Backtest exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return np.nan, {}, False
    finally:
        # Restore env
        if prev_feat_env is not None:
            os.environ["BMA_FEATURE_WHITELIST"] = prev_feat_env
        else:
            os.environ.pop("BMA_FEATURE_WHITELIST", None)


def grid_search_model(
    model_name: str,
    data_file: str,
    data_dir: str,
    base_config: str,
    snapshot_dir: str,
    output_dir: str,
    rebalance_mode: str,
    target_horizon_days: int,
    backtest_start_date: Optional[str],
    backtest_end_date: Optional[str],
    allow_insample_backtest: bool,
) -> pd.DataFrame:
    """
    Perform grid search for a single model.

    Args:
        model_name: Model to search
        data_file: Training data file
        data_dir: Data directory for backtest
        base_config: Base config path
        snapshot_dir: Snapshot directory
        output_dir: Output directory for results

    Returns:
        DataFrame with all results
    """
    logger.info("=" * 80)
    logger.info(f"🔍 Starting Grid Search for {model_name.upper()}")
    logger.info("=" * 80)

    # Get all parameter combinations
    param_combinations = get_param_combinations(model_name)
    total_combinations = len(param_combinations)

    logger.info(f"Total parameter combinations: {total_combinations}")
    logger.info(f"Estimated time: {total_combinations * 30} minutes (assuming 30 min/run)")

    results = []

    for idx, params in enumerate(param_combinations, 1):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[{idx}/{total_combinations}] {model_name.upper()} - Combination #{idx}")
        logger.info(f"Parameters: {params}")
        logger.info("=" * 80)

        # Step 1: Train model
        snapshot_id, train_success = run_single_training(
            model_name=model_name,
            params=params,
            data_file=data_file,
            base_config=base_config,
            snapshot_dir=snapshot_dir
        )

        if not train_success or not snapshot_id:
            logger.error(f"❌ Training failed for combination #{idx}")
            results.append({
                'model': model_name,
                'combination_id': idx,
                'params': json.dumps(params),
                **params,
                'snapshot_id': None,
                'top20_avg_return': np.nan,
                'train_success': False,
                'backtest_success': False,
                'timestamp': datetime.now().isoformat()
            })
            continue

        # Step 2: Run backtest
        top20_return, full_metrics, backtest_success = run_backtest_for_snapshot(
            model_name=model_name,
            snapshot_id=snapshot_id,
            data_dir=data_dir,
            data_file=data_file,
            rebalance_mode=rebalance_mode,
            target_horizon_days=target_horizon_days,
            start_date=backtest_start_date,
            end_date=backtest_end_date,
            allow_insample_backtest=allow_insample_backtest,
        )

        # Record result
        result_row = {
            'model': model_name,
            'combination_id': idx,
            'params': json.dumps(params),
            **params,
            'snapshot_id': snapshot_id,
            'top20_avg_return': top20_return,
            'train_success': train_success,
            'backtest_success': backtest_success,
            'timestamp': datetime.now().isoformat()
        }

        results.append(result_row)

        # Save intermediate results after each run
        results_df = pd.DataFrame(results)
        intermediate_path = Path(output_dir) / f"{model_name}_grid_search_intermediate.csv"
        results_df.to_csv(intermediate_path, index=False)
        logger.info(f"💾 Intermediate results saved: {intermediate_path}")

        logger.info(f"✅ Combination #{idx} completed: Top20% Return = {top20_return:.4f}%")

    # Final results
    results_df = pd.DataFrame(results)

    # Sort by Top20% return (descending)
    results_df = results_df.sort_values('top20_avg_return', ascending=False)

    # Save final results
    final_path = Path(output_dir) / f"{model_name}_grid_search_final.csv"
    results_df.to_csv(final_path, index=False)

    logger.info("=" * 80)
    logger.info(f"✅ {model_name.upper()} Grid Search COMPLETED")
    logger.info(f"Total runs: {len(results_df)}")
    logger.info(f"Successful: {results_df['train_success'].sum()}")
    logger.info(f"Best Top20% Return: {results_df['top20_avg_return'].max():.4f}%")
    logger.info(f"Results saved: {final_path}")
    logger.info("=" * 80)

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Full Grid Search with Backtest Integration'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to training data file (MultiIndex parquet/csv)'
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
        '--models',
        nargs='+',
        choices=['elastic_net', 'xgboost', 'catboost', 'lambdarank', 'ridge'],
        default=['elastic_net', 'xgboost', 'catboost', 'lambdarank'],
        help='Models to search (default: base models only; run ridge separately after bases are tuned)'
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
        default='cache/grid_search_snapshots',
        help='Directory to save model snapshots'
    )
    parser.add_argument(
        '--rebalance-mode',
        type=str,
        default='horizon',
        choices=['horizon', 'weekly'],
        help="Backtest rebalance cadence"
    )
    parser.add_argument(
        '--target-horizon-days',
        type=int,
        default=10,
        help="Target horizon used to align evaluation sampling (trading days)."
    )
    parser.add_argument(
        '--backtest-start-date',
        type=str,
        default=None,
        help="First evaluation date (inclusive, YYYY-MM-DD)."
    )
    parser.add_argument(
        '--backtest-end-date',
        type=str,
        default=None,
        help="Last evaluation date (inclusive, YYYY-MM-DD)."
    )
    parser.add_argument(
        '--allow-insample-backtest',
        action='store_true',
        help="Allow running when no out-of-sample data exists (not recommended)."
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
    logger.info("🚀 FULL GRID SEARCH WITH BACKTEST INTEGRATION")
    logger.info("=" * 80)
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Snapshot directory: {args.snapshot_dir}")
    logger.info(f"Rebalance mode: {args.rebalance_mode} (target horizon={args.target_horizon_days})")
    if args.backtest_start_date or args.backtest_end_date:
        logger.info("Backtest window: %s -> %s" % (args.backtest_start_date or '-inf', args.backtest_end_date or '+inf'))
    logger.info(f"Allow in-sample override: {bool(args.allow_insample_backtest)}")

    # Calculate total runs
    total_runs = sum(
        len(get_param_combinations(model))
        for model in args.models
    )
    logger.info(f"Total training+backtest runs: {total_runs}")
    logger.info("=" * 80)

    # Run grid search for each model
    all_model_results = {}

    for model_name in args.models:
        if model_name == 'ridge' and len(args.models) > 1:
            logger.warning("Skipping ridge stacking in mixed run. Tune base models first, then run ridge alone.")
            continue
        try:
            results_df = grid_search_model(
                model_name=model_name,
                data_file=args.data_file,
                data_dir=args.data_dir,
                base_config=args.base_config,
                snapshot_dir=args.snapshot_dir,
                output_dir=str(output_dir),
                rebalance_mode=args.rebalance_mode,
                target_horizon_days=args.target_horizon_days,
                backtest_start_date=args.backtest_start_date,
                backtest_end_date=args.backtest_end_date,
                allow_insample_backtest=args.allow_insample_backtest,
            )
            all_model_results[model_name] = results_df
        except Exception as e:
            logger.error(f"Grid search failed for {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Generate comprehensive report
    logger.info("=" * 80)
    logger.info("📊 Generating Comprehensive Report")
    logger.info("=" * 80)

    # Combine all results
    combined_results = pd.concat(
        [df for df in all_model_results.values()],
        ignore_index=True
    )

    # Save combined results
    combined_path = output_dir / "all_models_grid_search_results.csv"
    combined_results.to_csv(combined_path, index=False)
    logger.info(f"Combined results saved: {combined_path}")

    # Generate summary for each model
    summary_rows = []
    for model_name, results_df in all_model_results.items():
        best_row = results_df.loc[results_df['top20_avg_return'].idxmax()]
        summary_rows.append({
            'Model': model_name,
            'Total_Combinations': len(results_df),
            'Successful_Runs': results_df['train_success'].sum(),
            'Best_Top20_Return': results_df['top20_avg_return'].max(),
            'Best_Params': best_row['params'],
            'Best_Snapshot_ID': best_row['snapshot_id']
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('Best_Top20_Return', ascending=False)

    summary_path = output_dir / "grid_search_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary saved: {summary_path}")

    logger.info("=" * 80)
    logger.info("🎉 GRID SEARCH COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\nBest Parameters Summary:")
    print(summary_df.to_string())
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

