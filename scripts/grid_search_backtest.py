#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grid-search training + backtest for ElasticNet/XGBoost/CatBoost/LambdaRank/Ridge."""

import os
import sys
import json
import itertools
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('grid_search_backtest')

TRAINING_DATA_PATH = 'data/factor_exports/factors/factors_all.parquet'
BACKTEST_SCRIPT = os.path.join('scripts', 'comprehensive_model_backtest.py')
OUTPUT_DIR = Path('result/grid_search_backtest')

ELASTIC_ALPHA_BASE = 1e-6
ELASTIC_L1_BASE = 0.001
ELASTIC_ALPHA_GRID = [ELASTIC_ALPHA_BASE * (10 ** i) for i in range(-2, 3)]
ELASTIC_L1_GRID = [ELASTIC_L1_BASE * (10 ** i) for i in range(-2, 3)]

XGB_BASE = {
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.05,
    'min_child_weight': 3,
    'reg_alpha': 0.005,
    'reg_lambda': 0.05,
}

XGB_RANGES = {
    'n_estimators': [max(50, XGB_BASE['n_estimators'] + 100 * i) for i in range(-2, 3)],
    'max_depth': [max(2, XGB_BASE['max_depth'] + i) for i in range(-2, 3)],
    'learning_rate': [max(0.005, XGB_BASE['learning_rate'] * (1.5 ** i)) for i in range(-2, 3)],
    'min_child_weight': [max(1, XGB_BASE['min_child_weight'] + i) for i in range(-2, 3)],
    'reg_alpha': [max(1e-6, XGB_BASE['reg_alpha'] * (2 ** i)) for i in range(-2, 3)],
    'reg_lambda': [max(1e-6, XGB_BASE['reg_lambda'] * (2 ** i)) for i in range(-2, 3)],
}

CAT_BASE = {
    'iterations': 3000,
    'depth': 6,
    'learning_rate': 0.03,
    'l2_leaf_reg': 1.0,
    'subsample': 0.8,
    'rsm': 0.85,
}

CAT_RANGES = {
    'iterations': [max(500, CAT_BASE['iterations'] + 500 * i) for i in range(-2, 3)],
    'depth': [max(2, CAT_BASE['depth'] + i) for i in range(-2, 3)],
    'learning_rate': [max(0.005, CAT_BASE['learning_rate'] * (1.5 ** i)) for i in range(-2, 3)],
    'l2_leaf_reg': [max(0.01, CAT_BASE['l2_leaf_reg'] * (2 ** i)) for i in range(-2, 3)],
    'subsample': [min(0.99, max(0.4, CAT_BASE['subsample'] + 0.1 * i)) for i in range(-2, 3)],
    'rsm': [min(0.99, max(0.4, CAT_BASE['rsm'] + 0.1 * i)) for i in range(-2, 3)],
}

LAMBDA_BASE = {
    'n_quantiles': 128,
    'label_gain_power': 1.5,
    'num_boost_round': 100,
}

LAMBDA_RANGES = {
    'n_quantiles': [max(32, LAMBDA_BASE['n_quantiles'] + 32 * i) for i in range(-2, 3)],
    'label_gain_power': [max(0.5, LAMBDA_BASE['label_gain_power'] + 0.5 * i) for i in range(-2, 3)],
    'num_boost_round': [max(20, LAMBDA_BASE['num_boost_round'] + 20 * i) for i in range(-2, 3)],
}

RIDGE_ALPHA_BASE = 1.0
RIDGE_ALPHA_GRID = [RIDGE_ALPHA_BASE * (2 ** i) for i in range(-2, 3)]

def generate_parameter_grid():
    for e_alpha, e_l1 in itertools.product(ELASTIC_ALPHA_GRID, ELASTIC_L1_GRID):
        for xgb_params in itertools.product(*XGB_RANGES.values()):
            xgb_dict = dict(zip(XGB_RANGES.keys(), xgb_params))
            for cat_params in itertools.product(*CAT_RANGES.values()):
                cat_dict = dict(zip(CAT_RANGES.keys(), cat_params))
                for lmd_params in itertools.product(*LAMBDA_RANGES.values()):
                    lmd_dict = dict(zip(LAMBDA_RANGES.keys(), lmd_params))
                    for ridge_alpha in RIDGE_ALPHA_GRID:
                        yield {
                            'elastic': {'alpha': e_alpha, 'l1_ratio': e_l1},
                            'xgboost': xgb_dict,
                            'catboost': cat_dict,
                            'lambda': lmd_dict,
                            'ridge_alpha': ridge_alpha,
                        }

def override_config(model: UltraEnhancedQuantitativeModel, params: Dict[str, Any]) -> None:
    cfg = {}
    cfg['base_models'] = {
        'elastic_net': {
            'alpha': params['elastic']['alpha'],
            'l1_ratio': params['elastic']['l1_ratio'],
        },
        'xgboost': params['xgboost'],
        'catboost': params['catboost'],
        'lambda_rank': params['lambda'],
    }
    cfg['ridge'] = {'alpha': params['ridge_alpha']}
    model.CONFIG_OVERRIDE = cfg

def run_training(param_set: Dict[str, Any]) -> str:
    logger.info(f"Training with params: {json.dumps(param_set)}")
    model = UltraEnhancedQuantitativeModel()
    override_config(model, param_set)
    report = model.train_from_document(TRAINING_DATA_PATH, top_n=50)
    if not report.get('success'):
        raise RuntimeError('Training failed')
    snapshot_id = getattr(model, 'active_snapshot_id', None)
    if not snapshot_id:
        raise RuntimeError('Snapshot missing after training')
    return snapshot_id

def run_backtest(snapshot_id: str) -> Dict[str, Any]:
    cmd = [sys.executable, BACKTEST_SCRIPT, '--snapshot-id', snapshot_id]
    logger.info(f"Running backtest for snapshot {snapshot_id}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error(proc.stdout)
        logger.error(proc.stderr)
        raise RuntimeError('Backtest failed')
    return parse_backtest_output(proc.stdout)

def parse_backtest_output(output: str) -> Dict[str, Any]:
    metric_line = None
    for line in output.splitlines():
        # Backward compatible: accept either legacy "Top 20% Avg Return" or new "Top 30 Avg Return"
        if ('Top 20% Avg Return' in line) or ('Top 30 Avg Return' in line):
            metric_line = line
    if not metric_line:
        return {'avg_top_return': float('nan')}
    try:
        value = metric_line.split(':')[-1].strip().rstrip('%')
        return {'avg_top_return': float(value)}
    except Exception:
        return {'avg_top_return': float('nan')}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / f"grid_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows: List[str] = ['snapshot,avg_top_return,params']
    for idx, param_set in enumerate(generate_parameter_grid(), 1):
        try:
            snapshot = run_training(param_set)
            metric = run_backtest(snapshot)
            avg_top = metric.get('avg_top_return', float('nan'))
            rows.append(f"{snapshot},{avg_top},{json.dumps(param_set)}")
            logger.info(f"[# {idx}] Snapshot {snapshot} Top20 avg={avg_top:.4f}%")
        except Exception as exc:
            logger.error(f"[# {idx}] combination failed: {exc}")
    results_file.write_text('\r\n'.join(rows), encoding='utf-8')
    logger.info(f"Grid search results saved to {results_file}")

if __name__ == '__main__':
    main()
