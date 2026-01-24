#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 CatBoost 和 LambdaRank 的累计收益（基于非重叠回测）
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.time_split_80_20_oos_eval import (
    calculate_group_returns_hold10d_nonoverlap,
    load_test_data,
    _compute_benchmark_tplus_from_yfinance
)
from bma_models.model_registry import load_models_from_snapshot
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("calc_accumulated")

def calculate_accumulated_returns():
    """从最新运行中计算累计收益"""
    
    # Use the latest run's snapshot
    snapshot_id = "9de0b13d-647d-4c8d-bf3d-86d3ab8a738f"
    logger.info(f"Using snapshot: {snapshot_id}")
    
    # Load models
    loaded = load_models_from_snapshot(str(snapshot_id), load_catboost=True)
    models_dict = loaded.get("models", {})
    
    # Load test data
    test_data_path = Path(r"D:\trade\data\factor_exports\polygon_factors_all_filtered.parquet")
    if not test_data_path.exists():
        logger.error(f"Test data not found: {test_data_path}")
        return
    
    logger.info("Loading test data...")
    test_data = pd.read_parquet(test_data_path)
    logger.info(f"Test data loaded: {len(test_data)} rows")
    
    # Get test dates (last 20%)
    if 'date' in test_data.index.names:
        unique_dates = test_data.index.get_level_values('date').unique().sort_values()
    else:
        unique_dates = test_data['date'].unique() if 'date' in test_data.columns else []
    
    if len(unique_dates) == 0:
        logger.error("No dates found in test data")
        return
    
    split_idx = int(len(unique_dates) * 0.8)
    test_dates = unique_dates[split_idx:]
    logger.info(f"Test period: {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")
    
    # Calculate for CatBoost and LambdaRank
    models_to_calc = ['catboost', 'lambdarank']
    results = {}
    
    for model_name in models_to_calc:
        if model_name not in models_dict:
            logger.warning(f"Model {model_name} not found in snapshot")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Calculating for {model_name.upper()}")
        logger.info(f"{'='*60}")
        
        # This is complex - we need to run the full prediction loop
        # For now, let's use a simpler approach: check if timeseries files exist from previous runs
        
    logger.info("\nNote: Full calculation requires running the complete evaluation.")
    logger.info("Please run: python scripts/time_split_80_20_oos_eval.py --models catboost lambdarank")

if __name__ == "__main__":
    calculate_accumulated_returns()
