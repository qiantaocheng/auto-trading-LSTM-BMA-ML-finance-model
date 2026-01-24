#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 CatBoost 和 LambdaRank 的累计收益（基于非重叠回测）
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.time_split_80_20_oos_eval import calculate_group_returns_hold10d_nonoverlap

def calculate_accumulated_returns_for_models(
    snapshot_id: str,
    models: list = ['catboost', 'lambdarank'],
    top_n: int = 20,
    horizon_days: int = 10,
    cost_bps: float = 10.0,
    output_dir: str = "results/t10_time_split_80_20"
):
    """
    计算指定模型的累计收益（基于非重叠回测）
    """
    from bma_models.model_registry import load_models_from_snapshot
    from scripts.time_split_80_20_oos_eval import (
        load_test_data, 
        _compute_benchmark_tplus_from_yfinance,
        _write_model_topn_vs_benchmark
    )
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("accumulated_returns")
    
    # Load models
    logger.info(f"Loading snapshot: {snapshot_id}")
    loaded = load_models_from_snapshot(str(snapshot_id), load_catboost=True)
    models_dict = loaded.get("models", {})
    
    # Load test data (need to get from the evaluation script)
    # For now, we'll use a simpler approach: load predictions from a previous run
    # or calculate from scratch
    
    logger.info("This script requires predictions data. Please run the full evaluation first.")
    logger.info("Or use the existing timeseries files if available.")
    
    return None

if __name__ == "__main__":
    # Quick calculation from existing data if available
    import glob
    
    result_dirs = glob.glob("results/t10_time_split_80_20/run_*")
    if result_dirs:
        latest_dir = max(result_dirs, key=lambda x: Path(x).stat().st_mtime)
        print(f"Checking latest run: {latest_dir}")
        
        # Check for timeseries files
        catboost_file = Path(latest_dir) / "catboost_top20_timeseries.csv"
        lambdarank_file = Path(latest_dir) / "lambdarank_top20_timeseries.csv"
        
        if catboost_file.exists():
            df_cb = pd.read_csv(catboost_file)
            print(f"\n=== CatBoost 累计收益 ===")
            print(f"时间序列行数: {len(df_cb)}")
            if 'cum_top_return_net' in df_cb.columns:
                final_return = df_cb['cum_top_return_net'].iloc[-1]
                print(f"最终累计收益 (Net): {final_return:.2f}%")
            elif 'cum_top_return' in df_cb.columns:
                final_return = df_cb['cum_top_return'].iloc[-1]
                print(f"最终累计收益 (Gross): {final_return:.2f}%")
        
        if lambdarank_file.exists():
            df_lr = pd.read_csv(lambdarank_file)
            print(f"\n=== LambdaRank 累计收益 ===")
            print(f"时间序列行数: {len(df_lr)}")
            if 'cum_top_return_net' in df_lr.columns:
                final_return = df_lr['cum_top_return_net'].iloc[-1]
                print(f"最终累计收益 (Net): {final_return:.2f}%")
            elif 'cum_top_return' in df_lr.columns:
                final_return = df_lr['cum_top_return'].iloc[-1]
                print(f"最终累计收益 (Gross): {final_return:.2f}%")
    else:
        print("No results directory found. Please run the evaluation first.")
