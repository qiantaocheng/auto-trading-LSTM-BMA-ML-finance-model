#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Sector Neutralization Analysis using existing backtest results
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.professional_paper_sector_neutralization import analyze_sector_neutralization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Use the latest backtest run
    run_dir = Path("results/t10_time_split_bucket_analysis_20260110_192509/run_20260110_192510")
    
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return 1
    
    # We need to create predictions file from the backtest
    # For now, use the factor data and generate predictions
    data_file = "data/factor_exports/factors/factors_all.parquet"
    output_dir = run_dir / "sector_neutralization"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load snapshot ID
    snapshot_id_path = run_dir / "snapshot_id.txt"
    if not snapshot_id_path.exists():
        logger.error(f"Snapshot ID not found: {snapshot_id_path}")
        return 1
    
    snapshot_id = snapshot_id_path.read_text().strip()
    logger.info(f"Using snapshot: {snapshot_id}")
    
    # Load OOS metrics to get test period
    oos_metrics_path = run_dir / "oos_metrics.csv"
    if not oos_metrics_path.exists():
        logger.error(f"OOS metrics not found: {oos_metrics_path}")
        return 1
    
    oos_metrics = pd.read_csv(oos_metrics_path)
    test_start = oos_metrics['test_start'].iloc[0]
    test_end = oos_metrics['test_end'].iloc[0]
    
    logger.info(f"Test period: {test_start} to {test_end}")
    
    # We'll use ComprehensiveModelBacktest to generate predictions
    from scripts.comprehensive_model_backtest import ComprehensiveModelBacktest
    
    bt = ComprehensiveModelBacktest(
        data_file=data_file,
        snapshot_id=snapshot_id,
        load_catboost=True
    )
    
    # Load factor data
    logger.info("Loading factor data...")
    factors_df = bt.load_factor_data()
    
    # Filter to test period
    test_factors = factors_df.loc[
        (factors_df.index.get_level_values('date') >= pd.to_datetime(test_start)) &
        (factors_df.index.get_level_values('date') <= pd.to_datetime(test_end))
    ]
    
    logger.info(f"Test data shape: {test_factors.shape}")
    
    # Generate predictions for ridge_stacking
    logger.info("Generating predictions...")
    predictions_dict = {}
    
    # Get rebalance dates
    rebalance_dates = bt.get_rebalance_dates(
        test_factors,
        rebalance_mode="horizon",
        target_horizon_days=10
    )
    
    logger.info(f"Number of rebalance dates: {len(rebalance_dates)}")
    
    # Generate predictions for each rebalance date
    all_predictions = []
    
    for date in rebalance_dates[:10]:  # Limit to first 10 for testing
        date_data = test_factors.loc[test_factors.index.get_level_values('date') == date]
        
        if len(date_data) == 0:
            continue
        
        # Get predictions
        if 'ridge_stacking' in bt.models:
            model = bt.models['ridge_stacking']
            feature_cols = [col for col in date_data.columns if col not in ['target', 'Close', 'Open', 'High', 'Low', 'Volume']]
            X = date_data[feature_cols].fillna(0)
            
            try:
                pred = model.predict(X)
                pred_df = pd.DataFrame({
                    'prediction': pred,
                    'actual': date_data['target'].values if 'target' in date_data.columns else np.nan
                }, index=date_data.index)
                all_predictions.append(pred_df)
            except Exception as e:
                logger.warning(f"Failed to predict for {date}: {e}")
                continue
    
    if not all_predictions:
        logger.error("No predictions generated")
        return 1
    
    predictions = pd.concat(all_predictions)
    logger.info(f"Generated {len(predictions)} predictions")
    
    # Save predictions
    predictions_path = output_dir / "predictions.parquet"
    predictions.to_parquet(predictions_path)
    logger.info(f"Saved predictions to {predictions_path}")
    
    # Run sector neutralization analysis
    logger.info("Running sector neutralization analysis...")
    analyze_sector_neutralization(
        predictions_file=str(predictions_path),
        data_file=data_file,
        output_dir=str(output_dir),
        model_name="ridge_stacking",
        top_k=30,
        use_yfinance=True
    )
    
    logger.info("✅ Sector neutralization analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

