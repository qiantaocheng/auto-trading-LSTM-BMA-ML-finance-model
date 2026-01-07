#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Test - Direct Execution with Backtest Validation
===========================================================

Run a train + backtest loop using the aggregated factor file to validate the pipeline.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FACTORS_FILE = Path('data/factor_exports/factors/factors_all.parquet')
FACTORS_DIR = FACTORS_FILE.parent

def test_single_training_with_backtest() -> bool:
    """Train the full model on the aggregated factor file and run a backtest."""
    logger.info('=' * 80)
    logger.info('🚀 INTEGRATED TEST - Training + Backtest Validation')
    logger.info('=' * 80)

    data_file = str(FACTORS_FILE)
    data_dir = str(FACTORS_DIR)

    try:
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

        model = UltraEnhancedQuantitativeModel()
        logger.info(f'Loading training data from: {data_file}')
        training_results = model.train_from_document(training_data_path=data_file, top_n=10)
        if not training_results.get('success'):
            raise RuntimeError(f"Training failed: {training_results.get('error')}")
        snapshot_id = getattr(model, 'active_snapshot_id', None)
        if not snapshot_id:
            raise RuntimeError('Snapshot ID not found after training')
        logger.info('✅ Training completed successfully')
        logger.info(f'✅ Snapshot ID: {snapshot_id}')
    except Exception as exc:  # pragma: no cover
        logger.error(f'❌ Training failed: {exc}')
        import traceback
        logger.error(traceback.format_exc())
        return False

    try:
        from scripts.comprehensive_model_backtest import ComprehensiveModelBacktest

        logger.info(f'Initializing backtest for snapshot: {snapshot_id}')
        backtest = ComprehensiveModelBacktest(
            data_dir=data_dir,
            snapshot_id=snapshot_id,
            data_file=data_file,
        )
        _, report_df, _ = backtest.run_backtest()
        if report_df.empty:
            raise RuntimeError('Backtest returned empty report')

        logger.info('=' * 80)
        logger.info('📈 BACKTEST RESULTS')
        logger.info('=' * 80)
        for _, row in report_df.iterrows():
            if 'avg_top_return' in row:
                logger.info(f"{row['Model']}: Top20% Avg Return = {row['avg_top_return']:.4f}%")

        output_dir = Path('results/integrated_test')
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = output_dir / f'test_report_{timestamp}.csv'
        report_df.to_csv(report_path, index=False)
        logger.info(f'📄 Report saved: {report_path}')
        logger.info('=' * 80)
        logger.info('🎉 INTEGRATED TEST PASSED')
        logger.info('=' * 80)
        return True
    except Exception as exc:  # pragma: no cover
        logger.error(f'❌ Backtest failed: {exc}')
        import traceback
        logger.error(traceback.format_exc())
        return False


def main() -> None:
    if not FACTORS_FILE.exists():
        logger.error(f'❌ Data file not found: {FACTORS_FILE}')
        logger.error('Please ensure the aggregated factor file exists before running the test')
        sys.exit(1)
    if not FACTORS_DIR.exists():
        logger.error(f'❌ Data directory not found: {FACTORS_DIR}')
        sys.exit(1)

    logger.info('✅ All prerequisite files found')
    logger.info('')

    success = test_single_training_with_backtest()
    if success:
        logger.info('')
        logger.info('👉 You can now run the full grid search:')
        logger.info('   python scripts/full_grid_search.py \\')
        logger.info(f'       --data-file {FACTORS_FILE} \\')
        logger.info(f'       --data-dir {FACTORS_DIR} \\')
        logger.info('       --output-dir results/grid_search_$(date +%Y%m%d) \\')
        logger.info('       --models elastic_net')
    else:
        logger.error('')
        logger.error('❌ Test failed. Please check the errors above.')
        sys.exit(1)


if __name__ == '__main__':
    main()
