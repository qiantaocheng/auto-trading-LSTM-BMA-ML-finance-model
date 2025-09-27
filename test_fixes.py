#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify the fixes for:
1. EnhancedBottom20PenaltySystem method name issue
2. Excel export NoneType error
"""

import sys
import logging
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_penalty_system():
    """Test the EnhancedBottom20PenaltySystem fix"""
    try:
        logger.info("=" * 80)
        logger.info("Testing EnhancedBottom20PenaltySystem fix...")
        logger.info("=" * 80)

        from bma_models.enhanced_bottom20_penalty_system import EnhancedBottom20PenaltySystem

        # Create test data
        n_stocks = 100
        predictions = pd.Series(
            np.random.randn(n_stocks),
            index=pd.MultiIndex.from_product(
                [[pd.Timestamp('2025-09-26')],
                 [f'STOCK_{i:03d}' for i in range(n_stocks)]],
                names=['date', 'ticker']
            )
        )

        # Create feature data
        feature_data = pd.DataFrame({
            'market_cap': np.random.uniform(1e8, 1e11, n_stocks),
            'Volume': np.random.uniform(1e6, 1e9, n_stocks),
            'Close': np.random.uniform(10, 500, n_stocks),
            'returns': np.random.randn(n_stocks) * 0.02
        }, index=predictions.index)

        # Initialize penalty system
        penalty_system = EnhancedBottom20PenaltySystem(
            penalty_threshold=0.20,
            initial_penalty_factor=0.010,
            max_penalty=0.12
        )

        # Test the correct method name
        adjusted_predictions, diagnostics = penalty_system.apply_enhanced_bottom8_penalty(
            predictions=predictions,
            feature_data=feature_data
        )

        logger.info("✅ EnhancedBottom20PenaltySystem fix verified!")
        logger.info(f"   Original mean: {predictions.mean():.4f}")
        logger.info(f"   Adjusted mean: {adjusted_predictions.mean():.4f}")
        logger.info(f"   Penalized stocks: {diagnostics.get('penalized_stocks', 0)}")
        return True

    except AttributeError as e:
        logger.error(f"❌ Method name issue still exists: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error in penalty system test: {e}")
        return False

def test_excel_export():
    """Test the Excel export NoneType fix"""
    try:
        logger.info("\n" + "=" * 80)
        logger.info("Testing Excel export fix...")
        logger.info("=" * 80)

        from bma_models.corrected_prediction_exporter import CorrectedPredictionExporter

        # Test 1: Empty predictions
        logger.info("\nTest 1: Empty predictions...")
        exporter = CorrectedPredictionExporter(output_dir="D:/trade/test_results")

        # This should handle empty arrays gracefully now
        empty_predictions = np.array([])
        empty_dates = []
        empty_tickers = []

        try:
            # This would previously fail with NoneType error
            result = exporter.export_predictions(
                predictions=empty_predictions,
                dates=empty_dates,
                tickers=empty_tickers,
                filename="test_empty.xlsx"
            )
            logger.info("✅ Empty predictions handled gracefully")
        except TypeError as e:
            if "NoneType" in str(e):
                logger.error(f"❌ NoneType issue still exists: {e}")
                return False
            raise

        # Test 2: Normal predictions
        logger.info("\nTest 2: Normal predictions...")
        predictions = np.random.randn(10)
        dates = [pd.Timestamp('2025-09-26')] * 10
        tickers = [f'TEST_{i:02d}' for i in range(10)]

        result = exporter.export_predictions(
            predictions=predictions,
            dates=dates,
            tickers=tickers,
            filename="test_normal.xlsx"
        )

        logger.info(f"✅ Excel export completed: {result}")
        return True

    except Exception as e:
        logger.error(f"❌ Excel export test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests"""
    logger.info("\n🧪 Starting fix verification tests...\n")

    # Run tests
    penalty_test_passed = test_penalty_system()
    excel_test_passed = test_excel_export()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 80)

    if penalty_test_passed:
        logger.info("✅ EnhancedBottom20PenaltySystem fix: PASSED")
    else:
        logger.info("❌ EnhancedBottom20PenaltySystem fix: FAILED")

    if excel_test_passed:
        logger.info("✅ Excel export fix: PASSED")
    else:
        logger.info("❌ Excel export fix: FAILED")

    if penalty_test_passed and excel_test_passed:
        logger.info("\n🎉 All fixes verified successfully!")
        return 0
    else:
        logger.info("\n⚠️ Some fixes need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())