#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Prediction-Only System
============================
Test the prediction-only engine with sample stocks.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bma_models.prediction_only_engine import create_prediction_engine
from datetime import datetime, timedelta


def test_prediction():
    """Test prediction-only system"""
    print("=" * 80)
    print("Testing Prediction-Only System")
    print("=" * 80)

    # Sample stocks to test
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX']

    # Date range (last 365 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    print(f"\nTest stocks: {', '.join(test_stocks)}")
    print(f"Date range: {start_date} to {end_date}")
    print()

    try:
        # Create prediction engine (loads latest snapshot)
        print("Creating prediction engine...")
        engine = create_prediction_engine(snapshot_id=None)

        print("\nRunning predictions...")
        results = engine.predict(
            tickers=test_stocks,
            start_date=start_date,
            end_date=end_date,
            top_n=len(test_stocks)
        )

        if results['success']:
            print("\n" + "=" * 80)
            print("✅ PREDICTION SUCCESS")
            print("=" * 80)

            print(f"\nSnapshot ID: {results['snapshot_id']}")
            print(f"Input stocks: {results['n_stocks']}")
            print(f"Date range: {results['date_range']}")

            print("\n" + "-" * 80)
            print("Top Recommendations:")
            print("-" * 80)

            for rec in results['recommendations']:
                print(f"  {rec['rank']:2d}. {rec['ticker']:6s}  Score: {rec['score']:8.6f}")

            print("=" * 80)

        else:
            print("\n❌ PREDICTION FAILED")
            print(f"Error: {results.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_prediction()
