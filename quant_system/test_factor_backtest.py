"""
Quick Test Script for Factor-Based Backtest
============================================

Tests the factor data loading and backtest setup.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.factor_data_loader import FactorDataLoader

def test_factor_loading():
    """Test loading factor data."""
    print("="*80)
    print("TESTING FACTOR DATA LOADING")
    print("="*80)
    
    factor_file = r'D:\trade\quant_system\data\polygon_factors_all_2021_2026_T5_final.parquet'
    
    if not os.path.exists(factor_file):
        print(f"ERROR: Factor file not found: {factor_file}")
        return False
    
    print(f"\nLoading factor file: {factor_file}")
    
    try:
        loader = FactorDataLoader(factor_file)
        
        # Load factors
        factor_data = loader.load_factors()
        print(f"\n[OK] Successfully loaded factor data")
        print(f"  Shape: {factor_data.shape}")
        print(f"  Columns: {len(factor_data.columns)}")
        print(f"  Tickers: {len(factor_data.index.get_level_values(1).unique())}")
        
        # Get tickers
        tickers = loader.get_tickers()
        print(f"\n[OK] Found {len(tickers)} tickers")
        print(f"  First 10: {tickers[:10]}")
        
        # Get date range
        start_date, end_date = loader.get_date_range()
        print(f"\n[OK] Date range: {start_date} to {end_date}")
        
        # Test with small subset
        print(f"\n" + "-"*80)
        print("TESTING WITH SMALL SUBSET (5 tickers)")
        print("-"*80)
        
        test_tickers = tickers[:5]
        print(f"Test tickers: {test_tickers}")
        
        # This will fetch OHLCV data - may take a while
        print("\nFetching OHLCV data (this may take a few minutes)...")
        print("(You can cancel and run full backtest if this works)")
        
        merged_data = loader.merge_factors_with_ohlcv(
            tickers=test_tickers,
            use_cache=True
        )
        
        print(f"\n[OK] Successfully merged data")
        print(f"  Shape: {merged_data.shape}")
        print(f"  Columns: {merged_data.columns.tolist()}")
        
        # Check if OHLCV columns exist
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in ohlcv_cols if col not in merged_data.columns]
        if missing:
            print(f"  [WARNING] Missing OHLCV columns: {missing}")
        else:
            print(f"  [OK] All OHLCV columns present")
        
        # Convert to dict format
        print("\nConverting to dictionary format...")
        universe_dict = loader.get_universe_dict(merged_data)
        print(f"[OK] Converted to dictionary: {len(universe_dict)} tickers")
        
        # Check first ticker
        first_ticker = list(universe_dict.keys())[0]
        first_df = universe_dict[first_ticker]
        print(f"\nSample data for {first_ticker}:")
        print(first_df.head())
        
        print("\n" + "="*80)
        print("[OK] ALL TESTS PASSED")
        print("="*80)
        print("\nYou can now run the full backtest:")
        print("  python quant_system/run_backtest_with_factors.py")
        print("\nOr with specific tickers:")
        print("  python quant_system/run_backtest_with_factors.py --tickers AAPL MSFT GOOGL")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_factor_loading()
    sys.exit(0 if success else 1)
