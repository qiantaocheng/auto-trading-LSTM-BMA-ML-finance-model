#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify data acquisition functionality across all BMA versions
"""

import os
import sys
import traceback
import pandas as pd
from datetime import datetime, timedelta

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def test_polygon_client():
    """Test direct Polygon client functionality"""
    print("=" * 60)
    print("TESTING POLYGON CLIENT DIRECTLY")
    print("=" * 60)
    
    try:
        from polygon_client import polygon_client
        
        # Test basic connection
        print("SUCCESS: Polygon client imported successfully")
        print(f"  - API Key configured: {'***' + polygon_client.api_key[-8:] if polygon_client.api_key else 'NO API KEY'}")
        print(f"  - Delayed data mode: {polygon_client.delayed_data_mode}")
        
        # Test data download
        print("\nTesting data download for AAPL...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        data = polygon_client.download(['AAPL'], start_date, end_date)
        
        if not data.empty:
            print("SUCCESS: Data downloaded successfully!")
            print(f"  - Shape: {data.shape}")
            print(f"  - Date range: {data.index.min()} to {data.index.max()}")
            print(f"  - Columns: {list(data.columns)}")
            print(f"  - Last close price: ${data['Close'].iloc[-1]:.2f}")
            return True, data
        else:
            print("FAIL: No data returned from Polygon")
            return False, None
            
    except Exception as e:
        print(f"FAIL: Polygon client test failed: {e}")
        traceback.print_exc()
        return False, None

def test_original_bma_data():
    """Test data getting in original BMA file"""
    print("\n" + "=" * 60)
    print("TESTING ORIGINAL BMA ULTRA ENHANCED")
    print("=" * 60)
    
    try:
        # Import the original BMA model
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        
        print("SUCCESS: Original BMA model imported successfully")
        
        # Initialize model
        model = UltraEnhancedQuantitativeModel()
        print("SUCCESS: Model initialized")
        
        # Test data acquisition
        print("\nTesting get_data_and_features...")
        tickers = ['AAPL', 'MSFT']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        feature_data = model.get_data_and_features(tickers, start_date, end_date)
        
        if feature_data is not None and not feature_data.empty:
            print("SUCCESS: Feature data acquired successfully!")
            print(f"  - Shape: {feature_data.shape}")
            print(f"  - Columns: {list(feature_data.columns)[:10]}...")  # First 10 columns
            print(f"  - Index type: {type(feature_data.index)}")
            if hasattr(feature_data.index, 'names'):
                print(f"  - Index names: {feature_data.index.names}")
            return True, feature_data
        else:
            print("FAIL: No feature data returned from original BMA")
            return False, None
            
    except ImportError as ie:
        print(f"FAIL: Cannot import original BMA: {ie}")
        return False, None
    except Exception as e:
        print(f"FAIL: Original BMA test failed: {e}")
        traceback.print_exc()
        return False, None

def main():
    """Main test execution"""
    print("STARTING BMA DATA ACQUISITION TESTS")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Polygon Client
    polygon_success, polygon_data = test_polygon_client()
    results['polygon_client'] = polygon_success
    
    # Test 2: Original BMA
    original_success, original_data = test_original_bma_data()
    results['original_bma'] = original_success
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for component, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{component.upper():<20}: {status}")
    
    total_pass = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_pass}/{total_tests} tests passed")
    
    if total_pass == total_tests:
        print("All data acquisition tests PASSED!")
    else:
        print("Some data acquisition tests FAILED!")
        print("Check individual test outputs above for details.")

if __name__ == "__main__":
    main()