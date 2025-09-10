#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify data acquisition functionality in refactored versions
"""

import os
import sys
import traceback
import pandas as pd
from datetime import datetime, timedelta

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

def test_refactored_bma_data():
    """Test data getting in refactored BMA file"""
    print("=" * 60)
    print("TESTING REFACTORED BMA")
    print("=" * 60)
    
    try:
        from bma_models.bma_ultra_enhanced_refactored import UltraEnhancedQuantitativeModel
        
        print("SUCCESS: Refactored BMA model imported successfully")
        
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
            print(f"  - Columns: {list(feature_data.columns)[:10]}...")
            print(f"  - Index type: {type(feature_data.index)}")
            if hasattr(feature_data.index, 'names'):
                print(f"  - Index names: {feature_data.index.names}")
            return True, feature_data
        else:
            print("FAIL: No feature data returned from refactored BMA")
            return False, None
            
    except ImportError as ie:
        print(f"FAIL: Cannot import refactored BMA: {ie}")
        return False, None
    except Exception as e:
        print(f"FAIL: Refactored BMA test failed: {e}")
        traceback.print_exc()
        return False, None

def test_fixed_bma_data():
    """Test data getting in fixed BMA file"""
    print("\n" + "=" * 60)
    print("TESTING FIXED BMA")
    print("=" * 60)
    
    try:
        from bma_models.bma_model_fixed import BMAModelFixed, create_sample_data
        
        print("SUCCESS: Fixed BMA model imported successfully")
        
        # Initialize model
        model = BMAModelFixed()
        print("SUCCESS: Model initialized")
        
        # Note: Fixed version doesn't have real data getting - only sample data
        print("\nNOTE: Fixed BMA only has sample data generation...")
        sample_data = create_sample_data(n_dates=100, n_tickers=10, n_features=5)
        
        if not sample_data.empty:
            print("SUCCESS: Sample data generated successfully!")
            print(f"  - Shape: {sample_data.shape}")
            print(f"  - Columns: {list(sample_data.columns)}")
            print(f"  - Index type: {type(sample_data.index)}")
            if hasattr(sample_data.index, 'names'):
                print(f"  - Index names: {sample_data.index.names}")
            
            # Test if the model can work with this data
            print("\nTesting model training with sample data...")
            results = model.train(sample_data.iloc[:500])  # Use first 500 rows
            if results['success']:
                print("SUCCESS: Model training completed successfully!")
                print(f"  - Models trained: {len(results['models'])}")
                print(f"  - Training time: {results['summary']['training_time']:.2f}s")
                return True, sample_data
            else:
                print("FAIL: Model training failed")
                return False, None
        else:
            print("FAIL: No sample data generated")
            return False, None
            
    except ImportError as ie:
        print(f"FAIL: Cannot import fixed BMA: {ie}")
        return False, None
    except Exception as e:
        print(f"FAIL: Fixed BMA test failed: {e}")
        traceback.print_exc()
        return False, None

def main():
    """Main test execution"""
    print("TESTING REFACTORED VERSIONS DATA ACQUISITION")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Refactored BMA
    refactored_success, refactored_data = test_refactored_bma_data()
    results['refactored_bma'] = refactored_success
    
    # Test 2: Fixed BMA
    fixed_success, fixed_data = test_fixed_bma_data()
    results['fixed_bma'] = fixed_success
    
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
        print("All refactored version tests PASSED!")
    else:
        print("Some refactored version tests FAILED!")
        print("Check individual test outputs above for details.")

if __name__ == "__main__":
    main()