#!/usr/bin/env python3
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
        print("✓ Polygon client imported successfully")
        print(f"  - API Key configured: {'***' + polygon_client.api_key[-8:] if polygon_client.api_key else 'NO API KEY'}")
        print(f"  - Delayed data mode: {polygon_client.delayed_data_mode}")
        
        # Test data download
        print("\n🔄 Testing data download for AAPL...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        data = polygon_client.download(['AAPL'], start_date, end_date)
        
        if not data.empty:
            print(f"✅ Data downloaded successfully!")
            print(f"  - Shape: {data.shape}")
            print(f"  - Date range: {data.index.min()} to {data.index.max()}")
            print(f"  - Columns: {list(data.columns)}")
            print(f"  - Last close price: ${data['Close'].iloc[-1]:.2f}")
            return True, data
        else:
            print("❌ No data returned from Polygon")
            return False, None
            
    except Exception as e:
        print(f"❌ Polygon client test failed: {e}")
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
        
        print("✓ Original BMA model imported successfully")
        
        # Initialize model
        model = UltraEnhancedQuantitativeModel()
        print("✓ Model initialized")
        
        # Test data acquisition
        print("\n🔄 Testing get_data_and_features...")
        tickers = ['AAPL', 'MSFT']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        feature_data = model.get_data_and_features(tickers, start_date, end_date)
        
        if feature_data is not None and not feature_data.empty:
            print(f"✅ Feature data acquired successfully!")
            print(f"  - Shape: {feature_data.shape}")
            print(f"  - Columns: {list(feature_data.columns)[:10]}...")  # First 10 columns
            print(f"  - Index type: {type(feature_data.index)}")
            if hasattr(feature_data.index, 'names'):
                print(f"  - Index names: {feature_data.index.names}")
            return True, feature_data
        else:
            print("❌ No feature data returned from original BMA")
            return False, None
            
    except ImportError as ie:
        print(f"❌ Cannot import original BMA: {ie}")
        return False, None
    except Exception as e:
        print(f"❌ Original BMA test failed: {e}")
        traceback.print_exc()
        return False, None

def test_refactored_bma_data():
    """Test data getting in refactored BMA file"""
    print("\n" + "=" * 60)
    print("TESTING REFACTORED BMA")
    print("=" * 60)
    
    try:
        from bma_models.bma_ultra_enhanced_refactored import UltraEnhancedQuantitativeModel
        
        print("✓ Refactored BMA model imported successfully")
        
        # Initialize model
        model = UltraEnhancedQuantitativeModel()
        print("✓ Model initialized")
        
        # Test data acquisition
        print("\n🔄 Testing get_data_and_features...")
        tickers = ['AAPL', 'MSFT']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        feature_data = model.get_data_and_features(tickers, start_date, end_date)
        
        if feature_data is not None and not feature_data.empty:
            print(f"✅ Feature data acquired successfully!")
            print(f"  - Shape: {feature_data.shape}")
            print(f"  - Columns: {list(feature_data.columns)[:10]}...")
            print(f"  - Index type: {type(feature_data.index)}")
            if hasattr(feature_data.index, 'names'):
                print(f"  - Index names: {feature_data.index.names}")
            return True, feature_data
        else:
            print("❌ No feature data returned from refactored BMA")
            return False, None
            
    except ImportError as ie:
        print(f"❌ Cannot import refactored BMA: {ie}")
        return False, None
    except Exception as e:
        print(f"❌ Refactored BMA test failed: {e}")
        traceback.print_exc()
        return False, None

def test_fixed_bma_data():
    """Test data getting in fixed BMA file"""
    print("\n" + "=" * 60)
    print("TESTING FIXED BMA")
    print("=" * 60)
    
    try:
        from bma_models.bma_model_fixed import BMAModelFixed, create_sample_data
        
        print("✓ Fixed BMA model imported successfully")
        
        # Initialize model
        model = BMAModelFixed()
        print("✓ Model initialized")
        
        # Note: Fixed version doesn't have real data getting - only sample data
        print("\n⚠️  Fixed BMA only has sample data generation...")
        sample_data = create_sample_data(n_dates=100, n_tickers=10, n_features=5)
        
        if not sample_data.empty:
            print(f"✅ Sample data generated successfully!")
            print(f"  - Shape: {sample_data.shape}")
            print(f"  - Columns: {list(sample_data.columns)}")
            print(f"  - Index type: {type(sample_data.index)}")
            if hasattr(sample_data.index, 'names'):
                print(f"  - Index names: {sample_data.index.names}")
            return True, sample_data
        else:
            print("❌ No sample data generated")
            return False, None
            
    except ImportError as ie:
        print(f"❌ Cannot import fixed BMA: {ie}")
        return False, None
    except Exception as e:
        print(f"❌ Fixed BMA test failed: {e}")
        traceback.print_exc()
        return False, None

def main():
    """Main test execution"""
    print("🚀 STARTING BMA DATA ACQUISITION TESTS")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Polygon Client
    polygon_success, polygon_data = test_polygon_client()
    results['polygon_client'] = polygon_success
    
    # Test 2: Original BMA
    original_success, original_data = test_original_bma_data()
    results['original_bma'] = original_success
    
    # Test 3: Refactored BMA
    refactored_success, refactored_data = test_refactored_bma_data()
    results['refactored_bma'] = refactored_success
    
    # Test 4: Fixed BMA
    fixed_success, fixed_data = test_fixed_bma_data()
    results['fixed_bma'] = fixed_success
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for component, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{component.upper():<20}: {status}")
    
    total_pass = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_pass}/{total_tests} tests passed")
    
    if total_pass == total_tests:
        print("🎉 All data acquisition tests PASSED!")
    else:
        print("⚠️  Some data acquisition tests FAILED!")
        print("   Check individual test outputs above for details.")

if __name__ == "__main__":
    main()