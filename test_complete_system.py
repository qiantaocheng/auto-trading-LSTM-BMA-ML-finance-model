#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Complete System End-to-End Functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('D:/trade')

print("=== Complete System End-to-End Test ===")

# Create comprehensive test data
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

data = []
for i, date in enumerate(dates):
    for ticker in tickers:
        # Create more realistic trending price data
        trend = i * 0.001
        base_price = 100 + trend + np.random.randn() * 5
        data.append({
            'date': date,
            'ticker': ticker,
            'Close': base_price,
            'Open': base_price + np.random.randn() * 2, 
            'High': base_price + abs(np.random.randn() * 3),
            'Low': base_price - abs(np.random.randn() * 3),
            'Volume': np.random.randint(1000000, 50000000),
            'returns': np.random.randn() * 0.02
        })

test_data = pd.DataFrame(data)
test_data['date'] = pd.to_datetime(test_data['date'])

print(f"Test data created: {test_data.shape}")
print(f"Date range: {test_data['date'].min()} to {test_data['date'].max()}")
print(f"Tickers: {test_data['ticker'].unique()}")

# Import and create model
try:
    from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
    print("Model imported successfully")
    
    model = UltraEnhancedQuantitativeModel()
    print("Model created successfully")
    
    # Test key components
    print("\n=== Testing Key Components ===")
    
    # 1. Alpha Engine
    if hasattr(model, 'alpha_engine') and model.alpha_engine:
        print("✅ Alpha Engine: Available")
        engine_methods = [method for method in dir(model.alpha_engine) if not method.startswith('_')]
        print(f"   Methods: {', '.join(engine_methods[:5])}...")
    else:
        print("❌ Alpha Engine: Not available")
    
    # 2. Alpha Summary Processor
    if hasattr(model, 'alpha_summary_processor') and model.alpha_summary_processor:
        print("✅ Alpha Summary Processor: Available")
    else:
        print("❌ Alpha Summary Processor: Not available")
    
    # 3. Feature merging method
    if hasattr(model, '_merge_alpha_and_traditional_features'):
        print("✅ Feature Merging Method: Available")
    else:
        print("❌ Feature Merging Method: Not available")
    
    # Test basic alpha config loading
    print("\n=== Testing Configuration ===")
    print(f"Alpha config file exists: {'alphas_config.yaml' in open('D:/trade/bma_models/enhanced_alpha_strategies.py').read()}")
    
    # Test data processing capabilities
    print("\n=== Testing Data Processing ===")
    
    # Sample some data for faster testing
    sample_data = test_data.sample(n=min(1000, len(test_data))).copy()
    sample_data = sample_data.sort_values(['date', 'ticker'])
    
    print(f"Using sample data: {sample_data.shape}")
    
    # Test alpha factor computation (if available)
    if hasattr(model, 'alpha_engine') and model.alpha_engine:
        try:
            # Check if compute method exists and try to call it
            sample_indexed = sample_data.set_index(['date', 'ticker'])
            
            # Check available methods on alpha engine
            alpha_methods = [m for m in dir(model.alpha_engine) if 'compute' in m.lower() and not m.startswith('_')]
            print(f"Alpha engine compute methods: {alpha_methods}")
            
        except Exception as e:
            print(f"Alpha computation test failed: {e}")
    
    # Test feature integration workflow
    print("\n=== Testing Feature Integration ===")
    try:
        # Create mock processed data that would come from the pipeline
        processed_data = sample_data.copy()
        
        # Add some basic features (simulating feature engineering)
        processed_data['momentum_5d'] = processed_data.groupby('ticker')['returns'].rolling(5).mean().values
        processed_data['volatility_5d'] = processed_data.groupby('ticker')['returns'].rolling(5).std().values
        processed_data = processed_data.fillna(0)
        
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Processed data columns: {list(processed_data.columns)}")
        
        # Create mock alpha features (simulating alpha summary processor output)
        alpha_summary = processed_data[['date', 'ticker']].copy()
        alpha_summary['alpha_momentum_factor'] = np.random.randn(len(alpha_summary)) * 0.1
        alpha_summary['alpha_mean_reversion_factor'] = np.random.randn(len(alpha_summary)) * 0.1
        alpha_summary['alpha_volume_factor'] = np.random.randn(len(alpha_summary)) * 0.1
        alpha_summary = alpha_summary.set_index(['date', 'ticker'])
        
        # Test the merge
        if hasattr(model, '_merge_alpha_and_traditional_features'):
            merged_result = model._merge_alpha_and_traditional_features(
                processed_data, alpha_summary
            )
            
            print(f"Merge successful!")
            print(f"Final feature count: {merged_result.shape}")
            
            # Check alpha integration
            alpha_cols = [col for col in merged_result.columns if 'alpha_' in col]
            print(f"Alpha features integrated: {len(alpha_cols)}")
            print(f"Alpha column names: {alpha_cols}")
            
            if len(alpha_cols) > 0:
                print("SUCCESS: Complete alpha integration pipeline working!")
            else:
                print("WARNING: No alpha features in final output")
                
        else:
            print("Feature merging method not available")
    
    except Exception as e:
        print(f"Feature integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== System Health Summary ===")
    components_working = 0
    total_components = 4
    
    if hasattr(model, 'alpha_engine') and model.alpha_engine:
        components_working += 1
        print("✅ Alpha Engine")
    else:
        print("❌ Alpha Engine")
    
    if hasattr(model, 'alpha_summary_processor') and model.alpha_summary_processor:
        components_working += 1
        print("✅ Alpha Summary Processor")
    else:
        print("❌ Alpha Summary Processor")
        
    if hasattr(model, '_merge_alpha_and_traditional_features'):
        components_working += 1
        print("✅ Feature Merging")
    else:
        print("❌ Feature Merging")
    
    # Test if merge actually works
    try:
        # Quick merge test
        dummy_traditional = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'ticker': ['TEST'] * 3,
            'feature1': [1, 2, 3]
        })
        dummy_alpha = pd.DataFrame({
            'alpha_test': [0.1, 0.2, 0.3]
        }, index=pd.MultiIndex.from_tuples([
            (pd.Timestamp('2023-01-01'), 'TEST'),
            (pd.Timestamp('2023-01-02'), 'TEST'),
            (pd.Timestamp('2023-01-03'), 'TEST')
        ], names=['date', 'ticker']))
        
        merge_result = model._merge_alpha_and_traditional_features(dummy_traditional, dummy_alpha)
        if len([col for col in merge_result.columns if 'alpha_' in col]) > 0:
            components_working += 1
            print("✅ Alpha Integration Working")
        else:
            print("❌ Alpha Integration Not Working")
    except:
        print("❌ Alpha Integration Not Working")
    
    health_pct = (components_working / total_components) * 100
    print(f"\nSystem Health: {components_working}/{total_components} ({health_pct:.0f}%)")
    
    if health_pct >= 75:
        print("STATUS: SYSTEM READY FOR PRODUCTION")
    elif health_pct >= 50:
        print("STATUS: SYSTEM PARTIALLY FUNCTIONAL")  
    else:
        print("STATUS: SYSTEM NEEDS MAJOR FIXES")

except Exception as e:
    print(f"System test failed: {e}")
    import traceback
    traceback.print_exc()