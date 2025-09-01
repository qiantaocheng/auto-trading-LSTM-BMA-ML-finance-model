#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Alpha Generation and Merging Debug
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('D:/trade')

# Create comprehensive sample data for testing
dates = pd.date_range(start='2020-01-01', end='2023-01-30', freq='D')
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

data = []
for date in dates:
    for ticker in tickers:
        # Create more realistic price data with trends
        base_price = 100 + np.random.randn() * 10
        data.append({
            'date': date,
            'ticker': ticker,
            'Close': base_price + np.random.randn() * 5,
            'Open': base_price + np.random.randn() * 5, 
            'High': base_price + 5 + np.random.randn() * 2,
            'Low': base_price - 5 + np.random.randn() * 2,
            'Volume': np.random.randint(1000000, 10000000),
            'returns': np.random.randn() * 0.02
        })

test_data = pd.DataFrame(data)
test_data['date'] = pd.to_datetime(test_data['date'])

print(f"Test data shape: {test_data.shape}")
print(f"Test data columns: {list(test_data.columns)}")
print(f"Date range: {test_data['date'].min()} to {test_data['date'].max()}")

# Import the model
try:
    from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
    print("Model imported successfully")
    
    # Create model instance
    print("Creating model instance...")
    model = UltraEnhancedQuantitativeModel()
    print("Model created successfully")
    
    # Test the full feature generation pipeline
    print("\n=== Testing Full Feature Generation ===")
    
    # Test alpha generation specifically
    if hasattr(model, 'alpha_engine') and model.alpha_engine:
        print("Testing alpha factor generation...")
        
        # Set data with proper MultiIndex
        test_data_indexed = test_data.set_index(['date', 'ticker'])
        print(f"Indexed data shape: {test_data_indexed.shape}")
        
        # Try to generate alpha factors
        try:
            alpha_factors = model.alpha_engine.compute_factors(test_data_indexed)
            print(f"Alpha factors shape: {alpha_factors.shape}")
            print(f"Alpha factor columns: {list(alpha_factors.columns)}")
        except Exception as e:
            print(f"Error computing alpha factors: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Alpha engine not available")
    
    # Test alpha summary processor
    if hasattr(model, 'alpha_summary_processor') and model.alpha_summary_processor:
        print("Testing alpha summary processor...")
        
        # Create some mock alpha data
        alpha_data = test_data.copy()
        alpha_data['alpha_momentum'] = np.random.randn(len(alpha_data))
        alpha_data['alpha_mean_reversion'] = np.random.randn(len(alpha_data))
        alpha_data['alpha_volatility'] = np.random.randn(len(alpha_data))
        
        try:
            summary_features = model.alpha_summary_processor.process_alpha_summary(
                alpha_data, max_features=10
            )
            print(f"Summary features shape: {summary_features.shape}")
            print(f"Summary features columns: {list(summary_features.columns)}")
            print(f"Summary features index type: {type(summary_features.index)}")
            if hasattr(summary_features.index, 'names'):
                print(f"Summary features index names: {summary_features.index.names}")
        except Exception as e:
            print(f"Error processing alpha summary: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Alpha summary processor not available")
    
    # Test the merging method specifically
    print("\n=== Testing Merge Method Directly ===")
    if hasattr(model, '_merge_alpha_and_traditional_features'):
        print("Found _merge_alpha_and_traditional_features method")
        
        # Prepare traditional features
        traditional_features = test_data.set_index(['date', 'ticker']).copy()
        traditional_features['feature_1'] = np.random.randn(len(traditional_features))
        traditional_features['feature_2'] = np.random.randn(len(traditional_features))
        
        print(f"Traditional features shape: {traditional_features.shape}")
        print(f"Traditional features index type: {type(traditional_features.index)}")
        print(f"Traditional features index names: {traditional_features.index.names}")
        
        # Test the merging
        try:
            merged_features = model._merge_alpha_and_traditional_features(
                traditional_features, test_data
            )
            print(f"Merged features shape: {merged_features.shape}")
            print(f"Merged features columns: {list(merged_features.columns)}")
            
            # Check for alpha features
            alpha_cols = [col for col in merged_features.columns if 'alpha' in col.lower()]
            print(f"Alpha columns found: {alpha_cols}")
            print(f"Total alpha features: {len(alpha_cols)}")
            
        except Exception as e:
            print(f"Error in merging: {e}")
            import traceback
            traceback.print_exc()
    
except Exception as e:
    print(f"Error importing or creating model: {e}")
    import traceback
    traceback.print_exc()