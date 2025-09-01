#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Alpha Feature Merging Debug
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('D:/trade')

# Create sample data for testing
dates = pd.date_range(start='2023-01-01', end='2023-01-30', freq='D')
tickers = ['AAPL', 'MSFT', 'GOOGL']

data = []
for date in dates:
    for ticker in tickers:
        data.append({
            'date': date,
            'ticker': ticker,
            'Close': np.random.randn() + 100,
            'Open': np.random.randn() + 100, 
            'High': np.random.randn() + 102,
            'Low': np.random.randn() + 98,
            'Volume': np.random.randint(1000000, 10000000),
            'returns': np.random.randn() * 0.02
        })

test_data = pd.DataFrame(data)
test_data['date'] = pd.to_datetime(test_data['date'])
test_data = test_data.set_index(['date', 'ticker'])

print(f"Test data shape: {test_data.shape}")
print(f"Test data index: {test_data.index}")
print(f"Test data columns: {list(test_data.columns)}")

# Import the model
try:
    from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
    print("Model imported successfully")
    
    # Create model instance
    print("Creating model instance...")
    model = UltraEnhancedQuantitativeModel()
    print("Model created successfully")
    
    # Test the specific alpha feature merging method
    print("\n=== Testing Alpha Feature Merging ===")
    
    # Check if the method exists
    if hasattr(model, '_merge_alpha_and_traditional_features'):
        print("Found _merge_alpha_and_traditional_features method")
        
        # Prepare some mock traditional features
        traditional_features = test_data.copy()
        traditional_features['feature_1'] = np.random.randn(len(traditional_features))
        traditional_features['feature_2'] = np.random.randn(len(traditional_features))
        
        print(f"Traditional features shape: {traditional_features.shape}")
        print(f"Traditional features columns: {list(traditional_features.columns)}")
        
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
    else:
        print("Method _merge_alpha_and_traditional_features not found")
        print("Available methods:")
        methods = [m for m in dir(model) if not m.startswith('_') or m.startswith('_merge')]
        for method in methods[:20]:  # Show first 20
            print(f"  {method}")
    
except Exception as e:
    print(f"Error importing or creating model: {e}")
    import traceback
    traceback.print_exc()