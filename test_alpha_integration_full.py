#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Full Alpha Integration Pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('D:/trade')

# Create sample data with alpha features
dates = pd.date_range(start='2023-01-01', end='2023-01-30', freq='D')
tickers = ['AAPL', 'MSFT', 'GOOGL']

# Create traditional features
traditional_data = []
for date in dates:
    for ticker in tickers:
        traditional_data.append({
            'date': date,
            'ticker': ticker,
            'Close': np.random.randn() + 100,
            'Open': np.random.randn() + 100, 
            'High': np.random.randn() + 102,
            'Low': np.random.randn() + 98,
            'Volume': np.random.randint(1000000, 10000000),
            'returns': np.random.randn() * 0.02,
            'feature_1': np.random.randn(),
            'feature_2': np.random.randn()
        })

traditional_df = pd.DataFrame(traditional_data)
traditional_df['date'] = pd.to_datetime(traditional_df['date'])

# Create mock alpha summary features with proper alpha_ prefixed columns
alpha_data = []
for date in dates:
    for ticker in tickers:
        alpha_data.append({
            'date': date,
            'ticker': ticker,
            'alpha_momentum_1d': np.random.randn() * 0.1,
            'alpha_momentum_5d': np.random.randn() * 0.1,
            'alpha_mean_reversion_5d': np.random.randn() * 0.1,
            'alpha_volatility_5d': np.random.randn() * 0.1,
            'alpha_volume_ratio': np.random.randn() * 0.1,
            'alpha_pca_component_1': np.random.randn() * 0.1,
            'alpha_pca_component_2': np.random.randn() * 0.1,
            'alpha_summary_feature_1': np.random.randn() * 0.1,
            'alpha_summary_feature_2': np.random.randn() * 0.1
        })

alpha_df = pd.DataFrame(alpha_data)
alpha_df['date'] = pd.to_datetime(alpha_df['date'])
alpha_df = alpha_df.set_index(['date', 'ticker'])

print(f"Traditional features shape: {traditional_df.shape}")
print(f"Traditional features columns: {list(traditional_df.columns)}")
print(f"Alpha features shape: {alpha_df.shape}")
print(f"Alpha features columns: {list(alpha_df.columns)}")

# Import the model
try:
    from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
    print("Model imported successfully")
    
    # Create model instance
    print("Creating model instance...")
    model = UltraEnhancedQuantitativeModel()
    print("Model created successfully")
    
    # Test the merging method with proper alpha features
    print("\n=== Testing Alpha Feature Merging with Real Alpha Data ===")
    
    if hasattr(model, '_merge_alpha_and_traditional_features'):
        print("Testing merge with real alpha features...")
        
        try:
            # Test merging
            merged_features = model._merge_alpha_and_traditional_features(
                traditional_df, alpha_df
            )
            
            print(f"\n=== MERGE RESULTS ===")
            print(f"Original traditional features: {traditional_df.shape}")
            print(f"Alpha features: {alpha_df.shape}")
            print(f"Merged features: {merged_features.shape}")
            print(f"Merged columns: {list(merged_features.columns)}")
            
            # Check for alpha features in merged result
            alpha_cols = [col for col in merged_features.columns if 'alpha_' in col]
            print(f"\n=== ALPHA INTEGRATION CHECK ===")
            print(f"Alpha columns found: {alpha_cols}")
            print(f"Total alpha features: {len(alpha_cols)}")
            
            if len(alpha_cols) > 0:
                print("✅ SUCCESS: Alpha features successfully integrated!")
                print(f"Sample alpha values: {merged_features[alpha_cols].head()}")
            else:
                print("❌ FAILED: No alpha features found in merged result")
                
            # Check for data leakage or strange columns
            duplicate_cols = [col for col in merged_features.columns if '_x' in col or '_y' in col]
            if duplicate_cols:
                print(f"⚠️ Warning: Found duplicate columns: {duplicate_cols}")
                print("This suggests a merging issue")
            else:
                print("✅ No duplicate columns found")
                
        except Exception as e:
            print(f"Error in merging: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("_merge_alpha_and_traditional_features method not found")
    
except Exception as e:
    print(f"Error importing or creating model: {e}")
    import traceback
    traceback.print_exc()