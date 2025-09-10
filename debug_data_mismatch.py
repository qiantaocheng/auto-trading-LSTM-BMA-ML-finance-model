#!/usr/bin/env python3
"""
Debug BMA data length mismatch issue
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """Create minimal test data"""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')  
    tickers = ['AAPL', 'MSFT']
    
    data_list = []
    for ticker in tickers:
        np.random.seed(42 + hash(ticker) % 100)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        
        for i, date in enumerate(dates):
            data_list.append({
                'date': date,
                'ticker': ticker,
                'open': prices[i],
                'high': prices[i] * 1.01,
                'low': prices[i] * 0.99, 
                'close': prices[i],
                'volume': 1000000,
                'returns': np.random.normal(0.001, 0.02)
            })
    
    df = pd.DataFrame(data_list)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'ticker']).sort_index()
    
    print(f"Test data created: {df.shape}")
    return df

def debug_training():
    """Debug the training pipeline step by step"""
    print("=== BMA Training Pipeline Debug ===")
    
    # Import model
    from bma_models.bma_ultra_enhanced_refactored import UltraEnhancedQuantitativeModel
    model = UltraEnhancedQuantitativeModel()
    
    # Create test data
    test_data = create_test_data()
    
    # Prepare features and target
    feature_columns = ['open', 'high', 'low', 'close', 'volume']
    X = test_data[feature_columns].copy()
    y = test_data['returns'].copy()
    
    print(f"Original data: X={X.shape}, y={len(y)}")
    
    # Step 1: Data preprocessing
    X_processed, y_processed = model._safe_data_preprocessing(X, y)
    print(f"After preprocessing: X={X_processed.shape}, y={len(y_processed)}")
    
    # Step 2: Feature engineering
    X_optimized = model._apply_feature_lag_optimization(X_processed)
    print(f"After lag optimization: X={X_optimized.shape}")
    
    # Step 3: Feature selection  
    X_selected, selected_features = model._apply_robust_feature_selection(X_optimized, y_processed)
    print(f"After feature selection: X={X_selected.shape}, selected={len(selected_features)}")
    
    # Step 4: Data split
    validation_split = 0.2
    split_idx = int(len(X_selected) * (1 - validation_split))
    print(f"Split index: {split_idx}")
    
    X_train, X_val = X_selected.iloc[:split_idx], X_selected.iloc[split_idx:]
    y_train, y_val = y_processed.iloc[:split_idx], y_processed.iloc[split_idx:]
    
    print(f"Training split: X_train={X_train.shape}, y_train={len(y_train)}")
    print(f"Validation split: X_val={X_val.shape}, y_val={len(y_val)}")
    
    # Step 5: Try individual model training
    from sklearn.linear_model import ElasticNet
    from sklearn.model_selection import train_test_split
    
    print("\\nTesting ElasticNet training:")
    print(f"X_train index: {type(X_train.index)}, length: {len(X_train)}")
    print(f"y_train index: {type(y_train.index)}, length: {len(y_train)}")
    print(f"Index alignment: {X_train.index.equals(y_train.index)}")
    
    if not X_train.index.equals(y_train.index):
        print("WARNING: Index mismatch detected!")
        print(f"X_train index: {X_train.index[:5]}...")
        print(f"y_train index: {y_train.index[:5]}...")
        
        # Try to align indices
        common_idx = X_train.index.intersection(y_train.index)
        print(f"Common indices: {len(common_idx)}")
        X_train_aligned = X_train.loc[common_idx]
        y_train_aligned = y_train.loc[common_idx]
        print(f"Aligned: X_train={X_train_aligned.shape}, y_train={len(y_train_aligned)}")
        
        # Test model training with aligned data
        try:
            model_test = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
            model_test.fit(X_train_aligned, y_train_aligned)
            print("✓ ElasticNet training SUCCESS with aligned data")
        except Exception as e:
            print(f"✗ ElasticNet training FAILED even with aligned data: {e}")
    else:
        # Test model training with original data
        try:
            model_test = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
            model_test.fit(X_train, y_train)
            print("✓ ElasticNet training SUCCESS with original data")
        except Exception as e:
            print(f"✗ ElasticNet training FAILED: {e}")
    
    return {
        'X_original': X.shape,
        'X_processed': X_processed.shape,
        'X_selected': X_selected.shape,
        'train_split': (X_train.shape, len(y_train)),
        'val_split': (X_val.shape, len(y_val))
    }

if __name__ == "__main__":
    debug_training()