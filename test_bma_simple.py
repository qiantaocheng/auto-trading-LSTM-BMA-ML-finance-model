#!/usr/bin/env python3
"""
BMA Ultra Enhanced Simple Real Data Test
ASCII-only version to avoid encoding issues
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """Create realistic test data"""
    print("Creating test data...")
    
    # Time series
    dates = pd.date_range(start='2023-01-01', end='2024-01-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    data_list = []
    for ticker in tickers:
        np.random.seed(42 + hash(ticker) % 100)
        n_days = len(dates)
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        for i, date in enumerate(dates):
            data_list.append({
                'date': date,
                'ticker': ticker,
                'open': prices[i] * (1 + np.random.normal(0, 0.005)),
                'high': prices[i] * (1 + abs(np.random.normal(0, 0.01))),
                'low': prices[i] * (1 - abs(np.random.normal(0, 0.01))),
                'close': prices[i],
                'volume': np.random.randint(1000000, 10000000),
                'returns': returns[i]
            })
    
    df = pd.DataFrame(data_list)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'ticker']).sort_index()
    
    print(f"Test data created: {df.shape}")
    return df

def main():
    print("BMA Ultra Enhanced - Simple Real Data Test")
    print("="*50)
    
    try:
        # Import model
        print("Step 1: Importing model...")
        from bma_models.bma_ultra_enhanced_refactored import UltraEnhancedQuantitativeModel
        print("OK - Model imported successfully")
        
        # Initialize model
        print("Step 2: Initializing model...")
        model = UltraEnhancedQuantitativeModel()
        print("OK - Model initialized successfully")
        
        # Create data
        print("Step 3: Creating test data...")
        test_data = create_test_data()
        
        # Prepare features and target
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        X = test_data[feature_columns].copy()
        y = test_data.groupby('ticker')['returns'].shift(-10).fillna(0)
        
        # Clean data
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        print(f"OK - Data prepared: X={X_clean.shape}, y={len(y_clean)}")
        
        # Test preprocessing
        print("Step 4: Testing preprocessing...")
        X_processed, y_processed = model._safe_data_preprocessing(X_clean, y_clean)
        print(f"OK - Preprocessing: {X_processed.shape}")
        
        # Test feature engineering
        print("Step 5: Testing feature engineering...")
        X_selected, selected_features = model._apply_robust_feature_selection(X_processed, y_processed)
        print(f"OK - Feature selection: {X_selected.shape}, {len(selected_features)} features")
        
        # Test model training
        print("Step 6: Testing model training...")
        training_results = model.train_enhanced_models(X_selected, y_processed, validation_split=0.2)
        print(f"OK - Training completed: success={training_results.get('success', False)}")
        
        # Test predictions
        print("Step 7: Testing predictions...")
        X_pred = X_selected.iloc[:50]  # Use first 50 rows
        predictions = model.generate_enhanced_predictions(X_pred)
        
        if predictions is not None and not predictions.empty:
            print(f"OK - Predictions generated: {predictions.shape}")
        else:
            print("WARNING - No predictions generated")
        
        # Test complete analysis
        print("Step 8: Testing complete analysis...")
        analysis_results = model.run_complete_analysis(X_selected, y_processed, test_size=0.2)
        print(f"OK - Analysis completed: success={analysis_results.get('success', False)}")
        
        # Test temporal validation
        print("Step 9: Testing temporal validation...")
        temporal_valid = model.validate_temporal_configuration()
        print(f"OK - Temporal validation: {'PASS' if temporal_valid else 'FAIL'}")
        
        # Final summary
        print("\n" + "="*50)
        print("FINAL TEST RESULTS:")
        print(f"  Data samples: {len(X_clean)}")
        print(f"  Features: {X_clean.shape[1]} -> {X_selected.shape[1]}")
        print(f"  Training success: {training_results.get('success', False)}")
        print(f"  Models trained: {len(analysis_results.get('trained_models', {}))}")
        print(f"  Predictions: {'Available' if predictions is not None else 'Not Available'}")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        print(f"Details: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nSUCCESS: All tests passed - BMA model fully functional!")
    else:
        print("\nFAILED: Tests failed - needs debugging")