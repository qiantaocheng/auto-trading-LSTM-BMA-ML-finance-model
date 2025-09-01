#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test System Status - Simple Version
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('D:/trade')

print("=== System Status Check ===")

try:
    from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
    print("Model imported successfully")
    
    model = UltraEnhancedQuantitativeModel()
    print("Model created successfully")
    
    # Test key components
    print("\n=== Component Status ===")
    components_working = 0
    
    # 1. Alpha Engine
    if hasattr(model, 'alpha_engine') and model.alpha_engine:
        components_working += 1
        print("[OK] Alpha Engine: Available")
    else:
        print("[FAIL] Alpha Engine: Not available")
    
    # 2. Alpha Summary Processor
    if hasattr(model, 'alpha_summary_processor') and model.alpha_summary_processor:
        components_working += 1
        print("[OK] Alpha Summary Processor: Available")
    else:
        print("[FAIL] Alpha Summary Processor: Not available")
    
    # 3. Feature merging method
    if hasattr(model, '_merge_alpha_and_traditional_features'):
        components_working += 1
        print("[OK] Feature Merging Method: Available")
    else:
        print("[FAIL] Feature Merging Method: Not available")
    
    # 4. Test actual merge functionality
    print("\n=== Testing Feature Merge ===")
    try:
        # Create test data
        traditional = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'ticker': ['TEST'] * 3,
            'feature1': [1, 2, 3],
            'Close': [100, 101, 102]
        })
        
        alpha = pd.DataFrame({
            'alpha_test': [0.1, 0.2, 0.3]
        }, index=pd.MultiIndex.from_tuples([
            (pd.Timestamp('2023-01-01'), 'TEST'),
            (pd.Timestamp('2023-01-02'), 'TEST'),
            (pd.Timestamp('2023-01-03'), 'TEST')
        ], names=['date', 'ticker']))
        
        result = model._merge_alpha_and_traditional_features(traditional, alpha)
        alpha_cols = [col for col in result.columns if 'alpha_' in col]
        
        if len(alpha_cols) > 0:
            components_working += 1
            print("[OK] Alpha Integration: WORKING")
            print(f"     Alpha features found: {alpha_cols}")
            print(f"     Result shape: {result.shape}")
        else:
            print("[FAIL] Alpha Integration: No alpha features in output")
            
    except Exception as e:
        print(f"[FAIL] Alpha Integration: Error - {e}")
    
    print("\n=== Final Status ===")
    total_components = 4
    health_pct = (components_working / total_components) * 100
    print(f"Components Working: {components_working}/{total_components} ({health_pct:.0f}%)")
    
    if health_pct >= 75:
        print("STATUS: READY FOR PRODUCTION")
    elif health_pct >= 50:
        print("STATUS: PARTIALLY FUNCTIONAL")  
    else:
        print("STATUS: NEEDS FIXES")

except Exception as e:
    print(f"System test failed: {e}")
    import traceback
    traceback.print_exc()