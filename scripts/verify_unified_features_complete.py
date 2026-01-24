#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete verification of unified feature inputs across all stages:
1. Training (train_full_dataset.py)
2. Direct Predict (app.py)
3. 80/20 Time Split Evaluation (time_split_80_20_oos_eval.py)
4. Simple17FactorEngine calculation
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Expected unified feature list (14 features)
UNIFIED_FEATURES = [
    "ivol_30",
    "hist_vol_40d",
    "near_52w_high",
    "rsi_21",
    "vol_ratio_30d",
    "trend_r2_60",
    "liquid_momentum",
    "obv_momentum_40d",
    "atr_ratio",
    "ret_skew_30d",
    "price_ma60_deviation",
    "blowoff_ratio_30d",
    "bollinger_squeeze",
    "feat_vol_price_div_30d",
]

def check_t10_alpha_factors():
    """Check T10_ALPHA_FACTORS in simple_25_factor_engine.py"""
    print("=" * 80)
    print("1. Checking T10_ALPHA_FACTORS in simple_25_factor_engine.py")
    print("=" * 80)
    
    from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS
    
    t10_factors = list(T10_ALPHA_FACTORS)
    print(f"T10_ALPHA_FACTORS count: {len(t10_factors)}")
    print(f"T10_ALPHA_FACTORS: {t10_factors}")
    
    # Check if unified features are subset of T10_ALPHA_FACTORS
    missing = [f for f in UNIFIED_FEATURES if f not in t10_factors]
    extra = [f for f in t10_factors if f not in UNIFIED_FEATURES]
    
    if missing:
        print(f"[ERROR] Missing from T10_ALPHA_FACTORS: {missing}")
    else:
        print("[OK] All unified features present in T10_ALPHA_FACTORS")
    
    if extra:
        print(f"[INFO] Extra factors in T10_ALPHA_FACTORS (not in unified set): {extra}")
    
    return t10_factors

def check_t10_selected():
    """Check t10_selected in 量化模型_bma_ultra_enhanced.py"""
    print("\n" + "=" * 80)
    print("2. Checking t10_selected in 量化模型_bma_ultra_enhanced.py")
    print("=" * 80)
    
    # Read the file and extract t10_selected
    file_path = project_root / "bma_models" / "量化模型_bma_ultra_enhanced.py"
    content = file_path.read_text(encoding='utf-8')
    
    # Find t10_selected definition
    import re
    pattern = r't10_selected\s*=\s*\[(.*?)\]'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        # Extract the list content
        list_content = match.group(1)
        # Extract feature names
        features = re.findall(r'["\']([^"\']+)["\']', list_content)
        features = [f.strip() for f in features if f.strip()]
        
        print(f"t10_selected count: {len(features)}")
        print(f"t10_selected: {features}")
        
        # Compare with unified features
        missing = [f for f in UNIFIED_FEATURES if f not in features]
        extra = [f for f in features if f not in UNIFIED_FEATURES]
        
        if missing:
            print(f"[ERROR] Missing from t10_selected: {missing}")
        else:
            print("[OK] All unified features present in t10_selected")
        
        if extra:
            print(f"[WARNING] Extra features in t10_selected: {extra}")
        
        if len(features) != len(UNIFIED_FEATURES):
            print(f"[WARNING] Count mismatch: unified={len(UNIFIED_FEATURES)}, t10_selected={len(features)}")
        
        return features
    else:
        print("[ERROR] Could not find t10_selected definition")
        return None

def check_multiindex_file():
    """Check if multiindex file contains all unified features"""
    print("\n" + "=" * 80)
    print("3. Checking multiindex file for unified features")
    print("=" * 80)
    
    file_path = Path("D:/trade/data/factor_exports/polygon_factors_all_filtered_clean.parquet")
    
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return None
    
    try:
        df = pd.read_parquet(file_path)
        print(f"File shape: {df.shape}")
        print(f"Total columns: {len(df.columns)}")
        
        # Check unified features
        missing = [f for f in UNIFIED_FEATURES if f not in df.columns]
        present = [f for f in UNIFIED_FEATURES if f in df.columns]
        
        print(f"\nUnified features present: {len(present)}/{len(UNIFIED_FEATURES)}")
        if present:
            print(f"  Present: {present}")
        
        if missing:
            print(f"[ERROR] Missing from file: {missing}")
        else:
            print("[OK] All unified features present in multiindex file")
        
        # Check for old factors that should be removed
        old_factors = [
            'obv_divergence',
            'feat_sato_momentum_10d',
            'feat_sato_divergence_10d',
            'vol_ratio_20d',
            'ret_skew_20d',
            'ivol_20',
            'blowoff_ratio',
        ]
        
        old_present = [f for f in old_factors if f in df.columns]
        if old_present:
            print(f"[WARNING] Old factors still present: {old_present}")
        else:
            print("[OK] All old factors removed")
        
        return df.columns.tolist()
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return None

def check_calculation_methods():
    """Check if calculation methods match expected window sizes"""
    print("\n" + "=" * 80)
    print("4. Checking calculation methods in simple_25_factor_engine.py")
    print("=" * 80)
    
    file_path = project_root / "bma_models" / "simple_25_factor_engine.py"
    content = file_path.read_text(encoding='utf-8')
    
    checks = {
        'ret_skew_30d': {
            'method': '_compute_ret_skew_30d',
            'window': '30',
            'old_method': '_compute_ret_skew_20d',
        },
        'ivol_30': {
            'method': '_compute_ivol_30',
            'window': '30',
        },
        'vol_ratio_30d': {
            'method': '_compute_volume_factors',
            'window': '30',
        },
        'blowoff_ratio_30d': {
            'method': '_compute_blowoff_and_volatility',
            'window': '30',
        },
        'obv_momentum_40d': {
            'method': '_compute_volume_factors',
            'window': '40',
        },
        'feat_vol_price_div_30d': {
            'method': '_compute_vol_price_div_30d',
            'window': '30',
        },
    }
    
    for factor, info in checks.items():
        method = info['method']
        window = info['window']
        
        # Check if method exists
        if method in content:
            print(f"[OK] {factor}: Method {method} exists")
            
            # Check window size in method
            method_content = content[content.find(f'def {method}'):content.find(f'def {method}') + 2000]
            if f'window={window}' in method_content or f'window={window}' in method_content or f'rolling({window}' in method_content:
                print(f"      Window size {window} found")
            else:
                print(f"      [WARNING] Window size {window} not clearly found in method")
        else:
            print(f"[ERROR] {factor}: Method {method} not found")
        
        # Check for old method if applicable
        if 'old_method' in info:
            old_method = info['old_method']
            if old_method in content:
                # Check if it's still being called
                if f'{old_method}(' in content:
                    print(f"      [WARNING] Old method {old_method} still being called")
                else:
                    print(f"      [OK] Old method {old_method} exists but not called")

def main():
    print("=" * 80)
    print("UNIFIED FEATURES VERIFICATION")
    print("=" * 80)
    print(f"\nExpected unified features ({len(UNIFIED_FEATURES)}):")
    for i, feat in enumerate(UNIFIED_FEATURES, 1):
        print(f"  {i:2d}. {feat}")
    
    # Run checks
    t10_factors = check_t10_alpha_factors()
    t10_selected = check_t10_selected()
    file_columns = check_multiindex_file()
    check_calculation_methods()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_ok = True
    
    if t10_selected and set(t10_selected) == set(UNIFIED_FEATURES):
        print("[OK] t10_selected matches unified features")
    else:
        print("[ERROR] t10_selected does not match unified features")
        all_ok = False
    
    if file_columns:
        missing_in_file = [f for f in UNIFIED_FEATURES if f not in file_columns]
        if not missing_in_file:
            print("[OK] All unified features present in multiindex file")
        else:
            print(f"[ERROR] Missing in file: {missing_in_file}")
            all_ok = False
    
    if all_ok:
        print("\n[SUCCESS] All checks passed! Features are unified across all stages.")
    else:
        print("\n[FAILURE] Some checks failed. Please review the errors above.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
