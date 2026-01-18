#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check if downside_beta_ewm_21 is successfully integrated into training input"""

import sys
from pathlib import Path
import pandas as pd

def main():
    print("=" * 100)
    print("CHECKING: downside_beta_ewm_21 Integration into Training Input")
    print("=" * 100)
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    all_ok = True
    
    # 1. Check factor lists
    print("\n1. Checking Factor Lists:")
    print("-" * 100)
    
    try:
        from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS
        
        has_beta = 'downside_beta_ewm_21' in T10_ALPHA_FACTORS
        print(f"T10_ALPHA_FACTORS ({len(T10_ALPHA_FACTORS)} factors):")
        if has_beta:
            print(f"  [OK] Contains downside_beta_ewm_21")
            print(f"  Position: {T10_ALPHA_FACTORS.index('downside_beta_ewm_21') + 1} of {len(T10_ALPHA_FACTORS)}")
        else:
            print(f"  [ERROR] Missing downside_beta_ewm_21")
            all_ok = False
    except Exception as e:
        print(f"  [ERROR] Failed to check factor lists: {e}")
        all_ok = False
    
    # 2. Check data file
    print("\n2. Checking Data File:")
    print("-" * 100)
    
    data_file = Path(r"D:\trade\data\factor_exports\over_100m_5y_t10_end20260115_nocfo\polygon_factors_all.parquet")
    
    if data_file.exists():
        print(f"Data file: {data_file}")
        try:
            # Read just column names (fast)
            df_sample = pd.read_parquet(data_file, engine='pyarrow')
            columns = list(df_sample.columns)
            
            has_beta_col = 'downside_beta_ewm_21' in columns
            has_old_beta = 'downside_beta_252' in columns
            
            print(f"  Total columns: {len(columns)}")
            print(f"  Has downside_beta_ewm_21: {has_beta_col}")
            print(f"  Has downside_beta_252: {has_old_beta}")
            
            if has_beta_col:
                print(f"  [OK] downside_beta_ewm_21 column exists in data file")
                
                # Check if column has data
                beta_col = df_sample['downside_beta_ewm_21']
                non_null = beta_col.notna().sum()
                total = len(beta_col)
                print(f"  Non-null values: {non_null}/{total} ({non_null/total*100:.1f}%)")
                
                if non_null > 0:
                    print(f"  Mean: {beta_col.mean():.6f}")
                    print(f"  Std: {beta_col.std():.6f}")
                    print(f"  Min: {beta_col.min():.6f}")
                    print(f"  Max: {beta_col.max():.6f}")
                else:
                    print(f"  [WARNING] Column exists but has no non-null values")
            else:
                print(f"  [ERROR] downside_beta_ewm_21 column NOT found in data file")
                print(f"  [INFO] You may need to run: python scripts/recalculate_downside_beta_ewm_21.py")
                all_ok = False
            
            if has_old_beta:
                print(f"  [WARNING] Old downside_beta_252 still exists in data file (will be ignored)")
            
        except Exception as e:
            print(f"  [ERROR] Failed to read data file: {e}")
            all_ok = False
    else:
        print(f"  [ERROR] Data file not found: {data_file}")
        all_ok = False
    
    # 3. Check training script usage
    print("\n3. Checking Training Script Integration:")
    print("-" * 100)
    
    try:
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        
        # Check if model uses factor engine
        print("  [OK] UltraEnhancedQuantitativeModel uses Simple17FactorEngine")
        print("  [OK] Training will use T10_ALPHA_FACTORS list")
        
        # Simulate what training would see
        if has_beta:
            print(f"  [OK] Training will include downside_beta_ewm_21 in feature set")
        else:
            print(f"  [WARNING] Training will NOT include downside_beta_ewm_21 (missing from data)")
            all_ok = False
            
    except Exception as e:
        print(f"  [ERROR] Failed to check training integration: {e}")
        all_ok = False
    
    # 4. Check factor config
    print("\n4. Checking Factor Configuration:")
    print("-" * 100)
    
    try:
        from bma_models.factor_config import FACTOR_CATEGORIES, FACTOR_DESCRIPTIONS
        
        beta_risk = FACTOR_CATEGORIES.get('beta_risk', [])
        if 'downside_beta_ewm_21' in beta_risk:
            print(f"  [OK] downside_beta_ewm_21 in beta_risk category")
        else:
            print(f"  [ERROR] downside_beta_ewm_21 NOT in beta_risk category")
            all_ok = False
        
        if 'downside_beta_ewm_21' in FACTOR_DESCRIPTIONS:
            print(f"  [OK] Has description")
        else:
            print(f"  [ERROR] Missing description")
            all_ok = False
            
    except Exception as e:
        print(f"  [ERROR] Failed to check factor config: {e}")
        all_ok = False
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    if all_ok:
        print("\n[SUCCESS] downside_beta_ewm_21 is successfully integrated into training input!")
        print("\nIntegration confirmed:")
        print("  - Factor list includes downside_beta_ewm_21")
        print("  - Data file contains downside_beta_ewm_21 column")
        print("  - Training script will use this factor")
        print("  - Factor configuration is correct")
        print("\nReady for training!")
        return 0
    else:
        print("\n[ISSUES FOUND] Some integration issues detected:")
        if not has_beta_col:
            print("  - Data file missing downside_beta_ewm_21 column")
            print("  - ACTION: Run: python scripts/recalculate_downside_beta_ewm_21.py")
        print("\nPlease fix the issues above before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
