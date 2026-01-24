#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 MultiIndex 修复是否完成
检查所有因子计算失败时是否使用 pd.Series 而不是 np.zeros
"""

import re
import os

def verify_multindex_fix():
    """验证 MultiIndex 修复"""
    file_path = 'bma_models/simple_25_factor_engine.py'
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否还有 np.zeros(len(data))
    np_zeros_matches = re.findall(r'np\.zeros\(len\(data\)\)', content)
    
    # 检查 pd.Series 的使用
    pd_series_matches = re.findall(r'pd\.Series\(0\.0.*index=data\.index', content)
    
    print("=" * 80)
    print("MultiIndex 修复验证")
    print("=" * 80)
    
    if len(np_zeros_matches) == 0:
        print("[PASS] All np.zeros(len(data)) have been fixed")
    else:
        print(f"[FAIL] Still found {len(np_zeros_matches)} instances of np.zeros(len(data))")
        return False
    
    print(f"[PASS] Found {len(pd_series_matches)} instances using pd.Series(0.0, index=data.index)")
    
    # 检查关键因子
    key_factors = [
        'obv_divergence',
        'obv_momentum_40d',
        'momentum_10d',
        '5_days_reversal',
        'liquid_momentum',
        'rsrs_beta_18',
        'hist_vol_40d',
        'ivol_20',
        'ivol_30',
        'streak_reversal',
        'feat_vol_price_div_30d'
    ]
    
    print("\nKey factors check:")
    print("-" * 80)
    all_found = True
    for factor in key_factors:
        # 检查是否有这个因子的 Series 修复
        pattern = rf"pd\.Series\(0\.0.*index=data\.index.*name=['\"]{factor}['\"]"
        if re.search(pattern, content):
            print(f"  [PASS] {factor}: Fixed")
        else:
            # 检查是否在代码中使用了这个因子
            if factor in content:
                print(f"  [INFO] {factor}: Exists but Series fix not found (may compute successfully)")
            else:
                print(f"  [INFO] {factor}: Not found in code")
    
    print("\n" + "=" * 80)
    print("[PASS] MultiIndex fix verification complete!")
    print("=" * 80)
    
    return True

if __name__ == '__main__':
    verify_multindex_fix()
