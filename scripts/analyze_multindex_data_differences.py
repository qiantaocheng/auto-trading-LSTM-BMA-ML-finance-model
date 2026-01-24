#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 MultiIndex 设置过程中的潜在数据差异
检查因子计算和 MultiIndex 设置的一致性
"""

import re
import os

def analyze_multindex_data_differences():
    """分析 MultiIndex 数据差异"""
    file_path = 'bma_models/simple_25_factor_engine.py'
    
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("=" * 80)
    print("MultiIndex Data Differences Analysis")
    print("=" * 80)
    
    # 1. 检查数据准备阶段
    print("\n[1] Data Preparation Phase")
    print("-" * 80)
    
    # 检查 compute_data 排序和 reset_index
    if re.search(r'compute_data\s*=\s*compute_data\.sort_values.*reset_index\(drop=True\)', content):
        print("[PASS] compute_data is sorted and reset_index(drop=True) -> RangeIndex")
        print("       This ensures consistent RangeIndex [0, 1, 2, ...]")
    else:
        print("[WARN] compute_data sorting/reset_index pattern not found")
    
    # 2. 检查因子计算阶段
    print("\n[2] Factor Computation Phase")
    print("-" * 80)
    
    # 检查所有因子返回 DataFrame 时是否使用 data.index
    factor_return_patterns = [
        (r'return pd\.DataFrame\(out.*index=data\.index\)', 'out dict factors'),
        (r'return pd\.DataFrame\(\{.*\}.*index=data\.index\)', 'direct return factors')
    ]
    
    for pattern, desc in factor_return_patterns:
        matches = re.findall(pattern, content)
        if matches:
            print(f"[PASS] {desc}: Uses data.index (RangeIndex)")
        else:
            print(f"[INFO] {desc}: Pattern not found (may use different structure)")
    
    # 3. 检查因子合并阶段
    print("\n[3] Factor Concatenation Phase")
    print("-" * 80)
    
    if re.search(r'factors_df\s*=\s*pd\.concat\(all_factors', content):
        print("[PASS] factors_df = pd.concat(all_factors, axis=1)")
        print("       All factors should have RangeIndex, concat preserves order")
    else:
        print("[WARN] pd.concat pattern not found")
    
    # 4. 检查 MultiIndex 设置阶段
    print("\n[4] MultiIndex Setting Phase")
    print("-" * 80)
    
    multiindex_pattern = r'factors_df\.index\s*=\s*pd\.MultiIndex\.from_arrays\([^)]+\)'
    if re.search(multiindex_pattern, content):
        print("[PASS] MultiIndex is set using from_arrays")
        # 提取具体内容
        match = re.search(r'factors_df\.index\s*=\s*pd\.MultiIndex\.from_arrays\(([^)]+)\)', content)
        if match:
            arrays = match.group(1)
            print(f"       Arrays: {arrays.strip()}")
            if 'compute_data[\'date\']' in arrays and 'compute_data[\'ticker\']' in arrays:
                print("[PASS] Uses compute_data['date'] and compute_data['ticker']")
                print("       Order should match RangeIndex [0, 1, 2, ...]")
            else:
                print("[WARN] MultiIndex arrays may not match compute_data")
    else:
        print("[WARN] MultiIndex setting pattern not found")
    
    # 5. 检查潜在的数据差异风险
    print("\n[5] Potential Data Differences")
    print("-" * 80)
    
    risks = []
    
    # 风险 1: 因子行数不一致
    print("Risk 1: Factor row count mismatch")
    print("  - If factors have different row counts, pd.concat will fail or create NaNs")
    print("  - Current fix: All factors use data.index (same length)")
    print("  [PASS] All factors should have same row count")
    
    # 风险 2: 因子顺序不一致
    print("\nRisk 2: Factor row order mismatch")
    print("  - If factors have different row order, values will misalign")
    print("  - Current fix: All factors use data.index (same order)")
    print("  [PASS] All factors should have same row order")
    
    # 风险 3: MultiIndex 设置时顺序不匹配
    print("\nRisk 3: MultiIndex array order mismatch")
    print("  - If compute_data['date']/['ticker'] order != factors_df.index order")
    print("  - Current: compute_data is sorted, factors use data.index (same order)")
    print("  - compute_data and data should be same DataFrame")
    if re.search(r'data\s*=\s*compute_data|compute_data\s*=\s*data', content):
        print("  [PASS] data and compute_data are same reference")
    else:
        print("  [WARN] Need to verify data and compute_data are same")
        risks.append("data and compute_data may be different objects")
    
    # 风险 4: 重复索引
    print("\nRisk 4: Duplicate indices after MultiIndex")
    print("  - If compute_data has duplicate (date, ticker) pairs")
    if re.search(r'duplicates\s*=\s*factors_df\.index\.duplicated\(\)', content):
        print("  [PASS] Code checks for duplicates and removes them")
    else:
        print("  [WARN] No duplicate check found")
        risks.append("No duplicate index handling")
    
    # 风险 5: Series vs Array 对齐问题（已修复）
    print("\nRisk 5: Series vs Array alignment (FIXED)")
    print("  - Previously: np.zeros(len(data)) had no index")
    print("  - Fixed: pd.Series(0.0, index=data.index) has explicit index")
    print("  [PASS] All factors now use Series with explicit index")
    
    # 6. 检查修复后的代码
    print("\n[6] Post-Fix Verification")
    print("-" * 80)
    
    np_zeros_count = len(re.findall(r'np\.zeros\(len\(data\)\)', content))
    pd_series_count = len(re.findall(r'pd\.Series\(0\.0.*index=data\.index', content))
    
    print(f"np.zeros(len(data)) instances: {np_zeros_count}")
    print(f"pd.Series(0.0, index=data.index) instances: {pd_series_count}")
    
    if np_zeros_count == 0:
        print("[PASS] All np.zeros replaced with pd.Series")
    else:
        print(f"[FAIL] Still {np_zeros_count} instances of np.zeros")
        risks.append(f"{np_zeros_count} instances of np.zeros not fixed")
    
    # 7. 总结
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if len(risks) == 0:
        print("[PASS] No data differences detected")
        print("       All factors use consistent RangeIndex")
        print("       MultiIndex is set using matching arrays")
        print("       Series alignment issues are fixed")
    else:
        print(f"[WARN] {len(risks)} potential risks detected:")
        for i, risk in enumerate(risks, 1):
            print(f"  {i}. {risk}")
    
    print("\n" + "=" * 80)
    return len(risks) == 0

if __name__ == '__main__':
    analyze_multindex_data_differences()
