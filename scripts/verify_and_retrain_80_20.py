#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证并重新训练80/20分割，确保使用最新的15个因子
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS
from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

def verify_and_retrain():
    """验证配置并重新训练"""
    print("=" * 80)
    print("验证80/20分割和15个因子配置")
    print("=" * 80)
    
    # 1. 验证因子配置
    print("\n[1] 验证因子配置...")
    model = UltraEnhancedQuantitativeModel(preserve_state=False)
    model.horizon = 10
    
    t10_selected = model._base_feature_overrides.get('elastic_net', [])
    
    print(f"T10_ALPHA_FACTORS: {len(T10_ALPHA_FACTORS)} 个因子")
    print(f"  {T10_ALPHA_FACTORS}")
    
    print(f"\nt10_selected: {len(t10_selected)} 个因子")
    print(f"  {t10_selected}")
    
    if set(T10_ALPHA_FACTORS) == set(t10_selected):
        print("\n[OK] 因子配置正确：T10_ALPHA_FACTORS == t10_selected")
    else:
        missing = set(T10_ALPHA_FACTORS) - set(t10_selected)
        extra = set(t10_selected) - set(T10_ALPHA_FACTORS)
        if missing:
            print(f"\n[ERROR] 缺少因子: {missing}")
        if extra:
            print(f"\n[ERROR] 额外因子: {extra}")
        return False
    
    # 2. 验证数据文件
    print("\n[2] 验证数据文件...")
    data_file = "data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet"
    
    if not Path(data_file).exists():
        print(f"[ERROR] 数据文件不存在: {data_file}")
        return False
    
    df = pd.read_parquet(data_file)
    print(f"[OK] 数据文件加载成功: {df.shape}")
    
    # 检查因子列
    missing_factors = [f for f in T10_ALPHA_FACTORS if f not in df.columns]
    if missing_factors:
        print(f"[ERROR] 数据文件中缺少因子: {missing_factors}")
        return False
    
    print(f"[OK] 数据文件包含所有 {len(T10_ALPHA_FACTORS)} 个因子")
    
    # 检查日期范围
    dates = df.index.get_level_values('date').unique()
    dates_sorted = sorted(dates)
    print(f"[OK] 日期范围: {dates_sorted[0]} 至 {dates_sorted[-1]}")
    print(f"[OK] 总日期数: {len(dates_sorted)}")
    
    # 计算80/20分割
    split = 0.8
    horizon = 10
    n_dates = len(dates_sorted)
    split_idx = int(n_dates * split)
    train_end_idx = max(0, split_idx - 1 - horizon)
    train_start = dates_sorted[0]
    train_end = dates_sorted[train_end_idx]
    test_start = dates_sorted[split_idx]
    test_end = dates_sorted[-1]
    
    print(f"\n[3] 80/20分割配置:")
    print(f"  训练期: {train_start.date()} 至 {train_end.date()} ({train_end_idx + 1} 个日期)")
    print(f"  测试期: {test_start.date()} 至 {test_end.date()} ({n_dates - split_idx} 个日期)")
    print(f"  分割点: {split_idx}/{n_dates} = {split_idx/n_dates:.2%}")
    print(f"  隔离间隔: {horizon} 天（避免数据泄漏）")
    
    # 4. 生成训练命令
    print(f"\n[4] 训练命令:")
    print("=" * 80)
    cmd = f'''python scripts/time_split_80_20_oos_eval.py \\
    --data-file "{data_file}" \\
    --split 0.8 \\
    --horizon-days 10 \\
    --output-dir "results/t10_time_split_80_20_final" \\
    --log-level INFO'''
    
    print(cmd)
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = verify_and_retrain()
    sys.exit(0 if success else 1)
