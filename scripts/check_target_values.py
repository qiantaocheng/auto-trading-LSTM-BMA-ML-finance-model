#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据文件中的target列统计信息，分析为什么收益异常高
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def check_target_values():
    """检查target列的统计信息"""
    print("=" * 80)
    print("检查target列统计信息")
    print("=" * 80)
    
    data_file = "data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet"
    
    print(f"\n[1] 加载数据文件: {data_file}")
    df = pd.read_parquet(data_file)
    print(f"   [OK] 数据形状: {df.shape}")
    
    # 检查target列
    if 'target' not in df.columns:
        print(f"\n[ERROR] 数据文件中没有target列")
        print(f"   可用列: {list(df.columns)}")
        return
    
    target = df['target']
    
    print(f"\n[2] Target列统计信息:")
    print("-" * 80)
    
    # 基本统计
    print(f"总样本数: {len(target):,}")
    print(f"非空值: {target.notna().sum():,} ({target.notna().sum()/len(target)*100:.1f}%)")
    print(f"空值: {target.isna().sum():,} ({target.isna().sum()/len(target)*100:.1f}%)")
    
    # 只分析非空值
    target_valid = target.dropna()
    if len(target_valid) == 0:
        print("\n[ERROR] 没有有效的target值")
        return
    
    print(f"\n有效target值统计 ({len(target_valid):,} 个):")
    print(f"  均值: {target_valid.mean():.6f} ({target_valid.mean()*100:.2f}%)")
    print(f"  中位数: {target_valid.median():.6f} ({target_valid.median()*100:.2f}%)")
    print(f"  标准差: {target_valid.std():.6f} ({target_valid.std()*100:.2f}%)")
    print(f"  最小值: {target_valid.min():.6f} ({target_valid.min()*100:.2f}%)")
    print(f"  最大值: {target_valid.max():.6f} ({target_valid.max()*100:.2f}%)")
    
    # 分位数
    print(f"\n分位数:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        q = target_valid.quantile(p/100)
        print(f"  {p:2d}%: {q:.6f} ({q*100:.2f}%)")
    
    # 收益分布
    print(f"\n收益分布:")
    print(f"  > 10% (0.1): {(target_valid > 0.1).sum():,} ({(target_valid > 0.1).sum()/len(target_valid)*100:.1f}%)")
    print(f"  > 20% (0.2): {(target_valid > 0.2).sum():,} ({(target_valid > 0.2).sum()/len(target_valid)*100:.1f}%)")
    print(f"  > 30% (0.3): {(target_valid > 0.3).sum():,} ({(target_valid > 0.3).sum()/len(target_valid)*100:.1f}%)")
    print(f"  > 50% (0.5): {(target_valid > 0.5).sum():,} ({(target_valid > 0.5).sum()/len(target_valid)*100:.1f}%)")
    print(f"  > 100% (1.0): {(target_valid > 1.0).sum():,} ({(target_valid > 1.0).sum()/len(target_valid)*100:.1f}%)")
    
    print(f"\n  < -10% (-0.1): {(target_valid < -0.1).sum():,} ({(target_valid < -0.1).sum()/len(target_valid)*100:.1f}%)")
    print(f"  < -20% (-0.2): {(target_valid < -0.2).sum():,} ({(target_valid < -0.2).sum()/len(target_valid)*100:.1f}%)")
    print(f"  < -30% (-0.3): {(target_valid < -0.3).sum():,} ({(target_valid < -0.3).sum()/len(target_valid)*100:.1f}%)")
    
    # 按日期分组统计（检查是否有特定日期异常）
    print(f"\n[3] 按日期分组统计:")
    print("-" * 80)
    
    if isinstance(df.index, pd.MultiIndex) and 'date' in df.index.names:
        dates = df.index.get_level_values('date')
        target_with_dates = pd.DataFrame({'date': dates, 'target': target.values})
        target_with_dates = target_with_dates[target_with_dates['target'].notna()]
        
        daily_stats = target_with_dates.groupby('date')['target'].agg(['mean', 'median', 'std', 'count'])
        daily_stats = daily_stats.sort_values('mean', ascending=False)
        
        print(f"\nTop 10 最高平均收益日期:")
        for i, (date, row) in enumerate(daily_stats.head(10).iterrows(), 1):
            print(f"  {i:2d}. {date.date()}: 均值={row['mean']:.4f} ({row['mean']*100:.2f}%), "
                  f"中位数={row['median']:.4f}, 样本数={int(row['count'])}")
        
        print(f"\nBottom 10 最低平均收益日期:")
        for i, (date, row) in enumerate(daily_stats.tail(10).iterrows(), 1):
            print(f"  {i:2d}. {date.date()}: 均值={row['mean']:.4f} ({row['mean']*100:.2f}%), "
                  f"中位数={row['median']:.4f}, 样本数={int(row['count'])}")
        
        # 检查测试期的target值
        test_start = pd.Timestamp('2025-01-02')
        test_end = pd.Timestamp('2025-12-30')
        
        test_mask = (dates >= test_start) & (dates <= test_end)
        test_target = target[test_mask]
        test_target_valid = test_target.dropna()
        
        if len(test_target_valid) > 0:
            print(f"\n[4] 测试期 (2025-01-02 至 2025-12-30) Target统计:")
            print("-" * 80)
            print(f"  有效样本数: {len(test_target_valid):,}")
            print(f"  均值: {test_target_valid.mean():.6f} ({test_target_valid.mean()*100:.2f}%)")
            print(f"  中位数: {test_target_valid.median():.6f} ({test_target_valid.median()*100:.2f}%)")
            print(f"  标准差: {test_target_valid.std():.6f} ({test_target_valid.std()*100:.2f}%)")
            print(f"  最小值: {test_target_valid.min():.6f} ({test_target_valid.min()*100:.2f}%)")
            print(f"  最大值: {test_target_valid.max():.6f} ({test_target_valid.max()*100:.2f}%)")
            print(f"  > 20%: {(test_target_valid > 0.2).sum():,} ({(test_target_valid > 0.2).sum()/len(test_target_valid)*100:.1f}%)")
            print(f"  > 30%: {(test_target_valid > 0.3).sum():,} ({(test_target_valid > 0.3).sum()/len(test_target_valid)*100:.1f}%)")
    
    # 检查是否有异常值
    print(f"\n[5] 异常值检查:")
    print("-" * 80)
    
    # 使用IQR方法检测异常值
    Q1 = target_valid.quantile(0.25)
    Q3 = target_valid.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outliers = target_valid[(target_valid < lower_bound) | (target_valid > upper_bound)]
    print(f"  IQR异常值 (> 3*IQR): {len(outliers):,} ({len(outliers)/len(target_valid)*100:.1f}%)")
    
    if len(outliers) > 0:
        print(f"  异常值范围: {outliers.min():.4f} 至 {outliers.max():.4f}")
        print(f"  异常值均值: {outliers.mean():.4f} ({outliers.mean()*100:.2f}%)")
    
    # 检查target计算是否正确（应该是T+10收益）
    print(f"\n[6] Target计算验证:")
    print("-" * 80)
    print("  Target应该是T+10的forward return (小数形式)")
    print("  例如: 0.1 = 10%, 0.35 = 35%")
    print(f"  当前数据均值: {target_valid.mean():.4f} = {target_valid.mean()*100:.2f}%")
    
    if target_valid.mean() > 0.1:
        print(f"\n  [WARN] 平均收益 {target_valid.mean()*100:.2f}% 异常高！")
        print(f"  正常市场10天持有期平均收益应该在 -2% 到 +2% 之间")
        print(f"  可能的原因:")
        print(f"    1. Target计算有误（可能使用了累计收益而不是期收益）")
        print(f"    2. 数据是模拟/预测数据，不是真实市场数据")
        print(f"    3. 数据文件中的日期标注有误（可能是历史数据但标注为2025）")
        print(f"    4. Target计算时使用了错误的基准或计算方法")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_target_values()
