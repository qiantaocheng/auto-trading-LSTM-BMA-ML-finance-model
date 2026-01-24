#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析测试期的实际收益分布，验证为什么Top N收益异常高
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def analyze_test_period_returns():
    """分析测试期的实际收益分布"""
    print("=" * 80)
    print("分析测试期实际收益分布")
    print("=" * 80)
    
    data_file = "data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet"
    
    print(f"\n[1] 加载数据文件...")
    df = pd.read_parquet(data_file)
    print(f"   [OK] 数据形状: {df.shape}")
    
    # 提取测试期数据
    test_start = pd.Timestamp('2025-01-02')
    test_end = pd.Timestamp('2025-12-30')
    
    test_dates = df.index.get_level_values('date')
    test_mask = (test_dates >= test_start) & (test_dates <= test_end)
    test_df = df[test_mask].copy()
    
    print(f"\n[2] 测试期数据统计:")
    print(f"   日期范围: {test_start.date()} 至 {test_end.date()}")
    print(f"   样本数: {len(test_df):,}")
    print(f"   唯一日期: {test_df.index.get_level_values('date').nunique()}")
    print(f"   唯一ticker: {test_df.index.get_level_values('ticker').nunique()}")
    
    if 'target' not in test_df.columns:
        print(f"\n[ERROR] 测试数据中没有target列")
        return
    
    target = test_df['target']
    target_valid = target.dropna()
    
    print(f"\n[3] 测试期Target统计:")
    print("-" * 80)
    print(f"   有效样本: {len(target_valid):,}")
    print(f"   均值: {target_valid.mean():.6f} ({target_valid.mean()*100:.2f}%)")
    print(f"   中位数: {target_valid.median():.6f} ({target_valid.median()*100:.2f}%)")
    print(f"   标准差: {target_valid.std():.6f} ({target_valid.std()*100:.2f}%)")
    
    # 按日期分组，计算每天的Top N收益
    print(f"\n[4] 按日期分析Top N收益分布:")
    print("-" * 80)
    
    dates = test_df.index.get_level_values('date').unique()
    dates_sorted = sorted(dates)
    
    # 选择几个代表性日期进行分析
    sample_dates = dates_sorted[::len(dates_sorted)//10][:10]  # 选择10个日期
    
    for date in sample_dates:
        date_data = test_df.loc[test_df.index.get_level_values('date') == date]
        date_target = date_data['target'].dropna()
        
        if len(date_target) < 20:
            continue
        
        # 计算Top 5, 10, 15, 20的平均收益
        sorted_target = date_target.sort_values(ascending=False)
        
        top5_mean = sorted_target.head(5).mean()
        top10_mean = sorted_target.head(10).mean()
        top15_mean = sorted_target.head(15).mean()
        top20_mean = sorted_target.head(20).mean()
        all_mean = date_target.mean()
        
        print(f"\n  日期: {date.date()}")
        print(f"    总股票数: {len(date_target)}")
        print(f"    所有股票平均收益: {all_mean:.4f} ({all_mean*100:.2f}%)")
        print(f"    Top 5平均收益: {top5_mean:.4f} ({top5_mean*100:.2f}%)")
        print(f"    Top 10平均收益: {top10_mean:.4f} ({top10_mean*100:.2f}%)")
        print(f"    Top 15平均收益: {top15_mean:.4f} ({top15_mean*100:.2f}%)")
        print(f"    Top 20平均收益: {top20_mean:.4f} ({top20_mean*100:.2f}%)")
    
    # 计算所有日期的Top N平均收益
    print(f"\n[5] 所有测试日期的Top N平均收益统计:")
    print("-" * 80)
    
    top5_means = []
    top10_means = []
    top15_means = []
    top20_means = []
    all_means = []
    
    for date in dates_sorted:
        date_data = test_df.loc[test_df.index.get_level_values('date') == date]
        date_target = date_data['target'].dropna()
        
        if len(date_target) < 20:
            continue
        
        sorted_target = date_target.sort_values(ascending=False)
        
        top5_means.append(sorted_target.head(5).mean())
        top10_means.append(sorted_target.head(10).mean())
        top15_means.append(sorted_target.head(15).mean())
        top20_means.append(sorted_target.head(20).mean())
        all_means.append(date_target.mean())
    
    if top20_means:
        print(f"\n  所有股票平均收益:")
        print(f"    均值: {np.mean(all_means):.4f} ({np.mean(all_means)*100:.2f}%)")
        print(f"    中位数: {np.median(all_means):.4f} ({np.median(all_means)*100:.2f}%)")
        
        print(f"\n  Top 5平均收益:")
        print(f"    均值: {np.mean(top5_means):.4f} ({np.mean(top5_means)*100:.2f}%)")
        print(f"    中位数: {np.median(top5_means):.4f} ({np.median(top5_means)*100:.2f}%)")
        
        print(f"\n  Top 10平均收益:")
        print(f"    均值: {np.mean(top10_means):.4f} ({np.mean(top10_means)*100:.2f}%)")
        print(f"    中位数: {np.median(top10_means):.4f} ({np.median(top10_means)*100:.2f}%)")
        
        print(f"\n  Top 15平均收益:")
        print(f"    均值: {np.mean(top15_means):.4f} ({np.mean(top15_means)*100:.2f}%)")
        print(f"    中位数: {np.median(top15_means):.4f} ({np.median(top15_means)*100:.2f}%)")
        
        print(f"\n  Top 20平均收益:")
        print(f"    均值: {np.mean(top20_means):.4f} ({np.mean(top20_means)*100:.2f}%)")
        print(f"    中位数: {np.median(top20_means):.4f} ({np.median(top20_means)*100:.2f}%)")
    
    # 检查是否有异常高的收益值
    print(f"\n[6] 异常高收益检查:")
    print("-" * 80)
    
    high_returns = target_valid[target_valid > 0.3]  # > 30%
    print(f"  > 30%的收益: {len(high_returns):,} ({len(high_returns)/len(target_valid)*100:.2f}%)")
    
    if len(high_returns) > 0:
        print(f"  这些高收益的统计:")
        print(f"    均值: {high_returns.mean():.4f} ({high_returns.mean()*100:.2f}%)")
        print(f"    中位数: {high_returns.median():.4f} ({high_returns.median()*100:.2f}%)")
        print(f"    最大值: {high_returns.max():.4f} ({high_returns.max()*100:.2f}%)")
        
        # 检查这些高收益是否集中在某些日期
        high_returns_dates = test_df.loc[test_df['target'] > 0.3].index.get_level_values('date')
        date_counts = pd.Series(high_returns_dates).value_counts().sort_values(ascending=False)
        
        print(f"\n  高收益集中的Top 10日期:")
        for date, count in date_counts.head(10).items():
            date_data = test_df.loc[test_df.index.get_level_values('date') == date]
            date_mean = date_data['target'].mean()
            print(f"    {date.date()}: {count}个股票 > 30%, 当日平均收益={date_mean:.4f} ({date_mean*100:.2f}%)")
    
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    
    if top20_means and np.mean(top20_means) > 0.3:
        print(f"\n[WARN] Top 20平均收益 {np.mean(top20_means)*100:.2f}% 异常高！")
        print(f"  可能的原因:")
        print(f"    1. 测试数据是未来数据（2025年），可能是模拟/预测数据")
        print(f"    2. 数据文件中的target值计算有误")
        print(f"    3. 模型预测能力确实很强（IC=0.93），能够选出表现最好的股票")
        print(f"    4. 测试期市场表现异常好（但这种情况很少见）")
    else:
        print(f"\n[OK] Top 20平均收益 {np.mean(top20_means)*100:.2f}% 在合理范围内")

if __name__ == "__main__":
    analyze_test_period_returns()
