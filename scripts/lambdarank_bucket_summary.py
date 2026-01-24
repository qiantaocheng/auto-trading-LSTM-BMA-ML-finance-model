#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LambdaRank 各分桶收益汇总"""

import pandas as pd
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_032758")

# Read bucket summary
summary = pd.read_csv(run_dir / "lambdarank_bucket_summary.csv")
returns = pd.read_csv(run_dir / "lambdarank_bucket_returns.csv")

print("="*80)
print("LambdaRank 各分桶收益表现汇总")
print("="*80)

# Extract key metrics from summary
row = summary.iloc[0]

print("\n【Top 分桶 - 平均收益（每日预测质量评估）】")
print("-"*80)
print(f"Top 1-10 (平均):  {row['avg_top_1_10_return']*100:.4f}%")
print(f"Top 1-10 (中位数): {row['median_top_1_10_return']*100:.4f}%")
print(f"Top 11-20 (平均):  {row['avg_top_11_20_return']*100:.4f}%")
print(f"Top 11-20 (中位数): {row['median_top_11_20_return']*100:.4f}%")
print(f"Top 21-30 (平均):  {row['avg_top_21_30_return']*100:.4f}%")
print(f"Top 21-30 (中位数): {row['median_top_21_30_return']*100:.4f}%")

print("\n【Bottom 分桶 - 平均收益（每日预测质量评估）】")
print("-"*80)
print(f"Bottom 1-10 (平均):  {row['avg_bottom_1_10_return']*100:.4f}%")
print(f"Bottom 1-10 (中位数): {row['median_bottom_1_10_return']*100:.4f}%")
print(f"Bottom 11-20 (平均):  {row['avg_bottom_11_20_return']*100:.4f}%")
print(f"Bottom 11-20 (中位数): {row['median_bottom_11_20_return']*100:.4f}%")
print(f"Bottom 21-30 (平均):  {row['avg_bottom_21_30_return']*100:.4f}%")
print(f"Bottom 21-30 (中位数): {row['median_bottom_21_30_return']*100:.4f}%")

# Calculate statistics from time series
print("\n【各分桶时间序列统计（基于非重叠回测）】")
print("-"*80)

buckets = {
    'Top 1-10': 'top_1_10_return',
    'Top 11-20': 'top_11_20_return',
    'Top 21-30': 'top_21_30_return',
    'Bottom 1-10': 'bottom_1_10_return',
    'Bottom 11-20': 'bottom_11_20_return',
    'Bottom 21-30': 'bottom_21_30_return',
}

for name, col in buckets.items():
    if col in returns.columns:
        data = returns[col].dropna() / 100.0  # Convert from percentage to decimal
        if len(data) > 0:
            print(f"\n{name}:")
            print(f"  平均每期收益: {data.mean()*100:.4f}%")
            print(f"  中位数收益:   {data.median()*100:.4f}%")
            print(f"  标准差:       {data.std()*100:.4f}%")
            print(f"  胜率:         {(data > 0).mean()*100:.2f}%")
            print(f"  最大收益:     {data.max()*100:.4f}%")
            print(f"  最小收益:     {data.min()*100:.4f}%")
            
            # Calculate cumulative return
            cum_return = (1 + data).prod() - 1
            print(f"  累计收益:     {cum_return*100:.2f}%")

print("\n" + "="*80)
print("说明:")
print("- 平均收益/中位数收益：基于每日预测质量评估（每日计算）")
print("- 时间序列统计：基于非重叠回测（每10天一期，共25期）")
print("="*80)
