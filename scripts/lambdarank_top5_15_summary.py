#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LambdaRank Top 5-15 分桶收益汇总"""

import pandas as pd
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_032758")

# Read report_df to get Top 5-15 data
report_df = pd.read_csv(run_dir / "report_df.csv")
lambdarank_row = report_df[report_df['Model'] == 'lambdarank'].iloc[0]

print("="*80)
print("LambdaRank Top 5-15 分桶收益表现")
print("="*80)

print("\n【Top 5-15 分桶统计（每日预测质量评估）】")
print("-"*80)
print(f"平均收益: {lambdarank_row['avg_top_5_15_return']*100:.4f}%")
print(f"中位数收益: {lambdarank_row['median_top_5_15_return']*100:.4f}%")

# Compare with other buckets
print("\n【与其他分桶对比】")
print("-"*80)
print(f"Top 1-10:  平均 {lambdarank_row['avg_top_1_10_return']*100:.4f}%, 中位数 {lambdarank_row['median_top_1_10_return']*100:.4f}%")
print(f"Top 5-15:  平均 {lambdarank_row['avg_top_5_15_return']*100:.4f}%, 中位数 {lambdarank_row['median_top_5_15_return']*100:.4f}%")
print(f"Top 11-20: 平均 {lambdarank_row['avg_top_11_20_return']*100:.4f}%, 中位数 {lambdarank_row['median_top_11_20_return']*100:.4f}%")
print(f"Top 21-30: 平均 {lambdarank_row['avg_top_21_30_return']*100:.4f}%, 中位数 {lambdarank_row['median_top_21_30_return']*100:.4f}%")

# Check if there's time series data for Top 5-15
returns_file = run_dir / "lambdarank_bucket_returns.csv"
if returns_file.exists():
    returns = pd.read_csv(returns_file)
    if 'top_5_15_return' in returns.columns:
        data = returns['top_5_15_return'].dropna() / 100.0
        if len(data) > 0:
            print("\n【Top 5-15 时间序列统计（基于非重叠回测）】")
            print("-"*80)
            print(f"平均每期收益: {data.mean()*100:.4f}%")
            print(f"中位数收益:   {data.median()*100:.4f}%")
            print(f"标准差:       {data.std()*100:.4f}%")
            print(f"胜率:         {(data > 0).mean()*100:.2f}%")
            print(f"最大收益:     {data.max()*100:.4f}%")
            print(f"最小收益:     {data.min()*100:.4f}%")
            
            # Calculate cumulative return
            cum_return = (1 + data).prod() - 1
            print(f"累计收益:     {cum_return*100:.2f}%")
    else:
        print("\n注意: bucket_returns.csv 中没有 top_5_15_return 列")
        print("Top 5-15 数据仅在 report_df.csv 的汇总统计中可用")

print("\n" + "="*80)
