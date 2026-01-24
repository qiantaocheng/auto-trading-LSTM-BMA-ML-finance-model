#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从原始预测数据计算 Top 5-15 的 Overlap 胜率"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_041040")

print("="*80)
print("从原始预测数据计算 Top 5-15 的 Overlap 胜率")
print("="*80)

# We need to load predictions from the evaluation
# Since we don't have saved predictions, we'll use report_df to get summary
# Or we can recalculate from the model predictions

# For now, let's check if we can get Top 5-15 data from report_df
report_df = pd.read_csv(run_dir / "report_df.csv")

print("\n【从 report_df 获取 Top 5-15 汇总数据】")
print("-"*80)

for model_name in ['catboost', 'lambdarank']:
    model_row = report_df[report_df['Model'] == model_name]
    if not model_row.empty:
        row = model_row.iloc[0]
        avg_top_5_15 = row.get('avg_top_5_15_return', np.nan)
        median_top_5_15 = row.get('median_top_5_15_return', np.nan)
        
        print(f"\n{model_name.upper()}:")
        print(f"  Top 5-15 平均收益: {avg_top_5_15*100:.4f}%")
        print(f"  Top 5-15 中位数收益: {median_top_5_15*100:.4f}%")
        print(f"  注意: 这是每日平均收益，不是胜率")

print("\n【说明】")
print("-"*80)
print("要计算准确的 Top 5-15 Overlap 胜率，需要：")
print("1. 从原始预测数据中提取每日的 Top 5-15 收益")
print("2. 计算每日收益为正的天数比例")
print("\n由于 bucket_returns.csv 中没有保存 Top 5-15 的每日数据，")
print("我们可以使用以下方法估算：")
print("- 使用 Top 1-10 和 Top 11-20 的 Overlap 胜率作为参考")
print("- 或者重新运行评估并确保 Top 5-15 数据被保存")

# Calculate approximate Top 5-15 overlap win rate
# Top 5-15 is between Top 1-10 and Top 11-20
print("\n【Top 5-15 Overlap 胜率估算】")
print("-"*80)

cb_bucket = pd.read_csv(run_dir / "catboost_bucket_returns.csv")
lr_bucket = pd.read_csv(run_dir / "lambdarank_bucket_returns.csv")

cb_top1_10_wr = (cb_bucket['top_1_10_return'].dropna() / 100.0 > 0).mean() * 100
cb_top11_20_wr = (cb_bucket['top_11_20_return'].dropna() / 100.0 > 0).mean() * 100
cb_top5_15_estimated = (cb_top1_10_wr + cb_top11_20_wr) / 2

lr_top1_10_wr = (lr_bucket['top_1_10_return'].dropna() / 100.0 > 0).mean() * 100
lr_top11_20_wr = (lr_bucket['top_11_20_return'].dropna() / 100.0 > 0).mean() * 100
lr_top5_15_estimated = (lr_top1_10_wr + lr_top11_20_wr) / 2

print(f"CatBoost Top 1-10 Overlap 胜率: {cb_top1_10_wr:.2f}%")
print(f"CatBoost Top 11-20 Overlap 胜率: {cb_top11_20_wr:.2f}%")
print(f"CatBoost Top 5-15 Overlap 胜率（估算）: {cb_top5_15_estimated:.2f}%")
print(f"\nLambdaRank Top 1-10 Overlap 胜率: {lr_top1_10_wr:.2f}%")
print(f"LambdaRank Top 11-20 Overlap 胜率: {lr_top11_20_wr:.2f}%")
print(f"LambdaRank Top 5-15 Overlap 胜率（估算）: {lr_top5_15_estimated:.2f}%")

print("\n" + "="*80)
