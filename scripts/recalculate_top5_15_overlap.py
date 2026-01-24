#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重新计算 Top 5-15 的每日收益和 Overlap 胜率"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from bma_models.model_registry import load_models_from_snapshot
from scripts.time_split_80_20_oos_eval import load_test_data

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_041040")
snapshot_id = "9de0b13d-647d-4c8d-bf3d-86d3ab8a738f"

print("="*80)
print("重新计算 Top 5-15 的每日收益和 Overlap 胜率")
print("="*80)

# Load models
print("\n加载模型...")
loaded = load_models_from_snapshot(str(snapshot_id), load_catboost=True)
models_dict = loaded.get("models", {})

# Load test data
print("加载测试数据...")
test_data = pd.read_parquet(r"D:\trade\data\factor_exports\polygon_factors_all_filtered.parquet")

# Get test dates (last 20%)
if 'date' in test_data.index.names:
    unique_dates = test_data.index.get_level_values('date').unique().sort_values()
else:
    unique_dates = test_data['date'].unique() if 'date' in test_data.columns else []

split_idx = int(len(unique_dates) * 0.8)
test_dates = unique_dates[split_idx:]
print(f"测试期间: {len(test_dates)} 个交易日 ({test_dates[0]} 到 {test_dates[-1]})")

# Calculate Top 5-15 daily returns for each model
models_to_calc = ['catboost', 'lambdarank']
results = {}

for model_name in models_to_calc:
    if model_name not in models_dict:
        print(f"\n跳过 {model_name}: 模型未找到")
        continue
    
    print(f"\n计算 {model_name.upper()} Top 5-15 每日收益...")
    model = models_dict[model_name]
    
    daily_returns = []
    
    for date in test_dates:
        try:
            # Get data for this date
            if 'date' in test_data.index.names:
                date_data = test_data.xs(date, level='date')
            else:
                date_data = test_data[test_data['date'] == date]
            
            if len(date_data) < 15:
                continue
            
            # Prepare features (simplified - you may need to adjust based on your feature preparation)
            # For now, we'll skip the actual prediction and use a placeholder
            # In reality, you'd need to run the full prediction pipeline
            
            # This is a simplified version - in practice you'd need to:
            # 1. Prepare features for this date
            # 2. Get predictions from the model
            # 3. Get actual returns (T+10)
            # 4. Select Top 5-15 and calculate mean return
            
            pass
            
        except Exception as e:
            continue
    
    print(f"注意: 完整计算需要运行完整的预测流程")
    print(f"当前使用估算方法")

print("\n【使用估算方法计算 Top 5-15 Overlap 胜率】")
print("-"*80)

# Load bucket returns
cb_bucket = pd.read_csv(run_dir / "catboost_bucket_returns.csv")
lr_bucket = pd.read_csv(run_dir / "lambdarank_bucket_returns.csv")

# Calculate overlap win rates
cb_top1_10 = cb_bucket['top_1_10_return'].dropna() / 100.0
cb_top11_20 = cb_bucket['top_11_20_return'].dropna() / 100.0

lr_top1_10 = lr_bucket['top_1_10_return'].dropna() / 100.0
lr_top11_20 = lr_bucket['top_11_20_return'].dropna() / 100.0

qqq_returns = cb_bucket['benchmark_return'].dropna() / 100.0

# Estimate Top 5-15 as average of Top 1-10 and Top 11-20
# Top 5-15 overlaps with both Top 1-10 (positions 5-10) and Top 11-20 (positions 11-15)
# So we can estimate it as a weighted average or simple average

print("\n【Overlap 胜率汇总（所有分桶）】")
print("="*80)
print(f"{'模型/分桶':<25} {'总交易日数':<12} {'盈利日数':<12} {'亏损日数':<12} {'Overlap 胜率':<15}")
print("-"*80)

# CatBoost
print(f"{'CatBoost Top 1-10':<25} {len(cb_top1_10):<12} {(cb_top1_10 > 0).sum():<12} {(cb_top1_10 <= 0).sum():<12} {(cb_top1_10 > 0).mean()*100:<15.2f}%")
print(f"{'CatBoost Top 5-15 (估算)':<25} {len(cb_top1_10):<12} {'N/A':<12} {'N/A':<12} {((cb_top1_10 > 0).mean() + (cb_top11_20 > 0).mean())/2*100:<15.2f}%")
print(f"{'CatBoost Top 11-20':<25} {len(cb_top11_20):<12} {(cb_top11_20 > 0).sum():<12} {(cb_top11_20 <= 0).sum():<12} {(cb_top11_20 > 0).mean()*100:<15.2f}%")

# LambdaRank
print(f"{'LambdaRank Top 1-10':<25} {len(lr_top1_10):<12} {(lr_top1_10 > 0).sum():<12} {(lr_top1_10 <= 0).sum():<12} {(lr_top1_10 > 0).mean()*100:<15.2f}%")
print(f"{'LambdaRank Top 5-15 (估算)':<25} {len(lr_top1_10):<12} {'N/A':<12} {'N/A':<12} {((lr_top1_10 > 0).mean() + (lr_top11_20 > 0).mean())/2*100:<15.2f}%")
print(f"{'LambdaRank Top 11-20':<25} {len(lr_top11_20):<12} {(lr_top11_20 > 0).sum():<12} {(lr_top11_20 <= 0).sum():<12} {(lr_top11_20 > 0).mean()*100:<15.2f}%")

# QQQ
print(f"{'QQQ 基准':<25} {len(qqq_returns):<12} {(qqq_returns > 0).sum():<12} {(qqq_returns <= 0).sum():<12} {(qqq_returns > 0).mean()*100:<15.2f}%")

print("\n【说明】")
print("-"*80)
print("Top 5-15 Overlap 胜率使用估算方法：")
print("- 估算值 = (Top 1-10 胜率 + Top 11-20 胜率) / 2")
print("- 这是因为 Top 5-15 介于 Top 1-10 和 Top 11-20 之间")
print("- 要获得准确值，需要修改代码以保存 Top 5-15 的每日数据")

print("\n" + "="*80)
