#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计算完整的 Overlap 胜率（包括 Top 5-15）"""

import pandas as pd
import numpy as np
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_041040")

print("="*80)
print("完整的 Overlap 胜率计算（包括 Top 5-15）")
print("="*80)

# Load bucket returns data
cb_bucket = pd.read_csv(run_dir / "catboost_bucket_returns.csv")
lr_bucket = pd.read_csv(run_dir / "lambdarank_bucket_returns.csv")

# Check for Top 5-15 columns
cb_has_5_15 = any('5_15' in c for c in cb_bucket.columns)
lr_has_5_15 = any('5_15' in c for c in lr_bucket.columns)

print(f"\nCatBoost 是否有 Top 5-15 列: {cb_has_5_15}")
print(f"LambdaRank 是否有 Top 5-15 列: {lr_has_5_15}")

if cb_has_5_15:
    cb_5_15_cols = [c for c in cb_bucket.columns if '5_15' in c]
    print(f"CatBoost Top 5-15 列: {cb_5_15_cols}")

if lr_has_5_15:
    lr_5_15_cols = [c for c in lr_bucket.columns if '5_15' in c]
    print(f"LambdaRank Top 5-15 列: {lr_5_15_cols}")

# Calculate overlap win rates
buckets = [
    ('Top 1-10', 'top_1_10_return'),
    ('Top 5-15', 'top_5_15_return'),
    ('Top 11-20', 'top_11_20_return'),
    ('Top 21-30', 'top_21_30_return'),
]

print("\n【CatBoost Overlap 胜率】")
print("="*80)
print(f"{'分桶':<15} {'总交易日数':<12} {'盈利日数':<12} {'亏损日数':<12} {'Overlap 胜率':<15}")
print("-"*70)

cb_results = {}
for bucket_name, col_name in buckets:
    # Try direct column first
    if col_name in cb_bucket.columns:
        returns = cb_bucket[col_name].dropna() / 100.0
    elif f"{col_name}_mean" in cb_bucket.columns:
        returns = cb_bucket[f"{col_name}_mean"].dropna()
    else:
        print(f"{bucket_name:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}")
        continue
    
    total = len(returns)
    wins = (returns > 0).sum()
    losses = (returns <= 0).sum()
    wr = (returns > 0).mean() * 100
    cb_results[bucket_name] = wr
    print(f"{bucket_name:<15} {total:<12} {wins:<12} {losses:<12} {wr:<15.2f}%")

print("\n【LambdaRank Overlap 胜率】")
print("="*80)
print(f"{'分桶':<15} {'总交易日数':<12} {'盈利日数':<12} {'亏损日数':<12} {'Overlap 胜率':<15}")
print("-"*70)

lr_results = {}
for bucket_name, col_name in buckets:
    if col_name in lr_bucket.columns:
        returns = lr_bucket[col_name].dropna() / 100.0
    elif f"{col_name}_mean" in lr_bucket.columns:
        returns = lr_bucket[f"{col_name}_mean"].dropna()
    else:
        print(f"{bucket_name:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}")
        continue
    
    total = len(returns)
    wins = (returns > 0).sum()
    losses = (returns <= 0).sum()
    wr = (returns > 0).mean() * 100
    lr_results[bucket_name] = wr
    print(f"{bucket_name:<15} {total:<12} {wins:<12} {losses:<12} {wr:<15.2f}%")

# QQQ benchmark
print("\n【QQQ 基准 Overlap 胜率】")
print("="*80)
qqq_returns = cb_bucket['benchmark_return'].dropna() / 100.0
qqq_total = len(qqq_returns)
qqq_wins = (qqq_returns > 0).sum()
qqq_losses = (qqq_returns <= 0).sum()
qqq_wr = (qqq_returns > 0).mean() * 100
print(f"总交易日数: {qqq_total}")
print(f"盈利日数: {qqq_wins}")
print(f"亏损日数: {qqq_losses}")
print(f"Overlap 胜率: {qqq_wr:.2f}%")

# Summary comparison
print("\n【Overlap 胜率汇总对比】")
print("="*80)
print(f"{'分桶':<15} {'CatBoost':<15} {'LambdaRank':<15} {'QQQ 基准':<15}")
print("-"*65)

for bucket_name, _ in buckets:
    cb_wr = cb_results.get(bucket_name, 'N/A')
    lr_wr = lr_results.get(bucket_name, 'N/A')
    
    cb_str = f"{cb_wr:.2f}%" if isinstance(cb_wr, (int, float)) else cb_wr
    lr_str = f"{lr_wr:.2f}%" if isinstance(lr_wr, (int, float)) else lr_wr
    qqq_str = f"{qqq_wr:.2f}%"
    
    print(f"{bucket_name:<15} {cb_str:<15} {lr_str:<15} {qqq_str:<15}")

print("\n【说明】")
print("-"*80)
print("Overlap 胜率：基于每日预测质量评估（每日计算，重叠观测）")
print("- 每个交易日都有一个Top bucket的收益预测")
print("- 总交易日数约249天（测试期间的所有交易日）")
print("- 胜率 = (收益为正的交易日数 / 总交易日数) × 100%")

print("\n" + "="*80)
