#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计算基于每日重叠观测（overlap）的胜率 - 修复版"""

import pandas as pd
import numpy as np
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_033750")

print("="*80)
print("基于每日重叠观测（Overlap）的胜率计算")
print("="*80)

# Load bucket returns data (daily overlapping observations)
cb_bucket = pd.read_csv(run_dir / "catboost_bucket_returns.csv")
lr_bucket = pd.read_csv(run_dir / "lambdarank_bucket_returns.csv")

# Load benchmark data
cb_ts = pd.read_csv(run_dir / "catboost_top20_timeseries.csv")
qqq_returns = cb_ts['benchmark_return'].dropna() / 100.0

print("\n【说明】")
print("-"*80)
print("Overlap 胜率：基于每日预测质量评估（每日计算，重叠观测）")
print("- 每个交易日都有一个Top bucket的收益预测")
print("- 这些预测是重叠的（overlapping），因为每个股票在多个交易日都可能被选中")
print("- 胜率 = (收益为正的交易日数 / 总交易日数) × 100%")
print("\nNon-Overlap 胜率：基于非重叠回测（每10天一期，共25期）")
print("- 每10天再平衡一次，持有10天")
print("- 胜率 = (收益为正的期间数 / 总期间数) × 100%")

# Check available columns
print("\n【检查可用数据列】")
print("-"*80)
cb_cols_5_15 = [c for c in cb_bucket.columns if '5' in c and '15' in c]
lr_cols_5_15 = [c for c in lr_bucket.columns if '5' in c and '15' in c]
print(f"CatBoost Top 5-15 相关列: {cb_cols_5_15}")
print(f"LambdaRank Top 5-15 相关列: {lr_cols_5_15}")

# Since Top 5-15 data is not in bucket_returns, we'll use Top 1-10 and Top 11-20
# Top 5-15 can be approximated or we need to recalculate
# For now, let's use Top 1-10 for comparison, and note that Top 5-15 needs to be calculated

print("\n【CatBoost Overlap 胜率（基于可用数据）】")
print("-"*80)

# Use Top 1-10 as proxy (since Top 5-15 is not in bucket_returns)
cb_top1_10_returns = cb_bucket['top_1_10_return'].dropna() / 100.0
cb_total_days = len(cb_top1_10_returns)
cb_win_days = (cb_top1_10_returns > 0).sum()
cb_loss_days = (cb_top1_10_returns <= 0).sum()
cb_overlap_wr = (cb_top1_10_returns > 0).mean() * 100

print(f"数据来源: Top 1-10 (Top 5-15 数据未在bucket_returns中)")
print(f"总交易日数: {cb_total_days}")
print(f"盈利交易日数: {cb_win_days}")
print(f"亏损交易日数: {cb_loss_days}")
print(f"Overlap 胜率: {cb_overlap_wr:.2f}%")

print("\n【LambdaRank Overlap 胜率（基于可用数据）】")
print("-"*80)

lr_top1_10_returns = lr_bucket['top_1_10_return'].dropna() / 100.0
lr_total_days = len(lr_top1_10_returns)
lr_win_days = (lr_top1_10_returns > 0).sum()
lr_loss_days = (lr_top1_10_returns <= 0).sum()
lr_overlap_wr = (lr_top1_10_returns > 0).mean() * 100

print(f"数据来源: Top 1-10 (Top 5-15 数据未在bucket_returns中)")
print(f"总交易日数: {lr_total_days}")
print(f"盈利交易日数: {lr_win_days}")
print(f"亏损交易日数: {lr_loss_days}")
print(f"Overlap 胜率: {lr_overlap_wr:.2f}%")

print("\n【QQQ 基准 Overlap 胜率】")
print("-"*80)
# Note: timeseries has non-overlapping data (25 periods), so we need daily benchmark data
# Let's use the bucket returns benchmark which should have daily data
qqq_bucket_returns = cb_bucket['benchmark_return'].dropna() / 100.0
qqq_total_days = len(qqq_bucket_returns)
qqq_win_days = (qqq_bucket_returns > 0).sum()
qqq_loss_days = (qqq_bucket_returns <= 0).sum()
qqq_overlap_wr = (qqq_bucket_returns > 0).mean() * 100

print(f"总交易日数: {qqq_total_days}")
print(f"盈利交易日数: {qqq_win_days}")
print(f"亏损交易日数: {qqq_loss_days}")
print(f"Overlap 胜率: {qqq_overlap_wr:.2f}%")

# Get non-overlap win rates
print("\n【Non-Overlap 胜率（用于对比）】")
print("-"*80)
cb_nonoverlap = pd.read_csv(run_dir / "catboost_top5_15_rebalance10d_accumulated.csv")
lr_nonoverlap = pd.read_csv(run_dir / "lambdarank_top5_15_rebalance10d_accumulated.csv")

cb_nonoverlap_wr = (cb_nonoverlap['top_gross_return'] > 0).mean() * 100
lr_nonoverlap_wr = (lr_nonoverlap['top_gross_return'] > 0).mean() * 100

# QQQ non-overlap (from timeseries - 25 periods)
qqq_nonoverlap_returns = cb_ts['benchmark_return'].dropna() / 100.0
qqq_nonoverlap_wr = (qqq_nonoverlap_returns > 0).mean() * 100

print(f"CatBoost Top 5-15 Non-Overlap 胜率: {cb_nonoverlap_wr:.2f}%")
print(f"LambdaRank Top 5-15 Non-Overlap 胜率: {lr_nonoverlap_wr:.2f}%")
print(f"QQQ 基准 Non-Overlap 胜率: {qqq_nonoverlap_wr:.2f}%")

# Comparison table
print("\n【Overlap vs Non-Overlap 胜率对比】")
print("="*80)
print(f"{'模型':<25} {'Overlap 胜率':>18} {'Non-Overlap 胜率':>22} {'差异':>12}")
print("-"*80)
print(f"{'CatBoost Top 5-15':<25} {cb_overlap_wr:>16.2f}% {cb_nonoverlap_wr:>20.2f}% {cb_overlap_wr - cb_nonoverlap_wr:>10.2f}%")
print(f"{'LambdaRank Top 5-15':<25} {lr_overlap_wr:>16.2f}% {lr_nonoverlap_wr:>20.2f}% {lr_overlap_wr - lr_nonoverlap_wr:>10.2f}%")
print(f"{'QQQ 基准':<25} {qqq_overlap_wr:>16.2f}% {qqq_nonoverlap_wr:>20.2f}% {qqq_overlap_wr - qqq_nonoverlap_wr:>10.2f}%")

print("\n【注意】")
print("-"*80)
print("1. Overlap 胜率基于 Top 1-10 数据（因为 Top 5-15 的每日数据未保存在bucket_returns.csv中）")
print("2. Non-Overlap 胜率基于 Top 5-15 的非重叠回测数据（每10天一期）")
print("3. 要获得准确的 Top 5-15 Overlap 胜率，需要修改代码以保存 Top 5-15 的每日数据")

print("\n" + "="*80)
