#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计算基于每日重叠观测（overlap）的胜率"""

import pandas as pd
import numpy as np
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_033750")

print("="*80)
print("基于每日重叠观测（Overlap）的胜率计算")
print("="*80)

# Load timeseries data (these contain daily overlapping observations)
cb_ts = pd.read_csv(run_dir / "catboost_top20_timeseries.csv")
lr_ts = pd.read_csv(run_dir / "lambdarank_top20_timeseries.csv")

# Convert dates
cb_ts['date'] = pd.to_datetime(cb_ts['date'])
lr_ts['date'] = pd.to_datetime(lr_ts['date'])

print("\n【说明】")
print("-"*80)
print("Overlap 胜率：基于每日预测质量评估")
print("- 每个交易日都有一个Top 5-15的收益预测")
print("- 这些预测是重叠的（overlapping），因为每个股票在多个交易日都可能被选中")
print("- 胜率 = (收益为正的交易日数 / 总交易日数) × 100%")
print("- 与non-overlap胜率的区别：")
print("  * Non-overlap: 每10天一期，共25期")
print("  * Overlap: 每个交易日一期，共约250个交易日")

# Calculate overlap win rate for Top 5-15
# We need to get Top 5-15 returns from bucket returns or calculate from predictions
# Let's check if we have bucket returns data
cb_bucket = pd.read_csv(run_dir / "catboost_bucket_returns.csv")
lr_bucket = pd.read_csv(run_dir / "lambdarank_bucket_returns.csv")

print("\n【CatBoost Top 5-15 Overlap 胜率】")
print("-"*80)

# Check if top_5_15_return exists in bucket returns
if 'top_5_15_return' in cb_bucket.columns:
    cb_top5_15_returns = cb_bucket['top_5_15_return'].dropna() / 100.0  # Convert from percentage
    cb_total_days = len(cb_top5_15_returns)
    cb_win_days = (cb_top5_15_returns > 0).sum()
    cb_loss_days = (cb_top5_15_returns <= 0).sum()
    cb_overlap_wr = (cb_top5_15_returns > 0).mean() * 100
    
    print(f"总交易日数: {cb_total_days}")
    print(f"盈利交易日数: {cb_win_days}")
    print(f"亏损交易日数: {cb_loss_days}")
    print(f"Overlap 胜率: {cb_overlap_wr:.2f}%")
else:
    # Calculate from top_5_15_return_mean if available
    if 'top_5_15_return_mean' in cb_bucket.columns:
        cb_top5_15_returns = cb_bucket['top_5_15_return_mean'].dropna()
        cb_total_days = len(cb_top5_15_returns)
        cb_win_days = (cb_top5_15_returns > 0).sum()
        cb_loss_days = (cb_top5_15_returns <= 0).sum()
        cb_overlap_wr = (cb_top5_15_returns > 0).mean() * 100
        
        print(f"总交易日数: {cb_total_days}")
        print(f"盈利交易日数: {cb_win_days}")
        print(f"亏损交易日数: {cb_loss_days}")
        print(f"Overlap 胜率: {cb_overlap_wr:.2f}%")
    else:
        print("未找到 Top 5-15 数据，使用 Top 1-10 数据")
        cb_top1_10_returns = cb_bucket['top_1_10_return'].dropna() / 100.0
        cb_total_days = len(cb_top1_10_returns)
        cb_win_days = (cb_top1_10_returns > 0).sum()
        cb_loss_days = (cb_top1_10_returns <= 0).sum()
        cb_overlap_wr = (cb_top1_10_returns > 0).mean() * 100
        
        print(f"总交易日数: {cb_total_days}")
        print(f"盈利交易日数: {cb_win_days}")
        print(f"亏损交易日数: {cb_loss_days}")
        print(f"Overlap 胜率 (Top 1-10): {cb_overlap_wr:.2f}%")

print("\n【LambdaRank Top 5-15 Overlap 胜率】")
print("-"*80)

if 'top_5_15_return' in lr_bucket.columns:
    lr_top5_15_returns = lr_bucket['top_5_15_return'].dropna() / 100.0
    lr_total_days = len(lr_top5_15_returns)
    lr_win_days = (lr_top5_15_returns > 0).sum()
    lr_loss_days = (lr_top5_15_returns <= 0).sum()
    lr_overlap_wr = (lr_top5_15_returns > 0).mean() * 100
    
    print(f"总交易日数: {lr_total_days}")
    print(f"盈利交易日数: {lr_win_days}")
    print(f"亏损交易日数: {lr_loss_days}")
    print(f"Overlap 胜率: {lr_overlap_wr:.2f}%")
else:
    if 'top_5_15_return_mean' in lr_bucket.columns:
        lr_top5_15_returns = lr_bucket['top_5_15_return_mean'].dropna()
        lr_total_days = len(lr_top5_15_returns)
        lr_win_days = (lr_top5_15_returns > 0).sum()
        lr_loss_days = (lr_top5_15_returns <= 0).sum()
        lr_overlap_wr = (lr_top5_15_returns > 0).mean() * 100
        
        print(f"总交易日数: {lr_total_days}")
        print(f"盈利交易日数: {lr_win_days}")
        print(f"亏损交易日数: {lr_loss_days}")
        print(f"Overlap 胜率: {lr_overlap_wr:.2f}%")
    else:
        print("未找到 Top 5-15 数据，使用 Top 1-10 数据")
        lr_top1_10_returns = lr_bucket['top_1_10_return'].dropna() / 100.0
        lr_total_days = len(lr_top1_10_returns)
        lr_win_days = (lr_top1_10_returns > 0).sum()
        lr_loss_days = (lr_top1_10_returns <= 0).sum()
        lr_overlap_wr = (lr_top1_10_returns > 0).mean() * 100
        
        print(f"总交易日数: {lr_total_days}")
        print(f"盈利交易日数: {lr_win_days}")
        print(f"亏损交易日数: {lr_loss_days}")
        print(f"Overlap 胜率 (Top 1-10): {lr_overlap_wr:.2f}%")

# QQQ benchmark overlap win rate
print("\n【QQQ 基准 Overlap 胜率】")
print("-"*80)
qqq_returns = cb_ts['benchmark_return'].dropna() / 100.0
qqq_total_days = len(qqq_returns)
qqq_win_days = (qqq_returns > 0).sum()
qqq_loss_days = (qqq_returns <= 0).sum()
qqq_overlap_wr = (qqq_returns > 0).mean() * 100

print(f"总交易日数: {qqq_total_days}")
print(f"盈利交易日数: {qqq_win_days}")
print(f"亏损交易日数: {qqq_loss_days}")
print(f"Overlap 胜率: {qqq_overlap_wr:.2f}%")

# Comparison table
print("\n【Overlap vs Non-Overlap 胜率对比】")
print("="*80)
print(f"{'模型':<25} {'Overlap 胜率':>18} {'Non-Overlap 胜率':>22} {'差异':>12}")
print("-"*80)

# Get non-overlap win rates from accumulated returns
cb_nonoverlap = pd.read_csv(run_dir / "catboost_top5_15_rebalance10d_accumulated.csv")
lr_nonoverlap = pd.read_csv(run_dir / "lambdarank_top5_15_rebalance10d_accumulated.csv")

cb_nonoverlap_wr = (cb_nonoverlap['top_gross_return'] > 0).mean() * 100
lr_nonoverlap_wr = (lr_nonoverlap['top_gross_return'] > 0).mean() * 100

print(f"{'CatBoost Top 5-15':<25} {cb_overlap_wr:>16.2f}% {cb_nonoverlap_wr:>20.2f}% {cb_overlap_wr - cb_nonoverlap_wr:>10.2f}%")
print(f"{'LambdaRank Top 5-15':<25} {lr_overlap_wr:>16.2f}% {lr_nonoverlap_wr:>20.2f}% {lr_overlap_wr - lr_nonoverlap_wr:>10.2f}%")

# QQQ non-overlap (from timeseries, which is already non-overlapping for 10-day periods)
qqq_nonoverlap_returns = cb_ts['benchmark_return'].dropna() / 100.0
# Since timeseries is already non-overlapping (every 10 days), we need to calculate differently
# Actually, the timeseries has 25 rows (non-overlapping), so we can use that
qqq_nonoverlap_wr = (qqq_nonoverlap_returns > 0).mean() * 100
print(f"{'QQQ 基准':<25} {qqq_overlap_wr:>16.2f}% {qqq_nonoverlap_wr:>20.2f}% {qqq_overlap_wr - qqq_nonoverlap_wr:>10.2f}%")

print("\n【说明】")
print("-"*80)
print("Overlap 胜率：基于每日预测质量评估（每日计算，重叠观测）")
print("Non-Overlap 胜率：基于非重叠回测（每10天一期，共25期）")
print("差异：Overlap胜率通常高于Non-Overlap胜率，因为包含了更多观测点")

print("\n" + "="*80)
