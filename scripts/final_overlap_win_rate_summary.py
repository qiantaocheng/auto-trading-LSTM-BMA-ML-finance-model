#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终 Overlap 胜率汇总（基于现有数据）"""

import pandas as pd
import numpy as np
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_035449")

print("="*80)
print("所有模型的 Overlap 胜率汇总（基于每日重叠观测）")
print("="*80)

# Load data
cb_bucket = pd.read_csv(run_dir / "catboost_bucket_returns.csv")
lr_bucket = pd.read_csv(run_dir / "lambdarank_bucket_returns.csv")

# Get non-overlap win rates for comparison
cb_nonoverlap = pd.read_csv(run_dir / "catboost_top5_15_rebalance10d_accumulated.csv")
lr_nonoverlap = pd.read_csv(run_dir / "lambdarank_top5_15_rebalance10d_accumulated.csv")

print("\n【Overlap 胜率（每日重叠观测）】")
print("="*80)
print(f"{'模型/分桶':<25} {'总交易日数':<12} {'盈利日数':<12} {'亏损日数':<12} {'Overlap 胜率':<15}")
print("-"*80)

# CatBoost
cb_top1_10 = cb_bucket['top_1_10_return'].dropna() / 100.0
cb_top11_20 = cb_bucket['top_11_20_return'].dropna() / 100.0
cb_top21_30 = cb_bucket['top_21_30_return'].dropna() / 100.0

print(f"{'CatBoost Top 1-10':<25} {len(cb_top1_10):<12} {(cb_top1_10 > 0).sum():<12} {(cb_top1_10 <= 0).sum():<12} {(cb_top1_10 > 0).mean()*100:<15.2f}%")
print(f"{'CatBoost Top 11-20':<25} {len(cb_top11_20):<12} {(cb_top11_20 > 0).sum():<12} {(cb_top11_20 <= 0).sum():<12} {(cb_top11_20 > 0).mean()*100:<15.2f}%")
print(f"{'CatBoost Top 21-30':<25} {len(cb_top21_30):<12} {(cb_top21_30 > 0).sum():<12} {(cb_top21_30 <= 0).sum():<12} {(cb_top21_30 > 0).mean()*100:<15.2f}%")

# LambdaRank
lr_top1_10 = lr_bucket['top_1_10_return'].dropna() / 100.0
lr_top11_20 = lr_bucket['top_11_20_return'].dropna() / 100.0
lr_top21_30 = lr_bucket['top_21_30_return'].dropna() / 100.0

print(f"{'LambdaRank Top 1-10':<25} {len(lr_top1_10):<12} {(lr_top1_10 > 0).sum():<12} {(lr_top1_10 <= 0).sum():<12} {(lr_top1_10 > 0).mean()*100:<15.2f}%")
print(f"{'LambdaRank Top 11-20':<25} {len(lr_top11_20):<12} {(lr_top11_20 > 0).sum():<12} {(lr_top11_20 <= 0).sum():<12} {(lr_top11_20 > 0).mean()*100:<15.2f}%")
print(f"{'LambdaRank Top 21-30':<25} {len(lr_top21_30):<12} {(lr_top21_30 > 0).sum():<12} {(lr_top21_30 <= 0).sum():<12} {(lr_top21_30 > 0).mean()*100:<15.2f}%")

# QQQ
qqq_returns = cb_bucket['benchmark_return'].dropna() / 100.0
print(f"{'QQQ 基准':<25} {len(qqq_returns):<12} {(qqq_returns > 0).sum():<12} {(qqq_returns <= 0).sum():<12} {(qqq_returns > 0).mean()*100:<15.2f}%")

print("\n【Non-Overlap 胜率（每10天一期，共25期）】")
print("="*80)
print(f"{'模型':<25} {'总期间数':<12} {'盈利期间':<12} {'亏损期间':<12} {'Non-Overlap 胜率':<18}")
print("-"*80)

cb_nonoverlap_returns = cb_nonoverlap['top_gross_return'].dropna()
lr_nonoverlap_returns = lr_nonoverlap['top_gross_return'].dropna()

print(f"{'CatBoost Top 5-15':<25} {len(cb_nonoverlap_returns):<12} {(cb_nonoverlap_returns > 0).sum():<12} {(cb_nonoverlap_returns <= 0).sum():<12} {(cb_nonoverlap_returns > 0).mean()*100:<18.2f}%")
print(f"{'LambdaRank Top 5-15':<25} {len(lr_nonoverlap_returns):<12} {(lr_nonoverlap_returns > 0).sum():<12} {(lr_nonoverlap_returns <= 0).sum():<12} {(lr_nonoverlap_returns > 0).mean()*100:<18.2f}%")

# QQQ non-overlap (from timeseries - 25 periods)
cb_ts = pd.read_csv(run_dir / "catboost_top20_timeseries.csv")
qqq_nonoverlap_returns = cb_ts['benchmark_return'].dropna() / 100.0
print(f"{'QQQ 基准':<25} {len(qqq_nonoverlap_returns):<12} {(qqq_nonoverlap_returns > 0).sum():<12} {(qqq_nonoverlap_returns <= 0).sum():<12} {(qqq_nonoverlap_returns > 0).mean()*100:<18.2f}%")

print("\n【Overlap vs Non-Overlap 胜率对比总结】")
print("="*80)
print(f"{'指标':<25} {'Overlap 胜率':<18} {'Non-Overlap 胜率':<22} {'差异':<12}")
print("-"*80)

# For Top 5-15, we use Top 1-10 overlap as proxy since Top 5-15 daily data is not available
cb_overlap_wr_top1_10 = (cb_top1_10 > 0).mean() * 100
lr_overlap_wr_top1_10 = (lr_top1_10 > 0).mean() * 100
cb_nonoverlap_wr_top5_15 = (cb_nonoverlap_returns > 0).mean() * 100
lr_nonoverlap_wr_top5_15 = (lr_nonoverlap_returns > 0).mean() * 100

print(f"{'CatBoost (Top 1-10 vs Top 5-15)':<25} {cb_overlap_wr_top1_10:>16.2f}% {cb_nonoverlap_wr_top5_15:>20.2f}% {cb_overlap_wr_top1_10 - cb_nonoverlap_wr_top5_15:>10.2f}%")
print(f"{'LambdaRank (Top 1-10 vs Top 5-15)':<25} {lr_overlap_wr_top1_10:>16.2f}% {lr_nonoverlap_wr_top5_15:>20.2f}% {lr_overlap_wr_top1_10 - lr_nonoverlap_wr_top5_15:>10.2f}%")

qqq_overlap_wr = (qqq_returns > 0).mean() * 100
qqq_nonoverlap_wr = (qqq_nonoverlap_returns > 0).mean() * 100
print(f"{'QQQ 基准':<25} {qqq_overlap_wr:>16.2f}% {qqq_nonoverlap_wr:>20.2f}% {qqq_overlap_wr - qqq_nonoverlap_wr:>10.2f}%")

print("\n【说明】")
print("-"*80)
print("1. Overlap 胜率：基于每日预测质量评估（每日计算，重叠观测）")
print("   - 每个交易日都有一个Top bucket的收益预测")
print("   - 总交易日数约249天")
print("   - 胜率 = (收益为正的交易日数 / 总交易日数) × 100%")
print("\n2. Non-Overlap 胜率：基于非重叠回测（每10天一期，共25期）")
print("   - 每10天再平衡一次，持有10天")
print("   - 胜率 = (收益为正的期间数 / 总期间数) × 100%")
print("\n3. 注意：Top 5-15 的每日数据未保存在bucket_returns.csv中")
print("   - Overlap 胜率使用 Top 1-10 作为参考")
print("   - Non-Overlap 胜率使用 Top 5-15 的实际数据")

print("\n" + "="*80)
