#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析和比较 LambdaRank、CatBoost 和 QQQ 基准的表现"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_033750")

print("="*80)
print("LambdaRank vs CatBoost vs QQQ 对比分析")
print("="*80)

# Load Top 5-15 accumulated returns
cb_top5_15 = pd.read_csv(run_dir / "catboost_top5_15_rebalance10d_accumulated.csv")
lr_top5_15 = pd.read_csv(run_dir / "lambdarank_top5_15_rebalance10d_accumulated.csv")

# Load benchmark data from timeseries
cb_ts = pd.read_csv(run_dir / "catboost_top20_timeseries.csv")
lr_ts = pd.read_csv(run_dir / "lambdarank_top20_timeseries.csv")

# Extract benchmark returns
cb_ts['date'] = pd.to_datetime(cb_ts['date'])
lr_ts['date'] = pd.to_datetime(lr_ts['date'])

# Get benchmark returns (should be same for both)
benchmark_returns = cb_ts[['date', 'benchmark_return', 'cum_benchmark_return']].copy()

# Convert Top 5-15 dates to datetime
cb_top5_15['date'] = pd.to_datetime(cb_top5_15['date'])
lr_top5_15['date'] = pd.to_datetime(lr_top5_15['date'])

# Merge benchmark data
cb_top5_15 = cb_top5_15.merge(benchmark_returns, on='date', how='left')
lr_top5_15 = lr_top5_15.merge(benchmark_returns, on='date', how='left')

print("\n【Top 5-15 累计收益对比】")
print("-"*80)

# Calculate final returns
cb_final = cb_top5_15['acc_return'].iloc[-1] * 100
lr_final = lr_top5_15['acc_return'].iloc[-1] * 100
qqq_final = benchmark_returns['cum_benchmark_return'].iloc[-1]

print(f"CatBoost Top 5-15:   {cb_final:>8.2f}%")
print(f"LambdaRank Top 5-15: {lr_final:>8.2f}%")
print(f"QQQ 基准:            {qqq_final:>8.2f}%")

print("\n【平均每期收益对比】")
print("-"*80)
cb_avg = cb_top5_15['top_gross_return'].mean() * 100
lr_avg = lr_top5_15['top_gross_return'].mean() * 100
qqq_avg = benchmark_returns['benchmark_return'].mean()

print(f"CatBoost Top 5-15:   {cb_avg:>8.4f}%")
print(f"LambdaRank Top 5-15: {lr_avg:>8.4f}%")
print(f"QQQ 基准:            {qqq_avg:>8.4f}%")

print("\n【风险指标对比】")
print("-"*80)

# Calculate Sharpe ratios (annualized)
periods_per_year = 252.0 / 10

cb_returns = cb_top5_15['top_gross_return'].dropna()
lr_returns = lr_top5_15['top_gross_return'].dropna()
qqq_returns = benchmark_returns['benchmark_return'].dropna() / 100.0

cb_sharpe = (cb_returns.mean() / cb_returns.std()) * np.sqrt(periods_per_year) if cb_returns.std() > 0 else np.nan
lr_sharpe = (lr_returns.mean() / lr_returns.std()) * np.sqrt(periods_per_year) if lr_returns.std() > 0 else np.nan
qqq_sharpe = (qqq_returns.mean() / qqq_returns.std()) * np.sqrt(periods_per_year) if qqq_returns.std() > 0 else np.nan

print(f"CatBoost Top 5-15 Sharpe:   {cb_sharpe:>8.4f}")
print(f"LambdaRank Top 5-15 Sharpe: {lr_sharpe:>8.4f}")
print(f"QQQ 基准 Sharpe:            {qqq_sharpe:>8.4f}")

# Calculate max drawdown
def max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

cb_mdd = max_drawdown(cb_returns) * 100
lr_mdd = max_drawdown(lr_returns) * 100
qqq_mdd = max_drawdown(qqq_returns) * 100

print(f"\nCatBoost Top 5-15 Max Drawdown:   {cb_mdd:>8.2f}%")
print(f"LambdaRank Top 5-15 Max Drawdown: {lr_mdd:>8.2f}%")
print(f"QQQ 基准 Max Drawdown:            {qqq_mdd:>8.2f}%")

# Calculate win rate
cb_win_rate = (cb_returns > 0).mean() * 100
lr_win_rate = (lr_returns > 0).mean() * 100
qqq_win_rate = (qqq_returns > 0).mean() * 100

print(f"\nCatBoost Top 5-15 Win Rate:   {cb_win_rate:>8.2f}%")
print(f"LambdaRank Top 5-15 Win Rate: {lr_win_rate:>8.2f}%")
print(f"QQQ 基准 Win Rate:            {qqq_win_rate:>8.2f}%")

# Generate comparison plot
print("\n【生成对比图表】")
print("-"*80)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Cumulative returns comparison
ax1 = axes[0]
ax1.plot(cb_top5_15['date'], cb_top5_15['acc_return'] * 100, 
         label='CatBoost Top 5-15', linewidth=2.0, color='#1f77b4')
ax1.plot(lr_top5_15['date'], lr_top5_15['acc_return'] * 100, 
         label='LambdaRank Top 5-15', linewidth=2.0, color='#ff7f0e')
ax1.plot(benchmark_returns['date'], benchmark_returns['cum_benchmark_return'], 
         label='QQQ Benchmark', linewidth=2.0, linestyle='--', color='black')
ax1.axhline(0.0, color='#999999', linewidth=1.0, linestyle=':')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
ax1.set_title('Top 5-15 Accumulated Return Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Period returns comparison
ax2 = axes[1]
ax2.plot(cb_top5_15['date'], cb_top5_15['top_gross_return'] * 100, 
         label='CatBoost Top 5-15', linewidth=1.5, color='#1f77b4', alpha=0.7)
ax2.plot(lr_top5_15['date'], lr_top5_15['top_gross_return'] * 100, 
         label='LambdaRank Top 5-15', linewidth=1.5, color='#ff7f0e', alpha=0.7)
ax2.plot(benchmark_returns['date'], benchmark_returns['benchmark_return'], 
         label='QQQ Benchmark', linewidth=1.5, linestyle='--', color='black', alpha=0.7)
ax2.axhline(0.0, color='#999999', linewidth=1.0, linestyle=':')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Period Return (%)', fontsize=12)
ax2.set_title('Top 5-15 Period Returns Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_file = run_dir / "lambdarank_catboost_qqq_comparison.png"
plt.savefig(output_file, dpi=160, bbox_inches='tight')
plt.close()

print(f"对比图表已保存: {output_file.name}")

# Create summary table
print("\n【综合对比表】")
print("="*80)
print(f"{'指标':<25} {'CatBoost Top 5-15':>20} {'LambdaRank Top 5-15':>22} {'QQQ 基准':>15}")
print("-"*80)
print(f"{'最终累计收益':<25} {cb_final:>18.2f}% {lr_final:>20.2f}% {qqq_final:>13.2f}%")
print(f"{'平均每期收益':<25} {cb_avg:>18.4f}% {lr_avg:>20.4f}% {qqq_avg:>13.4f}%")
print(f"{'Sharpe Ratio':<25} {cb_sharpe:>18.4f} {lr_sharpe:>20.4f} {qqq_sharpe:>13.4f}")
print(f"{'最大回撤':<25} {cb_mdd:>18.2f}% {lr_mdd:>20.2f}% {qqq_mdd:>13.2f}%")
print(f"{'胜率':<25} {cb_win_rate:>18.2f}% {lr_win_rate:>20.2f}% {qqq_win_rate:>13.2f}%")
print("="*80)

# Calculate outperformance
cb_outperform = cb_final - qqq_final
lr_outperform = lr_final - qqq_final

print("\n【相对基准表现】")
print("-"*80)
print(f"CatBoost Top 5-15 相对 QQQ:   {cb_outperform:>8.2f}%")
print(f"LambdaRank Top 5-15 相对 QQQ: {lr_outperform:>8.2f}%")

print("\n" + "="*80)
