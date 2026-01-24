#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解释胜率的计算方法"""

import pandas as pd
import numpy as np
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_033750")

print("="*80)
print("胜率计算方法说明")
print("="*80)

# Load data
cb_top5_15 = pd.read_csv(run_dir / "catboost_top5_15_rebalance10d_accumulated.csv")
lr_top5_15 = pd.read_csv(run_dir / "lambdarank_top5_15_rebalance10d_accumulated.csv")

print("\n【胜率定义】")
print("-"*80)
print("胜率 = (收益为正的期间数 / 总期间数) × 100%")
print("其中：")
print("- 每个期间 = 10天持有期")
print("- 收益为正 = top_gross_return > 0")
print("- 总期间数 = 25期（每10天一期）")

print("\n【CatBoost Top 5-15 胜率计算示例】")
print("-"*80)
cb_returns = cb_top5_15['top_gross_return']
print(f"总期间数: {len(cb_returns)}")
print(f"收益为正的期间数: {(cb_returns > 0).sum()}")
print(f"收益为负的期间数: {(cb_returns <= 0).sum()}")
print(f"\n胜率 = {(cb_returns > 0).sum()} / {len(cb_returns)} × 100% = {(cb_returns > 0).mean() * 100:.2f}%")

print("\n【详细期间收益数据（前10期）】")
print("-"*80)
print("日期 | 期间收益 | 是否为正")
print("-"*50)
for i in range(min(10, len(cb_top5_15))):
    date = cb_top5_15.iloc[i]['date']
    ret = cb_top5_15.iloc[i]['top_gross_return'] * 100
    is_win = "是" if ret > 0 else "否"
    print(f"{date[:10]} | {ret:>8.4f}% | {is_win}")

print("\n【LambdaRank Top 5-15 胜率计算示例】")
print("-"*80)
lr_returns = lr_top5_15['top_gross_return']
print(f"总期间数: {len(lr_returns)}")
print(f"收益为正的期间数: {(lr_returns > 0).sum()}")
print(f"收益为负的期间数: {(lr_returns <= 0).sum()}")
print(f"\n胜率 = {(lr_returns > 0).sum()} / {len(lr_returns)} × 100% = {(lr_returns > 0).mean() * 100:.2f}%")

print("\n【LambdaRank 详细期间收益数据（前10期）】")
print("-"*80)
print("日期 | 期间收益 | 是否为正")
print("-"*50)
for i in range(min(10, len(lr_top5_15))):
    date = lr_top5_15.iloc[i]['date']
    ret = lr_top5_15.iloc[i]['top_gross_return'] * 100
    is_win = "是" if ret > 0 else "否"
    print(f"{date[:10]} | {ret:>8.4f}% | {is_win}")

print("\n【完整统计】")
print("-"*80)
print(f"{'模型':<20} {'总期间数':<10} {'盈利期间':<10} {'亏损期间':<10} {'胜率':<10}")
print("-"*60)
cb_wins = (cb_returns > 0).sum()
cb_losses = (cb_returns <= 0).sum()
cb_wr = (cb_returns > 0).mean() * 100
print(f"{'CatBoost Top 5-15':<20} {len(cb_returns):<10} {cb_wins:<10} {cb_losses:<10} {cb_wr:<10.2f}%")

lr_wins = (lr_returns > 0).sum()
lr_losses = (lr_returns <= 0).sum()
lr_wr = (lr_returns > 0).mean() * 100
print(f"{'LambdaRank Top 5-15':<20} {len(lr_returns):<10} {lr_wins:<10} {lr_losses:<10} {lr_wr:<10.2f}%")

print("\n【说明】")
print("-"*80)
print("1. 胜率基于每10天一个持有期的收益计算")
print("2. 如果某个10天持有期的收益 > 0，则该期间为'盈利'")
print("3. 如果某个10天持有期的收益 <= 0，则该期间为'亏损'")
print("4. 胜率 = 盈利期间数 / 总期间数")
print("5. 这与累计收益不同：即使胜率较低，如果盈利期间的收益较大，")
print("   累计收益仍可能很高（如 LambdaRank 的情况）")

print("\n" + "="*80)
