#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较过滤前后（Top 5-15）的结果"""

import pandas as pd
import numpy as np
from pathlib import Path

# 未过滤的结果
unfiltered_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_041850")
# 过滤后的结果
filtered_dir = Path(r"d:\trade\results\t10_time_split_80_20_filtered\run_20260120_042659")

print("="*80)
print("Top 5-15 过滤前后对比（移除波动率最高的2只和成交量最差的2只）")
print("="*80)

models = ['catboost', 'lambdarank']

for model_name in models:
    print(f"\n【{model_name.upper()}】")
    print("-"*80)
    
    # 读取bucket returns
    try:
        unfiltered_bucket = pd.read_csv(unfiltered_dir / f"{model_name}_bucket_returns.csv")
        filtered_bucket = pd.read_csv(filtered_dir / f"{model_name}_bucket_returns.csv")
        
        # Top 5-15 每日收益
        unfiltered_top5_15 = unfiltered_bucket['top_5_15_return'].dropna() / 100.0
        filtered_top5_15 = filtered_bucket['top_5_15_return'].dropna() / 100.0
        
        print(f"\n每日收益统计（Overlap，249个交易日）:")
        print(f"{'指标':<25} {'未过滤':<20} {'过滤后':<20} {'变化':<15}")
        print("-"*80)
        
        # 平均收益
        unfiltered_avg = unfiltered_top5_15.mean() * 100
        filtered_avg = filtered_top5_15.mean() * 100
        change_avg = filtered_avg - unfiltered_avg
        print(f"{'平均收益 (%)':<25} {unfiltered_avg:<20.4f} {filtered_avg:<20.4f} {change_avg:>+14.4f}%")
        
        # 中位数收益
        unfiltered_median = unfiltered_top5_15.median() * 100
        filtered_median = filtered_top5_15.median() * 100
        change_median = filtered_median - unfiltered_median
        print(f"{'中位数收益 (%)':<25} {unfiltered_median:<20.4f} {filtered_median:<20.4f} {change_median:>+14.4f}%")
        
        # 标准差
        unfiltered_std = unfiltered_top5_15.std() * 100
        filtered_std = filtered_top5_15.std() * 100
        change_std = filtered_std - unfiltered_std
        print(f"{'标准差 (%)':<25} {unfiltered_std:<20.4f} {filtered_std:<20.4f} {change_std:>+14.4f}%")
        
        # Overlap 胜率
        unfiltered_wr = (unfiltered_top5_15 > 0).mean() * 100
        filtered_wr = (filtered_top5_15 > 0).mean() * 100
        change_wr = filtered_wr - unfiltered_wr
        print(f"{'Overlap 胜率 (%)':<25} {unfiltered_wr:<20.2f} {filtered_wr:<20.2f} {change_wr:>+14.2f}%")
        
        # Sharpe Ratio (年化)
        unfiltered_sharpe = (unfiltered_top5_15.mean() / unfiltered_top5_15.std()) * np.sqrt(252) if unfiltered_top5_15.std() > 0 else 0
        filtered_sharpe = (filtered_top5_15.mean() / filtered_top5_15.std()) * np.sqrt(252) if filtered_top5_15.std() > 0 else 0
        change_sharpe = filtered_sharpe - unfiltered_sharpe
        print(f"{'Sharpe Ratio (年化)':<25} {unfiltered_sharpe:<20.4f} {filtered_sharpe:<20.4f} {change_sharpe:>+14.4f}")
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        continue
    
    # 读取Non-Overlap累积收益
    try:
        unfiltered_nonoverlap = pd.read_csv(unfiltered_dir / f"{model_name}_top5_15_rebalance10d_accumulated.csv")
        filtered_nonoverlap = pd.read_csv(filtered_dir / f"{model_name}_top5_15_rebalance10d_accumulated.csv")
        
        print(f"\nNon-Overlap 回测统计（每10天一期，共25期）:")
        print(f"{'指标':<25} {'未过滤':<20} {'过滤后':<20} {'变化':<15}")
        print("-"*80)
        
        # 累积收益
        unfiltered_cum = unfiltered_nonoverlap['acc_return'].iloc[-1] * 100 if 'acc_return' in unfiltered_nonoverlap.columns else 0
        filtered_cum = filtered_nonoverlap['acc_return'].iloc[-1] * 100 if 'acc_return' in filtered_nonoverlap.columns else 0
        change_cum = filtered_cum - unfiltered_cum
        print(f"{'累积收益 (%)':<25} {unfiltered_cum:<20.4f} {filtered_cum:<20.4f} {change_cum:>+14.4f}%")
        
        # 平均期间收益
        unfiltered_period_avg = unfiltered_nonoverlap['top_gross_return'].mean() * 100 if 'top_gross_return' in unfiltered_nonoverlap.columns else 0
        filtered_period_avg = filtered_nonoverlap['top_gross_return'].mean() * 100 if 'top_gross_return' in filtered_nonoverlap.columns else 0
        change_period_avg = filtered_period_avg - unfiltered_period_avg
        print(f"{'平均期间收益 (%)':<25} {unfiltered_period_avg:<20.4f} {filtered_period_avg:<20.4f} {change_period_avg:>+14.4f}%")
        
        # Non-Overlap 胜率
        unfiltered_nonoverlap_wr = (unfiltered_nonoverlap['top_gross_return'] > 0).mean() * 100 if 'top_gross_return' in unfiltered_nonoverlap.columns else 0
        filtered_nonoverlap_wr = (filtered_nonoverlap['top_gross_return'] > 0).mean() * 100 if 'top_gross_return' in filtered_nonoverlap.columns else 0
        change_nonoverlap_wr = filtered_nonoverlap_wr - unfiltered_nonoverlap_wr
        print(f"{'Non-Overlap 胜率 (%)':<25} {unfiltered_nonoverlap_wr:<20.2f} {filtered_nonoverlap_wr:<20.2f} {change_nonoverlap_wr:>+14.2f}%")
        
        # 最大回撤
        if 'acc_return' in unfiltered_nonoverlap.columns:
            unfiltered_cum_returns = (1 + unfiltered_nonoverlap['top_gross_return']).cumprod()
            unfiltered_max_dd = ((unfiltered_cum_returns / unfiltered_cum_returns.expanding().max()) - 1).min() * 100
            
            filtered_cum_returns = (1 + filtered_nonoverlap['top_gross_return']).cumprod()
            filtered_max_dd = ((filtered_cum_returns / filtered_cum_returns.expanding().max()) - 1).min() * 100
            
            change_max_dd = filtered_max_dd - unfiltered_max_dd
            print(f"{'最大回撤 (%)':<25} {unfiltered_max_dd:<20.4f} {filtered_max_dd:<20.4f} {change_max_dd:>+14.4f}%")
        
    except FileNotFoundError as e:
        print(f"Non-Overlap 文件未找到: {e}")

print("\n" + "="*80)
print("【说明】")
print("-"*80)
print("过滤规则：从 Top 15 中移除：")
print("1. 波动率最高的2只股票（使用 hist_vol_40d 或 vol_ratio_20d）")
print("2. 成交量最差的2只股票（使用 obv_momentum_60d 或 liquid_momentum）")
print("\n结果：Top 5-15 过滤后剩余约11只股票（15 - 2 - 2 = 11）")
print("="*80)
