#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成完整的指标报告（包括EWMA后的所有指标）"""

import pandas as pd
import numpy as np
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20\run_20260120_042659")  # Update with latest run

print("="*80)
print("完整指标报告（EWMA平滑后）")
print("="*80)

models = ['catboost', 'lambdarank', 'ridge_stacking']

for model_name in models:
    print(f"\n【{model_name.upper()}】")
    print("="*80)
    
    # Overlap metrics (daily)
    try:
        bucket_df = pd.read_csv(run_dir / f"{model_name}_bucket_returns.csv")
        
        # Top 5-15
        if 'top_5_15_return' in bucket_df.columns:
            top5_15 = bucket_df['top_5_15_return'].dropna() / 100.0
            
            print(f"\n【Overlap 指标 - Top 5-15（249个交易日）】")
            print("-"*80)
            print(f"平均收益: {top5_15.mean()*100:.4f}%")
            print(f"中位数收益: {top5_15.median()*100:.4f}%")
            print(f"标准差: {top5_15.std()*100:.4f}%")
            print(f"Overlap 胜率: {(top5_15 > 0).mean()*100:.2f}%")
            sharpe = (top5_15.mean() / top5_15.std()) * np.sqrt(252) if top5_15.std() > 0 else 0
            print(f"Sharpe Ratio (年化): {sharpe:.4f}")
    except FileNotFoundError:
        print(f"Bucket returns文件未找到: {model_name}_bucket_returns.csv")
    
    # Non-Overlap metrics
    try:
        nonoverlap_df = pd.read_csv(run_dir / f"{model_name}_top5_15_rebalance10d_accumulated.csv")
        
        print(f"\n【Non-Overlap 指标 - Top 5-15（25期，每10天）】")
        print("-"*80)
        
        if 'top_gross_return' in nonoverlap_df.columns:
            period_returns = nonoverlap_df['top_gross_return']
            print(f"平均期间收益: {period_returns.mean()*100:.4f}%")
            print(f"中位数期间收益: {period_returns.median()*100:.4f}%")
            print(f"标准差: {period_returns.std()*100:.4f}%")
            print(f"Non-Overlap 胜率: {(period_returns > 0).mean()*100:.2f}%")
            
            # Sharpe (based on periods)
            if period_returns.std() > 0:
                period_sharpe = (period_returns.mean() / period_returns.std()) * np.sqrt(25)
                print(f"Sharpe Ratio (基于期间): {period_sharpe:.4f}")
        
        if 'acc_return' in nonoverlap_df.columns:
            final_acc = nonoverlap_df['acc_return'].iloc[-1] * 100
            print(f"\n累积收益: {final_acc:.4f}%")
            
            # Max drawdown
            if 'drawdown' in nonoverlap_df.columns:
                max_dd = nonoverlap_df['drawdown'].min()
                print(f"最大回撤: {max_dd:.4f}%")
            else:
                # Calculate if not present
                cum_returns = (1 + period_returns).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns / running_max - 1) * 100
                max_dd = drawdown.min()
                print(f"最大回撤: {max_dd:.4f}%")
            
            # Annualized return
            total_days = len(nonoverlap_df) * 10
            annualized_return = ((1 + final_acc/100) ** (252 / total_days) - 1) * 100
            print(f"年化收益: {annualized_return:.4f}%")
            
    except FileNotFoundError:
        print(f"Non-Overlap文件未找到: {model_name}_top5_15_rebalance10d_accumulated.csv")

print("\n" + "="*80)
print("【说明】")
print("-"*80)
print("所有预测已应用EWMA平滑（3天EMA: 0.6*S_t + 0.3*S_{t-1} + 0.1*S_{t-2}）")
print("="*80)
