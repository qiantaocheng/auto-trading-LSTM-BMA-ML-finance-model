#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成过滤后的完整数据汇总报告"""

import pandas as pd
import numpy as np
from pathlib import Path

filtered_dir = Path(r"d:\trade\results\t10_time_split_80_20_filtered\run_20260120_042659")

print("="*80)
print("Top 5-15 过滤后完整数据汇总（移除波动率最高的2只和成交量最差的2只）")
print("="*80)

models = ['catboost', 'lambdarank']

for model_name in models:
    print(f"\n【{model_name.upper()} - Top 5-15 过滤后】")
    print("="*80)
    
    # 读取bucket returns (Overlap)
    try:
        bucket_df = pd.read_csv(filtered_dir / f"{model_name}_bucket_returns.csv")
        top5_15 = bucket_df['top_5_15_return'].dropna() / 100.0
        
        print(f"\n【Overlap 每日收益统计（249个交易日）】")
        print("-"*80)
        print(f"总交易日数: {len(top5_15)}")
        print(f"平均收益: {top5_15.mean()*100:.4f}%")
        print(f"中位数收益: {top5_15.median()*100:.4f}%")
        print(f"标准差: {top5_15.std()*100:.4f}%")
        print(f"最小收益: {top5_15.min()*100:.4f}%")
        print(f"最大收益: {top5_15.max()*100:.4f}%")
        print(f"盈利日数: {(top5_15 > 0).sum()}")
        print(f"亏损日数: {(top5_15 <= 0).sum()}")
        print(f"Overlap 胜率: {(top5_15 > 0).mean()*100:.2f}%")
        
        # Sharpe Ratio
        sharpe = (top5_15.mean() / top5_15.std()) * np.sqrt(252) if top5_15.std() > 0 else 0
        print(f"Sharpe Ratio (年化): {sharpe:.4f}")
        
        # 分位数
        print(f"\n收益分位数:")
        print(f"  25%分位数: {top5_15.quantile(0.25)*100:.4f}%")
        print(f"  50%分位数（中位数）: {top5_15.quantile(0.50)*100:.4f}%")
        print(f"  75%分位数: {top5_15.quantile(0.75)*100:.4f}%")
        print(f"  90%分位数: {top5_15.quantile(0.90)*100:.4f}%")
        print(f"  95%分位数: {top5_15.quantile(0.95)*100:.4f}%")
        
    except FileNotFoundError as e:
        print(f"Bucket returns 文件未找到: {e}")
    
    # 读取Non-Overlap累积收益
    try:
        nonoverlap_df = pd.read_csv(filtered_dir / f"{model_name}_top5_15_rebalance10d_accumulated.csv")
        
        print(f"\n【Non-Overlap 回测统计（每10天一期，共25期）】")
        print("-"*80)
        print(f"总期间数: {len(nonoverlap_df)}")
        
        if 'top_gross_return' in nonoverlap_df.columns:
            period_returns = nonoverlap_df['top_gross_return']
            print(f"平均期间收益: {period_returns.mean()*100:.4f}%")
            print(f"中位数期间收益: {period_returns.median()*100:.4f}%")
            print(f"标准差: {period_returns.std()*100:.4f}%")
            print(f"最小期间收益: {period_returns.min()*100:.4f}%")
            print(f"最大期间收益: {period_returns.max()*100:.4f}%")
            print(f"盈利期间数: {(period_returns > 0).sum()}")
            print(f"亏损期间数: {(period_returns <= 0).sum()}")
            print(f"Non-Overlap 胜率: {(period_returns > 0).mean()*100:.2f}%")
        
        if 'acc_return' in nonoverlap_df.columns:
            final_acc = nonoverlap_df['acc_return'].iloc[-1] * 100
            print(f"\n累积收益: {final_acc:.4f}%")
            
            # 计算最大回撤
            cum_returns = (1 + period_returns).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns / running_max - 1) * 100
            max_dd = drawdown.min()
            print(f"最大回撤: {max_dd:.4f}%")
            
            # 计算年化收益
            total_days = len(nonoverlap_df) * 10
            annualized_return = ((1 + final_acc/100) ** (252 / total_days) - 1) * 100
            print(f"年化收益（估算）: {annualized_return:.4f}%")
            
            # 计算年化Sharpe
            if period_returns.std() > 0:
                period_sharpe = (period_returns.mean() / period_returns.std()) * np.sqrt(25)  # 25期
                print(f"Sharpe Ratio（基于期间）: {period_sharpe:.4f}")
        
    except FileNotFoundError as e:
        print(f"Non-Overlap 文件未找到: {e}")

print("\n" + "="*80)
print("【过滤规则说明】")
print("-"*80)
print("从 Top 15 预测中移除：")
print("1. 波动率最高的2只股票（使用 hist_vol_40d 或 vol_ratio_20d）")
print("2. 成交量最差的2只股票（使用 obv_momentum_60d 或 liquid_momentum）")
print("\n结果：Top 5-15 过滤后剩余约11只股票（15 - 2 - 2 = 11）")
print("="*80)
