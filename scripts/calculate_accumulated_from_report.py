#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 report_df.csv 计算 CatBoost 和 LambdaRank 的累计收益
基于非重叠回测：每10天一期，共25期
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

def calculate_accumulated_from_avg_return(avg_return_per_period, num_periods=25):
    """
    从平均每期收益计算累计收益（复利）
    
    Args:
        avg_return_per_period: 平均每期收益（小数，如0.02表示2%）
        num_periods: 期数（默认25期，对应一年252交易日/10）
    
    Returns:
        最终累计收益（小数）
    """
    # 假设每期收益等于平均收益（简化计算）
    # 实际应该使用真实的时间序列，但这里用平均收益估算
    return (1.0 + avg_return_per_period) ** num_periods - 1.0

def main():
    # Find latest run with report_df
    result_dirs = list(Path("results/t10_time_split_80_20").glob("run_*"))
    if not result_dirs:
        print("No results directory found")
        return
    
    latest_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
    report_file = latest_dir / "report_df.csv"
    
    if not report_file.exists():
        print(f"report_df.csv not found in {latest_dir}")
        return
    
    print(f"Reading: {latest_dir.name}/report_df.csv")
    df = pd.read_csv(report_file)
    
    models = ['catboost', 'lambdarank']
    num_periods = 25  # 非重叠回测：252交易日/10 ≈ 25期
    
    print("\n" + "="*80)
    print("CatBoost 和 LambdaRank 累计收益计算（基于非重叠回测）")
    print("="*80)
    
    for model_name in models:
        model_data = df[df['Model'] == model_name]
        if model_data.empty:
            print(f"\n{model_name}: Not found in report")
            continue
        
        row = model_data.iloc[0]
        avg_return_net = row['avg_top_return_net']  # 小数格式
        
        print(f"\n{'-'*80}")
        print(f"{model_name.upper()}")
        print(f"{'-'*80}")
        print(f"平均每期收益 (Net): {avg_return_net*100:.4f}%")
        print(f"Sharpe Ratio: {row['top_sharpe_net']:.4f}")
        print(f"胜率: {row['win_rate']:.2%}")
        print(f"期数: {num_periods} (每10天一期)")
        
        # 估算累计收益（使用平均收益）
        # 注意：这是简化估算，实际应该使用真实的时间序列
        estimated_cum = calculate_accumulated_from_avg_return(avg_return_net, num_periods)
        print(f"\n估算累计收益 (基于平均收益): {estimated_cum*100:.2f}%")
        print(f"注意: 这是基于平均收益的估算，实际累计收益可能因波动而不同")
    
    print(f"\n{'='*80}")
    print("要获得准确的累计收益，请查看时间序列文件:")
    print(f"  - catboost_top20_timeseries.csv")
    print(f"  - lambdarank_top20_timeseries.csv")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
