#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证target计算是否正确，检查价格数据
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def verify_target_calculation():
    """验证target计算"""
    print("=" * 80)
    print("验证Target计算")
    print("=" * 80)
    
    data_file = "data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet"
    
    print(f"\n[1] 加载数据文件...")
    df = pd.read_parquet(data_file)
    print(f"   [OK] 数据形状: {df.shape}")
    
    # 检查Close列
    if 'Close' not in df.columns:
        print(f"\n[ERROR] 数据文件中没有Close列")
        return
    
    # 选择测试期的几个ticker进行验证
    test_start = pd.Timestamp('2025-01-02')
    test_end = pd.Timestamp('2025-12-30')
    
    test_dates = df.index.get_level_values('date')
    test_mask = (test_dates >= test_start) & (test_dates <= test_end)
    test_df = df[test_mask].copy()
    
    # 选择几个ticker验证target计算
    tickers = test_df.index.get_level_values('ticker').unique()[:5]
    
    print(f"\n[2] 验证Target计算（选择5个ticker）:")
    print("-" * 80)
    
    horizon = 10
    
    for ticker in tickers:
        ticker_data = test_df.loc[test_df.index.get_level_values('ticker') == ticker].copy()
        ticker_data = ticker_data.sort_index()
        
        if len(ticker_data) < horizon + 5:
            continue
        
        # 手动计算target
        close_prices = ticker_data['Close'].values
        dates = ticker_data.index.get_level_values('date').values
        
        # 计算T+10收益: (Close[t+10] - Close[t]) / Close[t]
        manual_targets = []
        for i in range(len(close_prices) - horizon):
            if close_prices[i] > 0 and close_prices[i + horizon] > 0:
                manual_target = (close_prices[i + horizon] - close_prices[i]) / close_prices[i]
                manual_targets.append({
                    'date': dates[i],
                    'manual_target': manual_target,
                    'data_target': ticker_data.iloc[i]['target'],
                    'close_t': close_prices[i],
                    'close_t10': close_prices[i + horizon]
                })
        
        if manual_targets:
            manual_df = pd.DataFrame(manual_targets)
            
            # 比较手动计算和数据文件中的target
            diff = (manual_df['manual_target'] - manual_df['data_target']).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            print(f"\n  Ticker: {ticker}")
            print(f"    样本数: {len(manual_df)}")
            print(f"    最大差异: {max_diff:.6f}")
            print(f"    平均差异: {mean_diff:.6f}")
            
            if max_diff > 1e-5:
                print(f"    [WARN] 发现差异！")
                print(f"    前5个样本:")
                for idx, row in manual_df.head(5).iterrows():
                    print(f"      Date {row['date'].date()}: manual={row['manual_target']:.6f}, "
                          f"data={row['data_target']:.6f}, diff={abs(row['manual_target']-row['data_target']):.6f}")
            
            # 检查是否有异常高的收益
            high_returns = manual_df[manual_df['manual_target'] > 0.3]
            if len(high_returns) > 0:
                print(f"    [WARN] 发现异常高收益（>30%）: {len(high_returns)}个")
                print(f"    最高收益: {manual_df['manual_target'].max():.4f} ({manual_df['manual_target'].max()*100:.2f}%)")
                print(f"    最高收益样本:")
                max_idx = manual_df['manual_target'].idxmax()
                max_row = manual_df.loc[max_idx]
                print(f"      Date: {max_row['date'].date()}")
                print(f"      Close[t]: {max_row['close_t']:.2f}")
                print(f"      Close[t+10]: {max_row['close_t10']:.2f}")
                print(f"      Return: {max_row['manual_target']:.4f} ({max_row['manual_target']*100:.2f}%)")
    
    # 检查价格数据质量
    print(f"\n[3] 检查价格数据质量:")
    print("-" * 80)
    
    close_prices = df['Close'].dropna()
    print(f"  Close价格统计:")
    print(f"    有效值: {len(close_prices):,}")
    print(f"    均值: ${close_prices.mean():.2f}")
    print(f"    中位数: ${close_prices.median():.2f}")
    print(f"    最小值: ${close_prices.min():.2f}")
    print(f"    最大值: ${close_prices.max():.2f}")
    
    # 检查是否有异常价格
    price_changes = df.groupby(level='ticker')['Close'].pct_change()
    extreme_changes = price_changes[(price_changes.abs() > 0.5) & price_changes.notna()]
    
    print(f"\n  异常价格变动（>50%）: {len(extreme_changes):,} ({len(extreme_changes)/len(price_changes.dropna())*100:.2f}%)")
    
    if len(extreme_changes) > 0:
        print(f"    最高变动: {extreme_changes.max():.4f} ({extreme_changes.max()*100:.2f}%)")
        print(f"    最低变动: {extreme_changes.min():.4f} ({extreme_changes.min()*100:.2f}%)")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    verify_target_calculation()
