#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查原始数据中的极端收益率，排查 CatBoost 9000% 收益率问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def check_extreme_returns(data_file: str, return_threshold: float = 50.0):
    """
    检查数据文件中的极端收益率
    
    Args:
        data_file: 数据文件路径
        return_threshold: 收益率阈值（默认50.0 = 5000%）
    """
    print("=" * 80)
    print(f"检查数据文件: {data_file}")
    print("=" * 80)
    
    # 读取数据
    print("\n读取数据文件...")
    df = pd.read_parquet(data_file)
    print(f"数据形状: {df.shape}")
    print(f"索引类型: {type(df.index)}")
    
    # 检查索引
    if isinstance(df.index, pd.MultiIndex):
        print(f"MultiIndex 级别: {df.index.names}")
        dates = df.index.get_level_values('date')
        tickers = df.index.get_level_values('ticker')
    else:
        print("警告: 不是MultiIndex，尝试查找date和ticker列")
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
        else:
            dates = None
        if 'ticker' in df.columns:
            tickers = df['ticker']
        else:
            tickers = None
    
    # 查找收益率列
    print("\n查找收益率列...")
    return_cols = [col for col in df.columns if 'target' in col.lower() or 'ret' in col.lower() or 'return' in col.lower()]
    print(f"找到收益率相关列: {return_cols}")
    
    if not return_cols:
        print("错误: 未找到收益率列")
        print(f"可用列: {df.columns.tolist()[:30]}")
        return
    
    # 检查每个收益率列
    for col in return_cols:
        print(f"\n{'='*80}")
        print(f"检查列: {col}")
        print(f"{'='*80}")
        
        returns = df[col].copy()
        
        # 移除NaN
        valid_returns = returns.dropna()
        print(f"有效数据点: {len(valid_returns)} / {len(returns)}")
        
        # 统计信息
        print(f"\n收益率统计:")
        print(f"  最小值: {valid_returns.min():.6f}")
        print(f"  最大值: {valid_returns.max():.6f}")
        print(f"  均值: {valid_returns.mean():.6f}")
        print(f"  中位数: {valid_returns.median():.6f}")
        print(f"  标准差: {valid_returns.std():.6f}")
        
        # 转换为百分比（如果是小数形式）
        if valid_returns.abs().max() < 10:
            returns_pct = valid_returns * 100
            print(f"\n(转换为百分比形式)")
        else:
            returns_pct = valid_returns
        
        # 查找极端值
        extreme_mask = returns_pct.abs() >= return_threshold
        extreme_count = extreme_mask.sum()
        
        print(f"\n极端值统计 (>= {return_threshold}%):")
        print(f"  数量: {extreme_count}")
        print(f"  比例: {extreme_count/len(valid_returns)*100:.4f}%")
        
        if extreme_count > 0:
            # 获取极端值记录（先重置索引）
            extreme_df = df[extreme_mask].copy()
            if isinstance(extreme_df.index, pd.MultiIndex):
                extreme_df = extreme_df.reset_index()
            
            # 添加收益率列（使用values避免索引问题）
            extreme_df['return_pct'] = returns_pct[extreme_mask].values
            
            # 按收益率绝对值排序
            extreme_df = extreme_df.sort_values('return_pct', key=abs, ascending=False)
            
            print(f"\nTop 20 极端收益率:")
            print("-" * 80)
            
            display_cols = []
            if 'ticker' in extreme_df.columns:
                display_cols.append('ticker')
            if 'date' in extreme_df.columns:
                display_cols.append('date')
            display_cols.append('return_pct')
            
            # 查找价格列
            price_cols = [c for c in extreme_df.columns if 'close' in c.lower() or 'price' in c.lower()]
            if price_cols:
                display_cols.append(price_cols[0])
            
            top_20 = extreme_df.head(20)[display_cols]
            print(top_20.to_string(index=False))
            
            # 检查低价股
            if price_cols:
                price_col = price_cols[0]
                low_price = extreme_df[extreme_df[price_col] < 1.0]
                very_low_price = extreme_df[extreme_df[price_col] < 0.1]
                
                print(f"\n低价股分析 (< $1):")
                print(f"  数量: {len(low_price)}")
                print(f"  极低价股 (< $0.1): {len(very_low_price)}")
                
                if len(low_price) > 0:
                    print(f"\n低价股样本:")
                    print(low_price[display_cols].head(10).to_string(index=False))
            
            # 按股票代码分组统计
            if 'ticker' in extreme_df.columns:
                print(f"\n极端收益率按股票代码分组 (Top 10):")
                ticker_stats = extreme_df.groupby('ticker').agg({
                    'return_pct': ['count', 'mean', 'min', 'max']
                }).sort_values(('return_pct', 'count'), ascending=False)
                print(ticker_stats.head(10).to_string())
            
            # 按日期分组统计
            if 'date' in extreme_df.columns:
                print(f"\n极端收益率按日期分组 (Top 10):")
                date_stats = extreme_df.groupby('date').agg({
                    'return_pct': ['count', 'mean', 'min', 'max']
                }).sort_values(('return_pct', 'count'), ascending=False)
                print(date_stats.head(10).to_string())
            
            # 保存结果
            output_file = Path(data_file).parent / f"extreme_returns_{col}_{return_threshold}pct.csv"
            extreme_df.to_csv(output_file, index=False)
            print(f"\n极端收益率记录已保存到: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="检查数据中的极端收益率")
    parser.add_argument("--data-file", type=str, 
                       default="data/factor_exports/polygon_factors_all_filtered.parquet",
                       help="数据文件路径")
    parser.add_argument("--threshold", type=float, default=50.0, 
                       help="收益率阈值（默认50.0 = 5000%）")
    
    args = parser.parse_args()
    
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"错误: 数据文件不存在: {data_file}")
        print(f"请提供正确的数据文件路径")
        sys.exit(1)
    
    check_extreme_returns(str(data_file), args.threshold)
