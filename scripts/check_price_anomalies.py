#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查价格数据中的异常�?"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def check_price_anomalies():
    """检查价格异常�?""
    print("=" * 80)
    print("检查价格数据异常�?)
    print("=" * 80)
    
    data_file = "data/factor_exports/polygon_factors_all_2021_2026_T5_final.parquet"
    
    print(f"\n[1] 加载数据文件...")
    df = pd.read_parquet(data_file)
    print(f"   [OK] 数据形状: {df.shape}")
    
    # 检查Close价格
    close = df['Close']
    
    print(f"\n[2] Close价格统计:")
    print("-" * 80)
    print(f"  有效�? {close.notna().sum():,}")
    print(f"  均�? ${close.mean():.2f}")
    print(f"  中位�? ${close.median():.2f}")
    print(f"  标准�? ${close.std():.2f}")
    print(f"  最小�? ${close.min():.2f}")
    print(f"  最大�? ${close.max():.2f}")
    
    # 检查异常高价格
    high_prices = close[close > 10000]  # > $10,000
    print(f"\n  > $10,000: {len(high_prices):,} ({len(high_prices)/len(close)*100:.2f}%)")
    
    if len(high_prices) > 0:
        print(f"  最高价�? ${high_prices.max():.2f}")
        print(f"  这些高价格的ticker:")
        high_price_data = df[df['Close'] > 10000]
        high_price_tickers = high_price_data.index.get_level_values('ticker').unique()
        for ticker in high_price_tickers[:10]:
            ticker_prices = df.loc[df.index.get_level_values('ticker') == ticker, 'Close']
            print(f"    {ticker}: 最�?${ticker_prices.max():.2f}, 均�?${ticker_prices.mean():.2f}")
    
    # 检查异常低价格
    low_prices = close[(close > 0) & (close < 0.01)]  # < $0.01
    print(f"\n  < $0.01: {len(low_prices):,} ({len(low_prices)/len(close)*100:.2f}%)")
    
    # 检查价格变�?    print(f"\n[3] 检查价格变动异�?")
    print("-" * 80)
    
    # 按ticker分组计算日收益率
    price_changes = df.groupby(level='ticker')['Close'].pct_change()
    
    extreme_positive = price_changes[price_changes > 1.0]  # > 100%
    extreme_negative = price_changes[price_changes < -0.9]  # < -90%
    
    print(f"  异常正变动（>100%�? {len(extreme_positive):,}")
    print(f"  异常负变动（<-90%�? {len(extreme_negative):,}")
    
    if len(extreme_positive) > 0:
        print(f"\n  Top 10 异常正变�?")
        top_extreme = extreme_positive.nlargest(10)
        for idx, val in top_extreme.items():
            date, ticker = idx
            print(f"    {date.date()} {ticker}: {val:.4f} ({val*100:.2f}%)")
    
    # 检查这些异常变动是否导致异常高的target
    print(f"\n[4] 检查异常价格变动对target的影�?")
    print("-" * 80)
    
    # 计算10天收益率 - FIX: shift must be per-ticker
    target_calc = (
        df.groupby(level='ticker')['Close']
        .pct_change(10)
        .groupby(level='ticker')
        .shift(-10)
    )
    
    extreme_targets = target_calc[target_calc > 0.5]  # > 50%
    
    print(f"  Target > 50%: {len(extreme_targets):,} ({len(extreme_targets)/len(target_calc.dropna())*100:.2f}%)")
    
    if len(extreme_targets) > 0:
        print(f"  最高target: {extreme_targets.max():.4f} ({extreme_targets.max()*100:.2f}%)")
        print(f"\n  Top 10 异常target样本:")
        top_targets = extreme_targets.nlargest(10)
        for idx, val in top_targets.items():
            date, ticker = idx
            ticker_data = df.loc[(df.index.get_level_values('date') == date) & 
                                 (df.index.get_level_values('ticker') == ticker)]
            if len(ticker_data) > 0:
                close_t = ticker_data['Close'].iloc[0]
                # 找到t+10的价�?                date_plus_10 = pd.Timestamp(date) + pd.Timedelta(days=15)  # 大约10个交易日
                ticker_data_plus10 = df.loc[(df.index.get_level_values('date') <= date_plus_10) &
                                            (df.index.get_level_values('ticker') == ticker)]
                if len(ticker_data_plus10) >= 10:
                    close_t10 = ticker_data_plus10['Close'].iloc[9] if len(ticker_data_plus10) >= 10 else np.nan
                    if not pd.isna(close_t10) and close_t > 0:
                        manual_return = (close_t10 - close_t) / close_t
                        print(f"    {date.date()} {ticker}: target={val:.4f}, "
                              f"Close[t]=${close_t:.2f}, Close[t+10]=${close_t10:.2f}, "
                              f"manual={manual_return:.4f}")
    
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    
    if len(extreme_targets) > 1000:
        print(f"\n[WARN] 发现大量异常target值（>50%�? {len(extreme_targets):,}�?)
        print(f"  这些异常值可能导致Top N收益异常�?)
        print(f"  建议:")
        print(f"    1. 检查价格数据质量（是否有数据错误）")
        print(f"    2. 检查是否有股票拆分/合并等公司行为未正确处理")
        print(f"    3. 考虑对target值进行winsorization（截尾处理）")
        print(f"    4. 检查数据文件生成过程是否有问题")

if __name__ == "__main__":
    check_price_anomalies()
