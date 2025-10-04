#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示如何应用 forward-fill 来减少样本损失
"""

import pandas as pd
import numpy as np

# 定义长周期因子
LONG_LOOKBACK_FACTORS = {
    'max_lottery_factor': 365,
    'near_52w_high': 252,
    'overnight_intraday_gap': 180,
}

def apply_smart_forward_fill(factors_df, max_fill_days=30):
    """
    智能前向填充：仅对长周期因子的warm-up期进行填充

    Args:
        factors_df: 因子DataFrame，必须有(date, ticker)的MultiIndex
        max_fill_days: 每个因子最多前向填充的天数（防止过度填充）

    Returns:
        填充后的DataFrame和统计信息
    """
    if not isinstance(factors_df.index, pd.MultiIndex):
        print("Warning: factors_df must have MultiIndex(date, ticker)")
        return factors_df, {}

    stats = {}
    filled_df = factors_df.copy()

    # 按ticker分组处理
    for ticker in filled_df.index.get_level_values('ticker').unique():
        ticker_data = filled_df.xs(ticker, level='ticker')

        # 对每个长周期因子进行填充
        for factor_name, lookback_days in LONG_LOOKBACK_FACTORS.items():
            if factor_name not in ticker_data.columns:
                continue

            factor_series = ticker_data[factor_name]

            # 统计填充前的NaN数量
            nan_before = factor_series.isna().sum()

            if nan_before == 0:
                continue

            # 前向填充（但限制填充范围）
            # 1. 先找到第一个非NaN值
            first_valid_idx = factor_series.first_valid_index()

            if first_valid_idx is None:
                # 全是NaN，用0填充
                filled_df.loc[(slice(None), ticker), factor_name] = 0
                stats[f'{ticker}_{factor_name}'] = {
                    'method': 'zero_fill',
                    'filled': nan_before
                }
                continue

            # 2. 对warm-up期（第一个有效值之前）用第一个有效值填充
            first_valid_value = factor_series.loc[first_valid_idx]
            mask = (ticker_data.index < first_valid_idx) & factor_series.isna()

            if mask.sum() > 0:
                filled_df.loc[(mask.index, ticker), factor_name] = first_valid_value

                stats[f'{ticker}_{factor_name}'] = {
                    'method': 'backward_fill_from_first_valid',
                    'filled': mask.sum(),
                    'first_valid_date': first_valid_idx,
                    'fill_value': first_valid_value
                }

            # 3. 对中间的NaN用前向填充（限制填充天数）
            # 使用pandas的fillna with limit
            factor_filled = filled_df.loc[(slice(None), ticker), factor_name].fillna(
                method='ffill',
                limit=max_fill_days
            )
            filled_df.loc[(slice(None), ticker), factor_name] = factor_filled

    # 全局统计
    total_stats = {
        'total_factors_processed': len(LONG_LOOKBACK_FACTORS),
        'individual_fills': stats,
        'rows_before': len(factors_df),
        'rows_after': len(filled_df),
        'nan_before': factors_df.isna().sum().sum(),
        'nan_after': filled_df.isna().sum().sum()
    }

    return filled_df, total_stats


# 测试示例
if __name__ == '__main__':
    # 模拟数据
    dates = pd.date_range('2021-01-01', periods=750, freq='B')  # 3年工作日
    tickers = [f'STOCK{i:02d}' for i in range(30)]

    # 创建MultiIndex
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])

    # 创建模拟因子数据
    np.random.seed(42)
    factors_data = {
        'momentum_10d_ex1': np.random.randn(len(index)),
        'rsi_7': np.random.randn(len(index)),
        'max_lottery_factor': np.random.randn(len(index)),
        'near_52w_high': np.random.randn(len(index)),
        'overnight_intraday_gap': np.random.randn(len(index)),
    }

    df = pd.DataFrame(factors_data, index=index)

    # 模拟长周期因子的NaN（前N天为NaN）
    for ticker in tickers:
        df.loc[(dates[:365], ticker), 'max_lottery_factor'] = np.nan
        df.loc[(dates[:252], ticker), 'near_52w_high'] = np.nan
        df.loc[(dates[:180], ticker), 'overnight_intraday_gap'] = np.nan

    print("=== 填充前 ===")
    print(f"总样本数: {len(df)}")
    print(f"NaN数量: {df.isna().sum().sum()}")
    print(f"完整样本数 (dropna后): {len(df.dropna())}")
    print()

    # 应用智能填充
    filled_df, stats = apply_smart_forward_fill(df, max_fill_days=30)

    print("=== 填充后 ===")
    print(f"总样本数: {len(filled_df)}")
    print(f"NaN数量: {filled_df.isna().sum().sum()}")
    print(f"完整样本数 (dropna后): {len(filled_df.dropna())}")
    print()

    print("=== 样本提升 ===")
    before_samples = len(df.dropna())
    after_samples = len(filled_df.dropna())
    print(f"填充前可用样本: {before_samples:,}")
    print(f"填充后可用样本: {after_samples:,}")
    print(f"样本增加: {after_samples - before_samples:,} ({(after_samples/before_samples - 1)*100:.1f}%)")
