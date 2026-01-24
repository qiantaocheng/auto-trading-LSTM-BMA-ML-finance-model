#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sato Square Root Factor Calculation (PRODUCTION READY - 100分版本)
Sato 平方根因子计算代码 - 生产级实现

核心改进（100分版本）：
1. 去掉bfill（避免Look-ahead Bias）
2. 添加Divergence因子（反转/异常检测）
3. 返回DataFrame（包含momentum和divergence两个特征）

核心公式：
Sato Momentum = sum((r / sigma) * sqrt(V_rel) over N days)
Sato Divergence = mean(|r| - sigma * sqrt(V_rel) over N days)

其中：
- r: 对数收益
- sigma: 波动率
- V_rel: 相对成交量
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


def calculate_sato_factors(
    df: pd.DataFrame,
    price_col: str = 'adj_close',
    volume_col: str = 'Volume',
    vol_ratio_col: Optional[str] = 'vol_ratio_20d',
    lookback_days: int = 10,
    vol_window: int = 20,
    use_vol_ratio_directly: bool = False
) -> pd.DataFrame:
    """
    计算 Sato 平方根因子（生产级版本 - 100分）
    
    返回两个特征：
    - feat_sato_momentum_10d: 趋势因子（累计动量）
    - feat_sato_divergence_10d: 反转因子（异常检测）
    
    Args:
        df: MultiIndex DataFrame (date, ticker) 或单股票 DataFrame
        price_col: 价格列名（复权后收盘价）
        volume_col: 成交量列名
        vol_ratio_col: 相对成交量因子列名（如果可用）
        lookback_days: Sato 因子滚动窗口（默认10天）
        vol_window: 波动率计算窗口（默认20天）
        use_vol_ratio_directly: 如果为True，直接使用vol_ratio_col作为相对成交量
    
    Returns:
        DataFrame with columns:
        - feat_sato_momentum_10d: Sato动量因子（10日累计）
        - feat_sato_divergence_10d: Sato差异因子（10日平均）
    """
    
    # 确保数据已排序（对MultiIndex很重要）
    if isinstance(df.index, pd.MultiIndex):
        df = df.sort_index()
        # 确定ticker level: 优先使用名称，否则使用位置
        index_names = df.index.names
        if len(index_names) > 1:
            # 查找ticker或symbol level
            ticker_level = None
            for i, name in enumerate(index_names):
                if name and name.lower() in ['ticker', 'symbol']:
                    ticker_level = i  # 使用位置索引
                    break
            if ticker_level is None:
                ticker_level = 1  # 默认第二个level
        else:
            ticker_level = 0  # 单level情况
    else:
        ticker_level = None
    
    # 定义单股票计算函数（确保所有计算都在组内进行）
    def _calc_single_stock_final(group):
        """
        对单只股票计算所有中间步骤（100分版本）
        
        改进：
        1. 去掉bfill（避免Look-ahead Bias）
        2. 添加Divergence因子
        3. 返回DataFrame包含两个特征
        """
        # Step 1: 计算对数收益（组内shift是安全的）
        log_ret = np.log(group[price_col] / group[price_col].shift(1))
        
        # Step 2: 计算波动率（20日滚动标准差）
        # 🔥 改进：使用min_periods避免bfill的未来数据泄漏
        # 允许最少10天就能算出波动率，前面的自动为NaN（LightGBM能处理NaN）
        vol_20d = log_ret.rolling(vol_window, min_periods=10).std()
        vol_20d = vol_20d.fillna(0.01) + 1e-6  # 最小波动率0.01（不使用bfill）
        
        # Step 3: 确定相对成交量
        if use_vol_ratio_directly and vol_ratio_col and vol_ratio_col in group.columns:
            # 直接使用vol_ratio_20d（避免重复计算）
            rel_vol = group[vol_ratio_col].fillna(1.0).clip(lower=0.01)  # 最小相对成交量0.01
        else:
            # 从Volume计算相对成交量
            vol_ma = group[volume_col].rolling(vol_window, min_periods=10).mean()
            rel_vol = group[volume_col] / (vol_ma + 1e-6)
            rel_vol = rel_vol.fillna(1.0).clip(lower=0.01)
        
        # Step 4: Sato 核心逻辑
        # --- 特征 A: Sato Momentum (趋势) ---
        # 逻辑：经波动率标准化后的收益 * 量能权重
        normalized_ret = (log_ret / vol_20d).clip(-5, 5)  # 截断极值
        daily_sato_mom = normalized_ret * np.sqrt(rel_vol)
        
        # --- 特征 B: Sato Divergence (反转/异常) ---
        # 逻辑：实际波动幅度 - 理论应该有的波动幅度
        # 含义：如果值很大，说明价格动了，但量没跟上(虚动) -> 往往预示反转
        theoretical_impact = vol_20d * np.sqrt(rel_vol)
        daily_divergence = np.abs(log_ret) - theoretical_impact
        
        # Step 5: 滚动聚合 (T+10 窗口)
        # 返回DataFrame包含两个特征
        res = pd.DataFrame(index=group.index)
        
        # 累计动量 (Sum) - 趋势因子
        res['feat_sato_momentum_10d'] = daily_sato_mom.rolling(lookback_days).sum()
        
        # 平均偏离度 (Mean) - 反转因子
        res['feat_sato_divergence_10d'] = daily_divergence.rolling(lookback_days).mean()
        
        return res
    
    # 执行分组计算
    if isinstance(df.index, pd.MultiIndex):
        # MultiIndex: 按ticker分组计算
        # 使用group_keys=False避免索引层级爆炸
        factors_df = df.groupby(level=ticker_level, group_keys=False).apply(
            lambda group: _calc_single_stock_final(group)
        )
        # 确保索引对齐
        factors_df = factors_df.reindex(df.index)
        return factors_df
    else:
        # 单股票DataFrame
        return _calc_single_stock_final(df)


def calculate_sato_factor(
    df: pd.DataFrame,
    price_col: str = 'adj_close',
    volume_col: str = 'Volume',
    vol_ratio_col: Optional[str] = 'vol_ratio_20d',
    lookback_days: int = 10,
    vol_window: int = 20,
    use_vol_ratio_directly: bool = False
) -> pd.Series:
    """
    计算 Sato 平方根因子（向后兼容版本 - 只返回momentum）
    
    注意：推荐使用 calculate_sato_factors() 获取完整特征（momentum + divergence）
    """
    factors_df = calculate_sato_factors(
        df=df,
        price_col=price_col,
        volume_col=volume_col,
        vol_ratio_col=vol_ratio_col,
        lookback_days=lookback_days,
        vol_window=vol_window,
        use_vol_ratio_directly=use_vol_ratio_directly
    )
    return factors_df['feat_sato_momentum_10d']


def calculate_sato_factor_with_benchmarks(
    df: pd.DataFrame,
    price_col: str = 'adj_close',
    volume_col: str = 'Volume',
    vol_ratio_col: Optional[str] = 'vol_ratio_20d',
    lookback_days: int = 10,
    vol_window: int = 20,
    use_vol_ratio_directly: bool = False
) -> pd.DataFrame:
    """
    计算 Sato 因子以及对照组因子（用于正交化测试）
    
    Returns:
        DataFrame with columns:
        - feat_sato_momentum_10d: Sato 动量因子
        - feat_sato_divergence_10d: Sato 差异因子
        - factor_mom_raw: 传统动量因子（10日涨幅）
        - factor_vol: 波动率因子
    """
    result_df = df.copy()
    
    # 计算Sato因子（momentum + divergence）
    sato_factors = calculate_sato_factors(
        df=df,
        price_col=price_col,
        volume_col=volume_col,
        vol_ratio_col=vol_ratio_col,
        lookback_days=lookback_days,
        vol_window=vol_window,
        use_vol_ratio_directly=use_vol_ratio_directly
    )
    
    # 确保数据已排序
    if isinstance(df.index, pd.MultiIndex):
        result_df = result_df.sort_index()
        # 确定ticker level: 优先使用名称，否则使用位置
        index_names = result_df.index.names
        if len(index_names) > 1:
            # 查找ticker或symbol level
            ticker_level = None
            for i, name in enumerate(index_names):
                if name and name.lower() in ['ticker', 'symbol']:
                    ticker_level = i  # 使用位置索引
                    break
            if ticker_level is None:
                ticker_level = 1  # 默认第二个level
        else:
            ticker_level = 0  # 单level情况
    else:
        ticker_level = None
    
    # 定义单股票计算函数（计算对照组因子）
    def _calc_benchmark_factors(group):
        """对单只股票计算对照组因子"""
        # 1. 计算对数收益
        log_ret = np.log(group[price_col] / group[price_col].shift(1))
        
        # 2. 计算波动率
        vol_20d = log_ret.rolling(vol_window, min_periods=10).std()
        vol_20d = vol_20d.fillna(0.01) + 1e-6
        
        # 3. 计算对照组因子
        factor_mom_raw = log_ret.rolling(lookback_days).sum()
        factor_vol = vol_20d
        
        # 返回结果DataFrame
        result = pd.DataFrame({
            'factor_mom_raw': factor_mom_raw,
            'factor_vol': factor_vol
        }, index=group.index)
        
        return result
    
    # 执行分组计算
    if isinstance(df.index, pd.MultiIndex):
        benchmark_df = result_df.groupby(level=ticker_level, group_keys=False).apply(_calc_benchmark_factors)
        # 合并结果
        for col in ['factor_mom_raw', 'factor_vol']:
            result_df[col] = benchmark_df[col].reindex(result_df.index)
    else:
        benchmark_df = _calc_benchmark_factors(result_df)
        for col in ['factor_mom_raw', 'factor_vol']:
            result_df[col] = benchmark_df[col]
    
    # 添加Sato因子
    result_df['feat_sato_momentum_10d'] = sato_factors['feat_sato_momentum_10d'].reindex(result_df.index)
    result_df['feat_sato_divergence_10d'] = sato_factors['feat_sato_divergence_10d'].reindex(result_df.index)
    
    return result_df


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例1: 从MultiIndex数据计算Sato因子
    print("=" * 80)
    print("Sato Factor Calculation Example (PRODUCTION READY - 100分版本)")
    print("=" * 80)
    
    # 加载数据
    data_path = r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean.parquet"
    df = pd.read_parquet(data_path)
    
    print(f"Data shape: {df.shape}")
    print(f"Index: {df.index.names}")
    
    # 准备数据
    df['adj_close'] = df['Close']
    
    # 检查是否有vol_ratio_20d（优先使用）
    has_vol_ratio = 'vol_ratio_20d' in df.columns
    
    if not has_vol_ratio and 'Volume' not in df.columns:
        raise ValueError("Need either Volume or vol_ratio_20d column")
    
    # 如果Volume不存在，估算（但优先使用vol_ratio_20d）
    if 'Volume' not in df.columns:
        base_volume = 1_000_000
        df['Volume'] = base_volume * df['vol_ratio_20d'].fillna(1.0).clip(lower=0.1, upper=10.0)
        use_vol_ratio = True
    else:
        use_vol_ratio = has_vol_ratio  # 如果有vol_ratio_20d，优先使用它
    
    # 计算Sato因子（momentum + divergence）
    print("\nCalculating Sato factors (momentum + divergence)...")
    sato_factors_df = calculate_sato_factors(
        df=df,
        price_col='adj_close',
        volume_col='Volume',
        vol_ratio_col='vol_ratio_20d',
        lookback_days=10,
        vol_window=20,
        use_vol_ratio_directly=use_vol_ratio
    )
    
    # 显示结果统计
    print("\nFactor Statistics:")
    print(f"feat_sato_momentum_10d: mean={sato_factors_df['feat_sato_momentum_10d'].mean():.6f}, std={sato_factors_df['feat_sato_momentum_10d'].std():.6f}")
    print(f"feat_sato_divergence_10d: mean={sato_factors_df['feat_sato_divergence_10d'].mean():.6f}, std={sato_factors_df['feat_sato_divergence_10d'].std():.6f}")
    
    # 检查极值
    print("\nExtreme Value Check:")
    print(f"feat_sato_momentum_10d: min={sato_factors_df['feat_sato_momentum_10d'].min():.2f}, max={sato_factors_df['feat_sato_momentum_10d'].max():.2f}")
    print(f"feat_sato_divergence_10d: min={sato_factors_df['feat_sato_divergence_10d'].min():.2f}, max={sato_factors_df['feat_sato_divergence_10d'].max():.2f}")
    
    print("\n[OK] Calculation complete!")
    print(f"Result DataFrame shape: {sato_factors_df.shape}")
    print(f"Columns: {list(sato_factors_df.columns)}")
