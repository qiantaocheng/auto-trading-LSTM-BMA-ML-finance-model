#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练/预测模式分离修复补丁

修复目标：
1. 训练模式：需要target，会dropna，训练到10-05
2. 预测模式：不需要target，不dropna，用10-10的特征预测10-15

使用方法：
在主模型中添加mode参数，根据mode调用不同的数据准备逻辑
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


def prepare_data_with_mode(
    feature_data: pd.DataFrame,
    mode: str = 'train',
    horizon: int = 5
) -> pd.DataFrame:
    """
    根据模式准备数据

    Args:
        feature_data: 包含特征和Close价格的DataFrame
        mode: 'train' 或 'predict'
        horizon: 预测窗口（天数）

    Returns:
        处理后的DataFrame
    """
    logger.info(f"=" * 80)
    logger.info(f"🔄 数据准备模式: {mode.upper()}")
    logger.info(f"=" * 80)

    if mode == 'train':
        logger.info("📚 训练模式: 计算target并删除无target样本")

        # 训练模式：需要计算target
        if 'Close' not in feature_data.columns:
            raise ValueError("训练模式需要Close价格列来计算target")

        # 计算target
        logger.info(f"   计算T+{horizon}的forward returns...")
        if isinstance(feature_data.index, pd.MultiIndex):
            # MultiIndex format
            target_series = (
                feature_data.groupby(level='ticker')['Close']
                .pct_change(horizon)
                .shift(-horizon)
            )
        else:
            # 如果有ticker列
            if 'ticker' in feature_data.columns:
                target_series = (
                    feature_data.groupby('ticker')['Close']
                    .pct_change(horizon)
                    .shift(-horizon)
                )
            else:
                # 单个ticker
                target_series = (
                    feature_data['Close']
                    .pct_change(horizon)
                    .shift(-horizon)
                )

        feature_data['target'] = target_series

        # 统计target质量
        total_samples = len(feature_data)
        valid_targets = target_series.notna().sum()
        valid_ratio = valid_targets / total_samples

        logger.info(f"   Target统计:")
        logger.info(f"     总样本数: {total_samples}")
        logger.info(f"     有效target: {valid_targets}")
        logger.info(f"     有效率: {valid_ratio:.1%}")

        # Dropna删除没有target的样本
        samples_before = len(feature_data)
        feature_data = feature_data.dropna(subset=['target'])
        samples_after = len(feature_data)
        samples_removed = samples_before - samples_after

        logger.info(f"   Dropna结果:")
        logger.info(f"     删除前: {samples_before} 样本")
        logger.info(f"     删除后: {samples_after} 样本")
        logger.info(f"     移除: {samples_removed} 样本 ({samples_removed/samples_before*100:.1f}%)")

        # 获取训练数据的日期范围
        if isinstance(feature_data.index, pd.MultiIndex) and 'date' in feature_data.index.names:
            min_date = feature_data.index.get_level_values('date').min()
            max_date = feature_data.index.get_level_values('date').max()
            logger.info(f"   训练数据日期范围: {min_date} 到 {max_date}")
        elif 'date' in feature_data.columns:
            min_date = feature_data['date'].min()
            max_date = feature_data['date'].max()
            logger.info(f"   训练数据日期范围: {min_date} 到 {max_date}")

    else:  # mode == 'predict'
        logger.info("🔮 预测模式: 不计算target，保留所有样本")

        # 预测模式：不需要target
        logger.info("   跳过target计算")
        logger.info("   跳过dropna操作")
        logger.info(f"   保留所有 {len(feature_data)} 个样本用于预测")

        # 添加空的target列（如果下游需要）
        feature_data['target'] = np.nan

        # 获取预测数据的日期范围
        if isinstance(feature_data.index, pd.MultiIndex) and 'date' in feature_data.index.names:
            min_date = feature_data.index.get_level_values('date').min()
            max_date = feature_data.index.get_level_values('date').max()
            logger.info(f"   预测数据日期范围: {min_date} 到 {max_date}")
            logger.info(f"   预测目标日期: {max_date + timedelta(days=horizon)}")
        elif 'date' in feature_data.columns:
            min_date = feature_data['date'].min()
            max_date = feature_data['date'].max()
            logger.info(f"   预测数据日期范围: {min_date} 到 {max_date}")
            logger.info(f"   预测目标日期: {max_date + timedelta(days=horizon)}")

    logger.info(f"=" * 80)
    return feature_data


def get_end_date_for_mode(
    end_date: str,
    mode: str = 'train',
    horizon: int = 5
) -> str:
    """
    根据模式调整end_date

    Args:
        end_date: 输入的结束日期
        mode: 'train' 或 'predict'
        horizon: 预测窗口（天数）

    Returns:
        调整后的end_date字符串
    """
    end_dt = pd.to_datetime(end_date)

    if mode == 'train':
        # 训练模式：尝试延长获取未来数据以计算target
        try:
            from pandas.tseries.offsets import BDay
            extended_end = (end_dt + BDay(horizon)).strftime('%Y-%m-%d')
        except:
            extended_end = (end_dt + timedelta(days=horizon + 2)).strftime('%Y-%m-%d')

        logger.info(f"📚 训练模式: 延长end_date从 {end_date} 到 {extended_end} (用于计算T+{horizon}target)")
        return extended_end

    else:  # mode == 'predict'
        # 预测模式：不需要延长，直接用输入日期
        logger.info(f"🔮 预测模式: 使用原始end_date {end_date} (不需要未来数据)")
        return end_date


def validate_mode_parameter(mode: str) -> str:
    """验证并标准化mode参数"""
    if mode is None:
        mode = 'train'  # 默认训练模式

    mode = str(mode).lower().strip()

    if mode not in ['train', 'predict', 'inference']:
        raise ValueError(
            f"无效的mode参数: {mode}\n"
            f"支持的值: 'train', 'predict', 'inference'\n"
            f"  - 'train': 训练模式，计算target并dropna\n"
            f"  - 'predict' 或 'inference': 预测模式，不计算target也不dropna"
        )

    # 统一 'inference' 和 'predict'
    if mode == 'inference':
        mode = 'predict'

    return mode


# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据
    dates = pd.date_range('2025-10-01', '2025-10-10', freq='D')
    tickers = ['AAPL', 'MSFT']

    data = []
    for ticker in tickers:
        for date in dates:
            data.append({
                'date': date,
                'ticker': ticker,
                'Close': 100 + np.random.randn(),
                'feature1': np.random.randn(),
                'feature2': np.random.randn(),
            })

    df = pd.DataFrame(data)
    df = df.set_index(['date', 'ticker'])

    print("\n" + "=" * 80)
    print("测试1: 训练模式")
    print("=" * 80)
    train_df = prepare_data_with_mode(df.copy(), mode='train', horizon=5)
    print(f"结果: {len(train_df)} 样本, 有target: {train_df['target'].notna().sum()}")

    print("\n" + "=" * 80)
    print("测试2: 预测模式")
    print("=" * 80)
    predict_df = prepare_data_with_mode(df.copy(), mode='predict', horizon=5)
    print(f"结果: {len(predict_df)} 样本, 有target: {predict_df['target'].notna().sum()}")

    print("\n" + "=" * 80)
    print("测试3: end_date调整")
    print("=" * 80)
    train_end = get_end_date_for_mode('2025-10-10', mode='train', horizon=5)
    predict_end = get_end_date_for_mode('2025-10-10', mode='predict', horizon=5)
    print(f"训练模式end_date: {train_end}")
    print(f"预测模式end_date: {predict_end}")
