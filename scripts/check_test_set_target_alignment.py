#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查测试集target的时间对齐问题

关键问题：测试集IC异常高（XGBoost: 0.9387, LambdaRank: 0.8272）
可能原因：target的时间对齐问题
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Fix module import path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("check_target_alignment")


def check_target_alignment(data_file: str, test_start_date: str, test_end_date: str):
    """检查测试集target的时间对齐"""
    logger.info("=" * 80)
    logger.info("检查测试集target的时间对齐")
    logger.info("=" * 80)
    
    # 加载数据
    logger.info(f"Loading data from {data_file}")
    df = pd.read_parquet(data_file)
    
    # 过滤测试期数据
    test_start = pd.to_datetime(test_start_date)
    test_end = pd.to_datetime(test_end_date)
    
    test_data = df.loc[
        (df.index.get_level_values('date') >= test_start) & 
        (df.index.get_level_values('date') <= test_end)
    ].copy()
    
    logger.info(f"Test data: {len(test_data)} samples")
    logger.info(f"Test period: {test_start.date()} to {test_end.date()}")
    
    # 检查target列
    if 'target' not in test_data.columns:
        logger.error("❌ 'target' column not found in test data")
        return
    
    # 检查target的计算方式
    logger.info("\n" + "=" * 80)
    logger.info("检查target的计算方式")
    logger.info("=" * 80)
    
    # 检查是否有ret_fwd_10d列
    if 'ret_fwd_10d' in test_data.columns:
        logger.info("✅ Found 'ret_fwd_10d' column")
        
        # 比较target和ret_fwd_10d
        comparison = pd.DataFrame({
            'target': test_data['target'],
            'ret_fwd_10d': test_data['ret_fwd_10d']
        })
        
        # 移除NaN
        valid_mask = ~(comparison['target'].isna() | comparison['ret_fwd_10d'].isna())
        comparison_clean = comparison[valid_mask]
        
        if len(comparison_clean) > 0:
            # 检查是否相等
            are_equal = np.allclose(comparison_clean['target'], comparison_clean['ret_fwd_10d'], equal_nan=True)
            logger.info(f"  target == ret_fwd_10d: {are_equal}")
            
            if not are_equal:
                diff = (comparison_clean['target'] - comparison_clean['ret_fwd_10d']).abs()
                logger.warning(f"  Max difference: {diff.max():.6f}")
                logger.warning(f"  Mean difference: {diff.mean():.6f}")
    
    # 检查target的时间对齐
    logger.info("\n" + "=" * 80)
    logger.info("检查target的时间对齐")
    logger.info("=" * 80)
    
    # 获取日期列表
    dates = test_data.index.get_level_values('date').unique().sort_values()
    logger.info(f"Test period has {len(dates)} unique dates")
    
    # 检查每个日期的target
    for i, date in enumerate(dates[:5]):  # 只检查前5个日期
        date_data = test_data.xs(date, level='date', drop_level=False)
        logger.info(f"\nDate {i+1}: {date.date()}")
        logger.info(f"  Samples: {len(date_data)}")
        logger.info(f"  Target mean: {date_data['target'].mean():.6f}")
        logger.info(f"  Target std: {date_data['target'].std():.6f}")
        logger.info(f"  Target min: {date_data['target'].min():.6f}")
        logger.info(f"  Target max: {date_data['target'].max():.6f}")
        
        # 检查target是否包含未来信息
        # 如果target是ret_fwd_10d，那么对于日期date，target应该是从date开始的未来10天收益
        # 这意味着target不应该在date之前就能知道
        
        # 检查是否有异常值
        if 'Close' in date_data.columns:
            close_values = date_data['Close']
            logger.info(f"  Close mean: {close_values.mean():.2f}")
            logger.info(f"  Close std: {close_values.std():.2f}")
    
    # 检查target的分布
    logger.info("\n" + "=" * 80)
    logger.info("Target分布统计")
    logger.info("=" * 80)
    
    target_clean = test_data['target'].dropna()
    logger.info(f"Valid target samples: {len(target_clean)}")
    logger.info(f"Target mean: {target_clean.mean():.6f}")
    logger.info(f"Target std: {target_clean.std():.6f}")
    logger.info(f"Target min: {target_clean.min():.6f}")
    logger.info(f"Target max: {target_clean.max():.6f}")
    logger.info(f"Target median: {target_clean.median():.6f}")
    
    # 检查异常值
    extreme_threshold = 0.5  # 50%收益
    extreme_count = (target_clean.abs() > extreme_threshold).sum()
    logger.info(f"\nExtreme values (|target| > {extreme_threshold}): {extreme_count} ({extreme_count/len(target_clean)*100:.2f}%)")
    
    if extreme_count > 0:
        extreme_values = target_clean[target_clean.abs() > extreme_threshold]
        logger.warning(f"  Extreme values: {extreme_values.head(10).tolist()}")


def main():
    parser = argparse.ArgumentParser(description="检查测试集target的时间对齐")
    parser.add_argument("--data-file", type=str, required=True,
                       help="数据文件路径")
    parser.add_argument("--test-start-date", type=str, required=True,
                       help="测试开始日期 (YYYY-MM-DD)")
    parser.add_argument("--test-end-date", type=str, required=True,
                       help="测试结束日期 (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    check_target_alignment(
        data_file=args.data_file,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date
    )


if __name__ == "__main__":
    main()
