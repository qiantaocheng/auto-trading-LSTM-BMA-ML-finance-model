#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查target_new和Close_new的含义和区别
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
logger = logging.getLogger("check_new_cols")


def check_new_columns(data_file: str):
    """检查target_new和Close_new的含义"""
    logger.info("=" * 80)
    logger.info("检查target_new和Close_new的含义")
    logger.info("=" * 80)
    
    # 加载数据
    logger.info(f"Loading data from {data_file}")
    df = pd.read_parquet(data_file)
    
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # 检查target vs target_new
    logger.info("\n" + "=" * 80)
    logger.info("检查target vs target_new")
    logger.info("=" * 80)
    
    if 'target' in df.columns and 'target_new' in df.columns:
        # 获取一些样本进行比较
        sample = df[['target', 'target_new']].dropna()
        
        logger.info(f"Both columns present: {len(sample)} samples with both values")
        
        if len(sample) > 0:
            # 检查是否相等
            are_equal = np.allclose(sample['target'], sample['target_new'], equal_nan=True)
            logger.info(f"  Are equal (within tolerance): {are_equal}")
            
            if not are_equal:
                diff = (sample['target'] - sample['target_new']).abs()
                logger.info(f"  Max difference: {diff.max():.6f}")
                logger.info(f"  Mean difference: {diff.mean():.6f}")
                logger.info(f"  Non-zero differences: {(diff > 1e-6).sum()}")
            
            # 统计信息
            logger.info(f"\n  target statistics:")
            logger.info(f"    Mean: {sample['target'].mean():.6f}")
            logger.info(f"    Std: {sample['target'].std():.6f}")
            logger.info(f"    Min: {sample['target'].min():.6f}")
            logger.info(f"    Max: {sample['target'].max():.6f}")
            
            logger.info(f"\n  target_new statistics:")
            logger.info(f"    Mean: {sample['target_new'].mean():.6f}")
            logger.info(f"    Std: {sample['target_new'].std():.6f}")
            logger.info(f"    Min: {sample['target_new'].min():.6f}")
            logger.info(f"    Max: {sample['target_new'].max():.6f}")
            
            # 检查相关性
            corr = sample['target'].corr(sample['target_new'])
            logger.info(f"\n  Correlation: {corr:.6f}")
            
            if corr > 0.99:
                logger.warning("  ⚠️ target_new is highly correlated with target!")
                logger.warning("     This suggests target_new might be the same as target or a copy!")
    else:
        logger.warning("  Missing columns: target={}, target_new={}".format(
            'target' in df.columns, 'target_new' in df.columns
        ))
    
    # 检查Close vs Close_new
    logger.info("\n" + "=" * 80)
    logger.info("检查Close vs Close_new")
    logger.info("=" * 80)
    
    if 'Close' in df.columns and 'Close_new' in df.columns:
        sample2 = df[['Close', 'Close_new']].dropna()
        
        logger.info(f"Both columns present: {len(sample2)} samples with both values")
        
        if len(sample2) > 0:
            # 检查是否相等
            are_equal = np.allclose(sample2['Close'], sample2['Close_new'], equal_nan=True, rtol=1e-5)
            logger.info(f"  Are equal (within tolerance): {are_equal}")
            
            if not are_equal:
                diff = (sample2['Close'] - sample2['Close_new']).abs()
                logger.info(f"  Max difference: {diff.max():.2f}")
                logger.info(f"  Mean difference: {diff.mean():.2f}")
                logger.info(f"  Non-zero differences: {(diff > 1e-3).sum()}")
            
            # 统计信息
            logger.info(f"\n  Close statistics:")
            logger.info(f"    Mean: {sample2['Close'].mean():.2f}")
            logger.info(f"    Std: {sample2['Close'].std():.2f}")
            logger.info(f"    Min: {sample2['Close'].min():.2f}")
            logger.info(f"    Max: {sample2['Close'].max():.2f}")
            
            logger.info(f"\n  Close_new statistics:")
            logger.info(f"    Mean: {sample2['Close_new'].mean():.2f}")
            logger.info(f"    Std: {sample2['Close_new'].std():.2f}")
            logger.info(f"    Min: {sample2['Close_new'].min():.2f}")
            logger.info(f"    Max: {sample2['Close_new'].max():.2f}")
            
            # 检查相关性
            corr = sample2['Close'].corr(sample2['Close_new'])
            logger.info(f"\n  Correlation: {corr:.6f}")
            
            if corr > 0.99:
                logger.warning("  ⚠️ Close_new is highly correlated with Close!")
                logger.warning("     This suggests Close_new might be the same as Close or a copy!")
    
    # 列出所有特征列
    logger.info("\n" + "=" * 80)
    logger.info("所有特征列列表")
    logger.info("=" * 80)
    
    exclude_cols = {'target', 'Close', 'ret_fwd_5d', 'sector', 'target_new', 'Close_new'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"\nTotal feature columns: {len(feature_cols)}")
    logger.info(f"\nFeature columns:")
    for i, col in enumerate(feature_cols, 1):
        non_null_count = df[col].notna().sum()
        non_null_pct = non_null_count / len(df) * 100
        logger.info(f"  {i:2d}. {col:30s} (non-null: {non_null_pct:.1f}%)")
    
    # 检查是否有其他可疑的列
    logger.info("\n" + "=" * 80)
    logger.info("检查其他可疑列（可能包含目标变量）")
    logger.info("=" * 80)
    
    suspicious_cols = [col for col in df.columns if 'target' in col.lower() or 'return' in col.lower() or 'ret_fwd' in col.lower()]
    logger.info(f"Suspicious columns: {suspicious_cols}")
    
    for col in suspicious_cols:
        if col not in exclude_cols:
            logger.warning(f"  ⚠️ {col} might contain target information!")


def main():
    parser = argparse.ArgumentParser(description="检查target_new和Close_new的含义")
    parser.add_argument("--data-file", type=str, required=True,
                       help="数据文件路径")
    
    args = parser.parse_args()
    
    check_new_columns(args.data_file)


if __name__ == "__main__":
    main()
