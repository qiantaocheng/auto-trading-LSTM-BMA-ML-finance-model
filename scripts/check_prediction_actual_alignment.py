#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查测试集预测和actual的时间对齐问题

关键问题：测试集IC异常高（XGBoost: 0.9387, LambdaRank: 0.8272）
可能原因：预测和actual的时间对齐问题
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
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("check_alignment")


def simulate_test_prediction_alignment(data_file: str, test_start_date: str, test_end_date: str):
    """模拟测试集预测过程，检查时间对齐"""
    logger.info("=" * 80)
    logger.info("模拟测试集预测过程，检查时间对齐")
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
    
    # 获取日期列表
    dates = test_data.index.get_level_values('date').unique().sort_values()
    logger.info(f"Test dates: {len(dates)} unique dates")
    
    # 模拟预测过程
    logger.info("\n" + "=" * 80)
    logger.info("模拟预测过程（检查时间对齐）")
    logger.info("=" * 80)
    
    all_predictions = []
    
    for i, pred_date in enumerate(dates[:5]):  # 只检查前5个日期
        logger.info(f"\nDate {i+1}: {pred_date.date()}")
        
        try:
            # 这是测试集预测的代码逻辑
            date_data = test_data.xs(pred_date, level='date', drop_level=True)
            
            if len(date_data) == 0:
                logger.warning(f"  No data for {pred_date.date()}")
                continue
            
            # 准备特征（模拟）
            exclude_cols = {'target', 'Close', 'ret_fwd_5d', 'sector'}
            all_feature_cols = [col for col in date_data.columns if col not in exclude_cols]
            X = date_data[all_feature_cols].fillna(0)
            
            # 获取actual target（这是关键）
            actual_target = date_data['target'] if 'target' in date_data.columns else pd.Series(np.nan, index=X.index)
            
            logger.info(f"  X shape: {X.shape}")
            logger.info(f"  X index: {X.index[:5].tolist()}")
            logger.info(f"  actual_target shape: {actual_target.shape}")
            logger.info(f"  actual_target index: {actual_target.index[:5].tolist()}")
            
            # 检查索引是否对齐
            if not X.index.equals(actual_target.index):
                logger.warning(f"  ⚠️ Index mismatch!")
                logger.warning(f"    X index: {X.index[:5].tolist()}")
                logger.warning(f"    actual_target index: {actual_target.index[:5].tolist()}")
            else:
                logger.info(f"  ✅ Index aligned")
            
            # 检查target的值
            logger.info(f"  Target stats:")
            logger.info(f"    Mean: {actual_target.mean():.6f}")
            logger.info(f"    Std: {actual_target.std():.6f}")
            logger.info(f"    Min: {actual_target.min():.6f}")
            logger.info(f"    Max: {actual_target.max():.6f}")
            
            # 模拟预测（使用一个简单的特征作为预测）
            # 这里我们使用momentum_10d作为预测，看看IC是多少
            if 'momentum_10d' in X.columns:
                mock_prediction = X['momentum_10d'].values
                
                # 计算IC
                valid_mask = ~(np.isnan(mock_prediction) | actual_target.isna())
                pred_clean = mock_prediction[valid_mask]
                actual_clean = actual_target[valid_mask].values
                
                if len(pred_clean) > 10:
                    ic = pearsonr(pred_clean, actual_clean)[0]
                    rank_ic = spearmanr(pred_clean, actual_clean)[0]
                    logger.info(f"\n  Mock prediction IC (using momentum_10d):")
                    logger.info(f"    IC: {ic:.6f}")
                    logger.info(f"    Rank IC: {rank_ic:.6f}")
                    
                    if abs(ic) > 0.9:
                        logger.warning(f"    ⚠️ Mock IC is extremely high!")
            
        except Exception as e:
            logger.error(f"  Error processing {pred_date.date()}: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="检查测试集预测和actual的时间对齐")
    parser.add_argument("--data-file", type=str, required=True,
                       help="数据文件路径")
    parser.add_argument("--test-start-date", type=str, required=True,
                       help="测试开始日期 (YYYY-MM-DD)")
    parser.add_argument("--test-end-date", type=str, required=True,
                       help="测试结束日期 (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    simulate_test_prediction_alignment(
        data_file=args.data_file,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date
    )


if __name__ == "__main__":
    main()
