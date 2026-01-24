#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度检查测试集预测问题

关键发现：
- OOF IC正常（0.01-0.09）
- 但测试集IC异常高（XGBoost: 0.9387, LambdaRank: 0.8272）

可能原因：
1. 测试集特征可能包含未来信息
2. 测试集预测和实际值的时间对齐问题
3. 测试集数据本身的问题
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
logger = logging.getLogger("deep_check")


def check_feature_target_correlation(data_file: str, test_start_date: str, test_end_date: str):
    """检查测试集特征和目标变量的相关性"""
    logger.info("=" * 80)
    logger.info("检查测试集特征和目标变量的相关性")
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
    
    # 获取特征列
    feature_cols = [c for c in test_data.columns if c not in ['target', 'Close', 'ret_fwd_5d', 'sector']]
    logger.info(f"Features: {len(feature_cols)}")
    
    # 检查每个特征与target的相关性
    logger.info("\n" + "=" * 80)
    logger.info("检查特征与target的相关性（按天计算）")
    logger.info("=" * 80)
    
    # 按天计算IC
    dates = test_data.index.get_level_values('date').unique().sort_values()
    
    feature_ics = {feat: [] for feat in feature_cols[:20]}  # 只检查前20个特征
    
    for date in dates:
        date_data = test_data.xs(date, level='date', drop_level=False)
        
        target = date_data['target'].values
        
        for feat in feature_cols[:20]:
            feat_values = date_data[feat].values
            
            # 移除NaN
            valid_mask = ~(np.isnan(feat_values) | np.isnan(target))
            if valid_mask.sum() < 2:
                continue
            
            feat_clean = feat_values[valid_mask]
            target_clean = target[valid_mask]
            
            # 计算IC
            try:
                ic = pearsonr(feat_clean, target_clean)[0]
                if not (np.isnan(ic) or np.isinf(ic)):
                    feature_ics[feat].append(ic)
            except:
                pass
    
    # 计算平均IC
    logger.info("\n特征IC统计（按天计算的平均IC）:")
    feature_avg_ics = {}
    for feat, ics in feature_ics.items():
        if len(ics) > 0:
            avg_ic = np.mean(ics)
            feature_avg_ics[feat] = avg_ic
    
    # 排序并显示
    sorted_features = sorted(feature_avg_ics.items(), key=lambda x: abs(x[1]), reverse=True)
    
    logger.info("\nTop 10 features by |IC|:")
    for feat, ic in sorted_features[:10]:
        logger.info(f"  {feat}: {ic:.6f}")
    
    # 检查是否有异常高的IC
    high_ic_threshold = 0.9
    high_ic_features = [feat for feat, ic in sorted_features if abs(ic) > high_ic_threshold]
    
    if high_ic_features:
        logger.warning(f"\n⚠️ Found {len(high_ic_features)} features with |IC| > {high_ic_threshold}")
        for feat in high_ic_features[:5]:
            logger.warning(f"  {feat}: {feature_avg_ics[feat]:.6f}")
        logger.warning("\n   这可能暗示特征包含未来信息或数据泄露！")


def check_prediction_actual_alignment(results_dir: str):
    """检查预测和实际值的时间对齐"""
    logger.info("\n" + "=" * 80)
    logger.info("检查预测和实际值的时间对齐")
    logger.info("=" * 80)
    
    results_path = Path(results_dir)
    
    # 查找predictions文件
    prediction_files = list(results_path.glob("*_predictions_diagnosis.csv"))
    
    if not prediction_files:
        logger.warning("No prediction files found. Need to run evaluation first.")
        return
    
    for pred_file in prediction_files:
        model_name = pred_file.stem.replace("_predictions_diagnosis", "")
        logger.info(f"\nAnalyzing {model_name}...")
        
        predictions = pd.read_csv(pred_file)
        
        # 检查数据
        logger.info(f"  Total predictions: {len(predictions)}")
        logger.info(f"  Unique dates: {predictions['date'].nunique()}")
        logger.info(f"  Unique tickers: {predictions['ticker'].nunique()}")
        
        # 检查预测和实际值的分布
        logger.info(f"\n  Prediction stats:")
        logger.info(f"    Mean: {predictions['prediction'].mean():.6f}")
        logger.info(f"    Std: {predictions['prediction'].std():.6f}")
        logger.info(f"    Min: {predictions['prediction'].min():.6f}")
        logger.info(f"    Max: {predictions['prediction'].max():.6f}")
        
        logger.info(f"\n  Actual stats:")
        logger.info(f"    Mean: {predictions['actual'].mean():.6f}")
        logger.info(f"    Std: {predictions['actual'].std():.6f}")
        logger.info(f"    Min: {predictions['actual'].min():.6f}")
        logger.info(f"    Max: {predictions['actual'].max():.6f}")
        
        # 计算整体IC
        valid_mask = ~(predictions['prediction'].isna() | predictions['actual'].isna())
        pred_clean = predictions.loc[valid_mask, 'prediction']
        actual_clean = predictions.loc[valid_mask, 'actual']
        
        if len(pred_clean) > 10:
            overall_ic = pearsonr(pred_clean, actual_clean)[0]
            overall_rank_ic = spearmanr(pred_clean, actual_clean)[0]
            logger.info(f"\n  Overall IC (all days combined):")
            logger.info(f"    IC: {overall_ic:.6f}")
            logger.info(f"    Rank IC: {overall_rank_ic:.6f}")
            
            if abs(overall_ic) > 0.9:
                logger.warning(f"\n  ⚠️ Overall IC is extremely high: {overall_ic:.6f}")
                logger.warning(f"     这可能暗示预测和实际值的时间对齐有问题！")


def main():
    parser = argparse.ArgumentParser(description="深度检查测试集预测问题")
    parser.add_argument("--data-file", type=str, required=True,
                       help="数据文件路径")
    parser.add_argument("--test-start-date", type=str, required=True,
                       help="测试开始日期 (YYYY-MM-DD)")
    parser.add_argument("--test-end-date", type=str, required=True,
                       help="测试结束日期 (YYYY-MM-DD)")
    parser.add_argument("--results-dir", type=str, required=True,
                       help="测试集评估结果目录")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("深度检查测试集预测问题")
    logger.info("=" * 80)
    
    # 1. 检查特征和目标变量的相关性
    check_feature_target_correlation(
        data_file=args.data_file,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date
    )
    
    # 2. 检查预测和实际值的时间对齐
    check_prediction_actual_alignment(args.results_dir)


if __name__ == "__main__":
    main()
