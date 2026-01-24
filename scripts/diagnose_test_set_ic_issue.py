#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断测试集IC异常高的问题

检查：
1. 测试集预测和实际值的时间对齐
2. 测试集预测分布 vs OOF预测分布
3. 测试集IC计算的详细过程
4. 测试集特征是否包含未来信息
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
logger = logging.getLogger("diagnose_test_ic")


def load_test_predictions(results_dir: str) -> Dict[str, pd.DataFrame]:
    """加载测试集预测结果"""
    results_path = Path(results_dir)
    
    # 查找所有模型的预测文件
    predictions = {}
    
    # 尝试从report_df.csv获取模型列表
    report_df_path = results_path / "report_df.csv"
    if report_df_path.exists():
        report_df = pd.read_csv(report_df_path)
        model_names = report_df['Model'].tolist()
        
        for model_name in model_names:
            # 尝试查找对应的预测文件
            # 预测通常保存在all_results中，但可能没有单独保存
            # 我们需要从评估过程中重新生成
            logger.info(f"Model {model_name} found in report")
    
    return predictions


def analyze_ic_calculation(predictions_df: pd.DataFrame, model_name: str):
    """分析IC计算的详细过程"""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Analyzing IC calculation for {model_name}")
    logger.info(f"{'=' * 80}")
    
    # 按日期分组
    predictions_df_sorted = predictions_df.sort_values('date').copy()
    
    daily_ics = []
    daily_rank_ics = []
    daily_dates = []
    daily_n_samples = []
    
    for date, date_group in predictions_df_sorted.groupby('date'):
        date_pred = date_group['prediction'].values
        date_actual = date_group['actual'].values
        
        # 移除NaN
        valid_mask = ~(np.isnan(date_pred) | np.isnan(date_actual))
        date_pred_clean = date_pred[valid_mask]
        date_actual_clean = date_actual[valid_mask]
        
        if len(date_pred_clean) < 2:
            continue
        
        # 计算IC和Rank IC
        ic_day = float(np.corrcoef(date_pred_clean, date_actual_clean)[0, 1])
        rank_ic_day, _ = spearmanr(date_pred_clean, date_actual_clean)
        
        if not (np.isnan(ic_day) or np.isinf(ic_day)):
            daily_ics.append(ic_day)
            daily_rank_ics.append(rank_ic_day)
            daily_dates.append(date)
            daily_n_samples.append(len(date_pred_clean))
    
    if len(daily_ics) == 0:
        logger.warning(f"No valid daily ICs for {model_name}")
        return
    
    daily_ics_array = np.array(daily_ics)
    daily_rank_ics_array = np.array(daily_rank_ics)
    
    logger.info(f"\nDaily IC Statistics:")
    logger.info(f"  Mean IC: {daily_ics_array.mean():.6f}")
    logger.info(f"  Std IC: {daily_ics_array.std():.6f}")
    logger.info(f"  Min IC: {daily_ics_array.min():.6f}")
    logger.info(f"  Max IC: {daily_ics_array.max():.6f}")
    logger.info(f"  Median IC: {np.median(daily_ics_array):.6f}")
    
    logger.info(f"\nDaily Rank IC Statistics:")
    logger.info(f"  Mean Rank IC: {daily_rank_ics_array.mean():.6f}")
    logger.info(f"  Std Rank IC: {daily_rank_ics_array.std():.6f}")
    logger.info(f"  Min Rank IC: {daily_rank_ics_array.min():.6f}")
    logger.info(f"  Max Rank IC: {daily_rank_ics_array.max():.6f}")
    logger.info(f"  Median Rank IC: {np.median(daily_rank_ics_array):.6f}")
    
    # 检查异常高的IC
    high_ic_threshold = 0.9
    high_ic_days = daily_ics_array > high_ic_threshold
    if high_ic_days.sum() > 0:
        logger.warning(f"\n⚠️ Found {high_ic_days.sum()} days with IC > {high_ic_threshold}")
        high_ic_dates = [daily_dates[i] for i in np.where(high_ic_days)[0]]
        high_ic_values = daily_ics_array[high_ic_days]
        for date, ic_val in zip(high_ic_dates[:10], high_ic_values[:10]):  # 只显示前10个
            logger.warning(f"  Date: {date}, IC: {ic_val:.6f}")
    
    # 检查预测和实际值的分布
    logger.info(f"\nPrediction Statistics:")
    logger.info(f"  Mean: {predictions_df['prediction'].mean():.6f}")
    logger.info(f"  Std: {predictions_df['prediction'].std():.6f}")
    logger.info(f"  Min: {predictions_df['prediction'].min():.6f}")
    logger.info(f"  Max: {predictions_df['prediction'].max():.6f}")
    
    logger.info(f"\nActual Statistics:")
    logger.info(f"  Mean: {predictions_df['actual'].mean():.6f}")
    logger.info(f"  Std: {predictions_df['actual'].std():.6f}")
    logger.info(f"  Min: {predictions_df['actual'].min():.6f}")
    logger.info(f"  Max: {predictions_df['actual'].max():.6f}")
    
    # 检查预测和实际值的相关性
    valid_mask = ~(predictions_df['prediction'].isna() | predictions_df['actual'].isna())
    pred_clean = predictions_df.loc[valid_mask, 'prediction']
    actual_clean = predictions_df.loc[valid_mask, 'actual']
    
    if len(pred_clean) > 10:
        overall_ic = pearsonr(pred_clean, actual_clean)[0]
        overall_rank_ic = spearmanr(pred_clean, actual_clean)[0]
        logger.info(f"\nOverall Correlation (all days combined):")
        logger.info(f"  IC: {overall_ic:.6f}")
        logger.info(f"  Rank IC: {overall_rank_ic:.6f}")


def main():
    parser = argparse.ArgumentParser(description="诊断测试集IC异常高的问题")
    parser.add_argument("--results-dir", type=str, required=True,
                       help="测试集评估结果目录")
    parser.add_argument("--model-name", type=str, default="xgboost",
                       help="要诊断的模型名称")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("诊断测试集IC异常高的问题")
    logger.info("=" * 80)
    
    # 注意：这个脚本需要从评估过程中获取predictions DataFrame
    # 目前只是框架，需要实际运行评估才能获取数据
    
    logger.warning("\n⚠️ 这个脚本需要从评估过程中获取predictions DataFrame")
    logger.warning("   建议：修改 time_split_80_20_oos_eval.py 保存predictions DataFrame")
    logger.warning("   或者：重新运行评估并保存predictions DataFrame")
    
    logger.info("\n建议的检查步骤：")
    logger.info("1. 检查测试集预测和实际值的时间对齐")
    logger.info("2. 检查测试集预测分布 vs OOF预测分布")
    logger.info("3. 检查测试集IC计算的详细过程")
    logger.info("4. 检查测试集特征是否包含未来信息")


if __name__ == "__main__":
    main()
