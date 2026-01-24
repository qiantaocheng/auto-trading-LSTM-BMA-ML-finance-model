#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练结果中分析OOF预测是否存在数据泄露

使用方法：
1. 运行训练，保存训练结果
2. 运行此脚本分析OOF预测

或者：
1. 运行80/20评估
2. 从评估结果中提取OOF预测进行分析
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("oof_leakage_analysis")


def analyze_oof_temporal_distribution(
    oof_predictions: Dict[str, pd.Series],
    dates: pd.Series,
    train_end_date: str,
    test_start_date: str
) -> Dict[str, Any]:
    """
    分析OOF预测的时间分布
    
    如果存在泄露：
    - OOF预测应该只来自训练集日期
    - 不应该有测试集日期的OOF预测
    """
    logger.info("=" * 80)
    logger.info("分析OOF预测的时间分布")
    logger.info("=" * 80)
    
    train_end = pd.to_datetime(train_end_date)
    test_start = pd.to_datetime(test_start_date)
    
    results = {}
    
    for model_name, oof_pred in oof_predictions.items():
        if len(oof_pred) == 0:
            continue
        
        # 获取OOF预测的日期
        if isinstance(oof_pred.index, pd.MultiIndex):
            oof_dates = pd.to_datetime(oof_pred.index.get_level_values('date'))
        else:
            # 尝试从dates中匹配
            oof_dates = dates.reindex(oof_pred.index)
        
        # 检查是否有测试集日期的OOF预测
        test_date_mask = oof_dates >= test_start
        train_date_mask = oof_dates <= train_end
        
        n_test_dates = test_date_mask.sum()
        n_train_dates = train_date_mask.sum()
        n_total = len(oof_pred)
        
        logger.info(f"\n模型: {model_name}")
        logger.info(f"  总OOF预测数: {n_total}")
        logger.info(f"  训练集日期: {n_train_dates}")
        logger.info(f"  测试集日期: {n_test_dates}")
        
        if n_test_dates > 0:
            logger.warning(f"  ⚠️  警告: 检测到{n_test_dates}个测试集日期的OOF预测！")
            logger.warning(f"     这可能是数据泄露的信号")
            results[model_name] = {
                'leakage_detected': True,
                'n_test_dates': n_test_dates,
                'n_train_dates': n_train_dates,
                'leakage_ratio': n_test_dates / n_total
            }
        else:
            logger.info(f"  ✅ OOF预测只来自训练集日期")
            results[model_name] = {
                'leakage_detected': False,
                'n_test_dates': 0,
                'n_train_dates': n_train_dates,
                'leakage_ratio': 0.0
            }
    
    return results


def analyze_oof_cv_consistency(
    oof_predictions: Dict[str, pd.Series],
    dates: pd.Series,
    horizon_days: int = 10
) -> Dict[str, Any]:
    """
    分析OOF预测的CV一致性
    
    如果存在泄露：
    - 每个样本的OOF预测应该来自不包含该样本的fold
    - 不应该有样本的OOF预测来自包含该样本的fold
    """
    logger.info("=" * 80)
    logger.info("分析OOF预测的CV一致性")
    logger.info("=" * 80)
    
    logger.warning("⚠️  此分析需要CV fold信息，当前实现为简化版本")
    logger.info("   建议：检查OOF预测是否在正确的fold中生成")
    
    # 这里需要实际的CV fold信息才能进行完整分析
    # 简化版本：检查OOF预测的分布是否合理
    
    results = {}
    
    for model_name, oof_pred in oof_predictions.items():
        if len(oof_pred) == 0:
            continue
        
        # 检查OOF预测的覆盖率
        # 理想情况下，OOF预测应该覆盖大部分训练样本
        coverage = oof_pred.notna().sum() / len(oof_pred) if len(oof_pred) > 0 else 0
        
        logger.info(f"\n模型: {model_name}")
        logger.info(f"  OOF预测覆盖率: {coverage:.2%}")
        
        if coverage < 0.5:
            logger.warning(f"  ⚠️  警告: OOF预测覆盖率低，可能存在CV分割问题")
            results[model_name] = {'coverage': coverage, 'warning': True}
        else:
            logger.info(f"  ✅ OOF预测覆盖率正常")
            results[model_name] = {'coverage': coverage, 'warning': False}
    
    return results


def compare_oof_with_shuffled_target(
    oof_predictions: Dict[str, pd.Series],
    actuals: pd.Series
) -> Dict[str, Any]:
    """
    比较OOF预测与随机打乱目标变量的相关性
    
    如果存在泄露：
    - OOF预测应该与随机目标变量无关（相关性 ≈ 0）
    - 如果OOF预测与随机目标变量相关，可能存在泄露
    """
    logger.info("=" * 80)
    logger.info("比较OOF预测与随机打乱目标变量")
    logger.info("=" * 80)
    
    # 随机打乱目标变量
    np.random.seed(42)
    shuffled_actuals = actuals.copy()
    shuffled_values = np.random.permutation(shuffled_actuals.values)
    shuffled_actuals = pd.Series(shuffled_values, index=shuffled_actuals.index)
    
    results = {}
    
    for model_name, oof_pred in oof_predictions.items():
        if len(oof_pred) == 0:
            continue
        
        # 对齐索引
        common_idx = oof_pred.index.intersection(actuals.index)
        if len(common_idx) == 0:
            continue
        
        oof_pred_aligned = oof_pred.reindex(common_idx)
        actuals_aligned = actuals.reindex(common_idx)
        shuffled_aligned = shuffled_actuals.reindex(common_idx)
        
        # 移除NaN
        valid_mask = ~(oof_pred_aligned.isna() | actuals_aligned.isna() | shuffled_aligned.isna())
        oof_clean = oof_pred_aligned[valid_mask]
        actuals_clean = actuals_aligned[valid_mask]
        shuffled_clean = shuffled_aligned[valid_mask]
        
        if len(oof_clean) < 10:
            continue
        
        # 计算相关性
        corr_true = pearsonr(oof_clean, actuals_clean)[0]
        corr_shuffled = pearsonr(oof_clean, shuffled_clean)[0]
        
        logger.info(f"\n模型: {model_name}")
        logger.info(f"  OOF vs 真实目标: IC = {corr_true:.6f}")
        logger.info(f"  OOF vs 随机目标: IC = {corr_shuffled:.6f}")
        
        if abs(corr_shuffled) > 0.1:
            logger.warning(f"  ⚠️  警告: OOF预测与随机目标变量相关性高 ({corr_shuffled:.6f})")
            logger.warning(f"     这可能是数据泄露的信号")
            results[model_name] = {
                'leakage_detected': True,
                'corr_true': corr_true,
                'corr_shuffled': corr_shuffled
            }
        else:
            logger.info(f"  ✅ OOF预测与随机目标变量相关性低，正常")
            results[model_name] = {
                'leakage_detected': False,
                'corr_true': corr_true,
                'corr_shuffled': corr_shuffled
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="从训练结果分析OOF预测泄露")
    parser.add_argument("--data-file", type=str, required=True,
                       help="数据文件路径")
    parser.add_argument("--snapshot-id", type=str,
                       help="模型快照ID（用于加载OOF预测）")
    parser.add_argument("--train-end-date", type=str,
                       help="训练集结束日期（YYYY-MM-DD）")
    parser.add_argument("--test-start-date", type=str,
                       help="测试集开始日期（YYYY-MM-DD）")
    parser.add_argument("--horizon-days", type=int, default=10,
                       help="预测horizon天数")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("OOF预测泄露分析")
    logger.info("=" * 80)
    
    # 加载数据
    logger.info(f"加载数据: {args.data_file}")
    df = pd.read_parquet(args.data_file)
    
    # 提取目标变量和日期
    if 'target' not in df.columns:
        logger.error("数据中缺少'target'列")
        return
    
    actuals = df['target'].copy()
    
    if isinstance(df.index, pd.MultiIndex):
        dates = pd.to_datetime(df.index.get_level_values('date'))
    else:
        logger.error("数据索引必须是MultiIndex (date, ticker)")
        return
    
    # 加载OOF预测
    oof_predictions = {}
    
    if args.snapshot_id:
        # 尝试从快照加载OOF预测
        try:
            from bma_models.model_registry import load_models_from_snapshot
            loaded = load_models_from_snapshot(str(args.snapshot_id), load_catboost=True)
            
            # 这里需要根据实际的数据结构来提取OOF预测
            # 可能需要从训练结果中提取
            logger.warning("⚠️  OOF预测加载功能需要根据实际数据结构实现")
            
        except Exception as e:
            logger.error(f"加载快照失败: {e}")
    
    if not oof_predictions:
        logger.error("无法加载OOF预测，请检查快照ID或提供OOF预测数据")
        return
    
    # 运行分析
    if args.train_end_date and args.test_start_date:
        temporal_results = analyze_oof_temporal_distribution(
            oof_predictions, dates, args.train_end_date, args.test_start_date
        )
    
    cv_results = analyze_oof_cv_consistency(
        oof_predictions, dates, args.horizon_days
    )
    
    shuffle_results = compare_oof_with_shuffled_target(
        oof_predictions, actuals
    )
    
    # 生成报告
    logger.info("\n" + "=" * 80)
    logger.info("分析报告")
    logger.info("=" * 80)
    
    total_models = len(oof_predictions)
    leakage_models = sum(1 for r in shuffle_results.values() if r.get('leakage_detected', False))
    
    logger.info(f"总模型数: {total_models}")
    logger.info(f"检测到泄露信号的模型数: {leakage_models}")
    
    if leakage_models > 0:
        logger.warning("\n⚠️  警告: 检测到潜在的数据泄露！")
    else:
        logger.info("\n✅ 未检测到明显的数据泄露")


if __name__ == "__main__":
    main()
