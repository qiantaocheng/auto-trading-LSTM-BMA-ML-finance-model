#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验：比较OOF预测性能与测试集预测性能

如果80/20 OOF存在数据泄露：
- OOF预测性能应该异常高（因为可能看到了测试集信息）
- OOF预测性能应该远高于测试集预测性能（不应该）
- OOF预测与测试集预测的相关性应该异常高（不应该）

使用方法：
1. 运行80/20评估，保存OOF预测结果
2. 运行此脚本比较OOF和测试集性能
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("oof_vs_test_comparison")


def load_oof_predictions(snapshot_id: str) -> Optional[Dict[str, pd.Series]]:
    """从快照加载OOF预测"""
    try:
        from bma_models.model_registry import load_models_from_snapshot
        
        loaded = load_models_from_snapshot(str(snapshot_id), load_catboost=True)
        
        # 尝试从训练结果中提取OOF预测
        # 注意：这需要训练时保存了OOF预测
        oof_predictions = {}
        
        # 检查是否有OOF预测数据
        # 这里需要根据实际的数据结构来调整
        logger.warning("⚠️  OOF预测加载功能需要根据实际数据结构实现")
        
        return oof_predictions
    except Exception as e:
        logger.error(f"加载OOF预测失败: {e}")
        return None


def compare_oof_vs_test(
    oof_predictions: Dict[str, pd.Series],
    test_predictions: Dict[str, pd.Series],
    test_actuals: pd.Series
) -> Dict[str, float]:
    """比较OOF预测和测试集预测的性能"""
    
    results = {}
    
    for model_name in oof_predictions.keys():
        if model_name not in test_predictions:
            logger.warning(f"模型 {model_name} 在测试集预测中不存在，跳过")
            continue
        
        oof_pred = oof_predictions[model_name]
        test_pred = test_predictions[model_name]
        
        # 对齐索引
        common_idx = oof_pred.index.intersection(test_pred.index)
        if len(common_idx) == 0:
            logger.warning(f"模型 {model_name} 的OOF和测试集预测没有共同索引，跳过")
            continue
        
        oof_pred_aligned = oof_pred.reindex(common_idx)
        test_pred_aligned = test_pred.reindex(common_idx)
        actuals_aligned = test_actuals.reindex(common_idx)
        
        # 移除NaN
        valid_mask = ~(oof_pred_aligned.isna() | test_pred_aligned.isna() | actuals_aligned.isna())
        oof_pred_clean = oof_pred_aligned[valid_mask]
        test_pred_clean = test_pred_aligned[valid_mask]
        actuals_clean = actuals_aligned[valid_mask]
        
        if len(oof_pred_clean) < 10:
            logger.warning(f"模型 {model_name} 的有效样本数太少，跳过")
            continue
        
        # 计算OOF预测的IC
        oof_ic = pearsonr(oof_pred_clean, actuals_clean)[0]
        oof_rank_ic = spearmanr(oof_pred_clean, actuals_clean)[0]
        
        # 计算测试集预测的IC
        test_ic = pearsonr(test_pred_clean, actuals_clean)[0]
        test_rank_ic = spearmanr(test_pred_clean, actuals_clean)[0]
        
        # 计算OOF和测试集预测的相关性（不应该太高）
        oof_test_corr = pearsonr(oof_pred_clean, test_pred_clean)[0]
        
        logger.info(f"\n模型: {model_name}")
        logger.info(f"  OOF IC: {oof_ic:.6f}, Rank IC: {oof_rank_ic:.6f}")
        logger.info(f"  测试集 IC: {test_ic:.6f}, Rank IC: {test_rank_ic:.6f}")
        logger.info(f"  OOF vs 测试集相关性: {oof_test_corr:.6f}")
        logger.info(f"  IC差异: {oof_ic - test_ic:.6f}")
        
        # 检测泄露信号
        leakage_signals = []
        
        # 信号1: OOF IC远高于测试集IC（不应该）
        if oof_ic > test_ic + 0.1:
            leakage_signals.append(f"OOF IC ({oof_ic:.4f}) >> 测试集 IC ({test_ic:.4f})")
        
        # 信号2: OOF和测试集预测高度相关（不应该）
        if abs(oof_test_corr) > 0.8:
            leakage_signals.append(f"OOF与测试集预测高度相关 ({oof_test_corr:.4f})")
        
        # 信号3: OOF IC异常高（可能看到了测试集信息）
        if oof_ic > 0.9:
            leakage_signals.append(f"OOF IC异常高 ({oof_ic:.4f})")
        
        if leakage_signals:
            logger.warning(f"  ⚠️  检测到潜在泄露信号:")
            for signal in leakage_signals:
                logger.warning(f"    - {signal}")
        else:
            logger.info(f"  ✅ 未检测到明显泄露信号")
        
        results[model_name] = {
            'oof_ic': oof_ic,
            'oof_rank_ic': oof_rank_ic,
            'test_ic': test_ic,
            'test_rank_ic': test_rank_ic,
            'oof_test_corr': oof_test_corr,
            'ic_diff': oof_ic - test_ic,
            'leakage_signals': leakage_signals
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="比较OOF预测和测试集预测性能")
    parser.add_argument("--snapshot-id", type=str, help="模型快照ID（用于加载OOF预测）")
    parser.add_argument("--test-predictions-file", type=str, help="测试集预测文件（CSV）")
    parser.add_argument("--test-actuals-file", type=str, help="测试集实际值文件（CSV）")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("OOF vs 测试集性能比较")
    logger.info("=" * 80)
    
    # 加载OOF预测
    oof_predictions = {}
    if args.snapshot_id:
        oof_predictions = load_oof_predictions(args.snapshot_id)
        if oof_predictions is None:
            logger.error("无法加载OOF预测，请检查快照ID或数据结构")
            return
    
    # 加载测试集预测
    if args.test_predictions_file:
        test_pred_df = pd.read_csv(args.test_predictions_file)
        test_predictions = {}
        # 根据实际文件格式解析
        # 这里需要根据实际的数据结构来调整
        logger.warning("⚠️  测试集预测加载功能需要根据实际文件格式实现")
    else:
        logger.error("请提供测试集预测文件")
        return
    
    # 加载测试集实际值
    if args.test_actuals_file:
        test_actuals_df = pd.read_csv(args.test_actuals_file)
        # 根据实际文件格式解析
        logger.warning("⚠️  测试集实际值加载功能需要根据实际文件格式实现")
    else:
        logger.error("请提供测试集实际值文件")
        return
    
    # 比较性能
    if oof_predictions and test_predictions:
        results = compare_oof_vs_test(oof_predictions, test_predictions, test_actuals_df)
        
        # 生成报告
        logger.info("\n" + "=" * 80)
        logger.info("检测报告")
        logger.info("=" * 80)
        
        total_models = len(results)
        leakage_models = sum(1 for r in results.values() if r['leakage_signals'])
        
        logger.info(f"总模型数: {total_models}")
        logger.info(f"检测到泄露信号的模型数: {leakage_models}")
        
        if leakage_models > 0:
            logger.warning("\n⚠️  警告: 检测到潜在的数据泄露！")
            logger.warning("   建议检查：")
            logger.warning("   1. OOF预测是否使用了测试集信息")
            logger.warning("   2. 特征标准化是否使用了测试集统计量")
            logger.warning("   3. CV分割是否正确处理了时间顺序")
        else:
            logger.info("\n✅ 未检测到明显的数据泄露")


if __name__ == "__main__":
    main()
