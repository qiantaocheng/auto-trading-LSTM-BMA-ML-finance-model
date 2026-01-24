#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验5（完整版）：使用完整的5-fold CV获取真实的OOF预测，比较OOF预测性能与测试集预测性能

关键区别：
1. OOF (Out-of-Fold) 预测：
   - 在训练期内通过CV获得
   - 每个fold的验证集预测
   - 用于训练第二层模型（MetaRankerStacker）
   - 应该与测试集性能接近（如果没有数据泄露）

2. 测试集预测：
   - 在训练期外的未来数据上获得（80/20时间分割的后20%）
   - 用于评估模型的真实泛化能力
   - 应该与OOF性能接近（如果没有数据泄露）

如果OOF IC >> 测试集IC，可能存在数据泄露：
- OOF预测可能使用了测试集信息
- 特征标准化可能使用了测试集统计量
- CV分割可能存在问题
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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
logger = logging.getLogger("experiment_5_full")


def load_test_metrics(report_df_path: str) -> Dict[str, Dict[str, float]]:
    """从report_df.csv加载测试集预测性能"""
    logger.info(f"Loading test metrics from {report_df_path}")
    report_df = pd.read_csv(report_df_path)
    
    test_metrics = {}
    for _, row in report_df.iterrows():
        model_name = row['Model']
        test_metrics[model_name] = {
            'ic': float(row.get('IC', np.nan)),
            'rank_ic': float(row.get('Rank_IC', np.nan)),
            'n_predictions': int(row.get('N_Predictions', 0))
        }
        logger.info(f"  {model_name}: IC={test_metrics[model_name]['ic']:.6f}, Rank IC={test_metrics[model_name]['rank_ic']:.6f}")
    
    return test_metrics


def extract_oof_predictions_from_full_training(
    train_data_path: str,
    train_start_date: str,
    train_end_date: str
) -> Dict[str, pd.Series]:
    """
    通过完整的5-fold CV训练提取真实的OOF预测
    
    方法：运行完整的训练流程，提取OOF预测
    """
    logger.info("=" * 80)
    logger.info("Extracting OOF predictions from FULL 5-fold CV training")
    logger.info("=" * 80)
    
    try:
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        
        logger.info(f"Loading training data: {train_data_path}")
        logger.info(f"Training period: {train_start_date} to {train_end_date}")
        
        # 创建模型并训练（使用完整的5-fold CV）
        model = UltraEnhancedQuantitativeModel()
        
        logger.info("Starting full training with 5-fold CV...")
        logger.info("This may take a while...")
        
        train_res = model.train_from_document(
            training_data_path=train_data_path,
            top_n=50,
            start_date=train_start_date,
            end_date=train_end_date,
        )
        
        if not train_res.get("success", False):
            raise RuntimeError(f"Training failed: {train_res.get('error')}")
        
        logger.info("Training completed successfully")
        
        # 提取OOF预测
        oof_predictions = {}
        
        # 从训练结果中提取OOF预测
        training_results = train_res.get('training_results', {})
        logger.info(f"Training results keys: {list(training_results.keys())}")
        
        if isinstance(training_results, dict):
            # 尝试从不同位置提取OOF预测
            if 'traditional_models' in training_results:
                trad_models = training_results['traditional_models']
                logger.info(f"Traditional models keys: {list(trad_models.keys()) if isinstance(trad_models, dict) else 'N/A'}")
                
                if isinstance(trad_models, dict) and 'oof_predictions' in trad_models:
                    oof_predictions = trad_models['oof_predictions']
                    logger.info(f"Found OOF predictions in traditional_models: {list(oof_predictions.keys())}")
            
            if not oof_predictions and 'oof_predictions' in training_results:
                oof_predictions = training_results['oof_predictions']
                logger.info(f"Found OOF predictions in training_results: {list(oof_predictions.keys())}")
        
        # 如果还是没有，尝试从模型的属性中获取
        if not oof_predictions:
            logger.warning("OOF predictions not found in training_results, trying model attributes...")
            # 检查模型是否有_last_training_results属性
            if hasattr(model, '_last_training_results'):
                last_res = model._last_training_results
                logger.info(f"Last training results keys: {list(last_res.keys()) if isinstance(last_res, dict) else 'N/A'}")
                if isinstance(last_res, dict) and 'oof_predictions' in last_res:
                    oof_predictions = last_res['oof_predictions']
                    logger.info(f"Found OOF predictions in _last_training_results: {list(oof_predictions.keys())}")
        
        if not oof_predictions:
            raise RuntimeError("Could not extract OOF predictions from training results")
        
        logger.info(f"Extracted OOF predictions for {len(oof_predictions)} models: {list(oof_predictions.keys())}")
        
        # 确保OOF预测是Series格式
        oof_predictions_series = {}
        for model_name, oof_pred in oof_predictions.items():
            if isinstance(oof_pred, pd.Series):
                oof_predictions_series[model_name] = oof_pred
            elif isinstance(oof_pred, np.ndarray):
                # 需要索引，尝试从训练数据中获取
                logger.warning(f"Model {model_name}: OOF predictions are numpy array, need to convert to Series")
                # 这里需要实际的索引，暂时跳过
                continue
            elif isinstance(oof_pred, list):
                logger.warning(f"Model {model_name}: OOF predictions are list, need to convert to Series")
                continue
            else:
                logger.warning(f"Model {model_name}: Unknown OOF prediction type: {type(oof_pred)}")
                continue
        
        return oof_predictions_series
        
    except Exception as e:
        logger.error(f"Failed to extract OOF predictions: {e}", exc_info=True)
        raise


def calculate_oof_metrics(
    oof_predictions: Dict[str, pd.Series],
    actuals: pd.Series
) -> Dict[str, Dict[str, float]]:
    """计算OOF预测的IC和Rank IC"""
    oof_metrics = {}
    
    for model_name, oof_pred in oof_predictions.items():
        if len(oof_pred) == 0:
            logger.warning(f"Model {model_name}: Empty OOF predictions")
            continue
        
        # 对齐索引
        common_idx = oof_pred.index.intersection(actuals.index)
        if len(common_idx) == 0:
            logger.warning(f"Model {model_name}: No common index between OOF predictions and actuals")
            logger.warning(f"  OOF index sample: {oof_pred.index[:5] if len(oof_pred) > 0 else 'empty'}")
            logger.warning(f"  Actuals index sample: {actuals.index[:5] if len(actuals) > 0 else 'empty'}")
            continue
        
        oof_pred_aligned = oof_pred.reindex(common_idx)
        actuals_aligned = actuals.reindex(common_idx)
        
        # 移除NaN
        valid_mask = ~(oof_pred_aligned.isna() | actuals_aligned.isna())
        oof_clean = oof_pred_aligned[valid_mask]
        actuals_clean = actuals_aligned[valid_mask]
        
        if len(oof_clean) < 10:
            logger.warning(f"Model {model_name}: Too few valid samples ({len(oof_clean)})")
            continue
        
        # 计算IC和Rank IC
        try:
            ic = pearsonr(oof_clean, actuals_clean)[0]
            rank_ic = spearmanr(oof_clean, actuals_clean)[0]
            
            oof_metrics[model_name] = {
                'ic': ic,
                'rank_ic': rank_ic,
                'n_samples': len(oof_clean)
            }
            
            logger.info(f"  {model_name}: IC={ic:.6f}, Rank IC={rank_ic:.6f}, N={len(oof_clean)}")
        except Exception as e:
            logger.warning(f"Model {model_name}: Failed to calculate metrics: {e}")
            continue
    
    return oof_metrics


def compare_oof_vs_test(
    oof_metrics: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, Any]]:
    """比较OOF预测和测试集预测的性能"""
    
    results = {}
    
    for model_name in set(oof_metrics.keys()) & set(test_metrics.keys()):
        oof_ic = oof_metrics[model_name]['ic']
        oof_rank_ic = oof_metrics[model_name]['rank_ic']
        test_ic = test_metrics[model_name]['ic']
        test_rank_ic = test_metrics[model_name]['rank_ic']
        
        ic_diff = oof_ic - test_ic
        rank_ic_diff = oof_rank_ic - test_rank_ic
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'=' * 80}")
        logger.info(f"OOF IC: {oof_ic:.6f}, Rank IC: {oof_rank_ic:.6f}")
        logger.info(f"Test IC: {test_ic:.6f}, Rank IC: {test_rank_ic:.6f}")
        logger.info(f"IC Difference: {ic_diff:.6f}")
        logger.info(f"Rank IC Difference: {rank_ic_diff:.6f}")
        
        # 检测泄露信号
        leakage_signals = []
        
        # 信号1: OOF IC远高于测试集IC（不应该）
        if oof_ic > test_ic + 0.1:
            leakage_signals.append(f"OOF IC ({oof_ic:.4f}) >> Test IC ({test_ic:.4f})")
            logger.warning(f"  WARNING: OOF IC is much higher than Test IC!")
        
        # 信号2: OOF Rank IC远高于测试集Rank IC（不应该）
        if oof_rank_ic > test_rank_ic + 0.1:
            leakage_signals.append(f"OOF Rank IC ({oof_rank_ic:.4f}) >> Test Rank IC ({test_rank_ic:.4f})")
            logger.warning(f"  WARNING: OOF Rank IC is much higher than Test Rank IC!")
        
        # 信号3: OOF IC异常高（可能看到了测试集信息）
        if oof_ic > 0.9:
            leakage_signals.append(f"OOF IC is extremely high ({oof_ic:.4f})")
            logger.warning(f"  WARNING: OOF IC is extremely high!")
        
        # 信号4: OOF和测试集IC差异过大（不应该）
        if abs(ic_diff) > 0.2:
            leakage_signals.append(f"Large IC difference ({ic_diff:.4f})")
            logger.warning(f"  WARNING: Large IC difference detected!")
        
        if leakage_signals:
            logger.warning(f"  LEAKAGE SIGNALS DETECTED:")
            for signal in leakage_signals:
                logger.warning(f"    - {signal}")
            results[model_name] = {
                'leakage_detected': True,
                'oof_ic': oof_ic,
                'oof_rank_ic': oof_rank_ic,
                'test_ic': test_ic,
                'test_rank_ic': test_rank_ic,
                'ic_diff': ic_diff,
                'rank_ic_diff': rank_ic_diff,
                'leakage_signals': leakage_signals
            }
        else:
            logger.info(f"  No leakage signals detected")
            results[model_name] = {
                'leakage_detected': False,
                'oof_ic': oof_ic,
                'oof_rank_ic': oof_rank_ic,
                'test_ic': test_ic,
                'test_rank_ic': test_rank_ic,
                'ic_diff': ic_diff,
                'rank_ic_diff': rank_ic_diff,
                'leakage_signals': []
            }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="实验5（完整版）: 使用完整的5-fold CV比较OOF预测和测试集预测性能")
    parser.add_argument("--report-df", type=str, required=True,
                       help="测试集评估报告路径 (report_df.csv)")
    parser.add_argument("--train-data", type=str, required=True,
                       help="训练数据路径")
    parser.add_argument("--train-start-date", type=str, required=True,
                       help="训练开始日期 (YYYY-MM-DD)")
    parser.add_argument("--train-end-date", type=str, required=True,
                       help="训练结束日期 (YYYY-MM-DD)")
    parser.add_argument("--data-file", type=str, required=True,
                       help="完整数据文件路径（用于提取actuals）")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("实验5（完整版）: OOF vs 测试集性能比较")
    logger.info("使用完整的5-fold CV训练获取真实的OOF预测")
    logger.info("=" * 80)
    
    logger.info("\n关键区别说明:")
    logger.info("  1. OOF预测: 在训练期内通过CV获得，用于训练第二层模型")
    logger.info("  2. 测试集预测: 在训练期外的未来数据上获得，用于评估真实泛化能力")
    logger.info("  3. 如果OOF IC >> 测试集IC，可能存在数据泄露")
    logger.info("")
    
    # 1. 加载测试集预测性能
    test_metrics = load_test_metrics(args.report_df)
    logger.info(f"\nLoaded test metrics for {len(test_metrics)} models")
    
    # 2. 提取OOF预测（使用完整的5-fold CV训练）
    logger.info("\nExtracting OOF predictions from FULL 5-fold CV training...")
    logger.info("This will run the complete training process...")
    
    oof_predictions = extract_oof_predictions_from_full_training(
        train_data_path=args.train_data,
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date
    )
    
    if not oof_predictions:
        logger.error("Failed to extract OOF predictions")
        return
    
    # 3. 加载实际值（用于计算OOF IC）
    logger.info(f"\nLoading actuals from {args.data_file}")
    df = pd.read_parquet(args.data_file)
    
    # 提取训练期的actuals
    train_start = pd.to_datetime(args.train_start_date)
    train_end = pd.to_datetime(args.train_end_date)
    
    train_data = df.loc[
        (df.index.get_level_values('date') >= train_start) & 
        (df.index.get_level_values('date') <= train_end)
    ].copy()
    
    actuals = train_data['target'].copy()
    logger.info(f"Loaded {len(actuals)} actual values from training period")
    
    # 4. 计算OOF预测的IC和Rank IC
    logger.info("\nCalculating OOF prediction metrics...")
    oof_metrics = calculate_oof_metrics(oof_predictions, actuals)
    logger.info(f"Calculated OOF metrics for {len(oof_metrics)} models")
    
    # 5. 比较OOF和测试集性能
    logger.info("\n" + "=" * 80)
    logger.info("Comparing OOF vs Test Performance")
    logger.info("=" * 80)
    
    comparison_results = compare_oof_vs_test(oof_metrics, test_metrics)
    
    # 6. 生成报告
    logger.info("\n" + "=" * 80)
    logger.info("实验5结果汇总（完整5-fold CV）")
    logger.info("=" * 80)
    
    total_models = len(comparison_results)
    leakage_models = sum(1 for r in comparison_results.values() if r['leakage_detected'])
    
    logger.info(f"总模型数: {total_models}")
    logger.info(f"检测到泄露信号的模型数: {leakage_models}")
    
    if leakage_models > 0:
        logger.warning("\nWARNING: 检测到潜在的数据泄露！")
        logger.warning("   建议检查：")
        logger.warning("   1. OOF预测是否使用了测试集信息")
        logger.warning("   2. 特征标准化是否使用了测试集统计量")
        logger.warning("   3. CV分割是否正确处理了时间顺序")
        
        logger.info("\n泄露详情:")
        for model_name, result in comparison_results.items():
            if result['leakage_detected']:
                logger.warning(f"\n{model_name}:")
                for signal in result['leakage_signals']:
                    logger.warning(f"  - {signal}")
    else:
        logger.info("\n未检测到明显的数据泄露")
    
    # 保存结果
    import json
    from datetime import datetime
    report_path = Path(f'experiment_5_full_5fold_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'oof_metrics': {k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv 
                               for kk, vv in v.items()} 
                           for k, v in oof_metrics.items()},
            'test_metrics': {k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv 
                                for kk, vv in v.items()} 
                            for k, v in test_metrics.items()},
            'comparison_results': {k: {kk: (float(vv) if isinstance(vv, (np.integer, np.floating)) else vv)
                                      for kk, vv in v.items()}
                                   for k, v in comparison_results.items()}
        }, f, indent=2, default=str)
    logger.info(f"\n结果已保存: {report_path}")


if __name__ == "__main__":
    main()
