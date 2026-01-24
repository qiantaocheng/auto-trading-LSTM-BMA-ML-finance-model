#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验5（简化版）：比较OOF预测性能与测试集预测性能

简化方法：
1. 从report_df.csv获取测试集IC（已有）
2. 从训练日志或快照中提取OOF IC（如果可用）
3. 或者使用理论分析：如果OOF IC >> 测试集IC，可能存在泄露

如果无法直接获取OOF预测，我们可以：
- 分析训练过程中的CV分数
- 或者重新训练但只计算IC不保存模型
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
logger = logging.getLogger("experiment_5_simplified")


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


def estimate_oof_ic_from_cv_scores(
    train_data_path: str,
    train_start_date: str,
    train_end_date: str
) -> Dict[str, Dict[str, float]]:
    """
    通过快速CV训练估算OOF IC
    
    方法：只训练一个fold，计算该fold的验证集IC作为OOF IC的代理
    """
    logger.info("=" * 80)
    logger.info("Estimating OOF IC from CV training (single fold)")
    logger.info("=" * 80)
    
    try:
        from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
        from bma_models.unified_purged_cv_factory import create_unified_cv
        
        # 加载数据
        logger.info(f"Loading data: {train_data_path}")
        df = pd.read_parquet(train_data_path)
        
        # 过滤训练期数据
        train_start = pd.to_datetime(train_start_date)
        train_end = pd.to_datetime(train_end_date)
        
        train_data = df.loc[
            (df.index.get_level_values('date') >= train_start) & 
            (df.index.get_level_values('date') <= train_end)
        ].copy()
        
        logger.info(f"Training data: {len(train_data)} samples")
        
        # 准备特征和目标
        feature_cols = [c for c in train_data.columns if c not in ['target', 'Close', 'ret_fwd_5d', 'sector']]
        X = train_data[feature_cols].fillna(0)
        y = train_data['target'].copy()
        
        # 获取日期用于CV
        dates = pd.to_datetime(train_data.index.get_level_values('date')).values
        dates_norm = pd.to_datetime(dates).values.astype('datetime64[D]')
        
        # 创建CV分割器（只使用1个fold以加快速度）
        cv = create_unified_cv(n_splits=1, gap=10, embargo=10)
        cv_splits = list(cv.split(X.values, y.values, groups=dates_norm))
        
        if len(cv_splits) == 0:
            raise RuntimeError("No CV splits generated")
        
        train_idx, val_idx = cv_splits[0]
        logger.info(f"Using single fold: train={len(train_idx)}, val={len(val_idx)}")
        
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        oof_metrics = {}
        
        # 训练简单模型并计算OOF IC
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # ElasticNet (使用Ridge作为代理)
        logger.info("Training ElasticNet proxy...")
        model_en = Ridge(alpha=1.0)
        model_en.fit(X_train_scaled, y_train)
        pred_en = model_en.predict(X_val_scaled)
        ic_en = pearsonr(pred_en, y_val)[0]
        rank_ic_en = spearmanr(pred_en, y_val)[0]
        oof_metrics['elastic_net'] = {'ic': ic_en, 'rank_ic': rank_ic_en}
        logger.info(f"  ElasticNet: IC={ic_en:.6f}, Rank IC={rank_ic_en:.6f}")
        
        # XGBoost
        try:
            import xgboost as xgb
            logger.info("Training XGBoost...")
            model_xgb = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42, n_jobs=1)
            model_xgb.fit(X_train_scaled, y_train)
            pred_xgb = model_xgb.predict(X_val_scaled)
            ic_xgb = pearsonr(pred_xgb, y_val)[0]
            rank_ic_xgb = spearmanr(pred_xgb, y_val)[0]
            oof_metrics['xgboost'] = {'ic': ic_xgb, 'rank_ic': rank_ic_xgb}
            logger.info(f"  XGBoost: IC={ic_xgb:.6f}, Rank IC={rank_ic_xgb:.6f}")
        except Exception as e:
            logger.warning(f"XGBoost training failed: {e}")
        
        # CatBoost
        try:
            import catboost as cb
            logger.info("Training CatBoost...")
            model_cb = cb.CatBoostRegressor(iterations=50, depth=3, random_state=42, verbose=False)
            model_cb.fit(X_train_scaled, y_train)
            pred_cb = model_cb.predict(X_val_scaled)
            ic_cb = pearsonr(pred_cb, y_val)[0]
            rank_ic_cb = spearmanr(pred_cb, y_val)[0]
            oof_metrics['catboost'] = {'ic': ic_cb, 'rank_ic': rank_ic_cb}
            logger.info(f"  CatBoost: IC={ic_cb:.6f}, Rank IC={rank_ic_cb:.6f}")
        except Exception as e:
            logger.warning(f"CatBoost training failed: {e}")
        
        return oof_metrics
        
    except Exception as e:
        logger.error(f"Failed to estimate OOF IC: {e}", exc_info=True)
        return {}


def compare_oof_vs_test_simplified(
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
    parser = argparse.ArgumentParser(description="实验5（简化版）: 比较OOF预测和测试集预测性能")
    parser.add_argument("--report-df", type=str, required=True,
                       help="测试集评估报告路径 (report_df.csv)")
    parser.add_argument("--train-data", type=str, required=True,
                       help="训练数据路径")
    parser.add_argument("--train-start-date", type=str, required=True,
                       help="训练开始日期 (YYYY-MM-DD)")
    parser.add_argument("--train-end-date", type=str, required=True,
                       help="训练结束日期 (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("实验5（简化版）: OOF vs 测试集性能比较")
    logger.info("=" * 80)
    
    # 1. 加载测试集预测性能
    test_metrics = load_test_metrics(args.report_df)
    logger.info(f"\nLoaded test metrics for {len(test_metrics)} models")
    
    # 2. 估算OOF预测性能（使用单fold CV）
    logger.info("\nEstimating OOF IC from single-fold CV...")
    oof_metrics = estimate_oof_ic_from_cv_scores(
        train_data_path=args.train_data,
        train_start_date=args.train_start_date,
        train_end_date=args.train_end_date
    )
    
    if not oof_metrics:
        logger.error("Failed to estimate OOF metrics")
        return
    
    logger.info(f"\nEstimated OOF metrics for {len(oof_metrics)} models")
    
    # 3. 比较OOF和测试集性能
    logger.info("\n" + "=" * 80)
    logger.info("Comparing OOF vs Test Performance")
    logger.info("=" * 80)
    
    comparison_results = compare_oof_vs_test_simplified(oof_metrics, test_metrics)
    
    # 4. 生成报告
    logger.info("\n" + "=" * 80)
    logger.info("实验5结果汇总")
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
    report_path = Path(f'experiment_5_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
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
