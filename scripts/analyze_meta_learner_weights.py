#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析每个Meta-Learner的权重

分析RidgeStacker和MetaRankerStacker的权重，展示每个第一层模型预测的贡献度。
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from bma_models.model_registry import (
    load_manifest,
    load_weights_from_snapshot,
    load_models_from_snapshot,
    list_snapshots
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_ridge_stacker_weights(ridge_stacker, base_cols: list) -> Dict[str, Any]:
    """分析RidgeStacker的权重"""
    if ridge_stacker is None:
        return None
    
    try:
        # 获取Ridge模型的系数
        if hasattr(ridge_stacker, 'ridge_model') and ridge_stacker.ridge_model is not None:
            coefs = ridge_stacker.ridge_model.coef_
            intercept = ridge_stacker.ridge_model.intercept_ if hasattr(ridge_stacker.ridge_model, 'intercept_') else 0.0
            
            # 获取实际特征列
            feat_cols = list(getattr(ridge_stacker, 'actual_feature_cols_', []) or 
                           getattr(ridge_stacker, 'feature_names_', []) or 
                           base_cols)
            
            # 确保特征列和系数数量匹配
            if len(feat_cols) != len(coefs):
                logger.warning(f"特征列数量({len(feat_cols)})与系数数量({len(coefs)})不匹配，使用默认命名")
                feat_cols = [f'feature_{i}' for i in range(len(coefs))]
            
            weights_dict = dict(zip(feat_cols, coefs.flatten() if coefs.ndim > 1 else coefs))
            
            # 计算权重统计
            weights_array = np.array(list(weights_dict.values()))
            abs_weights = np.abs(weights_array)
            total_abs_weight = abs_weights.sum()
            
            # 归一化权重（按绝对值）
            normalized_weights = {k: abs_weights[i] / total_abs_weight if total_abs_weight > 0 else 0.0 
                                 for i, k in enumerate(weights_dict.keys())}
            
            return {
                'type': 'RidgeStacker',
                'coefficients': weights_dict,
                'normalized_weights': normalized_weights,
                'intercept': float(intercept),
                'total_abs_weight': float(total_abs_weight),
                'feature_count': len(weights_dict),
                'base_cols': base_cols
            }
        else:
            logger.warning("RidgeStacker没有ridge_model属性")
            return None
    except Exception as e:
        logger.error(f"分析RidgeStacker权重失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def analyze_meta_ranker_stacker_weights(meta_ranker_stacker, base_cols: list) -> Dict[str, Any]:
    """分析MetaRankerStacker的特征重要性"""
    if meta_ranker_stacker is None:
        return None
    
    try:
        # 获取LightGBM模型
        if hasattr(meta_ranker_stacker, 'lightgbm_model') and meta_ranker_stacker.lightgbm_model is not None:
            booster = meta_ranker_stacker.lightgbm_model
            
            # 获取特征重要性（gain类型）
            try:
                feature_importance = booster.feature_importance(importance_type='gain')
                feature_names = booster.feature_name() or []
            except Exception as e:
                logger.warning(f"无法从booster获取特征重要性: {e}")
                return None
            
            # 获取实际特征列
            feat_cols = list(getattr(meta_ranker_stacker, 'actual_feature_cols_', []) or 
                           getattr(meta_ranker_stacker, 'base_cols', []) or 
                           base_cols)
            
            # 映射特征名称
            if len(feature_names) == len(feature_importance):
                # 如果特征名称是f_0, f_1格式，尝试映射到实际列名
                if len(feat_cols) == len(feature_names):
                    weights_dict = dict(zip(feat_cols, feature_importance))
                else:
                    # 使用特征名称作为key
                    weights_dict = dict(zip(feature_names, feature_importance))
            else:
                # 如果数量不匹配，使用索引
                if len(feat_cols) == len(feature_importance):
                    weights_dict = dict(zip(feat_cols, feature_importance))
                else:
                    weights_dict = {f'feature_{i}': feature_importance[i] for i in range(len(feature_importance))}
            
            # 计算归一化权重
            importance_array = np.array(list(weights_dict.values()))
            total_importance = importance_array.sum()
            normalized_weights = {k: v / total_importance if total_importance > 0 else 0.0 
                                 for k, v in weights_dict.items()}
            
            # 获取其他模型信息
            best_iteration = getattr(booster, 'best_iteration', None)
            if best_iteration is None:
                best_iteration = getattr(meta_ranker_stacker, 'num_boost_round', None)
            
            return {
                'type': 'MetaRankerStacker',
                'feature_importance': weights_dict,
                'normalized_weights': normalized_weights,
                'total_importance': float(total_importance),
                'feature_count': len(weights_dict),
                'base_cols': base_cols,
                'best_iteration': best_iteration,
                'label_gain_power': getattr(meta_ranker_stacker, 'label_gain_power', None),
                'n_quantiles': getattr(meta_ranker_stacker, 'n_quantiles', None)
            }
        else:
            logger.warning("MetaRankerStacker没有lightgbm_model属性")
            return None
    except Exception as e:
        logger.error(f"分析MetaRankerStacker权重失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def format_weights_table(weights_data: Dict[str, Any], title: str) -> str:
    """格式化权重表格"""
    if weights_data is None:
        return f"\n{title}: 无数据\n"
    
    # 处理不同类型的数据结构
    if weights_data.get('type') == 'RidgeStacker' or 'coefficients' in weights_data:
        weights = weights_data.get('coefficients', weights_data.get('weights', {}))
        normalized = weights_data.get('normalized_weights', {})
        intercept = weights_data.get('intercept', 0.0)
    elif weights_data.get('type') == 'MetaRankerStacker' or 'feature_importance' in weights_data:
        weights = weights_data.get('feature_importance', weights_data.get('weights', {}))
        normalized = weights_data.get('normalized_weights', {})
        intercept = None
    else:
        # 通用处理：假设有weights字段
        weights = weights_data.get('weights', {})
        normalized = weights_data.get('normalized_weights', {})
        intercept = None
    
    # 创建DataFrame
    if not weights:
        return f"\n{title}: 权重数据为空\n"
    
    df = pd.DataFrame({
        '特征': list(weights.keys()),
        '原始权重': list(weights.values()),
        '归一化权重': [normalized.get(k, 0.0) if normalized else 0.0 for k in weights.keys()]
    })
    
    # 按归一化权重排序
    df = df.sort_values('归一化权重', ascending=False)
    
    # 格式化输出
    output = f"\n{'='*80}\n"
    output += f"{title}\n"
    output += f"{'='*80}\n"
    output += f"模型类型: {weights_data.get('type', 'Unknown')}\n"
    output += f"特征数量: {weights_data.get('feature_count', len(weights))}\n"
    
    if intercept is not None:
        output += f"截距: {intercept:.6f}\n"
    
    model_type = weights_data.get('type', '')
    if 'RidgeStacker' in model_type or 'ridge' in model_type.lower():
        if 'total_abs_weight' in weights_data:
            output += f"总绝对权重: {weights_data['total_abs_weight']:.6f}\n"
    else:
        if 'total_importance' in weights_data:
            output += f"总重要性: {weights_data['total_importance']:.6f}\n"
        if weights_data.get('best_iteration'):
            output += f"最佳迭代轮数: {weights_data['best_iteration']}\n"
        if weights_data.get('label_gain_power'):
            output += f"Label Gain Power: {weights_data['label_gain_power']}\n"
        if weights_data.get('n_quantiles'):
            output += f"Quantiles数量: {weights_data['n_quantiles']}\n"
    
    output += f"\n权重分布:\n"
    output += df.to_string(index=False, float_format=lambda x: f'{x:.6f}')
    output += f"\n\n权重占比分析:\n"
    
    # 计算累计权重
    cumsum = 0.0
    for idx, row in df.iterrows():
        cumsum += row['归一化权重']
        output += f"  {row['特征']:30s}: {row['归一化权重']*100:6.2f}% (累计: {cumsum*100:6.2f}%)\n"
    
    return output


def main(snapshot_id: Optional[str] = None):
    """主函数"""
    logger.info("="*80)
    logger.info("Meta-Learner权重分析")
    logger.info("="*80)
    
    # 列出所有快照
    try:
        snapshots = list_snapshots()
        if not snapshots:
            logger.error("未找到任何模型快照")
            return 1
        
        logger.info(f"\n找到 {len(snapshots)} 个快照:")
        for i, (sid, created_at, tag) in enumerate(snapshots[:10], 1):  # 只显示前10个
            import datetime
            dt = datetime.datetime.fromtimestamp(created_at)
            logger.info(f"  {i}. {sid[:8]}... | {dt.strftime('%Y-%m-%d %H:%M:%S')} | {tag}")
        
        # 使用最新快照或指定快照
        if snapshot_id is None:
            snapshot_id = snapshots[0][0]
            logger.info(f"\n使用最新快照: {snapshot_id[:8]}... (tag: {snapshots[0][2]})")
        else:
            logger.info(f"\n使用指定快照: {snapshot_id[:8]}...")
    except Exception as e:
        logger.error(f"列出快照失败: {e}")
        return 1
    
    # 加载manifest获取配置信息
    try:
        manifest = load_manifest(snapshot_id)
        logger.info(f"\n快照信息:")
        logger.info(f"  标签: {manifest.get('tag', 'N/A')}")
        logger.info(f"  创建时间: {manifest.get('created_at', 'N/A')}")
    except Exception as e:
        logger.warning(f"加载manifest失败: {e}")
        manifest = {}
    
    # 加载权重（从JSON文件）
    logger.info("\n从JSON文件加载权重...")
    weights_from_json = load_weights_from_snapshot(snapshot_id)
    
    # 加载模型（从pickle文件）
    logger.info("\n从模型文件加载模型...")
    try:
        models_data = load_models_from_snapshot(snapshot_id, load_catboost=False)
    except Exception as e:
        logger.warning(f"加载模型失败: {e}")
        models_data = {}
    
    # 分析结果
    results = {}
    
    # 1. 分析RidgeStacker权重（从JSON）
    if 'ridge_stacking' in weights_from_json and weights_from_json['ridge_stacking']:
        logger.info("\n✅ 找到RidgeStacker权重（JSON）")
        ridge_weights_json = weights_from_json['ridge_stacking']
        results['ridge_stacker_json'] = {
            'type': 'RidgeStacker (from JSON)',
            'weights': ridge_weights_json,
            'normalized_weights': {k: abs(v) / sum(abs(x) for x in ridge_weights_json.values()) 
                                  if sum(abs(x) for x in ridge_weights_json.values()) > 0 else 0.0
                                  for k, v in ridge_weights_json.items()}
        }
    
    # 2. 分析RidgeStacker权重（从模型对象）
    if 'ridge_stacker' in models_data and models_data['ridge_stacker'] is not None:
        logger.info("\n✅ 找到RidgeStacker模型对象")
        base_cols = manifest.get('meta_ranker', {}).get('base_cols', 
                    ['pred_catboost', 'pred_xgb', 'pred_lambdarank', 'pred_elastic'])
        ridge_analysis = analyze_ridge_stacker_weights(models_data['ridge_stacker'], base_cols)
        if ridge_analysis:
            results['ridge_stacker_model'] = ridge_analysis
    
    # 3. 分析MetaRankerStacker权重（从JSON）
    if 'ridge_stacking' in weights_from_json and weights_from_json['ridge_stacking']:
        # 检查是否是MetaRankerStacker（通过manifest判断）
        if 'meta_ranker' in manifest:
            logger.info("\n✅ 找到MetaRankerStacker权重（JSON）")
            meta_weights_json = weights_from_json['ridge_stacking']
            results['meta_ranker_json'] = {
                'type': 'MetaRankerStacker (from JSON)',
                'weights': meta_weights_json,
                'normalized_weights': {k: v / sum(meta_weights_json.values()) 
                                      if sum(meta_weights_json.values()) > 0 else 0.0
                                      for k, v in meta_weights_json.items()}
            }
    
    # 4. 分析MetaRankerStacker权重（从模型对象）
    # 注意：MetaRankerStacker也可能保存在ridge_stacker字段中
    if 'ridge_stacker' in models_data and models_data['ridge_stacker'] is not None:
        # 检查是否是MetaRankerStacker
        stacker = models_data['ridge_stacker']
        if hasattr(stacker, 'lightgbm_model') and stacker.lightgbm_model is not None:
            logger.info("\n✅ 找到MetaRankerStacker模型对象")
            base_cols = manifest.get('meta_ranker', {}).get('base_cols',
                        getattr(stacker, 'base_cols', ['pred_catboost', 'pred_xgb', 'pred_lambdarank', 'pred_elastic']))
            meta_analysis = analyze_meta_ranker_stacker_weights(stacker, list(base_cols))
            if meta_analysis:
                results['meta_ranker_model'] = meta_analysis
    
    # 输出结果
    logger.info("\n" + "="*80)
    logger.info("权重分析结果")
    logger.info("="*80)
    
    output_lines = []
    
    # RidgeStacker分析
    if 'ridge_stacker_model' in results:
        output_lines.append(format_weights_table(results['ridge_stacker_model'], 
                                                 "RidgeStacker权重分析（从模型对象）"))
    elif 'ridge_stacker_json' in results:
        output_lines.append(format_weights_table(results['ridge_stacker_json'], 
                                                 "RidgeStacker权重分析（从JSON文件）"))
    
    # MetaRankerStacker分析
    if 'meta_ranker_model' in results:
        output_lines.append(format_weights_table(results['meta_ranker_model'], 
                                                 "MetaRankerStacker权重分析（从模型对象）"))
    elif 'meta_ranker_json' in results:
        output_lines.append(format_weights_table(results['meta_ranker_json'], 
                                                 "MetaRankerStacker权重分析（从JSON文件）"))
    
    # 打印所有结果
    for line in output_lines:
        print(line)
        logger.info(line)
    
    # 保存结果到文件
    output_file = f"meta_learner_weights_analysis_{snapshot_id[:8]}.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Meta-Learner权重分析报告\n\n")
        f.write(f"**快照ID**: {snapshot_id}\n\n")
        f.write(f"**标签**: {manifest.get('tag', 'N/A')}\n\n")
        f.write("---\n\n")
        for line in output_lines:
            f.write(line + "\n")
    
    logger.info(f"\n✅ 分析结果已保存到: {output_file}")
    
    return 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='分析Meta-Learner权重')
    parser.add_argument('--snapshot-id', type=str, default=None, help='快照ID（默认使用最新）')
    args = parser.parse_args()
    
    sys.exit(main(args.snapshot_id))
