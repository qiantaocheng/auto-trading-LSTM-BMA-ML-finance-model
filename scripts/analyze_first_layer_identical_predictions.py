#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析第一层模型预测为什么所有值相同
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def analyze_issue():
    """分析第一层预测相同的问题"""
    print("=" * 80)
    print("第一层模型预测相同问题 - 深度分析")
    print("=" * 80)
    
    print("\n[问题定位]")
    print("=" * 80)
    
    print("\n[1] 第一层模型预测流程")
    print("  位置: bma_models/量化模型_bma_ultra_enhanced.py line 9735-9783")
    print("\n  流程:")
    print("    1. X_df = X.copy()  (从feature_data构建)")
    print("    2. 对每个模型:")
    print("       - ElasticNet: line 9738-9751")
    print("       - XGBoost: line 9753-9767")
    print("       - CatBoost: line 9769-9783")
    print("       - LambdaRank: line ~9900+")
    
    print("\n[2] 关键问题点")
    print("=" * 80)
    
    print("\n  问题点1: 缺失特征填充为0.0")
    print("  位置: line 9745-9746, 9761-9762, 9777-9778")
    print("  代码:")
    print("    missing = [c for c in cols if c not in X_m.columns]")
    print("    for c in missing:")
    print("        X_m[c] = 0.0  # [WARN] 所有股票都是0.0")
    print("\n  影响:")
    print("    - 如果某个模型需要的特征列完全缺失")
    print("    - 所有股票的该列都被填充为0.0")
    print("    - 如果多个特征都缺失，所有股票的输入完全相同")
    print("    - 模型返回相同的预测值")
    
    print("\n  问题点2: X_df本身可能所有值相同")
    print("  位置: line 9723")
    print("  代码:")
    print("    X_df = X.copy()")
    print("    # X来自feature_data")
    print("\n  可能原因:")
    print("    - feature_data的所有特征值相同")
    print("    - 特征计算有bug")
    print("    - 特征对齐失败，使用了默认值")
    
    print("\n  问题点3: 特征对齐问题")
    print("  位置: line 9693")
    print("  代码:")
    print("    X, y, dates, tickers = self._prepare_standard_data_format(feature_data)")
    print("\n  可能问题:")
    print("    - _prepare_standard_data_format可能返回相同的特征值")
    print("    - 特征对齐失败，所有股票使用相同值")
    
    print("\n[3] 诊断步骤")
    print("=" * 80)
    
    print("\n  步骤1: 检查X_df的唯一值")
    print("    在predict_with_snapshot中添加:")
    print("      logger.info(f'X_df shape: {X_df.shape}')")
    print("      logger.info(f'X_df columns: {list(X_df.columns)}')")
    print("      for col in X_df.columns[:10]:  # 检查前10列")
    print("          logger.info(f'X_df[{col}] unique: {X_df[col].nunique()}, min={X_df[col].min():.6f}, max={X_df[col].max():.6f}')")
    
    print("\n  步骤2: 检查每个模型的输入特征")
    print("    在ElasticNet/XGBoost/CatBoost预测前添加:")
    print("      logger.info(f'{model_name} required cols: {cols}')")
    print("      logger.info(f'{model_name} missing cols: {missing}')")
    print("      logger.info(f'{model_name} X_m unique values per col: {X_m.nunique()}')")
    print("      logger.info(f'{model_name} X_m value range: min={X_m.min().min():.6f}, max={X_m.max().max():.6f}')")
    
    print("\n  步骤3: 检查每个模型的预测输出")
    print("    在预测后添加:")
    print("      logger.info(f'{model_name} predictions unique: {pred.nunique() if hasattr(pred, \"nunique\") else len(set(pred))}')")
    print("      logger.info(f'{model_name} predictions range: min={pred.min():.6f}, max={pred.max():.6f}')")
    
    print("\n[4] 可能的原因")
    print("=" * 80)
    
    print("\n  原因1: X_df所有特征值相同")
    print("    - feature_data本身有问题")
    print("    - 特征计算返回相同值")
    print("    - 特征对齐失败")
    
    print("\n  原因2: 缺失特征过多")
    print("    - 如果模型需要的特征大部分缺失")
    print("    - 所有股票都被填充为0.0")
    print("    - 输入完全相同，输出完全相同")
    
    print("\n  原因3: 模型本身问题")
    print("    - 模型权重损坏")
    print("    - 模型返回常数预测")
    print("    - 模型加载失败，返回默认值")
    
    print("\n  原因4: 特征对齐失败")
    print("    - _prepare_standard_data_format返回相同值")
    print("    - 特征索引对齐失败")
    print("    - 使用了错误的特征数据")
    
    print("\n[5] 修复建议")
    print("=" * 80)
    
    print("\n  修复1: 改进缺失特征填充")
    print("    不要用0.0填充，应该:")
    print("    - 使用横截面中位数")
    print("    - 或者使用训练期的特征均值")
    print("    - 或者记录警告并跳过该模型")
    
    print("\n  修复2: 添加输入验证")
    print("    在预测前验证:")
    print("    - X_df是否有变化")
    print("    - 每个模型的输入特征是否有变化")
    print("    - 如果所有值相同，记录错误")
    
    print("\n  修复3: 添加预测输出验证")
    print("    在预测后验证:")
    print("    - 预测值是否有变化")
    print("    - 如果所有值相同，记录错误")
    print("    - 使用第一层预测前检查唯一值")
    
    print("\n  修复4: 检查特征数据源")
    print("    验证:")
    print("    - feature_data是否正确加载")
    print("    - 特征计算是否正确")
    print("    - 特征对齐是否正确")
    
    print("\n" + "=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(analyze_issue())
