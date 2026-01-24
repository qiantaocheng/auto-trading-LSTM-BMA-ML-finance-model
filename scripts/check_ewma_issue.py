#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查EWMA相关问题是否导致重复分数
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def main():
    """检查EWMA相关问题"""
    print("=" * 80)
    print("EWMA相关问题检查")
    print("=" * 80)
    
    # 检查replace_ewa_in_pipeline方法
    print("\n[1] 检查replace_ewa_in_pipeline方法")
    print("  位置: bma_models/meta_ranker_stacker.py line 542")
    print("  功能: 兼容性方法，用于pipeline集成")
    print("  逻辑:")
    print("    1. 验证输入DataFrame")
    print("    2. 提取特征列（base_cols）")
    print("    3. 过滤NaN行")
    print("    4. 调用predict()方法")
    print("    5. 返回预测结果")
    
    # 检查predict_with_snapshot中的调用
    print("\n[2] 检查predict_with_snapshot中的调用")
    print("  位置: bma_models/量化模型_bma_ultra_enhanced.py line 5717")
    print("  代码:")
    print("    meta_ranker_scores = self.meta_ranker_stacker.replace_ewa_in_pipeline(ridge_input)")
    print("    ridge_predictions = meta_ranker_scores['score']")
    
    # 检查ridge_input的构建
    print("\n[3] 检查ridge_input的构建")
    print("  问题: ridge_input可能包含所有相同的值")
    print("  可能原因:")
    print("    1. first_layer_preds所有值相同（第一层模型预测相同）")
    print("    2. first_layer_preds没有正确对齐到股票")
    print("    3. first_layer_preds被错误覆盖")
    
    # 检查第一层预测
    print("\n[4] 检查第一层预测")
    print("  位置: bma_models/量化模型_bma_ultra_enhanced.py line ~9750-10000")
    print("  第一层模型:")
    print("    - ElasticNet")
    print("    - XGBoost")
    print("    - CatBoost")
    print("    - LambdaRank")
    print("  如果这些模型的预测都相同 -> 问题在第一层")
    print("  如果这些模型的预测不同 -> 问题在MetaRankerStacker")
    
    # 检查特征对齐
    print("\n[5] 检查特征对齐问题")
    print("  可能问题:")
    print("    1. first_layer_preds的索引没有正确对齐")
    print("    2. 特征列缺失，被填充为相同值")
    print("    3. 特征列顺序错误")
    
    # 建议
    print("\n[6] 诊断建议")
    print("  [建议1] 检查日志中的first_layer_preds")
    print("    查找: [SNAPSHOT] Base predictions columns")
    print("    查找: [SNAPSHOT] LambdaRank non-null values")
    print("    查找: [SNAPSHOT] CatBoost non-null values")
    print("\n  [建议2] 检查ridge_input的唯一值")
    print("    在predict_with_snapshot中添加日志:")
    print("      logger.info(f'ridge_input unique values per column: {ridge_input.nunique()}')")
    print("\n  [建议3] 检查MetaRankerStacker的predict输出")
    print("    在meta_ranker_stacker.py的predict方法中添加日志:")
    print("      logger.info(f'predictions unique values: {predictions[\"score\"].nunique()}')")
    
    print("\n" + "=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
