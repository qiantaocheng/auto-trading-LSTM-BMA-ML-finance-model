#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Model Integration Helper
展示如何在BMA模型中集成真实的Stacking和LambdaRank因子贡献
"""

import pandas as pd
import numpy as np


def extract_stacking_contributions(ridge_stacker):
    """从Ridge Stacker提取真实因子贡献"""
    if hasattr(ridge_stacker, 'feature_importance_') and ridge_stacker.feature_importance_ is not None:
        # 从Ridge系数提取贡献
        contributions = {}
        for _, row in ridge_stacker.feature_importance_.iterrows():
            contributions[row['feature']] = float(row['coefficient'])
        return contributions

    return {}


def extract_lambda_contributions(lambda_ranker):
    """从LambdaRank模型提取真实因子贡献"""
    if hasattr(lambda_ranker, 'model') and lambda_ranker.model is not None:
        if hasattr(lambda_ranker, 'base_cols'):
            # 从LightGBM特征重要性提取贡献
            importance_scores = lambda_ranker.model.feature_importance()
            contributions = {}
            for i, feature_name in enumerate(lambda_ranker.base_cols):
                if i < len(importance_scores):
                    # 标准化重要性分数到0-1范围
                    normalized_score = importance_scores[i] / max(importance_scores) * 0.1
                    contributions[feature_name] = float(normalized_score)
            return contributions

    return {}


def create_enhanced_model_info(ridge_stacker=None, lambda_ranker=None, base_model_info=None):
    """创建包含真实因子贡献的model_info"""

    if base_model_info is None:
        base_model_info = {}

    enhanced_info = base_model_info.copy()

    # 提取真实贡献
    if ridge_stacker:
        stacking_contributions = extract_stacking_contributions(ridge_stacker)
        if stacking_contributions:
            enhanced_info['stacking_contributions'] = stacking_contributions
            print(f"✓ 提取到Stacking贡献: {len(stacking_contributions)}个因子")

    if lambda_ranker:
        lambda_contributions = extract_lambda_contributions(lambda_ranker)
        if lambda_contributions:
            enhanced_info['lambda_contributions'] = lambda_contributions
            print(f"✓ 提取到LambdaRank贡献: {len(lambda_contributions)}个因子")

    # 验证是否包含sentiment_score
    all_factors = set()
    if 'stacking_contributions' in enhanced_info:
        all_factors.update(enhanced_info['stacking_contributions'].keys())
    if 'lambda_contributions' in enhanced_info:
        all_factors.update(enhanced_info['lambda_contributions'].keys())

    if 'sentiment_score' in all_factors:
        enhanced_info['has_sentiment'] = True
        print("✓ 确认包含sentiment_score因子")
    else:
        enhanced_info['has_sentiment'] = False
        print("⚠ 未找到sentiment_score因子")

    # 添加贡献统计
    enhanced_info['total_factors'] = len(all_factors)
    enhanced_info['contribution_type'] = 'dual_model' if ('stacking_contributions' in enhanced_info and 'lambda_contributions' in enhanced_info) else 'single_model'

    return enhanced_info


def export_with_real_contributions(predictions, dates, tickers, ridge_stacker=None, lambda_ranker=None,
                                 base_model_info=None, filename=None):
    """使用真实因子贡献导出Excel"""

    from bma_models.corrected_prediction_exporter import CorrectedPredictionExporter

    # 创建增强的model_info
    enhanced_model_info = create_enhanced_model_info(
        ridge_stacker=ridge_stacker,
        lambda_ranker=lambda_ranker,
        base_model_info=base_model_info
    )

    # 导出
    exporter = CorrectedPredictionExporter()
    output_file = exporter.export_predictions(
        predictions=predictions,
        dates=dates,
        tickers=tickers,
        model_info=enhanced_model_info,
        filename=filename or 'enhanced_factor_contributions.xlsx'
    )

    print(f"✓ Excel导出完成: {output_file}")

    # 显示因子贡献摘要
    print_contribution_summary(enhanced_model_info)

    return output_file


def print_contribution_summary(model_info):
    """打印因子贡献摘要"""

    print("\n" + "=" * 60)
    print("因子贡献摘要")
    print("=" * 60)

    stacking_contrib = model_info.get('stacking_contributions', {})
    lambda_contrib = model_info.get('lambda_contributions', {})

    if stacking_contrib and lambda_contrib:
        print("模式: 双模型贡献对比")
        print(f"Stacking因子数: {len(stacking_contrib)}")
        print(f"LambdaRank因子数: {len(lambda_contrib)}")

        # 找出差异最大的因子
        common_factors = set(stacking_contrib.keys()) & set(lambda_contrib.keys())
        if common_factors:
            max_diff_factor = max(common_factors,
                                key=lambda f: abs(stacking_contrib[f] - lambda_contrib[f]))
            s_val = stacking_contrib[max_diff_factor]
            l_val = lambda_contrib[max_diff_factor]
            diff_pct = ((l_val - s_val) / s_val * 100) if s_val != 0 else 0

            print(f"最大差异因子: {max_diff_factor}")
            print(f"  Stacking: {s_val:.4f}")
            print(f"  LambdaRank: {l_val:.4f}")
            print(f"  差异: {diff_pct:+.1f}%")

    elif stacking_contrib:
        print("模式: Stacking单模型")
        print(f"因子数: {len(stacking_contrib)}")
        top_factor = max(stacking_contrib.items(), key=lambda x: abs(x[1]))
        print(f"最重要因子: {top_factor[0]} ({top_factor[1]:.4f})")

    elif lambda_contrib:
        print("模式: LambdaRank单模型")
        print(f"因子数: {len(lambda_contrib)}")
        top_factor = max(lambda_contrib.items(), key=lambda x: abs(x[1]))
        print(f"最重要因子: {top_factor[0]} ({top_factor[1]:.4f})")

    else:
        print("模式: 默认因子贡献")

    # 检查sentiment
    has_sentiment = model_info.get('has_sentiment', False)
    print(f"Sentiment因子: {'包含' if has_sentiment else '未包含'}")


# 使用示例
def example_usage():
    """展示如何使用新的双列因子贡献功能"""

    print("=" * 60)
    print("使用示例: 在BMA模型中集成真实因子贡献")
    print("=" * 60)

    print("""
在你的BMA训练流程中，可以这样使用：

# 1. 训练完成后提取模型
ridge_stacker = your_trained_ridge_stacker
lambda_ranker = your_trained_lambda_ranker

# 2. 创建基础model_info
base_info = {
    'model_type': 'BMA Enhanced Production',
    'n_samples': len(predictions),
    'n_features': 16,  # 15个alpha因子 + sentiment_score
    'training_time': '89.2s',
    'cv_score': 0.045
}

# 3. 使用增强导出功能
output_file = export_with_real_contributions(
    predictions=final_predictions,
    dates=prediction_dates,
    tickers=prediction_tickers,
    ridge_stacker=ridge_stacker,
    lambda_ranker=lambda_ranker,
    base_model_info=base_info,
    filename='production_dual_contributions.xlsx'
)

结果Excel将包含：
• 双列因子贡献表（Stacking vs LambdaRank）
• sentiment_score因子（如果训练数据中包含）
• 真实的模型系数和重要性分数
• 因子类别分类和统计摘要
""")


if __name__ == "__main__":
    example_usage()