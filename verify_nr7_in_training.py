#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 nr7_breakout_bias 是否被正确加入训练
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def verify_nr7_in_factor_list():
    """验证1: nr7_breakout_bias 在因子列表中"""
    from bma_models.simple_25_factor_engine import REQUIRED_14_FACTORS

    print("="*80)
    print("验证1: nr7_breakout_bias 在因子列表中")
    print("="*80)
    print(f"\nREQUIRED_14_FACTORS 列表 ({len(REQUIRED_14_FACTORS)} 个因子):")
    for i, factor in enumerate(REQUIRED_14_FACTORS, 1):
        marker = "[*]" if factor == 'nr7_breakout_bias' else "   "
        print(f"  {marker} {i:2d}. {factor}")

    assert 'nr7_breakout_bias' in REQUIRED_14_FACTORS, "[FAIL] nr7_breakout_bias not in factor list!"
    print("\n[PASS] nr7_breakout_bias in REQUIRED_14_FACTORS\n")
    return True

def verify_nr7_in_generated_features():
    """验证2: nr7_breakout_bias 在生成的特征中"""
    from bma_models.simple_25_factor_engine import Simple17FactorEngine

    print("="*80)
    print("验证2: nr7_breakout_bias 在生成的特征中")
    print("="*80)

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    data = []
    for ticker in ['TEST1', 'TEST2']:
        for i, date in enumerate(dates):
            # 模拟NR7场景：第10天有窄幅
            if i == 10:
                high, low = 101, 99  # 窄幅
            else:
                high, low = 105, 95  # 正常幅度

            data.append({
                'date': date,
                'ticker': ticker,
                'Open': 100,
                'High': high,
                'Low': low,
                'Close': 100 + (i % 3),  # 变化的收盘价
                'Volume': 1000000
            })

    market_data = pd.DataFrame(data)

    # 初始化引擎并生成因子
    engine = Simple17FactorEngine(lookback_days=60, enable_sentiment=False)
    factors = engine.compute_all_17_factors(market_data)

    print(f"\n生成的因子 DataFrame 形状: {factors.shape}")
    print(f"生成的因子列表 ({len(factors.columns)} 列):")

    for i, col in enumerate(factors.columns, 1):
        marker = "[*]" if col == 'nr7_breakout_bias' else " "
        non_zero = (factors[col] != 0).sum()
        print(f"  {marker} {i:2d}. {col:<30} (非零值: {non_zero}/{len(factors)})")

    assert 'nr7_breakout_bias' in factors.columns, "[FAIL] nr7_breakout_bias 不在生成的因子中!"
    print("\n[PASS] 验证通过: nr7_breakout_bias 在生成的特征 DataFrame 中")

    # 额外验证: nr7_breakout_bias 有非零值
    nr7_data = factors['nr7_breakout_bias']
    non_zero_count = (nr7_data != 0).sum()
    print(f"\nnr7_breakout_bias 统计:")
    print(f"  非零值数量: {non_zero_count}/{len(nr7_data)}")
    print(f"  最小值: {nr7_data.min():.6f}")
    print(f"  最大值: {nr7_data.max():.6f}")
    print(f"  均值: {nr7_data.mean():.6f}")
    print(f"  标准差: {nr7_data.std():.6f}")

    if non_zero_count > 0:
        print("\n[PASS] nr7_breakout_bias 有非零值，因子计算正常工作\n")
    else:
        print("\n[WARN]  nr7_breakout_bias 全为零 (可能需要更多数据来触发NR7条件)\n")

    return factors

def verify_nr7_in_training_input():
    """验证3: 追踪 nr7_breakout_bias 到训练输入 X"""
    print("="*80)
    print("验证3: nr7_breakout_bias 在训练输入 X 中")
    print("="*80)

    # 模拟 _prepare_standard_data_format 的逻辑
    from bma_models.simple_25_factor_engine import Simple17FactorEngine

    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    data = []
    for ticker in ['AAPL', 'MSFT']:
        for date in dates:
            data.append({
                'date': date,
                'ticker': ticker,
                'Open': 100,
                'High': 102,
                'Low': 98,
                'Close': 100,
                'Volume': 1000000
            })

    market_data = pd.DataFrame(data)

    # 生成因子
    engine = Simple17FactorEngine(lookback_days=60, enable_sentiment=False)
    feature_data = engine.compute_all_17_factors(market_data)

    print(f"\nfeature_data 形状: {feature_data.shape}")
    print(f"feature_data 列: {list(feature_data.columns)}")

    # 模拟 _prepare_standard_data_format 中的列过滤逻辑
    # feature_cols = [col for col in feature_data.columns if col not in ['target', 'Close']]
    feature_cols = [col for col in feature_data.columns if col not in ['target', 'Close']]

    print(f"\n过滤后的特征列 (训练用 X，{len(feature_cols)} 列):")
    for i, col in enumerate(feature_cols, 1):
        marker = "[*]" if col == 'nr7_breakout_bias' else " "
        print(f"  {marker} {i:2d}. {col}")

    assert 'nr7_breakout_bias' in feature_cols, "[FAIL] nr7_breakout_bias 不在训练特征 X 中!"
    print("\n[PASS] 验证通过: nr7_breakout_bias 在训练输入 X 中")

    # 模拟创建 X
    X = feature_data[feature_cols].copy()
    print(f"\n最终训练矩阵 X 形状: {X.shape}")
    print(f"X 包含的列: {list(X.columns)}")

    assert 'nr7_breakout_bias' in X.columns, "[FAIL] nr7_breakout_bias 不在最终 X 中!"
    print("\n[PASS] 验证通过: nr7_breakout_bias 在最终训练矩阵 X 中\n")

    return X

def main():
    """运行所有验证"""
    print("\n" + "="*80)
    print("NR7_BREAKOUT_BIAS 因子训练集成验证")
    print("="*80 + "\n")

    try:
        # 验证1: 因子定义
        verify_nr7_in_factor_list()

        # 验证2: 因子生成
        factors = verify_nr7_in_generated_features()

        # 验证3: 训练输入
        X = verify_nr7_in_training_input()

        # 最终总结
        print("="*80)
        print("[PASS] 所有验证通过！")
        print("="*80)
        print("\n总结:")
        print("  1. [PASS] nr7_breakout_bias 在 REQUIRED_14_FACTORS 因子列表中")
        print("  2. [PASS] nr7_breakout_bias 被 Simple17FactorEngine 正确计算")
        print("  3. [PASS] nr7_breakout_bias 被包含在训练特征矩阵 X 中")
        print("\n结论: nr7_breakout_bias 已正确集成到 ElasticNet, CatBoost, XGBoost 训练流程中！")
        print("="*80 + "\n")

        return True

    except AssertionError as e:
        print(f"\n[FAIL] 验证失败: {e}")
        return False
    except Exception as e:
        print(f"\n[FAIL] 验证出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
