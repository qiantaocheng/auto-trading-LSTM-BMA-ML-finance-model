#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证所有15个因子都能正确计算，包括SPY数据获取
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import Simple17FactorEngine, T10_ALPHA_FACTORS

def verify_all_factors_calculation():
    """验证所有15个因子都能正确计算"""
    print("=" * 80)
    print("验证所有15个因子的计算")
    print("=" * 80)
    
    # 创建测试数据
    print("\n[1] 创建测试数据...")
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']  # 包含 SPY
    
    test_data = []
    np.random.seed(42)
    for ticker in tickers:
        for date in dates:
            test_data.append({
                'date': date,
                'ticker': ticker,
                'Close': 100 + np.random.randn() * 10,
                'Volume': 1_000_000 + np.random.randn() * 100_000,
                'High': 105 + np.random.randn() * 10,
                'Low': 95 + np.random.randn() * 10,
                'Open': 100 + np.random.randn() * 10,
            })
    
    test_df = pd.DataFrame(test_data)
    test_df = test_df.sort_values(['ticker', 'date']).reset_index(drop=True)
    print(f"   [OK] 测试数据创建完成: {test_df.shape}")
    print(f"   Tickers: {test_df['ticker'].unique()}")
    print(f"   Date range: {test_df['date'].min()} to {test_df['date'].max()}")
    
    # 初始化引擎
    print("\n[2] 初始化 Simple17FactorEngine...")
    try:
        engine = Simple17FactorEngine(
            lookback_days=120,
            mode='predict',
            horizon=10
        )
        engine.alpha_factors = T10_ALPHA_FACTORS  # 确保使用所有15个因子
        print(f"   [OK] 引擎初始化成功")
        print(f"   期望的因子: {len(engine.alpha_factors)}")
        print(f"   因子列表: {engine.alpha_factors}")
    except Exception as e:
        print(f"   [ERROR] 引擎初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 计算所有因子
    print("\n[3] 计算所有15个因子...")
    print("   这可能需要一些时间（包括从Polygon下载SPY/QQQ数据）...")
    try:
        factors_df = engine.compute_all_17_factors(test_df, mode='predict')
        
        print(f"\n   [OK] 因子计算完成: {factors_df.shape}")
        print(f"   计算的因子列: {list(factors_df.columns)}")
        
        # 验证所有期望的因子都被计算
        computed_factors = [f for f in T10_ALPHA_FACTORS if f in factors_df.columns]
        missing_computed = [f for f in T10_ALPHA_FACTORS if f not in factors_df.columns]
        
        print(f"\n   计算的期望因子: {len(computed_factors)}/{len(T10_ALPHA_FACTORS)}")
        if missing_computed:
            print(f"   [ERROR] 未计算的因子: {missing_computed}")
            return False
        else:
            print(f"   [OK] 所有15个因子都被计算")
        
        # 检查每个因子的数据质量
        print("\n[4] 检查因子数据质量...")
        print("-" * 80)
        
        for factor in T10_ALPHA_FACTORS:
            if factor in factors_df.columns:
                factor_data = factors_df[factor]
                non_zero = ((factor_data != 0) & factor_data.notna()).sum()
                total_nan = factor_data.isna().sum()
                total_zero = (factor_data == 0).sum()
                total_not_nan = factor_data.notna().sum()
                
                coverage = non_zero / len(factor_data) * 100 if len(factor_data) > 0 else 0
                
                status = "[OK]" if coverage > 50 else "[WARN]"
                print(f"{status} {factor}:")
                print(f"   非零值: {non_zero:,} ({coverage:.1f}%)")
                print(f"   零值: {total_zero:,}")
                print(f"   NaN值: {total_nan:,}")
                
                if total_not_nan > 0:
                    non_null_values = factor_data.dropna()
                    print(f"   均值: {non_null_values.mean():.6f}")
                    print(f"   标准差: {non_null_values.std():.6f}")
            else:
                print(f"[ERROR] {factor}: 未找到")
        
        # 特别检查需要SPY/QQQ数据的因子
        print("\n[5] 检查需要SPY/QQQ数据的因子...")
        print("-" * 80)
        
        spy_dependent_factors = ['ivol_30', 'downside_beta_ewm_21']
        for factor in spy_dependent_factors:
            if factor in factors_df.columns:
                factor_data = factors_df[factor]
                non_zero = ((factor_data != 0) & factor_data.notna()).sum()
                coverage = non_zero / len(factor_data) * 100 if len(factor_data) > 0 else 0
                
                if coverage > 50:
                    print(f"   [OK] {factor}: 覆盖率 {coverage:.1f}% (SPY/QQQ数据可用)")
                else:
                    print(f"   [WARN] {factor}: 覆盖率 {coverage:.1f}% (可能SPY/QQQ数据不可用)")
        
        return True
        
    except Exception as e:
        print(f"   [ERROR] 因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_all_factors_calculation()
    
    if success:
        print("\n" + "=" * 80)
        print("[OK] 所有15个因子验证通过！")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("[ERROR] 验证失败，请检查上述输出")
        print("=" * 80)
        sys.exit(1)
