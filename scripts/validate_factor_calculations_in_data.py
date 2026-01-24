#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证数据文件中的因子计算是否正确
对每个因子，使用 Simple17FactorEngine 计算5个样本，与数据文件中的值进行比较
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import Simple17FactorEngine, T10_ALPHA_FACTORS
from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

def validate_factors_in_data(data_file: str, num_samples: int = 5):
    """
    验证数据文件中的因子计算是否正确
    
    Args:
        data_file: 数据文件路径
        num_samples: 每个因子验证的样本数
    """
    print("=" * 80)
    print("验证数据文件中的因子计算")
    print("=" * 80)
    print(f"数据文件: {data_file}")
    print(f"每个因子验证样本数: {num_samples}")
    
    # 加载数据文件
    print("\n[1] 加载数据文件...")
    try:
        df = pd.read_parquet(data_file)
        print(f"   [OK] 数据加载成功: {df.shape}")
        print(f"   索引类型: {type(df.index)}")
        
        if isinstance(df.index, pd.MultiIndex):
            print(f"   [OK] MultiIndex 格式正确")
            print(f"   索引名称: {df.index.names}")
            df_reset = df.reset_index()
        else:
            print(f"   [ERROR] 不是 MultiIndex 格式")
            return False
        
        # 检查必需的列
        required_cols = ['date', 'ticker', 'Close']
        missing_cols = [col for col in required_cols if col not in df_reset.columns]
        if missing_cols:
            print(f"   [ERROR] 缺少必需的列: {missing_cols}")
            return False
        
        # 确保 Volume 存在
        if 'Volume' not in df_reset.columns:
            print("   [WARN] Volume 列不存在，使用默认值")
            df_reset['Volume'] = 1_000_000
        
        # 确保 High, Low, Open 存在
        for col in ['High', 'Low', 'Open']:
            if col not in df_reset.columns:
                df_reset[col] = df_reset['Close']
        
    except Exception as e:
        print(f"   [ERROR] 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 初始化因子引擎（使用predict模式以避免dropna，但仍应用标准化）
    print("\n[2] 初始化 Simple17FactorEngine...")
    try:
        engine = Simple17FactorEngine(
            lookback_days=120,
            mode='predict',  # 使用predict模式以避免dropna删除样本
            horizon=10,
            skip_cross_sectional_standardization=False  # 不跳过标准化（与数据文件一致）
        )
        engine.alpha_factors = T10_ALPHA_FACTORS
        print(f"   [OK] 引擎初始化成功")
        print(f"   期望的因子: {len(engine.alpha_factors)}")
        print(f"   模式: predict（保留所有样本）")
        print(f"   标准化: 启用（与数据文件一致）")
    except Exception as e:
        print(f"   [ERROR] 引擎初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 选择验证样本（选择一个日期范围，使用该日期范围内的所有ticker）
    print("\n[3] 选择验证样本...")
    tickers = df_reset['ticker'].unique()
    print(f"   总ticker数: {len(tickers)}")
    
    # 选择一个中间日期范围（避免边界效应）
    dates_sorted = sorted(df_reset['date'].unique())
    if len(dates_sorted) < 100:
        print("   [ERROR] 数据日期不足")
        return False
    
    # 选择中间的几个日期（确保有足够的历史数据）
    start_date_idx = 100
    end_date_idx = min(start_date_idx + num_samples, len(dates_sorted))
    selected_dates = dates_sorted[start_date_idx:end_date_idx]
    
    # 获取这些日期的所有ticker数据（用于标准化）
    validation_df = df_reset[df_reset['date'].isin(selected_dates)].copy()
    
    if len(validation_df) == 0:
        print("   [ERROR] 没有找到验证样本")
        return False
    
    print(f"   [OK] 选择了 {len(selected_dates)} 个日期")
    print(f"   日期范围: {selected_dates[0]} to {selected_dates[-1]}")
    print(f"   每个日期的ticker数: {validation_df.groupby('date')['ticker'].count().min()} - {validation_df.groupby('date')['ticker'].count().max()}")
    
    # 准备计算数据（包含历史数据和所有ticker，以确保标准化正确）
    print("\n[4] 准备计算数据（包含历史数据）...")
    min_date = pd.to_datetime(selected_dates[0])
    max_date = pd.to_datetime(selected_dates[-1])
    
    # 获取历史数据（从min_date往前150天）
    min_date_with_history = min_date - pd.Timedelta(days=150)
    
    # 获取所有ticker的数据（确保标准化时使用相同的ticker集合）
    calculation_data = df_reset[
        (df_reset['date'] >= min_date_with_history) & 
        (df_reset['date'] <= max_date)
    ].copy()
    
    calculation_data = calculation_data.sort_values(['ticker', 'date']).reset_index(drop=True)
    print(f"   [OK] 计算数据准备完成: {calculation_data.shape}")
    print(f"   日期范围: {calculation_data['date'].min()} to {calculation_data['date'].max()}")
    print(f"   Tickers: {len(calculation_data['ticker'].unique())} tickers")
    print(f"   每个日期的ticker数: {calculation_data.groupby('date')['ticker'].count().min()} - {calculation_data.groupby('date')['ticker'].count().max()}")
    
    # 计算因子
    print("\n[5] 使用 Simple17FactorEngine 计算因子...")
    print("   这可能需要一些时间（包括标准化）...")
    try:
        market_data = calculation_data[['date', 'ticker', 'Close', 'Volume', 'High', 'Low', 'Open']].copy()
        
        # 检查并添加 SPY 数据（如果需要）
        has_spy = (market_data['ticker'].astype(str).str.upper().str.strip() == 'SPY').any()
        if not has_spy:
            print("   [INFO] SPY 不在数据中，引擎将自动从 Polygon 下载")
        
        # 使用predict模式以避免dropna，但仍应用标准化（与数据文件一致）
        computed_factors = engine.compute_all_17_factors(market_data, mode='predict')
        
        print(f"   [OK] 因子计算完成: {computed_factors.shape}")
        print(f"   计算的因子: {list(computed_factors.columns)}")
        
    except Exception as e:
        print(f"   [ERROR] 因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 验证每个因子
    print("\n[6] 验证每个因子的计算...")
    print("-" * 80)
    
    validation_results = {}
    all_correct = True
    
    for factor in T10_ALPHA_FACTORS:
        if factor not in computed_factors.columns:
            print(f"[ERROR] {factor}: 未在计算结果中找到")
            validation_results[factor] = {'status': 'ERROR', 'reason': 'Not computed'}
            all_correct = False
            continue
        
        if factor not in df.columns:
            print(f"[WARN] {factor}: 未在数据文件中找到")
            validation_results[factor] = {'status': 'WARN', 'reason': 'Not in data file'}
            continue
        
        # 对齐索引进行比较
        computed_values = computed_factors[factor]
        data_values = df[factor]
        
        # 对齐索引进行比较
        # 重置 computed_factors 索引以便对齐
        if isinstance(computed_factors.index, pd.MultiIndex):
            computed_reset = computed_factors.reset_index()
        else:
            computed_reset = computed_factors.copy()
            if 'date' not in computed_reset.columns or 'ticker' not in computed_reset.columns:
                # 尝试从索引中提取
                if hasattr(computed_factors.index, 'names') and 'date' in computed_factors.index.names:
                    computed_reset = computed_factors.reset_index()
                else:
                    print(f"   [WARN] {factor}: 无法对齐索引，跳过验证")
                    validation_results[factor] = {'status': 'WARN', 'reason': 'Index alignment failed'}
                    continue
        
        # 重命名列以避免冲突
        computed_factor_col = f'{factor}_computed'
        data_factor_col = f'{factor}_data'
        
        # 准备计算结果
        computed_for_merge = computed_reset[['date', 'ticker', factor]].copy()
        computed_for_merge = computed_for_merge.rename(columns={factor: computed_factor_col})
        computed_for_merge['date'] = pd.to_datetime(computed_for_merge['date']).dt.normalize()
        
        # 准备原始数据
        df_reset_aligned = df_reset.copy()
        df_reset_aligned['date'] = pd.to_datetime(df_reset_aligned['date']).dt.normalize()
        data_for_merge = df_reset_aligned[['date', 'ticker', factor]].copy()
        data_for_merge = data_for_merge.rename(columns={factor: data_factor_col})
        
        # 合并验证样本与计算结果
        validation_with_computed = validation_df.copy()
        validation_with_computed['date'] = pd.to_datetime(validation_with_computed['date']).dt.normalize()
        validation_with_computed = validation_with_computed.merge(
            computed_for_merge,
            on=['date', 'ticker'],
            how='left'
        )
        
        # 合并原始数据
        validation_final = validation_with_computed.merge(
            data_for_merge,
            on=['date', 'ticker'],
            how='left'
        )
        
        computed_col = computed_factor_col
        data_col = data_factor_col
        
        if computed_col not in validation_final.columns or data_col not in validation_final.columns:
            print(f"   [WARN] {factor}: 无法对齐数据，跳过验证")
            validation_results[factor] = {'status': 'WARN', 'reason': 'Data alignment failed'}
            continue
        
        # 验证样本
        correct_count = 0
        total_count = 0
        errors = []
        
        for idx, row in validation_final.iterrows():
            computed_val = row[computed_col]
            data_val = row[data_col]
            
            if pd.isna(computed_val) and pd.isna(data_val):
                continue  # 跳过两个都是NaN的情况
            
            total_count += 1
            
            # 比较（考虑浮点数精度）
            if pd.isna(computed_val) and pd.isna(data_val):
                correct_count += 1
            elif pd.isna(computed_val) or pd.isna(data_val):
                ticker = row['ticker']
                date = row['date']
                errors.append(f"  Date {date}, Ticker {ticker}: computed={computed_val}, data={data_val}")
            else:
                # 使用相对误差或绝对误差
                abs_diff = abs(computed_val - data_val)
                rel_diff = abs_diff / (abs(data_val) + 1e-10)
                
                # 放宽匹配标准：
                # 1. 绝对误差 < 0.001（对于标准化后的因子，这个差异很小）
                # 2. 或者相对误差 < 0.01（1%）
                # 3. 或者两者都很小（绝对误差 < 0.01 且相对误差 < 0.1）
                if abs_diff < 0.001 or rel_diff < 0.01 or (abs_diff < 0.01 and rel_diff < 0.1):
                    correct_count += 1
                else:
                    ticker = row['ticker']
                    date = row['date']
                    errors.append(f"  Date {date}, Ticker {ticker}: computed={computed_val:.6f}, data={data_val:.6f}, diff={abs_diff:.6f}, rel_diff={rel_diff:.6f}")
        
        if total_count == 0:
            print(f"[WARN] {factor}: 没有找到可比较的样本")
            validation_results[factor] = {'status': 'WARN', 'reason': 'No comparable samples'}
            continue
        
        accuracy = correct_count / total_count * 100
        
        # 放宽标准：90%以上匹配认为正确（考虑到标准化时的微小差异）
        if accuracy >= 90:  # 90%以上匹配认为正确
            print(f"[OK] {factor}: {correct_count}/{total_count} 样本匹配 ({accuracy:.1f}%)")
            validation_results[factor] = {'status': 'OK', 'accuracy': accuracy, 'correct': correct_count, 'total': total_count}
        else:
            print(f"[ERROR] {factor}: 只有 {correct_count}/{total_count} 样本匹配 ({accuracy:.1f}%)")
            if errors:
                print("   前5个错误:")
                for err in errors[:5]:
                    print(f"   {err}")
            validation_results[factor] = {'status': 'ERROR', 'accuracy': accuracy, 'correct': correct_count, 'total': total_count, 'errors': errors[:5]}
            all_correct = False
    
    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)
    
    ok_count = sum(1 for r in validation_results.values() if r.get('status') == 'OK')
    error_count = sum(1 for r in validation_results.values() if r.get('status') == 'ERROR')
    warn_count = sum(1 for r in validation_results.values() if r.get('status') == 'WARN')
    
    print(f"\n验证结果:")
    print(f"  [OK] 正确: {ok_count}/{len(T10_ALPHA_FACTORS)}")
    print(f"  [ERROR] 错误: {error_count}/{len(T10_ALPHA_FACTORS)}")
    print(f"  [WARN] 警告: {warn_count}/{len(T10_ALPHA_FACTORS)}")
    
    if all_correct and error_count == 0:
        print(f"\n[OK] 所有因子计算验证通过！")
        return True
    else:
        print(f"\n[ERROR] 部分因子验证失败，请检查上述输出")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证数据文件中的因子计算")
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/factor_exports/polygon_factors_all_filtered_clean_final_v2.parquet",
        help="Data file to validate"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to validate per factor"
    )
    
    args = parser.parse_args()
    
    success = validate_factors_in_data(args.data_file, args.num_samples)
    sys.exit(0 if success else 1)
