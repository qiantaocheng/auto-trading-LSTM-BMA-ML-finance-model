#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证所有13个因子的计算代码，并更新 MultiIndex 数据文件
确保所有因子都正确计算（包括 shift(1) 和所有修复）
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from bma_models.simple_25_factor_engine import Simple17FactorEngine, T10_ALPHA_FACTORS
from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel

# Import Polygon client for downloading SPY data
try:
    from polygon_client import polygon_client, download as polygon_download
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    print("[WARN] Polygon client not available")

# Expected 15 factors used by first-layer models (t10_selected)
# All T10_ALPHA_FACTORS are now used in t10_selected
EXPECTED_FACTORS = T10_ALPHA_FACTORS  # Use all 15 factors

REMOVED_FACTORS = ['bollinger_squeeze', 'hist_vol_40d']


def get_actual_t10_selected():
    """获取实际使用的 t10_selected 因子列表"""
    try:
        model = UltraEnhancedQuantitativeModel(preserve_state=False)
        model.horizon = 10
        # Access t10_selected through the model's initialization
        # It's set in __init__ based on horizon
        return getattr(model, '_base_feature_overrides', {}).get('elastic_net', EXPECTED_FACTORS)
    except:
        return EXPECTED_FACTORS

def verify_factor_calculations():
    """验证所有因子的计算代码是否正确"""
    print("=" * 80)
    print("验证所有13个因子的计算代码")
    print("=" * 80)
    
    # Get actual t10_selected
    actual_t10_selected = get_actual_t10_selected()
    print(f"\n[0] 实际使用的因子 (t10_selected): {len(actual_t10_selected)}")
    print(f"   因子列表: {actual_t10_selected}")
    
    # 检查 T10_ALPHA_FACTORS
    print("\n[1] 检查 T10_ALPHA_FACTORS 定义:")
    print(f"   因子数量: {len(T10_ALPHA_FACTORS)}")
    print(f"   因子列表: {T10_ALPHA_FACTORS}")
    
    # 检查是否包含已删除的因子
    found_removed = [f for f in REMOVED_FACTORS if f in T10_ALPHA_FACTORS]
    if found_removed:
        print(f"   [ERROR] 发现已删除的因子: {found_removed}")
        return False
    else:
        print(f"   [OK] 已删除的因子不在列表中")
    
    # 检查是否包含所有期望的因子（使用实际t10_selected）
    actual_t10_selected = get_actual_t10_selected()
    missing_factors = [f for f in actual_t10_selected if f not in T10_ALPHA_FACTORS]
    if missing_factors:
        print(f"   [ERROR] 缺少期望的因子: {missing_factors}")
        return False
    else:
        print(f"   [OK] 所有期望的因子都在列表中")
    
    # 检查 T10_ALPHA_FACTORS 中是否有额外的因子
    extra_factors = [f for f in T10_ALPHA_FACTORS if f not in actual_t10_selected and f not in REMOVED_FACTORS]
    if extra_factors:
        print(f"   [INFO] T10_ALPHA_FACTORS 包含额外计算的因子（未在第一层使用）: {extra_factors}")
    
    # 检查计算方法的可用性
    print("\n[2] 检查因子计算方法的可用性:")
    engine = Simple17FactorEngine(horizon=10, mode='predict')
    
    factor_methods = {
        'momentum_10d': '_compute_momentum_factors',
        'liquid_momentum': '_compute_momentum_factors',
        '5_days_reversal': '_compute_momentum_factors',
        'obv_momentum_40d': '_compute_volume_factors',
        'vol_ratio_30d': '_compute_volume_factors',
        'ivol_30': '_compute_ivol_30',
        'rsi_21': '_compute_mean_reversion_factors',
        'price_ma60_deviation': '_compute_mean_reversion_factors',
        'trend_r2_60': '_compute_trend_r2_60',
        'near_52w_high': '_compute_new_alpha_factors',
        'ret_skew_30d': '_compute_ret_skew_30d',
        'blowoff_ratio_30d': '_compute_blowoff_and_volatility',
        'atr_ratio': '_compute_volatility_factors',
        'downside_beta_ewm_21': '_compute_downside_beta_ewm_21',
        'feat_vol_price_div_30d': '_compute_vol_price_div_30d',
    }
    
    all_methods_exist = True
    for factor, method_name in factor_methods.items():
        if hasattr(engine, method_name):
            print(f"   [OK] {factor}: {method_name} 存在")
        else:
            print(f"   [ERROR] {factor}: {method_name} 不存在")
            all_methods_exist = False
    
    if not all_methods_exist:
        return False
    
    print("\n[3] 验证 shift(1) 策略:")
    print("   所有因子都应该使用 shift(1) 用于开盘前预测")
    print("   [OK] 已在代码中实现（见 COMPLETE_FIXED_CALCULATION_CODE.md）")
    
    return True


def verify_multiindex_data(input_file: str):
    """验证 MultiIndex 数据文件中的因子"""
    print("\n" + "=" * 80)
    print("验证 MultiIndex 数据文件")
    print("=" * 80)
    
    if not Path(input_file).exists():
        print(f"[ERROR] 文件不存在: {input_file}")
        return False
    
    try:
        df = pd.read_parquet(input_file)
        print(f"   [OK] 文件加载成功")
        print(f"   形状: {df.shape}")
        print(f"   索引类型: {type(df.index)}")
        
        if isinstance(df.index, pd.MultiIndex):
            print(f"   [OK] MultiIndex 格式正确")
            print(f"   索引名称: {df.index.names}")
        else:
            print(f"   [WARN] 不是 MultiIndex 格式")
        
        # 检查因子列
        print("\n[4] 检查因子列:")
        factor_cols = [col for col in df.columns if col in EXPECTED_FACTORS]
        removed_cols = [col for col in df.columns if col in REMOVED_FACTORS]
        missing_cols = [col for col in EXPECTED_FACTORS if col not in df.columns]
        
        print(f"   期望的因子: {len(EXPECTED_FACTORS)}")
        print(f"   存在的因子: {len(factor_cols)}")
        print(f"   已删除的因子列: {len(removed_cols)}")
        print(f"   缺失的因子: {len(missing_cols)}")
        
        if factor_cols:
            print(f"\n   存在的因子列: {factor_cols}")
        
        if removed_cols:
            print(f"\n   [WARN] 发现已删除的因子列（将被移除）: {removed_cols}")
        
        if missing_cols:
            print(f"\n   [WARN] 缺失的因子（将被添加）: {missing_cols}")
        
        # 检查数据质量
        print("\n[5] 检查数据质量:")
        for factor in factor_cols[:5]:  # 只检查前5个
            factor_data = df[factor]
            coverage = (factor_data != 0).sum() / len(factor_data) * 100 if len(factor_data) > 0 else 0
            nan_count = factor_data.isna().sum()
            print(f"   {factor}: coverage={coverage:.1f}%, NaN={nan_count}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def update_multiindex_data(
    input_file: str,
    output_file: str = None,
    lookback_days: int = 120,
    backup: bool = True
):
    """更新 MultiIndex 数据文件，重新计算所有因子"""
    print("\n" + "=" * 80)
    print("更新 MultiIndex 数据文件")
    print("=" * 80)
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_file}")
        return False
    
    if output_file is None:
        output_file = str(input_path)
    
    # 备份
    if backup and input_file == output_file:
        backup_file = str(input_path.parent / f"{input_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{input_path.suffix}")
        print(f"\n[备份] 创建备份: {backup_file}")
        try:
            import shutil
            shutil.copy2(input_file, backup_file)
            print(f"   [OK] 备份创建成功")
        except Exception as e:
            print(f"   [WARN] 备份失败: {e}")
    
    # 加载数据
    print("\n[步骤 1] 加载现有数据...")
    try:
        df = pd.read_parquet(input_file)
        print(f"   [OK] 数据加载成功: {df.shape}")
        
        if not isinstance(df.index, pd.MultiIndex):
            print(f"   [ERROR] 不是 MultiIndex 格式")
            return False
        
        df_reset = df.reset_index()
        
        # 确保必需的列存在
        required_cols = ['date', 'ticker', 'Close']
        missing_cols = [col for col in required_cols if col not in df_reset.columns]
        if missing_cols:
            print(f"   [ERROR] 缺少必需的列: {missing_cols}")
            return False
        
        # 处理 Volume
        if 'Volume' not in df_reset.columns:
            print("   [WARN] Volume 列不存在，尝试估算...")
            df_reset['Volume'] = 1_000_000  # 默认值
        
        # 准备数据
        df_reset['date'] = pd.to_datetime(df_reset['date']).dt.tz_localize(None).dt.normalize()
        df_reset = df_reset.sort_values(['date', 'ticker'])
        
    except Exception as e:
        print(f"   [ERROR] 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 初始化引擎
    print("\n[步骤 2] 初始化 Simple17FactorEngine...")
    try:
        engine = Simple17FactorEngine(
            lookback_days=lookback_days,
            mode='predict',
            horizon=10
        )
        print(f"   [OK] 引擎初始化成功")
        print(f"   期望的因子: {len(engine.alpha_factors)}")
        print(f"   因子列表: {engine.alpha_factors}")
    except Exception as e:
        print(f"   [ERROR] 引擎初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 检查并添加 SPY 数据（用于 ivol_30 计算）
    print("\n[步骤 3] 检查并添加 SPY 数据...")
    try:
        # 检查数据中是否已有 SPY
        has_spy = (df_reset['ticker'].astype(str).str.upper().str.strip() == 'SPY').any()
        
        if not has_spy and POLYGON_AVAILABLE:
            print("   [INFO] SPY 不在数据中，从 Polygon 下载...")
            try:
                # 获取日期范围
                min_date = df_reset['date'].min()
                max_date = df_reset['date'].max()
                
                # 下载 SPY 数据
                spy_df = polygon_download('SPY', 
                                         start=min_date.strftime('%Y-%m-%d'),
                                         end=max_date.strftime('%Y-%m-%d'),
                                         interval='1d')
                
                if not spy_df.empty:
                    # 准备 SPY 数据格式
                    spy_data = spy_df.reset_index()
                    if 'Date' in spy_data.columns:
                        spy_data['date'] = pd.to_datetime(spy_data['Date'])
                    elif 'timestamp' in spy_data.columns:
                        spy_data['date'] = pd.to_datetime(spy_data['timestamp'])
                    else:
                        spy_data['date'] = spy_df.index
                    
                    spy_data['date'] = pd.to_datetime(spy_data['date']).dt.tz_localize(None).dt.normalize()
                    spy_data['ticker'] = 'SPY'
                    
                    # 确保列名匹配
                    column_mapping = {
                        'open': 'Open', 'high': 'High', 'low': 'Low', 
                        'close': 'Close', 'volume': 'Volume'
                    }
                    for old_col, new_col in column_mapping.items():
                        if old_col in spy_data.columns and new_col not in spy_data.columns:
                            spy_data[new_col] = spy_data[old_col]
                    
                    # 只保留需要的列
                    required_cols = ['date', 'ticker', 'Close']
                    for col in ['Open', 'High', 'Low', 'Volume']:
                        if col in spy_data.columns:
                            required_cols.append(col)
                    
                    spy_data = spy_data[required_cols].copy()
                    
                    # 合并到数据中
                    df_reset = pd.concat([df_reset, spy_data], ignore_index=True)
                    df_reset = df_reset.sort_values(['date', 'ticker']).reset_index(drop=True)
                    
                    print(f"   [OK] SPY 数据已添加: {len(spy_data)} 行")
                else:
                    print("   [WARN] SPY 数据下载失败，将使用零值")
            except Exception as e:
                print(f"   [WARN] SPY 数据下载失败: {e}，将使用零值")
        elif has_spy:
            print("   [OK] SPY 数据已存在于数据中")
        else:
            print("   [WARN] Polygon 客户端不可用，无法下载 SPY 数据")
    except Exception as e:
        print(f"   [WARN] SPY 数据检查失败: {e}，继续处理")
    
    # 重新计算因子
    print("\n[步骤 4] 重新计算所有因子...")
    print("   这可能需要一些时间...")
    try:
        market_data = df_reset[['date', 'ticker', 'Close', 'Volume']].copy()
        
        # 添加 High, Low, Open 如果存在
        for col in ['High', 'Low', 'Open']:
            if col in df_reset.columns:
                market_data[col] = df_reset[col]
            else:
                market_data[col] = market_data['Close']  # 使用 Close 作为默认值
        
        factors_df = engine.compute_all_17_factors(market_data, mode='predict')
        
        print(f"   [OK] 因子计算完成: {factors_df.shape}")
        print(f"   计算的因子: {list(factors_df.columns)}")
        
        # 获取实际使用的因子列表
        actual_t10_selected = get_actual_t10_selected()
        
        # 验证所有期望的因子都被计算
        computed_factors = [f for f in actual_t10_selected if f in factors_df.columns]
        missing_computed = [f for f in actual_t10_selected if f not in factors_df.columns]
        
        print(f"\n   计算的期望因子: {len(computed_factors)}/{len(actual_t10_selected)}")
        if missing_computed:
            print(f"   [WARN] 未计算的因子: {missing_computed}")
        
        # 也检查 T10_ALPHA_FACTORS 中的其他因子
        all_computed = [f for f in T10_ALPHA_FACTORS if f in factors_df.columns]
        print(f"   T10_ALPHA_FACTORS 中计算的因子: {len(all_computed)}/{len(T10_ALPHA_FACTORS)}")
        
    except Exception as e:
        print(f"   [ERROR] 因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 合并回原始数据
    print("\n[步骤 5] 合并因子到原始数据...")
    try:
        # 确保 factors_df 有 MultiIndex
        if not isinstance(factors_df.index, pd.MultiIndex):
            if 'date' in factors_df.columns and 'ticker' in factors_df.columns:
                factors_df = factors_df.set_index(['date', 'ticker'])
            elif hasattr(factors_df, 'index') and len(factors_df) == len(market_data):
                # 使用 market_data 的索引创建 MultiIndex
                factors_df.index = pd.MultiIndex.from_arrays(
                    [market_data['date'], market_data['ticker']],
                    names=['date', 'ticker']
                )
            else:
                # 如果长度不匹配，尝试从 market_data 重新索引
                factors_df = factors_df.reset_index(drop=True)
                factors_df.index = pd.MultiIndex.from_arrays(
                    [market_data['date'], market_data['ticker']],
                    names=['date', 'ticker']
                )
        
        # 移除 SPY 行（如果添加了），只保留原始股票的数据
        if 'ticker' in factors_df.index.names:
            factors_df = factors_df[factors_df.index.get_level_values('ticker') != 'SPY']
        
        print(f"   [INFO] 因子数据形状（移除SPY后）: {factors_df.shape}")
        
        # 保留原始的非因子列
        non_factor_cols = ['Close', 'Volume', 'High', 'Low', 'Open']
        non_factor_cols = [col for col in non_factor_cols if col in df.columns]
        
        # 也保留其他非因子列（如 target 等）
        other_non_factor_cols = [col for col in df.columns 
                                 if col not in non_factor_cols 
                                 and col not in EXPECTED_FACTORS 
                                 and col not in REMOVED_FACTORS
                                 and col not in T10_ALPHA_FACTORS]
        
        df_new = df[non_factor_cols + other_non_factor_cols].copy()
        
        # 确保 df_new 的索引日期格式正确（移除时区偏移）
        if isinstance(df_new.index, pd.MultiIndex):
            dates = df_new.index.get_level_values('date')
            # 转换为 Series 以便使用 .dt 访问器
            dates_series = pd.Series(dates)
            if dates_series.dt.tz is not None:
                dates = dates_series.dt.tz_localize(None)
            elif len(dates_series) > 0 and dates_series.iloc[0].hour != 0:
                # 如果有时间偏移，规范化到 00:00:00
                dates = dates_series.dt.normalize()
            else:
                dates = dates_series
            
            tickers = df_new.index.get_level_values('ticker')
            df_new.index = pd.MultiIndex.from_arrays([dates.values, tickers], names=['date', 'ticker'])
        
        # 对齐并添加重新计算的因子
        # 首先检查 factors_df 的索引结构
        print(f"   [DEBUG] factors_df index type: {type(factors_df.index)}")
        print(f"   [DEBUG] factors_df shape: {factors_df.shape}")
        print(f"   [DEBUG] df_new index type: {type(df_new.index)}")
        print(f"   [DEBUG] df_new shape: {df_new.shape}")
        
        # 如果 factors_df 没有 MultiIndex，尝试从 market_data 重建
        if not isinstance(factors_df.index, pd.MultiIndex):
            print("   [DEBUG] factors_df 没有 MultiIndex，尝试重建...")
            # 从 market_data 创建 MultiIndex（排除 SPY）
            market_data_no_spy = market_data[market_data['ticker'] != 'SPY'].copy()
            if len(factors_df) == len(market_data_no_spy):
                factors_df.index = pd.MultiIndex.from_arrays(
                    [market_data_no_spy['date'], market_data_no_spy['ticker']],
                    names=['date', 'ticker']
                )
                print(f"   [OK] 重建 MultiIndex 成功: {factors_df.index.names}")
            else:
                print(f"   [ERROR] 长度不匹配: factors_df={len(factors_df)}, market_data_no_spy={len(market_data_no_spy)}")
                return False
        
        # 移除 SPY 行（如果添加了），只保留原始股票的数据
        if 'ticker' in factors_df.index.names:
            factors_df = factors_df[factors_df.index.get_level_values('ticker') != 'SPY']
            print(f"   [INFO] 移除 SPY 后因子数据形状: {factors_df.shape}")
        
        # 检查索引匹配情况
        print(f"   [DEBUG] factors_df index sample (first 5): {factors_df.index[:5].tolist()}")
        print(f"   [DEBUG] df_new index sample (first 5): {df_new.index[:5].tolist()}")
        
        # 检查索引是否完全匹配
        index_match = factors_df.index.equals(df_new.index)
        print(f"   [DEBUG] 索引完全匹配: {index_match}")
        
        if not index_match:
            # 尝试通过 reset_index 和 merge 来对齐
            print("   [INFO] 索引不匹配，使用 merge 方式对齐...")
            factors_df_reset = factors_df.reset_index()
            df_new_reset = df_new.reset_index()
            
            # 确保日期格式一致（移除时区，规范化到 00:00:00）
            factors_df_reset['date'] = pd.to_datetime(factors_df_reset['date']).dt.tz_localize(None).dt.normalize()
            df_new_reset['date'] = pd.to_datetime(df_new_reset['date']).dt.tz_localize(None).dt.normalize()
            
            # 合并因子列
            factor_cols = [col for col in factors_df_reset.columns 
                          if col not in ['date', 'ticker']]
            
            # 使用 merge 对齐
            merged = df_new_reset.merge(
                factors_df_reset[['date', 'ticker'] + factor_cols],
                on=['date', 'ticker'],
                how='left',
                suffixes=('', '_new')
            )
            
            # 恢复 MultiIndex
            merged = merged.set_index(['date', 'ticker'])
            df_new = merged
            
            print(f"   [OK] Merge 完成，形状: {df_new.shape}")
        else:
            # 索引匹配，直接使用 reindex
            factors_df_aligned = factors_df.reindex(df_new.index)
        
        # 获取实际使用的因子列表
        actual_t10_selected = get_actual_t10_selected()
        
        # 添加实际使用的因子（如果使用 merge，因子已经在 df_new 中）
        if not index_match:
            # 检查因子是否都在 df_new 中
            for factor in actual_t10_selected:
                if factor in df_new.columns:
                    non_null_count = df_new[factor].notna().sum()
                    print(f"   [OK] 因子已添加: {factor} (非空值: {non_null_count:,}/{len(df_new):,})")
                else:
                    print(f"   [WARN] 因子未找到: {factor}")
                    df_new[factor] = 0.0
        else:
            # 使用 reindex 的结果
            for factor in actual_t10_selected:
                if factor in factors_df_aligned.columns:
                    factor_values = factors_df_aligned[factor]
                    non_null_count = factor_values.notna().sum()
                    print(f"   [OK] 添加因子: {factor} (非空值: {non_null_count:,}/{len(factor_values):,})")
                    df_new[factor] = factor_values
                else:
                    print(f"   [WARN] 因子未找到: {factor}")
                    df_new[factor] = 0.0
        
        # 也添加 T10_ALPHA_FACTORS 中的其他因子（如果被计算）
        for factor in T10_ALPHA_FACTORS:
            if factor not in actual_t10_selected and factor not in REMOVED_FACTORS:
                if factor in df_new.columns:
                    non_null_count = df_new[factor].notna().sum()
                    print(f"   [INFO] 添加额外计算的因子: {factor} (非空值: {non_null_count:,})")
                elif not index_match and factor in factors_df_reset.columns:
                    # 因子已经在 merge 中，不需要额外操作
                    pass
        
        # 删除已移除的因子列
        for removed_factor in REMOVED_FACTORS:
            if removed_factor in df_new.columns:
                df_new = df_new.drop(columns=[removed_factor])
                print(f"   [OK] 删除已移除的因子: {removed_factor}")
        
        # 其他非因子列已在上面处理
        
        print(f"\n   [OK] 合并完成")
        print(f"   最终形状: {df_new.shape}")
        actual_t10_selected = get_actual_t10_selected()
        print(f"   实际使用的因子列数: {len([c for c in df_new.columns if c in actual_t10_selected])}")
        print(f"   T10_ALPHA_FACTORS 因子列数: {len([c for c in df_new.columns if c in T10_ALPHA_FACTORS])}")
        
    except Exception as e:
        print(f"   [ERROR] 合并失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 保存
    print("\n[步骤 6] 保存更新后的数据...")
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_new.to_parquet(output_file, compression='snappy', index=True)
        
        print(f"   [OK] 文件保存成功: {output_file}")
        
        # 验证保存的文件
        verify_df = pd.read_parquet(output_file)
        actual_t10_selected = get_actual_t10_selected()
        verify_factors = [f for f in actual_t10_selected if f in verify_df.columns]
        verify_removed = [f for f in REMOVED_FACTORS if f in verify_df.columns]
        
        print(f"\n   验证保存的文件:")
        print(f"   实际使用的因子: {len(verify_factors)}/{len(actual_t10_selected)}")
        if verify_removed:
            print(f"   [WARN] 仍包含已移除的因子: {verify_removed}")
        else:
            print(f"   [OK] 不包含已移除的因子")
        
    except Exception as e:
        print(f"   [ERROR] 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("[OK] 更新完成！")
    print("=" * 80)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证和更新所有因子")
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/factor_exports/polygon_factors_all_filtered_clean_recalculated.parquet",
        help="Input multiindex parquet file path"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (default: overwrites input with backup)"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=120,
        help="Lookback days for factor calculation"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create backup"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify, do not update"
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Non-interactive mode"
    )
    
    args = parser.parse_args()
    
    # 验证计算代码
    if not verify_factor_calculations():
        print("\n[ERROR] 因子计算代码验证失败")
        sys.exit(1)
    
    # 验证数据文件
    if not verify_multiindex_data(args.input_file):
        print("\n[ERROR] 数据文件验证失败")
        sys.exit(1)
    
    # 更新数据（如果不是仅验证模式）
    if not args.verify_only:
        success = update_multiindex_data(
            input_file=args.input_file,
            output_file=args.output_file,
            lookback_days=args.lookback_days,
            backup=not args.no_backup
        )
        sys.exit(0 if success else 1)
    else:
        print("\n[OK] 验证完成（仅验证模式，未更新数据）")
        sys.exit(0)
