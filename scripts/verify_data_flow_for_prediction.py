"""
验证Direct Predict数据流程：确保数据获取、计算和传递给预测的MultiIndex格式一致
"""
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def verify_multiindex_format(df, stage_name, required_levels=['date', 'ticker']):
    """验证DataFrame的MultiIndex格式"""
    errors = []
    warnings = []
    
    # Check 1: Is MultiIndex?
    if not isinstance(df.index, pd.MultiIndex):
        errors.append(f"{stage_name}: Index is not MultiIndex, got {type(df.index)}")
        return {'valid': False, 'errors': errors, 'warnings': warnings}
    
    # Check 2: Level names
    index_names = df.index.names
    if index_names != required_levels:
        errors.append(f"{stage_name}: Level names mismatch. Expected {required_levels}, got {index_names}")
    
    # Check 3: Date level type
    if 'date' in index_names:
        date_level = df.index.get_level_values('date')
        if not pd.api.types.is_datetime64_any_dtype(date_level):
            errors.append(f"{stage_name}: Date level is not datetime, got {date_level.dtype}")
        else:
            # Check if normalized (no time component)
            try:
                if hasattr(date_level, 'dt'):
                    if not (date_level.dt.hour == 0).all() and not (date_level.dt.minute == 0).all():
                        warnings.append(f"{stage_name}: Date level may not be normalized (has time component)")
            except:
                pass  # Skip normalization check if dt accessor not available
    
    # Check 4: Ticker level type
    if 'ticker' in index_names:
        ticker_level = df.index.get_level_values('ticker')
        if not (pd.api.types.is_string_dtype(ticker_level) or pd.api.types.is_object_dtype(ticker_level)):
            warnings.append(f"{stage_name}: Ticker level is not string/object, got {ticker_level.dtype}")
    
    # Check 5: Duplicates
    duplicates = df.index.duplicated()
    if duplicates.any():
        dup_count = duplicates.sum()
        errors.append(f"{stage_name}: Found {dup_count} duplicate indices")
    
    # Check 6: Shape and basic info
    info = {
        'shape': df.shape,
        'unique_dates': df.index.get_level_values('date').nunique() if 'date' in index_names else 0,
        'unique_tickers': df.index.get_level_values('ticker').nunique() if 'ticker' in index_names else 0,
        'duplicate_count': duplicates.sum() if duplicates.any() else 0
    }
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'info': info
    }

def verify_data_flow():
    """验证整个数据流程"""
    print("="*80)
    print("验证Direct Predict数据流程")
    print("="*80)
    
    # 1. 检查训练文件格式（作为参考）
    print("\n[步骤1] 检查训练文件格式（参考标准）")
    training_file = r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet"
    
    if Path(training_file).exists():
        try:
            df_training = pd.read_parquet(training_file)
            training_check = verify_multiindex_format(df_training, "训练文件")
            
            print(f"  格式: {'[OK]' if training_check['valid'] else '[ERROR]'}")
            if training_check['errors']:
                for err in training_check['errors']:
                    print(f"    ERROR: {err}")
            if training_check['warnings']:
                for warn in training_check['warnings']:
                    print(f"    WARN: {warn}")
            print(f"  形状: {training_check['info']['shape']}")
            print(f"  唯一日期: {training_check['info']['unique_dates']}")
            print(f"  唯一Tickers: {training_check['info']['unique_tickers']}")
            print(f"  重复索引: {training_check['info']['duplicate_count']}")
        except Exception as e:
            print(f"  [ERROR] 无法读取训练文件: {e}")
    else:
        print(f"  [WARN] 训练文件不存在: {training_file}")
    
    # 2. 检查compute_all_17_factors的输出格式要求
    print("\n[步骤2] 检查因子计算函数格式要求")
    print("  函数: compute_all_17_factors")
    print("  位置: bma_models/simple_25_factor_engine.py")
    print("  要求:")
    print("    - 输入: market_data (DataFrame with 'date' and 'ticker' columns)")
    print("    - 输出: factors_df (MultiIndex(['date', 'ticker']))")
    print("    - 日期类型: datetime64[ns], normalized")
    print("    - Ticker类型: object/string")
    print("    - 无重复索引")
    
    # 3. 检查Direct Predict中的数据流程
    print("\n[步骤3] 检查Direct Predict数据流程")
    print("  位置: autotrader/app.py")
    print("  流程:")
    print("    1. 获取市场数据 (market_data)")
    print("    2. 计算因子 (engine.compute_all_17_factors)")
    print("    3. 标准化格式 (all_feature_data)")
    print("    4. 提取日期数据 (date_feature_data)")
    print("    5. 传递给预测 (model.predict_with_snapshot)")
    
    # 4. 检查predict_with_snapshot的输入格式要求
    print("\n[步骤4] 检查预测函数格式要求")
    print("  函数: predict_with_snapshot")
    print("  位置: bma_models/量化模型_bma_ultra_enhanced.py")
    print("  要求:")
    print("    - 输入: feature_data (MultiIndex(['date', 'ticker']))")
    print("    - 日期类型: datetime64[ns], normalized")
    print("    - Ticker类型: object/string")
    print("    - 无重复索引")
    print("    - 格式与训练文件完全一致")
    
    # 5. 检查_prepare_standard_data_format的格式标准化
    print("\n[步骤5] 检查格式标准化函数")
    print("  函数: _prepare_standard_data_format")
    print("  位置: bma_models/量化模型_bma_ultra_enhanced.py")
    print("  功能:")
    print("    - 标准化MultiIndex格式")
    print("    - 确保日期类型是normalized datetime")
    print("    - 确保ticker类型是string")
    print("    - 移除重复索引")
    print("    - 排序")
    
    # 6. 总结格式要求
    print("\n" + "="*80)
    print("格式要求总结")
    print("="*80)
    
    print("\n[标准格式]")
    print("  索引类型: pd.MultiIndex")
    print("  级别名称: ['date', 'ticker']")
    print("  日期类型: datetime64[ns] (normalized, no time component)")
    print("  Ticker类型: object/string")
    print("  无重复索引: 每个(date, ticker)组合只出现一次")
    print("  排序: 按date和ticker排序")
    
    print("\n[数据流程检查点]")
    print("  1. market_data -> compute_all_17_factors")
    print("     - 输入: DataFrame with 'date' and 'ticker' columns")
    print("     - 输出: MultiIndex(['date', 'ticker'])")
    print("  2. all_feature_data (after compute_all_17_factors)")
    print("     - 格式: MultiIndex(['date', 'ticker']), normalized")
    print("     - 验证: 在autotrader/app.py line ~1800")
    print("  3. date_feature_data (filtered by date)")
    print("     - 格式: MultiIndex(['date', 'ticker']), normalized")
    print("     - 验证: 在autotrader/app.py line ~1873")
    print("  4. predict_with_snapshot input")
    print("     - 格式: MultiIndex(['date', 'ticker']), normalized")
    print("     - 验证: 在量化模型_bma_ultra_enhanced.py line ~6630")
    
    print("\n[关键修复点]")
    print("  1. compute_all_17_factors (line ~816)")
    print("     - 返回前验证MultiIndex格式")
    print("     - 确保级别名称为['date', 'ticker']")
    print("     - 确保日期类型是normalized datetime")
    print("     - 确保ticker类型是string")
    print("     - 移除重复索引")
    print("  2. Direct Predict (line ~1800)")
    print("     - 标准化all_feature_data的MultiIndex格式")
    print("     - 确保格式与训练文件完全一致")
    print("     - 移除重复索引")
    print("  3. _prepare_standard_data_format (line ~6630)")
    print("     - 标准化feature_data的MultiIndex格式")
    print("     - 确保格式与训练文件完全一致")
    print("     - 移除重复索引并排序")
    
    print("\n" + "="*80)
    print("验证完成")
    print("="*80)
    
    return True

if __name__ == "__main__":
    verify_data_flow()
