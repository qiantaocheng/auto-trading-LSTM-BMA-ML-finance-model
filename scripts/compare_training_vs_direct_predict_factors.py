"""
对比训练parquet文件和Direct Predict使用的因子差异
"""
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS, T5_ALPHA_FACTORS

def compare_factors():
    """对比训练文件和Direct Predict的因子"""
    print("="*80)
    print("对比训练文件和Direct Predict的因子")
    print("="*80)
    
    # 1. 读取训练文件
    training_file = r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet"
    
    if not Path(training_file).exists():
        print(f"❌ 训练文件不存在: {training_file}")
        return
    
    print(f"\n[步骤1] 读取训练文件: {training_file}")
    df_training = pd.read_parquet(training_file)
    
    print(f"[OK] 训练文件格式:")
    print(f"  Shape: {df_training.shape}")
    print(f"  索引类型: {type(df_training.index)}")
    if isinstance(df_training.index, pd.MultiIndex):
        print(f"  级别名称: {df_training.index.names}")
        print(f"  日期类型: {df_training.index.get_level_values('date').dtype}")
        print(f"  Ticker类型: {df_training.index.get_level_values('ticker').dtype}")
    
    training_columns = set(df_training.columns)
    print(f"  总列数: {len(training_columns)}")
    
    # 2. 获取Direct Predict使用的因子
    print(f"\n[步骤2] Direct Predict使用的因子 (T10_ALPHA_FACTORS):")
    direct_predict_factors = set(T10_ALPHA_FACTORS)
    print(f"  因子数: {len(direct_predict_factors)}")
    print(f"  因子列表:")
    for i, factor in enumerate(sorted(direct_predict_factors), 1):
        print(f"    {i:2d}. {factor}")
    
    # 3. 识别因子列（排除市场数据和元数据）
    market_data_cols = {'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close', 'adj_close', 'close', 'open', 'high', 'low', 'volume'}
    metadata_cols = {'target', 'date', 'ticker', 'Symbol'}
    
    # 训练文件中的因子列（排除市场数据和元数据）
    training_factors = training_columns - market_data_cols - metadata_cols
    
    print(f"\n[步骤3] 训练文件中的因子列:")
    print(f"  因子列数: {len(training_factors)}")
    print(f"  因子列 (前30个):")
    for i, factor in enumerate(sorted(training_factors)[:30], 1):
        print(f"    {i:2d}. {factor}")
    
    # 4. 对比差异
    print(f"\n" + "="*80)
    print("因子对比分析")
    print("="*80)
    
    # Direct Predict有但训练文件没有的因子
    missing_in_training = direct_predict_factors - training_factors
    if missing_in_training:
        print(f"\n[WARN] Direct Predict有但训练文件没有的因子 ({len(missing_in_training)}):")
        for factor in sorted(missing_in_training):
            print(f"  - {factor}")
    else:
        print(f"\n[OK] Direct Predict的所有因子都在训练文件中")
    
    # 训练文件有但Direct Predict没有的因子
    extra_in_training = training_factors - direct_predict_factors
    if extra_in_training:
        print(f"\n[INFO] 训练文件有但Direct Predict没有的因子 ({len(extra_in_training)}):")
        print(f"  (这些因子可能被移除或替换)")
        for i, factor in enumerate(sorted(extra_in_training)[:30], 1):
            print(f"    {i:2d}. {factor}")
        if len(extra_in_training) > 30:
            print(f"    ... 还有 {len(extra_in_training) - 30} 个因子")
    
    # 共同因子
    common_factors = direct_predict_factors & training_factors
    print(f"\n[OK] 共同因子 ({len(common_factors)}):")
    for i, factor in enumerate(sorted(common_factors), 1):
        print(f"    {i:2d}. {factor}")
    
    # 5. 检查必需的市场数据列
    print(f"\n" + "="*80)
    print("市场数据列检查")
    print("="*80)
    
    required_market_cols = {'Close'}
    missing_market_cols = required_market_cols - training_columns
    if missing_market_cols:
        print(f"[WARN] 训练文件缺少必需的市场数据列: {missing_market_cols}")
    else:
        print(f"[OK] 训练文件包含必需的市场数据列: {required_market_cols}")
    
    # 6. 格式对比
    print(f"\n" + "="*80)
    print("格式对比")
    print("="*80)
    
    print(f"\n训练文件格式:")
    print(f"  索引类型: {type(df_training.index)}")
    if isinstance(df_training.index, pd.MultiIndex):
        print(f"  级别名称: {df_training.index.names}")
        print(f"  日期类型: {df_training.index.get_level_values('date').dtype}")
        print(f"  Ticker类型: {df_training.index.get_level_values('ticker').dtype}")
        print(f"  重复索引: {df_training.index.duplicated().sum()} 个")
    
    print(f"\nDirect Predict格式要求:")
    print(f"  索引类型: pd.MultiIndex")
    print(f"  级别名称: ['date', 'ticker']")
    print(f"  日期类型: datetime64[ns] (normalized)")
    print(f"  Ticker类型: object/string")
    print(f"  重复索引: 0 个")
    
    # 7. 总结
    print(f"\n" + "="*80)
    print("总结")
    print("="*80)
    
    print(f"\n训练文件:")
    print(f"  总列数: {len(training_columns)}")
    print(f"  因子列数: {len(training_factors)}")
    print(f"  市场数据列: {len(training_columns & market_data_cols)}")
    print(f"  元数据列: {len(training_columns & metadata_cols)}")
    
    print(f"\nDirect Predict:")
    print(f"  使用的因子数: {len(direct_predict_factors)}")
    print(f"  共同因子数: {len(common_factors)}")
    print(f"  缺失因子数: {len(missing_in_training)}")
    
    if missing_in_training:
        print(f"\n[WARN] 警告: Direct Predict使用了 {len(missing_in_training)} 个训练文件中没有的因子")
        print(f"   这些因子可能在预测时无法使用，需要检查因子计算逻辑")
    
    return {
        'training_factors': training_factors,
        'direct_predict_factors': direct_predict_factors,
        'common_factors': common_factors,
        'missing_in_training': missing_in_training,
        'extra_in_training': extra_in_training
    }

if __name__ == "__main__":
    result = compare_factors()
