"""
创建训练文件的1/5 ticker子集，用于快速训练和测试
"""
import pandas as pd
import sys
from pathlib import Path
import numpy as np

def create_subset_1_5_tickers(
    input_file: str,
    output_file: str = None,
    fraction: float = 0.2,  # 1/5 = 0.2
    random_seed: int = 42
):
    """
    从训练文件中创建ticker子集
    
    Args:
        input_file: 输入parquet文件路径
        output_file: 输出parquet文件路径（如果为None，自动生成）
        fraction: ticker比例（默认0.2 = 1/5）
        random_seed: 随机种子
    """
    print("="*80)
    print("创建1/5 Ticker子集")
    print("="*80)
    
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    print(f"\n[步骤1] 读取训练文件: {input_file}")
    df = pd.read_parquet(input_file)
    
    print(f"  原始数据形状: {df.shape}")
    print(f"  索引类型: {type(df.index)}")
    
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"输入文件必须是MultiIndex格式，got: {type(df.index)}")
    
    if 'date' not in df.index.names or 'ticker' not in df.index.names:
        raise ValueError(f"MultiIndex必须包含'date'和'ticker'级别，got: {df.index.names}")
    
    # 获取所有唯一的ticker
    all_tickers = df.index.get_level_values('ticker').unique()
    n_total_tickers = len(all_tickers)
    print(f"  总ticker数: {n_total_tickers}")
    
    # 随机选择1/5的ticker
    np.random.seed(random_seed)
    n_subset_tickers = max(1, int(n_total_tickers * fraction))
    selected_tickers = np.random.choice(all_tickers, size=n_subset_tickers, replace=False)
    selected_tickers = sorted(selected_tickers)  # 排序以便可重复
    
    print(f"\n[步骤2] 选择ticker子集")
    print(f"  选择比例: {fraction} ({fraction*100:.1f}%)")
    print(f"  选择的ticker数: {n_subset_tickers}")
    print(f"  选择的ticker (前20个): {list(selected_tickers[:20])}")
    if len(selected_tickers) > 20:
        print(f"  ... 还有 {len(selected_tickers) - 20} 个ticker")
    
    # 过滤数据，只保留选中的ticker
    print(f"\n[步骤3] 过滤数据")
    ticker_level = df.index.get_level_values('ticker')
    mask = ticker_level.isin(selected_tickers)
    df_subset = df[mask].copy()
    
    print(f"  子集数据形状: {df_subset.shape}")
    print(f"  原始数据行数: {len(df)}")
    print(f"  子集数据行数: {len(df_subset)}")
    print(f"  数据减少比例: {(1 - len(df_subset)/len(df))*100:.1f}%")
    
    # 验证格式
    print(f"\n[步骤4] 验证子集格式")
    print(f"  索引类型: {type(df_subset.index)}")
    if isinstance(df_subset.index, pd.MultiIndex):
        print(f"  级别名称: {df_subset.index.names}")
        print(f"  唯一日期数: {df_subset.index.get_level_values('date').nunique()}")
        print(f"  唯一ticker数: {df_subset.index.get_level_values('ticker').nunique()}")
        print(f"  列数: {len(df_subset.columns)}")
    
    # 生成输出文件名
    if output_file is None:
        input_stem = input_path.stem
        output_file = input_path.parent / f"{input_stem}_subset_1_5_tickers.parquet"
    
    output_path = Path(output_file)
    
    # 保存子集
    print(f"\n[步骤5] 保存子集到: {output_path}")
    df_subset.to_parquet(output_path, index=True, engine='pyarrow')
    
    print(f"  [OK] 子集已保存")
    print(f"  文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # 保存选择的ticker列表（用于验证）
    ticker_list_file = output_path.parent / f"{output_path.stem}_tickers.txt"
    with open(ticker_list_file, 'w', encoding='utf-8') as f:
        for ticker in selected_tickers:
            f.write(f"{ticker}\n")
    print(f"  [OK] Ticker列表已保存到: {ticker_list_file}")
    
    print("\n" + "="*80)
    print("子集创建完成")
    print("="*80)
    print(f"\n输出文件: {output_path}")
    print(f"Ticker列表: {ticker_list_file}")
    print(f"\n可以使用以下命令进行训练和评估:")
    print(f"  python scripts/train_full_dataset.py --train-data {output_path}")
    print(f"  python scripts/time_split_80_20_oos_eval.py --data-file {output_path}")
    
    return output_path, selected_tickers

if __name__ == "__main__":
    input_file = r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet"
    
    # 创建子集
    output_path, selected_tickers = create_subset_1_5_tickers(
        input_file=input_file,
        fraction=0.2,  # 1/5
        random_seed=42
    )
    
    print(f"\n[OK] 完成! 子集文件: {output_path}")
    print(f"   包含 {len(selected_tickers)} 个ticker (原始数据的20%)")
