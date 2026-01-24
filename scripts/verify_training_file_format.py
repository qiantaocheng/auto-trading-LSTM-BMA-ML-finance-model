"""
验证训练文件的MultiIndex格式，确保Direct Predict使用相同格式
"""
import pandas as pd
import sys
from pathlib import Path

def verify_training_file_format(file_path: str):
    """验证训练文件的格式"""
    print("="*80)
    print("验证训练文件格式")
    print("="*80)
    
    if not Path(file_path).exists():
        print(f"[ERROR] 文件不存在: {file_path}")
        return None
    
    df = pd.read_parquet(file_path)
    
    print(f"\n文件: {file_path}")
    print(f"Shape: {df.shape}")
    print(f"列数: {len(df.columns)}")
    
    # 检查索引格式
    print(f"\n索引类型: {type(df.index)}")
    if isinstance(df.index, pd.MultiIndex):
        print(f"[OK] MultiIndex格式")
        print(f"索引级别数: {df.index.nlevels}")
        print(f"级别名称: {df.index.names}")
        
        if df.index.nlevels >= 2:
            level0_name = df.index.names[0]
            level1_name = df.index.names[1]
            print(f"第一级: {level0_name} ({type(df.index.get_level_values(0).dtype)})")
            print(f"第二级: {level1_name} ({type(df.index.get_level_values(1).dtype)})")
            
            # 检查数据类型
            level0_values = df.index.get_level_values(0)
            level1_values = df.index.get_level_values(1)
            
            print(f"\n第一级数据类型: {level0_values.dtype}")
            print(f"第二级数据类型: {level1_values.dtype}")
            
            # 检查日期格式
            if pd.api.types.is_datetime64_any_dtype(level0_values):
                print(f"[OK] 第一级是日期类型")
                print(f"日期范围: {level0_values.min()} 到 {level0_values.max()}")
            else:
                print(f"[WARN] 第一级不是日期类型: {level0_values.dtype}")
            
            # 检查ticker格式
            if pd.api.types.is_string_dtype(level1_values) or pd.api.types.is_object_dtype(level1_values):
                print(f"[OK] 第二级是字符串类型")
                print(f"唯一ticker数: {level1_values.nunique()}")
                print(f"示例tickers: {sorted(level1_values.unique())[:10]}")
            else:
                print(f"[WARN] 第二级不是字符串类型: {level1_values.dtype}")
            
            # 检查是否有重复索引
            duplicates = df.index.duplicated()
            if duplicates.any():
                print(f"\n[WARN] 发现 {duplicates.sum()} 个重复索引!")
            else:
                print(f"\n[OK] 没有重复索引")
            
            # 检查列
            print(f"\n列名 (前20个):")
            for col in df.columns[:20]:
                print(f"  - {col}")
            
            # 检查必需的列
            required_cols = ['Close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"\n[WARN] 缺少必需列: {missing_cols}")
            else:
                print(f"\n[OK] 包含必需列: {required_cols}")
            
            return {
                'format': 'MultiIndex',
                'levels': df.index.names,
                'level0_type': level0_values.dtype,
                'level1_type': level1_values.dtype,
                'shape': df.shape,
                'columns': list(df.columns),
                'has_duplicates': duplicates.any(),
                'duplicate_count': duplicates.sum() if duplicates.any() else 0
            }
            else:
                print(f"[WARN] MultiIndex级别数不足: {df.index.nlevels}")
            return None
    else:
        print(f"[WARN] 不是MultiIndex格式")
        print(f"索引类型: {type(df.index)}")
        if hasattr(df.index, 'name'):
            print(f"索引名称: {df.index.name}")
        return None

if __name__ == "__main__":
    training_file = r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet"
    result = verify_training_file_format(training_file)
    
    if result:
        print("\n" + "="*80)
        print("格式规范总结")
        print("="*80)
        print(f"格式: {result['format']}")
        print(f"级别名称: {result['levels']}")
        print(f"第一级类型: {result['level0_type']} (应该是datetime64)")
        print(f"第二级类型: {result['level1_type']} (应该是object/string)")
        print(f"形状: {result['shape']}")
        print(f"重复索引: {result['has_duplicates']} ({result['duplicate_count']} 个)")
