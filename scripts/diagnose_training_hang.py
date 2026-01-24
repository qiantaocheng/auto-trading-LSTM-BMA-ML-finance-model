"""
诊断训练挂起问题
检查可能导致训练进程挂起的原因
"""
import sys
from pathlib import Path
import pandas as pd

def check_subset_data():
    """检查子集数据是否有问题"""
    print("="*80)
    print("检查子集数据")
    print("="*80)
    
    subset_file = Path(r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet")
    
    if not subset_file.exists():
        print(f"[ERROR] 子集文件不存在: {subset_file}")
        return False
    
    print(f"[OK] 子集文件存在: {subset_file}")
    print(f"文件大小: {subset_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        print("\n读取数据...")
        df = pd.read_parquet(subset_file)
        print(f"[OK] 数据读取成功")
        print(f"形状: {df.shape}")
        print(f"索引类型: {type(df.index)}")
        
        if isinstance(df.index, pd.MultiIndex):
            print(f"MultiIndex levels: {df.index.names}")
            print(f"日期范围: {df.index.get_level_values('date').min()} 到 {df.index.get_level_values('date').max()}")
            print(f"Ticker数量: {df.index.get_level_values('ticker').nunique()}")
            print(f"总样本数: {len(df)}")
        else:
            print("[WARN] 数据不是MultiIndex格式")
        
        # 检查是否有target列
        target_cols = [col for col in df.columns if 'target' in col.lower() or 'ret_fwd' in col.lower()]
        if target_cols:
            print(f"\n[OK] 找到目标列: {target_cols}")
            for col in target_cols:
                import numpy as np
                print(f"  {col}: NaN={df[col].isna().sum()}, Inf={np.isinf(df[col]).sum() if df[col].dtype in ['float64', 'float32'] else 0}")
        else:
            print("[WARN] 未找到目标列")
        
        # 检查特征列
        feature_cols = [col for col in df.columns if col not in target_cols]
        print(f"\n特征列数量: {len(feature_cols)}")
        
        # 检查NaN和Inf
        print("\n检查数据质量...")
        nan_counts = df[feature_cols[:100]].isna().sum()  # 只检查前100个特征
        if nan_counts.sum() > 0:
            print(f"[WARN] 前100个特征中有NaN: {nan_counts.sum()} 个值")
        else:
            print("[OK] 前100个特征无NaN")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 读取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_training_config():
    """检查训练配置"""
    print("\n" + "="*80)
    print("检查训练配置")
    print("="*80)
    
    try:
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(project_root))
        from bma_models.unified_config_loader import CONFIG
        
        print(f"[OK] 配置加载成功")
        print(f"CV折数: {CONFIG._CV_SPLITS}")
        print(f"CV Gap: {CONFIG._CV_GAP_DAYS} 天")
        print(f"CV Embargo: {CONFIG._CV_EMBARGO_DAYS} 天")
        
        # 检查模型配置
        if hasattr(CONFIG, 'ELASTIC_NET_CONFIG'):
            print(f"\nElasticNet配置:")
            print(f"  alpha: {CONFIG.ELASTIC_NET_CONFIG.get('alpha')}")
            print(f"  max_iter: {CONFIG.ELASTIC_NET_CONFIG.get('max_iter')}")
        
        if hasattr(CONFIG, 'CATBOOST_CONFIG'):
            print(f"\nCatBoost配置:")
            print(f"  iterations: {CONFIG.CATBOOST_CONFIG.get('iterations')}")
            print(f"  depth: {CONFIG.CATBOOST_CONFIG.get('depth')}")
            print(f"  od_wait: {CONFIG.CATBOOST_CONFIG.get('od_wait')}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 配置检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_imports():
    """检查必要的导入"""
    print("\n" + "="*80)
    print("检查必要的库")
    print("="*80)
    
    required_libs = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 'catboost', 
        'lightgbm', 'scipy', 'statsmodels'
    ]
    
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"[OK] {lib}")
        except ImportError:
            print(f"[ERROR] {lib} 未安装")

def main():
    print("="*80)
    print("训练挂起诊断工具")
    print("="*80)
    
    # 检查数据
    data_ok = check_subset_data()
    
    # 检查配置
    config_ok = check_training_config()
    
    # 检查库
    check_imports()
    
    print("\n" + "="*80)
    print("诊断总结")
    print("="*80)
    
    if data_ok and config_ok:
        print("[OK] 基本检查通过")
        print("\n可能的问题:")
        print("1. 线程池阻塞 - ThreadPoolExecutor可能在某些情况下挂起")
        print("2. CatBoost训练时间过长 - od_wait=120可能导致长时间等待")
        print("3. 内存不足 - 大数据集可能导致OOM")
        print("4. 死锁 - 多线程操作可能导致死锁")
        print("\n建议:")
        print("1. 检查是否有足够的系统内存")
        print("2. 尝试减少CV折数或样本数")
        print("3. 检查CatBoost的od_wait参数")
        print("4. 查看是否有其他进程占用资源")
    else:
        print("[ERROR] 发现问题，请检查上述错误信息")

if __name__ == "__main__":
    main()
