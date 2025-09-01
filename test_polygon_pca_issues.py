#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Polygon ticker错误和PCA问题
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('D:/trade')

print("=== 测试Polygon ticker和PCA问题 ===")

# 创建测试数据
dates = pd.date_range(start='2023-01-01', end='2023-01-30', freq='D')
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

data = []
for date in dates:
    for ticker in tickers:
        data.append({
            'date': date,
            'ticker': ticker,
            'Close': np.random.randn() + 100,
            'Open': np.random.randn() + 100, 
            'High': np.random.randn() + 102,
            'Low': np.random.randn() + 98,
            'Volume': np.random.randint(1000000, 10000000),
            'returns': np.random.randn() * 0.02
        })

test_data = pd.DataFrame(data)
test_data['date'] = pd.to_datetime(test_data['date'])

print(f"测试数据创建成功: {test_data.shape}")

try:
    from bma_models.量化模型_bma_ultra_enhanced import UltraEnhancedQuantitativeModel
    print("模型导入成功")
    
    model = UltraEnhancedQuantitativeModel()
    print("模型创建成功")
    
    # 测试Polygon相关功能
    print("\n=== 测试1: Polygon因子库功能 ===")
    
    # 检查是否有complete_factor_library属性
    if hasattr(model, 'complete_factor_library'):
        print("[OK] complete_factor_library: 存在")
        if model.complete_factor_library:
            print("[OK] complete_factor_library: 已初始化")
        else:
            print("[WARN] complete_factor_library: 未初始化")
    else:
        print("[FAIL] complete_factor_library: 不存在")
    
    # 测试生成特征时的Polygon集成
    print("\n测试特征生成过程...")
    
    # 模拟调用generate_features方法
    if hasattr(model, 'generate_features'):
        try:
            print("尝试生成特征...")
            # 只测试少量数据避免长时间运行
            sample_data = test_data.head(30).copy()  # 只用前30行
            
            features = model.generate_features(sample_data, test_size=0.2)
            print(f"[OK] 特征生成成功: {features.shape}")
            
            # 检查是否有Polygon相关特征
            polygon_features = [col for col in features.columns if 'polygon_' in col]
            print(f"Polygon特征数量: {len(polygon_features)}")
            if len(polygon_features) > 0:
                print(f"Polygon特征示例: {polygon_features[:3]}")
                
        except Exception as e:
            if "'ticker'" in str(e):
                print(f"[FAIL] Polygon ticker错误确认: {e}")
                # 尝试找出具体位置
                import traceback
                tb_str = traceback.format_exc()
                print("错误详情:")
                for line in tb_str.split('\n'):
                    if 'ticker' in line.lower():
                        print(f"  -> {line.strip()}")
            else:
                print(f"[ERROR] 特征生成失败: {e}")
    
    # 测试PCA相关功能
    print("\n=== 测试2: PCA功能 ===")
    
    # 检查PCA相关方法
    pca_methods = [method for method in dir(model) if 'pca' in method.lower()]
    print(f"PCA相关方法: {pca_methods}")
    
    # 测试PCA变换方法
    if hasattr(model, 'apply_pca_transformation'):
        try:
            print("测试PCA变换...")
            # 创建测试特征矩阵
            test_features = pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100) * 2,
                'feature_3': np.random.randn(100) * 0.5,
                'date': pd.date_range('2023-01-01', periods=100)
            })
            
            pca_result, pca_info = model.apply_pca_transformation(test_features)
            print(f"[OK] PCA变换成功: {pca_result.shape}")
            print(f"PCA信息: {pca_info.get('n_components', 0)} 个主成分")
            
        except Exception as e:
            print(f"[FAIL] PCA变换失败: {e}")
    
    # 测试智能共线性处理
    if hasattr(model, 'apply_intelligent_multicollinearity_processing'):
        try:
            print("测试智能共线性处理...")
            test_features = pd.DataFrame({
                'f1': np.random.randn(50),
                'f2': np.random.randn(50),
                'f3': np.random.randn(50),
                'date': pd.date_range('2023-01-01', periods=50),
                'ticker': ['AAPL'] * 50
            })
            
            processed_features, process_info = model.apply_intelligent_multicollinearity_processing(test_features)
            print(f"[OK] 共线性处理成功: {processed_features.shape}")
            print(f"处理方法: {process_info.get('method_used', 'unknown')}")
            print(f"是否成功: {process_info.get('success', False)}")
            
        except Exception as e:
            print(f"[FAIL] 共线性处理失败: {e}")
    
    print("\n=== 问题总结 ===")
    print("1. Polygon ticker错误: 需要进一步调试特征生成过程")
    print("2. PCA功能: 代码中存在多种PCA相关方法，需要检查具体执行路径")

except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()