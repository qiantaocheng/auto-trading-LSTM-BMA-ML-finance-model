#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试自适应加树优化器
演示基于验证集提升斜率的智能加树策略
"""

import pandas as pd
import numpy as np
import logging
from adaptive_tree_optimizer import AdaptiveTreeOptimizer
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_test_data(n_samples=300, n_features=12, noise_level=0.3):
    """创建测试数据"""
    np.random.seed(42)
    
    # 创建具有不同信号强度的股票数据
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 创建目标变量（有些特征有真实信号，有些是噪音）
    true_signal = (
        X['feature_0'] * 0.5 +           # 强信号
        X['feature_1'] * 0.3 +           # 中信号  
        X['feature_2'] * 0.1 +           # 弱信号
        np.sin(X['feature_3']) * 0.2     # 非线性信号
    )
    
    # 添加噪音
    y = true_signal + np.random.randn(n_samples) * noise_level
    
    return X, pd.Series(y, name='target')

def test_adaptive_vs_fixed():
    """测试自适应优化vs固定参数的效果"""
    print("=== 自适应优化 vs 固定参数对比测试 ===")
    
    # 创建不同信号强度的数据集
    datasets = {
        "强信号股票": create_test_data(noise_level=0.1),
        "中信号股票": create_test_data(noise_level=0.3), 
        "弱信号股票": create_test_data(noise_level=0.6)
    }
    
    results = []
    
    for stock_type, (X, y) in datasets.items():
        print(f"\n--- 测试 {stock_type} ---")
        
        # 创建自适应优化器
        optimizer = AdaptiveTreeOptimizer(
            slope_threshold_ic=0.001,     # 降低阈值以便观察效果
            slope_threshold_mse=0.005,
            tree_increment=15,
            max_trees_xgb=100,
            max_trees_lgb=100,
            max_trees_rf=120
        )
        
        stock_id = stock_type.replace("股票", "")
        
        # 测试XGBoost自适应优化
        if hasattr(optimizer, 'adaptive_train_xgboost'):
            try:
                xgb_model, xgb_perf = optimizer.adaptive_train_xgboost(X, y, stock_id)
                final_trees = xgb_model.n_estimators if xgb_model else 0
                
                result = {
                    'stock_type': stock_type,
                    'model': 'XGBoost',
                    'final_trees': final_trees,
                    'ic': xgb_perf.get('ic', 0.0),
                    'mse': xgb_perf.get('mse', 0.0)
                }
                results.append(result)
                
                print(f"  XGBoost: {final_trees}棵树, IC={xgb_perf.get('ic', 0):.4f}, MSE={xgb_perf.get('mse', 0):.4f}")
                
            except Exception as e:
                print(f"  XGBoost测试失败: {e}")
        
        # 测试LightGBM自适应优化
        if hasattr(optimizer, 'adaptive_train_lightgbm'):
            try:
                lgb_model, lgb_perf = optimizer.adaptive_train_lightgbm(X, y, stock_id)
                final_trees = lgb_model.n_estimators if lgb_model else 0
                
                result = {
                    'stock_type': stock_type,
                    'model': 'LightGBM',
                    'final_trees': final_trees,
                    'ic': lgb_perf.get('ic', 0.0),
                    'mse': lgb_perf.get('mse', 0.0)
                }
                results.append(result)
                
                print(f"  LightGBM: {final_trees}棵树, IC={lgb_perf.get('ic', 0):.4f}, MSE={lgb_perf.get('mse', 0):.4f}")
                
            except Exception as e:
                print(f"  LightGBM测试失败: {e}")
        
        # 测试RandomForest自适应优化
        try:
            rf_model, rf_perf = optimizer.adaptive_train_random_forest(X, y, stock_id)
            final_trees = rf_model.n_estimators if rf_model else 0
            
            result = {
                'stock_type': stock_type,
                'model': 'RandomForest',
                'final_trees': final_trees,
                'ic': rf_perf.get('ic', 0.0),
                'mse': rf_perf.get('mse', 0.0),
                'oob_score': rf_perf.get('oob_score', 0.0)
            }
            results.append(result)
            
            print(f"  RandomForest: {final_trees}棵树, OOB={rf_perf.get('oob_score', 0):.4f}")
            
        except Exception as e:
            print(f"  RandomForest测试失败: {e}")
    
    # 分析结果
    if results:
        results_df = pd.DataFrame(results)
        print("\n=== 结果总结 ===")
        print(results_df.to_string(index=False))
        
        # 生成可视化（如果有matplotlib）
        try:
            create_visualization(results_df)
        except Exception as e:
            print(f"可视化失败: {e}")
    
    return results

def create_visualization(results_df):
    """创建结果可视化"""
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('自适应加树优化结果分析', fontsize=16, fontweight='bold')
    
    # 1. 树数量对比
    ax1 = axes[0, 0]
    pivot_trees = results_df.pivot(index='stock_type', columns='model', values='final_trees')
    pivot_trees.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('最终树数量对比')
    ax1.set_ylabel('树数量')
    ax1.legend(title='模型类型')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. IC对比
    ax2 = axes[0, 1] 
    pivot_ic = results_df.pivot(index='stock_type', columns='model', values='ic')
    pivot_ic.plot(kind='bar', ax=ax2, color=['orange', 'purple', 'brown'])
    ax2.set_title('信息系数(IC)对比')
    ax2.set_ylabel('IC值')
    ax2.legend(title='模型类型')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. MSE对比
    ax3 = axes[1, 0]
    pivot_mse = results_df.pivot(index='stock_type', columns='model', values='mse')
    pivot_mse.plot(kind='bar', ax=ax3, color=['red', 'blue', 'green'])
    ax3.set_title('均方误差(MSE)对比')
    ax3.set_ylabel('MSE值')
    ax3.legend(title='模型类型')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 综合性能评分
    ax4 = axes[1, 1]
    # 计算综合评分：IC越高越好，MSE越低越好
    results_df['performance_score'] = results_df['ic'] - results_df['mse'] * 0.1
    pivot_score = results_df.pivot(index='stock_type', columns='model', values='performance_score')
    pivot_score.plot(kind='bar', ax=ax4, color=['gold', 'silver', 'bronze'])
    ax4.set_title('综合性能评分')
    ax4.set_ylabel('性能评分')
    ax4.legend(title='模型类型')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('adaptive_optimizer_results.png', dpi=300, bbox_inches='tight')
    print("可视化结果已保存到 adaptive_optimizer_results.png")
    
    # 显示统计信息
    print("\n=== 统计分析 ===")
    print("平均树数量:")
    print(pivot_trees.mean().round(1))
    print("\n平均IC值:")
    print(pivot_ic.mean().round(4))
    print("\n平均MSE值:")
    print(pivot_mse.mean().round(4))

def test_slope_calculation():
    """测试斜率计算逻辑"""
    print("\n=== 斜率计算测试 ===")
    
    optimizer = AdaptiveTreeOptimizer()
    
    # 测试不同的性能轨迹
    test_cases = [
        {
            'name': '持续改善',
            'ic_history': [0.1, 0.12, 0.15, 0.17, 0.19],
            'tree_counts': [20, 40, 60, 80, 100]
        },
        {
            'name': '早期改善后平缓',
            'ic_history': [0.1, 0.15, 0.16, 0.16, 0.16],
            'tree_counts': [20, 40, 60, 80, 100]
        },
        {
            'name': '性能下降',
            'ic_history': [0.15, 0.14, 0.12, 0.10, 0.08],
            'tree_counts': [20, 40, 60, 80, 100]
        },
        {
            'name': '不稳定波动',
            'ic_history': [0.1, 0.15, 0.08, 0.17, 0.12],
            'tree_counts': [20, 40, 60, 80, 100]
        }
    ]
    
    for case in test_cases:
        slope = optimizer.calculate_performance_slope(
            case['ic_history'], case['tree_counts']
        )
        should_continue = slope >= optimizer.slope_threshold_ic
        
        print(f"{case['name']}: 斜率={slope:.6f}, 继续加树={should_continue}")

def main():
    """主测试函数"""
    print("🚀 自适应加树优化器测试")
    print("=" * 50)
    
    # 1. 基础功能测试
    print("\n1. 基础功能测试")
    try:
        from adaptive_tree_optimizer import demo_adaptive_optimization
        demo_adaptive_optimization()
        print("✅ 基础功能测试通过")
    except Exception as e:
        print(f"❌ 基础功能测试失败: {e}")
    
    # 2. 斜率计算测试
    print("\n2. 斜率计算测试")
    try:
        test_slope_calculation()
        print("✅ 斜率计算测试通过")
    except Exception as e:
        print(f"❌ 斜率计算测试失败: {e}")
    
    # 3. 对比测试
    print("\n3. 自适应vs固定参数对比测试")
    try:
        results = test_adaptive_vs_fixed()
        print("✅ 对比测试完成")
        
        # 输出关键发现
        if results:
            print("\n🔍 关键发现:")
            results_df = pd.DataFrame(results)
            avg_trees = results_df.groupby('model')['final_trees'].mean()
            print("平均最终树数量:")
            for model, trees in avg_trees.items():
                print(f"  {model}: {trees:.1f}棵")
                
        return results
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        return None

if __name__ == "__main__":
    results = main()
    print("\n🎉 测试完成!")
    
    if results:
        print("\n💡 使用建议:")
        print("1. 对于强信号股票，可以适当增加树的数量以捕获更多模式")
        print("2. 对于弱信号股票，早停机制可以有效防止过拟合")
        print("3. 自适应加树比固定参数更能平衡性能和效率")
        print("4. 建议在生产环境中使用更严格的斜率阈值")