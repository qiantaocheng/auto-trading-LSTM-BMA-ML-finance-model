#!/usr/bin/env python3
"""
BMA Ultra Enhanced 完整验证脚本
验证所有导入、功能和真实流程运行
"""

import sys
import os
import time
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bma_models'))

def print_section(title: str):
    """打印分节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_external_dependencies() -> Dict[str, bool]:
    """检查所有外部依赖包"""
    print_section("检查外部依赖包")
    
    dependencies = {
        # 基础包
        'numpy': False,
        'pandas': False,
        'scipy': False,
        'sklearn': False,
        
        # 高级ML包
        'xgboost': False,
        'lightgbm': False,
        'catboost': False,
        
        # 系统包
        'psutil': False,
        'yaml': False,
        'joblib': False,
    }
    
    for package in dependencies:
        try:
            __import__(package)
            dependencies[package] = True
            print(f"✓ {package}: 可用")
        except ImportError:
            print(f"✗ {package}: 不可用")
    
    return dependencies

def check_bma_modules() -> Dict[str, Any]:
    """检查所有BMA内部模块"""
    print_section("检查BMA内部模块")
    
    modules_status = {}
    
    # 要检查的模块列表
    bma_modules = [
        ('index_aligner', 'IndexAligner'),
        ('enhanced_alpha_strategies', 'AlphaStrategiesEngine'),
        ('intelligent_memory_manager', 'IntelligentMemoryManager'),
        ('unified_exception_handler', 'UnifiedExceptionHandler'),
        ('production_readiness_validator', 'ProductionReadinessValidator'),
        ('regime_aware_cv', 'RegimeAwareCV'),
        ('leak_free_regime_detector', 'LeakFreeRegimeDetector'),
        ('enhanced_oos_system', 'EnhancedOOSSystem'),
        ('unified_feature_pipeline', 'UnifiedFeaturePipeline'),
        ('sample_weight_unification', 'SampleWeightUnificator'),
        ('fixed_purged_time_series_cv', 'FixedPurgedTimeSeriesCV'),
        ('alpha_summary_features', 'AlphaSummaryFeatures'),
        ('config_loader', 'ConfigLoader'),
    ]
    
    for module_name, class_name in bma_modules:
        try:
            module = __import__(f'bma_models.{module_name}', fromlist=[class_name])
            cls = getattr(module, class_name, None)
            if cls:
                modules_status[module_name] = {
                    'status': True,
                    'class': class_name,
                    'path': module.__file__ if hasattr(module, '__file__') else 'N/A'
                }
                print(f"✓ {module_name}.{class_name}: 可用")
            else:
                modules_status[module_name] = {
                    'status': False,
                    'error': f"Class {class_name} not found"
                }
                print(f"⚠ {module_name}: 模块可用但类{class_name}未找到")
        except Exception as e:
            modules_status[module_name] = {
                'status': False,
                'error': str(e)
            }
            print(f"✗ {module_name}: {str(e)[:50]}")
    
    return modules_status

def test_model_initialization() -> Tuple[bool, Any]:
    """测试模型初始化"""
    print_section("测试模型初始化")
    
    try:
        from bma_models.bma_ultra_enhanced_refactored import BMAUltraEnhancedModel
        
        # 创建配置
        config = {
            'temporal': {
                'prediction_horizon_days': 10,
                'cv_gap_days': 5,
            },
            'training': {
                'traditional_models': {
                    'enable': True,
                    'models': ['elastic_net', 'xgboost', 'lightgbm'],
                },
                'regime_aware': {
                    'enable': True,
                },
                'stacking': {
                    'enable': True,
                }
            }
        }
        
        # 初始化模型
        model = BMAUltraEnhancedModel(config)
        
        print("✓ 模型初始化成功")
        print(f"  - 配置加载: 成功")
        print(f"  - 组件初始化: 完成")
        
        # 获取模型摘要
        summary = model.get_model_summary()
        print(f"  - 内存使用: {summary['memory_stats']['rss_mb']:.1f} MB")
        
        return True, model
        
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        traceback.print_exc()
        return False, None

def generate_test_data(n_samples: int = 1000, n_features: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
    """生成测试数据"""
    print_section("生成测试数据")
    
    np.random.seed(42)
    
    # 生成日期和股票
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    n_stocks = 5
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # 创建MultiIndex数据
    data_list = []
    for date in dates:
        for stock in stocks:
            row_data = {
                'date': date,
                'ticker': stock,
            }
            # 添加特征
            for i in range(n_features):
                row_data[f'feature_{i}'] = np.random.randn()
            data_list.append(row_data)
    
    df = pd.DataFrame(data_list)
    df = df.set_index(['date', 'ticker'])
    
    # 生成目标变量
    X = df
    y = pd.Series(
        np.random.randn(len(X)) * 0.01 + 0.001,  # 小幅正收益
        index=X.index,
        name='target'
    )
    
    print(f"✓ 数据生成完成")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - 时间范围: {dates[0]} to {dates[-1]}")
    print(f"  - 股票数量: {n_stocks}")
    
    return X, y

def test_training_pipeline(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """测试完整训练流程"""
    print_section("测试训练流程")
    
    results = {}
    
    try:
        # 1. 数据预处理测试
        print("\n1. 测试数据预处理...")
        X_processed, y_processed = model._safe_data_preprocessing(X, y, training=True)
        print(f"  ✓ 数据预处理完成: {X_processed.shape}")
        results['preprocessing'] = True
        
        # 2. 特征工程测试
        print("\n2. 测试特征工程...")
        X_lagged = model._apply_feature_lag_optimization(X_processed, horizon=10)
        print(f"  ✓ 特征滞后优化: {X_lagged.shape}")
        
        X_selected = model._apply_robust_feature_selection(X_lagged, y_processed, top_k=15)
        print(f"  ✓ 特征选择完成: {X_selected.shape}")
        results['feature_engineering'] = True
        
        # 3. 训练集划分
        print("\n3. 测试训练集划分...")
        split_idx = int(len(X_selected) * 0.8)
        X_train = X_selected.iloc[:split_idx]
        y_train = y_processed.iloc[:split_idx]
        X_val = X_selected.iloc[split_idx:]
        y_val = y_processed.iloc[split_idx:]
        print(f"  ✓ 训练集: {X_train.shape}, 验证集: {X_val.shape}")
        results['data_split'] = True
        
        # 4. 传统ML模型训练
        print("\n4. 测试传统ML模型训练...")
        traditional_results = model._train_standard_models(X_train, y_train, X_val, y_val)
        
        if traditional_results and 'models' in traditional_results:
            print(f"  ✓ 训练成功的模型: {list(traditional_results['models'].keys())}")
            for model_name, score in traditional_results.get('scores', {}).items():
                print(f"    - {model_name} R2: {score:.4f}")
            results['traditional_models'] = True
        else:
            print("  ⚠ 传统模型训练无结果")
            results['traditional_models'] = False
        
        # 5. 制度感知模型训练
        print("\n5. 测试制度感知模型训练...")
        regime_results = model._train_enhanced_regime_aware_models(X_train, y_train, X_val, y_val)
        
        if regime_results and 'models' in regime_results:
            print(f"  ✓ 制度模型数量: {len(regime_results['models'])}")
            results['regime_models'] = True
        else:
            print("  ⚠ 制度感知模型训练无结果")
            results['regime_models'] = False
        
        # 6. Stacking集成
        print("\n6. 测试Stacking集成...")
        if traditional_results and 'predictions' in traditional_results:
            base_predictions = {}
            for name, pred in traditional_results['predictions'].items():
                base_predictions[name] = pred
            
            if len(base_predictions) >= 2:
                # 生成训练集预测
                train_base_predictions = {}
                for name, mdl in traditional_results['models'].items():
                    train_base_predictions[name] = mdl.predict(X_train)
                
                stacking_results = model._train_stacking_models_modular(
                    train_base_predictions, y_train,
                    base_predictions, y_val
                )
                
                if stacking_results and 'meta_learner' in stacking_results:
                    print(f"  ✓ Stacking元学习器训练成功")
                    if 'score' in stacking_results and stacking_results['score'] is not None:
                        print(f"    - Stacking R2: {stacking_results['score']:.4f}")
                    results['stacking'] = True
                else:
                    print("  ⚠ Stacking训练无结果")
                    results['stacking'] = False
            else:
                print("  ⚠ 基础模型不足，跳过Stacking")
                results['stacking'] = False
        
        results['overall'] = True
        print("\n✓ 训练流程测试完成")
        
    except Exception as e:
        print(f"\n✗ 训练流程测试失败: {e}")
        traceback.print_exc()
        results['overall'] = False
        results['error'] = str(e)
    
    return results

def test_prediction_pipeline(model, X_test: pd.DataFrame) -> Dict[str, Any]:
    """测试预测流程"""
    print_section("测试预测流程")
    
    results = {}
    
    try:
        # 生成预测
        predictions = model.generate_enhanced_predictions(X_test, use_ensemble=True)
        
        print(f"✓ 预测生成成功")
        print(f"  - 预测shape: {predictions.shape}")
        print(f"  - 预测列: {list(predictions.columns)}")
        
        # 检查预测值
        for col in predictions.columns:
            pred_values = predictions[col]
            print(f"  - {col}: mean={pred_values.mean():.4f}, std={pred_values.std():.4f}")
            
            # 检查异常值
            if pred_values.isna().any():
                print(f"    ⚠ 包含NaN值: {pred_values.isna().sum()}")
            if np.isinf(pred_values).any():
                print(f"    ⚠ 包含Inf值: {np.isinf(pred_values).sum()}")
        
        results['predictions'] = predictions
        results['success'] = True
        
    except Exception as e:
        print(f"✗ 预测流程测试失败: {e}")
        results['success'] = False
        results['error'] = str(e)
    
    return results

def test_complete_analysis(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """测试完整分析流程"""
    print_section("测试完整分析流程")
    
    try:
        # 运行完整分析
        analysis_results = model.run_complete_analysis(
            X, y,
            test_size=0.2,
            generate_report=True
        )
        
        print("✓ 完整分析运行成功")
        
        # 检查结果
        if analysis_results.get('training'):
            print("  ✓ 训练阶段完成")
        
        if analysis_results.get('predictions') is not None:
            print(f"  ✓ 预测阶段完成: {analysis_results['predictions'].shape}")
        
        if analysis_results.get('performance'):
            print("  ✓ 性能评估完成")
            
            # 找出最佳模型
            best_model = max(
                analysis_results['performance'].items(),
                key=lambda x: x[1].get('r2', -float('inf'))
            )
            print(f"    最佳模型: {best_model[0]} (R2={best_model[1]['r2']:.4f})")
        
        if analysis_results.get('report'):
            print("  ✓ 报告生成完成")
        
        return {
            'success': True,
            'results': analysis_results
        }
        
    except Exception as e:
        print(f"✗ 完整分析失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def verify_all_functions(model) -> Dict[str, bool]:
    """验证所有功能的完整性"""
    print_section("验证功能完整性")
    
    functions = {
        # 数据处理
        '_safe_data_preprocessing': False,
        '_apply_feature_lag_optimization': False,
        '_apply_robust_feature_selection': False,
        
        # 模型训练
        '_train_standard_models': False,
        '_train_enhanced_regime_aware_models': False,
        '_train_stacking_models_modular': False,
        'train_enhanced_models': False,
        
        # 预测
        'generate_enhanced_predictions': False,
        
        # 分析
        'run_complete_analysis': False,
        
        # 工具方法
        'save_model': False,
        'load_model': False,
        'get_feature_importance': False,
        'get_model_summary': False,
    }
    
    for func_name in functions:
        if hasattr(model, func_name):
            functions[func_name] = True
            print(f"✓ {func_name}: 存在")
        else:
            print(f"✗ {func_name}: 缺失")
    
    # 计算完整性
    available = sum(functions.values())
    total = len(functions)
    completeness = available / total * 100
    
    print(f"\n功能完整性: {completeness:.1f}% ({available}/{total})")
    
    return functions

def run_full_validation():
    """运行完整验证"""
    print("="*60)
    print("  BMA Ultra Enhanced 完整验证")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*60)
    
    validation_results = {
        'timestamp': datetime.now(),
        'external_deps': {},
        'bma_modules': {},
        'model_init': False,
        'training': {},
        'prediction': {},
        'complete_analysis': {},
        'functions': {},
        'overall_success': False
    }
    
    try:
        # 1. 检查外部依赖
        validation_results['external_deps'] = check_external_dependencies()
        
        # 2. 检查BMA模块
        validation_results['bma_modules'] = check_bma_modules()
        
        # 3. 初始化模型
        success, model = test_model_initialization()
        validation_results['model_init'] = success
        
        if not success:
            print("\n❌ 模型初始化失败，无法继续")
            return validation_results
        
        # 4. 生成测试数据
        X, y = generate_test_data(n_samples=2000, n_features=20)
        
        # 5. 测试训练流程
        validation_results['training'] = test_training_pipeline(model, X, y)
        
        # 6. 测试预测流程
        X_test = X.iloc[-200:]  # 最后200个样本作为测试
        validation_results['prediction'] = test_prediction_pipeline(model, X_test)
        
        # 7. 测试完整分析
        validation_results['complete_analysis'] = test_complete_analysis(model, X.iloc[:1000], y.iloc[:1000])
        
        # 8. 验证功能完整性
        validation_results['functions'] = verify_all_functions(model)
        
        # 判断整体成功
        validation_results['overall_success'] = (
            validation_results['model_init'] and
            validation_results['training'].get('overall', False) and
            validation_results['prediction'].get('success', False) and
            validation_results['complete_analysis'].get('success', False)
        )
        
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        traceback.print_exc()
        validation_results['error'] = str(e)
    
    # 生成最终报告
    print_section("验证总结")
    
    if validation_results['overall_success']:
        print("✅ 验证成功！BMA Ultra Enhanced模型完全可用")
        print("\n核心功能状态:")
        print("  ✓ 模型初始化: 成功")
        print("  ✓ 数据预处理: 成功")
        print("  ✓ 特征工程: 成功")
        print("  ✓ 模型训练: 成功")
        print("  ✓ 预测生成: 成功")
        print("  ✓ 完整分析: 成功")
    else:
        print("⚠️ 验证部分成功，存在以下问题:")
        if not validation_results['model_init']:
            print("  ✗ 模型初始化失败")
        if not validation_results['training'].get('overall', False):
            print("  ✗ 训练流程存在问题")
        if not validation_results['prediction'].get('success', False):
            print("  ✗ 预测流程存在问题")
        if not validation_results['complete_analysis'].get('success', False):
            print("  ✗ 完整分析流程存在问题")
    
    # 统计信息
    print("\n统计信息:")
    external_ok = sum(validation_results['external_deps'].values())
    external_total = len(validation_results['external_deps'])
    print(f"  - 外部依赖: {external_ok}/{external_total} 可用")
    
    bma_ok = sum(1 for m in validation_results['bma_modules'].values() if m.get('status', False))
    bma_total = len(validation_results['bma_modules'])
    print(f"  - BMA模块: {bma_ok}/{bma_total} 可用")
    
    func_ok = sum(validation_results['functions'].values())
    func_total = len(validation_results['functions'])
    print(f"  - 功能完整性: {func_ok}/{func_total} ({func_ok/func_total*100:.1f}%)")
    
    print("\n" + "="*60)
    print("  验证完成")
    print("="*60)
    
    return validation_results

if __name__ == "__main__":
    results = run_full_validation()
    
    # 保存验证结果
    import json
    with open('bma_validation_results.json', 'w') as f:
        # 转换为可序列化格式
        serializable_results = {
            'timestamp': results['timestamp'].isoformat(),
            'overall_success': results['overall_success'],
            'model_init': results['model_init'],
            'external_deps': results['external_deps'],
            'training_success': results['training'].get('overall', False),
            'prediction_success': results['prediction'].get('success', False),
            'complete_analysis_success': results['complete_analysis'].get('success', False),
            'functions': results['functions'],
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n验证结果已保存至: bma_validation_results.json")