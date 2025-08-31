#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两段式特征选择使用示例
展示如何在主BMA系统中使用两段式特征选择
"""

import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_two_stage_feature_selection():
    """演示两段式特征选择的完整流程"""
    
    print("=" * 80)
    print("两段式特征选择演示")
    print("=" * 80)
    
    # 1. 模拟数据创建
    print("\n1. 创建模拟数据...")
    data = create_mock_data()
    print(f"   原始数据: {data.shape[0]} 样本, {data.shape[1]} 列")
    
    # 2. 配置两段式特征选择
    print("\n2. 配置两段式特征选择...")
    from bma_models.two_stage_feature_config import TwoStageFeatureConfig, TwoStageFeatureManager
    
    # 使用默认配置
    config = TwoStageFeatureConfig.default()
    manager = TwoStageFeatureManager(config)
    
    print(f"   Stage-A目标特征数: {config.stage_a.target_features}")
    print(f"   Stage-B模式: {config.stage_b.mode}")
    print(f"   反窥视保护: {config.anti_snooping_enabled}")
    
    # 3. 执行Stage-A (全局稳健特征选择)
    print("\n3. 执行Stage-A - 全局稳健特征选择...")
    stage_a_result = run_stage_a_demo(data, manager)
    
    if not stage_a_result['success']:
        print(f"   ❌ Stage-A失败: {stage_a_result['error']}")
        return
    
    selected_features = stage_a_result['selected_features']
    print(f"   ✅ Stage-A完成: {len(data.columns)-3} -> {len(selected_features)} 特征")
    print(f"   选择的特征: {selected_features[:5]}...")
    
    # 4. 准备Stage-B数据
    print("\n4. 准备Stage-B数据...")
    stage_b_data = data[selected_features + ['target', 'date', 'ticker']].copy()
    print(f"   Stage-B数据: {stage_b_data.shape}")
    
    # 5. 执行Stage-B (模型内收缩)
    print("\n5. 执行Stage-B - 模型内收缩训练...")
    stage_b_result = run_stage_b_demo(stage_b_data, manager)
    
    if stage_b_result['success']:
        print("   ✅ Stage-B训练完成")
        print(f"   训练结果: {list(stage_b_result['metrics'].keys())}")
    else:
        print(f"   ❌ Stage-B失败: {stage_b_result.get('error', 'Unknown')}")
    
    # 6. 性能报告
    print("\n6. 生成性能报告...")
    report = manager.get_performance_report()
    print(f"   Stage-A状态: {report['stage_a_status']}")
    print(f"   Stage-B状态: {report['stage_b_status']}")
    
    print("\n=" * 80)
    print("两段式特征选择演示完成")
    print("=" * 80)

def create_mock_data(n_samples: int = 1000, n_features: int = 50) -> pd.DataFrame:
    """创建模拟数据"""
    np.random.seed(42)
    
    # 创建日期序列
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # 创建股票代码
    tickers = np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'], n_samples)
    
    # 创建特征
    features = {}
    
    # 创建一些有预测能力的特征
    base_signal = np.cumsum(np.random.randn(n_samples) * 0.01)
    
    for i in range(n_features):
        if i < 10:
            # 高质量特征 - 与目标相关
            noise_level = 0.5
            lag = np.random.choice([1, 2, 3])
            feature = np.roll(base_signal, -lag) + np.random.randn(n_samples) * noise_level
        elif i < 25:
            # 中等质量特征 - 弱相关
            noise_level = 1.0
            lag = np.random.choice([1, 2])
            feature = np.roll(base_signal, -lag) + np.random.randn(n_samples) * noise_level
        else:
            # 低质量特征 - 噪声
            feature = np.random.randn(n_samples)
        
        features[f'feature_{i:02d}'] = feature
    
    # 创建目标变量
    future_return = np.diff(base_signal, prepend=base_signal[0])
    target = future_return + np.random.randn(n_samples) * 0.3
    
    # 组合数据
    data = pd.DataFrame(features)
    data['target'] = target
    data['date'] = dates
    data['ticker'] = tickers
    
    return data

def run_stage_a_demo(data: pd.DataFrame, manager) -> dict:
    """运行Stage-A演示"""
    try:
        # 创建Stage-A选择器
        selector = manager.create_stage_a_selector()
        if selector is None:
            return {'success': False, 'error': 'Stage-A选择器创建失败'}
        
        # 准备数据
        feature_cols = [col for col in data.columns 
                       if col not in ['target', 'date', 'ticker']]
        X = data[feature_cols].fillna(0)
        y = data['target'].fillna(0)
        dates = data['date']
        
        # 执行特征选择
        X_selected = selector.fit_transform(X, y, dates)
        selected_features = X_selected.columns.tolist()
        
        return {
            'success': True,
            'selected_features': selected_features,
            'selector': selector
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def run_stage_b_demo(data: pd.DataFrame, manager) -> dict:
    """运行Stage-B演示"""
    try:
        # 创建Stage-B训练器
        trainer = manager.create_stage_b_trainer()
        if trainer is None:
            return {'success': False, 'error': 'Stage-B训练器创建失败'}
        
        # 准备数据
        feature_cols = [col for col in data.columns 
                       if col not in ['target', 'date', 'ticker']]
        X = data[feature_cols].fillna(0)
        y = data['target'].fillna(0)
        dates = data['date'] if 'date' in data.columns else None
        
        # 执行训练
        result = trainer.train_models(X=X, y=y, dates=dates)
        
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def demo_bma_integration():
    """演示BMA系统集成"""
    
    print("\n" + "=" * 80)
    print("BMA系统集成演示")
    print("=" * 80)
    
    # 1. 创建模拟BMA模型
    print("\n1. 创建模拟BMA模型...")
    
    class MockBMAModel:
        def __init__(self):
            self.name = "Mock BMA Ultra Enhanced Model"
            self.model_cache = {}
        
        def train_enhanced_models(self, feature_data, current_ticker=None):
            """原有的训练方法"""
            print(f"   调用原有训练方法: {feature_data.shape}")
            return {
                'success': True, 
                'method': 'original',
                'features_used': feature_data.shape[1] - 3  # 减去target, date, ticker
            }
        
        def get_feature_data(self):
            """模拟特征数据获取"""
            return create_mock_data(n_samples=500, n_features=30)
    
    bma_model = MockBMAModel()
    print(f"   模型名称: {bma_model.name}")
    
    # 2. 集成两段式特征选择
    print("\n2. 集成两段式特征选择...")
    from bma_models.two_stage_integration import integrate_two_stage_feature_selection
    
    integrator = integrate_two_stage_feature_selection(bma_model, 'default')
    
    # 3. 验证集成
    print("\n3. 验证集成...")
    validation = integrator.validate_integration()
    print(f"   集成成功: {validation['integration_successful']}")
    
    for component, status in validation['components_status'].items():
        status_emoji = "✅" if status else "❌"
        print(f"   {component}: {status_emoji}")
    
    # 4. 测试集成后的功能
    print("\n4. 测试集成后的功能...")
    
    # 获取模拟数据
    test_data = bma_model.get_feature_data()
    print(f"   测试数据: {test_data.shape}")
    
    # 测试两段式特征选择
    if hasattr(bma_model, 'two_stage_feature_selection'):
        print("   执行两段式特征选择...")
        selected_data, metadata = bma_model.two_stage_feature_selection(
            test_data, 'target', 'date')
        print(f"   特征选择结果: {test_data.shape} -> {selected_data.shape}")
        print(f"   减少比例: {metadata.get('reduction_ratio', 0):.2%}")
    
    # 测试Stage-B训练
    if hasattr(bma_model, 'enhanced_ml_training_with_stage_b'):
        print("   执行Stage-B训练...")
        training_result = bma_model.enhanced_ml_training_with_stage_b(selected_data)
        print(f"   训练成功: {training_result.get('success', False)}")
    
    # 5. 性能报告
    print("\n5. 性能报告...")
    if hasattr(bma_model, 'get_two_stage_performance_report'):
        report = bma_model.get_two_stage_performance_report()
        print(f"   配置模式: {report.get('config_mode', 'unknown')}")
        print(f"   Stage-A完成: {report.get('stage_a_completed', False)}")
        print(f"   Stage-B启用: {report.get('stage_b_enabled', False)}")
    
    print("\n=" * 80)
    print("BMA系统集成演示完成")
    print("=" * 80)

def demo_configuration_modes():
    """演示不同配置模式"""
    
    print("\n" + "=" * 80)
    print("配置模式对比演示")
    print("=" * 80)
    
    from bma_models.two_stage_feature_config import TwoStageFeatureConfig
    
    configs = {
        'default': TwoStageFeatureConfig.default(),
        'conservative': TwoStageFeatureConfig.conservative(),
        'aggressive': TwoStageFeatureConfig.aggressive()
    }
    
    for mode_name, config in configs.items():
        print(f"\n{mode_name.upper()} 配置:")
        print(f"   Stage-A目标特征: {config.stage_a.target_features}")
        print(f"   IC阈值: {config.stage_a.min_ic_mean}")
        print(f"   IR阈值: {config.stage_a.min_ic_ir}")
        print(f"   Stage-B模式: {config.stage_b.mode}")
        print(f"   最大特征阈值: {config.stage_b.max_features_threshold}")
        
        # 验证配置
        warnings = config.validate()
        if warnings:
            print(f"   ⚠️ 警告数量: {len(warnings)}")
        else:
            print("   ✅ 配置验证通过")


if __name__ == "__main__":
    try:
        # 基础演示
        demo_two_stage_feature_selection()
        
        # BMA集成演示
        demo_bma_integration()
        
        # 配置模式演示
        demo_configuration_modes()
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n🎉 所有演示完成！")
    print("\n使用说明:")
    print("1. 在主BMA系统中导入: from bma_models.two_stage_integration import integrate_two_stage_feature_selection")
    print("2. 集成到模型: integrator = integrate_two_stage_feature_selection(bma_model, 'default')")
    print("3. 使用两段式特征选择: selected_data, metadata = bma_model.two_stage_feature_selection(data)")
    print("4. 使用Stage-B训练: result = bma_model.enhanced_ml_training_with_stage_b(selected_data)")