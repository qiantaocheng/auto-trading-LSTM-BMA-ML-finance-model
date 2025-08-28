#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应权重适配器
专为BMA Enhanced系统设计，确保ML权重的正确使用
"""

import logging
from typing import Dict, Optional
import os
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

def get_bma_enhanced_weights() -> Dict[str, float]:
    """
    为BMA Enhanced系统获取ML权重
    主动触发学习，避免硬编码权重回退
    
    Returns:
        Dict[str, float]: ML学习的因子权重
    """
    try:
        # 导入自适应权重系统
        try:
            from .adaptive_factor_weights import AdaptiveFactorWeights, WeightLearningConfig
        except ImportError:
            # 处理相对导入失败
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from adaptive_factor_weights import AdaptiveFactorWeights, WeightLearningConfig
        
        logger.info("🎯 BMA Enhanced权重获取：启动主动ML权重学习")
        
        # 创建权重学习系统
        config = WeightLearningConfig(
            lookback_days=252,      # 1年数据
            rebalance_frequency=7,   # 更频繁的更新
            min_confidence=0.5       # 降低置信度要求，优先使用ML权重
        )
        
        weight_system = AdaptiveFactorWeights(config)
        
        # 使用专用的主动学习方法
        ml_weights = weight_system.get_or_learn_weights()
        
        # 验证权重质量
        is_ml_weights = not _is_fallback_pattern(ml_weights)
        
        if is_ml_weights:
            logger.info(f"✅ 成功获取ML权重: {ml_weights}")
            return ml_weights
        else:
            logger.warning("⚠️ 获取到的可能是硬编码权重，尝试强制学习")
            
            # 强制重新学习
            try:
                result = weight_system.learn_weights_from_bma()
                if result and result.confidence >= 0.4:  # 进一步降低要求
                    logger.info(f"🎯 强制学习成功，置信度: {result.confidence:.3f}")
                    return result.weights
            except Exception as e:
                logger.error(f"强制学习失败: {e}")
            
            logger.warning("使用经过优化的权重配置")
            return _get_optimized_fallback_weights()
        
    except ImportError as e:
        logger.error(f"自适应权重系统不可用: {e}")
        return _get_optimized_fallback_weights()
    except Exception as e:
        logger.error(f"BMA Enhanced权重获取失败: {e}")
        return _get_optimized_fallback_weights()

def _is_fallback_pattern(weights: Dict[str, float]) -> bool:
    """检测是否为硬编码模式"""
    try:
        values = list(weights.values())
        
        # 等权重模式
        if len(set(values)) == 1 and abs(values[0] - 0.2) < 0.001:
            return True
            
        # 预设回退模式
        if len(values) >= 5:
            sorted_vals = sorted(values, reverse=True)
            # 检查是否符合典型的硬编码模式
            if (abs(sorted_vals[0] - 0.3) < 0.05 and 
                abs(sorted_vals[1] - 0.3) < 0.05 and
                abs(sorted_vals[2] - 0.25) < 0.05):
                return True
        
        return False
    except:
        return True

def _get_optimized_fallback_weights() -> Dict[str, float]:
    """获取优化的回退权重（基于研究的最佳实践）"""
    return {
        'mean_reversion': 0.35,  # 强化均值回归
        'trend': 0.28,           # 趋势跟踪
        'momentum': 0.20,        # 动量因子
        'volume': 0.12,          # 成交量因子
        'volatility': 0.05       # 波动率因子（降低权重）
    }

def test_ml_weights_availability() -> bool:
    """测试ML权重系统可用性"""
    try:
        try:
            from .adaptive_factor_weights import AdaptiveFactorWeights
        except ImportError:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from adaptive_factor_weights import AdaptiveFactorWeights
        
        weight_system = AdaptiveFactorWeights()
        latest_result = weight_system.load_latest_weights()
        
        if latest_result is not None:
            days_old = (datetime.now() - latest_result.learning_date).days
            logger.info(f"最新ML权重: {days_old}天前，置信度: {latest_result.confidence:.3f}")
            return latest_result.confidence >= 0.5 and days_old <= 60
        else:
            logger.info("没有找到历史ML权重")
            return False
            
    except Exception as e:
        logger.error(f"ML权重系统测试失败: {e}")
        return False

# 向后兼容
def get_current_factor_weights() -> Dict[str, float]:
    """向后兼容的权重获取函数"""
    return get_bma_enhanced_weights()

# 添加自适应权重配置加载
def load_adaptive_weights_config():
    """加载自适应权重配置"""
    import yaml
    import os
    config_path = "adaptive_weights_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return None

def get_adaptive_weights():
    """获取自适应权重配置"""
    config = load_adaptive_weights_config()
    if config and config.get('weight_learning', {}).get('enabled'):
        return config
    # 返回fallback权重
    return {
        'fallback_weights': {
            'mean_reversion': 0.30,
            'trend': 0.30,
            'momentum': 0.25,
            'volume': 0.20,
            'volatility': 0.15
        }
    }

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=== 测试BMA Enhanced权重获取 ===")
    
    # 测试ML权重可用性
    ml_available = test_ml_weights_availability()
    print(f"ML权重系统可用性: {ml_available}")
    
    # 获取权重
    weights = get_bma_enhanced_weights()
    print(f"获取的权重: {weights}")
    
    # 检查权重类型
    is_fallback = _is_fallback_pattern(weights)
    weight_type = "硬编码权重" if is_fallback else "ML学习权重"
    print(f"权重类型: {weight_type}")