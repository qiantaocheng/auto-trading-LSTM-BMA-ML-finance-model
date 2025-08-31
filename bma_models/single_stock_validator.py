#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单股票情况的专用验证器
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SingleStockValidationConfig:
    """单股票验证配置"""
    min_correlation: float = 0.05  # 时间序列相关性最小值
    min_hit_rate: float = 0.51     # 方向准确率最小值
    min_sharpe: float = 0.3        # Sharpe比率最小值
    min_observations: int = 50     # 最小观测数
    max_drawdown: float = 0.3      # 最大回撤

class SingleStockValidator:
    """单股票验证器 - 使用时间序列验证方法"""
    
    def __init__(self, config: Optional[SingleStockValidationConfig] = None):
        self.config = config or SingleStockValidationConfig()
        
    def validate_single_stock_predictions(self, predictions: np.ndarray, 
                                        returns: np.ndarray,
                                        dates: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        单股票预测验证
        
        Args:
            predictions: 预测值
            returns: 实际收益率
            dates: 日期序列
            
        Returns:
            验证结果字典
        """
        try:
            if len(predictions) != len(returns):
                logger.error(f"单股票验证维度不匹配: pred={len(predictions)}, ret={len(returns)}")
                return {'success': False, 'error': '维度不匹配'}
            
            if len(predictions) < self.config.min_observations:
                logger.warning(f"单股票观测数不足: {len(predictions)} < {self.config.min_observations}")
                return {
                    'success': False, 
                    'reason': 'insufficient_data',
                    'observations': len(predictions)
                }
            
            # 1. 时间序列相关性
            correlation = np.corrcoef(predictions, returns)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
                
            # 2. 方向准确率
            pred_directions = np.sign(predictions)
            ret_directions = np.sign(returns)
            hit_rate = np.mean(pred_directions == ret_directions)
            
            # 3. Sharpe比率 (基于预测信号的策略)
            strategy_returns = predictions * returns  # 简化策略收益
            if np.std(strategy_returns) > 0:
                sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns)
            else:
                sharpe_ratio = 0.0
                
            # 4. 最大回撤
            cumulative_returns = np.cumsum(strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / np.maximum(running_max, 1e-8)
            max_drawdown = abs(np.min(drawdown))
            
            # 5. 信息系数统计
            ic_mean = abs(correlation)  # 使用绝对相关性
            ic_std = np.std([correlation])  # 单值的标准差为0
            t_stat = ic_mean / max(ic_std, 1e-8) if ic_std > 1e-8 else 0
            
            # 验证结果
            validation_results = {
                'correlation': correlation,
                'hit_rate': hit_rate,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'ic_mean': ic_mean,
                'ic_t_stat': t_stat,
                'observations': len(predictions)
            }
            
            # 通过标准检查
            checks = {
                'correlation_check': abs(correlation) >= self.config.min_correlation,
                'hit_rate_check': hit_rate >= self.config.min_hit_rate,
                'sharpe_check': sharpe_ratio >= self.config.min_sharpe,
                'drawdown_check': max_drawdown <= self.config.max_drawdown,
                'data_sufficiency': len(predictions) >= self.config.min_observations
            }
            
            # 计算得分
            score = sum(checks.values()) / len(checks)
            passed = score >= 0.6  # 60%检查通过
            
            result = {
                'success': True,
                'validation_type': 'single_stock_time_series',
                'passed': passed,
                'score': score,
                'metrics': validation_results,
                'checks': checks,
                'recommendation': 'GO' if passed else 'NO_GO'
            }
            
            logger.info(f"单股票验证完成: {'PASS' if passed else 'FAIL'}, 得分: {score:.3f}")
            logger.info(f"  相关性: {correlation:.3f}, 命中率: {hit_rate:.3f}, Sharpe: {sharpe_ratio:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"单股票验证失败: {e}")
            return {'success': False, 'error': str(e)}

def create_single_stock_validator(config: Optional[Dict] = None) -> SingleStockValidator:
    """创建单股票验证器"""
    if config:
        validation_config = SingleStockValidationConfig(**config)
    else:
        validation_config = SingleStockValidationConfig()
    
    return SingleStockValidator(validation_config)