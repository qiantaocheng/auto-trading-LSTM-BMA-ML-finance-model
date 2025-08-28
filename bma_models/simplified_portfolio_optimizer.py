#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化投资组合优化器 - 集成BMA风险模型和预测校准
基于现代投资组合理论(MPT)与Bayesian Model Averaging的实用实现

核心功能:
1. 整合预测校准器的μ预测
2. 使用因子风险模型的Σ估计  
3. 执行均值-方差优化
4. 支持多种约束条件
5. 实时调整和风险控制
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import scipy.optimize as opt
from scipy import linalg

from .prediction_calibrator import get_prediction_calibrator, CalibrationConfig
from .factor_risk_model import get_factor_risk_model, RiskModelConfig

logger = logging.getLogger(__name__)

@dataclass
class PortfolioConstraints:
    """投资组合约束条件"""
    # 基本约束
    max_weight_per_asset: float = 0.1          # 单个资产最大权重
    min_weight_per_asset: float = 0.0          # 单个资产最小权重(一般为0，允许不持有)
    max_total_long: float = 1.0                # 最大多头总权重
    max_total_short: float = 0.0               # 最大空头总权重(0表示不允许做空)
    
    # 集中度约束
    max_sector_weight: float = 0.3             # 单个行业最大权重
    max_top5_concentration: float = 0.5        # 前5大持仓最大集中度
    
    # 换手率约束
    max_turnover: float = 0.2                  # 最大换手率(相对于当前持仓)
    
    # 风险约束
    max_portfolio_volatility: float = 0.25     # 最大组合波动率(年化25%)
    max_tracking_error: float = 0.05           # 最大跟踪误差(如果有基准)

@dataclass
class OptimizationConfig:
    """优化配置"""
    method: str = "mean_variance"              # 优化方法: mean_variance/risk_parity/min_variance
    risk_aversion: float = 3.0                 # 风险厌恶系数(A参数)
    transaction_cost_bps: float = 5.0          # 交易成本(bps)
    rebalance_threshold: float = 0.05          # 重新平衡阈值
    lookback_periods: int = 252                # 风险模型回望期
    shrinkage_intensity: float = 0.1           # 收缩强度

class SimplifiedPortfolioOptimizer:
    """
    简化投资组合优化器
    
    集成核心组件:
    1. PredictionCalibrator - 提供μ(预期收益)
    2. FactorRiskModel - 提供Σ(风险协方差矩阵)
    3. 约束优化求解器
    """
    
    def __init__(self, 
                 constraints: Optional[PortfolioConstraints] = None,
                 config: Optional[OptimizationConfig] = None):
        
        self.constraints = constraints or PortfolioConstraints()
        self.config = config or OptimizationConfig()
        
        # 核心组件
        self.prediction_calibrator = get_prediction_calibrator()
        self.risk_model = get_factor_risk_model()
        
        # 优化历史
        self.optimization_history = []
        self.current_weights = {}
        
        # 统计信息
        self.stats = {
            'total_optimizations': 0,
            'last_optimization_time': None,
            'last_expected_return': 0.0,
            'last_expected_risk': 0.0,
            'last_sharpe_ratio': 0.0
        }
        
        logger.info(f"简化投资组合优化器初始化: 方法={self.config.method}, 风险厌恶={self.config.risk_aversion}")
    
    def optimize_portfolio(self, 
                          signals: Dict[str, float],           # 原始信号 {symbol: score}
                          current_exposures: pd.DataFrame,     # 当前因子暴露度
                          current_weights: Dict[str, float] = None,  # 当前持仓权重
                          market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行投资组合优化
        
        Args:
            signals: 原始预测信号 {symbol: raw_score}
            current_exposures: 因子暴露度矩阵 (index=symbol, columns=factors)
            current_weights: 当前持仓权重 {symbol: weight}
            market_data: 市场数据(价格、成交量等)
            
        Returns:
            {
                'optimal_weights': {symbol: weight},
                'expected_return': float,
                'expected_risk': float,
                'optimization_status': str,
                'constraints_satisfied': bool,
                'diagnostics': {...}
            }
        """
        try:
            start_time = datetime.now()
            current_weights = current_weights or {}
            
            # 1. 信号校准: 原始信号 -> 预期收益率μ
            logger.debug(f"校准{len(signals)}个信号...")
            mu = self.prediction_calibrator.calibrate_scores(signals)
            
            if isinstance(mu, pd.Series):
                mu_dict = mu.to_dict()
            elif isinstance(mu, dict):
                mu_dict = mu
            else:
                mu_dict = dict(zip(signals.keys(), mu))
            
            # 过滤掉无效预测
            valid_symbols = [s for s, m in mu_dict.items() if abs(m) > 1e-6]
            if len(valid_symbols) < 2:
                logger.warning("有效信号不足，无法优化")
                return self._create_empty_result("insufficient_signals")
            
            # 2. 风险模型: 获取协方差矩阵Σ  
            logger.debug(f"构建{len(valid_symbols)}个资产的风险模型...")
            valid_exposures = current_exposures.loc[current_exposures.index.isin(valid_symbols)]
            
            Sigma, aligned_symbols, factors = self.risk_model.get_risk_model(valid_exposures)
            
            # 3. 对齐数据
            mu_aligned = pd.Series([mu_dict.get(s, 0.0) for s in aligned_symbols], 
                                 index=aligned_symbols)
            
            # 4. 执行优化
            if self.config.method == "mean_variance":
                result = self._optimize_mean_variance(mu_aligned, Sigma, aligned_symbols, current_weights)
            elif self.config.method == "min_variance":
                result = self._optimize_min_variance(Sigma, aligned_symbols, current_weights)
            elif self.config.method == "risk_parity":
                result = self._optimize_risk_parity(Sigma, aligned_symbols, current_weights)
            else:
                raise ValueError(f"未知优化方法: {self.config.method}")
            
            # 5. 后处理和验证
            result = self._post_process_weights(result, aligned_symbols, mu_aligned, Sigma)
            
            # 6. 更新统计信息
            self._update_stats(result, start_time)
            
            logger.info(f"组合优化完成: 预期收益{result['expected_return']:.4f}, "
                       f"风险{result['expected_risk']:.4f}, "
                       f"夏普比率{result.get('sharpe_ratio', 0):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"投资组合优化失败: {e}")
            return self._create_empty_result(f"optimization_error: {e}")
    
    def _optimize_mean_variance(self, 
                               mu: pd.Series, 
                               Sigma: np.ndarray, 
                               symbols: List[str],
                               current_weights: Dict[str, float]) -> Dict[str, Any]:
        """均值-方差优化 (经典Markowitz)"""
        n = len(symbols)
        
        # 目标函数: 最大化 μ^T w - (A/2) w^T Σ w - 交易成本
        def objective(w):
            portfolio_return = np.dot(mu.values, w)
            portfolio_risk = 0.5 * self.config.risk_aversion * np.dot(w, np.dot(Sigma, w))
            
            # 交易成本(相对于当前持仓)
            current_w = np.array([current_weights.get(s, 0.0) for s in symbols])
            turnover = np.sum(np.abs(w - current_w))
            transaction_cost = turnover * self.config.transaction_cost_bps / 10000
            
            return -(portfolio_return - portfolio_risk - transaction_cost)
        
        # 约束条件
        constraints = self._build_constraints(n, symbols, current_weights)
        
        # 边界条件
        bounds = [(self.constraints.min_weight_per_asset, 
                  self.constraints.max_weight_per_asset) for _ in range(n)]
        
        # 初始猜测(等权重或当前权重)
        if current_weights:
            x0 = np.array([current_weights.get(s, 1.0/n) for s in symbols])
            x0 = x0 / np.sum(x0)  # 标准化
        else:
            x0 = np.ones(n) / n
        
        # 优化求解
        result = opt.minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-9})
        
        if not result.success:
            logger.warning(f"优化求解失败: {result.message}")
        
        optimal_weights = dict(zip(symbols, result.x))
        
        return {
            'optimal_weights': optimal_weights,
            'optimization_status': 'success' if result.success else 'failed',
            'solver_message': result.message,
            'iterations': result.nit,
            'objective_value': -result.fun
        }
    
    def _optimize_min_variance(self, 
                              Sigma: np.ndarray, 
                              symbols: List[str],
                              current_weights: Dict[str, float]) -> Dict[str, Any]:
        """最小方差优化"""
        n = len(symbols)
        
        # 目标函数: 最小化 w^T Σ w
        def objective(w):
            return np.dot(w, np.dot(Sigma, w))
        
        constraints = self._build_constraints(n, symbols, current_weights)
        bounds = [(self.constraints.min_weight_per_asset, 
                  self.constraints.max_weight_per_asset) for _ in range(n)]
        
        x0 = np.ones(n) / n
        
        result = opt.minimize(objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        
        return {
            'optimal_weights': dict(zip(symbols, result.x)),
            'optimization_status': 'success' if result.success else 'failed',
            'solver_message': result.message
        }
    
    def _optimize_risk_parity(self, 
                             Sigma: np.ndarray, 
                             symbols: List[str],
                             current_weights: Dict[str, float]) -> Dict[str, Any]:
        """风险平价优化"""
        n = len(symbols)
        
        # 目标函数: 最小化风险贡献的平方差
        def objective(w):
            portfolio_vol = np.sqrt(np.dot(w, np.dot(Sigma, w)))
            risk_contributions = w * np.dot(Sigma, w) / portfolio_vol
            target_risk = portfolio_vol / n  # 等风险贡献
            return np.sum((risk_contributions - target_risk) ** 2)
        
        constraints = self._build_constraints(n, symbols, current_weights)
        bounds = [(self.constraints.min_weight_per_asset, 
                  self.constraints.max_weight_per_asset) for _ in range(n)]
        
        x0 = np.ones(n) / n
        
        result = opt.minimize(objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        
        return {
            'optimal_weights': dict(zip(symbols, result.x)),
            'optimization_status': 'success' if result.success else 'failed',
            'solver_message': result.message
        }
    
    def _build_constraints(self, n: int, symbols: List[str], 
                          current_weights: Dict[str, float]) -> List[Dict]:
        """构建约束条件"""
        constraints = []
        
        # 权重和约束: Σw = 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # 换手率约束
        if self.constraints.max_turnover < 1.0 and current_weights:
            def turnover_constraint(w):
                current_w = np.array([current_weights.get(s, 0.0) for s in symbols])
                turnover = np.sum(np.abs(w - current_w))
                return self.constraints.max_turnover - turnover
            
            constraints.append({
                'type': 'ineq',
                'fun': turnover_constraint
            })
        
        return constraints
    
    def _post_process_weights(self, 
                             result: Dict[str, Any], 
                             symbols: List[str],
                             mu: pd.Series, 
                             Sigma: np.ndarray) -> Dict[str, Any]:
        """后处理权重和计算性能指标"""
        
        weights = result['optimal_weights']
        w = np.array([weights[s] for s in symbols])
        
        # 计算组合性能指标
        portfolio_return = np.dot(mu.values, w)
        portfolio_variance = np.dot(w, np.dot(Sigma, w))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # 夏普比率(假设无风险利率为0)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # 约束检查
        constraints_satisfied = True
        constraint_violations = []
        
        # 检查权重约束
        if np.max(w) > self.constraints.max_weight_per_asset + 1e-6:
            constraints_satisfied = False
            constraint_violations.append("max_weight_violation")
        
        if portfolio_volatility > self.constraints.max_portfolio_volatility:
            constraints_satisfied = False
            constraint_violations.append("volatility_violation")
        
        # 清理微小权重
        cleaned_weights = {s: max(0, w) for s, w in weights.items() if abs(w) > 1e-4}
        total_weight = sum(cleaned_weights.values())
        if total_weight > 0:
            cleaned_weights = {s: w/total_weight for s, w in cleaned_weights.items()}
        
        result.update({
            'optimal_weights': cleaned_weights,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'constraints_satisfied': constraints_satisfied,
            'constraint_violations': constraint_violations,
            'diagnostics': {
                'num_assets': len([w for w in cleaned_weights.values() if w > 1e-4]),
                'max_weight': max(cleaned_weights.values()) if cleaned_weights else 0,
                'weight_concentration': np.sum(np.array(list(cleaned_weights.values()))**2),
                'total_weight': sum(cleaned_weights.values())
            }
        })
        
        return result
    
    def _create_empty_result(self, reason: str) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'optimal_weights': {},
            'expected_return': 0.0,
            'expected_risk': 0.0,
            'optimization_status': 'failed',
            'constraints_satisfied': False,
            'error_reason': reason,
            'diagnostics': {}
        }
    
    def _update_stats(self, result: Dict[str, Any], start_time: datetime):
        """更新统计信息"""
        self.stats['total_optimizations'] += 1
        self.stats['last_optimization_time'] = start_time
        self.stats['last_expected_return'] = result.get('expected_return', 0.0)
        self.stats['last_expected_risk'] = result.get('expected_risk', 0.0)
        self.stats['last_sharpe_ratio'] = result.get('sharpe_ratio', 0.0)
        
        # 记录优化历史
        self.optimization_history.append({
            'timestamp': start_time,
            'status': result.get('optimization_status'),
            'num_assets': len(result.get('optimal_weights', {})),
            'expected_return': result.get('expected_return', 0.0),
            'expected_risk': result.get('expected_risk', 0.0)
        })
        
        # 保持历史记录在合理范围内
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
    
    def update_current_weights(self, weights: Dict[str, float]):
        """更新当前持仓权重"""
        self.current_weights = weights.copy()
        logger.debug(f"更新当前持仓: {len(weights)}个资产")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """获取优化统计信息"""
        stats = self.stats.copy()
        stats.update({
            'prediction_calibrator_stats': self.prediction_calibrator.get_calibration_stats(),
            'risk_model_stats': self.risk_model.get_model_stats(),
            'constraints': {
                'max_weight_per_asset': self.constraints.max_weight_per_asset,
                'max_turnover': self.constraints.max_turnover,
                'max_portfolio_volatility': self.constraints.max_portfolio_volatility
            }
        })
        return stats


# 全局实例
_global_optimizer: Optional[SimplifiedPortfolioOptimizer] = None

def get_portfolio_optimizer(constraints: Optional[PortfolioConstraints] = None,
                           config: Optional[OptimizationConfig] = None) -> SimplifiedPortfolioOptimizer:
    """获取全局投资组合优化器实例"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = SimplifiedPortfolioOptimizer(constraints, config)
    return _global_optimizer


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建优化器
    optimizer = SimplifiedPortfolioOptimizer()
    
    # 模拟数据
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    signals = {s: np.zeros(1) for s in symbols}  # 随机信号
    
    # 模拟因子暴露度
    factor_names = ['market', 'size', 'value', 'momentum']
    exposures = pd.DataFrame(
        np.zeros(len(symbols), len(factor_names)),
        index=symbols,
        columns=factor_names
    )
    
    print(f"测试信号: {signals}")
    print(f"因子暴露度:\n{exposures}")
    
    # 执行优化
    result = optimizer.optimize_portfolio(signals, exposures)
    
    print(f"\n优化结果:")
    print(f"状态: {result['optimization_status']}")
    print(f"权重: {result['optimal_weights']}")
    print(f"预期收益: {result['expected_return']:.4f}")
    print(f"预期风险: {result['expected_risk']:.4f}")
    print(f"约束满足: {result['constraints_satisfied']}")
    
    # 统计信息
    stats = optimizer.get_optimization_stats()
    print(f"\n优化器统计: {stats}")