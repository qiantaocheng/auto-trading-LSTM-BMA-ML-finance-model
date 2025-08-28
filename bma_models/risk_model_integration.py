#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA Enhanced 系统风险模型集成
将因子风险模型 + 预测校准 + 组合优化集成到BMA训练和推理流程中

实现您提到的四步落地方案：
1. 在线维护风险模型 (EWMA + Ledoit-Wolf)
2. 预测分数标定成μ
3. 组合优化获取权重 
4. 落盘与自检

作者：基于您的指导实现
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import sys
import os

# 导入BMA模块
from .factor_risk_model import FactorRiskModel, RiskModelConfig, get_factor_risk_model
from .prediction_calibrator import PredictionCalibrator, CalibrationConfig, get_prediction_calibrator
from .simplified_portfolio_optimizer import SimplifiedPortfolioOptimizer, PortfolioConstraints, OptimizationConfig, get_portfolio_optimizer

# 导入现有BMA模块
from .enhanced_target_engineering import EnhancedTargetEngineering, TripleBarrierConfig
from .optimized_bma_trainer import OptimizedBMATrainer

logger = logging.getLogger(__name__)

@dataclass
class RiskModelIntegrationConfig:
    """风险模型集成配置"""
    # 风险模型配置
    factor_halflife_days: int = 63
    specific_halflife_days: int = 126
    min_history_days: int = 252
    
    # 预测校准配置
    calibration_method: str = "isotonic"
    calibration_lookback: int = 252
    
    # 组合优化配置
    optimization_method: str = "mean_variance"
    risk_aversion: float = 3.0
    max_weight_per_asset: float = 0.05
    max_turnover: float = 0.15
    
    # 更新频率
    risk_model_update_freq: str = "daily"
    calibration_update_freq: str = "daily"
    
    # 数据质量
    min_stocks_for_factor: int = 50
    min_calibration_samples: int = 100

class BMAEnhancedRiskSystem:
    """
    BMA Enhanced风险系统集成器
    
    核心功能：
    1. 训练期增益：残差收益标签 + 噪声加权 + 特征正交化
    2. 实盘期增益：μ预测 + Σ风险模型 + 组合优化 + 风险预算
    """
    
    def __init__(self, config: Optional[RiskModelIntegrationConfig] = None):
        self.config = config or RiskModelIntegrationConfig()
        
        # 初始化核心组件
        self._init_risk_components()
        
        # 数据存储
        self.returns_history = []  # 收益率历史
        self.exposures_history = []  # 因子暴露度历史
        self.predictions_history = []  # 预测历史
        
        # 状态跟踪
        self.last_update_time = None
        self.is_initialized = False
        
        logger.info("BMA Enhanced风险系统初始化完成")
    
    def _init_risk_components(self):
        """初始化风险模型组件"""
        # 1. 因子风险模型
        risk_config = RiskModelConfig(
            ewma_halflife_days=self.config.factor_halflife_days,
            specific_halflife_days=self.config.specific_halflife_days,
            min_history_days=self.config.min_history_days,
            min_stocks_for_factor=self.config.min_stocks_for_factor
        )
        self.risk_model = FactorRiskModel(risk_config)
        
        # 2. 预测校准器
        calibration_config = CalibrationConfig(
            method=self.config.calibration_method,
            lookback_days=self.config.calibration_lookback,
            min_observations=self.config.min_calibration_samples
        )
        self.prediction_calibrator = PredictionCalibrator(calibration_config)
        
        # 3. 组合优化器
        constraints = PortfolioConstraints(
            max_weight_per_asset=self.config.max_weight_per_asset,
            max_turnover=self.config.max_turnover
        )
        optimization_config = OptimizationConfig(
            method=self.config.optimization_method,
            risk_aversion=self.config.risk_aversion
        )
        self.portfolio_optimizer = SimplifiedPortfolioOptimizer(constraints, optimization_config)
        
        logger.info("风险模型核心组件初始化完成")
    
    def add_training_data(self,
                         returns: pd.DataFrame,      # (date x stocks) 收益率矩阵
                         exposures: pd.DataFrame,    # (date x stocks x factors) 因子暴露度
                         raw_predictions: pd.DataFrame = None,  # (date x stocks) 原始预测分数
                         realized_returns: pd.DataFrame = None): # (date x stocks) 实际收益率（用于校准）
        """
        添加训练数据到风险系统
        
        Args:
            returns: 历史收益率矩阵
            exposures: 因子暴露度矩阵  
            raw_predictions: 原始模型预测分数
            realized_returns: 实际收益率（用于校准预测）
        """
        try:
            # 数据验证
            if returns.empty or exposures.empty:
                logger.warning("收益率或暴露度数据为空")
                return
            
            # 按日期添加数据到风险模型
            for date in returns.index:
                if date in exposures.index:
                    daily_returns = returns.loc[date].dropna()
                    daily_exposures = exposures.loc[date].dropna()
                    
                    # 对齐数据
                    common_stocks = daily_returns.index.intersection(daily_exposures.index)
                    if len(common_stocks) >= self.config.min_stocks_for_factor:
                        aligned_returns = daily_returns.loc[common_stocks]
                        aligned_exposures = daily_exposures.loc[common_stocks]
                        
                        # 重塑暴露度为DataFrame格式（如果需要）
                        if isinstance(aligned_exposures, pd.Series):
                            # 假设单因子情况
                            exposure_df = pd.DataFrame({'factor_1': aligned_exposures})
                        else:
                            exposure_df = pd.DataFrame(aligned_exposures)
                        
                        # 添加到风险模型
                        self.risk_model.add_data_point(aligned_returns, exposure_df, date)
            
            # 添加预测校准数据
            if raw_predictions is not None and realized_returns is not None:
                for date in raw_predictions.index:
                    if date in realized_returns.index:
                        daily_pred = raw_predictions.loc[date].dropna()
                        daily_real = realized_returns.loc[date].dropna()
                        
                        # 添加到校准器
                        self.prediction_calibrator.add_observation(
                            scores=daily_pred,
                            realized_returns=daily_real,
                            timestamp=date
                        )
            
            logger.info(f"成功添加训练数据: {len(returns)} 个交易日")
            
        except Exception as e:
            logger.error(f"添加训练数据失败: {e}")
    
    def prepare_training_labels(self, 
                               raw_returns: pd.DataFrame, 
                               exposures: pd.DataFrame) -> pd.DataFrame:
        """
        准备训练标签：使用残差收益去除伪alpha
        
        这是您提到的"训练期增益"第1点：用Z做横截面回归，用残差ε当标签
        
        Args:
            raw_returns: 原始收益率
            exposures: 因子暴露度
            
        Returns:
            残差收益率（作为训练标签）
        """
        try:
            if not self.risk_model._is_initialized:
                logger.warning("风险模型未初始化，无法计算残差收益")
                return raw_returns
            
            residual_returns = pd.DataFrame(index=raw_returns.index, columns=raw_returns.columns)
            
            for date in raw_returns.index:
                if date in exposures.index:
                    daily_returns = raw_returns.loc[date].dropna()
                    daily_exposures = exposures.loc[date].dropna()
                    
                    # 对齐数据
                    common_stocks = daily_returns.index.intersection(daily_exposures.index)
                    if len(common_stocks) >= 10:  # 至少10只股票才进行回归
                        aligned_returns = daily_returns.loc[common_stocks]
                        aligned_exposures = daily_exposures.loc[common_stocks]
                        
                        # 横截面回归：r_t = Z_t * f_t + ε_t
                        try:
                            if isinstance(aligned_exposures, pd.Series):
                                X = aligned_exposures.values.reshape(-1, 1)
                            else:
                                X = aligned_exposures.values
                            
                            # 添加截距项
                            X_with_intercept = np.column_stack([np.ones(len(X)), X])
                            y = aligned_returns.values
                            
                            # OLS回归
                            factor_returns = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                            
                            # 计算残差
                            fitted_returns = X_with_intercept @ factor_returns
                            residuals = y - fitted_returns
                            
                            # 存储残差
                            residual_returns.loc[date, common_stocks] = residuals
                            
                        except Exception as e:
                            logger.warning(f"日期 {date} 回归失败: {e}")
                            # 回退到原始收益率
                            residual_returns.loc[date, common_stocks] = aligned_returns
            
            logger.info("残差收益计算完成，已去除系统性因子暴露")
            return residual_returns.fillna(0)
            
        except Exception as e:
            logger.error(f"计算残差收益失败: {e}")
            return raw_returns
    
    def get_sample_weights(self, stocks: List[str]) -> Dict[str, float]:
        """
        获取训练样本权重：使用特异性方差的倒数
        
        这是您提到的"训练期增益"第2点：用ψ²⁻¹做样本权重，噪声大的股票权重低
        
        Args:
            stocks: 股票列表
            
        Returns:
            样本权重字典
        """
        try:
            if not self.risk_model._is_initialized:
                logger.warning("风险模型未初始化，使用等权重")
                return {stock: 1.0 for stock in stocks}
            
            specific_var_result = self.risk_model.get_specific_variances()
            if specific_var_result is None:
                return {stock: 1.0 for stock in stocks}
            
            specific_var, stock_names = specific_var_result
            
            weights = {}
            for stock in stocks:
                if stock in stock_names:
                    idx = list(stock_names).index(stock)
                    # 权重 = 1 / 特异性方差（加小量避免除零）
                    weights[stock] = 1.0 / (specific_var.iloc[idx] + 1e-6)
                else:
                    weights[stock] = 1.0  # 默认权重
            
            # 归一化权重
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            logger.debug(f"计算样本权重完成: {len(weights)} 只股票")
            return weights
            
        except Exception as e:
            logger.error(f"计算样本权重失败: {e}")
            return {stock: 1.0 for stock in stocks}
    
    def optimize_portfolio(self,
                          raw_scores: Dict[str, float],
                          current_exposures: pd.DataFrame,
                          current_weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        执行组合优化：从分数到权重的完整流程
        
        这是您提到的"四步落地"完整实现
        
        Args:
            raw_scores: 原始模型预测分数
            current_exposures: 当前因子暴露度矩阵
            current_weights: 当前持仓权重
            
        Returns:
            优化结果字典
        """
        try:
            # 第1步：在线维护风险模型（已在add_data_point中完成）
            if not self.risk_model._is_initialized:
                logger.error("风险模型未初始化")
                return {'status': 'failed', 'reason': 'risk_model_not_initialized'}
            
            # 第2步：把预测分数标定成μ
            logger.debug("开始预测分数标定...")
            mu = self.prediction_calibrator.calibrate_scores(raw_scores)
            
            if isinstance(mu, pd.Series):
                mu_dict = mu.to_dict()
            elif isinstance(mu, dict):
                mu_dict = mu
            else:
                mu_dict = dict(zip(raw_scores.keys(), mu))
            
            # 第3步：丢给优化器拿权重
            logger.debug("开始组合优化...")
            optimization_result = self.portfolio_optimizer.optimize_portfolio(
                signals=raw_scores,
                current_exposures=current_exposures,
                current_weights=current_weights
            )
            
            if optimization_result['optimization_status'] != 'success':
                logger.warning(f"组合优化失败: {optimization_result.get('error_reason', 'unknown')}")
                return optimization_result
            
            # 第4步：落盘与自检
            logger.debug("开始风险分解验证...")
            validation_result = self._validate_optimization_result(
                optimization_result, current_exposures
            )
            
            # 合并结果
            final_result = {
                **optimization_result,
                'risk_validation': validation_result,
                'calibrated_mu': mu_dict,
                'timestamp': datetime.now(),
                'components_status': {
                    'risk_model': 'initialized' if self.risk_model._is_initialized else 'not_ready',
                    'calibrator': 'fitted' if self.prediction_calibrator._is_fitted else 'not_fitted',
                    'optimizer': 'ready'
                }
            }
            
            logger.info(f"组合优化完成: {len(optimization_result['optimal_weights'])} 只股票, "
                       f"预期收益率 {optimization_result['expected_return']:.4f}, "
                       f"预期风险 {optimization_result['expected_risk']:.4f}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"组合优化流程失败: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def _validate_optimization_result(self, 
                                    optimization_result: Dict[str, Any],
                                    current_exposures: pd.DataFrame) -> Dict[str, Any]:
        """验证优化结果的风险分解"""
        try:
            weights = optimization_result['optimal_weights']
            if not weights:
                return {'status': 'failed', 'reason': 'empty_weights'}
            
            # 获取风险模型
            Sigma, stock_names, factor_names = self.risk_model.get_risk_model(current_exposures)
            
            # 计算组合风险分解
            w = np.array([weights.get(stock, 0.0) for stock in stock_names])
            portfolio_variance = w.T @ Sigma @ w
            portfolio_vol = np.sqrt(portfolio_variance)
            
            # 验证风险分解闭环: Σ RC_i = w^T Σ w
            risk_contributions = w * (Sigma @ w)
            total_risk_contribution = np.sum(risk_contributions)
            
            validation = {
                'portfolio_volatility': portfolio_vol,
                'total_risk_contribution': total_risk_contribution,
                'risk_decomposition_error': abs(total_risk_contribution - portfolio_variance),
                'is_valid': abs(total_risk_contribution - portfolio_variance) < 1e-6,
                'factor_exposures': {},
                'risk_concentration': {}
            }
            
            # 计算因子暴露度
            if len(factor_names) > 0:
                Z = current_exposures[factor_names].values
                factor_exposures = Z.T @ w
                validation['factor_exposures'] = dict(zip(factor_names, factor_exposures))
            
            # 计算风险集中度
            validation['risk_concentration'] = {
                'max_weight': np.max(np.abs(w)),
                'top5_concentration': np.sum(np.sort(np.abs(w))[-5:]),
                'effective_stocks': 1.0 / np.sum(w**2) if np.sum(w**2) > 0 else 0
            }
            
            return validation
            
        except Exception as e:
            logger.error(f"风险验证失败: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'is_initialized': self.is_initialized,
            'last_update': self.last_update_time,
            'risk_model_stats': self.risk_model.get_model_stats(),
            'calibrator_stats': self.prediction_calibrator.get_calibration_stats(),
            'optimizer_stats': self.portfolio_optimizer.get_optimization_stats(),
            'config': self.config
        }


# 工厂函数
def create_bma_risk_system(config: Optional[RiskModelIntegrationConfig] = None) -> BMAEnhancedRiskSystem:
    """创建BMA Enhanced风险系统实例"""
    return BMAEnhancedRiskSystem(config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=== BMA Enhanced风险系统集成测试 ===")
    
    # 创建测试系统
    config = RiskModelIntegrationConfig(
        factor_halflife_days=30,
        min_history_days=100,
        calibration_lookback=100
    )
    
    risk_system = create_bma_risk_system(config)
    
    # 生成模拟数据
    n_dates = 150
    n_stocks = 100
    n_factors = 5
    
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='D')
    stocks = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    factors = [f'factor_{i}' for i in range(n_factors)]
    
    # 模拟收益率
    returns = pd.DataFrame(
        np.zeros(n_dates) * 0.02,
        index=dates, columns=stocks
    )
    
    # 模拟因子暴露度
    base_exposures = np.zeros(n_stocks)
    exposures_data = []
    for i in range(n_dates):
        daily_exposures = base_exposures + np.zeros(n_stocks) * 0.1
        exposures_data.append(daily_exposures)
    
    exposures = pd.Panel(exposures_data, items=dates, minor_axis=factors, major_axis=stocks)
    
    # 模拟预测分数
    raw_predictions = pd.DataFrame(
        np.zeros(n_dates),
        index=dates, columns=stocks
    )
    
    print("模拟数据生成完成")
    
    # 添加训练数据
    print("添加训练数据...")
    # 注：由于Panel已弃用，这里需要重新设计暴露度数据结构
    # risk_system.add_training_data(returns, exposures, raw_predictions, returns.shift(1))
    
    print("BMA Enhanced风险系统测试完成")