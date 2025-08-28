#!/usr/bin/env python3
"""
因子风险模型在线维护系统
实现EWMA/Ledoit-Wolf估计的因子协方差矩阵F和特异性方差Ψ²
为组合优化器提供风险模型Σ = ZFZ^T + Ψ
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from scipy import linalg

logger = logging.getLogger(__name__)


@dataclass
class RiskModelConfig:
    """风险模型配置"""
    # EWMA参数
    ewma_halflife_days: int = 63        # 因子半衰期约3个月
    specific_halflife_days: int = 126   # 特异性风险半衰期约6个月
    
    # Ledoit-Wolf收缩参数
    use_ledoit_wolf: bool = True
    min_shrinkage: float = 0.01
    max_shrinkage: float = 0.99
    
    # 数据要求
    min_history_days: int = 252         # 至少需要1年历史数据
    min_stocks_for_factor: int = 20     # 每个因子至少20只股票
    
    # 更新频率
    update_frequency: str = "daily"     # daily/intraday
    refit_frequency_days: int = 21      # 重新拟合频率(约1个月)
    
    # 数值稳定性
    eigenvalue_floor: float = 1e-8     # 特征值下限
    specific_var_floor: float = 1e-6   # 特异性方差下限
    specific_var_cap: float = 1.0      # 特异性方差上限


class FactorRiskModel:
    """
    在线因子风险模型
    
    核心功能:
    1. 滚动估计因子协方差矩阵F (EWMA + Ledoit-Wolf)
    2. 滚动估计特异性方差Ψ² (EWMA)
    3. 提供实时风险模型Σ = ZFZ^T + Ψ
    4. 支持风险模型自检和降维
    """
    
    def __init__(self, config: Optional[RiskModelConfig] = None):
        self.config = config or RiskModelConfig()
        
        # 历史数据缓存
        self._returns_history = deque(maxlen=500)    # 收益率历史
        self._exposures_history = deque(maxlen=500)  # 暴露度历史
        self._factor_returns_history = deque(maxlen=500)  # 因子收益历史
        
        # 当前估计结果
        self._factor_cov = None          # F: 因子协方差矩阵
        self._specific_var = None        # Ψ²: 特异性方差向量
        self._factor_names = None        # 因子名称
        self._stock_names = None         # 股票名称
        
        # 估计状态
        self._last_fit_time = None
        self._is_initialized = False
        self._lock = threading.RLock()
        
        # 性能统计
        self.stats = {
            'fit_count': 0,
            'last_condition_number': 0.0,
            'avg_specific_vol': 0.0,
            'factor_vol_trace': 0.0
        }
        
        logger.info(f"Factor risk model initialized with {self.config.ewma_halflife_days}d factor halflife")
    
    def add_data_point(self, 
                       returns: pd.Series,      # 当期股票收益率
                       exposures: pd.DataFrame, # 当期因子暴露度 (stocks x factors)
                       timestamp: datetime = None):
        """
        添加新的数据点
        
        Args:
            returns: 股票收益率序列 (stock_name -> return)
            exposures: 因子暴露度矩阵 (index=stock_name, columns=factor_name)
            timestamp: 时间戳
        """
        timestamp = timestamp or datetime.now()
        
        with self._lock:
            # 数据对齐：只保留同时有收益率和暴露度的股票
            common_stocks = returns.index.intersection(exposures.index)
            if len(common_stocks) < self.config.min_stocks_for_factor:
                logger.warning(f"Only {len(common_stocks)} stocks have both returns and exposures")
                return
            
            aligned_returns = returns.loc[common_stocks]
            aligned_exposures = exposures.loc[common_stocks]
            
            # 存储数据
            self._returns_history.append({
                'timestamp': timestamp,
                'returns': aligned_returns,
                'exposures': aligned_exposures
            })
            
            # 更新因子收益率 (cross-sectional regression)
            factor_returns = self._estimate_factor_returns(aligned_returns, aligned_exposures)
            if factor_returns is not None:
                self._factor_returns_history.append({
                    'timestamp': timestamp,
                    'factor_returns': factor_returns
                })
            
            # 检查是否需要重新拟合
            if self._should_refit():
                self.fit_model()
    
    def _estimate_factor_returns(self, 
                                returns: pd.Series, 
                                exposures: pd.DataFrame) -> Optional[pd.Series]:
        """通过横截面回归估计因子收益率"""
        try:
            # 加入截距项
            X = exposures.copy()
            X.insert(0, 'intercept', 1.0)
            
            # OLS回归: ret = X * f + epsilon
            factor_returns = linalg.lstsq(X.values, returns.values)[0]
            
            factor_names = ['intercept'] + list(exposures.columns)
            return pd.Series(factor_returns, index=factor_names)
            
        except Exception as e:
            logger.warning(f"Factor returns estimation failed: {e}")
            return None
    
    def _should_refit(self) -> bool:
        """判断是否需要重新拟合模型"""
        if not self._is_initialized:
            return len(self._factor_returns_history) >= self.config.min_history_days // 4
        
        if self._last_fit_time is None:
            return True
        
        days_since_fit = (datetime.now() - self._last_fit_time).days
        return days_since_fit >= self.config.refit_frequency_days
    
    def fit_model(self) -> bool:
        """拟合因子风险模型"""
        with self._lock:
            if len(self._factor_returns_history) < self.config.min_history_days // 4:
                logger.info(f"Insufficient data for fitting: {len(self._factor_returns_history)} points")
                return False
            
            try:
                self._fit_factor_covariance()
                self._fit_specific_variances()
                
                self._is_initialized = True
                self._last_fit_time = datetime.now()
                self.stats['fit_count'] += 1
                
                logger.info(f"Risk model fitted successfully. "
                           f"Factors: {len(self._factor_names)}, "
                           f"Condition number: {self.stats['last_condition_number']:.2f}")
                
                return True
                
            except Exception as e:
                logger.error(f"Risk model fitting failed: {e}")
                return False
    
    def _fit_factor_covariance(self):
        """拟合因子协方差矩阵F"""
        # 提取因子收益率时间序列
        factor_returns_df = pd.DataFrame([
            entry['factor_returns'] for entry in self._factor_returns_history
        ])
        
        # 去除截距项（如果存在）
        if 'intercept' in factor_returns_df.columns:
            factor_returns_df = factor_returns_df.drop('intercept', axis=1)
        
        self._factor_names = list(factor_returns_df.columns)
        
        # EWMA权重
        n_periods = len(factor_returns_df)
        ewma_decay = np.exp(-np.log(2) / self.config.ewma_halflife_days)
        weights = ewma_decay ** np.arange(n_periods-1, -1, -1)
        weights = weights / weights.sum()
        
        # 加权协方差矩阵
        factor_returns_centered = factor_returns_df - factor_returns_df.mean()
        
        if self.config.use_ledoit_wolf:
            # Ledoit-Wolf收缩估计
            sample_cov = np.cov(factor_returns_centered.T, aweights=weights)
            shrinkage_target = np.trace(sample_cov) / len(self._factor_names) * np.eye(len(self._factor_names))
            
            # 简化的收缩系数估计
            shrinkage = min(self.config.max_shrinkage, 
                          max(self.config.min_shrinkage, 
                              1.0 / max(n_periods, 1)))
            
            self._factor_cov = (1 - shrinkage) * sample_cov + shrinkage * shrinkage_target
        else:
            # 纯EWMA
            self._factor_cov = np.cov(factor_returns_centered.T, aweights=weights)
        
        # 数值稳定性处理
        eigenvals, eigenvecs = linalg.eigh(self._factor_cov)
        eigenvals = np.maximum(eigenvals, self.config.eigenvalue_floor)
        self._factor_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        self.stats['last_condition_number'] = eigenvals.max() / eigenvals.min()
        self.stats['factor_vol_trace'] = np.sqrt(np.trace(self._factor_cov))
    
    def _fit_specific_variances(self):
        """拟合特异性方差Ψ²"""
        specific_vars = []
        stock_names = None
        
        for i, entry in enumerate(self._returns_history):
            if i < len(self._factor_returns_history):
                # 计算残差
                returns = entry['returns']
                exposures = entry['exposures']
                
                if stock_names is None:
                    stock_names = returns.index
                elif not returns.index.equals(stock_names):
                    continue  # 跳过股票组合不一致的观测
                
                # 因子收益解释部分
                factor_contrib = exposures @ self._factor_returns_history[i]['factor_returns'].drop('intercept', errors='ignore')
                residuals = returns - factor_contrib
                specific_vars.append(residuals ** 2)
        
        if not specific_vars:
            logger.warning("No valid observations for specific variance estimation")
            return
        
        # 转换为DataFrame
        specific_vars_df = pd.DataFrame(specific_vars, columns=stock_names).fillna(0)
        
        # EWMA估计特异性方差
        n_periods = len(specific_vars_df)
        ewma_decay = np.exp(-np.log(2) / self.config.specific_halflife_days)
        weights = ewma_decay ** np.arange(n_periods-1, -1, -1)
        weights = weights / weights.sum()
        
        # 加权平均
        self._specific_var = (specific_vars_df * weights[:, np.newaxis]).sum(axis=0)
        
        # 应用上下限
        self._specific_var = np.clip(self._specific_var, 
                                   self.config.specific_var_floor,
                                   self.config.specific_var_cap)
        
        self._stock_names = stock_names
        self.stats['avg_specific_vol'] = np.sqrt(self._specific_var.mean())
    
    def get_risk_model(self, 
                      current_exposures: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        获取当前风险模型 Σ = ZFZ^T + Ψ
        
        Args:
            current_exposures: 当前因子暴露度矩阵
            
        Returns:
            Σ: 风险协方差矩阵
            stock_names: 股票名称列表
            factor_names: 因子名称列表
        """
        with self._lock:
            if not self._is_initialized:
                raise ValueError("Risk model not initialized. Please fit model first.")
            
            # 对齐因子
            common_factors = [f for f in self._factor_names if f in current_exposures.columns]
            if len(common_factors) == 0:
                raise ValueError("No common factors between model and current exposures")
            
            Z = current_exposures[common_factors].values
            F = self._factor_cov
            
            # 对齐股票（使用当前暴露度的股票，特异性方差填充默认值）
            current_stocks = current_exposures.index
            specific_var = np.full(len(current_stocks), self.config.specific_var_floor * 4)  # 默认值
            
            for i, stock in enumerate(current_stocks):
                if stock in self._stock_names:
                    idx = list(self._stock_names).index(stock)
                    specific_var[i] = self._specific_var.iloc[idx]
            
            # 构建风险模型
            Sigma = Z @ F @ Z.T + np.diag(specific_var)
            
            return Sigma, list(current_stocks), common_factors
    
    def get_factor_covariance(self) -> Optional[Tuple[np.ndarray, List[str]]]:
        """获取因子协方差矩阵"""
        if not self._is_initialized:
            return None
        return self._factor_cov.copy(), self._factor_names.copy()
    
    def get_specific_variances(self) -> Optional[Tuple[pd.Series, List[str]]]:
        """获取特异性方差"""
        if not self._is_initialized:
            return None
        return self._specific_var.copy(), list(self._stock_names)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        with self._lock:
            stats = self.stats.copy()
            stats.update({
                'is_initialized': self._is_initialized,
                'n_factors': len(self._factor_names) if self._factor_names else 0,
                'n_stocks': len(self._stock_names) if self._stock_names else 0,
                'history_length': len(self._factor_returns_history),
                'last_fit_time': self._last_fit_time.isoformat() if self._last_fit_time else None
            })
            return stats
    
    def validate_model(self) -> Dict[str, bool]:
        """验证模型有效性"""
        validation = {
            'is_initialized': self._is_initialized,
            'has_factor_cov': self._factor_cov is not None,
            'has_specific_var': self._specific_var is not None,
            'factor_cov_psd': False,
            'specific_var_positive': False,
            'reasonable_condition_number': False
        }
        
        if self._factor_cov is not None:
            try:
                eigenvals = linalg.eigvals(self._factor_cov)
                validation['factor_cov_psd'] = np.all(eigenvals > -1e-8)
                validation['reasonable_condition_number'] = (eigenvals.max() / eigenvals.min()) < 1e6
            except:
                pass
        
        if self._specific_var is not None:
            validation['specific_var_positive'] = np.all(self._specific_var > 0)
        
        return validation


# 全局实例
_global_risk_model: Optional[FactorRiskModel] = None


def get_factor_risk_model(config: Optional[RiskModelConfig] = None) -> FactorRiskModel:
    """获取全局因子风险模型实例"""
    global _global_risk_model
    if _global_risk_model is None:
        _global_risk_model = FactorRiskModel(config)
    return _global_risk_model


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟数据
    n_stocks = 100
    n_factors = 8
    n_periods = 300
    
    # np.random.seed removed
    factor_names = [f'factor_{i}' for i in range(n_factors)]
    stock_names = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    
    # 模拟因子暴露度（较稳定）
    base_exposures = np.zeros(n_stocks) * 0.5
    
    risk_model = FactorRiskModel()
    
    print("Simulating risk model with synthetic data...")
    for t in range(n_periods):
        # 生成因子收益率
        factor_returns = np.random.multivariate_normal(
            np.zeros(n_factors), 
            np.eye(n_factors) * 0.01  # 1%因子波动率
        )
        
        # 生成特异性收益率
        specific_returns = np.zeros(n_stocks) * 0.02  # 2%特异性波动率
        
        # 当期暴露度（微小扰动）
        current_exposures = base_exposures + np.zeros(n_stocks) * 0.02
        
        # 总收益率 = 因子收益 + 特异性收益
        total_returns = current_exposures @ factor_returns + specific_returns
        
        # 添加数据点
        returns_series = pd.Series(total_returns, index=stock_names)
        exposures_df = pd.DataFrame(current_exposures, 
                                  index=stock_names, 
                                  columns=factor_names)
        
        risk_model.add_data_point(returns_series, exposures_df)
        
        if t % 50 == 49:
            print(f"Added {t+1} periods, model stats: {risk_model.get_model_stats()}")
    
    # 测试风险模型
    if risk_model._is_initialized:
        test_exposures = pd.DataFrame(base_exposures[-20:], 
                                    index=stock_names[-20:], 
                                    columns=factor_names)
        
        Sigma, stocks, factors = risk_model.get_risk_model(test_exposures)
        print(f"Risk matrix shape: {Sigma.shape}")
        print(f"Risk matrix condition number: {np.linalg.cond(Sigma):.2f}")
        print(f"Average volatility: {np.sqrt(np.diag(Sigma).mean()):.4f}")
        
        validation = risk_model.validate_model()
        print(f"Model validation: {validation}")
    
    print("Risk model test completed.")