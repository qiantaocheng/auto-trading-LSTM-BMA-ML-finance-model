#!/usr/bin/env python3
"""
稳健的Beta计算器
解决Beta计算中的数值不稳定问题
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Tuple
from scipy.stats import trim_mean
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class RobustBetaCalculator:
    """稳健的Beta计算器"""
    
    def __init__(self, 
                 window_size: int = 252,
                 min_samples: int = 30,
                 trim_percent: float = 0.1,
                 use_robust_regression: bool = True,
                 market_cap_weighted: bool = True):
        
        self.window_size = window_size
        self.min_samples = max(min_samples, 10)  # 确保最小样本数
        self.trim_percent = trim_percent
        self.use_robust_regression = use_robust_regression
        self.market_cap_weighted = market_cap_weighted
        
        logger.info(f"初始化稳健Beta计算器 - 窗口: {window_size}天, "
                   f"最小样本: {self.min_samples}, 稳健回归: {use_robust_regression}")
    
    def calculate_market_returns(self, returns_matrix: pd.DataFrame, 
                                market_caps: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        计算稳健的市场收益率
        
        Args:
            returns_matrix: 股票收益率矩阵 (date x ticker)
            market_caps: 市值数据 (可选)
            
        Returns:
            市场收益率序列
        """
        try:
            if self.market_cap_weighted and market_caps is not None:
                # 市值加权市场收益
                logger.debug("使用市值加权计算市场收益")
                market_returns = self._calculate_cap_weighted_returns(returns_matrix, market_caps)
            else:
                # 使用截尾均值替代简单均值或中位数
                logger.debug(f"使用截尾均值(trim={self.trim_percent})计算市场收益")
                market_returns = returns_matrix.apply(
                    lambda row: trim_mean(row.dropna(), self.trim_percent) 
                    if len(row.dropna()) >= 3 else np.nan,
                    axis=1
                )
            
            # 数据质量检查
            valid_returns = market_returns.dropna()
            if len(valid_returns) < self.min_samples:
                logger.warning(f"市场收益数据不足: {len(valid_returns)} < {self.min_samples}")
                # 降级到简单均值
                market_returns = returns_matrix.mean(axis=1)
            
            return market_returns
            
        except Exception as e:
            logger.error(f"市场收益计算失败: {e}")
            # 备用方案：简单均值
            return returns_matrix.mean(axis=1)
    
    def _calculate_cap_weighted_returns(self, returns_matrix: pd.DataFrame, 
                                      market_caps: pd.DataFrame) -> pd.Series:
        """计算市值加权收益"""
        aligned_returns = returns_matrix.copy()
        aligned_caps = market_caps.reindex(returns_matrix.index, method='ffill')
        
        # 对齐股票代码
        common_tickers = aligned_returns.columns.intersection(aligned_caps.columns)
        if len(common_tickers) < 3:
            logger.warning("市值数据覆盖不足，降级到等权重")
            return returns_matrix.mean(axis=1)
        
        aligned_returns = aligned_returns[common_tickers]
        aligned_caps = aligned_caps[common_tickers]
        
        # 计算权重 (处理零值和NaN)
        weights = aligned_caps.div(aligned_caps.sum(axis=1), axis=0)
        weights = weights.fillna(0)
        
        # 加权收益
        weighted_returns = (aligned_returns * weights).sum(axis=1)
        
        return weighted_returns
    
    def calculate_beta_series(self, returns_matrix: pd.DataFrame,
                            market_caps: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        计算稳健的Beta因子序列
        
        Args:
            returns_matrix: 股票收益率矩阵
            market_caps: 市值数据(可选)
            
        Returns:
            Beta因子序列
        """
        logger.info("开始计算稳健Beta因子序列")
        
        # 计算市场收益
        market_returns = self.calculate_market_returns(returns_matrix, market_caps)
        
        betas = []
        dates = returns_matrix.index
        
        for i, date in enumerate(dates):
            try:
                # 确定窗口范围
                end_idx = i + 1
                start_idx = max(0, end_idx - self.window_size)
                
                if end_idx - start_idx < self.min_samples:
                    betas.append(1.0)  # 默认市场Beta
                    continue
                
                # 获取窗口数据
                window_returns = returns_matrix.iloc[start_idx:end_idx]
                window_market = market_returns.iloc[start_idx:end_idx]
                
                # 计算窗口内的平均Beta
                window_beta = self._calculate_window_beta(window_returns, window_market)
                betas.append(window_beta)
                
                if i % 50 == 0:  # 定期日志
                    logger.debug(f"计算进度: {i+1}/{len(dates)}, 当前Beta: {window_beta:.3f}")
                    
            except Exception as e:
                logger.warning(f"日期 {date} Beta计算异常: {e}")
                betas.append(1.0)  # 默认值
        
        result = pd.Series(betas, index=dates, name='robust_beta')
        
        # 结果验证
        valid_betas = result.dropna()
        logger.info(f"Beta计算完成 - 有效值: {len(valid_betas)}/{len(result)}, "
                   f"范围: [{valid_betas.min():.3f}, {valid_betas.max():.3f}]")
        
        return result
    
    def _calculate_window_beta(self, window_returns: pd.DataFrame, 
                             window_market: pd.Series) -> float:
        """计算窗口内的稳健Beta"""
        stock_betas = []
        
        for ticker in window_returns.columns:
            try:
                stock_returns = window_returns[ticker].dropna()
                
                # 对齐数据
                common_idx = stock_returns.index.intersection(window_market.index)
                if len(common_idx) < self.min_samples // 2:
                    continue
                
                stock_ret = stock_returns.loc[common_idx]
                market_ret = window_market.loc[common_idx]
                
                # 移除极端值
                stock_ret, market_ret = self._remove_outliers(stock_ret, market_ret)
                
                if len(stock_ret) < max(5, self.min_samples // 4):
                    continue
                
                # 计算Beta
                beta = self._robust_beta_regression(stock_ret, market_ret)
                
                # Beta合理性检查
                if 0.1 <= beta <= 5.0:  # 合理范围
                    stock_betas.append(beta)
                    
            except Exception as e:
                logger.debug(f"股票 {ticker} Beta计算跳过: {e}")
                continue
        
        if not stock_betas:
            return 1.0
        
        # 返回稳健的平均Beta
        return np.median(stock_betas) if stock_betas else 1.0
    
    def _remove_outliers(self, stock_returns: pd.Series, 
                        market_returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """移除极端值"""
        # 使用IQR方法移除极端值
        def remove_outliers_iqr(series, factor=1.5):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            return series[(series >= lower) & (series <= upper)]
        
        # 分别处理股票和市场收益
        clean_stock = remove_outliers_iqr(stock_returns)
        clean_market = remove_outliers_iqr(market_returns)
        
        # 保持索引对齐
        common_idx = clean_stock.index.intersection(clean_market.index)
        
        return clean_stock.loc[common_idx], clean_market.loc[common_idx]
    
    def _robust_beta_regression(self, stock_returns: pd.Series, 
                              market_returns: pd.Series) -> float:
        """稳健Beta回归计算"""
        if not self.use_robust_regression:
            # OLS回归
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            return covariance / market_variance if market_variance > 1e-8 else 1.0
        
        try:
            # 准备回归数据
            X = market_returns.values.reshape(-1, 1)
            y = stock_returns.values
            
            # 标准化数据以提高数值稳定性
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
            
            # Huber回归 (稳健回归)
            huber = HuberRegressor(epsilon=1.35, alpha=0.01)
            huber.fit(X_scaled, y_scaled)
            
            # 还原到原始尺度
            beta_scaled = huber.coef_[0]
            beta = beta_scaled * (scaler_y.scale_[0] / scaler_X.scale_[0])
            
            return float(np.clip(beta, 0.1, 5.0))  # 限制在合理范围
            
        except Exception as e:
            logger.debug(f"稳健回归失败，降级到OLS: {e}")
            # 降级到简单线性回归
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            return covariance / market_variance if market_variance > 1e-8 else 1.0

def create_robust_beta_calculator(**kwargs) -> RobustBetaCalculator:
    """工厂函数"""
    return RobustBetaCalculator(**kwargs)