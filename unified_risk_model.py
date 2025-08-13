#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一风险模型系统
贯通Alpha策略、Professional引擎和Ultra Enhanced的风险模型与中性化数据口径
"""

import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from scipy import stats
from scipy.linalg import cholesky, inv
import sqlite3

from unified_market_data_manager import UnifiedMarketDataManager, MarketDataConfig

logger = logging.getLogger(__name__)

@dataclass
class RiskModelConfig:
    """风险模型配置"""
    # 因子模型设置
    use_fundamental_factors: bool = True
    use_style_factors: bool = True
    use_industry_factors: bool = True
    use_country_factors: bool = True
    
    # 行业分类设置
    sector_classification: str = "GICS"  # 与市场数据管理器保持一致
    sector_level: int = 4  # GICS 4级分类
    
    # 市值类型选择
    market_cap_type: str = "free_float_market_cap"  # market_cap, float_market_cap, free_float_market_cap
    
    # 风险建模参数
    lookback_window: int = 252  # 1年
    half_life: int = 63        # 约3个月半衰期
    min_observations: int = 126 # 最小观测数
    
    # 中性化设置
    neutralization_method: str = "cross_sectional"  # cross_sectional, time_series
    industry_neutral: bool = True
    size_neutral: bool = True
    country_neutral: bool = False  # 单一市场可关闭
    
    # 协方差估计
    covariance_method: str = "ledoit_wolf"  # sample, ledoit_wolf, factor_model
    shrinkage_intensity: Optional[float] = None
    
    # 异常值处理
    winsorize_quantiles: Tuple[float, float] = (0.01, 0.99)
    outlier_method: str = "mad"  # mad, z_score, iqr

@dataclass
class RiskFactors:
    """风险因子定义"""
    # 基本因子
    size: pd.Series
    value: pd.Series  
    momentum: pd.Series
    profitability: pd.Series
    investment: pd.Series
    volatility: pd.Series
    liquidity: pd.Series
    
    # 行业因子
    industry_exposures: pd.DataFrame
    
    # 国家因子
    country_exposures: pd.DataFrame
    
    # 自定义因子
    custom_factors: Dict[str, pd.Series] = field(default_factory=dict)

class FactorExposureCalculator:
    """因子暴露计算器"""
    
    def __init__(self, config: RiskModelConfig):
        self.config = config
        self.label_encoders = {}
        
    def calculate_style_factors(self, 
                               data: pd.DataFrame,
                               market_data_manager: UnifiedMarketDataManager) -> Dict[str, pd.Series]:
        """计算风格因子暴露"""
        
        # 确保数据包含必要的市场信息
        if 'market_cap' not in data.columns:
            logger.info("数据中缺少市场数据，正在补充...")
            data = market_data_manager.create_unified_features_dataframe(data)
        
        factors = {}
        
        # 1. Size因子 (基于选定的市值类型)
        market_cap_col = self.config.market_cap_type
        if market_cap_col in data.columns:
            factors['size'] = np.log(data[market_cap_col].fillna(data[market_cap_col].median()))
        else:
            # 降级到可用的市值类型
            for fallback_col in ['market_cap', 'float_market_cap', 'free_float_market_cap']:
                if fallback_col in data.columns:
                    factors['size'] = np.log(data[fallback_col].fillna(data[fallback_col].median()))
                    logger.warning(f"使用{fallback_col}替代{market_cap_col}计算Size因子")
                    break
        
        # 2. Value因子 (基于基本面数据)
        if 'pb_ratio' in data.columns:
            factors['value'] = -np.log(data['pb_ratio'].fillna(data['pb_ratio'].median()).clip(0.1, 50))
        elif 'pe_ratio' in data.columns:
            factors['value'] = -np.log(data['pe_ratio'].fillna(data['pe_ratio'].median()).clip(1, 100))
        else:
            # 使用价格动量作为Value的代理
            if 'close' in data.columns:
                returns_252d = data.groupby('ticker')['close'].pct_change(252)
                factors['value'] = -returns_252d.fillna(0)  # 低回报率 = 高价值
        
        # 3. Momentum因子
        if 'close' in data.columns:
            # 12-1动量：过去12个月去除最近1个月
            returns_252d = data.groupby('ticker')['close'].pct_change(252)
            returns_21d = data.groupby('ticker')['close'].pct_change(21)
            factors['momentum'] = (returns_252d - returns_21d).fillna(0)
        
        # 4. Profitability因子
        if 'roe' in data.columns:
            factors['profitability'] = data['roe'].fillna(data['roe'].median())
        elif 'roa' in data.columns:
            factors['profitability'] = data['roa'].fillna(data['roa'].median())
        else:
            # 使用近期收益率作为盈利能力代理
            if 'close' in data.columns:
                returns_63d = data.groupby('ticker')['close'].pct_change(63)
                factors['profitability'] = returns_63d.fillna(0)
        
        # 5. Investment因子
        if 'asset_growth' in data.columns:
            factors['investment'] = -data['asset_growth'].fillna(0)  # 低投资 = 高因子值
        else:
            # 使用成交量增长作为投资活动代理
            if 'volume' in data.columns:
                volume_growth = data.groupby('ticker')['volume'].pct_change(63)
                factors['investment'] = -volume_growth.fillna(0)
        
        # 6. Volatility因子
        if 'close' in data.columns:
            returns = data.groupby('ticker')['close'].pct_change()
            volatility = returns.groupby(data['ticker']).rolling(63).std().reset_index(level=0, drop=True)
            factors['volatility'] = volatility.fillna(volatility.median())
        
        # 7. Liquidity因子
        if 'volume' in data.columns and 'close' in data.columns:
            # Amihud非流动性指标的负值
            returns_abs = data.groupby('ticker')['close'].pct_change().abs()
            dollar_volume = data['volume'] * data['close']
            illiquidity = returns_abs / (dollar_volume + 1e-8)
            factors['liquidity'] = -illiquidity.rolling(21).mean().fillna(0)
        
        # 标准化所有因子
        for factor_name, factor_values in factors.items():
            factors[factor_name] = self._winsorize_and_standardize(factor_values)
        
        logger.info(f"计算完成{len(factors)}个风格因子")
        return factors
    
    def calculate_industry_exposures(self, 
                                   data: pd.DataFrame,
                                   market_data_manager: UnifiedMarketDataManager) -> pd.DataFrame:
        """计算行业暴露"""
        
        # 确保数据包含行业信息
        if 'gics_sector' not in data.columns and 'sector' not in data.columns:
            data = market_data_manager.create_unified_features_dataframe(data)
        
        # 选择行业分类列
        industry_col = None
        if self.config.sector_level == 1 and 'gics_sector' in data.columns:
            industry_col = 'gics_sector'
        elif self.config.sector_level >= 2 and 'gics_industry' in data.columns:
            industry_col = 'gics_industry'
        elif 'sector' in data.columns:
            industry_col = 'sector'
        
        if industry_col is None:
            logger.warning("无法找到行业分类数据，跳过行业因子")
            return pd.DataFrame(index=data.index)
        
        # 创建one-hot编码
        industry_dummies = pd.get_dummies(data[industry_col], prefix='industry')
        
        # 去除第一个行业作为基准（避免共线性）
        if len(industry_dummies.columns) > 1:
            industry_dummies = industry_dummies.iloc[:, 1:]
        
        logger.info(f"创建{len(industry_dummies.columns)}个行业因子")
        return industry_dummies
    
    def calculate_country_exposures(self, 
                                  data: pd.DataFrame,
                                  market_data_manager: UnifiedMarketDataManager) -> pd.DataFrame:
        """计算国家暴露"""
        
        if not self.config.use_country_factors:
            return pd.DataFrame(index=data.index)
        
        # 确保数据包含国家信息
        if 'country' not in data.columns:
            data = market_data_manager.create_unified_features_dataframe(data)
        
        if 'country' not in data.columns:
            logger.warning("无法找到国家数据，跳过国家因子")
            return pd.DataFrame(index=data.index)
        
        # 创建国家哑变量
        country_dummies = pd.get_dummies(data['country'], prefix='country')
        
        # 如果只有一个国家，返回空DataFrame
        if len(country_dummies.columns) <= 1:
            logger.info("单一国家市场，跳过国家因子")
            return pd.DataFrame(index=data.index)
        
        # 去除第一个国家作为基准
        country_dummies = country_dummies.iloc[:, 1:]
        
        logger.info(f"创建{len(country_dummies.columns)}个国家因子")
        return country_dummies
    
    def _winsorize_and_standardize(self, series: pd.Series) -> pd.Series:
        """缩尾和标准化处理"""
        
        # 缩尾处理
        lower_q, upper_q = self.config.winsorize_quantiles
        lower_bound = series.quantile(lower_q)
        upper_bound = series.quantile(upper_q)
        winsorized = series.clip(lower_bound, upper_bound)
        
        # 标准化
        standardized = (winsorized - winsorized.mean()) / (winsorized.std() + 1e-8)
        
        return standardized.fillna(0)

class RiskModelEngine:
    """统一风险模型引擎"""
    
    def __init__(self, 
                 config: RiskModelConfig = None,
                 market_data_manager: UnifiedMarketDataManager = None):
        
        self.config = config or RiskModelConfig()
        self.market_data_manager = market_data_manager or UnifiedMarketDataManager()
        self.exposure_calculator = FactorExposureCalculator(self.config)
        
        # 模型状态
        self.factor_exposures: Optional[pd.DataFrame] = None
        self.factor_returns: Optional[pd.DataFrame] = None
        self.specific_returns: Optional[pd.Series] = None
        self.factor_covariance: Optional[pd.DataFrame] = None
        self.specific_variance: Optional[pd.Series] = None
        
    def fit_risk_model(self, 
                      data: pd.DataFrame,
                      returns_column: str = 'return') -> 'RiskModelEngine':
        """拟合风险模型"""
        
        logger.info("开始拟合风险模型...")
        
        # 1. 计算因子暴露
        self.factor_exposures = self._build_factor_exposures(data)
        
        # 2. 回归得到因子收益率
        self.factor_returns, self.specific_returns = self._estimate_factor_returns(
            data, returns_column
        )
        
        # 3. 估计因子协方差矩阵
        self.factor_covariance = self._estimate_factor_covariance()
        
        # 4. 估计特异性风险
        self.specific_variance = self._estimate_specific_variance()
        
        logger.info("风险模型拟合完成")
        return self
    
    def _build_factor_exposures(self, data: pd.DataFrame) -> pd.DataFrame:
        """构建因子暴露矩阵"""
        
        exposures = []
        
        # 1. 风格因子
        if self.config.use_style_factors:
            style_factors = self.exposure_calculator.calculate_style_factors(
                data, self.market_data_manager
            )
            for factor_name, factor_values in style_factors.items():
                exposures.append(factor_values.rename(factor_name))
        
        # 2. 行业因子
        if self.config.use_industry_factors:
            industry_exposures = self.exposure_calculator.calculate_industry_exposures(
                data, self.market_data_manager
            )
            exposures.extend([industry_exposures[col] for col in industry_exposures.columns])
        
        # 3. 国家因子
        if self.config.use_country_factors:
            country_exposures = self.exposure_calculator.calculate_country_exposures(
                data, self.market_data_manager
            )
            exposures.extend([country_exposures[col] for col in country_exposures.columns])
        
        # 合并所有因子
        if exposures:
            factor_matrix = pd.concat(exposures, axis=1)
            factor_matrix = factor_matrix.fillna(0)
        else:
            factor_matrix = pd.DataFrame(index=data.index)
        
        logger.info(f"构建因子暴露矩阵: {factor_matrix.shape}")
        return factor_matrix
    
    def _estimate_factor_returns(self, 
                               data: pd.DataFrame,
                               returns_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """估计因子收益率"""
        
        returns = data[returns_column]
        
        if self.factor_exposures.empty:
            # 如果没有因子，返回市场收益率
            market_return = returns.mean()
            factor_returns = pd.DataFrame({'market': [market_return]})
            specific_returns = returns - market_return
            return factor_returns, specific_returns
        
        # 时间序列回归：每个时期(date)进行横截面回归
        dates = data['date'].unique() if 'date' in data.columns else [data.index.name or 'all']
        
        factor_returns_list = []
        specific_returns_list = []
        
        for date in dates:
            if 'date' in data.columns:
                date_mask = data['date'] == date
                date_returns = returns[date_mask]
                date_exposures = self.factor_exposures.loc[date_mask]
            else:
                date_returns = returns
                date_exposures = self.factor_exposures
            
            if len(date_returns) < 2 or date_exposures.empty:
                continue
            
            # 横截面回归: r_i = X_i * f + epsilon_i
            try:
                # 使用加权最小二乘，权重为市值的平方根
                weights = None
                if 'market_cap' in data.columns:
                    if 'date' in data.columns:
                        weights = np.sqrt(data.loc[date_mask, 'market_cap'].fillna(data['market_cap'].median()))
                    else:
                        weights = np.sqrt(data['market_cap'].fillna(data['market_cap'].median()))
                
                # 回归
                if weights is not None:
                    # 加权回归
                    W = np.diag(weights.values if hasattr(weights, 'values') else weights)
                    X_weighted = date_exposures.values.T @ W
                    y_weighted = W @ date_returns.values
                    
                    beta = np.linalg.lstsq(X_weighted.T, y_weighted, rcond=None)[0]
                else:
                    # 普通最小二乘
                    beta = np.linalg.lstsq(date_exposures.values, date_returns.values, rcond=None)[0]
                
                # 计算残差
                predicted_returns = date_exposures @ beta
                residuals = date_returns - predicted_returns
                
                # 存储结果
                factor_return = pd.Series(beta, index=date_exposures.columns, name=date)
                factor_returns_list.append(factor_return)
                specific_returns_list.extend(residuals.tolist())
                
            except np.linalg.LinAlgError:
                logger.warning(f"日期{date}的回归失败，跳过")
                continue
        
        # 合并结果
        if factor_returns_list:
            factor_returns = pd.DataFrame(factor_returns_list)
        else:
            factor_returns = pd.DataFrame(columns=self.factor_exposures.columns)
        
        specific_returns = pd.Series(specific_returns_list, index=returns.index[:len(specific_returns_list)])
        
        return factor_returns, specific_returns
    
    def _estimate_factor_covariance(self) -> pd.DataFrame:
        """估计因子协方差矩阵"""
        
        if self.factor_returns.empty:
            return pd.DataFrame()
        
        # 应用指数衰减权重
        weights = self._get_exponential_weights(len(self.factor_returns))
        
        # 计算加权协方差
        if self.config.covariance_method == "ledoit_wolf":
            # Ledoit-Wolf收缩估计
            cov_estimator = LedoitWolf()
            factor_cov = cov_estimator.fit(self.factor_returns.fillna(0)).covariance_
            factor_cov = pd.DataFrame(factor_cov, 
                                    index=self.factor_returns.columns,
                                    columns=self.factor_returns.columns)
        else:
            # 样本协方差（带指数权重）
            centered_returns = self.factor_returns - self.factor_returns.mean()
            weighted_cov = np.cov(centered_returns.T, aweights=weights)
            factor_cov = pd.DataFrame(weighted_cov,
                                    index=self.factor_returns.columns, 
                                    columns=self.factor_returns.columns)
        
        return factor_cov
    
    def _estimate_specific_variance(self) -> pd.Series:
        """估计特异性方差"""
        
        if self.specific_returns.empty:
            return pd.Series()
        
        # 按股票分组计算特异性方差
        specific_var = self.specific_returns.groupby(level=0).var() if isinstance(self.specific_returns.index, pd.MultiIndex) else pd.Series([self.specific_returns.var()])
        
        # 异常值处理
        specific_var = specific_var.clip(lower=specific_var.quantile(0.05), 
                                       upper=specific_var.quantile(0.95))
        
        return specific_var
    
    def _get_exponential_weights(self, n: int) -> np.ndarray:
        """生成指数衰减权重"""
        
        alpha = 1 - np.exp(-np.log(2) / self.config.half_life)
        weights = np.array([(1-alpha)**i for i in range(n)])
        weights = weights[::-1]  # 最新数据权重最大
        weights = weights / weights.sum()  # 标准化
        
        return weights
    
    def apply_neutralization(self, 
                           signals: pd.Series,
                           data: pd.DataFrame,
                           method: str = None) -> pd.Series:
        """应用中性化"""
        
        method = method or self.config.neutralization_method
        
        if method == "cross_sectional":
            return self._cross_sectional_neutralization(signals, data)
        elif method == "time_series":
            return self._time_series_neutralization(signals, data)
        else:
            raise ValueError(f"未知的中性化方法: {method}")
    
    def _cross_sectional_neutralization(self, 
                                      signals: pd.Series,
                                      data: pd.DataFrame) -> pd.Series:
        """横截面中性化"""
        
        neutralized_signals = signals.copy()
        
        # 按日期分组进行中性化
        if 'date' in data.columns:
            dates = data['date'].unique()
            
            for date in dates:
                date_mask = data['date'] == date
                date_signals = signals[date_mask]
                date_data = data[date_mask]
                
                if len(date_signals) < 2:
                    continue
                
                # 构建中性化矩阵
                neutralization_factors = []
                
                # 市值中性化
                if self.config.size_neutral and 'market_cap' in date_data.columns:
                    size_factor = np.log(date_data['market_cap'].fillna(date_data['market_cap'].median()))
                    neutralization_factors.append(size_factor)
                
                # 行业中性化
                if self.config.industry_neutral:
                    if 'gics_sector' in date_data.columns:
                        industry_dummies = pd.get_dummies(date_data['gics_sector'])
                        neutralization_factors.extend([industry_dummies[col] for col in industry_dummies.columns[:-1]])
                    elif 'sector' in date_data.columns:
                        industry_dummies = pd.get_dummies(date_data['sector'])
                        neutralization_factors.extend([industry_dummies[col] for col in industry_dummies.columns[:-1]])
                
                # 国家中性化
                if self.config.country_neutral and 'country' in date_data.columns:
                    country_dummies = pd.get_dummies(date_data['country'])
                    if len(country_dummies.columns) > 1:
                        neutralization_factors.extend([country_dummies[col] for col in country_dummies.columns[:-1]])
                
                # 执行中性化回归
                if neutralization_factors:
                    X = pd.concat(neutralization_factors, axis=1).fillna(0)
                    
                    try:
                        # 回归: signal = X * beta + residual
                        beta = np.linalg.lstsq(X, date_signals, rcond=None)[0]
                        residuals = date_signals - X @ beta
                        
                        # 标准化残差
                        residuals = (residuals - residuals.mean()) / (residuals.std() + 1e-8)
                        
                        neutralized_signals[date_mask] = residuals
                        
                    except np.linalg.LinAlgError:
                        logger.warning(f"日期{date}的中性化失败")
                        continue
        
        return neutralized_signals
    
    def _time_series_neutralization(self, 
                                  signals: pd.Series,
                                  data: pd.DataFrame) -> pd.Series:
        """时间序列中性化（去除趋势）"""
        
        # 按股票分组去趋势
        if 'ticker' in data.columns:
            neutralized = signals.groupby(data['ticker']).apply(
                lambda x: x - x.rolling(window=21, min_periods=1).mean()
            )
            return neutralized.reset_index(level=0, drop=True)
        else:
            # 整体去趋势
            return signals - signals.rolling(window=21, min_periods=1).mean()
    
    def calculate_portfolio_risk(self, 
                               weights: pd.Series,
                               data: pd.DataFrame) -> Dict[str, float]:
        """计算组合风险"""
        
        if self.factor_exposures is None or self.factor_covariance.empty:
            logger.warning("风险模型未拟合，无法计算组合风险")
            return {'total_risk': np.nan}
        
        # 权重对齐
        aligned_weights = weights.reindex(self.factor_exposures.index, fill_value=0)
        
        # 组合因子暴露
        portfolio_exposures = self.factor_exposures.T @ aligned_weights
        
        # 因子风险
        factor_risk = portfolio_exposures.T @ self.factor_covariance @ portfolio_exposures
        
        # 特异性风险
        if not self.specific_variance.empty:
            specific_risk = (aligned_weights**2 @ self.specific_variance.reindex(aligned_weights.index, fill_value=self.specific_variance.median()))
        else:
            specific_risk = 0
        
        # 总风险
        total_risk = factor_risk + specific_risk
        
        return {
            'total_risk': float(np.sqrt(total_risk * 252)),  # 年化
            'factor_risk': float(np.sqrt(factor_risk * 252)),
            'specific_risk': float(np.sqrt(specific_risk * 252)),
            'risk_ratio': float(factor_risk / (factor_risk + specific_risk + 1e-8))
        }

# 集成到Alpha策略引擎
class EnhancedAlphaEngine:
    """增强的Alpha引擎，集成统一风险模型"""
    
    def __init__(self, 
                 alpha_engine,  # 原有的AlphaStrategiesEngine
                 risk_model_config: RiskModelConfig = None):
        
        self.alpha_engine = alpha_engine
        self.risk_model = RiskModelEngine(risk_model_config)
        self.market_data_manager = self.risk_model.market_data_manager
        
    def enhanced_alpha_computation(self, 
                                 data: pd.DataFrame,
                                 alpha_names: List[str] = None) -> pd.DataFrame:
        """增强的Alpha计算，使用统一的市场数据"""
        
        # 1. 补充统一的市场数据
        enhanced_data = self.market_data_manager.create_unified_features_dataframe(data)
        
        # 2. 计算Alpha因子
        alpha_names = alpha_names or list(self.alpha_engine.alpha_functions.keys())
        alpha_df = self.alpha_engine.compute_all_alphas(enhanced_data)
        
        # 3. 应用风险模型中性化
        if 'return' in enhanced_data.columns:
            # 拟合风险模型
            self.risk_model.fit_risk_model(enhanced_data)
            
            # 对每个Alpha因子进行中性化
            for alpha_name in alpha_df.columns:
                alpha_df[alpha_name] = self.risk_model.apply_neutralization(
                    alpha_df[alpha_name], enhanced_data
                )
        
        return alpha_df
    
    def risk_adjusted_portfolio_optimization(self, 
                                           alpha_signals: pd.Series,
                                           data: pd.DataFrame,
                                           target_risk: float = 0.15) -> pd.Series:
        """风险调整的组合优化"""
        
        # 计算初始等权重
        n_assets = len(alpha_signals)
        initial_weights = pd.Series(1.0/n_assets, index=alpha_signals.index)
        
        # 应用Alpha信号调整
        signal_strength = np.abs(alpha_signals)
        adjusted_weights = initial_weights * (1 + alpha_signals)
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        # 风险约束
        risk_metrics = self.risk_model.calculate_portfolio_risk(adjusted_weights, data)
        current_risk = risk_metrics.get('total_risk', 0.2)
        
        # 如果风险过高，进行风险缩放
        if current_risk > target_risk:
            risk_scale = target_risk / current_risk
            adjusted_weights = adjusted_weights * risk_scale + initial_weights * (1 - risk_scale)
        
        # 行业中性约束
        final_weights = self.market_data_manager.get_sector_neutral_weights(
            adjusted_weights.index.tolist(), adjusted_weights
        )
        
        return final_weights

# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'BRK-B', 'UNH', 'JNJ']
    
    data = []
    for date in dates:
        for ticker in tickers:
            data.append({
                'date': date,
                'ticker': ticker,
                'close': 100 * (1 + np.random.normal(0, 0.02)),
                'volume': np.random.uniform(1e6, 1e8),
                'return': np.random.normal(0.001, 0.02)
            })
    
    df = pd.DataFrame(data)
    
    # 测试统一风险模型
    config = RiskModelConfig()
    risk_model = RiskModelEngine(config)
    
    # 拟合模型
    risk_model.fit_risk_model(df)
    
    print(f"因子暴露矩阵形状: {risk_model.factor_exposures.shape if risk_model.factor_exposures is not None else None}")
    print(f"因子协方差矩阵形状: {risk_model.factor_covariance.shape}")
    
    # 测试中性化
    test_signals = pd.Series(np.random.normal(0, 1, len(df)), index=df.index)
    neutralized_signals = risk_model.apply_neutralization(test_signals, df)
    
    print(f"原始信号均值: {test_signals.mean():.4f}, 标准差: {test_signals.std():.4f}")
    print(f"中性化后均值: {neutralized_signals.mean():.4f}, 标准差: {neutralized_signals.std():.4f}")
    
    # 测试组合风险
    weights = pd.Series(0.1, index=df[df['date'] == dates[0]].index)
    risk_metrics = risk_model.calculate_portfolio_risk(weights, df[df['date'] == dates[0]])
    print(f"组合风险指标: {risk_metrics}")
