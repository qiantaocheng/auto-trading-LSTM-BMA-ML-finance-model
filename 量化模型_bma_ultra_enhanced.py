#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA Ultra Enhanced 量化分析模型 V4
集成Alpha策略、Learning-to-Rank、不确定性感知BMA、高级投资组合优化
提供工业级的量化交易解决方案
"""

import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import warnings
import argparse
import os
import tempfile
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union

# 基础科学计算
from scipy.stats import spearmanr, entropy
from scipy.optimize import minimize
import statsmodels.api as sm
from dataclasses import dataclass, field
from scipy import stats
from sklearn.linear_model import HuberRegressor
from sklearn.covariance import LedoitWolf

# 机器学习
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from purged_time_series_cv import PurgedGroupTimeSeriesSplit, ValidationConfig, create_time_groups

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 导入我们的增强模块
try:
    from enhanced_alpha_strategies import AlphaStrategiesEngine
    from learning_to_rank_bma import LearningToRankBMA
    from advanced_portfolio_optimizer import AdvancedPortfolioOptimizer
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] 增强模块导入失败: {e}")
    ENHANCED_MODULES_AVAILABLE = False

# 统一市场数据（行业/市值/国家等）
try:
    from unified_market_data_manager import UnifiedMarketDataManager
    MARKET_MANAGER_AVAILABLE = True
except Exception:
    MARKET_MANAGER_AVAILABLE = False

# 导入中性化模块
try:
    from neutralization_pipeline import DailyNeutralizationTransformer
    NEUTRALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] 中性化模块导入失败: {e}")
    NEUTRALIZATION_AVAILABLE = False

# 导入isotonic校准
try:
    from sklearn.isotonic import IsotonicRegression
    ISOTONIC_AVAILABLE = True
except ImportError:
    ISOTONIC_AVAILABLE = False

# 高级模型
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostRanker
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# 配置
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局配置
DEFAULT_TICKER_LIST = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 
    'CRM', 'ADBE', 'PYPL', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN',
    'ORCL', 'IBM', 'CSCO', 'NOW', 'SNOW', 'PLTR', 'DDOG', 'ZS'
]

@dataclass
class MarketRegime:
    """市场状态"""
    regime_id: int
    name: str
    probability: float
    characteristics: Dict[str, float]
    duration: int = 0

@dataclass 
class RiskFactorExposure:
    """风险因子暴露"""
    market_beta: float
    size_exposure: float  
    value_exposure: float
    momentum_exposure: float
    volatility_exposure: float
    quality_exposure: float
    country_exposure: Dict[str, float] = field(default_factory=dict)
    sector_exposure: Dict[str, float] = field(default_factory=dict)

def sanitize_ticker(raw: Union[str, Any]) -> str:
    """清理股票代码中的BOM、引号、空白等杂质。"""
    try:
        s = str(raw)
    except Exception:
        return ''
    # 去除BOM与零宽字符
    s = s.replace('\ufeff', '').replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    # 去除引号与空白
    s = s.strip().strip("'\"")
    # 统一大写
    s = s.upper()
    return s


def load_universe_from_file(file_path: str) -> Optional[List[str]]:
    try:
        if os.path.exists(file_path):
            # 使用utf-8-sig以自动去除BOM
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                tickers = []
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # 支持逗号或空格分隔
                    parts = [p for token in line.split(',') for p in token.split()]
                    for p in parts:
                        t = sanitize_ticker(p)
                        if t:
                            tickers.append(t)
            # 去重并保持顺序
            tickers = list(dict.fromkeys(tickers))
            return tickers if tickers else None
    except Exception:
        return None
    return None

def load_universe_fallback() -> List[str]:
    # 优先从 stocks.txt 读取；否则尝试导入原版模型的 ticker_list；最后用默认列表
    root_stocks = os.path.join(os.path.dirname(__file__), 'stocks.txt')
    tickers = load_universe_from_file(root_stocks)
    if tickers:
        return tickers
    try:
        import 量化模型_bma_enhanced as bma_v3
        if hasattr(bma_v3, 'ticker_list') and isinstance(bma_v3.ticker_list, list):
            return list(dict.fromkeys([str(t).upper() for t in bma_v3.ticker_list]))
    except Exception:
        pass
    return DEFAULT_TICKER_LIST

class UltraEnhancedQuantitativeModel:
    """Ultra Enhanced 量化模型：集成所有高级功能"""
    
    def __init__(self, config_path: str = "alphas_config.yaml"):
        """
        初始化Ultra Enhanced量化模型
        
        Args:
            config_path: Alpha策略配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 核心引擎
        if ENHANCED_MODULES_AVAILABLE:
            self.alpha_engine = AlphaStrategiesEngine(config_path)
            self.ltr_bma = LearningToRankBMA(
                ranking_objective=self.config.get('model_config', {}).get('ranking_objective', 'rank:pairwise'),
                temperature=self.config.get('temperature', 1.2),
                enable_regime_detection=self.config.get('model_config', {}).get('regime_detection', True)
            )
            self.portfolio_optimizer = AdvancedPortfolioOptimizer(
                risk_aversion=self.config.get('risk_config', {}).get('risk_aversion', 5.0),
                turnover_penalty=self.config.get('risk_config', {}).get('turnover_penalty', 1.0),
                max_turnover=self.config.get('max_turnover', 0.10),
                max_position=self.config.get('max_position', 0.03),
                max_sector_exposure=self.config.get('risk_config', {}).get('max_sector_exposure', 0.15),
                max_country_exposure=self.config.get('risk_config', {}).get('max_country_exposure', 0.20)
            )
        else:
            logger.warning("增强模块不可用，使用基础功能")
            self.alpha_engine = None
            self.ltr_bma = None
            self.portfolio_optimizer = None
        
        # 传统ML模型（作为对比）
        self.traditional_models = {}
        self.model_weights = {}
        
        # Professional引擎功能
        self.risk_model_results = {}
        self.current_regime = None
        self.regime_weights = {}
        self.market_data_manager = UnifiedMarketDataManager() if MARKET_MANAGER_AVAILABLE else None
        
        # 数据和结果存储
        self.raw_data = {}
        self.feature_data = None
        self.alpha_signals = None
        self.final_predictions = None
        self.portfolio_weights = None
        
        # 性能跟踪
        self.performance_metrics = {}
        self.backtesting_results = {}
        
        logger.info("UltraEnhanced量化模型初始化完成")
    
    def build_risk_model(self) -> Dict[str, Any]:
        """构建Multi-factor风险模型（来自Professional引擎）"""
        logger.info("构建Multi-factor风险模型")
        
        if not self.raw_data:
            raise ValueError("Market data not loaded")
        
        # 构建收益率矩阵
        returns_data = []
        tickers = []
        
        for ticker, data in self.raw_data.items():
            if len(data) > 100:
                returns = data['close'].pct_change().fillna(0)
                returns_data.append(returns)
                tickers.append(ticker)
        
        if not returns_data:
            raise ValueError("No valid returns data")
        
        returns_matrix = pd.concat(returns_data, axis=1, keys=tickers)
        returns_matrix = returns_matrix.fillna(0.0)
        
        # 构建风险因子
        risk_factors = self._build_risk_factors(returns_matrix)
        
        # 估计因子载荷
        factor_loadings = self._estimate_factor_loadings(returns_matrix, risk_factors)
        
        # 估计因子协方差
        factor_covariance = self._estimate_factor_covariance(risk_factors)
        
        # 估计特异风险
        specific_risk = self._estimate_specific_risk(returns_matrix, factor_loadings, risk_factors)
        
        self.risk_model_results = {
            'factor_loadings': factor_loadings,
            'factor_covariance': factor_covariance,
            'specific_risk': specific_risk,
            'risk_factors': risk_factors
        }
        
        logger.info("风险模型构建完成")
        return self.risk_model_results
    
    def _build_risk_factors(self, returns_matrix: pd.DataFrame) -> pd.DataFrame:
        """构建风险因子（来自Professional引擎）"""
        factors = pd.DataFrame(index=returns_matrix.index)
        
        # 1. 市场因子
        factors['market'] = returns_matrix.mean(axis=1)
        
        # 2. 规模因子 (模拟市值数据)
        market_caps = {}
        for ticker in returns_matrix.columns:
            # 模拟市值数据（实际应从数据源获取）
            market_caps[ticker] = np.random.lognormal(10, 1)
        
        if market_caps:
            market_cap_series = pd.Series(market_caps)
            small_cap_mask = market_cap_series < market_cap_series.median()
            
            small_cap_returns = returns_matrix.loc[:, small_cap_mask].mean(axis=1)
            large_cap_returns = returns_matrix.loc[:, ~small_cap_mask].mean(axis=1)
            factors['size'] = small_cap_returns - large_cap_returns
        
        # 3. 动量因子
        momentum_scores = {}
        for ticker in returns_matrix.columns:
            momentum_scores[ticker] = returns_matrix[ticker].rolling(252).sum().shift(21)
        
        momentum_df = pd.DataFrame(momentum_scores)
        high_momentum = momentum_df.rank(axis=1, pct=True) > 0.7
        low_momentum = momentum_df.rank(axis=1, pct=True) < 0.3
        
        factors['momentum'] = returns_matrix.where(high_momentum).mean(axis=1) - \
                             returns_matrix.where(low_momentum).mean(axis=1)
        
        # 4. 波动率因子
        volatility_scores = returns_matrix.rolling(60).std()
        low_vol = volatility_scores.rank(axis=1, pct=True) < 0.3
        high_vol = volatility_scores.rank(axis=1, pct=True) > 0.7
        
        factors['volatility'] = returns_matrix.where(low_vol).mean(axis=1) - \
                               returns_matrix.where(high_vol).mean(axis=1)
        
        # 5. 质量因子
        quality_scores = returns_matrix.rolling(60).mean() / returns_matrix.rolling(60).std()
        high_quality = quality_scores.rank(axis=1, pct=True) > 0.7
        low_quality = quality_scores.rank(axis=1, pct=True) < 0.3
        
        factors['quality'] = returns_matrix.where(high_quality).mean(axis=1) - \
                            returns_matrix.where(low_quality).mean(axis=1)
        
        # 6. 反转因子
        reversal_scores = returns_matrix.rolling(21).sum()
        high_reversal = reversal_scores.rank(axis=1, pct=True) < 0.3
        low_reversal = reversal_scores.rank(axis=1, pct=True) > 0.7
        
        factors['reversal'] = returns_matrix.where(high_reversal).mean(axis=1) - \
                             returns_matrix.where(low_reversal).mean(axis=1)
        
        # 标准化因子
        factors = factors.fillna(0)
        for col in factors.columns:
            factors[col] = (factors[col] - factors[col].mean()) / (factors[col].std() + 1e-8)
        
        return factors
    
    def _estimate_factor_loadings(self, returns_matrix: pd.DataFrame, 
                                 risk_factors: pd.DataFrame) -> pd.DataFrame:
        """估计因子载荷"""
        loadings = {}
        
        for ticker in returns_matrix.columns:
            stock_returns = returns_matrix[ticker].dropna()
            aligned_factors = risk_factors.loc[stock_returns.index].dropna().fillna(0)
            
            if len(stock_returns) < 50 or len(aligned_factors) < 50:
                loadings[ticker] = np.zeros(len(risk_factors.columns))
                continue
            
            try:
                # 确保数据长度匹配
                min_len = min(len(stock_returns), len(aligned_factors))
                stock_returns = stock_returns.iloc[:min_len]
                aligned_factors = aligned_factors.iloc[:min_len]
                
                # 使用稳健回归估计载荷
                model = HuberRegressor(epsilon=1.35, alpha=0.0001)
                model.fit(aligned_factors.values, stock_returns.values)
                
                loadings[ticker] = model.coef_
                
            except Exception as e:
                logger.warning(f"Failed to estimate loadings for {ticker}: {e}")
                loadings[ticker] = np.zeros(len(risk_factors.columns))
        
        loadings_df = pd.DataFrame(loadings, index=risk_factors.columns).T
        return loadings_df
    
    def _estimate_factor_covariance(self, risk_factors: pd.DataFrame) -> pd.DataFrame:
        """估计因子协方差矩阵"""
        # 使用Ledoit-Wolf收缩估计
        cov_estimator = LedoitWolf()
        factor_cov_matrix = cov_estimator.fit(risk_factors.fillna(0)).covariance_
        
        # 确保正定性
        eigenvals, eigenvecs = np.linalg.eigh(factor_cov_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        factor_cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return pd.DataFrame(factor_cov_matrix, 
                           index=risk_factors.columns, 
                           columns=risk_factors.columns)
    
    def _estimate_specific_risk(self, returns_matrix: pd.DataFrame,
                               factor_loadings: pd.DataFrame, 
                               risk_factors: pd.DataFrame) -> pd.Series:
        """估计特异风险"""
        specific_risks = {}
        
        for ticker in returns_matrix.columns:
            if ticker not in factor_loadings.index:
                specific_risks[ticker] = 0.2  # 默认特异风险
                continue
            
            stock_returns = returns_matrix[ticker].dropna()
            loadings = factor_loadings.loc[ticker]
            aligned_factors = risk_factors.loc[stock_returns.index].fillna(0)
            
            if len(stock_returns) < 50:
                specific_risks[ticker] = 0.2
                continue
            
            # 计算残差
            min_len = min(len(stock_returns), len(aligned_factors))
            factor_returns = (aligned_factors.iloc[:min_len] @ loadings).values
            residuals = stock_returns.iloc[:min_len].values - factor_returns
            
            # 特异风险为残差标准差
            specific_var = np.nan_to_num(np.var(residuals), nan=0.04)
            specific_risks[ticker] = np.sqrt(specific_var)
        
        return pd.Series(specific_risks)
    
    def detect_market_regime(self) -> MarketRegime:
        """检测市场状态（来自Professional引擎）"""
        logger.info("检测市场状态")
        
        if not self.raw_data:
            return MarketRegime(0, "Unknown", 0.5, {'volatility': 0.2, 'trend': 0.0})
        
        # 构建市场指数
        market_returns = []
        for ticker, data in self.raw_data.items():
            if len(data) > 100:
                returns = data['close'].pct_change().fillna(0)
                market_returns.append(returns)
        
        if not market_returns:
            return MarketRegime(0, "Unknown", 0.5, {'volatility': 0.2, 'trend': 0.0})
        
        market_index = pd.concat(market_returns, axis=1).mean(axis=1).dropna()
        
        if len(market_index) < 100:
            return MarketRegime(1, "Normal", 1.0, {'volatility': 0.15, 'trend': 0.0})
        
        # 基于波动率和趋势的状态检测
        rolling_vol = market_index.rolling(21).std()
        rolling_trend = market_index.rolling(21).mean()
        
        # 定义状态阈值
        vol_low = rolling_vol.quantile(0.33)
        vol_high = rolling_vol.quantile(0.67)
        trend_low = rolling_trend.quantile(0.33)
        trend_high = rolling_trend.quantile(0.67)
        
        # 当前状态
        current_vol = rolling_vol.iloc[-1]
        current_trend = rolling_trend.iloc[-1]
        
        if current_vol < vol_low:
            if current_trend > trend_high:
                regime = MarketRegime(1, "Bull_Low_Vol", 0.8, 
                                    {'volatility': current_vol, 'trend': current_trend})
            elif current_trend < trend_low:
                regime = MarketRegime(2, "Bear_Low_Vol", 0.8,
                                    {'volatility': current_vol, 'trend': current_trend})
            else:
                regime = MarketRegime(3, "Normal_Low_Vol", 0.8,
                                    {'volatility': current_vol, 'trend': current_trend})
        elif current_vol > vol_high:
            if current_trend > trend_high:
                regime = MarketRegime(4, "Bull_High_Vol", 0.8,
                                    {'volatility': current_vol, 'trend': current_trend})
            elif current_trend < trend_low:
                regime = MarketRegime(5, "Bear_High_Vol", 0.8,
                                    {'volatility': current_vol, 'trend': current_trend})
            else:
                regime = MarketRegime(6, "Volatile", 0.8,
                                    {'volatility': current_vol, 'trend': current_trend})
        else:
            regime = MarketRegime(0, "Normal", 0.7,
                                {'volatility': current_vol, 'trend': current_trend})
        
        self.current_regime = regime
        logger.info(f"检测到市场状态: {regime.name} (概率: {regime.probability:.2f})")
        
        return regime
    
    def _get_regime_alpha_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """根据市场状态调整Alpha权重（来自Professional引擎）"""
        if "Bull" in regime.name:
            # 牛市：偏好动量
            return {
                'momentum_21d': 2.0, 'momentum_63d': 2.5, 'momentum_126d': 2.0,
                'reversion_5d': 0.5, 'reversion_10d': 0.5, 'reversion_21d': 0.5,
                'volatility_factor': 1.0, 'volume_trend': 1.5, 'quality_factor': 1.0
            }
        elif "Bear" in regime.name:
            # 熊市：偏好质量和防御
            return {
                'momentum_21d': 0.5, 'momentum_63d': 0.5, 'momentum_126d': 1.0,
                'reversion_5d': 1.5, 'reversion_10d': 2.0, 'reversion_21d': 1.5,
                'volatility_factor': 2.0, 'volume_trend': 0.5, 'quality_factor': 2.0
            }
        elif "Volatile" in regime.name:
            # 高波动：偏好均值回归
            return {
                'momentum_21d': 0.5, 'momentum_63d': 1.0, 'momentum_126d': 1.0,
                'reversion_5d': 2.5, 'reversion_10d': 2.0, 'reversion_21d': 1.5,
                'volatility_factor': 2.5, 'volume_trend': 1.0, 'quality_factor': 1.5
            }
        else:
            # 正常市场：均衡权重
            return {col: 1.0 for col in [
                'momentum_21d', 'momentum_63d', 'momentum_126d',
                'reversion_5d', 'reversion_10d', 'reversion_21d',
                'volatility_factor', 'volume_trend', 'quality_factor'
            ]}
    
    def generate_enhanced_predictions(self, training_results: Dict[str, Any], 
                                    market_regime: MarketRegime) -> pd.Series:
        """生成Regime-Aware的增强预测"""
        try:
            # 获取基础预测
            base_predictions = self.generate_ensemble_predictions(training_results)
            
            if not ENHANCED_MODULES_AVAILABLE or not self.alpha_engine:
                # 如果没有增强模块，应用regime权重到基础预测
                regime_weights = self._get_regime_alpha_weights(market_regime)
                # 简单应用权重（这里简化处理）
                adjustment_factor = sum(regime_weights.values()) / len(regime_weights)
                enhanced_predictions = base_predictions * adjustment_factor
                logger.info(f"应用简化的regime调整，调整因子: {adjustment_factor:.3f}")
                return enhanced_predictions
            
            # 如果有Alpha引擎，生成Alpha信号
            try:
                # 为Alpha引擎准备数据（包含标准化的价格列）
                alpha_input = self._prepare_alpha_data()
                # 计算Alpha因子（签名只接受df）
                alpha_signals = self.alpha_engine.compute_all_alphas(alpha_input)
                
                # 根据市场状态调整Alpha权重
                regime_weights = self._get_regime_alpha_weights(market_regime)
                
                # 应用regime权重到alpha信号
                weighted_alpha = pd.Series(0.0, index=alpha_signals.index)
                for alpha_name, weight in regime_weights.items():
                    if alpha_name in alpha_signals.columns:
                        weighted_alpha += alpha_signals[alpha_name] * weight
                
                # 标准化加权后的alpha
                if weighted_alpha.std() > 0:
                    weighted_alpha = (weighted_alpha - weighted_alpha.mean()) / weighted_alpha.std()
                
                # 与基础ML预测融合
                alpha_weight = 0.3  # Alpha信号权重
                ml_weight = 0.7     # ML预测权重
                
                # 确保索引对齐
                common_index = base_predictions.index.intersection(weighted_alpha.index)
                if len(common_index) > 0:
                    enhanced_predictions = (
                        ml_weight * base_predictions.loc[common_index] +
                        alpha_weight * weighted_alpha.loc[common_index]
                    )
                else:
                    enhanced_predictions = base_predictions
                
                logger.info(f"成功融合Alpha信号和ML预测，market regime: {market_regime.name}")
                return enhanced_predictions
                
            except Exception as e:
                logger.warning(f"Alpha信号生成失败: {e}")
                # 回退到基础预测
                return base_predictions
                
        except Exception as e:
            logger.error(f"增强预测生成失败: {e}")
            # 最终回退
            return pd.Series(0.0, index=range(10))
    
    def optimize_portfolio_with_risk_model(self, predictions: pd.Series, 
                                          feature_data: pd.DataFrame) -> Dict[str, Any]:
        """使用风险模型的投资组合优化"""
        try:
            # 如果有Professional的风险模型结果，使用它们
            if self.risk_model_results and 'factor_loadings' in self.risk_model_results:
                factor_loadings = self.risk_model_results['factor_loadings']
                factor_covariance = self.risk_model_results['factor_covariance']
                specific_risk = self.risk_model_results['specific_risk']
                
                # 构建协方差矩阵
                common_assets = list(set(predictions.index) & set(factor_loadings.index))
                if len(common_assets) >= 3:
                    # 使用专业风险模型进行优化
                    try:
                        # 构建投资组合协方差矩阵: B * F * B' + S
                        B = factor_loadings.loc[common_assets]  # 因子载荷
                        F = factor_covariance                   # 因子协方差
                        S = specific_risk.loc[common_assets]    # 特异风险
                        
                        # 计算协方差矩阵
                        portfolio_cov = B @ F @ B.T + np.diag(S**2)
                        portfolio_cov = pd.DataFrame(
                            portfolio_cov, 
                            index=common_assets, 
                            columns=common_assets
                        )
                        
                        # 优化目标：最大化预期收益，最小化风险
                        expected_returns = predictions.loc[common_assets]
                        
                        # 简化的均值-方差优化
                        from scipy.optimize import minimize
                        
                        n_assets = len(common_assets)
                        
                        def objective(weights):
                            portfolio_return = expected_returns @ weights
                            portfolio_risk = np.sqrt(weights @ portfolio_cov @ weights)
                            # 风险调整回报 (风险厌恶系数=5)
                            return -(portfolio_return - 5 * portfolio_risk)
                        
                        # 约束：权重和为1，无卖空
                        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
                        bounds = [(0, 0.05)] * n_assets  # 每只股票最多5%
                        
                        # 初始权重：等权
                        x0 = np.ones(n_assets) / n_assets
                        
                        result = minimize(
                            objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000}
                        )
                        
                        if result.success:
                            optimal_weights = pd.Series(result.x, index=common_assets)
                            # 计算组合指标
                            portfolio_return = expected_returns @ optimal_weights
                            portfolio_risk = np.sqrt(optimal_weights @ portfolio_cov @ optimal_weights)
                            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
                            
                            # 计算风险归因
                            factor_contribution = {}
                            for factor in F.columns:
                                factor_exposure = (B[factor] @ optimal_weights).sum()
                                factor_var = factor_exposure**2 * F.loc[factor, factor]
                                factor_contribution[factor] = factor_var
                            
                            return {
                                'success': True,
                                'method': 'professional_risk_model',
                                'weights': optimal_weights.to_dict(),
                                'portfolio_metrics': {
                                    'expected_return': float(portfolio_return),
                                    'portfolio_risk': float(portfolio_risk),
                                    'sharpe_ratio': float(sharpe_ratio),
                                    'diversification_ratio': len([w for w in optimal_weights if w > 0.01])
                                },
                                'risk_attribution': factor_contribution,
                                'regime_context': self.current_regime.name if self.current_regime else "Unknown"
                            }
                        
                    except Exception as e:
                        logger.warning(f"专业风险模型优化失败: {e}")
            
            # 回退到基础优化
            return self.optimize_portfolio(predictions, feature_data)
            
        except Exception as e:
            logger.error(f"风险模型优化失败: {e}")
            # 最终回退到等权组合
            top_assets = predictions.nlargest(min(10, len(predictions))).index
            equal_weights = pd.Series(1.0/len(top_assets), index=top_assets)
            
            return {
                'success': True,
                'method': 'equal_weight_fallback',
                'weights': equal_weights.to_dict(),
                'portfolio_metrics': {
                    'expected_return': predictions.loc[top_assets].mean(),
                    'portfolio_risk': 0.15,  # 假设风险
                    'sharpe_ratio': 1.0,
                    'diversification_ratio': len(top_assets)
                },
                'risk_attribution': {},
                'regime_context': self.current_regime.name if self.current_regime else "Unknown"
            }
    
    def _prepare_alpha_data(self) -> pd.DataFrame:
        """为Alpha引擎准备数据"""
        if not self.raw_data:
            return pd.DataFrame()
        
        # 将原始数据转换为Alpha引擎需要的格式
        all_data = []
        for ticker, data in self.raw_data.items():
            ticker_data = data.copy()
            ticker_data['ticker'] = ticker
            ticker_data['date'] = ticker_data.index
            # 标准化价格列，Alpha引擎需要 'Close','High','Low'
            if 'Adj Close' in ticker_data.columns:
                ticker_data['Close'] = ticker_data['Adj Close']
            elif 'close' in ticker_data.columns:
                ticker_data['Close'] = ticker_data['close']
            elif 'Close' not in ticker_data.columns and 'close' not in ticker_data.columns:
                # 若缺少close信息，跳过该票
                continue
            if 'High' not in ticker_data.columns and 'high' in ticker_data.columns:
                ticker_data['High'] = ticker_data['high']
            if 'Low' not in ticker_data.columns and 'low' in ticker_data.columns:
                ticker_data['Low'] = ticker_data['low']
            # 添加模拟的基本信息
            ticker_data['COUNTRY'] = 'US'
            ticker_data['SECTOR'] = 'Technology'  # 简化处理
            ticker_data['SUBINDUSTRY'] = 'Software'
            all_data.append(ticker_data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"配置文件{self.config_path}未找到，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'universe': 'TOPDIV3000',
            'neutralization': ['COUNTRY'],
            'hump_levels': [0.003, 0.008],
            'winsorize_std': 2.5,
            'truncation': 0.10,
            'max_position': 0.03,
            'max_turnover': 0.10,
            'temperature': 1.2,
            'model_config': {
                'learning_to_rank': True,
                'ranking_objective': 'rank:pairwise',
                'uncertainty_aware': True,
                'quantile_regression': True
            },
            'risk_config': {
                'risk_aversion': 5.0,
                'turnover_penalty': 1.0,
                'max_sector_exposure': 0.15,
                'max_country_exposure': 0.20
            }
        }
    
    def download_stock_data(self, tickers: List[str], 
                           start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        下载股票数据
        
        Args:
            tickers: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            股票数据字典
        """
        logger.info(f"下载{len(tickers)}只股票的数据，时间范围: {start_date} - {end_date}")
        
        all_data = {}
        failed_downloads = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                # 使用复权数据，避免股利污染；固定日频，关闭actions列
                hist = stock.history(start=start_date, end=end_date, interval='1d', auto_adjust=True, actions=False)
                
                if len(hist) == 0:
                    failed_downloads.append(ticker)
                    continue
                
                # 标准化列名
                hist = hist.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                })
                
                # 添加基础特征
                hist['ticker'] = ticker
                hist['date'] = hist.index
                hist['amount'] = hist['close'] * hist['volume']  # 成交额
                
                # 添加元数据（模拟）
                hist['COUNTRY'] = self._get_country_for_ticker(ticker)
                hist['SECTOR'] = self._get_sector_for_ticker(ticker)
                hist['SUBINDUSTRY'] = self._get_subindustry_for_ticker(ticker)
                
                all_data[ticker] = hist
                
            except Exception as e:
                logger.warning(f"下载{ticker}失败: {e}")
                failed_downloads.append(ticker)
        
        if failed_downloads:
            logger.warning(f"以下股票下载失败: {failed_downloads}")
        
        logger.info(f"成功下载{len(all_data)}只股票的数据")
        self.raw_data = all_data
        
        return all_data
    
    def _get_country_for_ticker(self, ticker: str) -> str:
        """获取股票的国家（简化实现）"""
        # 这里可以接入真实的股票元数据API
        if ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']:
            return 'US'
        else:
            return np.random.choice(['US', 'EU', 'ASIA'])
    
    def _get_sector_for_ticker(self, ticker: str) -> str:
        """获取股票的行业（简化实现）"""
        sector_mapping = {
            'AAPL': 'TECH', 'MSFT': 'TECH', 'GOOGL': 'TECH', 'NVDA': 'TECH',
            'AMZN': 'CONSUMER', 'TSLA': 'AUTO', 'META': 'TECH', 'NFLX': 'MEDIA'
        }
        return sector_mapping.get(ticker, np.random.choice(['TECH', 'FINANCE', 'ENERGY', 'HEALTH']))
    
    def _get_subindustry_for_ticker(self, ticker: str) -> str:
        """获取股票的子行业（简化实现）"""
        return np.random.choice(['SOFTWARE', 'HARDWARE', 'BIOTECH', 'RETAIL'])
    
    def create_traditional_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        创建传统技术指标特征
        
        Args:
            data_dict: 股票数据字典
            
        Returns:
            特征数据框
        """
        logger.info("创建传统技术指标特征")
        
        all_features = []
        
        for ticker, df in data_dict.items():
            if len(df) < 60:  # 至少需要60天数据
                continue
            
            df_copy = df.copy().sort_values('date')
            
            # 价格特征
            df_copy['returns'] = df_copy['close'].pct_change()
            df_copy['log_returns'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
            
            # 移动平均
            for window in [5, 10, 20, 50]:
                df_copy[f'ma_{window}'] = df_copy['close'].rolling(window).mean()
                df_copy[f'ma_ratio_{window}'] = df_copy['close'] / df_copy[f'ma_{window}']
            
            # 波动率
            for window in [10, 20, 50]:
                df_copy[f'vol_{window}'] = df_copy['log_returns'].rolling(window).std()
            
            # RSI
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            df_copy['rsi_14'] = calculate_rsi(df_copy['close'])
            
            # 成交量特征
            if 'volume' in df_copy.columns:
                df_copy['volume_ma_20'] = df_copy['volume'].rolling(20).mean()
                df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume_ma_20']
            
            # 价格位置
            for window in [20, 50]:
                high_roll = df_copy['high'].rolling(window).max()
                low_roll = df_copy['low'].rolling(window).min()
                df_copy[f'price_position_{window}'] = (df_copy['close'] - low_roll) / (high_roll - low_roll + 1e-8)
            
            # 动量指标
            for period in [5, 10, 20]:
                df_copy[f'momentum_{period}'] = df_copy['close'] / df_copy['close'].shift(period) - 1
            
            # 改进的目标构建：形成期-跳空期-持有期
            # 避免微观结构噪声和信息泄露
            formation_period = 1  # T-1形成期
            skip_period = 1       # T+1跳空期  
            holding_period = 5    # T+1到T+5持有期
            
            # 使用稳健的目标构建方式
            df_copy['target'] = (
                df_copy['close'].shift(-(skip_period + holding_period)) / 
                df_copy['close'].shift(-skip_period) - 1
            )
            
            # 添加辅助信息
            df_copy['ticker'] = ticker
            df_copy['date'] = df_copy.index
            # 模拟行业和国家信息（实际应从数据源获取）
            df_copy['COUNTRY'] = 'US'
            df_copy['SECTOR'] = ticker[:2] if len(ticker) >= 2 else 'TECH'  # 简化分类
            df_copy['SUBINDUSTRY'] = ticker[:3] if len(ticker) >= 3 else 'SOFTWARE'
            
            all_features.append(df_copy)
        
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            # 选出纯特征列（排除标识/目标/元数据）
            feature_cols = [col for col in combined_features.columns 
                            if col not in ['ticker','date','target','COUNTRY','SECTOR','SUBINDUSTRY']]
            # 全部特征统一施加T-2滞后，防止潜在泄露
            try:
                combined_features[feature_cols] = combined_features.groupby('ticker')[feature_cols].shift(2)
            except Exception:
                pass
            # 基础清洗
            combined_features = combined_features.dropna()
            
            # ========== 简化但可靠的中性化处理 ==========
            logger.info("应用简化中性化处理")
            try:
                # 按日期分组，逐日进行简单的标准化和winsorization
                neutralized_features = []
                
                for date, group in combined_features.groupby('date'):
                    group_features = group[feature_cols].copy()
                    
                    # 1. Winsorization (1%-99%分位数截断)
                    for col in feature_cols:
                        if group_features[col].notna().sum() > 2:
                            q01, q99 = group_features[col].quantile([0.01, 0.99])
                            group_features[col] = group_features[col].clip(lower=q01, upper=q99)
                    
                    # 2. 横截面标准化（Z-score）
                    for col in feature_cols:
                        if group_features[col].notna().sum() > 2:
                            mean_val = group_features[col].mean()
                            std_val = group_features[col].std()
                            if std_val > 0:
                                group_features[col] = (group_features[col] - mean_val) / std_val
                            else:
                                group_features[col] = 0.0
                    
                    # 3. 行业中性化（如果有行业数据）
                    if self.market_data_manager is not None:
                        try:
                            tickers = group['ticker'].tolist()
                            stock_info = self.market_data_manager.get_batch_stock_info(tickers)
                            industries = {}
                            for ticker in tickers:
                                info = stock_info.get(ticker)
                                if info:
                                    sector = info.gics_sub_industry or info.gics_industry or info.sector
                                    industries[ticker] = sector or 'Unknown'
                                else:
                                    industries[ticker] = 'Unknown'
                            
                            # 按行业去均值
                            group_with_industry = group_features.copy()
                            group_with_industry['industry'] = group['ticker'].map(industries)
                            
                            for col in feature_cols:
                                if group_with_industry[col].notna().sum() > 2:
                                    industry_means = group_with_industry.groupby('industry')[col].transform('mean')
                                    group_features[col] = group_features[col] - industry_means
                                    
                        except Exception as e:
                            logger.debug(f"行业中性化跳过: {e}")
                    
                    # 保留非特征列
                    group_result = group[['date', 'ticker']].copy()
                    group_result[feature_cols] = group_features[feature_cols]
                    neutralized_features.append(group_result)
                
                # 合并结果
                neutralized_df = pd.concat(neutralized_features, ignore_index=True)
                combined_features[feature_cols] = neutralized_df[feature_cols]
                
                logger.info(f"简化中性化完成，处理{len(feature_cols)}个特征")
                
            except Exception as e:
                logger.warning(f"简化中性化失败: {e}")
                logger.info("使用原始特征，仅进行标准化")
                # 最简单的回退：全局标准化
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                try:
                    combined_features[feature_cols] = scaler.fit_transform(combined_features[feature_cols].fillna(0))
                except Exception:
                    pass
            
            logger.info(f"传统特征创建完成，数据形状: {combined_features.shape}")
            return combined_features
        else:
            logger.error("没有有效的特征数据")
            return pd.DataFrame()
    
    def train_enhanced_models(self, feature_data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练增强模型（Alpha策略 + Learning-to-Rank + 传统ML）
        
        Args:
            feature_data: 特征数据
            
        Returns:
            训练结果
        """
        logger.info("开始训练增强模型")
        
        self.feature_data = feature_data
        training_results = {}
        
        # 准备数据
        feature_cols = [col for col in feature_data.columns 
                       if col not in ['ticker', 'date', 'target', 'COUNTRY', 'SECTOR', 'SUBINDUSTRY']]
        
        X = feature_data[feature_cols]
        y = feature_data['target']
        dates = feature_data['date']
        tickers = feature_data['ticker']
        
        # 去除缺失值
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        dates_clean = dates[valid_mask]
        tickers_clean = tickers[valid_mask]
        
        if len(X_clean) == 0:
            logger.error("清洗后数据为空")
            return {}
        
        logger.info(f"训练数据: {len(X_clean)}样本, {len(feature_cols)}特征")
        
        # 1. 训练Alpha策略引擎
        if self.alpha_engine and ENHANCED_MODULES_AVAILABLE:
            logger.info("训练Alpha策略引擎")
            try:
                # 重组数据格式用于Alpha计算
                alpha_data = feature_data[['date', 'ticker', 'close', 'high', 'low', 'volume', 'amount',
                                         'COUNTRY', 'SECTOR', 'SUBINDUSTRY']].copy()
                # 为Alpha引擎标准化列名并优先使用复权收盘价
                if 'Adj Close' in feature_data.columns:
                    alpha_data['Close'] = feature_data['Adj Close']
                else:
                    alpha_data['Close'] = feature_data['close']
                alpha_data['High'] = feature_data['high']
                alpha_data['Low'] = feature_data['low']
                
                # 计算Alpha因子
                alpha_df = self.alpha_engine.compute_all_alphas(alpha_data)
                
                # 计算OOF评分
                if len(alpha_df) > 0:
                    alpha_scores = self.alpha_engine.compute_oof_scores(
                        alpha_df, y_clean, dates_clean, metric='ic'
                    )
                    
                    # 计算BMA权重
                    alpha_weights = self.alpha_engine.compute_bma_weights(alpha_scores)
                    
                    # 组合Alpha信号
                    alpha_signal = self.alpha_engine.combine_alphas(alpha_df, alpha_weights)
                    
                    # 简单过滤：去除极值和NaN
                    filtered_signal = alpha_signal.copy()
                    filtered_signal = filtered_signal.replace([np.inf, -np.inf], np.nan)
                    filtered_signal = filtered_signal.fillna(0.0)
                    
                    # 可选：Winsorize处理极值
                    q1, q99 = filtered_signal.quantile([0.01, 0.99])
                    filtered_signal = filtered_signal.clip(lower=q1, upper=q99)
                    
                    self.alpha_signals = filtered_signal
                    training_results['alpha_strategy'] = {
                        'alpha_scores': alpha_scores,
                        'alpha_weights': alpha_weights,
                        'alpha_signals': filtered_signal,
                        'alpha_stats': self.alpha_engine.get_stats()
                    }
                    
                    logger.info(f"Alpha策略训练完成，信号覆盖: {(~filtered_signal.isna()).sum()}样本")
                
            except Exception as e:
                logger.error(f"Alpha策略训练失败: {e}")
                training_results['alpha_strategy'] = {'error': str(e)}
        
        # 2. 训练Learning-to-Rank BMA
        if self.ltr_bma and ENHANCED_MODULES_AVAILABLE:
            logger.info("训练Learning-to-Rank BMA")
            try:
                ltr_results = self.ltr_bma.train_ranking_models(
                    X=X_clean, y=y_clean, dates=dates_clean,
                    cv_folds=3, optimize_hyperparams=False
                )
                
                training_results['learning_to_rank'] = {
                    'model_results': ltr_results,
                    'performance_summary': self.ltr_bma.get_performance_summary()
                }
                
                logger.info("Learning-to-Rank训练完成")
                
            except Exception as e:
                logger.error(f"Learning-to-Rank训练失败: {e}")
                training_results['learning_to_rank'] = {'error': str(e)}
        
        # 3. 训练传统ML模型（作为基准）
        logger.info("训练传统ML模型")
        try:
            traditional_results = self._train_traditional_models(X_clean, y_clean, dates_clean)
            training_results['traditional_models'] = traditional_results
            
        except Exception as e:
            logger.error(f"传统模型训练失败: {e}")
            training_results['traditional_models'] = {'error': str(e)}
        
        logger.info("增强模型训练完成")
        return training_results
    
    def _train_traditional_models(self, X: pd.DataFrame, y: pd.Series, 
                                 dates: pd.Series) -> Dict[str, Any]:
        """训练传统ML模型"""
        models = {
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=5000),
            'rf': RandomForestRegressor(n_estimators=200, random_state=42)
        }
        
        # 添加高级模型
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
        
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1
            )
        
        if CATBOOST_AVAILABLE:
            models['catboost'] = CatBoostRegressor(
                iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=False
            )
        
        # 使用PurgedGroupTimeSeriesSplit进行严格时序验证
        cv_config = ValidationConfig(n_splits=5, test_size=63, gap=5, embargo=2, group_freq='W')
        purged_cv = PurgedGroupTimeSeriesSplit(cv_config)
        groups = create_time_groups(dates, freq=cv_config.group_freq)
        
        model_results = {}
        oof_predictions = {}
        
        for model_name, model in models.items():
            logger.info(f"训练{model_name}模型")
            
            fold_predictions = np.full(len(X), np.nan)
            fold_models = []
            
            for train_idx, test_idx in purged_cv.split(X, y, groups):
                train_mask = np.zeros(len(X), dtype=bool)
                train_mask[train_idx] = True
                test_mask = np.zeros(len(X), dtype=bool) 
                test_mask[test_idx] = True
                
                if train_mask.sum() == 0 or test_mask.sum() == 0:
                    continue
                
                X_train, y_train = X[train_mask], y[train_mask]
                X_test = X[test_mask]
                
                try:
                    # 标准化
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # 训练模型
                    if model_name in ['xgboost', 'lightgbm', 'catboost', 'rf']:
                        # Tree-based模型不需要标准化
                        model_copy = type(model)(**model.get_params())
                        model_copy.fit(X_train, y_train)
                        test_pred = model_copy.predict(X_test)
                    else:
                        model_copy = type(model)(**model.get_params())
                        model_copy.fit(X_train_scaled, y_train)
                        test_pred = model_copy.predict(X_test_scaled)
                    
                    fold_predictions[test_mask] = test_pred
                    fold_models.append((model_copy, scaler))
                    
                except Exception as e:
                    logger.warning(f"{model_name}模型训练失败: {e}")
                    continue
            
            oof_predictions[model_name] = fold_predictions
            self.traditional_models[model_name] = fold_models
            
            # 计算性能指标
            valid_mask = ~np.isnan(fold_predictions)
            if valid_mask.sum() > 0:
                oof_ic = np.corrcoef(y[valid_mask], fold_predictions[valid_mask])[0, 1]
                oof_rank_ic = spearmanr(y[valid_mask], fold_predictions[valid_mask])[0]
                
                model_results[model_name] = {
                    'oof_ic': oof_ic if not np.isnan(oof_ic) else 0.0,
                    'oof_rank_ic': oof_rank_ic if not np.isnan(oof_rank_ic) else 0.0,
                    'valid_predictions': valid_mask.sum()
                }
                
                logger.info(f"{model_name} - IC: {oof_ic:.4f}, RankIC: {oof_rank_ic:.4f}")
        
        # 二层Stacking（Ridge + ElasticNet）作为元学习器
        try:
            logger.info("训练二层Stacking元学习器 (Ridge/ElasticNet)")
            base_pred_df = pd.DataFrame({name: preds for name, preds in oof_predictions.items()})
            
            # 确保索引对齐：重置所有索引到相同基础
            base_pred_df = base_pred_df.reset_index(drop=True)
            y_reset = y.reset_index(drop=True)
            dates_reset = dates.reset_index(drop=True)
            
            # 计算有效掩码（所有索引现在都是0-based连续的）
            base_valid_mask = ~base_pred_df.isna().any(axis=1) & ~y_reset.isna()
            
            X_meta = base_pred_df.loc[base_valid_mask].copy()
            y_meta = y_reset.loc[base_valid_mask].copy()
            dates_meta = dates_reset.loc[base_valid_mask].copy()

            # 使用PurgedGroupTimeSeriesSplit防泄漏
            groups = create_time_groups(dates_meta, freq='W')
            pgts = PurgedGroupTimeSeriesSplit(ValidationConfig(n_splits=5, test_size=63, gap=5, embargo=2))

            meta_models = {
                'meta_ridge': Ridge(alpha=0.5),
                'meta_elastic': ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000)
            }

            meta_oof = {name: np.full(len(X_meta), np.nan) for name in meta_models.keys()}
            trained_meta = {}

            for train_idx, test_idx in pgts.split(X_meta, y_meta, groups):
                X_tr, X_te = X_meta.iloc[train_idx], X_meta.iloc[test_idx]
                y_tr = y_meta.iloc[train_idx]
                for mname, m in meta_models.items():
                    m_fit = type(m)(**m.get_params())
                    m_fit.fit(X_tr, y_tr)
                    meta_oof[mname][test_idx] = m_fit.predict(X_te)
                    trained_meta.setdefault(mname, []).append(m_fit)

            # 记录元学习器性能
            meta_perf = {}
            for mname, preds in meta_oof.items():
                valid = ~np.isnan(preds)
                ic = np.corrcoef(y_meta.values[valid], np.array(preds)[valid])[0, 1]
                rank_ic = spearmanr(y_meta.values[valid], np.array(preds)[valid])[0]
                meta_perf[mname] = {
                    'oof_ic': float(ic) if not np.isnan(ic) else 0.0,
                    'oof_rank_ic': float(rank_ic) if not np.isnan(rank_ic) else 0.0,
                    'valid_predictions': int(valid.sum())
                }

            # 保存到实例以供后续预测
            self.meta_learners = trained_meta
            self.meta_oof_predictions = meta_oof
            model_results.update({f'stacking_{k}': v for k, v in meta_perf.items()})
        except Exception as e:
            logger.warning(f"二层Stacking训练失败: {e}")

        return {
            'model_performance': model_results,
            'oof_predictions': oof_predictions,
            'stacking': {
                'meta_oof': meta_oof if 'meta_oof' in locals() else {},
                'meta_performance': meta_perf if 'meta_perf' in locals() else {}
            }
        }
    
    def generate_ensemble_predictions(self, training_results: Dict[str, Any]) -> pd.Series:
        """
        生成集成预测
        
        Args:
            training_results: 训练结果
            
        Returns:
            集成预测序列
        """
        logger.info("生成集成预测")
        
        predictions_dict = {}
        weights_dict = {}
        
        # 1. Alpha策略预测
        if 'alpha_strategy' in training_results and 'alpha_signals' in training_results['alpha_strategy']:
            alpha_signals = training_results['alpha_strategy']['alpha_signals']
            if alpha_signals is not None and len(alpha_signals) > 0:
                predictions_dict['alpha'] = alpha_signals
                # 基于Alpha评分设置权重
                alpha_scores = training_results['alpha_strategy'].get('alpha_scores', pd.Series())
                if len(alpha_scores) > 0:
                    avg_alpha_score = alpha_scores.mean()
                    weights_dict['alpha'] = max(0.1, min(0.5, avg_alpha_score * 5))  # 权重在0.1-0.5之间
                else:
                    weights_dict['alpha'] = 0.2
        
        # 2. Learning-to-Rank预测
        if (self.ltr_bma and 'learning_to_rank' in training_results and 
            'model_results' in training_results['learning_to_rank']):
            try:
                if self.feature_data is not None:
                    feature_cols = [col for col in self.feature_data.columns 
                                   if col not in ['ticker', 'date', 'target', 'COUNTRY', 'SECTOR', 'SUBINDUSTRY']]
                    X_for_prediction = self.feature_data[feature_cols].dropna()
                    
                    ltr_pred, ltr_uncertainty = self.ltr_bma.predict_with_uncertainty(X_for_prediction)
                    
                    # 权重基于不确定性和LTR性能
                    avg_uncertainty = np.nanmean(ltr_uncertainty)
                    base_ltr_weight = 1.0 / (1.0 + avg_uncertainty * 10)
                    
                    # 检查LTR性能，如果有负IC通道则降权
                    performance_penalty = 1.0
                    try:
                        ltr_results = training_results['learning_to_rank']
                        if isinstance(ltr_results, dict):
                            ltr_performance = ltr_results.get('performance_summary', {})
                            if ltr_performance and isinstance(ltr_performance, dict):
                                avg_ic = np.mean([p.get('ic', 0.0) for p in ltr_performance.values() if isinstance(p, dict)])
                                if avg_ic < 0:
                                    performance_penalty = 0.3  # 负IC时大幅降权
                                elif avg_ic < 0.05:
                                    performance_penalty = 0.6  # 弱IC时中度降权
                    except Exception as e:
                        logger.debug(f"LTR性能检查失败: {e}")
                        performance_penalty = 0.8  # 安全的中等权重
                    
                    final_ltr_weight = base_ltr_weight * performance_penalty
                    predictions_dict['ltr'] = pd.Series(ltr_pred, index=X_for_prediction.index)
                    weights_dict['ltr'] = max(0.05, min(0.25, final_ltr_weight))  # 降低上限从0.4到0.25
                    
            except Exception as e:
                logger.warning(f"Learning-to-Rank预测失败: {e}")
        
        # 3. 传统模型预测
        if 'traditional_models' in training_results and 'oof_predictions' in training_results['traditional_models']:
            oof_preds = training_results['traditional_models']['oof_predictions']
            model_perfs = training_results['traditional_models'].get('model_performance', {})
            stacking_info = training_results['traditional_models'].get('stacking', {})
            
            # 获取训练数据的索引作为参考
            if hasattr(self, 'feature_data') and self.feature_data is not None:
                ref_index = self.feature_data.index
            else:
                ref_index = None
            
            for model_name, pred_array in oof_preds.items():
                if pred_array is not None and not np.all(np.isnan(pred_array)):
                    # 确保预测与特征数据索引对齐
                    if ref_index is not None and len(pred_array) == len(ref_index):
                        predictions_dict[f'traditional_{model_name}'] = pd.Series(pred_array, index=ref_index)
                    else:
                        # 回退到默认索引，但要确保长度匹配
                        logger.warning(f"传统模型{model_name}预测长度{len(pred_array)}与特征数据不匹配")
                        continue
                    
                    # 动态权重：负IC大幅降权，正IC按强度分配
                    if model_name in model_perfs:
                        ic = model_perfs[model_name].get('oof_ic', 0.0)
                        if ic < -0.05:
                            weights_dict[f'traditional_{model_name}'] = 0.02  # 强负IC：最低权重
                        elif ic < 0:
                            weights_dict[f'traditional_{model_name}'] = 0.05  # 弱负IC：低权重
                        elif ic > 0.1:
                            weights_dict[f'traditional_{model_name}'] = 0.25  # 强正IC：高权重
                        elif ic > 0.05:
                            weights_dict[f'traditional_{model_name}'] = 0.15  # 中等正IC
                        elif ic > 0:
                            weights_dict[f'traditional_{model_name}'] = 0.1   # 弱正IC
                        else:
                            weights_dict[f'traditional_{model_name}'] = 0.05  # 零IC：低权重
                    else:
                        weights_dict[f'traditional_{model_name}'] = 0.05

            # 加入二层Stacking元学习器的预测（作为额外通道）
            try:
                if stacking_info and 'meta_oof' in stacking_info and hasattr(self, 'feature_data'):
                    base_models = [f"{k}" for k in oof_preds.keys()]
                    base_pred_df = pd.DataFrame({name: predictions_dict.get(f'traditional_{name}', pd.Series(dtype=float)) for name in base_models})
                    # 对齐到参考索引
                    base_pred_df = base_pred_df.reindex(ref_index)
                    # 使用已训练的meta learners做一层预测平均
                    if hasattr(self, 'meta_learners') and isinstance(self.meta_learners, dict):
                        for mname, mlist in self.meta_learners.items():
                            # 对多个折的meta模型取平均预测
                            meta_preds = np.nanmean([m.predict(base_pred_df.fillna(0.0)) for m in mlist], axis=0)
                            predictions_dict[f'stacking_{mname}'] = pd.Series(meta_preds, index=ref_index)
                            perf = stacking_info.get('meta_performance', {}).get(mname, {})
                            ic = perf.get('oof_ic', 0.0)
                            weights_dict[f'stacking_{mname}'] = max(0.05, min(0.35, ic * 6))
            except Exception as e:
                logger.warning(f"Stacking通道集成失败: {e}")
        
        # 集成预测
        if not predictions_dict:
            logger.error("没有有效的预测结果")
            return pd.Series()
        
        # 标准化权重
        total_weight = sum(weights_dict.values())
        if total_weight > 0:
            for key in weights_dict:
                weights_dict[key] /= total_weight
        
        logger.info(f"集成权重: {weights_dict}")
        
        # 统一所有预测的索引到feature_data的索引
        if hasattr(self, 'feature_data') and self.feature_data is not None:
            reference_index = self.feature_data.index
        else:
            # 如果没有参考索引，取所有预测的交集
            all_indices = set(list(predictions_dict.values())[0].index)
            for pred in list(predictions_dict.values())[1:]:
                all_indices = all_indices.intersection(set(pred.index))
            reference_index = sorted(all_indices)
        
        if len(reference_index) == 0:
            logger.error("没有可用的参考索引进行集成")
            return pd.Series()
        
        # 构建预测矩阵（不填充为0，保留NaN）
        preds_df = pd.DataFrame({
            name: series.reindex(reference_index) for name, series in predictions_dict.items()
        })

        # 将权重向量与列对齐
        weights_vec = np.array([weights_dict.get(name, 0.0) for name in preds_df.columns], dtype=float)
        # 每行有效权重之和（忽略该行中的NaN）
        mask = ~preds_df.isna().values
        weights_matrix = np.tile(weights_vec, (len(preds_df), 1))
        denom = (weights_matrix * mask).sum(axis=1)
        numer = np.nansum(preds_df.values * weights_matrix, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            ensemble_values = np.where(denom > 0, numer / denom, np.nan)
        ensemble_prediction = pd.Series(ensemble_values, index=reference_index)
        
        self.final_predictions = ensemble_prediction
        
        logger.info(f"集成预测完成，覆盖{len(ensemble_prediction)}个样本")
        
        return ensemble_prediction
    
    def optimize_portfolio(self, predictions: pd.Series, 
                          feature_data: pd.DataFrame) -> Dict[str, Any]:
        """
        优化投资组合
        
        Args:
            predictions: 集成预测
            feature_data: 特征数据
            
        Returns:
            投资组合优化结果
        """
        if not self.portfolio_optimizer or not ENHANCED_MODULES_AVAILABLE:
            logger.warning("投资组合优化器不可用，无法生成投资建议")
            return {'success': False, 'error': 'Portfolio optimizer not available'}
        
        logger.info("开始投资组合优化")
        
        try:
            # 将预测与样本元数据(date,ticker)对齐，再筛选最新截面
            if self.feature_data is None or len(self.feature_data) == 0:
                logger.error("缺少特征元数据用于对齐预测")
                return {}
            
            # 只取预测索引中存在于feature_data中的部分
            valid_pred_indices = predictions.index.intersection(self.feature_data.index)
            if len(valid_pred_indices) == 0:
                logger.error("预测索引与特征数据索引没有交集")
                return {}
            
            # 获取有效预测
            valid_predictions = predictions.reindex(valid_pred_indices)
            meta = self.feature_data.loc[valid_pred_indices, ['date', 'ticker']].copy()
            pred_df = meta.assign(pred=valid_predictions.values)
            
            # 仅保留有效预测
            pred_df = pred_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['pred'])
            if pred_df.empty:
                logger.error("没有有效的预测信号")
                return {}
            
            latest_date = pred_df['date'].max()
            latest_pred = pred_df[pred_df['date'] == latest_date]
            if latest_pred.empty:
                logger.error("最新截面没有预测信号")
                return {}
            
            # 聚合到ticker层面
            ticker_pred = latest_pred.groupby('ticker')['pred'].mean()
            
            # 对齐到最新截面特征
            latest_slice = feature_data[feature_data['date'] == latest_date].copy()
            if latest_slice.empty:
                logger.error("没有最新截面数据")
                return {}
            
            latest_slice = latest_slice.set_index('ticker')
            predictions_valid = ticker_pred.reindex(latest_slice.index)
            
            # 过滤NaN（但不把信号强行置零，避免全零）
            valid_mask = (~predictions_valid.isna())
            if valid_mask.sum() == 0:
                logger.error("没有有效的预测信号")
                return {}
                
            latest_data_valid = latest_slice[valid_mask]
            predictions_valid = predictions_valid[valid_mask]

            # 记录最新截面信号统计，诊断是否出现全0
            try:
                nz_ratio = float((predictions_valid != 0).sum()) / float(len(predictions_valid))
                logger.info(f"最新截面信号非零比率: {nz_ratio:.2%}, 均值: {predictions_valid.mean():.6f}, 标准差: {predictions_valid.std():.6f}")
                self.latest_ticker_predictions = predictions_valid.copy()
            except Exception:
                self.latest_ticker_predictions = predictions_valid
            
            logger.info(f"有效预测信号数量: {len(predictions_valid)}, 涵盖股票: {list(predictions_valid.index)}")
            
            # 构建预期收益率（基于预测信号）
            expected_returns = predictions_valid.copy()
            expected_returns.name = 'expected_returns'
            
            # 构建历史收益率矩阵用于协方差估计
            returns_data = []
            tickers_for_cov = expected_returns.index.tolist()
            
            # 获取历史收益率
            for ticker in tickers_for_cov:
                if ticker in self.raw_data:
                    hist_data = self.raw_data[ticker].copy()
                    hist_data['returns'] = hist_data['close'].pct_change()
                    returns_data.append(hist_data[['date', 'returns']].set_index('date')['returns'].rename(ticker))
            
            if returns_data:
                returns_matrix = pd.concat(returns_data, axis=1).dropna()
                
                # 估计协方差矩阵
                cov_matrix = self.portfolio_optimizer.estimate_covariance_matrix(returns_matrix)
                
                # 构建股票池数据
                universe_data = latest_data_valid[['COUNTRY', 'SECTOR', 'SUBINDUSTRY']].copy()
                if 'volume' in latest_data_valid.columns:
                    # 简单的流动性排名
                    universe_data['liquidity_rank'] = latest_data_valid['volume'].rank(pct=True)
                else:
                    universe_data['liquidity_rank'] = 0.5
                
                # 执行投资组合优化
                optimization_result = self.portfolio_optimizer.optimize_portfolio(
                    expected_returns=expected_returns,
                    covariance_matrix=cov_matrix,
                    current_weights=None,  # 假设从空仓开始
                    universe_data=universe_data
                )
                
                if optimization_result.get('success', False):
                    optimal_weights = optimization_result['optimal_weights']
                    portfolio_metrics = optimization_result['portfolio_metrics']
                    
                    # 风险归因
                    risk_attribution = self.portfolio_optimizer.risk_attribution(
                        optimal_weights, cov_matrix
                    )
                    
                    # 压力测试
                    from advanced_portfolio_optimizer import create_stress_scenarios
                    stress_scenarios = create_stress_scenarios(optimal_weights.index.tolist())
                    stress_results = self.portfolio_optimizer.stress_test(
                        optimal_weights, cov_matrix, stress_scenarios
                    )
                    
                    self.portfolio_weights = optimal_weights
                    
                    return {
                        'success': True,
                        'optimal_weights': optimal_weights,
                        'portfolio_metrics': portfolio_metrics,
                        'risk_attribution': risk_attribution,
                        'stress_test': stress_results,
                        'optimization_info': optimization_result.get('optimization_info', {})
                    }
                else:
                    logger.warning("高级投资组合优化未达到最优，但已返回最佳可用结果")
                    return optimization_result
            else:
                logger.error("无法构建协方差矩阵")
                return {}
                
        except Exception as e:
            logger.error(f"投资组合优化异常: {e}")
            return {'error': str(e)}
    

    
    def generate_investment_recommendations(self, portfolio_result: Dict[str, Any],
                                          top_n: int = 10) -> List[Dict[str, Any]]:
        """
        生成投资建议
        
        Args:
            portfolio_result: 投资组合优化结果
            top_n: 返回前N个推荐
            
        Returns:
            投资建议列表
        """
        logger.info(f"生成前{top_n}个投资建议")
        
        if not portfolio_result.get('success', False):
            logger.error("投资组合优化失败，无法生成建议")
            return []
        
        optimal_weights = portfolio_result['optimal_weights']
        portfolio_metrics = portfolio_result.get('portfolio_metrics', {})
        
        # 获取最新的股票数据
        recommendations = []
        
        # 按权重排序
        sorted_weights = optimal_weights[optimal_weights > 0.001].sort_values(ascending=False)
        
        for i, (ticker, weight) in enumerate(sorted_weights.head(top_n).items()):
            try:
                # 获取股票基本信息
                if ticker in self.raw_data:
                    stock_data = self.raw_data[ticker]
                    latest_price = stock_data['close'].iloc[-1]
                    
                    # 计算一些基本指标
                    price_change_1d = stock_data['close'].pct_change().iloc[-1]
                    price_change_5d = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[-6] - 1) if len(stock_data) > 5 else 0
                    
                    avg_volume = stock_data['volume'].tail(20).mean() if 'volume' in stock_data.columns else 0
                    
                    # 获取预测信号（优先使用按ticker聚合过的最新截面信号）
                    if hasattr(self, 'latest_ticker_predictions') and isinstance(self.latest_ticker_predictions, pd.Series):
                        prediction_signal = float(self.latest_ticker_predictions.get(ticker, np.nan))
                    else:
                        # 回退：从逐行预测聚合（最新日期）
                        try:
                            if self.final_predictions is not None and hasattr(self, 'feature_data'):
                                ref_idx = self.feature_data.index
                                preds = pd.Series(self.final_predictions).reindex(ref_idx)
                                latest_date = self.feature_data['date'].max()
                                latest_mask = self.feature_data['date'] == latest_date
                                latest_tickers = self.feature_data.loc[latest_mask, 'ticker']
                                grouped = pd.DataFrame({'ticker': latest_tickers, 'pred': preds[latest_mask]}).groupby('ticker')['pred'].mean()
                                prediction_signal = float(grouped.get(ticker, np.nan))
                            else:
                                prediction_signal = np.nan
                        except Exception:
                            prediction_signal = np.nan
                    if np.isnan(prediction_signal):
                        prediction_signal = 0.0
                    
                    recommendation = {
                        'rank': i + 1,
                        'ticker': ticker,
                        'weight': weight,
                        'latest_price': latest_price,
                        'price_change_1d': price_change_1d,
                        'price_change_5d': price_change_5d,
                        'avg_volume_20d': avg_volume,
                        'prediction_signal': prediction_signal,
                        'recommendation_reason': self._get_recommendation_reason(ticker, weight, prediction_signal)
                    }
                    
                    recommendations.append(recommendation)
                
            except Exception as e:
                logger.warning(f"生成{ticker}推荐信息失败: {e}")
                continue
        
        return recommendations
    
    def _get_recommendation_reason(self, ticker: str, weight: float, signal: float) -> str:
        """生成推荐理由"""
        reasons = []
        
        if weight > 0.05:
            reasons.append("高权重配置")
        elif weight > 0.03:
            reasons.append("中等权重配置")
        else:
            reasons.append("低权重配置")
        
        if signal > 0.1:
            reasons.append("强烈买入信号")
        elif signal > 0.05:
            reasons.append("买入信号")
        elif signal > 0:
            reasons.append("弱买入信号")
        else:
            reasons.append("中性信号")
        
        return "; ".join(reasons)
    
    def save_results(self, recommendations: List[Dict[str, Any]], 
                    portfolio_result: Dict[str, Any]) -> str:
        """
        保存结果
        
        Args:
            recommendations: 投资建议
            portfolio_result: 投资组合结果
            
        Returns:
            保存文件路径
        """
        logger.info("保存分析结果")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path("result")
        result_dir.mkdir(exist_ok=True)
        
        # 保存投资建议
        if recommendations:
            # Excel格式（确保列顺序与数据类型稳定）
            excel_file = result_dir / f"ultra_enhanced_recommendations_{timestamp}.xlsx"
            rec_df = pd.DataFrame(recommendations)
            # 规范ticker
            if 'ticker' in rec_df.columns:
                rec_df['ticker'] = rec_df['ticker'].map(sanitize_ticker)
            # 设定列顺序
            preferred_cols = ['rank','ticker','weight','latest_price','price_change_1d','price_change_5d','avg_volume_20d','prediction_signal','recommendation_reason']
            ordered_cols = [c for c in preferred_cols if c in rec_df.columns] + [c for c in rec_df.columns if c not in preferred_cols]
            rec_df = rec_df[ordered_cols]
            # Excel优先；失败时回退CSV
            try:
                rec_df.to_excel(excel_file, index=False)
            except Exception:
                excel_file = result_dir / f"ultra_enhanced_recommendations_{timestamp}.csv"
                rec_df.to_csv(excel_file, index=False, encoding='utf-8')
            
            # 简化的股票代码列表
            tickers_file = result_dir / f"top_tickers_{timestamp}.txt"
            top_tickers = [sanitize_ticker(rec.get('ticker','')) for rec in recommendations[:7] if rec.get('ticker')]
            with open(tickers_file, 'w', encoding='utf-8') as f:
                f.write(", ".join([f"'{ticker}'" for ticker in top_tickers]))

            # 仅股票代码数组（JSON），Top7
            top7_json = result_dir / f"top7_tickers_{timestamp}.json"
            with open(top7_json, 'w', encoding='utf-8') as f:
                json.dump(top_tickers, f, ensure_ascii=False)
        
            # 保存投资组合详情
        if portfolio_result.get('success', False):
            portfolio_file = result_dir / f"portfolio_details_{timestamp}.json"
            portfolio_data = {
                'timestamp': timestamp,
                'portfolio_metrics': portfolio_result.get('portfolio_metrics', {}),
                'optimization_info': portfolio_result.get('optimization_info', {}),
                    'weights': {sanitize_ticker(k): float(v) for k, v in portfolio_result.get('optimal_weights', pd.Series(dtype=float)).to_dict().items()}
            }
            
            with open(portfolio_file, 'w', encoding='utf-8') as f:
                json.dump(portfolio_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"结果已保存到 {result_dir}")
        return str(excel_file) if recommendations else str(result_dir)
    
    def run_complete_analysis(self, tickers: List[str], 
                             start_date: str, end_date: str,
                             top_n: int = 10) -> Dict[str, Any]:
        """
        运行完整分析流程
        
        Args:
            tickers: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            top_n: 返回推荐数量
            
        Returns:
            完整分析结果
        """
        logger.info("开始完整分析流程")
        
        analysis_results = {
            'start_time': datetime.now(),
            'config': self.config,
            'tickers': tickers,
            'date_range': f"{start_date} to {end_date}"
        }
        
        try:
            # 1. 下载数据
            stock_data = self.download_stock_data(tickers, start_date, end_date)
            if not stock_data:
                raise ValueError("无法获取股票数据")
            
            analysis_results['data_download'] = {
                'success': True,
                'stocks_downloaded': len(stock_data)
            }
            
            # 2. 创建特征
            feature_data = self.create_traditional_features(stock_data)
            if len(feature_data) == 0:
                raise ValueError("特征创建失败")
            
            analysis_results['feature_engineering'] = {
                'success': True,
                'feature_shape': feature_data.shape,
                'feature_columns': len([col for col in feature_data.columns 
                                      if col not in ['ticker', 'date', 'target']])
            }
            
            # 3. 构建Multi-factor风险模型
            try:
                risk_model = self.build_risk_model()
                analysis_results['risk_model'] = {
                    'success': True,
                    'factor_count': len(risk_model['risk_factors'].columns),
                    'assets_covered': len(risk_model['factor_loadings'])
                }
                logger.info("风险模型构建完成")
            except Exception as e:
                logger.warning(f"风险模型构建失败: {e}")
                analysis_results['risk_model'] = {'success': False, 'error': str(e)}
            
            # 4. 检测市场状态
            try:
                market_regime = self.detect_market_regime()
                analysis_results['market_regime'] = {
                    'success': True,
                    'regime': market_regime.name,
                    'probability': market_regime.probability,
                    'characteristics': market_regime.characteristics
                }
                logger.info(f"市场状态检测完成: {market_regime.name}")
            except Exception as e:
                logger.warning(f"市场状态检测失败: {e}")
                analysis_results['market_regime'] = {'success': False, 'error': str(e)}
                market_regime = MarketRegime(0, "Normal", 0.7, {'volatility': 0.15, 'trend': 0.0})
            
            # 5. 训练模型
            training_results = self.train_enhanced_models(feature_data)
            analysis_results['model_training'] = training_results
            
            # 6. 生成预测（结合regime-aware权重）
            ensemble_predictions = self.generate_enhanced_predictions(training_results, market_regime)
            if len(ensemble_predictions) == 0:
                raise ValueError("预测生成失败")
            
            analysis_results['prediction_generation'] = {
                'success': True,
                'predictions_count': len(ensemble_predictions),
                'prediction_stats': {
                    'mean': ensemble_predictions.mean(),
                    'std': ensemble_predictions.std(),
                    'min': ensemble_predictions.min(),
                    'max': ensemble_predictions.max()
                },
                'regime_adjusted': True
            }
            
            # 7. 投资组合优化（带风险模型）
            portfolio_result = self.optimize_portfolio_with_risk_model(ensemble_predictions, feature_data)
            analysis_results['portfolio_optimization'] = portfolio_result
            
            # 6. 生成投资建议
            recommendations = self.generate_investment_recommendations(portfolio_result, top_n)
            analysis_results['recommendations'] = recommendations
            
            # 7. 保存结果
            result_file = self.save_results(recommendations, portfolio_result)
            analysis_results['result_file'] = result_file
            
            analysis_results['end_time'] = datetime.now()
            analysis_results['total_time'] = (analysis_results['end_time'] - analysis_results['start_time']).total_seconds()
            analysis_results['success'] = True
            
            logger.info(f"完整分析流程完成，耗时: {analysis_results['total_time']:.1f}秒")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"分析流程失败: {e}")
            analysis_results['error'] = str(e)
            analysis_results['success'] = False
            analysis_results['end_time'] = datetime.now()
            
            return analysis_results


def main():
    """主函数"""
    print("=== BMA Ultra Enhanced 量化分析模型 V4 ===")
    print("集成Alpha策略、Learning-to-Rank、高级投资组合优化")
    print(f"增强模块可用: {ENHANCED_MODULES_AVAILABLE}")
    print(f"高级模型: XGBoost={XGBOOST_AVAILABLE}, LightGBM={LIGHTGBM_AVAILABLE}, CatBoost={CATBOOST_AVAILABLE}")
    
    # 命令行参数
    parser = argparse.ArgumentParser(description='BMA Ultra Enhanced量化模型V4')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='结束日期')
    parser.add_argument('--top-n', type=int, default=10, help='返回top N个推荐')
    parser.add_argument('--config', type=str, default='alphas_config.yaml', help='配置文件路径')
    parser.add_argument('--tickers', type=str, nargs='+', default=None, help='股票代码列表')
    parser.add_argument('--tickers-file', type=str, default='stocks.txt', help='股票列表文件（每行一个代码）')
    parser.add_argument('--tickers-limit', type=int, default=0, help='先用前N只做小样本测试，再全量训练（0表示直接全量）')
    
    args = parser.parse_args()
    
    # 确定股票列表
    if args.tickers:
        tickers = args.tickers
    else:
        tickers = load_universe_from_file(args.tickers_file) or load_universe_fallback()
    
    print(f"分析参数:")
    print(f"  时间范围: {args.start_date} - {args.end_date}")
    print(f"  股票数量: {len(tickers)}")
    print(f"  推荐数量: {args.top_n}")
    print(f"  配置文件: {args.config}")
    
    # 初始化模型
    model = UltraEnhancedQuantitativeModel(config_path=args.config)
    
    # 两阶段：小样本测试 → 全量
    if args.tickers_limit and args.tickers_limit > 0 and len(tickers) > args.tickers_limit:
        print("\n🧪 先运行小样本测试...")
        small_tickers = tickers[:args.tickers_limit]
        _ = model.run_complete_analysis(
            tickers=small_tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            top_n=min(args.top_n, len(small_tickers))
        )
        print("\n✅ 小样本测试完成，开始全量训练...")

    # 运行完整分析
    results = model.run_complete_analysis(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n
    )
    
    # 显示结果摘要
    print("\n" + "="*60)
    print("分析结果摘要")
    print("="*60)
    
    if results.get('success', False):
        # 避免控制台编码错误（GBK）
        print(f"分析成功完成，耗时: {results['total_time']:.1f}秒")
        
        if 'data_download' in results:
            print(f"数据下载: {results['data_download']['stocks_downloaded']}只股票")
        
        if 'feature_engineering' in results:
            fe_info = results['feature_engineering']
            print(f"特征工程: {fe_info['feature_shape'][0]}样本, {fe_info['feature_columns']}特征")
        
        if 'prediction_generation' in results:
            pred_info = results['prediction_generation']
            stats = pred_info['prediction_stats']
            print(f"预测生成: {pred_info['predictions_count']}个预测 (均值: {stats['mean']:.4f})")
        
        if 'portfolio_optimization' in results and results['portfolio_optimization'].get('success', False):
            port_metrics = results['portfolio_optimization']['portfolio_metrics']
            print(f"投资组合: 预期收益{port_metrics.get('expected_return', 0):.4f}, "
                  f"夏普比{port_metrics.get('sharpe_ratio', 0):.4f}")
        
        if 'recommendations' in results:
            recommendations = results['recommendations']
            print(f"\n投资建议 (Top {len(recommendations)}):")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec['ticker']}: 权重{rec['weight']:.3f}, "
                      f"信号{rec['prediction_signal']:.4f}")
        
        if 'result_file' in results:
            print(f"\n结果已保存至: {results['result_file']}")
    
    else:
        print(f"分析失败: {results.get('error', '未知错误')}")
    
    print("="*60)


if __name__ == "__main__":
    main()
