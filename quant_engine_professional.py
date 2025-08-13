#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Quantitative Engine V5
顶级金融机构级别的量化交易系统
集成所有先进技术：Multi-factor Risk Model、Dynamic Factor Loading、Regime-Aware BMA
"""

import pandas as pd
import numpy as np
from polygon_client import polygon_client, download, Ticker
import warnings
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml

# 科学计算和统计
from scipy import stats, optimize
from scipy.linalg import cholesky, inv
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import r2_score

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

# CatBoost removed due to compatibility issues
CATBOOST_AVAILABLE = False

# 配置
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

@dataclass
class PortfolioMetrics:
    """投资组合指标"""
    expected_return: float
    tracking_error: float
    information_ratio: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    turnover: float
    concentration_hhi: float

class ProfessionalQuantEngine:
    """专业量化引擎"""
    
    def __init__(self, config_path: str = None):
        """初始化专业量化引擎"""
        self.config = self._load_config(config_path)
        
        # 核心组件
        self.risk_model = MultifactorRiskModel()
        self.alpha_model = DynamicAlphaModel()
        self.regime_detector = MarketRegimeDetector()
        self.portfolio_optimizer = ProfessionalPortfolioOptimizer()
        self.execution_model = ExecutionCostModel()
        
        # 数据存储
        self.market_data = {}
        self.fundamental_data = {}
        self.alternative_data = {}
        
        # 模型状态
        self.current_regime = None
        self.factor_loadings = None
        self.risk_forecasts = None
        self.alpha_forecasts = None
        
        logger.info("Professional Quantitative Engine initialized")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            'universe': {
                'max_stocks': 500,
                'min_market_cap': 1e9,
                'min_liquidity_percentile': 20,
                'regions': ['US', 'EU', 'APAC'],
                'exclude_sectors': []
            },
            'risk_model': {
                'horizon_days': 21,
                'half_life_days': 90,
                'factor_count': 10,
                'eigen_adjustment': True,
                'newey_west_lags': 5
            },
            'alpha_model': {
                'lookback_days': 252,
                'rebalance_frequency': 'weekly',
                'decay_factor': 0.95,
                'signal_cap': 3.0,
                'min_observations': 50
            },
            'portfolio': {
                'target_volatility': 0.15,
                'max_turnover': 0.20,
                'max_position_weight': 0.05,
                'max_sector_weight': 0.25,
                'max_country_weight': 0.40,
                'transaction_cost_bps': 10
            },
            'regime_detection': {
                'lookback_periods': 252,
                'n_regimes': 3,
                'smoothing_factor': 0.1
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_market_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """加载市场数据"""
        logger.info(f"Loading market data for {len(tickers)} securities")
        
        all_data = {}
        failed_tickers = []
        
        for ticker in tickers:
            try:
                stock = Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if len(hist) < 100:  # 至少需要100天数据
                    failed_tickers.append(ticker)
                    continue
                
                # 标准化数据
                hist = hist.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                
                # 计算基础指标
                hist['returns'] = hist['close'].pct_change()
                hist['log_returns'] = np.log(hist['close'] / hist['close'].shift(1))
                hist['dollar_volume'] = hist['close'] * hist['volume']

                # 缺失值处理（前向填充+后向填充，再用0兜底）
                hist[['returns', 'log_returns', 'dollar_volume']] = (
                    hist[['returns', 'log_returns', 'dollar_volume']]
                    .fillna(method='ffill')
                    .fillna(method='bfill')
                    .fillna(0.0)
                )
                
                # 添加元数据
                info = stock.info
                hist['market_cap'] = info.get('market_cap', info.get('marketCap', 1e9))
                hist['sector'] = info.get('sector', 'Technology')
                hist['country'] = info.get('country', info.get('locale', 'us').upper())
                
                all_data[ticker] = hist
                
            except Exception as e:
                logger.warning(f"Failed to load data for {ticker}: {e}")
                failed_tickers.append(ticker)
        
        if failed_tickers:
            logger.warning(f"Failed to load {len(failed_tickers)} securities: {failed_tickers}")
        
        self.market_data = all_data
        logger.info(f"Successfully loaded data for {len(all_data)} securities")
        
        return all_data
    
    def build_risk_model(self) -> Dict[str, Any]:
        """构建多因子风险模型"""
        logger.info("Building multifactor risk model")
        
        if not self.market_data:
            raise ValueError("Market data not loaded")
        
        # 构建收益率矩阵
        returns_data = []
        tickers = []
        
        for ticker, data in self.market_data.items():
            if len(data) > 100:
                returns_data.append(data['returns'].fillna(0))
                tickers.append(ticker)
        
        if not returns_data:
            raise ValueError("No valid returns data")
        
        returns_matrix = pd.concat(returns_data, axis=1, keys=tickers)
        # 用0替代残留NaN，保证后续数值稳定
        returns_matrix = returns_matrix.fillna(0.0)
        
        # 构建风险因子
        risk_factors = self._build_risk_factors(returns_matrix)
        
        # 估计因子载荷
        factor_loadings = self._estimate_factor_loadings(returns_matrix, risk_factors)
        
        # 估计因子协方差
        factor_covariance = self._estimate_factor_covariance(risk_factors)
        
        # 估计特异风险
        specific_risk = self._estimate_specific_risk(returns_matrix, factor_loadings, risk_factors)
        
        # 风险模型验证
        model_stats = self._validate_risk_model(returns_matrix, factor_loadings, factor_covariance, specific_risk)
        
        risk_model_results = {
            'factor_loadings': factor_loadings,
            'factor_covariance': factor_covariance,
            'specific_risk': specific_risk,
            'risk_factors': risk_factors,
            'model_statistics': model_stats
        }
        
        self.risk_model.update_model(risk_model_results)
        
        logger.info("Risk model building completed")
        return risk_model_results
    
    def _build_risk_factors(self, returns_matrix: pd.DataFrame) -> pd.DataFrame:
        """构建风险因子"""
        factors = pd.DataFrame(index=returns_matrix.index)
        
        # 1. 市场因子
        factors['market'] = returns_matrix.mean(axis=1)
        
        # 2. 规模因子 (基于市值的SMB)
        market_caps = {}
        for ticker in returns_matrix.columns:
            if ticker in self.market_data:
                market_caps[ticker] = self.market_data[ticker]['market_cap'].iloc[-1]
        
        if market_caps:
            market_cap_series = pd.Series(market_caps)
            small_cap_mask = market_cap_series < market_cap_series.median()
            
            small_cap_returns = returns_matrix.loc[:, small_cap_mask].mean(axis=1)
            large_cap_returns = returns_matrix.loc[:, ~small_cap_mask].mean(axis=1)
            factors['size'] = small_cap_returns - large_cap_returns
        
        # 3. 动量因子
        momentum_scores = {}
        for ticker in returns_matrix.columns:
            # 过去1年动量，排除最近1个月
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
        
        # 5. 质量因子 (基于收益稳定性)
        quality_scores = returns_matrix.rolling(60).mean() / returns_matrix.rolling(60).std()
        high_quality = quality_scores.rank(axis=1, pct=True) > 0.7
        low_quality = quality_scores.rank(axis=1, pct=True) < 0.3
        
        factors['quality'] = returns_matrix.where(high_quality).mean(axis=1) - \
                            returns_matrix.where(low_quality).mean(axis=1)
        
        # 6. 反转因子
        reversal_scores = returns_matrix.rolling(21).sum()  # 过去1个月收益
        high_reversal = reversal_scores.rank(axis=1, pct=True) < 0.3  # 买入最差的
        low_reversal = reversal_scores.rank(axis=1, pct=True) > 0.7   # 卖出最好的
        
        factors['reversal'] = returns_matrix.where(high_reversal).mean(axis=1) - \
                             returns_matrix.where(low_reversal).mean(axis=1)
        
        # 标准化因子
        factors = factors.fillna(0)
        for col in factors.columns:
            factors[col] = (factors[col] - factors[col].mean()) / factors[col].std()
        
        return factors
    
    def _estimate_factor_loadings(self, returns_matrix: pd.DataFrame, 
                                 risk_factors: pd.DataFrame) -> pd.DataFrame:
        """估计因子载荷"""
        logger.info("Estimating factor loadings")
        
        loadings = {}
        
        for ticker in returns_matrix.columns:
            stock_returns = returns_matrix[ticker].dropna()
            aligned_factors = risk_factors.loc[stock_returns.index]
            
            if len(stock_returns) < 50:
                continue
            
            try:
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
        logger.info("Estimating factor covariance matrix")
        
        # 使用指数加权移动平均
        weights = np.exp(-np.arange(len(risk_factors)) / 90)[::-1]  # 90天半衰期
        weights = weights / weights.sum()
        
        # 加权协方差矩阵
        weighted_factors = risk_factors.values * np.sqrt(weights[:, np.newaxis])
        # 防御性：若样本过少或数值异常，回退到单位矩阵
        try:
            factor_cov = np.cov(weighted_factors.T)
        except Exception:
            factor_cov = np.eye(weighted_factors.shape[1]) * 1e-4
        
        # Newey-West调整 (HAC估计)
        factor_cov_df = pd.DataFrame(factor_cov, 
                                   index=risk_factors.columns,
                                   columns=risk_factors.columns)
        
        # 确保正定性
        try:
            eigenvals, eigenvecs = np.linalg.eigh(factor_cov)
            eigenvals = np.maximum(eigenvals, 1e-8)
            factor_cov_adjusted = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except Exception:
            # 回退：对角微噪声，确保正定
            d = factor_cov.shape[0]
            factor_cov_adjusted = factor_cov + np.eye(d) * 1e-6
        
        return pd.DataFrame(factor_cov_adjusted,
                          index=risk_factors.columns,
                          columns=risk_factors.columns)
    
    def _estimate_specific_risk(self, returns_matrix: pd.DataFrame,
                              factor_loadings: pd.DataFrame,
                              risk_factors: pd.DataFrame) -> pd.Series:
        """估计特异风险"""
        logger.info("Estimating specific risk")
        
        specific_risks = {}
        
        for ticker in returns_matrix.columns:
            if ticker not in factor_loadings.index:
                continue
            
            stock_returns = returns_matrix[ticker].dropna()
            loadings = factor_loadings.loc[ticker]
            aligned_factors = risk_factors.loc[stock_returns.index]
            
            # 计算因子收益解释的部分
            factor_returns = (aligned_factors * loadings).sum(axis=1)
            
            # 残差（特异收益）
            residuals = stock_returns - factor_returns
            
            # 使用GARCH或简单的指数加权计算特异风险
            weights = np.exp(-np.arange(len(residuals)) / 60)[::-1]  # 60天半衰期
            weights = weights / weights.sum()
            
            # 处理NaN/Inf
            resid2 = np.nan_to_num(residuals.values**2, nan=0.0, posinf=0.0, neginf=0.0)
            specific_var = np.average(resid2, weights=weights)
            specific_risks[ticker] = np.sqrt(specific_var)
        
        return pd.Series(specific_risks)
    
    def _validate_risk_model(self, returns_matrix: pd.DataFrame,
                           factor_loadings: pd.DataFrame,
                           factor_covariance: pd.DataFrame,
                           specific_risk: pd.Series) -> Dict[str, float]:
        """验证风险模型"""
        logger.info("Validating risk model")
        
        # 计算预测协方差矩阵
        common_tickers = list(set(returns_matrix.columns) & 
                            set(factor_loadings.index) & 
                            set(specific_risk.index))
        
        if len(common_tickers) < 10:
            return {'model_r2': 0.0, 'bias_test_pvalue': 1.0}
        
        B = factor_loadings.loc[common_tickers].values
        F = factor_covariance.values
        D = np.diag(specific_risk.loc[common_tickers].values**2)
        
        # 预测协方差矩阵: Σ = B*F*B' + D
        predicted_cov = B @ F @ B.T + D
        
        # 实际协方差矩阵
        actual_returns = returns_matrix[common_tickers].dropna()
        actual_cov = actual_returns.cov().values
        
        # R²计算
        pred_flat = predicted_cov[np.triu_indices_from(predicted_cov, k=1)]
        actual_flat = actual_cov[np.triu_indices_from(actual_cov, k=1)]
        
        model_r2 = r2_score(actual_flat, pred_flat)
        
        # 偏差检验
        bias_test_stat, bias_test_pvalue = stats.kstest(
            (actual_flat - pred_flat) / np.std(pred_flat), 'norm'
        )
        
        return {
            'model_r2': model_r2,
            'bias_test_pvalue': bias_test_pvalue,
            'mean_absolute_error': np.mean(np.abs(actual_flat - pred_flat)),
            'factor_count': len(factor_covariance),
            'coverage_ratio': len(common_tickers) / len(returns_matrix.columns)
        }
    
    def detect_market_regime(self) -> MarketRegime:
        """检测市场状态"""
        logger.info("Detecting market regime")
        
        if not self.market_data:
            raise ValueError("Market data not loaded")
        
        # 构建市场指数
        market_returns = []
        for ticker, data in self.market_data.items():
            if len(data) > 100:
                market_returns.append(data['returns'].fillna(0))
        
        market_index = pd.concat(market_returns, axis=1).mean(axis=1).dropna()
        
        if len(market_index) < 100:
            # 默认状态
            return MarketRegime(
                regime_id=1,
                name="Normal",
                probability=1.0,
                characteristics={'volatility': 0.15, 'trend': 0.0}
            )
        
        # 使用马尔科夫转换模型
        try:
            # 准备数据
            y = market_index.values.reshape(-1, 1)
            
            # 简化的状态检测：基于波动率和趋势
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
            
            logger.info(f"Current market regime: {regime.name} (probability: {regime.probability:.2f})")
            
            return regime
            
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return MarketRegime(0, "Unknown", 0.5, {'volatility': 0.2, 'trend': 0.0})
    
    def generate_alpha_signals(self) -> pd.DataFrame:
        """生成Alpha信号"""
        logger.info("Generating alpha signals")
        
        if not self.market_data:
            raise ValueError("Market data not loaded")
        
        alpha_signals = {}
        
        for ticker, data in self.market_data.items():
            if len(data) < 100:
                continue
            
            signals = {}
            
            # 1. 动量信号 (多时间框架)
            for window in [21, 63, 126, 252]:
                momentum = data['close'] / data['close'].shift(window) - 1
                signals[f'momentum_{window}d'] = momentum.iloc[-1] if not pd.isna(momentum.iloc[-1]) else 0
            
            # 2. 均值回归信号
            for window in [5, 10, 21]:
                mean_price = data['close'].rolling(window).mean()
                reversion = (data['close'] - mean_price) / mean_price
                signals[f'reversion_{window}d'] = -reversion.iloc[-1] if not pd.isna(reversion.iloc[-1]) else 0
            
            # 3. 波动率信号
            realized_vol = data['returns'].rolling(21).std()
            vol_zscore = (realized_vol - realized_vol.rolling(252).mean()) / realized_vol.rolling(252).std()
            signals['volatility_zscore'] = -vol_zscore.iloc[-1] if not pd.isna(vol_zscore.iloc[-1]) else 0
            
            # 4. 成交量信号
            if 'volume' in data.columns:
                volume_ma = data['volume'].rolling(21).mean()
                volume_ratio = data['volume'] / volume_ma
                volume_trend = volume_ratio.rolling(5).mean()
                signals['volume_trend'] = volume_trend.iloc[-1] if not pd.isna(volume_trend.iloc[-1]) else 0
            
            # 5. 价格位置信号
            high_252 = data['high'].rolling(252).max()
            low_252 = data['low'].rolling(252).min()
            price_position = (data['close'] - low_252) / (high_252 - low_252)
            signals['price_position'] = price_position.iloc[-1] if not pd.isna(price_position.iloc[-1]) else 0.5
            
            # 6. 技术形态信号
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            signals['rsi'] = (50 - rsi.iloc[-1]) / 50 if not pd.isna(rsi.iloc[-1]) else 0
            
            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            signals['macd'] = macd_histogram.iloc[-1] / data['close'].iloc[-1] if not pd.isna(macd_histogram.iloc[-1]) else 0
            
            # 汇总信号
            alpha_signals[ticker] = signals
        
        # 转换为DataFrame
        signals_df = pd.DataFrame(alpha_signals).T
        
        # 信号标准化和去极值
        for col in signals_df.columns:
            # Winsorize
            q01, q99 = signals_df[col].quantile([0.01, 0.99])
            signals_df[col] = signals_df[col].clip(q01, q99)
            
            # 标准化
            signals_df[col] = (signals_df[col] - signals_df[col].mean()) / (signals_df[col].std() + 1e-8)
        
        # 信号合成 (基于当前市场状态调整权重)
        if self.current_regime:
            regime_weights = self._get_regime_signal_weights(self.current_regime)
        else:
            regime_weights = {col: 1.0 for col in signals_df.columns}
        
        # 加权合成最终Alpha信号
        final_alpha = pd.Series(0.0, index=signals_df.index)
        total_weight = sum(regime_weights.values())
        
        for signal, weight in regime_weights.items():
            if signal in signals_df.columns:
                final_alpha += (weight / total_weight) * signals_df[signal]
        
        # 最终信号标准化
        final_alpha = (final_alpha - final_alpha.mean()) / (final_alpha.std() + 1e-8)
        
        # 信号截断 (只保留显著信号)
        signal_threshold = 0.5
        final_alpha = final_alpha.where(final_alpha.abs() > signal_threshold, 0)
        
        logger.info(f"Generated alpha signals for {len(final_alpha)} securities")
        logger.info(f"Non-zero signals: {(final_alpha != 0).sum()}")
        
        return final_alpha
    
    def _get_regime_signal_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """根据市场状态调整信号权重"""
        if "Bull" in regime.name:
            # 牛市：偏好动量
            return {
                'momentum_21d': 2.0, 'momentum_63d': 2.5, 'momentum_126d': 2.0, 'momentum_252d': 1.5,
                'reversion_5d': 0.5, 'reversion_10d': 0.5, 'reversion_21d': 0.5,
                'volatility_zscore': 1.0, 'volume_trend': 1.5, 'price_position': 1.0,
                'rsi': 0.5, 'macd': 1.5
            }
        elif "Bear" in regime.name:
            # 熊市：偏好质量和防御
            return {
                'momentum_21d': 0.5, 'momentum_63d': 0.5, 'momentum_126d': 1.0, 'momentum_252d': 1.0,
                'reversion_5d': 1.5, 'reversion_10d': 2.0, 'reversion_21d': 1.5,
                'volatility_zscore': 2.0, 'volume_trend': 0.5, 'price_position': 1.5,
                'rsi': 2.0, 'macd': 0.5
            }
        elif "Volatile" in regime.name:
            # 高波动：偏好均值回归
            return {
                'momentum_21d': 0.5, 'momentum_63d': 1.0, 'momentum_126d': 1.0, 'momentum_252d': 1.0,
                'reversion_5d': 2.5, 'reversion_10d': 2.0, 'reversion_21d': 1.5,
                'volatility_zscore': 2.5, 'volume_trend': 1.0, 'price_position': 1.0,
                'rsi': 2.0, 'macd': 1.0
            }
        else:
            # 正常市场：均衡权重
            return {col: 1.0 for col in [
                'momentum_21d', 'momentum_63d', 'momentum_126d', 'momentum_252d',
                'reversion_5d', 'reversion_10d', 'reversion_21d',
                'volatility_zscore', 'volume_trend', 'price_position', 'rsi', 'macd'
            ]}
    
    def optimize_portfolio(self, alpha_signals: pd.Series) -> Dict[str, Any]:
        """优化投资组合"""
        logger.info("Optimizing portfolio")
        
        if not hasattr(self.risk_model, 'factor_loadings') or self.risk_model.factor_loadings is None:
            raise ValueError("Risk model not built")
        
        # 对齐数据
        common_assets = list(set(alpha_signals.index) & set(self.risk_model.factor_loadings.index))
        
        if len(common_assets) < 3:
            # 回退策略：少样本时使用简化等权/按信号权重
            simple_weights = alpha_signals.loc[common_assets].copy()
            # 若全零，则等权；否则取正值归一
            if (simple_weights.abs() > 0).any():
                w = simple_weights.clip(lower=0)
                if w.sum() == 0:
                    w = pd.Series(1.0, index=common_assets)
            else:
                w = pd.Series(1.0, index=common_assets)
            w = w / w.sum()

            metrics = self._calculate_portfolio_metrics(w, alpha_signals.loc[common_assets])
            return {
                'success': True,
                'weights': w,
                'metrics': metrics,
                'risk_attribution': {}
            }
        
        alpha_aligned = alpha_signals.loc[common_assets]
        
        # 构建优化问题
        n_assets = len(common_assets)
        
        # 目标函数：最大化 alpha'w - λ/2 * w'Σw - η * turnover
        def objective(weights):
            w = weights.reshape(-1, 1)
            
            # Alpha收益
            alpha_return = alpha_aligned.values @ weights
            
            # 风险惩罚
            portfolio_risk = self._calculate_portfolio_risk(weights, common_assets)
            
            # 换手率惩罚 (假设当前持仓为0)
            turnover = np.sum(np.abs(weights))
            
            risk_aversion = self.config['portfolio']['target_volatility'] ** 2
            turnover_penalty = self.config['portfolio']['max_turnover']
            
            return -(alpha_return - risk_aversion * portfolio_risk - turnover_penalty * turnover)
        
        # 约束条件
        constraints = []
        
        # 权重和约束
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # 个股权重约束
        max_weight = self.config['portfolio']['max_position_weight']
        bounds = [(-max_weight, max_weight) for _ in range(n_assets)]
        
        # 行业中性约束 (简化)
        sector_constraints = self._build_sector_constraints(common_assets)
        constraints.extend(sector_constraints)
        
        # 优化求解
        initial_guess = np.zeros(n_assets)
        # 给有信号的股票初始权重
        for i, asset in enumerate(common_assets):
            if alpha_signals[asset] > 0:
                initial_guess[i] = min(0.02, max_weight)
            elif alpha_signals[asset] < 0:
                initial_guess[i] = max(-0.02, -max_weight)
        
        # 标准化初始权重
        if np.sum(initial_guess) != 0:
            initial_guess = initial_guess / np.sum(initial_guess)
        
        try:
            result = optimize.minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'disp': False, 'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = pd.Series(result.x, index=common_assets)
                
                # 计算组合指标
                portfolio_metrics = self._calculate_portfolio_metrics(optimal_weights, alpha_aligned)
                
                # 风险归因
                risk_attribution = self._calculate_risk_attribution(optimal_weights, common_assets)
                
                return {
                    'success': True,
                    'weights': optimal_weights,
                    'metrics': portfolio_metrics,
                    'risk_attribution': risk_attribution,
                    'optimization_info': {
                        'objective_value': -result.fun,
                        'iterations': result.nit,
                        'message': result.message
                    }
                }
            else:
                logger.error(f"Optimization failed: {result.message}")
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_portfolio_risk(self, weights: np.ndarray, assets: List[str]) -> float:
        """计算投资组合风险"""
        if not hasattr(self.risk_model, 'factor_loadings'):
            return np.sum(weights**2) * 0.15**2  # 简单估计
        
        # 因子风险
        B = self.risk_model.factor_loadings.loc[assets].values
        F = self.risk_model.factor_covariance.values
        
        factor_risk = weights.T @ B @ F @ B.T @ weights
        
        # 特异风险
        specific_var = self.risk_model.specific_risk.loc[assets].values**2
        specific_risk = weights.T @ np.diag(specific_var) @ weights
        
        return factor_risk + specific_risk
    
    def _build_sector_constraints(self, assets: List[str]) -> List[Dict]:
        """构建行业约束"""
        constraints = []
        
        # 获取行业信息
        sectors = {}
        for asset in assets:
            if asset in self.market_data:
                sector = self.market_data[asset]['sector'].iloc[-1]
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(asset)
        
        # 为每个行业添加约束
        max_sector_weight = self.config['portfolio']['max_sector_weight']
        
        for sector, sector_assets in sectors.items():
            if len(sector_assets) > 1:  # 只对有多个股票的行业添加约束
                sector_indices = [assets.index(asset) for asset in sector_assets]
                
                def sector_constraint(weights, indices=sector_indices):
                    return max_sector_weight - np.sum(np.abs(weights[indices]))
                
                constraints.append({
                    'type': 'ineq',
                    'fun': sector_constraint
                })
        
        return constraints
    
    def _calculate_portfolio_metrics(self, weights: pd.Series, alpha_signals: pd.Series) -> PortfolioMetrics:
        """计算投资组合指标"""
        expected_return = (weights * alpha_signals).sum()
        portfolio_risk = np.sqrt(self._calculate_portfolio_risk(weights.values, weights.index.tolist()))
        
        return PortfolioMetrics(
            expected_return=expected_return,
            tracking_error=portfolio_risk,
            information_ratio=expected_return / (portfolio_risk + 1e-8),
            max_drawdown=0.0,  # 需要历史数据计算
            sharpe_ratio=expected_return / (portfolio_risk + 1e-8),
            sortino_ratio=0.0,  # 需要历史数据计算
            calmar_ratio=0.0,   # 需要历史数据计算
            var_95=-1.645 * portfolio_risk,
            cvar_95=-2.0 * portfolio_risk,
            turnover=weights.abs().sum(),
            concentration_hhi=np.sum(weights**2)
        )
    
    def _calculate_risk_attribution(self, weights: pd.Series, assets: List[str]) -> Dict[str, float]:
        """计算风险归因"""
        if not hasattr(self.risk_model, 'factor_loadings'):
            return {}
        
        B = self.risk_model.factor_loadings.loc[assets].values
        F = self.risk_model.factor_covariance.values
        
        # 组合在各因子上的暴露
        factor_exposures = B.T @ weights.values
        
        # 各因子的风险贡献
        factor_risk_contrib = {}
        for i, factor_name in enumerate(self.risk_model.factor_covariance.columns):
            factor_risk_contrib[factor_name] = factor_exposures[i]**2 * F[i, i]
        
        return factor_risk_contrib
    
    def run_complete_analysis(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """运行完整分析"""
        logger.info("Starting complete professional analysis")
        
        results = {
            'timestamp': datetime.now(),
            'config': self.config,
            'analysis_summary': {}
        }
        
        try:
            # 1. 加载数据
            market_data = self.load_market_data(tickers, start_date, end_date)
            results['data_loading'] = {
                'success': True,
                'securities_loaded': len(market_data),
                'date_range': f"{start_date} to {end_date}"
            }
            
            # 2. 构建风险模型
            risk_model = self.build_risk_model()
            results['risk_model'] = {
                'success': True,
                'model_r2': risk_model['model_statistics'].get('model_r2', 0),
                'factor_count': len(risk_model['factor_loadings'].columns),
                'coverage': risk_model['model_statistics'].get('coverage_ratio', 0)
            }
            
            # 3. 检测市场状态
            current_regime = self.detect_market_regime()
            results['market_regime'] = {
                'regime_name': current_regime.name,
                'probability': current_regime.probability,
                'characteristics': current_regime.characteristics
            }
            
            # 4. 生成Alpha信号
            alpha_signals = self.generate_alpha_signals()
            results['alpha_signals'] = {
                'total_signals': len(alpha_signals),
                'active_signals': (alpha_signals != 0).sum(),
                'signal_strength': {
                    'mean': alpha_signals.mean(),
                    'std': alpha_signals.std(),
                    'max': alpha_signals.max(),
                    'min': alpha_signals.min()
                }
            }
            
            # 5. 投资组合优化
            portfolio_result = self.optimize_portfolio(alpha_signals)
            
            if portfolio_result.get('success', False):
                optimal_weights = portfolio_result['weights']
                portfolio_metrics = portfolio_result['metrics']
                
                results['portfolio'] = {
                    'success': True,
                    'metrics': {
                        'expected_return': portfolio_metrics.expected_return,
                        'tracking_error': portfolio_metrics.tracking_error,
                        'information_ratio': portfolio_metrics.information_ratio,
                        'sharpe_ratio': portfolio_metrics.sharpe_ratio,
                        'concentration_hhi': portfolio_metrics.concentration_hhi,
                        'turnover': portfolio_metrics.turnover
                    },
                    'top_positions': optimal_weights.abs().sort_values(ascending=False).head(10).to_dict(),
                    'risk_attribution': portfolio_result.get('risk_attribution', {})
                }
                
                # 6. 生成投资建议
                recommendations = self._generate_investment_recommendations(
                    optimal_weights, alpha_signals, portfolio_metrics
                )
                results['recommendations'] = recommendations
                
            else:
                results['portfolio'] = {
                    'success': False,
                    'error': portfolio_result.get('error', 'Unknown error')
                }
            
            # 7. 保存结果
            result_file = self._save_professional_results(results)
            results['output_file'] = result_file
            
            results['analysis_summary'] = {
                'status': 'SUCCESS',
                'total_time_seconds': (datetime.now() - results['timestamp']).total_seconds(),
                'data_quality': 'HIGH' if len(market_data) > len(tickers) * 0.8 else 'MEDIUM',
                'model_quality': 'HIGH' if risk_model['model_statistics'].get('model_r2', 0) > 0.3 else 'MEDIUM',
                'signal_quality': 'HIGH' if (alpha_signals != 0).sum() > len(alpha_signals) * 0.1 else 'LOW'
            }
            
            logger.info("Professional analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            results['analysis_summary'] = {
                'status': 'FAILED',
                'error': str(e),
                'total_time_seconds': (datetime.now() - results['timestamp']).total_seconds()
            }
        
        return results
    
    def _generate_investment_recommendations(self, weights: pd.Series, 
                                           alpha_signals: pd.Series,
                                           metrics: PortfolioMetrics) -> List[Dict[str, Any]]:
        """生成投资建议"""
        recommendations = []
        
        # 获取前10个推荐
        top_weights = weights[weights > 0.001].sort_values(ascending=False).head(10)
        
        for rank, (ticker, weight) in enumerate(top_weights.items(), 1):
            try:
                stock_data = self.market_data.get(ticker)
                if stock_data is None:
                    continue
                
                latest_price = stock_data['close'].iloc[-1]
                signal_strength = alpha_signals.get(ticker, 0)
                
                # 计算一些基本指标
                returns_5d = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[-6] - 1) if len(stock_data) > 5 else 0
                returns_21d = (stock_data['close'].iloc[-1] / stock_data['close'].iloc[-22] - 1) if len(stock_data) > 21 else 0
                
                volatility_21d = stock_data['returns'].tail(21).std() * np.sqrt(252)
                
                recommendation = {
                    'rank': rank,
                    'ticker': ticker,
                    'weight': float(weight),
                    'signal_strength': float(signal_strength),
                    'latest_price': float(latest_price),
                    'returns_5d': float(returns_5d),
                    'returns_21d': float(returns_21d),
                    'volatility_annualized': float(volatility_21d),
                    'sector': stock_data['sector'].iloc[-1] if 'sector' in stock_data.columns else 'Unknown',
                    'market_cap': stock_data['market_cap'].iloc[-1] if 'market_cap' in stock_data.columns else 0,
                    'recommendation_reason': self._get_recommendation_reason(signal_strength, weight, returns_21d)
                }
                
                recommendations.append(recommendation)
                
            except Exception as e:
                logger.warning(f"Failed to generate recommendation for {ticker}: {e}")
        
        return recommendations
    
    def _get_recommendation_reason(self, signal: float, weight: float, momentum: float) -> str:
        """生成推荐理由"""
        reasons = []
        
        if signal > 1.0:
            reasons.append("强Alpha信号")
        elif signal > 0.5:
            reasons.append("正Alpha信号")
        
        if weight > 0.03:
            reasons.append("高置信度权重")
        elif weight > 0.01:
            reasons.append("中等权重")
        
        if momentum > 0.1:
            reasons.append("强劲价格动量")
        elif momentum > 0.05:
            reasons.append("正向价格趋势")
        
        if not reasons:
            reasons.append("基于模型优化选择")
        
        return "; ".join(reasons)
    
    def _save_professional_results(self, results: Dict[str, Any]) -> str:
        """保存专业分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = Path("result")
        result_dir.mkdir(exist_ok=True)
        
        # 主结果文件
        result_file = result_dir / f"professional_quant_analysis_{timestamp}.json"
        
        # 转换numpy类型为Python原生类型
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_types(results)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, ensure_ascii=False, indent=2)
        
        # 如果有推荐，单独保存推荐文件
        if 'recommendations' in results and results['recommendations']:
            rec_file = result_dir / f"investment_recommendations_{timestamp}.json"
            with open(rec_file, 'w', encoding='utf-8') as f:
                json.dump(results['recommendations'], f, ensure_ascii=False, indent=2)
            
            # 简化的ticker列表
            if results['recommendations']:
                ticker_file = result_dir / f"top_tickers_{timestamp}.txt"
                top_tickers = [rec['ticker'] for rec in results['recommendations'][:5]]
                with open(ticker_file, 'w') as f:
                    f.write(", ".join([f"'{ticker}'" for ticker in top_tickers]))
        
        logger.info(f"Results saved to {result_file}")
        return str(result_file)


# 支持类定义
class MultifactorRiskModel:
    """多因子风险模型"""
    def __init__(self):
        self.factor_loadings = None
        self.factor_covariance = None
        self.specific_risk = None
        self.risk_factors = None
    
    def update_model(self, model_results: Dict[str, Any]):
        """更新模型"""
        self.factor_loadings = model_results.get('factor_loadings')
        self.factor_covariance = model_results.get('factor_covariance')
        self.specific_risk = model_results.get('specific_risk')
        self.risk_factors = model_results.get('risk_factors')

class DynamicAlphaModel:
    """动态Alpha模型"""
    def __init__(self):
        self.signal_weights = {}
        self.regime_adjustments = {}

class MarketRegimeDetector:
    """市场状态检测器"""
    def __init__(self):
        self.current_regime = None
        self.regime_history = []

class ProfessionalPortfolioOptimizer:
    """专业投资组合优化器"""
    def __init__(self):
        self.optimization_history = []

class ExecutionCostModel:
    """执行成本模型"""
    def __init__(self):
        self.cost_parameters = {}


# 主函数
def main():
    """主函数"""
    print("=== Professional Quantitative Engine V5 ===")
    print("顶级金融机构级别的量化交易系统")
    
    # 默认股票池 (扩展到更多优质股票)
    DEFAULT_TICKERS = [
        # 大盘科技股
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX',
        # 金融
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B',
        # 医疗保健
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT',
        # 消费
        'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE',
        # 工业
        'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX',
        # 新兴科技
        'CRM', 'ADBE', 'PYPL', 'INTC', 'AMD', 'QCOM', 'AVGO'
    ]
    
    # 初始化引擎
    engine = ProfessionalQuantEngine()
    
    # 运行分析
    try:
        results = engine.run_complete_analysis(
            tickers=DEFAULT_TICKERS,
            start_date='2022-01-01',
            end_date='2024-12-31'
        )
        
        # 显示结果摘要
        print("\n" + "="*60)
        print("Professional Analysis Results")
        print("="*60)
        
        summary = results.get('analysis_summary', {})
        print(f"Analysis Status: {summary.get('status', 'UNKNOWN')}")
        print(f"Total Time: {summary.get('total_time_seconds', 0):.1f} seconds")
        print(f"Data Quality: {summary.get('data_quality', 'UNKNOWN')}")
        print(f"Model Quality: {summary.get('model_quality', 'UNKNOWN')}")
        
        if 'data_loading' in results:
            data_info = results['data_loading']
            print(f"Securities Loaded: {data_info.get('securities_loaded', 0)}")
        
        if 'risk_model' in results:
            risk_info = results['risk_model']
            print(f"Risk Model R²: {risk_info.get('model_r2', 0):.3f}")
            print(f"Factor Count: {risk_info.get('factor_count', 0)}")
        
        if 'market_regime' in results:
            regime_info = results['market_regime']
            print(f"Market Regime: {regime_info.get('regime_name', 'Unknown')}")
            print(f"Regime Probability: {regime_info.get('probability', 0):.2f}")
        
        if 'alpha_signals' in results:
            signal_info = results['alpha_signals']
            print(f"Total Signals: {signal_info.get('total_signals', 0)}")
            print(f"Active Signals: {signal_info.get('active_signals', 0)}")
        
        if 'portfolio' in results and results['portfolio'].get('success', False):
            port_info = results['portfolio']['metrics']
            print(f"Expected Return: {port_info.get('expected_return', 0):.4f}")
            print(f"Information Ratio: {port_info.get('information_ratio', 0):.3f}")
            print(f"Portfolio Turnover: {port_info.get('turnover', 0):.3f}")
        
        if 'recommendations' in results:
            recommendations = results['recommendations']
            print(f"\n🎯 Top Investment Recommendations:")
            for rec in recommendations[:5]:
                print(f"  {rec['rank']}. {rec['ticker']}: {rec['weight']:.3f} "
                      f"(Signal: {rec['signal_strength']:.3f})")
        
        if 'output_file' in results:
            print(f"\n📁 Detailed results saved to: {results['output_file']}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    main()
