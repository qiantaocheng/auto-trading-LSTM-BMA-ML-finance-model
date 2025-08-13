#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regime-Aware Adaptive Weight Engine
动态状态感知的Alpha/ML权重自适应系统
"""

import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import savgol_filter
import yfinance as yf

logger = logging.getLogger(__name__)

@dataclass
class RegimeState:
    """市场状态定义"""
    regime_id: int
    name: str
    probability: float
    characteristics: Dict[str, float]
    alpha_weight: float  # Alpha因子权重
    ml_weight: float     # ML权重
    cash_weight: float   # 现金权重
    
@dataclass  
class UncertaintyMetrics:
    """不确定性指标"""
    prediction_uncertainty: float
    regime_uncertainty: float
    volatility_uncertainty: float
    combined_uncertainty: float

class RegimeDetector:
    """市场状态检测器"""
    
    def __init__(self, n_regimes: int = 3, lookback_window: int = 252):
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.regime_history = []
        
    def fit_regime_model(self, returns: pd.Series, volatility: pd.Series, 
                        volume: pd.Series = None) -> None:
        """拟合隐马尔可夫模型检测市场状态"""
        try:
            # 构建特征矩阵
            features = []
            
            # 1. 收益率特征
            features.append(returns.values)
            features.append(returns.rolling(5).mean().values)  # 短期趋势
            features.append(returns.rolling(21).mean().values)  # 中期趋势
            
            # 2. 波动率特征  
            features.append(volatility.values)
            features.append(volatility.rolling(10).mean().values)
            
            # 3. 量价特征
            if volume is not None:
                volume_ma = volume.rolling(21).mean()
                volume_ratio = volume / volume_ma
                features.append(volume_ratio.fillna(1.0).values)
            
            # 4. 技术指标
            # RSI
            price_changes = returns.cumsum()
            rsi = self._calculate_rsi(price_changes, 14)
            features.append(rsi.values)
            
            # 波动率相对位置
            vol_percentile = volatility.rolling(63).rank(pct=True)
            features.append(vol_percentile.fillna(0.5).values)
            
            # 构建特征矩阵并去除NaN
            feature_matrix = np.column_stack(features)
            valid_mask = ~np.isnan(feature_matrix).any(axis=1)
            clean_features = feature_matrix[valid_mask]
            
            # 标准化特征
            scaled_features = self.scaler.fit_transform(clean_features)
            
            # 拟合高斯混合模型作为HMM的近似
            self.hmm_model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42,
                max_iter=200
            )
            
            self.hmm_model.fit(scaled_features)
            
            # 预测状态序列
            regime_probs = self.hmm_model.predict_proba(scaled_features)
            regime_labels = self.hmm_model.predict(scaled_features)
            
            # 存储状态历史
            self.regime_history = {
                'probabilities': regime_probs,
                'labels': regime_labels,
                'valid_mask': valid_mask,
                'dates': returns.index[valid_mask]
            }
            
            logger.info(f"市场状态模型拟合完成，识别出{self.n_regimes}个状态")
            
        except Exception as e:
            logger.error(f"市场状态模型拟合失败: {e}")
            self.hmm_model = None
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def predict_current_regime(self, recent_data: Dict[str, pd.Series]) -> Tuple[int, np.ndarray]:
        """预测当前市场状态"""
        if self.hmm_model is None:
            return 0, np.array([1.0, 0.0, 0.0])[:self.n_regimes]
        
        try:
            # 构建当前特征
            returns = recent_data['returns'].tail(1)
            volatility = recent_data['volatility'].tail(1)
            volume = recent_data.get('volume')
            
            features = []
            features.append(returns.values[0])
            features.append(recent_data['returns'].tail(5).mean())
            features.append(recent_data['returns'].tail(21).mean())
            features.append(volatility.values[0])
            features.append(recent_data['volatility'].tail(10).mean())
            
            if volume is not None:
                volume_ma = volume.tail(21).mean()
                volume_ratio = volume.tail(1).values[0] / volume_ma
                features.append(volume_ratio)
            
            # RSI
            price_changes = recent_data['returns'].cumsum()
            rsi = self._calculate_rsi(price_changes, 14).tail(1).values[0]
            features.append(rsi)
            
            # 波动率分位数
            vol_percentile = recent_data['volatility'].tail(63).rank(pct=True).tail(1).values[0]
            features.append(vol_percentile)
            
            # 标准化并预测
            feature_vector = np.array(features).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_vector)
            
            regime_probs = self.hmm_model.predict_proba(scaled_features)[0]
            regime_label = self.hmm_model.predict(scaled_features)[0]
            
            return regime_label, regime_probs
            
        except Exception as e:
            logger.warning(f"状态预测失败: {e}")
            return 0, np.array([1.0, 0.0, 0.0])[:self.n_regimes]

class AdaptiveWeightEngine:
    """自适应权重引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.regime_detector = RegimeDetector(
            n_regimes=self.config['regimes']['n_regimes'],
            lookback_window=self.config['regimes']['lookback_window']
        )
        
        # 状态定义
        self.regime_states = self._define_regime_states()
        
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'regimes': {
                'n_regimes': 3,
                'lookback_window': 252,
                'update_frequency': 5  # 每5天更新一次状态
            },
            'weights': {
                'base_alpha_weight': 0.3,
                'base_ml_weight': 0.7,
                'max_cash_weight': 0.3,
                'volatility_threshold': 0.02,  # 2%日波动率阈值
                'uncertainty_threshold': 0.8
            },
            'adaptation': {
                'momentum_boost': 1.5,    # 牛市动量增强系数
                'uncertainty_decay': 0.7,  # 不确定性下的权重衰减
                'cash_preference': 0.2     # 高不确定性时的现金偏好
            }
        }
    
    def _define_regime_states(self) -> Dict[int, RegimeState]:
        """定义市场状态"""
        return {
            0: RegimeState(  # 牛市/上涨趋势
                regime_id=0,
                name="Bull Market",
                probability=0.0,
                characteristics={
                    'trend': 'positive',
                    'volatility': 'low_to_medium',
                    'momentum': 'strong'
                },
                alpha_weight=0.45,  # 提高Alpha权重
                ml_weight=0.55,
                cash_weight=0.0
            ),
            1: RegimeState(  # 熊市/下跌趋势
                regime_id=1, 
                name="Bear Market",
                probability=0.0,
                characteristics={
                    'trend': 'negative',
                    'volatility': 'medium_to_high',
                    'momentum': 'weak'
                },
                alpha_weight=0.2,   # 降低风险
                ml_weight=0.6,
                cash_weight=0.2
            ),
            2: RegimeState(  # 震荡市/高波动
                regime_id=2,
                name="Volatile/Sideways",
                probability=0.0,
                characteristics={
                    'trend': 'neutral',
                    'volatility': 'high',
                    'momentum': 'unstable'
                },
                alpha_weight=0.25,
                ml_weight=0.45,
                cash_weight=0.3
            )
        }
    
    def compute_uncertainty_metrics(self, 
                                  prediction_std: float,
                                  regime_probs: np.ndarray,
                                  recent_volatility: float) -> UncertaintyMetrics:
        """计算综合不确定性指标"""
        
        # 1. 预测不确定性（基于模型预测方差）
        prediction_uncertainty = min(prediction_std / 0.05, 1.0)  # 标准化到[0,1]
        
        # 2. 状态不确定性（基于状态概率分布的熵）
        regime_entropy = -np.sum(regime_probs * np.log(regime_probs + 1e-8))
        max_entropy = np.log(len(regime_probs))
        regime_uncertainty = regime_entropy / max_entropy
        
        # 3. 波动率不确定性
        volatility_threshold = self.config['weights']['volatility_threshold']
        volatility_uncertainty = min(recent_volatility / (2 * volatility_threshold), 1.0)
        
        # 4. 综合不确定性（加权平均）
        combined_uncertainty = (
            0.4 * prediction_uncertainty +
            0.3 * regime_uncertainty +
            0.3 * volatility_uncertainty
        )
        
        return UncertaintyMetrics(
            prediction_uncertainty=prediction_uncertainty,
            regime_uncertainty=regime_uncertainty,
            volatility_uncertainty=volatility_uncertainty,
            combined_uncertainty=combined_uncertainty
        )
    
    def adaptive_weight_allocation(self,
                                 market_data: Dict[str, pd.Series],
                                 prediction_uncertainty: float,
                                 current_signals: Dict[str, float]) -> Dict[str, float]:
        """自适应权重分配主函数"""
        
        try:
            # 1. 检测当前市场状态
            regime_label, regime_probs = self.regime_detector.predict_current_regime(market_data)
            
            # 2. 更新状态概率
            current_regime = self.regime_states[regime_label].copy()
            current_regime.probability = regime_probs[regime_label]
            
            # 3. 计算不确定性指标
            recent_volatility = market_data['volatility'].tail(10).mean()
            uncertainty_metrics = self.compute_uncertainty_metrics(
                prediction_uncertainty, regime_probs, recent_volatility
            )
            
            # 4. 基于状态和不确定性调整权重
            adapted_weights = self._adapt_weights_by_regime_and_uncertainty(
                current_regime, uncertainty_metrics, current_signals
            )
            
            # 5. 记录决策信息
            logger.info(f"当前市场状态: {current_regime.name} (概率: {current_regime.probability:.3f})")
            logger.info(f"综合不确定性: {uncertainty_metrics.combined_uncertainty:.3f}")
            logger.info(f"自适应权重 - Alpha: {adapted_weights['alpha']:.3f}, "
                       f"ML: {adapted_weights['ml']:.3f}, Cash: {adapted_weights['cash']:.3f}")
            
            return adapted_weights
            
        except Exception as e:
            logger.error(f"自适应权重分配失败: {e}")
            # 返回默认权重
            return {
                'alpha': self.config['weights']['base_alpha_weight'],
                'ml': self.config['weights']['base_ml_weight'],
                'cash': 0.0
            }
    
    def _adapt_weights_by_regime_and_uncertainty(self,
                                               regime: RegimeState,
                                               uncertainty: UncertaintyMetrics,
                                               signals: Dict[str, float]) -> Dict[str, float]:
        """基于状态和不确定性调整权重"""
        
        # 基础权重从状态开始
        alpha_weight = regime.alpha_weight
        ml_weight = regime.ml_weight
        cash_weight = regime.cash_weight
        
        # === 状态特定调整 ===
        if regime.regime_id == 0:  # 牛市
            # 动量信号强时增强Alpha权重
            momentum_signal = signals.get('momentum', 0.0)
            if momentum_signal > 0.3:  # 强动量
                boost_factor = self.config['adaptation']['momentum_boost']
                alpha_weight = min(alpha_weight * boost_factor, 0.6)
                ml_weight = 1.0 - alpha_weight - cash_weight
                
        elif regime.regime_id == 1:  # 熊市
            # 熊市中偏向保守，增加现金权重
            if uncertainty.combined_uncertainty > 0.6:
                cash_weight = min(cash_weight + 0.1, 0.4)
                
        elif regime.regime_id == 2:  # 震荡市
            # 震荡市中根据波动率调整
            if uncertainty.volatility_uncertainty > 0.8:
                cash_weight = min(cash_weight + 0.15, 0.5)
        
        # === 不确定性调整 ===
        uncertainty_threshold = self.config['weights']['uncertainty_threshold']
        
        if uncertainty.combined_uncertainty > uncertainty_threshold:
            # 高不确定性：降低所有风险权重，增加现金
            decay_factor = self.config['adaptation']['uncertainty_decay']
            alpha_weight *= decay_factor
            ml_weight *= decay_factor
            
            # 增加现金权重
            cash_preference = self.config['adaptation']['cash_preference']
            additional_cash = (1.0 - alpha_weight - ml_weight) * cash_preference
            cash_weight = min(cash_weight + additional_cash, 
                            self.config['weights']['max_cash_weight'])
        
        # === 权重标准化 ===
        total_weight = alpha_weight + ml_weight + cash_weight
        if total_weight > 1.0:
            # 按比例缩放
            scale_factor = 1.0 / total_weight
            alpha_weight *= scale_factor
            ml_weight *= scale_factor
            cash_weight *= scale_factor
        elif total_weight < 1.0:
            # 剩余权重分配给ML
            ml_weight += (1.0 - total_weight)
        
        return {
            'alpha': alpha_weight,
            'ml': ml_weight,
            'cash': cash_weight
        }
    
    def train_regime_model(self, historical_data: Dict[str, pd.Series]) -> None:
        """训练市场状态模型"""
        logger.info("开始训练市场状态检测模型...")
        
        returns = historical_data['returns']
        volatility = historical_data['volatility']
        volume = historical_data.get('volume')
        
        self.regime_detector.fit_regime_model(returns, volatility, volume)
        
        logger.info("市场状态检测模型训练完成")

def get_market_indices_data(period: str = "2y") -> Dict[str, pd.Series]:
    """获取主要市场指数数据用于状态检测"""
    try:
        # 使用SPY作为市场代理
        spy = yf.download("SPY", period=period, progress=False)
        
        if spy.empty:
            raise ValueError("无法获取市场数据")
        
        # 计算收益率和波动率
        # 兼容yfinance在auto_adjust=True下无'Adj Close'的情况
        price_col = 'Adj Close' if 'Adj Close' in spy.columns else 'Close'
        returns = spy[price_col].pct_change().dropna()
        volatility = returns.rolling(21).std() * np.sqrt(252)  # 年化波动率
        volume = spy['Volume']
        
        return {
            'returns': returns,
            'volatility': volatility,
            'volume': volume,
            'price': spy[price_col]
        }
        
    except Exception as e:
        logger.error(f"获取市场数据失败: {e}")
        # 返回模拟数据
        dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        volatility = pd.Series(np.abs(np.random.normal(0.15, 0.05, len(dates))), index=dates)
        volume = pd.Series(np.random.uniform(1e6, 1e8, len(dates)), index=dates)
        
        return {
            'returns': returns,
            'volatility': volatility,
            'volume': volume,
            'price': (1 + returns).cumprod() * 100
        }

# 使用示例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 初始化自适应权重引擎
    engine = AdaptiveWeightEngine()
    
    # 获取市场数据并训练模型
    market_data = get_market_indices_data()
    engine.train_regime_model(market_data)
    
    # 模拟当前信号
    current_signals = {
        'momentum': 0.4,
        'reversal': -0.2,
        'volatility': 0.15
    }
    
    # 计算自适应权重
    adapted_weights = engine.adaptive_weight_allocation(
        market_data=market_data,
        prediction_uncertainty=0.08,
        current_signals=current_signals
    )
    
    print(f"自适应权重分配结果: {adapted_weights}")
