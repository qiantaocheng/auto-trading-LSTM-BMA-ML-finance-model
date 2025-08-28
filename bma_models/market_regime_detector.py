#!/usr/bin/env python3
"""
市场状态检测器 - Market Regime Detector (B方案：无监督聚类+时间平滑)
为BMA Enhanced系统提供基于GMM聚类的智能市场状态识别和分段建模能力
实现无数据泄漏的滚动更新和概率输出
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import logging
import warnings

logger = logging.getLogger(__name__)

@dataclass
class RegimeConfig:
    """市场状态检测配置 - B方案：GMM聚类"""
    # 基础参数
    n_regimes: int = 3                    # 状态数量（推荐2-4）
    lookback_window: int = 252            # 滚动训练窗口（1年交易日）
    update_frequency: int = 63            # 更新频率（每季度）
    
    # 指标计算窗口
    rv_window_short: int = 20             # 短期已实现波动率窗口
    rv_window_long: int = 60              # 长期已实现波动率窗口
    momentum_window: int = 21             # 动量计算窗口（月度）
    ma_window_short: int = 20             # 短期均线
    ma_window_long: int = 60              # 长期均线
    robust_window: int = 252              # Robust标准化窗口
    
    # GMM参数
    covariance_type: str = 'full'         # GMM协方差类型
    reg_covar: float = 1e-6               # 协方差正则化
    n_init: int = 10                      # GMM初始化次数
    max_iter: int = 100                   # 最大迭代次数
    
    # 时间平滑参数
    prob_smooth_window: int = 7           # 概率时间平滑窗口
    hard_threshold: float = 0.6           # 硬路由阈值
    
    # 降维参数（可选）
    enable_pca: bool = False              # 是否启用PCA降维
    pca_variance_ratio: float = 0.85      # PCA保留方差比例
    
    # 稳定性参数
    min_regime_samples: int = 50          # 每状态最少样本数
    regime_stability_threshold: float = 0.7  # 状态稳定性阈值

@dataclass 
class RegimeFeatures:
    """市场状态特征矩阵"""
    features: pd.DataFrame              # 标准化后的特征矩阵
    raw_features: pd.DataFrame          # 原始特征
    feature_names: List[str]            # 特征名称列表

class MarketRegimeDetector:
    """
    市场状态检测器 - B方案：无监督GMM聚类 + 时间平滑
    
    核心功能：
    1. 计算多维技术指标（无数据泄漏）
    2. 基于GMM的无监督聚类识别市场状态
    3. 概率输出和时间平滑
    4. 滚动更新避免参数漂移
    5. 支持硬路由和软路由
    """
    
    def __init__(self, config: RegimeConfig = None):
        self.config = config or RegimeConfig()
        
        # 模型组件
        self.gmm_model = None
        self.pca_model = None
        self.robust_scaler = RobustScaler()
        
        # 状态缓存和历史
        self._regime_cache = {}
        self._feature_cache = {}
        self._model_cache = {}
        self._last_update_date = None
        
        # 概率和状态历史（用于平滑）
        self._regime_probabilities = []
        self._regime_states = []
        
        logger.info(f"MarketRegimeDetector初始化：{self.config.n_regimes}状态GMM聚类系统")
        logger.info(f"配置 - 训练窗口：{self.config.lookback_window}天，更新频率：{self.config.update_frequency}天")
    
    def _compute_regime_features(self, data: pd.DataFrame) -> RegimeFeatures:
        """
        计算多维市场状态识别特征（B方案：无数据泄漏）
        
        包括：波动率、趋势、流动性、动量等多个维度
        所有指标使用历史窗口计算，避免未来信息泄漏
        """
        
        # 确保数据按时间排序
        if 'date' in data.columns:
            data = data.sort_values('date').copy()
        else:
            data = data.copy()
        
        # 计算收益率（滞后1天避免泄漏）
        data['returns'] = data['close'].pct_change().shift(1)
        
        features = pd.DataFrame(index=data.index)
        
        # 1. 已实现波动率特征（Realized Volatility）
        rv_short = data['returns'].rolling(self.config.rv_window_short).apply(
            lambda x: np.sqrt((x**2).sum())
        )
        rv_long = data['returns'].rolling(self.config.rv_window_long).apply(
            lambda x: np.sqrt((x**2).sum())
        )
        features['rv_short'] = rv_short
        features['rv_long'] = rv_long
        features['rv_ratio'] = rv_short / (rv_long + 1e-8)
        
        # 2. Parkinson波动率（基于高低价，更robust）
        if 'high' in data.columns and 'low' in data.columns:
            log_hl = np.log(data['high'] / data['low'])
            parkinson_vol = (log_hl**2).rolling(self.config.rv_window_short).mean() / (4 * np.log(2))
            features['parkinson_vol'] = np.sqrt(parkinson_vol)
        else:
            features['parkinson_vol'] = rv_short  # 回退到RV
        
        # 3. 动量特征（3-1月效应，滞后避免泄漏）
        mom_3m = data['close'].pct_change(self.config.momentum_window * 3).shift(1)  # 3月
        mom_1m = data['close'].pct_change(self.config.momentum_window).shift(1)     # 1月
        features['momentum_3_1'] = mom_3m - mom_1m
        features['momentum_short'] = data['close'].pct_change(5).shift(1)
        
        # 4. 均线偏离特征
        ma_short = data['close'].rolling(self.config.ma_window_short).mean()
        ma_long = data['close'].rolling(self.config.ma_window_long).mean()
        features['ma_deviation_short'] = ((data['close'] - ma_short) / ma_short).shift(1)
        features['ma_deviation_long'] = ((data['close'] - ma_long) / ma_long).shift(1)
        features['ma_trend'] = ((ma_short - ma_long) / ma_long).shift(1)
        
        # 5. Amihud流动性指标（非流动性）
        if 'volume' in data.columns:
            # 简化版Dollar Volume（如果没有实际美元成交额）
            dollar_vol = data['volume'] * data['close']
            amihud_illiq = (data['returns'].abs() / (dollar_vol + 1e-8)).replace([np.inf, -np.inf], np.nan)
            features['amihud_illiq'] = amihud_illiq.rolling(self.config.rv_window_short).mean().shift(1)
            
            # 成交量相关特征
            volume_ma = data['volume'].rolling(self.config.ma_window_short).mean()
            features['volume_ratio'] = (data['volume'] / volume_ma).shift(1)
            features['volume_trend'] = data['volume'].pct_change(self.config.momentum_window).shift(1)
        else:
            features['amihud_illiq'] = 0
            features['volume_ratio'] = 1
            features['volume_trend'] = 0
        
        # 6. 价格结构特征（如果有高低开收数据）
        if all(col in data.columns for col in ['high', 'low', 'open']):
            # 上影线占比
            upper_shadow = (data['high'] - np.maximum(data['close'], data['open'])) / data['close']
            # 下影线占比  
            lower_shadow = (np.minimum(data['close'], data['open']) - data['low']) / data['close']
            features['upper_shadow'] = upper_shadow.rolling(5).mean().shift(1)
            features['lower_shadow'] = lower_shadow.rolling(5).mean().shift(1)
            
            # 日内波动
            intraday_range = ((data['high'] - data['low']) / data['close'])
            features['intraday_range'] = intraday_range.rolling(self.config.rv_window_short).mean().shift(1)
        else:
            features['upper_shadow'] = 0
            features['lower_shadow'] = 0  
            features['intraday_range'] = features['rv_short']  # 用RV替代
        
        # 移除缺失值过多的特征
        features = features.select_dtypes(include=[np.number])
        
        # Robust Z-Score标准化（避免极值污染）
        features_standardized = features.copy()
        for col in features.columns:
            series = features[col]
            if series.isna().sum() / len(series) > 0.8:  # 缺失值超过80%的特征删除
                features_standardized = features_standardized.drop(columns=[col])
                continue
                
            # Robust标准化
            rolling_median = series.rolling(self.config.robust_window, min_periods=20).median()
            rolling_mad = series.rolling(self.config.robust_window, min_periods=20).apply(
                lambda x: np.median(np.abs(x - np.median(x))) if len(x) > 0 else np.nan
            )
            
            # 标准化公式
            features_standardized[col] = (series - rolling_median) / (1.4826 * rolling_mad + 1e-8)
        
        # 获取特征名称
        feature_names = list(features_standardized.columns)
        
        logger.debug(f"计算了{len(feature_names)}个状态识别特征: {feature_names}")
        
        return RegimeFeatures(
            features=features_standardized,
            raw_features=features,
            feature_names=feature_names
        )
    
    def _fit_gmm_model(self, features_data: np.ndarray) -> GaussianMixture:
        """
        训练GMM模型
        
        Args:
            features_data: 标准化后的特征数据 (n_samples, n_features)
            
        Returns:
            训练好的GMM模型
        """
        
        # 检查数据质量
        if np.any(np.isnan(features_data)):
            # 使用前向填充处理缺失值
            features_data = pd.DataFrame(features_data).fillna(method='ffill').fillna(0).values
        
        # PCA降维（可选）
        if self.config.enable_pca and features_data.shape[1] > 3:
            if self.pca_model is None:
                self.pca_model = PCA(n_components=None)
                features_data = self.pca_model.fit_transform(features_data)
                
                # 计算保留的成分数
                cum_var_ratio = np.cumsum(self.pca_model.explained_variance_ratio_)
                n_components = np.argmax(cum_var_ratio >= self.config.pca_variance_ratio) + 1
                
                # 重新拟合以使用确定的成分数
                self.pca_model = PCA(n_components=n_components)
                features_data = self.pca_model.fit_transform(features_data)
                
                logger.info(f"PCA降维：从{features_data.shape[1]}维降至{n_components}维，保留{cum_var_ratio[n_components-1]:.2%}方差")
            else:
                features_data = self.pca_model.transform(features_data)
        
        # 训练GMM模型
        gmm = GaussianMixture(
            n_components=self.config.n_regimes,
            covariance_type=self.config.covariance_type,
            reg_covar=self.config.reg_covar,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
            random_state=42
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gmm.fit(features_data)
        
        logger.info(f"GMM模型训练完成，收敛: {gmm.converged_}, AIC: {gmm.aic(features_data):.2f}")
        
        return gmm
    
    def _predict_regime_probabilities(self, features_data: np.ndarray, gmm_model: GaussianMixture) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测状态概率
        
        Args:
            features_data: 特征数据
            gmm_model: 训练好的GMM模型
            
        Returns:
            (状态概率矩阵, 最可能状态)
        """
        
        # 处理缺失值
        if np.any(np.isnan(features_data)):
            features_data = pd.DataFrame(features_data).fillna(method='ffill').fillna(0).values
        
        # PCA变换（如果启用）
        if self.config.enable_pca and self.pca_model is not None:
            features_data = self.pca_model.transform(features_data)
        
        # 预测概率
        probabilities = gmm_model.predict_proba(features_data)
        states = gmm_model.predict(features_data)
        
        return probabilities, states
    
    def _smooth_regime_probabilities(self, probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        时间平滑概率序列
        
        Args:
            probabilities: 原始概率矩阵 (n_samples, n_regimes)
            
        Returns:
            (平滑后概率矩阵, 平滑后状态)
        """
        
        prob_df = pd.DataFrame(probabilities)
        
        # 7日滚动平均平滑
        smoothed_probs = prob_df.rolling(
            window=self.config.prob_smooth_window,
            min_periods=1,
            center=False
        ).mean().values
        
        # 重新归一化（确保概率和为1）
        row_sums = smoothed_probs.sum(axis=1, keepdims=True)
        smoothed_probs = smoothed_probs / (row_sums + 1e-8)
        
        # 计算平滑后的最可能状态
        smoothed_states = np.argmax(smoothed_probs, axis=1)
        
        return smoothed_probs, smoothed_states
    
    def _should_update_model(self, current_date) -> bool:
        """
        检查是否需要更新模型
        
        Args:
            current_date: 当前日期
            
        Returns:
            是否需要更新
        """
        
        if self.gmm_model is None or self._last_update_date is None:
            return True
        
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        if isinstance(self._last_update_date, str):
            self._last_update_date = pd.to_datetime(self._last_update_date)
        
        days_since_update = (current_date - self._last_update_date).days
        return days_since_update >= self.config.update_frequency
    
    def detect_regimes(self, data: pd.DataFrame) -> pd.Series:
        """
        B方案：基于GMM聚类的市场状态检测（无数据泄漏，支持滚动更新）
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            pd.Series: 每个时间点的市场状态 (0/1/2/...)
        """
        
        if data.empty:
            logger.warning("输入数据为空，返回默认状态0")
            return pd.Series(dtype=int)
        
        # 检查必要列
        required_cols = ['close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的数据列: {missing_cols}")
        
        try:
            # 1. 计算多维技术特征
            logger.debug("计算市场状态识别特征...")
            regime_features = self._compute_regime_features(data)
            features_data = regime_features.features.dropna()
            
            if len(features_data) < self.config.min_regime_samples:
                logger.warning(f"有效样本数({len(features_data)})小于最小要求({self.config.min_regime_samples})，返回默认状态")
                return pd.Series(0, index=data.index)
            
            # 2. 检查是否需要更新GMM模型
            current_date = data.index[-1] if hasattr(data.index, 'max') else None
            should_update = self._should_update_model(current_date)
            
            if should_update:
                logger.info("开始训练/更新GMM聚类模型...")
                
                # 使用滚动窗口训练GMM
                train_window_end = len(features_data)
                train_window_start = max(0, train_window_end - self.config.lookback_window)
                train_features = features_data.iloc[train_window_start:train_window_end]
                
                if len(train_features) >= self.config.min_regime_samples:
                    self.gmm_model = self._fit_gmm_model(train_features.values)
                    self._last_update_date = current_date
                    
                    logger.info(f"GMM模型更新完成，训练样本：{len(train_features)}，AIC: {self.gmm_model.aic(train_features.values):.2f}")
                else:
                    logger.warning(f"训练数据不足({len(train_features)})，跳过模型更新")
            
            # 3. 如果没有训练好的模型，使用默认状态
            if self.gmm_model is None:
                logger.warning("GMM模型不可用，返回默认状态")
                return pd.Series(0, index=data.index)
            
            # 4. 预测状态概率
            logger.debug("预测市场状态概率...")
            probabilities, raw_states = self._predict_regime_probabilities(features_data.values, self.gmm_model)
            
            # 5. 时间平滑处理
            logger.debug("应用时间平滑...")
            smoothed_probabilities, smoothed_states = self._smooth_regime_probabilities(probabilities)
            
            # 6. 创建完整的状态序列（包括NaN位置）
            regime_series = pd.Series(0, index=data.index, dtype=int)
            regime_series.loc[features_data.index] = smoothed_states
            
            # 7. 前向填充缺失状态
            regime_series = regime_series.fillna(method='ffill').fillna(0).astype(int)
            
            # 8. 缓存结果
            self._regime_cache[id(data)] = regime_series
            self._feature_cache[id(data)] = regime_features
            
            # 存储概率信息（用于软路由）
            prob_df = pd.DataFrame(smoothed_probabilities, index=features_data.index)
            prob_df.columns = [f'regime_{i}_prob' for i in range(self.config.n_regimes)]
            self._model_cache['latest_probabilities'] = prob_df
            
            # 9. 状态统计和日志
            regime_counts = regime_series.value_counts().sort_index()
            state_switches = (regime_series.diff() != 0).sum()
            
            logger.info(f"GMM状态分布: {dict(regime_counts)}")
            logger.info(f"状态切换次数: {state_switches}, 切换率: {state_switches/len(regime_series):.1%}")
            
            # 状态解释
            for regime_id, count in regime_counts.items():
                prob_info = f"平均概率: {smoothed_probabilities[smoothed_states == regime_id, regime_id].mean():.2f}" if regime_id < len(smoothed_probabilities[0]) else ""
                logger.debug(f"状态{regime_id}: {count}天 ({count/len(regime_series):.1%}), {prob_info}")
            
            return regime_series
            
        except Exception as e:
            logger.error(f"GMM状态检测失败: {e}")
            import traceback
            logger.debug(f"错误详情: {traceback.format_exc()}")
            # 返回默认状态（全部为状态0）
            return pd.Series(0, index=data.index)
    
    def get_regime_probabilities(self) -> Optional[pd.DataFrame]:
        """
        获取最新的状态概率矩阵（用于软路由）
        
        Returns:
            概率DataFrame或None
        """
        return self._model_cache.get('latest_probabilities', None)
    
    def predict_current_regime(self, data: pd.DataFrame, return_probabilities: bool = False) -> Union[int, Tuple[int, np.ndarray]]:
        """
        预测当前最新的市场状态（实时预测）
        
        Args:
            data: 最新的市场数据
            return_probabilities: 是否返回概率分布
            
        Returns:
            当前状态ID，或(状态ID, 概率数组)元组
        """
        
        if self.gmm_model is None:
            if return_probabilities:
                return 0, np.array([1.0] + [0.0] * (self.config.n_regimes - 1))
            return 0
        
        try:
            # 计算特征
            regime_features = self._compute_regime_features(data)
            latest_features = regime_features.features.dropna().iloc[-1:].values
            
            # 预测概率
            probabilities, states = self._predict_regime_probabilities(latest_features, self.gmm_model)
            
            current_state = int(states[-1])
            current_probs = probabilities[-1]
            
            if return_probabilities:
                return current_state, current_probs
            return current_state
            
        except Exception as e:
            logger.warning(f"当前状态预测失败: {e}")
            if return_probabilities:
                return 0, np.array([1.0] + [0.0] * (self.config.n_regimes - 1))
            return 0
    
    def get_regime_statistics(self, regimes: pd.Series) -> Dict[str, Any]:
        """
        获取GMM状态统计信息
        
        Args:
            regimes: 状态序列
            
        Returns:
            统计信息字典
        """
        
        if regimes.empty:
            return {}
        
        stats = {
            'regime_counts': regimes.value_counts().to_dict(),
            'regime_proportions': regimes.value_counts(normalize=True).to_dict(),
            'regime_transitions': self._count_transitions(regimes),
            'average_regime_duration': self._calculate_avg_duration(regimes),
            'regime_stability': self._calculate_stability(regimes),
            'regime_switch_rate': self._calculate_switch_rate(regimes)
        }
        
        # 添加概率统计（如果有）
        prob_df = self.get_regime_probabilities()
        if prob_df is not None:
            stats['average_probabilities'] = {
                col: prob_df[col].mean() for col in prob_df.columns
            }
            stats['probability_volatility'] = {
                col: prob_df[col].std() for col in prob_df.columns  
            }
        
        return stats
    
    def _count_transitions(self, regimes: pd.Series) -> int:
        """计算状态转换次数"""
        if len(regimes) <= 1:
            return 0
        return (regimes.diff() != 0).sum()
    
    def _calculate_switch_rate(self, regimes: pd.Series) -> float:
        """计算状态切换率（每月）"""
        if len(regimes) <= 1:
            return 0.0
        transitions = self._count_transitions(regimes)
        # 假设交易日，每月约21天
        months = len(regimes) / 21.0
        return transitions / months if months > 0 else 0.0
    
    def _calculate_avg_duration(self, regimes: pd.Series) -> Dict[int, float]:
        """计算各状态的平均持续时间"""
        if len(regimes) <= 1:
            return {}
            
        # 动态获取所有状态ID
        unique_regimes = regimes.unique()
        durations = {regime: [] for regime in unique_regimes}
        
        current_regime = regimes.iloc[0]
        current_duration = 1
        
        for i in range(1, len(regimes)):
            if regimes.iloc[i] == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = regimes.iloc[i]
                current_duration = 1
        
        # 添加最后一个状态的持续时间
        durations[current_regime].append(current_duration)
        
        # 计算平均值
        avg_durations = {}
        for regime, duration_list in durations.items():
            if duration_list:
                avg_durations[int(regime)] = np.mean(duration_list)
            else:
                avg_durations[int(regime)] = 0.0
        
        return avg_durations
    
    def _calculate_stability(self, regimes: pd.Series) -> float:
        """计算状态稳定性（减少转换频率的程度）"""
        if len(regimes) <= 1:
            return 1.0
            
        transitions = self._count_transitions(regimes)
        max_possible_transitions = len(regimes) - 1
        stability = 1.0 - (transitions / max_possible_transitions)
        return stability
    
    def get_regime_descriptions(self) -> Dict[int, str]:
        """
        获取状态描述（GMM聚类结果的语义解释）
        
        注意：GMM聚类的状态标签是数值型的，不具备先验的经济含义
        状态的实际特征需要通过后验分析确定
        """
        
        # 基础描述
        base_descriptions = {
            0: "市场状态0 - 通过GMM聚类识别的第一种市场行为模式",
            1: "市场状态1 - 通过GMM聚类识别的第二种市场行为模式", 
            2: "市场状态2 - 通过GMM聚类识别的第三种市场行为模式",
            3: "市场状态3 - 通过GMM聚类识别的第四种市场行为模式"
        }
        
        # 返回对应数量的描述
        return {i: base_descriptions.get(i, f"市场状态{i} - GMM聚类识别的市场行为模式") 
                for i in range(self.config.n_regimes)}
    
    def interpret_regime_characteristics(self, regimes: pd.Series, data: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        解释各个状态的特征（后验分析）
        
        Args:
            regimes: 状态序列
            data: 原始市场数据
            
        Returns:
            每个状态的特征统计
        """
        
        if regimes.empty or data.empty:
            return {}
        
        # 计算基础统计量
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std()
        
        regime_characteristics = {}
        
        for regime_id in regimes.unique():
            if pd.isna(regime_id):
                continue
                
            regime_mask = regimes == regime_id
            regime_returns = returns[regime_mask].dropna()
            regime_volatility = volatility[regime_mask].dropna()
            
            if len(regime_returns) > 0:
                characteristics = {
                    'avg_return': regime_returns.mean(),
                    'return_volatility': regime_returns.std(),
                    'avg_volatility': regime_volatility.mean() if len(regime_volatility) > 0 else 0,
                    'sharpe_ratio': regime_returns.mean() / (regime_returns.std() + 1e-8),
                    'positive_return_ratio': (regime_returns > 0).mean(),
                    'sample_count': len(regime_returns)
                }
                regime_characteristics[int(regime_id)] = characteristics
        
        return regime_characteristics


def create_market_regime_detector(config: RegimeConfig = None) -> MarketRegimeDetector:
    """工厂函数：创建市场状态检测器（B方案：GMM聚类）"""
    return MarketRegimeDetector(config)


# 预设配置 - B方案GMM参数
class RegimeConfigPresets:
    """预设的GMM状态检测配置"""
    
    @staticmethod
    def conservative() -> RegimeConfig:
        """保守配置 - 更稳定的GMM聚类"""
        return RegimeConfig(
            n_regimes=3,
            lookback_window=504,           # 2年训练窗口
            update_frequency=126,          # 半年更新一次
            prob_smooth_window=10,         # 更长的平滑窗口
            hard_threshold=0.7,            # 更高的硬路由阈值
            min_regime_samples=100,        # 更多最小样本
            enable_pca=False,              # 关闭PCA简化
            robust_window=252             # 1年robust标准化窗口
        )
    
    @staticmethod
    def aggressive() -> RegimeConfig:
        """激进配置 - 更敏感的GMM聚类"""
        return RegimeConfig(
            n_regimes=4,                   # 更多状态数
            lookback_window=126,           # 0.5年训练窗口
            update_frequency=21,           # 月度更新
            prob_smooth_window=3,          # 短平滑窗口
            hard_threshold=0.5,            # 低硬路由阈值
            min_regime_samples=30,         # 少最小样本
            enable_pca=True,               # 启用PCA降维
            pca_variance_ratio=0.8,        # 保留80%方差
            robust_window=126             # 0.5年robust标准化
        )
    
    @staticmethod
    def balanced() -> RegimeConfig:
        """平衡配置 - 推荐的默认设置"""
        return RegimeConfig()  # 使用默认参数
    
    @staticmethod
    def high_frequency() -> RegimeConfig:
        """高频更新配置 - 适合短期交易"""
        return RegimeConfig(
            n_regimes=3,
            lookback_window=63,            # 3个月训练窗口
            update_frequency=5,            # 周更新
            prob_smooth_window=5,
            hard_threshold=0.6,
            rv_window_short=5,             # 短波动率窗口
            rv_window_long=20,
            momentum_window=5,             # 短动量窗口
            ma_window_short=10,
            ma_window_long=20
        )