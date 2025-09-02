#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoTrader统一因子库 - 基于Polygon 15分钟延迟数据
整合所有因子计算，统一数据源为Polygon API
支持autotrader算法使用的所有因子类型
"""

import logging
import math
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import scipy.stats as stats

# Polygon客户端导入
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from polygon_client import polygon_client, download, Ticker
except ImportError as e:
    logging.warning(f"Polygon client import failed: {e}")
    polygon_client = None

# 延迟数据配置
try:
    from .delayed_data_config import DelayedDataConfig, DEFAULT_DELAYED_CONFIG, should_trade_with_delayed_data
except ImportError:
    # 创建简化配置类以防导入失败
    @dataclass
    class DelayedDataConfig:
        enabled: bool = True
        data_delay_minutes: int = 15
        min_confidence_threshold: float = 0.8
        position_size_reduction: float = 0.4
        min_alpha_multiplier: float = 1.0  # 添加缺失字段
    
    DEFAULT_DELAYED_CONFIG = DelayedDataConfig()
    
    def should_trade_with_delayed_data(config: DelayedDataConfig) -> Tuple[bool, str]:
        """简化的延迟数据交易检查"""
        if not config.enabled:
            return False, "Delayed data trading disabled"
        return True, "Delayed data trading allowed"

# 导入自适应权重系统
try:
    from .adaptive_factor_weights import get_current_factor_weights, AdaptiveFactorWeights
    from .adaptive_weights_adapter import get_bma_enhanced_weights
    ADAPTIVE_WEIGHTS_AVAILABLE = True
    BMA_ENHANCED_WEIGHTS_AVAILABLE = True
except ImportError:
    ADAPTIVE_WEIGHTS_AVAILABLE = False
    BMA_ENHANCED_WEIGHTS_AVAILABLE = False
    logging.warning("Adaptive weights system not available, using fallback weights")

logger = logging.getLogger(__name__)

@dataclass
class FactorResult:
    """因子计算结果"""
    factor_name: str
    value: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    data_quality_score: float

class UnifiedPolygonFactors:
    """
    AutoTrader统一因子库
    基于Polygon 15分钟延迟数据的所有因子计算
    """
    
    def __init__(self, config: DelayedDataConfig = None):
        """初始化统一因子库"""
        self.config = config or DEFAULT_DELAYED_CONFIG
        self.client = polygon_client
        self.cache = {}
        self.cache_ttl = 300  # 5分钟缓存
        
        # 因子权重配置 - 现在支持动态权重学习
        self.fallback_weights = {
            'momentum': 0.20,        # 动量因子
            'mean_reversion': 0.30,  # 均值回归（主要信号）
            'trend': 0.25,           # 趋势因子
            'volatility': 0.15,      # 波动率因子
            'volume': 0.10,          # 成交量因子
            'microstructure': 0.00   # 微观结构（暂时禁用）
        }
        
        # 延迟初始化自适应权重系统（避免重复初始化）
        self.adaptive_weights = None
        
        # 🔥 延迟权重获取：仅在实际需要时才获取权重
        self.factor_weights = None
        self._weights_initialized = False
        
        # 统计信息
        self.stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_evictions': 0,
            'last_update': datetime.now()
        }
        
        # 缓存管理
        self.max_cache_size = 1000  # 最大缓存条目数
        self.cache_timestamps = {}  # 跟踪缓存时间戳
        
        logger.info(f"UnifiedPolygonFactors initialized with {self.config.data_delay_minutes}min delay")
    
    def _ensure_weights_initialized(self):
        """确保权重已初始化（轻量级延迟初始化）"""
        if not self._weights_initialized:
            logger.info("⚡ 轻量级权重初始化（避免训练前触发ML）")
            
            # 🔥 启动时仅使用轻量级权重，不触发ML学习
            if self.adaptive_weights is not None:
                # 尝试加载现有权重，不触发新的学习
                latest_result = self.adaptive_weights.load_latest_weights()
                if latest_result is not None and latest_result.confidence >= 0.5:
                    logger.info(f"📂 加载历史ML权重，置信度: {latest_result.confidence:.3f}")
                    self.factor_weights = latest_result.weights
                else:
                    logger.info("📋 使用优化回退权重，等待训练完成")
                    self.factor_weights = self.fallback_weights.copy()
            else:
                self.factor_weights = self.fallback_weights.copy()
            
            # 验证权重总和
            total_weight = sum(self.factor_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                self.factor_weights = {k: v/total_weight for k, v in self.factor_weights.items()}
            
            self._weights_initialized = True
            logger.info(f"✅ 轻量级权重初始化完成，等待训练后更新")

    def get_bma_enhanced_weights(self) -> Dict[str, float]:
        """
        专为BMA Enhanced系统获取ML权重
        只有在BMA训练完成后才应该调用此方法
        """
        try:
            if BMA_ENHANCED_WEIGHTS_AVAILABLE:
                logger.info("🎯 BMA Enhanced训练后获取ML权重")
                ml_weights = get_bma_enhanced_weights()
                
                # 更新内部权重缓存
                self.factor_weights = ml_weights
                self._weights_initialized = True
                
                return ml_weights
            else:
                logger.warning("BMA Enhanced权重适配器不可用，回退到标准方法")
                return self._get_current_weights(force_ml_learning=True)
        except Exception as e:
            logger.error(f"BMA Enhanced权重获取失败: {e}")
            return self._get_current_weights()
    
    def update_weights_post_training(self, training_context: str = "BMA_ENHANCED"):
        """
        训练完成后更新权重
        
        Args:
            training_context: 训练上下文 ("BMA_ENHANCED", "MANUAL", etc.)
        """
        try:
            logger.info(f"📊 {training_context} 训练完成，更新因子权重")
            
            if training_context == "BMA_ENHANCED":
                # BMA训练完成后，获取ML权重
                updated_weights = self.get_bma_enhanced_weights()
            else:
                # 其他训练模式，强制更新权重
                updated_weights = self._get_current_weights(force_ml_learning=True)
            
            # 验证并应用新权重
            total_weight = sum(updated_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                updated_weights = {k: v/total_weight for k, v in updated_weights.items()}
            
            self.factor_weights = updated_weights
            self._weights_initialized = True
            
            logger.info(f"✅ 权重更新完成: {updated_weights}")
            return updated_weights
            
        except Exception as e:
            logger.error(f"训练后权重更新失败: {e}")
            # 保持现有权重或使用回退权重
            if not self._weights_initialized:
                self._ensure_weights_initialized()
            return self.factor_weights

    def _get_current_weights(self, force_ml_learning: bool = False) -> Dict[str, float]:
        """获取当前因子权重（优先ML学习权重）"""
        try:
            if self.adaptive_weights is not None:
                # 🔥 优先使用主动学习权重，避免硬编码回退
                if force_ml_learning:
                    logger.info("🚀 BMA Enhanced模式：主动获取或学习ML权重")
                    adaptive_weights = self.adaptive_weights.get_or_learn_weights()
                else:
                    adaptive_weights = self.adaptive_weights.get_current_weights()
                
                # 检查是否为硬编码回退权重
                is_fallback = self._is_fallback_weights(adaptive_weights)
                if is_fallback:
                    logger.warning("⚠️ 检测到硬编码权重，尝试主动学习ML权重")
                    try:
                        ml_weights = self.adaptive_weights.get_or_learn_weights()
                        if not self._is_fallback_weights(ml_weights):
                            logger.info("✅ 成功获取ML权重，替换硬编码权重")
                            adaptive_weights = ml_weights
                    except Exception as ml_error:
                        logger.error(f"ML权重获取失败: {ml_error}")
                
                weight_type = "ML学习权重" if not self._is_fallback_weights(adaptive_weights) else "硬编码回退权重"
                logger.info(f"使用{weight_type}: {adaptive_weights}")
                return adaptive_weights
            else:
                # 使用回退权重
                logger.info(f"使用回退权重: {self.fallback_weights}")
                return self.fallback_weights.copy()
        except Exception as e:
            logger.error(f"获取权重失败: {e}")
            return self.fallback_weights.copy()
    
    def _is_fallback_weights(self, weights: Dict[str, float]) -> bool:
        """检测权重是否为硬编码回退权重"""
        try:
            # 检查权重分布特征
            values = list(weights.values())
            
            # 检查是否为等权重分布 (0.2, 0.2, 0.2, 0.2, 0.2)
            if len(set(values)) == 1 and abs(values[0] - 0.2) < 0.001:
                return True
                
            # 检查是否匹配预设的回退权重模式
            fallback_signature = [0.30, 0.30, 0.25, 0.20, 0.15]  # 典型回退权重
            sorted_weights = sorted(values, reverse=True)
            if len(sorted_weights) >= 3:
                if (abs(sorted_weights[0] - 0.3) < 0.05 and 
                    abs(sorted_weights[1] - 0.3) < 0.05):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _validate_client(self) -> bool:
        """验证Polygon客户端可用性"""
        if not self.client:
            logger.error("Polygon client not available")
            return False
        return True
    
    def _get_cache_key(self, symbol: str, factor_name: str, lookback_days: int = 60) -> str:
        """生成安全的缓存键，避免冲突"""
        import hashlib
        
        # 清理输入参数，避免特殊字符
        clean_symbol = symbol.replace('_', '').replace('-', '').upper()
        clean_factor = factor_name.replace('_', '').replace('-', '').lower()
        
        # 时间窗口（防止边界情况的键冲突）
        time_window = int(time.time() // self.cache_ttl)
        
        # 创建复合键
        key_components = f"{clean_symbol}|{clean_factor}|{lookback_days}|{time_window}"
        
        # 生成哈希避免过长的键名和潜在冲突
        key_hash = hashlib.md5(key_components.encode('utf-8')).hexdigest()[:16]
        
        # 返回可读性好的缓存键
        return f"factors_{clean_symbol}_{clean_factor}_{key_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """获取缓存结果"""
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                self.stats['cache_hits'] += 1
                # 更新访问时间
                self.cache_timestamps[cache_key] = time.time()
                return cached_data
            else:
                # 过期缓存清理
                del self.cache[cache_key]
                if cache_key in self.cache_timestamps:
                    del self.cache_timestamps[cache_key]
        
        self.stats['cache_misses'] += 1
        return None
    
    def _set_cache(self, cache_key: str, data: Any):
        """设置缓存，包含大小管理"""
        current_time = time.time()
        
        # 检查缓存大小限制
        if len(self.cache) >= self.max_cache_size:
            self._evict_old_cache_entries()
        
        # 设置新缓存
        self.cache[cache_key] = (current_time, data)
        self.cache_timestamps[cache_key] = current_time
        
    def _evict_old_cache_entries(self):
        """清理老旧缓存条目"""
        if not self.cache_timestamps:
            return
            
        # 按访问时间排序，移除最老的条目
        sorted_entries = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
        
        # 移除最老的25%条目
        evict_count = max(1, len(sorted_entries) // 4)
        
        for cache_key, _ in sorted_entries[:evict_count]:
            if cache_key in self.cache:
                del self.cache[cache_key]
            if cache_key in self.cache_timestamps:
                del self.cache_timestamps[cache_key]
            self.stats['cache_evictions'] += 1
        
        logger.debug(f"缓存清理：移除了 {evict_count} 个老旧条目")
    
    def clear_cache(self):
        """清空所有缓存"""
        cache_size = len(self.cache)
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info(f"缓存已清空，移除了 {cache_size} 个条目")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
            'total_evictions': self.stats['cache_evictions']
        }
    
    def get_market_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """
        获取市场数据 - 统一数据源
        使用Polygon 15分钟延迟数据
        """
        if not self._validate_client():
            return pd.DataFrame()
        
        cache_key = self._get_cache_key(symbol, "market_data", days)
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=max(days * 2, 120))  # 确保足够的数据
            
            data = self.client.get_historical_bars(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                'day',
                1
            )
            
            if not data.empty and len(data) > 0:
                # 添加必要的技术指标列
                data['Returns'] = data['Close'].pct_change()
                data['Volume_MA20'] = data['Volume'].rolling(20).mean()
                data['Price_MA20'] = data['Close'].rolling(20).mean()
                data['Price_MA5'] = data['Close'].rolling(5).mean()
                data['Price_MA50'] = data['Close'].rolling(50).mean()
                data['Volatility_20'] = data['Returns'].rolling(20).std()
                
                # 去除NaN值
                data = data.dropna()
                
                self._set_cache(cache_key, data)
                self.stats['successful_calculations'] += 1
                return data
            else:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            self.stats['failed_calculations'] += 1
            return pd.DataFrame()
        finally:
            self.stats['total_calculations'] += 1
    
    # ===============================
    # 核心因子 - AutoTrader引擎使用
    # ===============================
    
    def calculate_zscore(self, values: List[float], n: int = 20) -> List[float]:
        """
        Z-Score计算 - AutoTrader核心均值回归信号
        移植自autotrader.factors.zscore
        """
        try:
            if len(values) < n:
                return [math.nan] * len(values)
            
            out = []
            for i in range(len(values)):
                if i < n - 1:
                    out.append(math.nan)
                else:
                    window = values[i - n + 1:i + 1]
                    mean_val = sum(window) / n
                    variance = sum((x - mean_val) ** 2 for x in window) / n
                    std_val = math.sqrt(variance) if variance > 0 else 0
                    
                    if std_val > 0:
                        z = (values[i] - mean_val) / std_val
                        out.append(z)
                    else:
                        out.append(0.0)
            
            return out
        except Exception as e:
            logger.error(f"Z-score calculation failed: {e}")
            return [math.nan] * len(values)
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> List[float]:
        """
        ATR (Average True Range) 计算
        移植自autotrader.factors.atr
        """
        try:
            if len(highs) != len(lows) or len(highs) != len(closes):
                return [math.nan] * len(closes)
            
            # 计算True Range
            tr_values = []
            for i in range(len(closes)):
                if i == 0:
                    tr = highs[i] - lows[i]
                else:
                    tr = max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1])
                    )
                tr_values.append(tr)
            
            # 计算SMA of TR
            atr_values = []
            for i in range(len(tr_values)):
                if i < n - 1:
                    atr_values.append(math.nan)
                else:
                    window_tr = tr_values[i - n + 1:i + 1]
                    atr = sum(window_tr) / n
                    atr_values.append(atr)
            
            return atr_values
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            return [math.nan] * len(closes)
    
    def calculate_sma(self, values: List[float], n: int) -> List[float]:
        """简单移动平均计算"""
        try:
            out = []
            for i in range(len(values)):
                if i < n - 1:
                    out.append(math.nan)
                else:
                    window = values[i - n + 1:i + 1]
                    sma = sum(window) / n
                    out.append(sma)
            return out
        except Exception as e:
            logger.error(f"SMA calculation failed: {e}")
            return [math.nan] * len(values)
    
    def calculate_mean_reversion_signal(self, symbol: str) -> FactorResult:
        """
        均值回归信号 - AutoTrader主要策略
        基于20日Z-Score，移植自engine.py的mr_signal
        """
        try:
            data = self.get_market_data(symbol, days=60)
            if data.empty or len(data) < 20:
                return self._create_failed_result('mean_reversion', 'Insufficient data')
            
            closes = data['Close'].tolist()
            z_scores = self.calculate_zscore(closes, 20)
            
            if not z_scores or len(z_scores) == 0:
                return self._create_failed_result('mean_reversion', 'Z-score calculation failed')
            
            current_z = z_scores[-1]
            if math.isnan(current_z):
                return self._create_failed_result('mean_reversion', 'Invalid Z-score')
            
            # AutoTrader信号逻辑
            if current_z > 2.5:
                signal = -1.0  # 强卖出信号
            elif current_z > 1.5:
                signal = -0.5  # 弱卖出信号
            elif current_z < -2.5:
                signal = 1.0   # 强买入信号
            elif current_z < -1.5:
                signal = 0.5   # 弱买入信号
            else:
                signal = -current_z  # 线性缩放
            
            # 计算置信度
            confidence = min(abs(current_z) / 2.5, 1.0) * 0.9
            
            return FactorResult(
                factor_name='mean_reversion',
                value=signal,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'z_score': current_z,
                    'data_points': len(closes),
                    'lookback_period': 20
                },
                data_quality_score=0.95 if len(data) >= 30 else 0.8
            )
            
        except Exception as e:
            logger.error(f"Mean reversion calculation failed for {symbol}: {e}")
            return self._create_failed_result('mean_reversion', str(e))
    
    def calculate_momentum_signal(self, symbol: str, period: int = 20) -> FactorResult:
        """
        动量信号计算
        移植自engine.py的calculate_momentum
        """
        try:
            data = self.get_market_data(symbol, days=60)
            if data.empty or len(data) < period + 1:
                return self._create_failed_result('momentum', 'Insufficient data')
            
            prices = data['Close'].tolist()
            
            # 计算收益率
            returns = []
            for i in range(1, len(prices)):
                if prices[i] > 0 and prices[i-1] > 0:
                    ret = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(ret)
            
            if len(returns) < period:
                return self._create_failed_result('momentum', 'Insufficient returns data')
            
            # 使用最近period个收益率计算动量
            recent_returns = returns[-period:]
            momentum = sum(recent_returns) / len(recent_returns)
            
            # 计算置信度
            volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0
            confidence = min(abs(momentum) / (volatility + 1e-6), 1.0) * 0.8
            
            return FactorResult(
                factor_name='momentum',
                value=momentum,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'period': period,
                    'avg_returns': momentum,
                    'volatility': volatility,
                    'data_points': len(recent_returns)
                },
                data_quality_score=0.9
            )
            
        except Exception as e:
            logger.error(f"Momentum calculation failed for {symbol}: {e}")
            return self._create_failed_result('momentum', str(e))
    
    def calculate_trend_signal(self, symbol: str) -> FactorResult:
        """
        趋势信号计算
        移植自engine.py的multi_factor_signal中的趋势部分
        """
        try:
            data = self.get_market_data(symbol, days=60)
            if data.empty or len(data) < 50:
                return self._create_failed_result('trend', 'Insufficient data')
            
            closes = data['Close'].tolist()
            
            # 计算移动平均
            sma5 = sum(closes[-5:]) / 5
            sma20 = sum(closes[-20:]) / 20
            sma50 = sum(closes[-50:]) / 50
            
            trend_score = 0.0
            current_price = closes[-1]
            
            # 趋势评分逻辑（来自engine.py）
            if sma5 > sma20 > sma50:
                trend_score += 0.4
            elif sma5 > sma20:
                trend_score += 0.2
            elif current_price > sma20:
                trend_score += 0.1
            
            # 均线斜率
            if len(closes) >= 25:
                sma20_prev = sum(closes[-25:-5]) / 20
                if sma20_prev > 0:
                    slope = (sma20 - sma20_prev) / abs(sma20_prev)
                    if slope > 0.01:
                        trend_score += 0.3
                    elif slope > 0:
                        trend_score += 0.1
            
            # 价格相对位置
            if current_price > sma5 * 1.02:
                trend_score += 0.3
            elif current_price > sma5:
                trend_score += 0.2
            
            # 归一化到[-1, 1]
            normalized_trend = max(-1.0, min(1.0, trend_score))
            
            # 计算置信度
            price_variance = np.var(closes[-20:]) if len(closes) >= 20 else 0
            confidence = 0.8 if abs(normalized_trend) > 0.3 else 0.6
            
            return FactorResult(
                factor_name='trend',
                value=normalized_trend,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'sma5': sma5,
                    'sma20': sma20,
                    'sma50': sma50,
                    'trend_score': trend_score,
                    'current_price': current_price
                },
                data_quality_score=0.9
            )
            
        except Exception as e:
            logger.error(f"Trend calculation failed for {symbol}: {e}")
            return self._create_failed_result('trend', str(e))
    
    def calculate_volume_signal(self, symbol: str) -> FactorResult:
        """
        成交量信号计算
        移植自engine.py的multi_factor_signal中的成交量部分
        """
        try:
            data = self.get_market_data(symbol, days=40)
            if data.empty or len(data) < 20:
                return self._create_failed_result('volume', 'Insufficient data')
            
            volumes = data['Volume'].tolist()
            
            volume_score = 0.0
            
            # 20日均量比较
            if len(volumes) >= 20:
                v20 = sum(volumes[-20:]) / 20
                v_current = max(volumes[-1], 0.0)
                ratio = v_current / v20 if v20 > 0 else 1.0
                
                if ratio > 1.5:
                    volume_score += 0.4
                elif ratio > 1.2:
                    volume_score += 0.2
                elif ratio > 0.8:
                    volume_score += 0.1
                
                # 近5日相对提升
                if len(volumes) >= 20:
                    recent5 = sum(volumes[-5:]) / 5
                    prev15 = sum(volumes[-20:-5]) / 15
                    if recent5 > prev15 * 1.2:
                        volume_score += 0.3
                    elif recent5 > prev15:
                        volume_score += 0.1
            
            # 归一化
            normalized_volume = max(-1.0, min(1.0, volume_score))
            
            # 计算置信度
            volume_stability = 1.0 / (1.0 + np.std(volumes[-10:]) / np.mean(volumes[-10:]))
            confidence = volume_stability * 0.7
            
            return FactorResult(
                factor_name='volume',
                value=normalized_volume,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'volume_score': volume_score,
                    'current_volume': volumes[-1],
                    'avg_volume_20d': sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 0,
                    'volume_ratio': ratio if 'ratio' in locals() else 1.0
                },
                data_quality_score=0.85
            )
            
        except Exception as e:
            logger.error(f"Volume calculation failed for {symbol}: {e}")
            return self._create_failed_result('volume', str(e))
    
    def calculate_volatility_signal(self, symbol: str) -> FactorResult:
        """
        波动率信号计算
        基于ATR和适宜波动率区间判断
        """
        try:
            data = self.get_market_data(symbol, days=30)
            if data.empty or len(data) < 15:
                return self._create_failed_result('volatility', 'Insufficient data')
            
            highs = data['High'].tolist()
            lows = data['Low'].tolist()
            closes = data['Close'].tolist()
            
            # 计算ATR
            atr_values = self.calculate_atr(highs, lows, closes, 14)
            current_atr = atr_values[-1] if atr_values and not math.isnan(atr_values[-1]) else 0
            
            volatility_score = 0.0
            current_price = closes[-1]
            
            if current_atr > 0 and current_price > 0:
                atr_pct = (current_atr / current_price) * 100
                
                # 适宜波动率区间（来自engine.py）
                if 1.5 <= atr_pct <= 4.0:
                    volatility_score += 0.4
                elif 1.0 <= atr_pct <= 6.0:
                    volatility_score += 0.2
            
            # 归一化
            normalized_volatility = max(-1.0, min(1.0, volatility_score))
            
            # 计算置信度
            confidence = 0.8 if current_atr > 0 else 0.3
            
            return FactorResult(
                factor_name='volatility',
                value=normalized_volatility,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'atr_14': current_atr,
                    'atr_percentage': (current_atr / current_price) * 100 if current_price > 0 else 0,
                    'volatility_score': volatility_score,
                    'current_price': current_price
                },
                data_quality_score=0.9
            )
            
        except Exception as e:
            logger.error(f"Volatility calculation failed for {symbol}: {e}")
            return self._create_failed_result('volatility', str(e))
    
    def calculate_composite_signal(self, symbol: str) -> FactorResult:
        """
        综合信号计算
        整合所有核心因子，移植自engine.py的multi_factor_signal
        """
        try:
            # 计算各个因子
            mr_result = self.calculate_mean_reversion_signal(symbol)
            momentum_result = self.calculate_momentum_signal(symbol)
            trend_result = self.calculate_trend_signal(symbol)
            volume_result = self.calculate_volume_signal(symbol)
            volatility_result = self.calculate_volatility_signal(symbol)
            
            # 构建因子字典
            factors = {
                'mean_reversion': mr_result.value if mr_result.confidence > 0.3 else 0.0,
                'momentum': momentum_result.value if momentum_result.confidence > 0.3 else 0.0,
                'trend': trend_result.value if trend_result.confidence > 0.3 else 0.0,
                'volume': volume_result.value if volume_result.confidence > 0.3 else 0.0,
                'volatility': volatility_result.value if volatility_result.confidence > 0.3 else 0.0
            }
            
            # 加权计算综合得分
            composite_score = 0.0
            total_weight = 0.0
            
            # 使用AutoTrader权重（来自engine.py）
            weights = {
                'trend': 0.30,
                'momentum': 0.25,
                'volume': 0.20,
                'volatility': 0.15,
                'mean_reversion': 0.30  # 主要信号
            }
            
            for factor_name, weight in weights.items():
                if factor_name in factors and not math.isnan(factors[factor_name]):
                    composite_score += factors[factor_name] * weight
                    total_weight += weight
            
            # 归一化
            if total_weight > 0:
                composite_score = composite_score / total_weight
            
            # 最终限制在[-1, 1]
            final_score = max(-1.0, min(1.0, composite_score))
            
            # 计算综合置信度
            confidences = [mr_result.confidence, momentum_result.confidence, 
                          trend_result.confidence, volume_result.confidence, volatility_result.confidence]
            avg_confidence = sum(confidences) / len(confidences)
            
            # 应用延迟数据调整
            if self.config.enabled:
                final_score *= self.config.min_alpha_multiplier
                final_score = max(-1.0, min(1.0, final_score))  # 重新限制
                avg_confidence = min(avg_confidence, self.config.min_confidence_threshold)
            
            return FactorResult(
                factor_name='composite',
                value=final_score,
                confidence=avg_confidence,
                timestamp=datetime.now(),
                metadata={
                    'individual_factors': factors,
                    'weights_used': weights,
                    'total_weight': total_weight,
                    'raw_score': composite_score,
                    'delay_adjusted': self.config.enabled,
                    'factors_count': len([f for f in factors.values() if not math.isnan(f)])
                },
                data_quality_score=min([r.data_quality_score for r in [mr_result, momentum_result, trend_result, volume_result, volatility_result]])
            )
            
        except Exception as e:
            logger.error(f"Composite signal calculation failed for {symbol}: {e}")
            return self._create_failed_result('composite', str(e))
    
    def _create_failed_result(self, factor_name: str, reason: str) -> FactorResult:
        """创建失败结果"""
        return FactorResult(
            factor_name=factor_name,
            value=0.0,
            confidence=0.0,
            timestamp=datetime.now(),
            metadata={'error': reason},
            data_quality_score=0.0
        )
    
    # ===============================
    # 高级因子接口
    # ===============================
    
    def calculate_all_signals(self, symbol: str) -> Dict[str, FactorResult]:
        """计算所有信号"""
        results = {}
        
        try:
            results['mean_reversion'] = self.calculate_mean_reversion_signal(symbol)
            results['momentum'] = self.calculate_momentum_signal(symbol)
            results['trend'] = self.calculate_trend_signal(symbol)
            results['volume'] = self.calculate_volume_signal(symbol)
            results['volatility'] = self.calculate_volatility_signal(symbol)
            results['composite'] = self.calculate_composite_signal(symbol)
            
        except Exception as e:
            logger.error(f"Failed to calculate all signals for {symbol}: {e}")
        
        return results
    
    def get_trading_signal(self, symbol: str, threshold: float = 0.3) -> Dict[str, Any]:
        """
        获取交易信号
        返回AutoTrader引擎兼容的信号格式
        """
        try:
            composite_result = self.calculate_composite_signal(symbol)
            
            signal_strength = abs(composite_result.value)
            
            # 检查是否满足交易条件
            meets_threshold = signal_strength >= threshold
            meets_confidence = composite_result.confidence >= self.config.min_confidence_threshold
            
            # 延迟数据交易时间检查
            can_trade_delayed, delay_reason = should_trade_with_delayed_data(self.config)
            
            can_trade = meets_threshold and meets_confidence and can_trade_delayed
            
            # 确定交易方向
            side = "BUY" if composite_result.value > 0 else "SELL"
            
            return {
                'symbol': symbol,
                'signal_value': composite_result.value,
                'signal_strength': signal_strength,
                'confidence': composite_result.confidence,
                'side': side,
                'can_trade': can_trade,
                'meets_threshold': meets_threshold,
                'meets_confidence': meets_confidence,
                'can_trade_delayed': can_trade_delayed,
                'threshold': threshold,
                'timestamp': composite_result.timestamp,
                'data_quality': composite_result.data_quality_score,
                'delay_reason': delay_reason if not can_trade_delayed else None,
                'metadata': composite_result.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get trading signal for {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal_value': 0.0,
                'signal_strength': 0.0,
                'confidence': 0.0,
                'side': 'HOLD',
                'can_trade': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def validate_polygon_data(self, symbol: str) -> Dict[str, Any]:
        """验证Polygon数据质量和延迟"""
        try:
            data = self.get_market_data(symbol, days=5)
            
            if data.empty:
                return {
                    'symbol': symbol,
                    'data_available': False,
                    'error': 'No data available'
                }
            
            latest_date = data.index[-1]
            data_age_hours = (datetime.now() - latest_date.to_pydatetime()).total_seconds() / 3600
            
            # 计算数据质量指标
            price_gaps = (data['Close'].pct_change().abs() > 0.1).sum()  # 大于10%的价格跳空
            zero_volume_days = (data['Volume'] == 0).sum()
            
            data_quality = 1.0
            if price_gaps > 0:
                data_quality -= 0.2 * price_gaps / len(data)
            if zero_volume_days > 0:
                data_quality -= 0.3 * zero_volume_days / len(data)
            
            return {
                'symbol': symbol,
                'data_available': True,
                'latest_date': latest_date.strftime('%Y-%m-%d %H:%M:%S'),
                'data_age_hours': data_age_hours,
                'within_delay_window': data_age_hours <= (self.config.data_delay_minutes / 60 + 24),  # 延迟+1天缓冲
                'data_points': len(data),
                'price_gaps': price_gaps,
                'zero_volume_days': zero_volume_days,
                'data_quality_score': max(0.0, data_quality),
                'delay_minutes': self.config.data_delay_minutes,
                'polygon_connected': self.client is not None
            }
            
        except Exception as e:
            logger.error(f"Data validation failed for {symbol}: {e}")
            return {
                'symbol': symbol,
                'data_available': False,
                'error': str(e),
                'polygon_connected': self.client is not None
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['cache_size'] = len(self.cache)
        stats['config'] = self.config.__dict__
        stats['polygon_available'] = self.client is not None
        return stats
    
    def clear_cache(self):
        """清理缓存"""
        self.cache.clear()
        logger.info("Factor cache cleared")

# 全局实例
_unified_factors_instance = None

def get_unified_polygon_factors(config: DelayedDataConfig = None) -> UnifiedPolygonFactors:
    """获取统一因子库实例"""
    global _unified_factors_instance
    if _unified_factors_instance is None:
        _unified_factors_instance = UnifiedPolygonFactors(config)
    return _unified_factors_instance

# 便捷函数 - AutoTrader引擎兼容接口
def zscore(values: List[float], n: int = 20) -> List[float]:
    """Z-Score计算 - 向后兼容"""
    factors = get_unified_polygon_factors()
    return factors.calculate_zscore(values, n)

def atr(highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> List[float]:
    """ATR计算 - 向后兼容"""
    factors = get_unified_polygon_factors()
    return factors.calculate_atr(highs, lows, closes, n)

def sma(values: List[float], n: int) -> List[float]:
    """简单移动平均 - 向后兼容autotrader.factors"""
    out: List[float] = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= n:
            s -= values[i - n]
        out.append(s / n if i >= n - 1 else math.nan)
    return out

def stddev(values: List[float], n: int) -> List[float]:
    """标准差计算 - 向后兼容autotrader.factors"""
    out: List[float] = []
    s = 0.0
    s2 = 0.0
    for i, v in enumerate(values):
        s += v
        s2 += v * v
        if i >= n:
            s -= values[i - n]
            s2 -= values[i - n] * values[i - n]
        if i >= n - 1:
            mean = s / n
            var = max(s2 / n - mean * mean, 0.0)
            out.append(math.sqrt(var))
        else:
            out.append(math.nan)
    return out

def rsi(values: List[float], n: int) -> List[float]:
    """RSI指标计算 - 向后兼容autotrader.factors"""
    gains = [0.0]
    losses = [0.0]
    for i in range(1, len(values)):
        chg = values[i] - values[i - 1]
        gains.append(max(chg, 0.0))
        losses.append(max(-chg, 0.0))
    avg_gain = sma(gains, n)
    avg_loss = sma(losses, n)
    out: List[float] = []
    for g, l in zip(avg_gain, avg_loss):
        if math.isnan(g) or math.isnan(l) or l == 0:
            out.append(math.nan)
        else:
            rs = g / l
            out.append(100.0 - 100.0 / (1.0 + rs))
    return out

def bollinger(values: List[float], n: int, k: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    """布林带计算 - 向后兼容autotrader.factors"""
    ma = sma(values, n)
    sd = stddev(values, n)
    upper: List[float] = []
    lower: List[float] = []
    for m, s in zip(ma, sd):
        if math.isnan(m) or math.isnan(s):
            upper.append(math.nan)
            lower.append(math.nan)
        else:
            upper.append(m + k * s)
            lower.append(m - k * s)
    return ma, upper, lower

def get_trading_signal_for_autotrader(symbol: str, threshold: float = 0.3) -> Dict[str, Any]:
    """为AutoTrader引擎提供交易信号"""
    factors = get_unified_polygon_factors()
    return factors.get_trading_signal(symbol, threshold)

# 兼容polygon_unified_factors.py的函数
def get_polygon_unified_factors():
    """兼容性函数 - 返回统一因子实例"""
    return get_unified_polygon_factors()

def enable_polygon_factors():
    """启用Polygon因子 - 兼容性函数"""
    manager = get_unified_polygon_factors()
    logger.info("Polygon因子已启用")

def enable_polygon_risk_balancer():
    """启用Polygon风险平衡器 - 兼容性函数"""
    manager = get_unified_polygon_factors()
    logger.info("Polygon风险平衡器已启用")

def disable_polygon_risk_balancer():
    """禁用Polygon风险平衡器 - 兼容性函数"""
    manager = get_unified_polygon_factors()
    logger.info("Polygon风险平衡器已禁用")

def check_polygon_trading_conditions(symbol: str) -> Dict[str, Any]:
    """检查Polygon交易条件 - 兼容性函数"""
    manager = get_unified_polygon_factors()
    validation = manager.validate_polygon_data(symbol)
    return {
        'trading_allowed': validation.get('data_available', False),
        'data_quality': validation.get('data_quality_score', 0.0),
        'last_update': validation.get('latest_data_time', 'Unknown'),
        'conditions_met': validation.get('data_available', False)
    }

def process_signals_with_polygon(signals) -> List[Dict]:
    """使用Polygon处理信号 - 兼容性函数"""
    manager = get_unified_polygon_factors()
    processed_signals = []
    
    # 处理不同输入格式
    if hasattr(signals, 'to_dict'):  # DataFrame
        signals_list = signals.to_dict('records')
    elif isinstance(signals, list):
        signals_list = signals
    else:
        signals_list = [signals]
    
    for signal in signals_list:
        if isinstance(signal, dict) and 'symbol' in signal:
            try:
                # 使用统一因子验证和增强信号
                enhanced_signal = manager.get_trading_signal(signal['symbol'])
                signal.update(enhanced_signal)
                processed_signals.append(signal)
            except Exception as e:
                logger.warning(f"处理信号失败 {signal.get('symbol', 'Unknown')}: {e}")
                processed_signals.append(signal)  # 保留原信号
        else:
            processed_signals.append(signal)
    
    return processed_signals

if __name__ == "__main__":
    # 测试代码
    print("AutoTrader统一因子库测试")
    print("=" * 50)
    
    factors = get_unified_polygon_factors()
    
    # 测试单个股票
    test_symbol = "AAPL"
    print(f"测试股票: {test_symbol}")
    
    # 数据验证
    validation = factors.validate_polygon_data(test_symbol)
    print(f"数据验证: {validation}")
    
    if validation['data_available']:
        # 交易信号
        signal = factors.get_trading_signal(test_symbol)
        print(f"交易信号: {signal}")
        
        # 统计信息
        stats = factors.get_stats()
        print(f"统计信息: {stats}")
    else:
        print("数据不可用，跳过信号测试")