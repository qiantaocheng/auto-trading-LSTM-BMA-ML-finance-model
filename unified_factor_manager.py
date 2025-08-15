#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一因子管理器
整合Barra风格因子、Polygon因子和AutoTrader因子
提供统一的因子计算接口，避免代码重复

主要功能：
1. 统一因子计算接口
2. 避免重复计算和代码冗余
3. 智能缓存管理
4. 多数据源支持
5. 向后兼容现有系统
"""

import os
import sys
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

# 导入各个因子引擎
try:
    from barra_style_factors import BarraStyleFactors
    BARRA_AVAILABLE = True
except ImportError:
    BARRA_AVAILABLE = False
    
try:
    from polygon_complete_factors import CompletePolygonFactors
    POLYGON_COMPLETE_AVAILABLE = True
except ImportError:
    POLYGON_COMPLETE_AVAILABLE = False
    
try:
    from autotrader.unified_polygon_factors import UnifiedPolygonFactors
    AUTOTRADER_AVAILABLE = True
except ImportError:
    AUTOTRADER_AVAILABLE = False

logger = logging.getLogger(__name__)


class FactorCategory(Enum):
    """因子分类"""
    MOMENTUM = "momentum"
    VALUE = "value" 
    QUALITY = "quality"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    GROWTH = "growth"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MICROSTRUCTURE = "microstructure"


class DataSource(Enum):
    """数据源类型"""
    POLYGON = "polygon"
    BARRA = "barra"
    AUTOTRADER = "autotrader"
    AUTO = "auto"


@dataclass
class FactorResult:
    """统一因子结果格式"""
    factor_name: str
    category: FactorCategory
    value: float
    confidence: float
    timestamp: datetime
    symbol: str
    data_source: DataSource
    computation_time: float
    data_quality: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'category': self.category.value,
            'value': self.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'data_source': self.data_source.value,
            'computation_time': self.computation_time,
            'data_quality': self.data_quality,
            'metadata': self.metadata
        }


@dataclass
class FactorConfig:
    """因子配置"""
    enabled: bool = True
    priority: int = 1  # 1-5, 5最高
    cache_ttl: int = 300  # 缓存过期时间(秒)
    required_data_quality: float = 0.7  # 最低数据质量要求
    fallback_engines: List[DataSource] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


class BaseFactor(ABC):
    """基础因子抽象类"""
    
    def __init__(self, name: str, category: FactorCategory, config: FactorConfig = None):
        self.name = name
        self.category = category
        self.config = config or FactorConfig()
        
    @abstractmethod
    def calculate(self, symbol: str, data: pd.DataFrame, **kwargs) -> FactorResult:
        """计算因子值"""
        pass
    
    @abstractmethod
    def get_required_data(self) -> List[str]:
        """获取所需数据类型"""
        pass
    
    @abstractmethod
    def validate_input(self, data: pd.DataFrame) -> bool:
        """验证输入数据质量"""
        pass


class SharedCalculations:
    """共享计算函数库，避免重复实现"""
    
    @staticmethod
    def zscore(values: pd.Series, window: int = 20) -> pd.Series:
        """统一Z-Score计算"""
        rolling_mean = values.rolling(window=window).mean()
        rolling_std = values.rolling(window=window).std()
        return (values - rolling_mean) / rolling_std
    
    @staticmethod
    def moving_average(values: pd.Series, window: int, ma_type: str = 'sma') -> pd.Series:
        """统一移动平均计算"""
        if ma_type == 'sma':
            return values.rolling(window=window).mean()
        elif ma_type == 'ema':
            return values.ewm(span=window).mean()
        elif ma_type == 'wma':
            weights = np.arange(1, window + 1)
            return values.rolling(window=window).apply(lambda x: np.dot(x, weights) / weights.sum())
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")
    
    @staticmethod
    def volatility(returns: pd.Series, window: int = 20, 
                  vol_type: str = 'realized') -> pd.Series:
        """统一波动率计算"""
        if vol_type == 'realized':
            return returns.rolling(window=window).std() * np.sqrt(252)
        elif vol_type == 'parkinson':
            # 需要高低价数据
            raise NotImplementedError("Parkinson estimator requires high/low data")
        elif vol_type == 'garman_klass':
            # 需要OHLC数据
            raise NotImplementedError("Garman-Klass estimator requires OHLC data")
        else:
            return returns.rolling(window=window).std() * np.sqrt(252)
    
    @staticmethod
    def beta_calculation(stock_returns: pd.Series, 
                        market_returns: pd.Series, window: int = 252) -> float:
        """统一Beta计算"""
        if len(stock_returns) < window or len(market_returns) < window:
            return 1.0  # 默认beta
        
        # 对齐数据
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        if len(aligned_data) < window:
            return 1.0
            
        stock_aligned = aligned_data.iloc[:, 0]
        market_aligned = aligned_data.iloc[:, 1]
        
        covariance = stock_aligned.rolling(window=window).cov(market_aligned).iloc[-1]
        market_variance = market_aligned.rolling(window=window).var().iloc[-1]
        
        if market_variance == 0:
            return 1.0
            
        return covariance / market_variance
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """统一RSI计算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, 
                       num_std: float = 2) -> Dict[str, pd.Series]:
        """统一布林带计算"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        return {
            'upper': sma + (std * num_std),
            'middle': sma,
            'lower': sma - (std * num_std),
            'bandwidth': (std * num_std * 2) / sma,
            'percent_b': (prices - (sma - std * num_std)) / (std * num_std * 2)
        }


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache/factors"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def _get_cache_key(self, symbol: str, factor_name: str, params: Dict = None) -> str:
        """生成缓存键"""
        key_data = f"{symbol}_{factor_name}_{str(params or {})}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, symbol: str, factor_name: str, params: Dict = None, 
            ttl: int = 300) -> Optional[FactorResult]:
        """获取缓存值"""
        cache_key = self._get_cache_key(symbol, factor_name, params)
        
        # 检查内存缓存
        if cache_key in self.memory_cache:
            cached_item = self.memory_cache[cache_key]
            if (datetime.now() - cached_item['timestamp']).seconds < ttl:
                self.cache_stats['hits'] += 1
                return cached_item['result']
        
        # 检查磁盘缓存
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if (datetime.now() - cached_data['timestamp']).seconds < ttl:
                        # 加载到内存缓存
                        self.memory_cache[cache_key] = cached_data
                        self.cache_stats['hits'] += 1
                        return cached_data['result']
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, symbol: str, factor_name: str, result: FactorResult, 
            params: Dict = None):
        """设置缓存值"""
        cache_key = self._get_cache_key(symbol, factor_name, params)
        cache_data = {
            'result': result,
            'timestamp': datetime.now()
        }
        
        # 设置内存缓存
        self.memory_cache[cache_key] = cache_data
        
        # 设置磁盘缓存
        try:
            import pickle
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache file: {e}")
    
    def clear_expired(self, ttl: int = 300):
        """清理过期缓存"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, cached_item in self.memory_cache.items():
            if (current_time - cached_item['timestamp']).seconds > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache)
        }


class UnifiedFactorManager:
    """统一因子管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.cache_manager = CacheManager()
        self.engines = {}
        self.factor_registry = {}
        self.config = self._load_config(config_path)
        self._initialize_engines()
        self._register_factors()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            'engines': {
                'barra': {'enabled': BARRA_AVAILABLE, 'priority': 3},
                'polygon': {'enabled': POLYGON_COMPLETE_AVAILABLE, 'priority': 2},
                'autotrader': {'enabled': AUTOTRADER_AVAILABLE, 'priority': 1}
            },
            'cache': {
                'default_ttl': 300,
                'max_memory_items': 1000
            },
            'data_quality': {
                'min_quality_threshold': 0.7,
                'max_staleness_hours': 24
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_engines(self):
        """初始化各个因子引擎"""
        if BARRA_AVAILABLE and self.config['engines']['barra']['enabled']:
            try:
                self.engines['barra'] = BarraStyleFactors()
                logger.info("Barra因子引擎初始化成功")
            except Exception as e:
                logger.error(f"Barra因子引擎初始化失败: {e}")
        
        if POLYGON_COMPLETE_AVAILABLE and self.config['engines']['polygon']['enabled']:
            try:
                self.engines['polygon'] = CompletePolygonFactors()
                logger.info("Polygon因子引擎初始化成功")
            except Exception as e:
                logger.error(f"Polygon因子引擎初始化失败: {e}")
        
        if AUTOTRADER_AVAILABLE and self.config['engines']['autotrader']['enabled']:
            try:
                self.engines['autotrader'] = UnifiedPolygonFactors()
                logger.info("AutoTrader因子引擎初始化成功")
            except Exception as e:
                logger.error(f"AutoTrader因子引擎初始化失败: {e}")
    
    def _register_factors(self):
        """注册所有可用因子"""
        
        # Barra风格因子映射
        barra_factors = {
            # 动量因子
            'momentum_12_1': FactorCategory.MOMENTUM,
            'momentum_6_1': FactorCategory.MOMENTUM,
            'momentum_3_1': FactorCategory.MOMENTUM,
            'momentum_short': FactorCategory.MOMENTUM,
            'price_trend': FactorCategory.MOMENTUM,
            
            # 价值因子
            'book_to_price': FactorCategory.VALUE,
            'earnings_to_price': FactorCategory.VALUE,
            'sales_to_price': FactorCategory.VALUE,
            'cash_earnings_to_price': FactorCategory.VALUE,
            
            # 质量因子
            'roe': FactorCategory.QUALITY,
            'gross_profitability': FactorCategory.QUALITY,
            'accruals': FactorCategory.QUALITY,
            'debt_to_equity': FactorCategory.QUALITY,
            'earnings_quality': FactorCategory.QUALITY,
            
            # 波动率因子
            'volatility_90d': FactorCategory.VOLATILITY,
            'volatility_60d': FactorCategory.VOLATILITY,
            'residual_volatility': FactorCategory.VOLATILITY,
            'downside_volatility': FactorCategory.VOLATILITY,
            
            # 流动性因子
            'amihud_illiquidity': FactorCategory.LIQUIDITY,
            'turnover_rate': FactorCategory.LIQUIDITY,
            'trading_volume_ratio': FactorCategory.LIQUIDITY,
            
            # 成长因子
            'asset_growth': FactorCategory.GROWTH,
            'sales_growth': FactorCategory.GROWTH,
            'earnings_growth': FactorCategory.GROWTH,
        }
        
        # Polygon因子映射
        polygon_factors = {
            # 动量因子
            'momentum_12_1': FactorCategory.MOMENTUM,
            'momentum_6_1': FactorCategory.MOMENTUM,
            'week52_high_proximity': FactorCategory.MOMENTUM,
            'residual_momentum': FactorCategory.MOMENTUM,
            
            # 基本面因子
            'earnings_surprise': FactorCategory.FUNDAMENTAL,
            'ebit_ev_yield': FactorCategory.FUNDAMENTAL,
            'fcf_yield': FactorCategory.FUNDAMENTAL,
            'gross_margin': FactorCategory.FUNDAMENTAL,
            
            # 盈利能力因子
            'earnings_yield': FactorCategory.VALUE,
            'sales_yield': FactorCategory.VALUE,
            'roe_quality': FactorCategory.QUALITY,
            'roic_quality': FactorCategory.QUALITY,
            
            # 风险因子
            'idiosyncratic_volatility': FactorCategory.VOLATILITY,
            'beta_anomaly': FactorCategory.VOLATILITY,
            
            # 微观结构因子
            'turnover_hump': FactorCategory.MICROSTRUCTURE,
            'effective_spread': FactorCategory.MICROSTRUCTURE,
        }
        
        # AutoTrader因子映射 (修正：基于实际的方法名)
        autotrader_factors = {
            'mean_reversion_signal': FactorCategory.MOMENTUM,
            'momentum_signal': FactorCategory.MOMENTUM,
            'trend_signal': FactorCategory.TECHNICAL,
            'volume_signal': FactorCategory.LIQUIDITY,
            'volatility_signal': FactorCategory.VOLATILITY,
            'composite_signal': FactorCategory.TECHNICAL,
            # 技术指标计算方法
            'zscore': FactorCategory.TECHNICAL,
            'atr': FactorCategory.TECHNICAL,
            'sma': FactorCategory.TECHNICAL,
        }
        
        # 注册因子
        for engine_name, factor_map in [
            ('barra', barra_factors),
            ('polygon', polygon_factors), 
            ('autotrader', autotrader_factors)
        ]:
            if engine_name in self.engines:
                for factor_name, category in factor_map.items():
                    self.factor_registry[f"{engine_name}_{factor_name}"] = {
                        'engine': engine_name,
                        'factor_name': factor_name,
                        'category': category,
                        'priority': self.config['engines'][engine_name]['priority']
                    }
    
    def get_available_factors(self, category: Optional[FactorCategory] = None, 
                            engine: Optional[str] = None) -> List[str]:
        """获取可用因子列表"""
        factors = []
        for full_name, info in self.factor_registry.items():
            if category and info['category'] != category:
                continue
            if engine and info['engine'] != engine:
                continue
            factors.append(full_name)
        
        return sorted(factors)
    
    def calculate_factor(self, factor_name: str, symbol: str, 
                        engine: str = 'auto', use_cache: bool = True,
                        **kwargs) -> Optional[FactorResult]:
        """计算单个因子"""
        start_time = time.time()
        
        # 检查缓存
        if use_cache:
            cached_result = self.cache_manager.get(
                symbol, factor_name, kwargs,
                ttl=self.config['cache']['default_ttl']
            )
            if cached_result:
                logger.debug(f"Cache hit for {symbol}:{factor_name}")
                return cached_result
        
        # 确定使用的引擎
        if engine == 'auto':
            engine = self._select_best_engine(factor_name)
        
        if not engine or engine not in self.engines:
            logger.error(f"Engine {engine} not available for factor {factor_name}")
            return None
        
        try:
            # 调用相应引擎计算因子
            result = self._calculate_with_engine(engine, factor_name, symbol, **kwargs)
            
            if result and use_cache:
                self.cache_manager.set(symbol, factor_name, result, kwargs)
            
            computation_time = time.time() - start_time
            if result:
                result.computation_time = computation_time
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate {factor_name} for {symbol}: {e}")
            return None
    
    def calculate_factor_set(self, factor_names: List[str], symbol: str,
                           use_cache: bool = True, **kwargs) -> Dict[str, FactorResult]:
        """批量计算因子"""
        results = {}
        
        for factor_name in factor_names:
            try:
                result = self.calculate_factor(
                    factor_name, symbol, use_cache=use_cache, **kwargs
                )
                if result:
                    results[factor_name] = result
            except Exception as e:
                logger.error(f"Failed to calculate {factor_name}: {e}")
                continue
        
        return results
    
    def _select_best_engine(self, factor_name: str) -> Optional[str]:
        """选择最佳引擎"""
        # 查找所有可用的引擎
        available_engines = []
        
        for full_name, info in self.factor_registry.items():
            if info['factor_name'] in factor_name or factor_name in full_name:
                available_engines.append(info)
        
        if not available_engines:
            return None
        
        # 按优先级排序，选择最高优先级的引擎
        available_engines.sort(key=lambda x: x['priority'], reverse=True)
        return available_engines[0]['engine']
    
    def _calculate_with_engine(self, engine: str, factor_name: str, 
                             symbol: str, **kwargs) -> Optional[FactorResult]:
        """使用指定引擎计算因子"""
        engine_instance = self.engines.get(engine)
        if not engine_instance:
            return None
        
        try:
            if engine == 'barra':
                return self._calculate_barra_factor(engine_instance, factor_name, symbol, **kwargs)
            elif engine == 'polygon':
                return self._calculate_polygon_factor(engine_instance, factor_name, symbol, **kwargs)
            elif engine == 'autotrader':
                return self._calculate_autotrader_factor(engine_instance, factor_name, symbol, **kwargs)
            else:
                logger.error(f"Unknown engine: {engine}")
                return None
                
        except Exception as e:
            logger.error(f"Engine {engine} calculation failed: {e}")
            return None
    
    def _calculate_barra_factor(self, engine, factor_name: str, 
                              symbol: str, **kwargs) -> Optional[FactorResult]:
        """计算Barra因子"""
        # 这里需要根据实际的BarraStyleFactors接口调整
        try:
            # 假设engine有对应的方法
            if hasattr(engine, factor_name):
                method = getattr(engine, factor_name)
                value = method(symbol, **kwargs)
                
                return FactorResult(
                    factor_name=factor_name,
                    category=self.factor_registry.get(f"barra_{factor_name}", {}).get('category', FactorCategory.TECHNICAL),
                    value=float(value) if value is not None else 0.0,
                    confidence=0.8,  # 默认置信度
                    timestamp=datetime.now(),
                    symbol=symbol,
                    data_source=DataSource.BARRA,
                    computation_time=0.0,
                    data_quality=0.9  # 假设Barra数据质量较高
                )
        except Exception as e:
            logger.error(f"Barra factor calculation failed: {e}")
            return None
    
    def _calculate_polygon_factor(self, engine, factor_name: str, 
                                symbol: str, **kwargs) -> Optional[FactorResult]:
        """计算Polygon因子"""
        try:
            # 根据实际的CompletePolygonFactors接口调整
            if hasattr(engine, factor_name):
                method = getattr(engine, factor_name)
                result = method(symbol, **kwargs)
                
                if hasattr(result, 'value'):
                    return FactorResult(
                        factor_name=factor_name,
                        category=self.factor_registry.get(f"polygon_{factor_name}", {}).get('category', FactorCategory.TECHNICAL),
                        value=float(result.value),
                        confidence=getattr(result, 'confidence', 0.7),
                        timestamp=datetime.now(),
                        symbol=symbol,
                        data_source=DataSource.POLYGON,
                        computation_time=getattr(result, 'computation_time', 0.0),
                        data_quality=getattr(result, 'data_quality', 0.8)
                    )
        except Exception as e:
            logger.error(f"Polygon factor calculation failed: {e}")
            return None
    
    def _calculate_autotrader_factor(self, engine, factor_name: str, 
                                   symbol: str, **kwargs) -> Optional[FactorResult]:
        """计算AutoTrader因子（修正：支持实际的UnifiedPolygonFactors接口）"""
        try:
            # 方法名映射
            method_mapping = {
                'mean_reversion_signal': 'calculate_mean_reversion_signal',
                'momentum_signal': 'calculate_momentum_signal',
                'trend_signal': 'calculate_trend_signal',
                'volume_signal': 'calculate_volume_signal',
                'volatility_signal': 'calculate_volatility_signal',
                'composite_signal': 'calculate_composite_signal',
                'zscore': 'calculate_zscore',
                'atr': 'calculate_atr',
                'sma': 'calculate_sma'
            }
            
            method_name = method_mapping.get(factor_name, f"calculate_{factor_name}")
            
            if hasattr(engine, method_name):
                method = getattr(engine, method_name)
                
                # 对于信号类方法，它们返回FactorResult对象
                if factor_name.endswith('_signal'):
                    autotrader_result = method(symbol, **kwargs)
                    
                    # 转换为统一的FactorResult格式
                    return FactorResult(
                        factor_name=factor_name,
                        category=self.factor_registry.get(f"autotrader_{factor_name}", {}).get('category', FactorCategory.TECHNICAL),
                        value=autotrader_result.value if autotrader_result else 0.0,
                        confidence=autotrader_result.confidence if autotrader_result else 0.0,
                        timestamp=datetime.now(),
                        symbol=symbol,
                        data_source=DataSource.AUTOTRADER,
                        computation_time=0.0,
                        data_quality=autotrader_result.data_quality_score if autotrader_result else 0.0,
                        metadata=autotrader_result.metadata if autotrader_result else {}
                    )
                
                # 对于技术指标方法，需要先获取数据
                elif factor_name in ['zscore', 'atr', 'sma']:
                    # 获取市场数据
                    market_data = engine.get_market_data(symbol)
                    if market_data.empty:
                        return None
                    
                    if factor_name == 'zscore':
                        closes = market_data['Close'].tolist()
                        result_values = method(closes, kwargs.get('n', 20))
                        value = result_values[-1] if result_values else 0.0
                    elif factor_name == 'atr':
                        highs = market_data['High'].tolist()
                        lows = market_data['Low'].tolist()
                        closes = market_data['Close'].tolist()
                        result_values = method(highs, lows, closes, kwargs.get('n', 14))
                        value = result_values[-1] if result_values else 0.0
                    elif factor_name == 'sma':
                        closes = market_data['Close'].tolist()
                        result_values = method(closes, kwargs.get('n', 20))
                        value = result_values[-1] if result_values else 0.0
                    else:
                        value = 0.0
                    
                    return FactorResult(
                        factor_name=factor_name,
                        category=self.factor_registry.get(f"autotrader_{factor_name}", {}).get('category', FactorCategory.TECHNICAL),
                        value=float(value) if not np.isnan(value) else 0.0,
                        confidence=0.8,
                        timestamp=datetime.now(),
                        symbol=symbol,
                        data_source=DataSource.AUTOTRADER,
                        computation_time=0.0,
                        data_quality=0.9
                    )
                else:
                    logger.warning(f"Unknown AutoTrader factor type: {factor_name}")
                    return None
            else:
                logger.error(f"Method {method_name} not found in AutoTrader engine")
                return None
                
        except Exception as e:
            logger.error(f"AutoTrader factor calculation failed for {factor_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_factor_info(self, factor_name: str) -> Optional[Dict[str, Any]]:
        """获取因子信息"""
        for full_name, info in self.factor_registry.items():
            if factor_name in full_name or info['factor_name'] == factor_name:
                return info
        return None
    
    def get_engine_status(self) -> Dict[str, Dict[str, Any]]:
        """获取引擎状态"""
        status = {}
        for engine_name, engine_instance in self.engines.items():
            status[engine_name] = {
                'available': engine_instance is not None,
                'priority': self.config['engines'][engine_name]['priority'],
                'enabled': self.config['engines'][engine_name]['enabled']
            }
        
        # 添加缓存统计
        status['cache'] = self.cache_manager.get_stats()
        
        return status
    
    def cleanup_cache(self):
        """清理过期缓存"""
        self.cache_manager.clear_expired(self.config['cache']['default_ttl'])


# 创建全局实例
_global_factor_manager = None

def get_unified_factor_manager() -> UnifiedFactorManager:
    """获取全局统一因子管理器实例"""
    global _global_factor_manager
    if _global_factor_manager is None:
        _global_factor_manager = UnifiedFactorManager()
    return _global_factor_manager


# 便捷函数
def calculate_factor(factor_name: str, symbol: str, **kwargs) -> Optional[FactorResult]:
    """便捷因子计算函数"""
    manager = get_unified_factor_manager()
    return manager.calculate_factor(factor_name, symbol, **kwargs)

def calculate_factors(factor_names: List[str], symbol: str, **kwargs) -> Dict[str, FactorResult]:
    """便捷批量因子计算函数"""
    manager = get_unified_factor_manager()
    return manager.calculate_factor_set(factor_names, symbol, **kwargs)

def get_available_factors(category: Optional[str] = None) -> List[str]:
    """便捷获取可用因子函数"""
    manager = get_unified_factor_manager()
    factor_category = None
    if category:
        try:
            factor_category = FactorCategory(category.lower())
        except ValueError:
            pass
    return manager.get_available_factors(factor_category)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    manager = UnifiedFactorManager()
    
    print("可用引擎:", list(manager.engines.keys()))
    print("引擎状态:", manager.get_engine_status())
    print("可用因子数量:", len(manager.get_available_factors()))
    
    # 测试因子计算
    test_symbol = "AAPL"
    available_factors = manager.get_available_factors()
    
    if available_factors:
        test_factor = available_factors[0]
        print(f"\n测试计算因子: {test_factor}")
        
        result = manager.calculate_factor(test_factor, test_symbol)
        if result:
            print("计算结果:", result.to_dict())
        else:
            print("计算失败")
    
    print("\n缓存统计:", manager.cache_manager.get_stats())