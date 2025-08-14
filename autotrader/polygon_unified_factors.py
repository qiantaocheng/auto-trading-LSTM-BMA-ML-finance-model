#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoTrader专usePolygon统一因子入口
集成所hasPolygon因子功能，提供统一接口
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加 items目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from polygon_client import polygon_client
    from polygon_factors import PolygonFactorIntegrator, PolygonShortTermFactors
    from polygon_complete_factors import PolygonCompleteFactors
    from risk_reward_balancer_integrated import RiskRewardBalancer, Config as RiskBalancerConfig
    from ibkr_risk_balancer_adapter import (
        get_risk_balancer_adapter, 
        enable_risk_balancer, 
        disable_risk_balancer,
        is_risk_balancer_enabled
    )
except ImportError as e:
    logging.warning(f"导入Polygon模块failed: {e}")

logger = logging.getLogger(__name__)

class PolygonUnifiedFactors:
    """
    AutoTrader专usePolygon统一因子管理器
    集成所has因子计算、risk controland交易决策功能
    """
    
    def __init__(self):
        """初始化统一因子管理器"""
        self.enabled = False
        self.risk_balancer_enabled = False
        
        # 初始化各个组件
        self.polygon_client = None
        self.factor_integrator = None
        self.short_term_factors = None
        self.complete_factors = None
        self.risk_balancer = None
        
        # 缓存
        self.factor_cache = {}
        self.cache_ttl = 300  # 5分钟缓存
        
        # 统计信息
        self.stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'cache_hits': 0,
            'last_update': datetime.now()
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """初始化所has组件 - 使用lazy loading避免重复初始化"""
        if self.enabled:
            logger.info("Polygon统一因子管理器already initialized, skipping")
            return
            
        try:
            # 初始化Polygon客户端
            if 'polygon_client' in globals() and self.polygon_client is None:
                self.polygon_client = polygon_client
                logger.info("Polygon客户端connection")
            
            # 使用lazy loading for因子组件 - 只在需要时初始化
            # 不在启动时自动创建所有组件
            logger.info("Polygon统一因子管理器初始化completed (lazy loading enabled)")
            
            # 初始化risk control收益平衡器 (轻量级组件)
            if 'get_risk_balancer_adapter' in globals() and self.risk_balancer is None:
                self.risk_balancer = get_risk_balancer_adapter()
                logger.info("risk control收益平衡器初始化")
            
            self.enabled = True
            
        except Exception as e:
            logger.error(f"初始化failed: {e}")
            self.enabled = False
    
    def is_enabled(self) -> bool:
        """checkis否启use"""
        return self.enabled and self.polygon_client is not None
    
    def _ensure_factor_integrator(self):
        """按需初始化因子集成器"""
        if self.factor_integrator is None and 'PolygonFactorIntegrator' in globals():
            self.factor_integrator = PolygonFactorIntegrator()
            logger.info("Polygon因子集成器 initialized on demand")
    
    def _ensure_short_term_factors(self):
        """按需初始化短期因子"""
        if self.short_term_factors is None and 'PolygonShortTermFactors' in globals():
            self.short_term_factors = PolygonShortTermFactors()
            logger.info("Polygon短期因子 initialized on demand")
    
    def _ensure_complete_factors(self):
        """按需初始化完整因子库"""
        if self.complete_factors is None and 'PolygonCompleteFactors' in globals():
            self.complete_factors = PolygonCompleteFactors()
            logger.info("Polygon完整因子库 initialized on demand")
    
    def enable_risk_balancer(self):
        """启userisk control收益平衡器"""
        try:
            if 'enable_risk_balancer' in globals():
                enable_risk_balancer()
                self.risk_balancer_enabled = True
                logger.info("risk control收益平衡器启use")
        except Exception as e:
            logger.error(f"启userisk control收益平衡器failed: {e}")
    
    def disable_risk_balancer(self):
        """禁userisk control收益平衡器"""
        try:
            if 'disable_risk_balancer' in globals():
                disable_risk_balancer()
                self.risk_balancer_enabled = False
                logger.info("risk control收益平衡器禁use")
        except Exception as e:
            logger.error(f"禁userisk control收益平衡器failed: {e}")
    
    def is_risk_balancer_enabled(self) -> bool:
        """checkrisk control收益平衡器状态"""
        try:
            if 'is_risk_balancer_enabled' in globals():
                return is_risk_balancer_enabled()
        except:
            pass
        return self.risk_balancer_enabled
    
    def get_stock_basic_info(self, symbol: str) -> Optional[Dict]:
        """retrieval股票基础信息"""
        if not self.is_enabled():
            return None
        
        cache_key = f"{symbol}_basic_info"
        if cache_key in self.factor_cache:
            cached_time, cached_data = self.factor_cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                self.stats['cache_hits'] += 1
                return cached_data
        
        try:
            info = self.polygon_client.get_ticker_details(symbol)
            if info:
                # 缓存结果
                self.factor_cache[cache_key] = (datetime.now().timestamp(), info)
                self.stats['successful_calculations'] += 1
                return info
        except Exception as e:
            logger.error(f"retrieval{symbol}基础信息failed: {e}")
            self.stats['failed_calculations'] += 1
        
        self.stats['total_calculations'] += 1
        return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """retrieval历史数据"""
        if not self.is_enabled():
            return None
        
        cache_key = f"{symbol}_hist_{days}d"
        if cache_key in self.factor_cache:
            cached_time, cached_data = self.factor_cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                self.stats['cache_hits'] += 1
                return cached_data
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = self.polygon_client.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            if data is not None and len(data) > 0:
                # 缓存结果
                self.factor_cache[cache_key] = (datetime.now().timestamp(), data)
                self.stats['successful_calculations'] += 1
                return data
                
        except Exception as e:
            logger.error(f"retrieval{symbol}历史数据failed: {e}")
            self.stats['failed_calculations'] += 1
        
        self.stats['total_calculations'] += 1
        return None
    
    def calculate_short_term_factors(self, symbol: str) -> Dict[str, Any]:
        """计算T+5短期因子"""
        if not self.is_enabled() or not self.short_term_factors:
            return {}
        
        try:
            factors = self.short_term_factors.calculate_all_short_term_factors(symbol)
            self.stats['successful_calculations'] += 1
            return factors
        except Exception as e:
            logger.error(f"计算{symbol}短期因子failed: {e}")
            self.stats['failed_calculations'] += 1
            return {}
    
    def calculate_complete_factors(self, symbol: str, categories: List[str] = None) -> Dict[str, Any]:
        """计算完整因子(48个专业因子)"""
        if not self.is_enabled() or not self.complete_factors:
            return {}
        
        categories = categories or ['momentum', 'fundamental', 'profitability', 'quality', 'risk', 'microstructure']
        
        try:
            factors = self.complete_factors.calculate_all_complete_factors(symbol, categories)
            self.stats['successful_calculations'] += 1
            return factors
        except Exception as e:
            logger.error(f"计算{symbol}完整因子failed: {e}")
            self.stats['failed_calculations'] += 1
            return {}
    
    def create_factor_matrix(self, symbols: List[str], factor_types: List[str] = None) -> pd.DataFrame:
        """创建因子矩阵"""
        if not self.is_enabled() or not self.factor_integrator:
            return pd.DataFrame()
        
        try:
            matrix = self.factor_integrator.create_factor_matrix(symbols, factor_types)
            self.stats['successful_calculations'] += 1
            return matrix
        except Exception as e:
            logger.error(f"创建因子矩阵failed: {e}")
            self.stats['failed_calculations'] += 1
            return pd.DataFrame()
    
    def check_trading_conditions(self, symbol: str) -> Dict[str, Any]:
        """
        check交易 records件(替代原has超短期判断)
        使usePolygon数据进行流动性、价差、动量等check
        """
        if not self.is_enabled():
            return {'can_trade': False, 'reason': 'Polygon未启use'}
        
        try:
            # retrieval基础信息
            basic_info = self.get_stock_basic_info(symbol)
            if not basic_info:
                return {'can_trade': False, 'reason': 'no法retrieval基础信息'}
            
            # retrieval历史数据
            hist_data = self.get_historical_data(symbol, days=30)
            if hist_data is None or len(hist_data) < 20:
                return {'can_trade': False, 'reason': '历史数据not足'}
            
            # pricecheck
            current_price = float(hist_data['Close'].iloc[-1])
            if current_price < 5.0:
                return {'can_trade': False, 'reason': f'price过低: ${current_price:.2f}'}
            
            # execution量check
            avg_volume = hist_data['Volume'].rolling(20).mean().iloc[-1]
            avg_dollar_volume = avg_volume * current_price
            if avg_dollar_volume < 500000:  # 50万美元
                return {'can_trade': False, 'reason': f'execution量not足: ${avg_dollar_volume:.0f}'}
            
            # 波动率check
            returns = hist_data['Close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]
            if pd.isna(volatility) or volatility > 0.1:  # 10%日波动率
                return {'can_trade': False, 'reason': f'波动率过高: {volatility:.2%}'}
            
            # 动量check(简化版)
            if len(hist_data) >= 10:
                momentum_5d = (current_price / hist_data['Close'].iloc[-6]) - 1
                momentum_20d = (current_price / hist_data['Close'].iloc[-21]) - 1 if len(hist_data) >= 21 else 0
            else:
                momentum_5d = momentum_20d = 0
            
            # 市值check
            market_cap = basic_info.get('market_cap', 0)
            
            return {
                'can_trade': True,
                'reason': '通过所hascheck',
                'current_price': current_price,
                'avg_dollar_volume': avg_dollar_volume,
                'volatility': volatility,
                'momentum_5d': momentum_5d,
                'momentum_20d': momentum_20d,
                'market_cap': market_cap,
                'is_large_cap': market_cap > 10e9
            }
            
        except Exception as e:
            logger.error(f"check{symbol}交易 records件failed: {e}")
            return {'can_trade': False, 'reason': f'checkfailed: {e}'}
    
    def process_signals_with_risk_control(self, signals: Union[pd.DataFrame, List[Dict]]) -> List[Dict]:
        """
        使userisk control收益平衡器处理信号
        """
        if not self.is_risk_balancer_enabled() or not self.risk_balancer:
            logger.info("risk control收益平衡器未启use，使use基础处理")
            return self._process_signals_basic(signals)
        
        try:
            from ibkr_risk_balancer_adapter import process_bma_signals_with_risk_control
            
            # 转换asDataFrame格式
            if isinstance(signals, list):
                df_signals = pd.DataFrame(signals)
            else:
                df_signals = signals
            
            orders = process_bma_signals_with_risk_control(df_signals)
            logger.info(f"risk control处理{len(df_signals)}个信号，生成{len(orders)}个订单")
            return orders
            
        except Exception as e:
            logger.error(f"risk control信号处理failed: {e}")
            return self._process_signals_basic(signals)
    
    def _process_signals_basic(self, signals: Union[pd.DataFrame, List[Dict]]) -> List[Dict]:
        """基础信号处理(norisk control)"""
        orders = []
        
        try:
            if isinstance(signals, pd.DataFrame):
                signal_data = signals.to_dict('records')
            else:
                signal_data = signals
            
            for signal in signal_data:
                symbol = signal.get('symbol', '')
                prediction = signal.get('weighted_prediction', 0)
                
                # check交易 records件
                conditions = self.check_trading_conditions(symbol)
                if not conditions['can_trade']:
                    logger.info(f"跳过{symbol}: {conditions['reason']}")
                    continue
                
                # 简单阈值过滤
                if abs(prediction) < 0.005:  # 0.5%
                    continue
                
                side = "BUY" if prediction > 0 else "SELL"
                
                # 简单数量计算
                quantity = min(100, int(conditions['avg_dollar_volume'] / conditions['current_price'] * 0.001))
                
                orders.append({
                    'symbol': symbol,
                    'side': side,
                    'quantity': max(quantity, 50),
                    'order_type': 'MKT',
                    'source': 'polygon_basic',
                    'prediction': prediction,
                    'conditions': conditions
                })
            
            logger.info(f"基础处理生成{len(orders)}个订单")
            
        except Exception as e:
            logger.error(f"基础信号处理failed: {e}")
        
        return orders
    
    def get_stats(self) -> Dict[str, Any]:
        """retrieval统计信息"""
        stats = self.stats.copy()
        stats['enabled'] = self.is_enabled()
        stats['risk_balancer_enabled'] = self.is_risk_balancer_enabled()
        stats['cache_size'] = len(self.factor_cache)
        
        # 添加组件状态
        stats['components'] = {
            'polygon_client': self.polygon_client is not None,
            'factor_integrator': self.factor_integrator is not None,
            'short_term_factors': self.short_term_factors is not None,
            'complete_factors': self.complete_factors is not None,
            'risk_balancer': self.risk_balancer is not None
        }
        
        return stats
    
    def clear_cache(self):
        """清理缓存"""
        self.factor_cache.clear()
        logger.info("因子缓存清理")
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'cache_hits': 0,
            'last_update': datetime.now()
        }
        logger.info("统计信息重置")

# 全局单例
_polygon_unified_instance = None

def get_polygon_unified_factors() -> PolygonUnifiedFactors:
    """retrievalPolygon统一因子管理器单例"""
    global _polygon_unified_instance
    if _polygon_unified_instance is None:
        _polygon_unified_instance = PolygonUnifiedFactors()
    return _polygon_unified_instance

# 便捷函数
def enable_polygon_factors():
    """启usePolygon因子"""
    manager = get_polygon_unified_factors()
    # 移除重复初始化调用，因为get_polygon_unified_factors()已经会初始化
    if not manager.is_enabled():
        logger.info("Polygon factors already initialized or failed to initialize")

def enable_polygon_risk_balancer():
    """启usePolygonrisk control收益平衡器"""
    manager = get_polygon_unified_factors()
    manager.enable_risk_balancer()

def disable_polygon_risk_balancer():
    """禁usePolygonrisk control收益平衡器"""
    manager = get_polygon_unified_factors()
    manager.disable_risk_balancer()

def check_polygon_trading_conditions(symbol: str) -> Dict[str, Any]:
    """checkPolygon交易 records件"""
    manager = get_polygon_unified_factors()
    return manager.check_trading_conditions(symbol)

def process_signals_with_polygon(signals: Union[pd.DataFrame, List[Dict]]) -> List[Dict]:
    """使usePolygon处理信号"""
    manager = get_polygon_unified_factors()
    return manager.process_signals_with_risk_control(signals)