#!/usr/bin/env python3
"""
价格验证模块 - 确保交易价格数据的有效性和安全性
"""

import logging
import time
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class PriceValidationResult(Enum):
    """价格验证结果"""
    VALID = "valid"
    INVALID_RANGE = "invalid_range" 
    INVALID_CHANGE = "invalid_change"
    STALE_DATA = "stale_data"
    NO_DATA = "no_data"
    SUSPICIOUS = "suspicious"

@dataclass
class PriceValidationConfig:
    """价格验证配置"""
    # 基础验证
    min_price: float = 0.01  # 最低价格
    max_price: float = 100000.0  # 最高价格
    max_daily_change_pct: float = 0.50  # 日内最大涨跌幅50%
    max_tick_change_pct: float = 0.10  # 单次跳动最大变化10%
    
    # 时效性验证
    max_data_age_seconds: float = 300.0  # 数据最大延迟5分钟
    stale_warning_seconds: float = 60.0  # 延迟告警阈值1分钟
    
    # 异常检测
    outlier_std_multiplier: float = 3.0  # 异常值标准差倍数
    min_history_for_validation: int = 5  # 最小历史数据点数
    
    # 回退策略
    allow_avgcost_fallback: bool = True  # 允许使用平均成本
    allow_last_known_fallback: bool = True  # 允许使用最后已知价格
    fallback_max_age_hours: float = 24.0  # 回退数据最大年龄

@dataclass 
class PriceData:
    """价格数据结构"""
    symbol: str
    price: float
    timestamp: float
    source: str  # 'realtime', 'delayed', 'avgcost', 'last_known'
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None

class EnhancedPriceValidator:
    """增强价格验证器"""
    
    def __init__(self, config: Optional[PriceValidationConfig] = None):
        self.config = config or PriceValidationConfig()
        self._price_history: Dict[str, List[PriceData]] = {}
        self._validation_stats: Dict[str, int] = {
            'total_validations': 0,
            'valid_prices': 0,
            'invalid_prices': 0,
            'fallback_used': 0
        }
        
    def validate_price(self, symbol: str, price_data: PriceData) -> Tuple[PriceValidationResult, str]:
        """
        验证价格数据的有效性
        
        Returns:
            (验证结果, 详细信息)
        """
        self._validation_stats['total_validations'] += 1
        
        # 1. 基础范围验证
        if price_data.price <= 0:
            return PriceValidationResult.NO_DATA, "价格为零或负数"
            
        if not (self.config.min_price <= price_data.price <= self.config.max_price):
            return PriceValidationResult.INVALID_RANGE, f"价格超出合理范围: {price_data.price}"
            
        # 2. 时效性验证
        current_time = time.time()
        data_age = current_time - price_data.timestamp
        
        if data_age > self.config.max_data_age_seconds:
            return PriceValidationResult.STALE_DATA, f"数据过时: {data_age:.1f}秒"
            
        # 3. 变化幅度验证
        if symbol in self._price_history and self._price_history[symbol]:
            last_price_data = self._price_history[symbol][-1]
            change_pct = abs(price_data.price - last_price_data.price) / last_price_data.price
            
            if change_pct > self.config.max_tick_change_pct:
                return PriceValidationResult.INVALID_CHANGE, f"价格变化过大: {change_pct*100:.2f}%"
                
        # 4. 异常值检测
        if self._is_price_outlier(symbol, price_data.price):
            return PriceValidationResult.SUSPICIOUS, "价格可能是异常值"
            
        # 5. 验证通过
        self._validation_stats['valid_prices'] += 1
        self._update_price_history(symbol, price_data)
        
        # 添加延迟告警
        if data_age > self.config.stale_warning_seconds:
            logger.warning(f"{symbol} 价格数据延迟 {data_age:.1f}秒")
            
        return PriceValidationResult.VALID, "验证通过"
    
    def get_validated_price(self, symbol: str, primary_source_price: Optional[float], 
                           fallback_sources: Dict[str, Optional[float]]) -> Optional[PriceData]:
        """
        获取验证后的价格，支持多级回退
        
        Args:
            symbol: 股票代码
            primary_source_price: 主要数据源价格
            fallback_sources: 回退数据源 {'avgcost': price, 'last_known': price}
        
        Returns:
            验证通过的价格数据或None
        """
        current_time = time.time()
        
        # 1. 尝试主要数据源
        if primary_source_price and primary_source_price > 0:
            primary_data = PriceData(
                symbol=symbol,
                price=primary_source_price,
                timestamp=current_time,
                source='realtime'
            )
            
            result, msg = self.validate_price(symbol, primary_data)
            if result == PriceValidationResult.VALID:
                logger.debug(f"{symbol} 使用实时价格: {primary_source_price}")
                return primary_data
            else:
                logger.warning(f"{symbol} 实时价格验证失败: {msg}")
        
        # 2. 尝试平均成本回退
        if self.config.allow_avgcost_fallback and fallback_sources.get('avgcost'):
            avgcost = fallback_sources['avgcost']
            if avgcost > 0:
                avgcost_data = PriceData(
                    symbol=symbol,
                    price=avgcost,
                    timestamp=current_time,
                    source='avgcost'
                )
                
                # 放宽验证标准，但仍需基础检查
                if self.config.min_price <= avgcost <= self.config.max_price:
                    self._validation_stats['fallback_used'] += 1
                    logger.warning(f"{symbol} 使用平均成本回退价格: {avgcost}")
                    self._update_price_history(symbol, avgcost_data)
                    return avgcost_data
        
        # 3. 尝试最后已知价格回退
        if self.config.allow_last_known_fallback and symbol in self._price_history:
            history = self._price_history[symbol]
            if history:
                last_data = history[-1]
                age_hours = (current_time - last_data.timestamp) / 3600
                
                if age_hours <= self.config.fallback_max_age_hours:
                    self._validation_stats['fallback_used'] += 1
                    logger.warning(f"{symbol} 使用最后已知价格: {last_data.price} (年龄: {age_hours:.1f}小时)")
                    return last_data
        
        # 4. 所有方法都失败
        self._validation_stats['invalid_prices'] += 1
        logger.error(f"{symbol} 无法获取有效价格数据")
        return None
    
    def _is_price_outlier(self, symbol: str, price: float) -> bool:
        """检测价格是否为异常值"""
        if symbol not in self._price_history:
            return False
            
        history = self._price_history[symbol]
        if len(history) < self.config.min_history_for_validation:
            return False
            
        # 使用最近的历史价格计算统计量
        recent_prices = [p.price for p in history[-20:]]  # 最近20个价格点
        
        try:
            mean_price = statistics.mean(recent_prices)
            std_dev = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
            
            if std_dev == 0:  # 价格无变化
                return abs(price - mean_price) > mean_price * 0.01  # 1%阈值
                
            z_score = abs(price - mean_price) / std_dev
            return z_score > self.config.outlier_std_multiplier
            
        except statistics.StatisticsError:
            return False
    
    def _update_price_history(self, symbol: str, price_data: PriceData):
        """更新价格历史"""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
            
        self._price_history[symbol].append(price_data)
        
        # 保持历史长度合理
        max_history = 100
        if len(self._price_history[symbol]) > max_history:
            self._price_history[symbol] = self._price_history[symbol][-max_history:]
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        stats = self._validation_stats.copy()
        if stats['total_validations'] > 0:
            stats['valid_rate'] = stats['valid_prices'] / stats['total_validations']
            stats['fallback_rate'] = stats['fallback_used'] / stats['total_validations']
        else:
            stats['valid_rate'] = 0.0
            stats['fallback_rate'] = 0.0
            
        return stats
    
    def cleanup_old_data(self, max_age_hours: float = 24.0):
        """清理过期的历史数据"""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        cleaned_count = 0
        for symbol in self._price_history:
            original_len = len(self._price_history[symbol])
            self._price_history[symbol] = [
                p for p in self._price_history[symbol]
                if p.timestamp > cutoff_time
            ]
            cleaned_count += original_len - len(self._price_history[symbol])
            
        if cleaned_count > 0:
            logger.info(f"清理了 {cleaned_count} 条过期价格数据")

# 全局价格验证器实例
_global_price_validator: Optional[EnhancedPriceValidator] = None

def get_price_validator(config: Optional[PriceValidationConfig] = None) -> EnhancedPriceValidator:
    """获取全局价格验证器实例"""
    global _global_price_validator
    if _global_price_validator is None:
        _global_price_validator = EnhancedPriceValidator(config)
    return _global_price_validator

def create_price_validator(config: Optional[PriceValidationConfig] = None) -> EnhancedPriceValidator:
    """创建新的价格验证器实例"""
    return EnhancedPriceValidator(config)