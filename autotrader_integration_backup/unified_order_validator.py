#!/usr/bin/env python3
"""
统一订单验证器
整合所有订单验证逻辑，消除重复验证
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    reason: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

class UnifiedOrderValidator:
    """统一订单验证器 - 消除重复验证逻辑"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logger
        
        # 验证配置
        self.validation_config = {
            "min_order_value": 1000.0,
            "max_order_value": 100000.0,
            "min_price": 1.0,
            "max_price": 1000.0,
            "max_position_pct": 0.15,
            "per_trade_risk_pct": 0.02,
            "max_daily_orders": 50
        }
        
        # 验证缓存 - 避免重复计算
        self._validation_cache = {}
        self._cache_ttl = 30  # 30秒缓存
        
    async def validate_order_unified(self, 
                                   symbol: str, 
                                   side: str, 
                                   quantity: int, 
                                   price: float, 
                                   account_value: float) -> ValidationResult:
        """
        统一订单验证 - 所有验证逻辑的单一入口点
        避免重复验证，提高性能
        """
        
        # 生成缓存键
        cache_key = f"{symbol}:{side}:{quantity}:{price:.2f}"
        
        # 检查缓存
        if cache_key in self._validation_cache:
            cached_time, cached_result = self._validation_cache[cache_key]
            if (asyncio.get_event_loop().time() - cached_time) < self._cache_ttl:
                self.logger.debug(f"使用缓存的验证结果: {symbol}")
                return cached_result
        
        # 执行验证
        result = await self._perform_validation(symbol, side, quantity, price, account_value)
        
        # 缓存结果
        self._validation_cache[cache_key] = (asyncio.get_event_loop().time(), result)
        
        return result
    
    async def _perform_validation(self, symbol, side, quantity, price, account_value) -> ValidationResult:
        """执行实际的验证逻辑"""
        
        try:
            # 1. 基本参数验证
            if quantity <= 0:
                return ValidationResult(False, "数量必须大于0")
            
            if price <= 0:
                return ValidationResult(False, "价格必须大于0")
            
            # 2. 订单金额验证
            order_value = quantity * price
            if order_value < self.validation_config["min_order_value"]:
                return ValidationResult(False, f"订单金额过小: ${order_value:.2f}")
            
            if order_value > self.validation_config["max_order_value"]:
                return ValidationResult(False, f"订单金额过大: ${order_value:.2f}")
            
            # 3. 价格范围验证
            if price < self.validation_config["min_price"] or price > self.validation_config["max_price"]:
                return ValidationResult(False, f"价格超出范围: ${price:.2f}")
            
            # 4. 账户余额验证
            if account_value <= 0:
                return ValidationResult(False, "账户余额不足")
            
            # 5. 风险比例验证
            risk_ratio = order_value / account_value
            if risk_ratio > self.validation_config["max_position_pct"]:
                return ValidationResult(False, f"仓位风险过大: {risk_ratio:.2%}")
            
            # 所有验证通过
            return ValidationResult(True, "验证通过", {
                "order_value": order_value,
                "risk_ratio": risk_ratio,
                "validated_at": asyncio.get_event_loop().time()
            })
            
        except Exception as e:
            self.logger.error(f"验证过程中发生错误: {e}")
            return ValidationResult(False, f"验证错误: {str(e)}")
    
    def clear_cache(self):
        """清理验证缓存"""
        self._validation_cache.clear()

# 全局单例
_unified_validator = None

def get_unified_validator(config_manager=None):
    """获取统一验证器实例"""
    global _unified_validator
    if _unified_validator is None:
        _unified_validator = UnifiedOrderValidator(config_manager)
    return _unified_validator
