#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一风险管理器 - 整合所has风险控制逻辑
解决风险验证分散in多个模块问题
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import math

class RiskLevel(Enum):
    """风险级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskEventType(Enum):
    """风险事件类型"""
    POSITION_LIMIT = "position_limit"
    SECTOR_LIMIT = "sector_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    ORDER_SIZE_LIMIT = "order_size_limit"
    CORRELATION_LIMIT = "correlation_limit"
    VOLATILITY_WARNING = "volatility_warning"
    CONCENTRATION_WARNING = "concentration_warning"

@dataclass
class RiskValidationResult:
    """风险验证结果"""
    is_valid: bool
    risk_level: RiskLevel
    violations: List[str]
    warnings: List[str]
    recommended_size: Optional[int] = None
    max_allowed_size: Optional[int] = None

@dataclass
class PositionRisk:
    """positions风险信息"""
    symbol: str
    quantity: int
    current_price: float
    entry_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    portfolio_weight: float
    sector: Optional[str] = None
    volatility: Optional[float] = None
    correlation_score: Optional[float] = None

@dataclass
class PortfolioRisk:
    """投资组合风险信息"""
    total_value: float
    total_unrealized_pnl: float
    max_drawdown: float
    current_drawdown: float
    sector_exposures: Dict[str, float]
    concentration_risk: float
    var_1d: Optional[float] = None
    sharpe_ratio: Optional[float] = None

class UnifiedRiskManager:
    """统一风险管理器"""
    
    def __init__(self, config_manager, logger: Optional[logging.Logger] = None):
        self.config_manager = config_manager
        self.logger = logger or logging.getLogger("UnifiedRiskManager")
        
        # 风险配置
        self.risk_config = {
            'max_single_position_pct': 0.15,  # 单仓最大仓位
            'max_sector_exposure_pct': 0.30,  # 行业最大敞口
            'max_correlation': 0.70,          # 最大相关性
            'daily_loss_limit_pct': 0.05,    # 日亏损限制
            'max_daily_orders': 20,           # 日订单限制
            'min_order_value': 1000.0,       # 最小订单价值
            'max_order_value': 50000.0,      # 最大订单价值
            'concentration_warning_pct': 0.25, # 集in度警告阈值
            'volatility_warning_pct': 0.05,   # 波动率警告阈值
        }
        
        # 运行when状态
        self.daily_order_count = 0
        self.daily_pnl = 0.0
        self.last_reset_date = time.strftime('%Y-%m-%d')
        
        # positionsand组合数据
        self.positions: Dict[str, PositionRisk] = {}
        self.portfolio_risk: Optional[PortfolioRisk] = None
        
        # 风险事件历史
        self.risk_events: List[Dict] = []
        self.max_event_history = 1000
        
        # 加载配置
        self._load_risk_config()
    
    def _load_risk_config(self):
        """from配置管理器加载风险配置"""
        try:
            # from统一配置管理器retrieval风险settings
            risk_settings = self.config_manager.get('risk_management', {})
            if risk_settings:
                self.risk_config.update(risk_settings)
                self.logger.info(f"Risk configuration loaded: {len(risk_settings)} items")
            
            # from订单settingsretrieval配置
            sizing_config = self.config_manager.get('sizing', {})
            if sizing_config:
                if 'max_position_pct_of_equity' in sizing_config:
                    self.risk_config['max_single_position_pct'] = sizing_config['max_position_pct_of_equity']
                if 'min_position_usd' in sizing_config:
                    self.risk_config['min_order_value'] = sizing_config['min_position_usd']
            
            # from风险控制settingsretrieval配置
            risk_controls = self.config_manager.get('risk_controls', {})
            if risk_controls:
                if 'daily_order_limit' in risk_controls:
                    self.risk_config['max_daily_orders'] = risk_controls['daily_order_limit']
                if 'sector_exposure_limit' in risk_controls:
                    self.risk_config['max_sector_exposure_pct'] = risk_controls['sector_exposure_limit']
                if 'max_correlation' in risk_controls:
                    self.risk_config['max_correlation'] = risk_controls['max_correlation']
            
        except Exception as e:
            self.logger.warning(f"加载风险配置failed，使use默认值: {e}")
    
    async def validate_order(self, symbol: str, side: str, quantity: int, 
                           price: float, account_value: float) -> RiskValidationResult:
        """统一订单风险验证入口"""
        violations = []
        warnings = []
        risk_level = RiskLevel.LOW
        
        # 重置日计数器
        self._reset_daily_counters_if_needed()
        
        try:
            # 1. 基础订单check
            order_value = quantity * price
            
            # 订单价值check
            if order_value < self.risk_config['min_order_value']:
                violations.append(f"订单价值过小: ${order_value:.2f} < ${self.risk_config['min_order_value']:.2f}")
            
            if order_value > self.risk_config['max_order_value']:
                violations.append(f"订单价值过大: ${order_value:.2f} > ${self.risk_config['max_order_value']:.2f}")
            
            # 2. positions限制check
            if account_value > 0:
                position_pct = order_value / account_value
                max_position_pct = self.risk_config['max_single_position_pct']
                
                if position_pct > max_position_pct:
                    violations.append(f"仓位超限: {position_pct:.1%} > {max_position_pct:.1%}")
                    risk_level = RiskLevel.HIGH
                elif position_pct > max_position_pct * 0.8:
                    warnings.append(f"仓位接近上限: {position_pct:.1%}")
                    risk_level = max(risk_level, RiskLevel.MEDIUM)
            
            # 3. 日订单限制check
            if self.daily_order_count >= self.risk_config['max_daily_orders']:
                violations.append(f"日订单数超限: {self.daily_order_count} >= {self.risk_config['max_daily_orders']}")
                risk_level = RiskLevel.CRITICAL
            elif self.daily_order_count >= self.risk_config['max_daily_orders'] * 0.8:
                warnings.append(f"日订单数接近上限: {self.daily_order_count}")
                risk_level = max(risk_level, RiskLevel.MEDIUM)
            
            # 4. 投资组合风险check
            portfolio_risk_result = await self._check_portfolio_risk(symbol, side, quantity, price, account_value)
            violations.extend(portfolio_risk_result.get('violations', []))
            warnings.extend(portfolio_risk_result.get('warnings', []))
            if portfolio_risk_result.get('risk_level', RiskLevel.LOW).value == RiskLevel.HIGH.value:
                risk_level = RiskLevel.HIGH
            
            # 5. 计算推荐订单大小
            recommended_size = self._calculate_recommended_size(symbol, price, account_value, violations)
            max_allowed_size = self._calculate_max_allowed_size(symbol, price, account_value)
            
            # 记录风险事件
            if violations or (risk_level.value in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]):
                self._log_risk_event(RiskEventType.ORDER_SIZE_LIMIT, {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'violations': violations,
                    'risk_level': risk_level.value
                })
            
            return RiskValidationResult(
                is_valid=len(violations) == 0,
                risk_level=risk_level,
                violations=violations,
                warnings=warnings,
                recommended_size=recommended_size,
                max_allowed_size=max_allowed_size
            )
            
        except Exception as e:
            self.logger.error(f"订单风险验证failed: {e}")
            return RiskValidationResult(
                is_valid=False,
                risk_level=RiskLevel.CRITICAL,
                violations=[f"风险验证系统错误: {str(e)}"],
                warnings=[]
            )
    
    async def _check_portfolio_risk(self, symbol: str, side: str, quantity: int, 
                                  price: float, account_value: float) -> Dict:
        """check投资组合级别风险"""
        violations = []
        warnings = []
        risk_level = RiskLevel.LOW
        
        try:
            # updates组合风险计算
            await self._update_portfolio_risk(account_value)
            
            if self.portfolio_risk:
                # check行业集in度
                # 这里需要from外部retrieval行业信息，暂when跳过
                # sector_risk = self._check_sector_concentration(symbol, quantity * price)
                
                # check整体集in度
                concentration_risk = self.portfolio_risk.concentration_risk
                concentration_limit = self.risk_config['concentration_warning_pct']
                
                if concentration_risk > concentration_limit:
                    warnings.append(f"投资组合集in度过高: {concentration_risk:.1%}")
                    risk_level = max(risk_level, RiskLevel.MEDIUM)
                
                # check当before回撤
                if self.portfolio_risk.current_drawdown > 0.10:  # 10%回撤警告
                    warnings.append(f"投资组合回撤较大: {self.portfolio_risk.current_drawdown:.1%}")
                    risk_level = max(risk_level, RiskLevel.HIGH)
            
        except Exception as e:
            self.logger.warning(f"投资组合风险checkfailed: {e}")
        
        return {
            'violations': violations,
            'warnings': warnings,
            'risk_level': risk_level
        }
    
    async def _update_portfolio_risk(self, account_value: float):
        """updates投资组合风险指标"""
        try:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            
            # 计算集in度风险（最大仓位ratio）
            if self.positions and account_value > 0:
                max_position_weight = max(
                    abs(pos.quantity * pos.current_price) / account_value 
                    for pos in self.positions.values()
                )
                concentration_risk = max_position_weight
            else:
                concentration_risk = 0.0
            
            # 简化投资组合风险计算
            self.portfolio_risk = PortfolioRisk(
                total_value=account_value,
                total_unrealized_pnl=total_unrealized_pnl,
                max_drawdown=0.0,  # 需要历史数据计算
                current_drawdown=max(0, -total_unrealized_pnl / account_value) if account_value > 0 else 0,
                sector_exposures={},  # 需要行业数据
                concentration_risk=concentration_risk
            )
            
        except Exception as e:
            self.logger.warning(f"updates投资组合风险failed: {e}")
    
    def _calculate_recommended_size(self, symbol: str, price: float, 
                                  account_value: float, violations: List[str]) -> Optional[int]:
        """计算推荐订单大小"""
        try:
            if violations:  # if果has违规，not推荐
                return None
            
            # 基at风险预算计算推荐大小
            max_position_value = account_value * self.risk_config['max_single_position_pct'] * 0.8  # 保守80%
            max_size = int(max_position_value / price)
            
            # 考虑最小订单价值
            min_size = int(self.risk_config['min_order_value'] / price) + 1
            
            return max(min_size, min(max_size, 1000))  # 限制in合理范围内
            
        except Exception as e:
            self.logger.warning(f"计算推荐订单大小failed: {e}")
            return None
    
    def _calculate_max_allowed_size(self, symbol: str, price: float, account_value: float) -> Optional[int]:
        """计算最大允许订单大小"""
        try:
            max_position_value = account_value * self.risk_config['max_single_position_pct']
            max_order_value = min(max_position_value, self.risk_config['max_order_value'])
            
            return int(max_order_value / price)
            
        except Exception as e:
            self.logger.warning(f"计算最大允许订单大小failed: {e}")
            return None
    
    def update_position(self, symbol: str, quantity: int, current_price: float, 
                       entry_price: Optional[float] = None):
        """updatespositions信息"""
        try:
            if quantity == 0:
                # 平仓
                if symbol in self.positions:
                    del self.positions[symbol]
                return
            
            if entry_price is None:
                entry_price = current_price
            
            unrealized_pnl = quantity * (current_price - entry_price)
            unrealized_pnl_pct = (current_price / entry_price - 1) if entry_price > 0 else 0
            
            self.positions[symbol] = PositionRisk(
                symbol=symbol,
                quantity=quantity,
                current_price=current_price,
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                portfolio_weight=0.0  # willinupdate_portfolio_riskin计算
            )
            
        except Exception as e:
            self.logger.error(f"updatespositions信息failed {symbol}: {e}")
    
    def increment_order_count(self):
        """增加日订单计数"""
        self._reset_daily_counters_if_needed()
        self.daily_order_count += 1
    
    def update_daily_pnl(self, pnl_change: float):
        """updates日损益"""
        self._reset_daily_counters_if_needed()
        self.daily_pnl += pnl_change
        
        # check日亏损限制
        daily_loss_limit = self.risk_config['daily_loss_limit_pct']
        if self.daily_pnl < -daily_loss_limit:
            self._log_risk_event(RiskEventType.DAILY_LOSS_LIMIT, {
                'daily_pnl': self.daily_pnl,
                'limit': -daily_loss_limit,
                'timestamp': time.time()
            })
    
    def _reset_daily_counters_if_needed(self):
        """根据需要重置日计数器"""
        current_date = time.strftime('%Y-%m-%d')
        if current_date != self.last_reset_date:
            self.daily_order_count = 0
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            self.logger.info("日风险计数器重置")
    
    def _log_risk_event(self, event_type: RiskEventType, details: Dict):
        """记录风险事件"""
        event = {
            'type': event_type.value,
            'timestamp': time.time(),
            'details': details
        }
        
        self.risk_events.append(event)
        
        # 限制历史长度
        if len(self.risk_events) > self.max_event_history:
            self.risk_events = self.risk_events[-self.max_event_history:]
        
        # 记录日志
        if event_type in [RiskEventType.DAILY_LOSS_LIMIT, RiskEventType.ORDER_SIZE_LIMIT]:
            self.logger.warning(f"风险事件: {event_type.value} - {details}")
        else:
            self.logger.info(f"风险事件: {event_type.value} - {details}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """retrieval风险状况summary"""
        self._reset_daily_counters_if_needed()
        
        # 计算风险指标
        recent_events = [e for e in self.risk_events if time.time() - e['timestamp'] < 3600]  # 最近1小when
        
        return {
            'daily_orders': {
                'count': self.daily_order_count,
                'limit': self.risk_config['max_daily_orders'],
                'utilization': self.daily_order_count / self.risk_config['max_daily_orders']
            },
            'daily_pnl': {
                'amount': self.daily_pnl,
                'limit': -self.risk_config['daily_loss_limit_pct'],
                'status': 'OK' if self.daily_pnl > -self.risk_config['daily_loss_limit_pct'] else 'WARNING'
            },
            'positions': {
                'count': self.position_manager.get_portfolio_summary().total_positions,
                'symbols': list(self.position_manager.get_symbols())
            },
            'portfolio_risk': self.portfolio_risk.__dict__ if self.portfolio_risk else None,
            'recent_events': len(recent_events),
            'risk_config': self.risk_config.copy()
        }
    
    def get_position_risk(self, symbol: str) -> Optional[PositionRisk]:
        """retrieval特定positions风险信息"""
        return self.position_manager.get_quantity(symbol)
    
    def check_emergency_stop_conditions(self) -> Tuple[bool, List[str]]:
        """checkis否触发紧急停止 records件"""
        violations = []
        
        # check日亏损限制
        if self.daily_pnl < -self.risk_config['daily_loss_limit_pct'] * 2:  # 2倍日亏损限制
            violations.append(f"日亏损超过紧急阈值: {self.daily_pnl:.1%}")
        
        # check投资组合回撤
        if self.portfolio_risk and self.portfolio_risk.current_drawdown > 0.20:  # 20%回撤
            violations.append(f"投资组合回撤过大: {self.portfolio_risk.current_drawdown:.1%}")
        
        return len(violations) > 0, violations

# 全局实例
_global_risk_manager: Optional[UnifiedRiskManager] = None

def get_risk_manager(config_manager=None) -> UnifiedRiskManager:
    """retrieval全局风险管理器实例"""
    global _global_risk_manager
    if _global_risk_manager is None:
        if config_manager is None:
            from .unified_config import get_unified_config
            config_manager = get_unified_config()
        _global_risk_manager = UnifiedRiskManager(config_manager)
    return _global_risk_manager
