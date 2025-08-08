#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的风险管理系统
实现Pre-Trade检查、损失冷却期、单日交易限制和组合保护
"""

import time
import logging
import threading
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import numpy as np
from collections import defaultdict, deque


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCheckResult(Enum):
    """风险检查结果"""
    APPROVED = "approved"
    REJECTED = "rejected"
    SCALED_DOWN = "scaled_down"
    WARNING = "warning"


@dataclass
class RiskCheckResponse:
    """风险检查响应"""
    result: RiskCheckResult
    original_quantity: int
    approved_quantity: int
    risk_level: RiskLevel
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionRisk:
    """持仓风险信息"""
    symbol: str
    quantity: int
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    avg_cost: float
    current_price: float
    portfolio_weight: float
    var_1d: float = 0.0  # 1日VaR
    max_drawdown: float = 0.0
    days_held: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'avg_cost': self.avg_cost,
            'current_price': self.current_price,
            'portfolio_weight': self.portfolio_weight,
            'var_1d': self.var_1d,
            'max_drawdown': self.max_drawdown,
            'days_held': self.days_held
        }


class EnhancedRiskManager:
    """增强的风险管理器"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # 基础风控参数
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)  # 组合最大风险2%
        self.max_position_size = config.get('max_position_size', 0.05)  # 单个持仓最大5%
        self.max_sector_exposure = config.get('max_sector_exposure', 0.25)  # 单个行业最大25%
        self.max_daily_loss = config.get('max_daily_loss', 0.05)  # 单日最大损失5%
        self.max_drawdown = config.get('max_drawdown', 0.10)  # 最大回撤10%
        
        # 交易限制
        self.max_new_positions_per_day = config.get('max_new_positions_per_day', 10)
        self.max_trades_per_symbol_per_day = config.get('max_trades_per_symbol_per_day', 3)
        self.loss_cooldown_days = config.get('loss_cooldown_days', 3)
        self.min_time_between_trades = config.get('min_time_between_trades_minutes', 15)
        
        # 市场风险参数
        self.volatility_threshold = config.get('volatility_threshold', 0.03)  # 3%波动率阈值
        self.correlation_threshold = config.get('correlation_threshold', 0.7)  # 相关性阈值
        self.liquidity_threshold = config.get('liquidity_threshold', 100000)  # 流动性阈值
        
        # 状态跟踪
        self.current_positions = {}  # symbol -> PositionRisk
        self.daily_trades = defaultdict(list)  # date -> list of trades
        self.symbol_trades = defaultdict(list)  # symbol -> list of trades
        self.loss_cooldown = {}  # symbol -> cooldown_end_date
        self.emergency_stop_triggered = False
        self.trading_suspended = False
        
        # 组合统计
        self.portfolio_value = 0.0
        self.portfolio_pnl = 0.0
        self.daily_pnl = 0.0
        self.max_portfolio_drawdown = 0.0
        self.high_water_mark = 0.0
        
        # 风险度量历史
        self.risk_history = deque(maxlen=252)  # 1年历史
        self.pnl_history = deque(maxlen=252)
        
        # 行业分类（简化版本）
        self.sector_mapping = self._load_sector_mapping()
        
        # 监控线程
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # 数据文件
        self.risk_data_file = config.get('risk_data_file', 'risk/risk_data.json')
        
        # 加载历史数据
        self._load_risk_data()
        
        # 启动监控
        self._start_monitoring()
    
    def pre_trade_check(self, symbol: str, action: str, quantity: int, 
                       price: float, strategy_name: str = None) -> RiskCheckResponse:
        """交易前风险检查"""
        reasons = []
        warnings = []
        risk_level = RiskLevel.LOW
        approved_quantity = quantity
        
        try:
            # 1. 检查交易暂停状态
            if self.trading_suspended:
                return RiskCheckResponse(
                    result=RiskCheckResult.REJECTED,
                    original_quantity=quantity,
                    approved_quantity=0,
                    risk_level=RiskLevel.CRITICAL,
                    reasons=["Trading is suspended due to risk controls"]
                )
            
            # 2. 检查紧急停止
            if self.emergency_stop_triggered:
                return RiskCheckResponse(
                    result=RiskCheckResult.REJECTED,
                    original_quantity=quantity,
                    approved_quantity=0,
                    risk_level=RiskLevel.CRITICAL,
                    reasons=["Emergency stop is active"]
                )
            
            # 3. 检查损失冷却期
            if symbol in self.loss_cooldown:
                cooldown_end = self.loss_cooldown[symbol]
                if date.today() <= cooldown_end:
                    days_remaining = (cooldown_end - date.today()).days
                    return RiskCheckResponse(
                        result=RiskCheckResult.REJECTED,
                        original_quantity=quantity,
                        approved_quantity=0,
                        risk_level=RiskLevel.HIGH,
                        reasons=[f"Symbol {symbol} is in loss cooldown for {days_remaining} more days"]
                    )
            
            # 4. 检查单日交易限制
            today = date.today()
            daily_trades_count = len(self.daily_trades[today])
            if daily_trades_count >= self.max_new_positions_per_day:
                return RiskCheckResponse(
                    result=RiskCheckResult.REJECTED,
                    original_quantity=quantity,
                    approved_quantity=0,
                    risk_level=RiskLevel.MEDIUM,
                    reasons=[f"Daily trade limit reached: {daily_trades_count}/{self.max_new_positions_per_day}"]
                )
            
            # 5. 检查单个股票的交易频率
            symbol_daily_trades = [t for t in self.symbol_trades[symbol] 
                                  if t['date'] == today]
            if len(symbol_daily_trades) >= self.max_trades_per_symbol_per_day:
                return RiskCheckResponse(
                    result=RiskCheckResult.REJECTED,
                    original_quantity=quantity,
                    approved_quantity=0,
                    risk_level=RiskLevel.MEDIUM,
                    reasons=[f"Daily trade limit for {symbol} reached: {len(symbol_daily_trades)}/{self.max_trades_per_symbol_per_day}"]
                )
            
            # 6. 检查交易时间间隔
            if symbol_daily_trades:
                last_trade_time = max(t['timestamp'] for t in symbol_daily_trades)
                time_diff = (datetime.now() - last_trade_time).total_seconds() / 60
                if time_diff < self.min_time_between_trades:
                    return RiskCheckResponse(
                        result=RiskCheckResult.REJECTED,
                        original_quantity=quantity,
                        approved_quantity=0,
                        risk_level=RiskLevel.MEDIUM,
                        reasons=[f"Minimum time between trades not met: {time_diff:.1f} < {self.min_time_between_trades} minutes"]
                    )
            
            # 7. 计算新仓位风险
            position_value = quantity * price
            
            # 检查单个仓位大小
            if self.portfolio_value > 0:
                position_weight = position_value / self.portfolio_value
                if position_weight > self.max_position_size:
                    # 缩减仓位
                    max_value = self.portfolio_value * self.max_position_size
                    approved_quantity = int(max_value / price)
                    warnings.append(f"Position size scaled down from {quantity} to {approved_quantity} (weight limit: {self.max_position_size:.1%})")
                    risk_level = RiskLevel.MEDIUM
            
            # 8. 检查行业集中度
            sector = self._get_sector(symbol)
            if sector:
                sector_exposure = self._calculate_sector_exposure(sector)
                sector_weight = sector_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
                
                if sector_weight > self.max_sector_exposure:
                    # 进一步缩减仓位
                    max_sector_value = self.portfolio_value * self.max_sector_exposure
                    remaining_sector_capacity = max_sector_value - sector_exposure
                    if remaining_sector_capacity > 0:
                        max_quantity_by_sector = int(remaining_sector_capacity / price)
                        if max_quantity_by_sector < approved_quantity:
                            approved_quantity = max_quantity_by_sector
                            warnings.append(f"Position size limited by sector exposure: {approved_quantity} (sector {sector}: {sector_weight:.1%})")
                            risk_level = RiskLevel.MEDIUM
                    else:
                        return RiskCheckResponse(
                            result=RiskCheckResult.REJECTED,
                            original_quantity=quantity,
                            approved_quantity=0,
                            risk_level=RiskLevel.HIGH,
                            reasons=[f"Sector {sector} exposure limit exceeded: {sector_weight:.1%} > {self.max_sector_exposure:.1%}"]
                        )
            
            # 9. 检查组合风险
            portfolio_risk_check = self._check_portfolio_risk(symbol, approved_quantity, price)
            if not portfolio_risk_check[0]:
                return RiskCheckResponse(
                    result=RiskCheckResult.REJECTED,
                    original_quantity=quantity,
                    approved_quantity=0,
                    risk_level=RiskLevel.HIGH,
                    reasons=[portfolio_risk_check[1]]
                )
            
            # 10. 检查流动性
            liquidity_risk = self._check_liquidity_risk(symbol, approved_quantity)
            if liquidity_risk:
                warnings.append(liquidity_risk)
                risk_level = max(risk_level, RiskLevel.MEDIUM)
            
            # 11. 检查波动性
            volatility_risk = self._check_volatility_risk(symbol)
            if volatility_risk:
                warnings.append(volatility_risk)
                risk_level = max(risk_level, RiskLevel.MEDIUM)
            
            # 12. 检查相关性
            correlation_risk = self._check_correlation_risk(symbol)
            if correlation_risk:
                warnings.append(correlation_risk)
                risk_level = max(risk_level, RiskLevel.MEDIUM)
            
            # 确定最终结果
            if approved_quantity == 0:
                result = RiskCheckResult.REJECTED
            elif approved_quantity < quantity:
                result = RiskCheckResult.SCALED_DOWN
            elif warnings:
                result = RiskCheckResult.WARNING
            else:
                result = RiskCheckResult.APPROVED
            
            # 构建风险指标
            risk_metrics = {
                'position_weight': position_value / self.portfolio_value if self.portfolio_value > 0 else 0,
                'sector_exposure': sector_weight if 'sector_weight' in locals() else 0,
                'portfolio_risk': self._calculate_portfolio_var(),
                'daily_pnl_impact': self.daily_pnl / self.portfolio_value if self.portfolio_value > 0 else 0,
                'drawdown': self.max_portfolio_drawdown
            }
            
            return RiskCheckResponse(
                result=result,
                original_quantity=quantity,
                approved_quantity=approved_quantity,
                risk_level=risk_level,
                reasons=reasons,
                warnings=warnings,
                risk_metrics=risk_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error in pre-trade check: {e}")
            return RiskCheckResponse(
                result=RiskCheckResult.REJECTED,
                original_quantity=quantity,
                approved_quantity=0,
                risk_level=RiskLevel.CRITICAL,
                reasons=[f"Risk check error: {str(e)}"]
            )
    
    def record_trade(self, symbol: str, action: str, quantity: int, price: float, 
                    strategy_name: str = None, timestamp: datetime = None):
        """记录交易"""
        if timestamp is None:
            timestamp = datetime.now()
        
        trade_record = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'strategy_name': strategy_name,
            'timestamp': timestamp,
            'date': timestamp.date()
        }
        
        # 记录到日常交易
        today = timestamp.date()
        self.daily_trades[today].append(trade_record)
        
        # 记录到股票交易
        self.symbol_trades[symbol].append(trade_record)
        
        # 更新持仓
        self._update_position(symbol, action, quantity, price)
        
        self.logger.info(f"Trade recorded: {symbol} {action} {quantity} @ {price}")
    
    def update_position_prices(self, price_data: Dict[str, float]):
        """更新持仓价格"""
        for symbol, current_price in price_data.items():
            if symbol in self.current_positions:
                position = self.current_positions[symbol]
                old_price = position.current_price
                position.current_price = current_price
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_cost) * position.quantity
                
                # 更新组合价值
                if old_price > 0:
                    price_change = current_price - old_price
                    self.portfolio_value += price_change * position.quantity
                    self.portfolio_pnl += price_change * position.quantity
        
        # 更新组合权重
        self._update_portfolio_weights()
        
        # 检查风险限制
        self._check_risk_limits()
    
    def _update_position(self, symbol: str, action: str, quantity: int, price: float):
        """更新持仓信息"""
        if symbol not in self.current_positions:
            self.current_positions[symbol] = PositionRisk(
                symbol=symbol,
                quantity=0,
                market_value=0,
                unrealized_pnl=0,
                realized_pnl=0,
                avg_cost=0,
                current_price=price,
                portfolio_weight=0
            )
        
        position = self.current_positions[symbol]
        
        if action.upper() == 'BUY':
            # 计算新的平均成本
            total_cost = position.avg_cost * position.quantity + price * quantity
            total_quantity = position.quantity + quantity
            position.avg_cost = total_cost / total_quantity if total_quantity > 0 else 0
            position.quantity = total_quantity
        
        elif action.upper() == 'SELL':
            if position.quantity >= quantity:
                # 计算已实现盈亏
                realized_pnl = (price - position.avg_cost) * quantity
                position.realized_pnl += realized_pnl
                position.quantity -= quantity
                
                if position.quantity == 0:
                    del self.current_positions[symbol]
                    return
            else:
                self.logger.warning(f"Attempting to sell more than held: {symbol}")
        
        # 更新市场价值和未实现盈亏
        position.current_price = price
        position.market_value = position.quantity * price
        position.unrealized_pnl = (price - position.avg_cost) * position.quantity
        
        # 更新组合价值
        self._update_portfolio_value()
    
    def _update_portfolio_value(self):
        """更新组合价值"""
        self.portfolio_value = sum(pos.market_value for pos in self.current_positions.values())
        self.portfolio_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.current_positions.values())
        
        # 更新最高水位和回撤
        if self.portfolio_value > self.high_water_mark:
            self.high_water_mark = self.portfolio_value
        
        if self.high_water_mark > 0:
            current_drawdown = (self.high_water_mark - self.portfolio_value) / self.high_water_mark
            self.max_portfolio_drawdown = max(self.max_portfolio_drawdown, current_drawdown)
    
    def _update_portfolio_weights(self):
        """更新组合权重"""
        if self.portfolio_value <= 0:
            return
        
        for position in self.current_positions.values():
            position.portfolio_weight = position.market_value / self.portfolio_value
    
    def _check_risk_limits(self):
        """检查风险限制"""
        # 检查单日损失限制
        if self.portfolio_value > 0 and self.daily_pnl / self.portfolio_value < -self.max_daily_loss:
            self._trigger_daily_loss_protection()
        
        # 检查最大回撤
        if self.max_portfolio_drawdown > self.max_drawdown:
            self._trigger_drawdown_protection()
        
        # 检查个股持仓限制
        for symbol, position in self.current_positions.items():
            if position.portfolio_weight > self.max_position_size * 1.2:  # 20%容差
                self._trigger_position_size_alert(symbol, position)
    
    def _trigger_daily_loss_protection(self):
        """触发单日损失保护"""
        self.trading_suspended = True
        loss_pct = (self.daily_pnl / self.portfolio_value * 100) if self.portfolio_value > 0 else 0
        
        self.logger.critical(f"Daily loss limit exceeded: {loss_pct:.2f}% > {self.max_daily_loss*100:.1f}%")
        self.logger.critical("Trading suspended for the day")
        
        # 这里可以添加告警通知
    
    def _trigger_drawdown_protection(self):
        """触发回撤保护"""
        self.emergency_stop_triggered = True
        self.trading_suspended = True
        
        drawdown_pct = self.max_portfolio_drawdown * 100
        
        self.logger.critical(f"Maximum drawdown exceeded: {drawdown_pct:.2f}% > {self.max_drawdown*100:.1f}%")
        self.logger.critical("Emergency stop triggered - all trading suspended")
        
        # 这里可以添加紧急平仓逻辑
    
    def _trigger_position_size_alert(self, symbol: str, position: PositionRisk):
        """触发持仓规模告警"""
        weight_pct = position.portfolio_weight * 100
        limit_pct = self.max_position_size * 100
        
        self.logger.warning(f"Position size alert: {symbol} {weight_pct:.2f}% > {limit_pct:.1f}%")
    
    def add_to_loss_cooldown(self, symbol: str, loss_amount: float):
        """将股票加入损失冷却期"""
        if loss_amount < 0:  # 确实是损失
            cooldown_end = date.today() + timedelta(days=self.loss_cooldown_days)
            self.loss_cooldown[symbol] = cooldown_end
            
            self.logger.warning(f"Added {symbol} to loss cooldown until {cooldown_end} (loss: ${loss_amount:.2f})")
    
    def remove_from_loss_cooldown(self, symbol: str):
        """从损失冷却期移除股票"""
        if symbol in self.loss_cooldown:
            del self.loss_cooldown[symbol]
            self.logger.info(f"Removed {symbol} from loss cooldown")
    
    def _get_sector(self, symbol: str) -> Optional[str]:
        """获取股票所属行业"""
        return self.sector_mapping.get(symbol)
    
    def _calculate_sector_exposure(self, sector: str) -> float:
        """计算行业敞口"""
        exposure = 0.0
        for position in self.current_positions.values():
            if self._get_sector(position.symbol) == sector:
                exposure += position.market_value
        return exposure
    
    def _check_portfolio_risk(self, symbol: str, quantity: int, price: float) -> Tuple[bool, str]:
        """检查组合风险"""
        # 简化的风险检查 - 实际应该使用VaR等更复杂的度量
        new_position_value = quantity * price
        
        if self.portfolio_value > 0:
            risk_increase = new_position_value / self.portfolio_value
            if risk_increase > self.max_portfolio_risk:
                return False, f"Portfolio risk increase too high: {risk_increase:.1%} > {self.max_portfolio_risk:.1%}"
        
        return True, ""
    
    def _check_liquidity_risk(self, symbol: str, quantity: int) -> Optional[str]:
        """检查流动性风险"""
        # 简化实现 - 实际应该查询实时成交量数据
        # 这里假设某些条件下会有流动性风险
        return None
    
    def _check_volatility_risk(self, symbol: str) -> Optional[str]:
        """检查波动性风险"""
        # 简化实现 - 实际应该计算实时波动率
        return None
    
    def _check_correlation_risk(self, symbol: str) -> Optional[str]:
        """检查相关性风险"""
        # 简化实现 - 实际应该计算与现有持仓的相关性
        return None
    
    def _calculate_portfolio_var(self) -> float:
        """计算组合VaR"""
        # 简化实现 - 实际应该使用历史或蒙特卡洛方法
        if len(self.pnl_history) < 30:
            return 0.0
        
        returns = np.array(list(self.pnl_history))
        return np.percentile(returns, 5)  # 95% VaR
    
    def _load_sector_mapping(self) -> Dict[str, str]:
        """加载行业分类映射"""
        # 简化的行业分类
        return {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'JPM': 'Financials',
            'BAC': 'Financials',
            'JNJ': 'Healthcare',
            'PFE': 'Healthcare',
            'XOM': 'Energy',
            'CVX': 'Energy'
        }
    
    def _start_monitoring(self):
        """启动风险监控线程"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Risk monitoring started")
    
    def _stop_monitoring(self):
        """停止风险监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """风险监控循环"""
        while self.is_monitoring:
            try:
                # 更新每日PnL
                self._update_daily_pnl()
                
                # 检查风险限制
                self._check_risk_limits()
                
                # 清理过期的冷却期
                self._cleanup_expired_cooldowns()
                
                # 保存风险数据
                self._save_risk_data()
                
                time.sleep(300)  # 每5分钟检查一次
                
            except Exception as e:
                self.logger.error(f"Error in risk monitoring: {e}")
                time.sleep(60)
    
    def _update_daily_pnl(self):
        """更新每日PnL"""
        today = date.today()
        today_trades = self.daily_trades.get(today, [])
        
        # 计算今日已实现盈亏
        daily_realized = sum(
            (trade['price'] - self._get_avg_cost_at_time(trade['symbol'], trade['timestamp'])) * trade['quantity']
            for trade in today_trades if trade['action'] == 'SELL'
        )
        
        # 计算今日未实现盈亏变化
        daily_unrealized = sum(
            position.unrealized_pnl for position in self.current_positions.values()
        )
        
        self.daily_pnl = daily_realized + daily_unrealized
    
    def _get_avg_cost_at_time(self, symbol: str, timestamp: datetime) -> float:
        """获取指定时间的平均成本（简化实现）"""
        if symbol in self.current_positions:
            return self.current_positions[symbol].avg_cost
        return 0.0
    
    def _cleanup_expired_cooldowns(self):
        """清理过期的冷却期"""
        today = date.today()
        expired_symbols = [
            symbol for symbol, end_date in self.loss_cooldown.items()
            if today > end_date
        ]
        
        for symbol in expired_symbols:
            self.remove_from_loss_cooldown(symbol)
    
    def _save_risk_data(self):
        """保存风险数据"""
        try:
            os.makedirs(os.path.dirname(self.risk_data_file), exist_ok=True)
            
            risk_data = {
                'portfolio_value': self.portfolio_value,
                'portfolio_pnl': self.portfolio_pnl,
                'daily_pnl': self.daily_pnl,
                'max_portfolio_drawdown': self.max_portfolio_drawdown,
                'high_water_mark': self.high_water_mark,
                'emergency_stop_triggered': self.emergency_stop_triggered,
                'trading_suspended': self.trading_suspended,
                'positions': {symbol: pos.to_dict() for symbol, pos in self.current_positions.items()},
                'loss_cooldown': {symbol: end_date.isoformat() for symbol, end_date in self.loss_cooldown.items()},
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.risk_data_file, 'w', encoding='utf-8') as f:
                json.dump(risk_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving risk data: {e}")
    
    def _load_risk_data(self):
        """加载风险数据"""
        try:
            if os.path.exists(self.risk_data_file):
                with open(self.risk_data_file, 'r', encoding='utf-8') as f:
                    risk_data = json.load(f)
                
                self.portfolio_value = risk_data.get('portfolio_value', 0.0)
                self.portfolio_pnl = risk_data.get('portfolio_pnl', 0.0)
                self.daily_pnl = risk_data.get('daily_pnl', 0.0)
                self.max_portfolio_drawdown = risk_data.get('max_portfolio_drawdown', 0.0)
                self.high_water_mark = risk_data.get('high_water_mark', 0.0)
                self.emergency_stop_triggered = risk_data.get('emergency_stop_triggered', False)
                self.trading_suspended = risk_data.get('trading_suspended', False)
                
                # 加载持仓
                positions_data = risk_data.get('positions', {})
                for symbol, pos_dict in positions_data.items():
                    self.current_positions[symbol] = PositionRisk(
                        symbol=pos_dict['symbol'],
                        quantity=pos_dict['quantity'],
                        market_value=pos_dict['market_value'],
                        unrealized_pnl=pos_dict['unrealized_pnl'],
                        realized_pnl=pos_dict['realized_pnl'],
                        avg_cost=pos_dict['avg_cost'],
                        current_price=pos_dict['current_price'],
                        portfolio_weight=pos_dict['portfolio_weight'],
                        var_1d=pos_dict.get('var_1d', 0.0),
                        max_drawdown=pos_dict.get('max_drawdown', 0.0),
                        days_held=pos_dict.get('days_held', 0)
                    )
                
                # 加载冷却期
                cooldown_data = risk_data.get('loss_cooldown', {})
                for symbol, end_date_str in cooldown_data.items():
                    self.loss_cooldown[symbol] = date.fromisoformat(end_date_str)
                
                self.logger.info("Risk data loaded successfully")
                
        except Exception as e:
            self.logger.error(f"Error loading risk data: {e}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """获取风险状态报告"""
        return {
            'portfolio_value': self.portfolio_value,
            'portfolio_pnl': self.portfolio_pnl,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.portfolio_value * 100 if self.portfolio_value > 0 else 0,
            'max_drawdown': self.max_portfolio_drawdown * 100,
            'max_drawdown_limit': self.max_drawdown * 100,
            'emergency_stop': self.emergency_stop_triggered,
            'trading_suspended': self.trading_suspended,
            'positions_count': len(self.current_positions),
            'daily_trades_count': len(self.daily_trades.get(date.today(), [])),
            'daily_trades_limit': self.max_new_positions_per_day,
            'cooldown_symbols': list(self.loss_cooldown.keys()),
            'risk_utilization': {
                'max_position_size': max((pos.portfolio_weight for pos in self.current_positions.values()), default=0) / self.max_position_size,
                'daily_loss': abs(self.daily_pnl / self.portfolio_value) / self.max_daily_loss if self.portfolio_value > 0 else 0,
                'drawdown': self.max_portfolio_drawdown / self.max_drawdown
            }
        }
    
    def reset_emergency_stop(self):
        """重置紧急停止（需要手动操作）"""
        self.emergency_stop_triggered = False
        self.trading_suspended = False
        self.logger.info("Emergency stop reset - trading resumed")
    
    def cleanup(self):
        """清理资源"""
        self._stop_monitoring()
        self._save_risk_data()
        self.logger.info("Risk manager cleaned up")


# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建配置
    config = {
        'max_portfolio_risk': 0.02,
        'max_position_size': 0.05,
        'max_sector_exposure': 0.25,
        'max_daily_loss': 0.05,
        'max_drawdown': 0.10,
        'max_new_positions_per_day': 10,
        'max_trades_per_symbol_per_day': 3,
        'loss_cooldown_days': 3,
        'min_time_between_trades_minutes': 15,
        'risk_data_file': 'risk/risk_data.json'
    }
    
    # 创建风险管理器
    risk_manager = EnhancedRiskManager(config)
    
    # 测试交易前检查
    check_result = risk_manager.pre_trade_check('AAPL', 'BUY', 100, 150.0, 'test_strategy')
    print(f"Pre-trade check result: {check_result.result.value}")
    print(f"Approved quantity: {check_result.approved_quantity}")
    print(f"Risk level: {check_result.risk_level.value}")
    
    if check_result.warnings:
        print(f"Warnings: {check_result.warnings}")
    
    if check_result.reasons:
        print(f"Reasons: {check_result.reasons}")
    
    print("Enhanced risk manager initialized successfully")