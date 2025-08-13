#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一持仓管理器 - 解决持仓数据分散管理的问题
提供线程安全、数据一致性的统一持仓管理接口
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from threading import Lock
from collections import defaultdict
from enum import Enum
import json

class PositionType(Enum):
    """持仓类型"""
    LONG = "long"
    SHORT = "short"
    CLOSED = "closed"

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    entry_time: float
    last_updated: float
    
    # 计算字段
    market_value: float = field(init=False)
    unrealized_pnl: float = field(init=False)
    unrealized_pnl_pct: float = field(init=False)
    position_type: PositionType = field(init=False)
    
    # 可选字段
    sector: Optional[str] = None
    avg_cost: Optional[float] = None
    realized_pnl: float = 0.0
    
    def __post_init__(self):
        """计算派生字段"""
        self.market_value = abs(self.quantity) * self.current_price
        
        if self.quantity > 0:
            self.position_type = PositionType.LONG
            self.unrealized_pnl = self.quantity * (self.current_price - self.entry_price)
        elif self.quantity < 0:
            self.position_type = PositionType.SHORT
            self.unrealized_pnl = abs(self.quantity) * (self.entry_price - self.current_price)
        else:
            self.position_type = PositionType.CLOSED
            self.unrealized_pnl = 0.0
        
        if self.entry_price > 0:
            if self.position_type == PositionType.LONG:
                self.unrealized_pnl_pct = (self.current_price / self.entry_price - 1)
            elif self.position_type == PositionType.SHORT:
                self.unrealized_pnl_pct = (self.entry_price / self.current_price - 1)
            else:
                self.unrealized_pnl_pct = 0.0
        else:
            self.unrealized_pnl_pct = 0.0

@dataclass 
class PortfolioSummary:
    """投资组合摘要"""
    total_positions: int
    total_market_value: float
    total_unrealized_pnl: float
    total_unrealized_pnl_pct: float
    long_positions: int
    short_positions: int
    largest_position: Optional[Position] = None
    largest_winner: Optional[Position] = None
    largest_loser: Optional[Position] = None
    concentration_risk: float = 0.0  # 最大持仓占比

class UnifiedPositionManager:
    """统一持仓管理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("UnifiedPositionManager")
        
        # 持仓数据
        self._positions: Dict[str, Position] = {}
        self._position_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._sync_lock = Lock()  # 同步操作锁
        
        # 配置
        self.max_history_length = 1000
        self.price_update_threshold = 0.001  # 价格变化阈值
        
        # 统计
        self._stats = {
            'position_updates': 0,
            'price_updates': 0,
            'total_trades': 0,
            'last_snapshot_time': 0.0
        }
        
        # 缓存
        self._portfolio_summary: Optional[PortfolioSummary] = None
        self._summary_cache_time = 0.0
        self._summary_cache_ttl = 5.0  # 5秒缓存
    
    async def update_position(self, symbol: str, quantity: int, 
                            current_price: float, entry_price: Optional[float] = None,
                            sector: Optional[str] = None) -> Position:
        """更新持仓信息"""
        async with self._lock:
            current_time = time.time()
            
            # 获取现有持仓
            existing_position = self._positions.get(symbol)
            
            if quantity == 0:
                # 平仓
                if existing_position:
                    self._record_position_change(existing_position, None, "CLOSE")
                    del self._positions[symbol]
                    self.logger.info(f"平仓: {symbol}")
                    # 返回空的持仓对象而不是None
                    return Position(
                        symbol=symbol,
                        quantity=0,
                        entry_price=existing_position.entry_price,
                        current_price=current_price,
                        sector=existing_position.sector,
                        last_updated=current_time
                    )
                else:
                    # 没有现有持仓，创建一个空持仓
                    return Position(
                        symbol=symbol,
                        quantity=0,
                        entry_price=current_price,
                        current_price=current_price,
                        sector=sector or "Unknown",
                        last_updated=current_time
                    )
            
            # 确定入场价格
            if existing_position and entry_price is None:
                # 使用现有入场价格
                final_entry_price = existing_position.entry_price
                final_entry_time = existing_position.entry_time
            else:
                # 新仓位或指定了新的入场价格
                final_entry_price = entry_price or current_price
                final_entry_time = current_time
            
            # 创建或更新持仓
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=final_entry_price,
                current_price=current_price,
                entry_time=final_entry_time,
                last_updated=current_time,
                sector=sector,
                avg_cost=existing_position.avg_cost if existing_position else None,
                realized_pnl=existing_position.realized_pnl if existing_position else 0.0
            )
            
            # 记录变化
            if existing_position:
                self._record_position_change(existing_position, position, "UPDATE")
            else:
                self._record_position_change(None, position, "OPEN")
                self.logger.info(f"开仓: {symbol} {quantity} @ {final_entry_price:.2f}")
            
            self._positions[symbol] = position
            self._stats['position_updates'] += 1
            
            # 清除摘要缓存
            self._portfolio_summary = None
            
            return position
    
    async def update_price(self, symbol: str, new_price: float) -> bool:
        """更新持仓的当前价格"""
        async with self._lock:
            position = self._positions.get(symbol)
            if not position:
                return False
            
            # 检查价格变化是否足够大
            price_change_pct = abs(new_price / position.current_price - 1) if position.current_price > 0 else 1.0
            if price_change_pct < self.price_update_threshold:
                return False
            
            # 更新价格
            old_position = position
            position.current_price = new_price
            position.last_updated = time.time()
            
            # 重新计算派生字段
            position.__post_init__()
            
            self._stats['price_updates'] += 1
            
            # 清除摘要缓存
            self._portfolio_summary = None
            
            return True
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定持仓"""
        with self._sync_lock:
            return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """获取所有持仓"""
        with self._sync_lock:
            return self._positions.copy()
    
    def get_quantity(self, symbol: str) -> int:
        """获取持仓数量（兼容接口）"""
        with self._sync_lock:
            position = self._positions.get(symbol)
            return position.quantity if position else 0
    
    def get_market_value(self, symbol: str) -> float:
        """获取持仓市值"""
        with self._sync_lock:
            position = self._positions.get(symbol)
            return position.market_value if position else 0.0
    
    def get_unrealized_pnl(self, symbol: str) -> float:
        """获取未实现损益"""
        with self._sync_lock:
            position = self._positions.get(symbol)
            return position.unrealized_pnl if position else 0.0
    
    def get_symbols(self) -> List[str]:
        """获取所有持仓标的"""
        with self._sync_lock:
            return list(self._positions.keys())
    
    def get_long_positions(self) -> Dict[str, Position]:
        """获取多头持仓"""
        with self._sync_lock:
            return {
                symbol: pos for symbol, pos in self._positions.items()
                if pos.position_type == PositionType.LONG
            }
    
    def get_short_positions(self) -> Dict[str, Position]:
        """获取空头持仓"""
        with self._sync_lock:
            return {
                symbol: pos for symbol, pos in self._positions.items()
                if pos.position_type == PositionType.SHORT
            }
    
    def get_portfolio_summary(self, force_refresh: bool = False) -> PortfolioSummary:
        """获取投资组合摘要"""
        current_time = time.time()
        
        # 检查缓存
        if (not force_refresh and self._portfolio_summary and 
            current_time - self._summary_cache_time < self._summary_cache_ttl):
            return self._portfolio_summary
        
        with self._sync_lock:
            positions = list(self._positions.values())
        
        if not positions:
            return PortfolioSummary(
                total_positions=0,
                total_market_value=0.0,
                total_unrealized_pnl=0.0,
                total_unrealized_pnl_pct=0.0,
                long_positions=0,
                short_positions=0
            )
        
        # 计算汇总数据
        total_market_value = sum(pos.market_value for pos in positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        
        long_positions = [pos for pos in positions if pos.position_type == PositionType.LONG]
        short_positions = [pos for pos in positions if pos.position_type == PositionType.SHORT]
        
        # 找出最大持仓、最大盈利、最大亏损
        largest_position = max(positions, key=lambda p: p.market_value) if positions else None
        
        profitable_positions = [pos for pos in positions if pos.unrealized_pnl > 0]
        losing_positions = [pos for pos in positions if pos.unrealized_pnl < 0]
        
        largest_winner = max(profitable_positions, key=lambda p: p.unrealized_pnl) if profitable_positions else None
        largest_loser = min(losing_positions, key=lambda p: p.unrealized_pnl) if losing_positions else None
        
        # 计算集中度风险
        concentration_risk = (largest_position.market_value / total_market_value) if largest_position and total_market_value > 0 else 0.0
        
        # 计算总体收益率
        total_cost = sum(abs(pos.quantity) * pos.entry_price for pos in positions)
        total_unrealized_pnl_pct = total_unrealized_pnl / total_cost if total_cost > 0 else 0.0
        
        summary = PortfolioSummary(
            total_positions=len(positions),
            total_market_value=total_market_value,
            total_unrealized_pnl=total_unrealized_pnl,
            total_unrealized_pnl_pct=total_unrealized_pnl_pct,
            long_positions=len(long_positions),
            short_positions=len(short_positions),
            largest_position=largest_position,
            largest_winner=largest_winner,
            largest_loser=largest_loser,
            concentration_risk=concentration_risk
        )
        
        # 更新缓存
        self._portfolio_summary = summary
        self._summary_cache_time = current_time
        
        return summary
    
    def _record_position_change(self, old_position: Optional[Position], 
                              new_position: Optional[Position], action: str):
        """记录持仓变化历史"""
        record = {
            'timestamp': time.time(),
            'action': action,
            'symbol': new_position.symbol if new_position else (old_position.symbol if old_position else None),
            'old_quantity': old_position.quantity if old_position else 0,
            'new_quantity': new_position.quantity if new_position else 0,
            'old_price': old_position.current_price if old_position else 0.0,
            'new_price': new_position.current_price if new_position else 0.0
        }
        
        self._position_history.append(record)
        
        # 限制历史长度
        if len(self._position_history) > self.max_history_length:
            self._position_history = self._position_history[-self.max_history_length:]
    
    def get_position_history(self, symbol: Optional[str] = None, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """获取持仓变化历史"""
        with self._sync_lock:
            history = self._position_history
            
            if symbol:
                history = [record for record in history if record.get('symbol') == symbol]
            
            return history[-limit:] if limit > 0 else history
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        summary = self.get_portfolio_summary()
        
        with self._sync_lock:
            recent_trades = [
                record for record in self._position_history 
                if time.time() - record['timestamp'] < 86400  # 最近24小时
            ]
        
        return {
            'positions': {
                'total': summary.total_positions,
                'long': summary.long_positions,
                'short': summary.short_positions
            },
            'portfolio': {
                'total_market_value': summary.total_market_value,
                'total_unrealized_pnl': summary.total_unrealized_pnl,
                'unrealized_pnl_pct': summary.total_unrealized_pnl_pct,
                'concentration_risk': summary.concentration_risk
            },
            'activity': {
                'position_updates': self._stats['position_updates'],
                'price_updates': self._stats['price_updates'],
                'recent_trades': len(recent_trades)
            },
            'cache': {
                'summary_cached': self._portfolio_summary is not None,
                'cache_age': time.time() - self._summary_cache_time
            }
        }
    
    async def bulk_update_prices(self, price_updates: Dict[str, float]) -> int:
        """批量更新价格"""
        updated_count = 0
        
        async with self._lock:
            for symbol, new_price in price_updates.items():
                if await self.update_price(symbol, new_price):
                    updated_count += 1
        
        return updated_count
    
    async def sync_with_broker_positions(self, broker_positions: Dict[str, Any],
                                       price_source: Dict[str, float]) -> Dict[str, Any]:
        """与经纪商持仓同步"""
        sync_result = {
            'added': [],
            'updated': [],
            'removed': [],
            'price_updated': []
        }
        
        async with self._lock:
            current_symbols = set(self._positions.keys())
            broker_symbols = set(broker_positions.keys())
            
            # 添加新持仓
            for symbol in broker_symbols - current_symbols:
                broker_qty = broker_positions[symbol]
                if broker_qty != 0:
                    current_price = price_source.get(symbol, 0.0)
                    if current_price > 0:
                        await self.update_position(symbol, broker_qty, current_price, current_price)
                        sync_result['added'].append(symbol)
            
            # 更新现有持仓
            for symbol in current_symbols & broker_symbols:
                broker_qty = broker_positions[symbol]
                current_position = self._positions[symbol]
                
                if broker_qty != current_position.quantity:
                    current_price = price_source.get(symbol, current_position.current_price)
                    await self.update_position(symbol, broker_qty, current_price)
                    sync_result['updated'].append(symbol)
                else:
                    # 只更新价格
                    current_price = price_source.get(symbol)
                    if current_price and await self.update_price(symbol, current_price):
                        sync_result['price_updated'].append(symbol)
            
            # 移除已平仓持仓
            for symbol in current_symbols - broker_symbols:
                await self.update_position(symbol, 0, self._positions[symbol].current_price)
                sync_result['removed'].append(symbol)
        
        self.logger.info(f"持仓同步完成: +{len(sync_result['added'])} "
                        f"~{len(sync_result['updated'])} -{len(sync_result['removed'])}")
        
        return sync_result
    
    def clear_all_positions(self):
        """清空所有持仓（谨慎使用）"""
        with self._sync_lock:
            symbols = list(self._positions.keys())
            self._positions.clear()
            self._portfolio_summary = None
            
            self.logger.warning(f"已清空所有持仓: {len(symbols)} 个")

# 全局实例
_global_position_manager: Optional[UnifiedPositionManager] = None

def get_position_manager() -> UnifiedPositionManager:
    """获取全局持仓管理器实例"""
    global _global_position_manager
    if _global_position_manager is None:
        _global_position_manager = UnifiedPositionManager()
    return _global_position_manager
