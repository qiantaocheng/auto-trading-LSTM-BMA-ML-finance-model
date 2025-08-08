#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态仓位管理系统 (DynamicPositionSizing)
基于反马丁格尔（Anti-Martingale）与金字塔加仓（Pyramiding）策略
实现连续盈利周期监控和自动资金分配调整

核心功能:
1. 连续盈利周期监控
2. 动态加仓触发机制
3. 金字塔式仓位递增
4. 风险控制和止损管理
5. 冷静期和撤退机制

Authors: AI Assistant
Version: 1.0
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum
import asyncio
import threading
import time

class PositionSizingMode(Enum):
    """仓位调整模式"""
    STATIC = "static"                # 静态仓位
    DYNAMIC = "dynamic"              # 动态仓位
    PYRAMIDING = "pyramiding"        # 金字塔加仓
    ANTI_MARTINGALE = "anti_martingale"  # 反马丁格尔

class TradingState(Enum):
    """交易状态"""
    NORMAL = "normal"               # 正常状态
    WINNING_STREAK = "winning_streak"  # 连胜状态
    LOSING_STREAK = "losing_streak"    # 连败状态
    COOLDOWN = "cooldown"           # 冷静期

@dataclass
class TradeResult:
    """交易结果数据类"""
    timestamp: datetime
    symbol: str
    action: str  # OPEN, CLOSE, ADD
    position_id: str
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: float
    pnl_pct: float
    position_size_pct: float
    is_addon: bool = False
    reason: str = ""

@dataclass
class PositionState:
    """持仓状态数据类"""
    position_id: str
    symbol: str
    base_quantity: int
    addon_quantity: int
    total_quantity: int
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    base_position_size: float
    addon_position_size: float
    total_position_size: float
    max_addon_level: int
    current_addon_level: int
    stop_loss_price: float
    take_profit_price: float
    created_at: datetime
    last_addon_at: Optional[datetime]

@dataclass
class DynamicSizingConfig:
    """动态仓位配置"""
    # 基础配置
    base_risk_pct: float = 0.02           # 基础仓位风险 (2%)
    max_exposure_pct: float = 0.08        # 最大总敞口 (8%)
    
    # 连胜监控配置
    win_streak_trigger: int = 3           # 连胜触发阈值
    loss_streak_cooldown: int = 2         # 连败触发冷静期阈值
    
    # 加仓配置
    addon_aggressive_factor: float = 0.2  # 加仓激进度 (β)
    max_addon_levels: int = 3             # 最大加仓层数
    addon_size_ratio: float = 0.5         # 加仓相对基础仓位比例
    
    # 风险控制
    atr_multiplier: float = 1.5           # ATR止损倍数
    volume_threshold: float = 1.2         # 成交量确认倍数
    momentum_threshold: float = 0.02      # 动量确认阈值
    
    # 冷静期配置
    cooldown_duration_hours: int = 24     # 冷静期持续时间（小时）
    recovery_win_streak: int = 2          # 恢复正常状态所需连胜次数

class DynamicPositionSizing:
    """动态仓位管理系统"""
    
    def __init__(self, config: DynamicSizingConfig = None):
        """
        初始化动态仓位管理系统
        
        Args:
            config: 动态仓位配置
        """
        self.config = config or DynamicSizingConfig()
        self.logger = logging.getLogger(__name__)
        
        # 状态跟踪
        self.win_streak = 0
        self.loss_streak = 0
        self.trading_state = TradingState.NORMAL
        self.cooldown_start_time = None
        
        # 交易历史
        self.trade_history: List[TradeResult] = []
        self.position_states: Dict[str, PositionState] = {}
        
        # 性能统计
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_exposure': 0.0,
            'addon_trades': 0,
            'addon_success_rate': 0.0
        }
        
        # 数据存储
        self.data_dir = "dynamic_sizing_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self.data_dir, f"dynamic_sizing_{datetime.now().strftime('%Y%m%d')}.log")
        
        # 配置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # 配置logger
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def calculate_position_size(self, symbol: str, signal_strength: float = 1.0, 
                              is_addon: bool = False, base_position_id: str = None) -> float:
        """
        计算仓位大小
        
        Args:
            symbol: 交易标的
            signal_strength: 信号强度 (0-1)
            is_addon: 是否为加仓
            base_position_id: 基础仓位ID（加仓时需要）
            
        Returns:
            仓位大小百分比
        """
        try:
            # 检查冷静期
            if self._is_in_cooldown():
                self.logger.info(f"处于冷静期，使用基础仓位: {self.config.base_risk_pct:.2%}")
                return self.config.base_risk_pct * signal_strength
            
            # 基础仓位计算
            if not is_addon:
                base_size = self.config.base_risk_pct
                
                # 动态调整基础仓位
                if self.trading_state == TradingState.WINNING_STREAK:
                    # 连胜状态下增加基础仓位
                    dynamic_factor = 1 + (self.config.addon_aggressive_factor * self.win_streak)
                    base_size = base_size * dynamic_factor
                    
                elif self.trading_state == TradingState.LOSING_STREAK:
                    # 连败状态下减少基础仓位
                    reduction_factor = 0.8 ** self.loss_streak
                    base_size = base_size * reduction_factor
                
                # 应用信号强度
                final_size = base_size * signal_strength
                
                # 确保不超过最大敞口
                current_exposure = self._calculate_current_exposure()
                max_allowed = self.config.max_exposure_pct - current_exposure
                final_size = min(final_size, max_allowed)
                
                self.logger.info(f"基础仓位计算 - 符号: {symbol}, 连胜: {self.win_streak}, "
                               f"连败: {self.loss_streak}, 仓位: {final_size:.2%}")
                
                return max(final_size, 0.0)
            
            # 加仓计算
            else:
                if not base_position_id or base_position_id not in self.position_states:
                    self.logger.error(f"加仓失败：未找到基础仓位 {base_position_id}")
                    return 0.0
                
                base_position = self.position_states[base_position_id]
                
                # 检查是否可以加仓
                if not self._can_add_position(base_position):
                    return 0.0
                
                # 计算加仓大小
                addon_size = base_position.base_position_size * self.config.addon_size_ratio
                
                # 应用金字塔递减
                level_reduction = 0.8 ** base_position.current_addon_level
                addon_size = addon_size * level_reduction
                
                # 检查总敞口限制
                current_exposure = self._calculate_current_exposure()
                max_allowed = self.config.max_exposure_pct - current_exposure
                addon_size = min(addon_size, max_allowed)
                
                self.logger.info(f"加仓计算 - 符号: {symbol}, 层级: {base_position.current_addon_level}, "
                               f"加仓大小: {addon_size:.2%}")
                
                return max(addon_size, 0.0)
                
        except Exception as e:
            self.logger.error(f"计算仓位大小失败: {e}")
            return self.config.base_risk_pct * 0.5  # 安全的小仓位
    
    def check_addon_trigger(self, position_id: str, current_price: float, 
                           market_data: Dict = None) -> bool:
        """
        检查是否触发加仓条件
        
        Args:
            position_id: 持仓ID
            current_price: 当前价格
            market_data: 市场数据（包含成交量、ATR等）
            
        Returns:
            是否应该触发加仓
        """
        try:
            # 检查持仓是否存在
            if position_id not in self.position_states:
                return False
            
            position = self.position_states[position_id]
            
            # 基本条件检查
            if not self._can_add_position(position):
                return False
            
            # 连胜条件检查
            if self.win_streak < self.config.win_streak_trigger:
                self.logger.debug(f"连胜次数不足: {self.win_streak} < {self.config.win_streak_trigger}")
                return False
            
            # 持仓盈利检查
            unrealized_pnl_pct = (current_price - position.avg_entry_price) / position.avg_entry_price
            if unrealized_pnl_pct <= 0:
                self.logger.debug(f"持仓未盈利: {unrealized_pnl_pct:.2%}")
                return False
            
            # 动量确认
            if market_data and not self._check_momentum_confirmation(market_data):
                self.logger.debug("动量确认失败")
                return False
            
            # 成交量确认
            if market_data and not self._check_volume_confirmation(market_data):
                self.logger.debug("成交量确认失败")
                return False
            
            # 技术位确认（可选）
            if market_data and not self._check_technical_confirmation(current_price, market_data):
                self.logger.debug("技术位确认失败")
                return False
            
            self.logger.info(f"加仓条件满足 - 持仓: {position_id}, 连胜: {self.win_streak}, "
                           f"盈利: {unrealized_pnl_pct:.2%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"检查加仓触发条件失败: {e}")
            return False
    
    def record_trade_result(self, trade_result: TradeResult):
        """
        记录交易结果并更新状态
        
        Args:
            trade_result: 交易结果
        """
        try:
            # 添加到历史记录
            self.trade_history.append(trade_result)
            
            # 更新连胜/连败计数
            if trade_result.action == "CLOSE":
                self._update_streak_counters(trade_result)
            
            # 更新持仓状态
            if trade_result.action == "OPEN":
                self._create_position_state(trade_result)
            elif trade_result.action == "ADD":
                self._update_position_addon(trade_result)
            elif trade_result.action == "CLOSE":
                self._close_position_state(trade_result)
            
            # 更新交易状态
            self._update_trading_state()
            
            # 更新性能统计
            self._update_performance_stats()
            
            # 保存数据
            self._save_trade_data()
            
            self.logger.info(f"交易记录已更新 - 动作: {trade_result.action}, "
                           f"PnL: {trade_result.pnl:.2f}, 连胜: {self.win_streak}, "
                           f"连败: {self.loss_streak}, 状态: {self.trading_state.value}")
            
        except Exception as e:
            self.logger.error(f"记录交易结果失败: {e}")
    
    def get_position_management_signal(self, position_id: str, current_price: float, 
                                     market_data: Dict = None) -> Dict:
        """
        获取仓位管理信号
        
        Args:
            position_id: 持仓ID
            current_price: 当前价格
            market_data: 市场数据
            
        Returns:
            仓位管理信号字典
        """
        try:
            if position_id not in self.position_states:
                return {'action': 'HOLD', 'reason': '持仓不存在'}
            
            position = self.position_states[position_id]
            signal = {'action': 'HOLD', 'reason': '正常持有'}
            
            # 更新持仓状态
            self._update_position_price(position_id, current_price)
            
            # 计算动态止损（追踪止损）
            dynamic_stop_loss = self._calculate_dynamic_stop_loss(position, current_price)
            
            # 检查止损（优先检查）
            if current_price <= dynamic_stop_loss:
                signal = {
                    'action': 'CLOSE',
                    'reason': f'触发动态止损: {current_price:.2f} <= {dynamic_stop_loss:.2f}',
                    'urgency': 'HIGH',
                    'stop_loss_type': 'dynamic'
                }
            
            # 检查止盈
            elif current_price >= position.take_profit_price:
                signal = {
                    'action': 'CLOSE',
                    'reason': f'触发止盈: {current_price:.2f} >= {position.take_profit_price:.2f}',
                    'urgency': 'MEDIUM',
                    'stop_loss_type': 'take_profit'
                }
            
            # 检查加仓机会（仅在未触发止损/止盈时）
            elif self.check_addon_trigger(position_id, current_price, market_data):
                addon_size = self.calculate_position_size(
                    position.symbol, 1.0, True, position_id
                )
                if addon_size > 0:
                    signal = {
                        'action': 'ADD',
                        'reason': f'加仓机会: 连胜{self.win_streak}次',
                        'addon_size': addon_size,
                        'urgency': 'LOW'
                    }
            
            # 检查风险控制信号
            elif self._should_reduce_position(position, current_price, market_data):
                signal = {
                    'action': 'REDUCE',
                    'reason': '风险控制：减少仓位',
                    'urgency': 'MEDIUM'
                }
            
            # 添加当前状态信息
            signal.update({
                'position_pnl': position.unrealized_pnl,
                'position_pnl_pct': position.unrealized_pnl_pct,
                'win_streak': self.win_streak,
                'loss_streak': self.loss_streak,
                'trading_state': self.trading_state.value,
                'total_exposure': self._calculate_current_exposure(),
                'dynamic_stop_loss': dynamic_stop_loss,
                'original_stop_loss': position.stop_loss_price
            })
            
            return signal
            
        except Exception as e:
            self.logger.error(f"获取仓位管理信号失败: {e}")
            return {'action': 'HOLD', 'reason': f'错误: {e}'}
    
    def _can_add_position(self, position: PositionState) -> bool:
        """检查是否可以加仓"""
        # 检查最大加仓层数
        if position.current_addon_level >= self.config.max_addon_levels:
            return False
        
        # 检查冷静期
        if self._is_in_cooldown():
            return False
        
        # 检查总敞口
        current_exposure = self._calculate_current_exposure()
        if current_exposure >= self.config.max_exposure_pct * 0.9:  # 90%上限
            return False
        
        # 检查时间间隔（避免频繁加仓）
        if position.last_addon_at:
            time_since_last = datetime.now() - position.last_addon_at
            if time_since_last.total_seconds() < 3600:  # 1小时内不重复加仓
                return False
        
        return True
    
    def _check_momentum_confirmation(self, market_data: Dict) -> bool:
        """检查动量确认"""
        try:
            if 'price_change_pct' not in market_data:
                return True  # 如果没有数据，默认通过
            
            price_change = market_data['price_change_pct']
            return abs(price_change) >= self.config.momentum_threshold
            
        except Exception:
            return True
    
    def _check_volume_confirmation(self, market_data: Dict) -> bool:
        """检查成交量确认"""
        try:
            if 'volume_ratio' not in market_data:
                return True  # 如果没有数据，默认通过
            
            volume_ratio = market_data['volume_ratio']
            return volume_ratio >= self.config.volume_threshold
            
        except Exception:
            return True
    
    def _check_technical_confirmation(self, current_price: float, market_data: Dict) -> bool:
        """检查技术位确认"""
        try:
            # 可以添加技术指标确认逻辑
            # 例如：突破布林带中轨、SuperTrend等
            if 'bollinger_middle' in market_data:
                return current_price > market_data['bollinger_middle']
            
            return True  # 默认通过
            
        except Exception:
            return True
    
    def _update_streak_counters(self, trade_result: TradeResult):
        """更新连胜/连败计数器"""
        if trade_result.pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0
    
    def _update_trading_state(self):
        """更新交易状态"""
        # 检查冷静期
        if self._is_in_cooldown():
            self.trading_state = TradingState.COOLDOWN
            return
        
        # 根据连胜/连败情况更新状态
        if self.win_streak >= self.config.win_streak_trigger:
            self.trading_state = TradingState.WINNING_STREAK
        elif self.loss_streak >= self.config.loss_streak_cooldown:
            self.trading_state = TradingState.LOSING_STREAK
            self._start_cooldown()
        else:
            self.trading_state = TradingState.NORMAL
    
    def _is_in_cooldown(self) -> bool:
        """检查是否在冷静期"""
        if not self.cooldown_start_time:
            return False
        
        cooldown_duration = timedelta(hours=self.config.cooldown_duration_hours)
        return datetime.now() - self.cooldown_start_time < cooldown_duration
    
    def _start_cooldown(self):
        """开始冷静期"""
        self.cooldown_start_time = datetime.now()
        self.logger.warning(f"进入冷静期 - 连败: {self.loss_streak}")
    
    def _calculate_current_exposure(self) -> float:
        """计算当前总敞口"""
        total_exposure = sum(pos.total_position_size for pos in self.position_states.values())
        return total_exposure
    
    def _create_position_state(self, trade_result: TradeResult):
        """创建持仓状态"""
        position_size = trade_result.position_size_pct
        # 优化止损计算：使用更敏感的止损
        stop_loss_price = trade_result.entry_price * (1 - 0.02)  # 2%止损（更敏感）
        take_profit_price = trade_result.entry_price * (1 + 0.06)  # 6%止盈
        
        position = PositionState(
            position_id=trade_result.position_id,
            symbol=trade_result.symbol,
            base_quantity=trade_result.quantity,
            addon_quantity=0,
            total_quantity=trade_result.quantity,
            avg_entry_price=trade_result.entry_price,
            current_price=trade_result.entry_price,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            base_position_size=position_size,
            addon_position_size=0.0,
            total_position_size=position_size,
            max_addon_level=self.config.max_addon_levels,
            current_addon_level=0,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            created_at=trade_result.timestamp,
            last_addon_at=None
        )
        
        self.position_states[trade_result.position_id] = position
    
    def _update_position_addon(self, trade_result: TradeResult):
        """更新加仓状态"""
        if trade_result.position_id in self.position_states:
            position = self.position_states[trade_result.position_id]
            
            # 更新数量和价格
            new_total_quantity = position.total_quantity + trade_result.quantity
            new_avg_price = (
                (position.avg_entry_price * position.total_quantity + 
                 trade_result.entry_price * trade_result.quantity) / new_total_quantity
            )
            
            position.addon_quantity += trade_result.quantity
            position.total_quantity = new_total_quantity
            position.avg_entry_price = new_avg_price
            position.addon_position_size += trade_result.position_size_pct
            position.total_position_size += trade_result.position_size_pct
            position.current_addon_level += 1
            position.last_addon_at = trade_result.timestamp
            
            # 更新止损止盈（基于新的平均价格，使用优化后的计算）
            position.stop_loss_price = new_avg_price * (1 - 0.02)  # 2%止损
            position.take_profit_price = new_avg_price * (1 + 0.06)  # 6%止盈
    
    def _update_position_price(self, position_id: str, current_price: float):
        """更新持仓当前价格和盈亏"""
        if position_id in self.position_states:
            position = self.position_states[position_id]
            position.current_price = current_price
            position.unrealized_pnl = (current_price - position.avg_entry_price) * position.total_quantity
            position.unrealized_pnl_pct = (current_price - position.avg_entry_price) / position.avg_entry_price
    
    def _close_position_state(self, trade_result: TradeResult):
        """关闭持仓状态"""
        if trade_result.position_id in self.position_states:
            del self.position_states[trade_result.position_id]
    
    def _update_performance_stats(self):
        """更新性能统计"""
        if not self.trade_history:
            return
        
        closed_trades = [t for t in self.trade_history if t.action == "CLOSE"]
        if not closed_trades:
            return
        
        self.performance_stats['total_trades'] = len(closed_trades)
        self.performance_stats['winning_trades'] = len([t for t in closed_trades if t.pnl > 0])
        self.performance_stats['losing_trades'] = len([t for t in closed_trades if t.pnl <= 0])
        self.performance_stats['win_rate'] = self.performance_stats['winning_trades'] / self.performance_stats['total_trades']
        self.performance_stats['total_pnl'] = sum(t.pnl for t in closed_trades)
        self.performance_stats['addon_trades'] = len([t for t in self.trade_history if t.is_addon])
        
        # 计算加仓成功率
        addon_closes = [t for t in closed_trades if any(h.is_addon and h.position_id == t.position_id for h in self.trade_history)]
        if addon_closes:
            addon_wins = len([t for t in addon_closes if t.pnl > 0])
            self.performance_stats['addon_success_rate'] = addon_wins / len(addon_closes)
    
    def _save_trade_data(self):
        """保存交易数据"""
        try:
            # 保存交易历史
            trade_data = [asdict(trade) for trade in self.trade_history[-100:]]  # 保存最近100条
            trade_file = os.path.join(self.data_dir, "trade_history.json")
            with open(trade_file, 'w', encoding='utf-8') as f:
                json.dump(trade_data, f, ensure_ascii=False, indent=2, default=str)
            
            # 保存性能统计
            stats_file = os.path.join(self.data_dir, "performance_stats.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.performance_stats, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存交易数据失败: {e}")
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            'win_streak': self.win_streak,
            'loss_streak': self.loss_streak,
            'trading_state': self.trading_state.value,
            'is_in_cooldown': self._is_in_cooldown(),
            'cooldown_remaining': self._get_cooldown_remaining(),
            'current_exposure': self._calculate_current_exposure(),
            'active_positions': len(self.position_states),
            'performance_stats': self.performance_stats.copy(),
            'config': asdict(self.config)
        }
    
    def _get_cooldown_remaining(self) -> Optional[int]:
        """获取冷静期剩余时间（秒）"""
        if not self.cooldown_start_time:
            return None
        
        cooldown_duration = timedelta(hours=self.config.cooldown_duration_hours)
        elapsed = datetime.now() - self.cooldown_start_time
        remaining = cooldown_duration - elapsed
        
        return max(0, int(remaining.total_seconds()))

    def _calculate_dynamic_stop_loss(self, position: PositionState, current_price: float) -> float:
        """
        计算动态止损价格（追踪止损）
        
        Args:
            position: 持仓状态
            current_price: 当前价格
            
        Returns:
            动态止损价格
        """
        # 基础止损价格
        base_stop_loss = position.stop_loss_price
        
        # 如果持仓盈利，使用追踪止损
        if current_price > position.avg_entry_price:
            # 计算盈利百分比
            profit_pct = (current_price - position.avg_entry_price) / position.avg_entry_price
            
            # 根据盈利程度调整止损
            if profit_pct >= 0.03:  # 盈利3%以上
                # 将止损上移到成本价附近
                dynamic_stop_loss = position.avg_entry_price * 1.005  # 成本价+0.5%
            elif profit_pct >= 0.02:  # 盈利2%以上
                # 将止损上移到成本价-1%
                dynamic_stop_loss = position.avg_entry_price * 0.99
            elif profit_pct >= 0.01:  # 盈利1%以上
                # 将止损上移到成本价-1.5%
                dynamic_stop_loss = position.avg_entry_price * 0.985
            else:
                # 盈利不足1%，使用基础止损
                dynamic_stop_loss = base_stop_loss
        else:
            # 持仓亏损，使用基础止损
            dynamic_stop_loss = base_stop_loss
        
        return dynamic_stop_loss
    
    def _should_reduce_position(self, position: PositionState, current_price: float, 
                               market_data: Dict = None) -> bool:
        """
        检查是否应该减少仓位
        
        Args:
            position: 持仓状态
            current_price: 当前价格
            market_data: 市场数据
            
        Returns:
            是否应该减少仓位
        """
        # 检查连败状态
        if self.loss_streak >= 2:
            return True
        
        # 检查持仓亏损程度
        if position.unrealized_pnl_pct < -0.015:  # 亏损超过1.5%
            return True
        
        # 检查市场数据（如果有）
        if market_data:
            # 检查成交量异常
            if 'volume_ratio' in market_data and market_data['volume_ratio'] > 2.0:
                return True
            
            # 检查价格动量异常
            if 'price_change_pct' in market_data and abs(market_data['price_change_pct']) > 0.05:
                return True
        
        return False

def main():
    """测试函数"""
    # 创建动态仓位管理器
    config = DynamicSizingConfig(
        base_risk_pct=0.02,
        max_exposure_pct=0.08,
        win_streak_trigger=3,
        addon_aggressive_factor=0.2
    )
    
    dps = DynamicPositionSizing(config)
    
    # 模拟交易流程
    print("=== 动态仓位管理系统测试 ===")
    
    # 第一笔交易
    base_size = dps.calculate_position_size("AAPL", 1.0)
    print(f"基础仓位大小: {base_size:.2%}")
    
    # 记录几笔盈利交易
    for i in range(4):
        trade = TradeResult(
            timestamp=datetime.now(),
            symbol="AAPL",
            action="CLOSE",
            position_id=f"pos_{i}",
            entry_price=150.0,
            exit_price=153.0,
            quantity=100,
            pnl=300.0,
            pnl_pct=0.02,
            position_size_pct=0.02
        )
        dps.record_trade_result(trade)
    
    # 检查连胜后的仓位大小
    new_size = dps.calculate_position_size("AAPL", 1.0)
    print(f"连胜后仓位大小: {new_size:.2%}")
    
    # 获取系统状态
    status = dps.get_system_status()
    print(f"系统状态: {status}")

if __name__ == "__main__":
    main()