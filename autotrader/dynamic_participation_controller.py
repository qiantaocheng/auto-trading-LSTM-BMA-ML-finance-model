#!/usr/bin/env python3
"""
📊 P1级别修复：动态参与率控制系统
=======================================

实现基于市场微结构的动态参与率控制，包括：
- 波动率自适应参与率
- 流动性感知执行控制
- 收盘前自动停止新仓
- 市场冲击最小化
- 实时执行成本监控
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, time
from enum import Enum
import threading
import time as time_module
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市场状态"""
    QUIET = "QUIET"          # 平静市场
    NORMAL = "NORMAL"        # 正常市场  
    VOLATILE = "VOLATILE"    # 波动市场
    STRESSED = "STRESSED"    # 压力市场
    ILLIQUID = "ILLIQUID"    # 流动性不足


class ExecutionUrgency(Enum):
    """执行紧迫性"""
    LOW = "LOW"              # 低紧迫性
    NORMAL = "NORMAL"        # 正常紧迫性
    HIGH = "HIGH"            # 高紧迫性
    URGENT = "URGENT"        # 紧急执行


@dataclass
class MarketMicrostructure:
    """市场微结构数据"""
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    vwap: float
    volatility: float
    timestamp: datetime
    
    @property
    def spread(self) -> float:
        """买卖价差"""
        return self.ask_price - self.bid_price
    
    @property
    def spread_bps(self) -> float:
        """价差（基点）"""
        mid_price = (self.bid_price + self.ask_price) / 2
        return (self.spread / mid_price) * 10000 if mid_price > 0 else 0
    
    @property
    def liquidity_score(self) -> float:
        """流动性评分 (0-1)"""
        # 简化的流动性评分：基于价差和深度
        spread_penalty = max(0, min(1, (50 - self.spread_bps) / 50))
        depth_score = min(1, (self.bid_size + self.ask_size) / 10000)  # 假设10000是良好深度
        return (spread_penalty + depth_score) / 2


@dataclass
class ExecutionParameters:
    """执行参数"""
    symbol: str
    side: str  # BUY/SELL
    target_quantity: float
    max_participation_rate: float
    urgency: ExecutionUrgency
    start_time: datetime
    end_time: datetime
    twap_target: Optional[float] = None
    vwap_target: Optional[float] = None
    max_slippage_bps: float = 50  # 最大滑点50bp
    min_fill_size: float = 1.0
    
    @property
    def duration_minutes(self) -> float:
        """执行时长（分钟）"""
        return (self.end_time - self.start_time).total_seconds() / 60


class DynamicParticipationController:
    """动态参与率控制器"""
    
    def __init__(self, 
                 min_participation_rate: float = 0.01,  # 最小1%
                 max_participation_rate: float = 0.25,  # 最大25%
                 volatility_lookback_minutes: int = 30,
                 liquidity_lookback_minutes: int = 5,
                 market_close_buffer_minutes: int = 30):
        
        self.min_participation_rate = min_participation_rate
        self.max_participation_rate = max_participation_rate
        self.volatility_lookback_minutes = volatility_lookback_minutes
        self.liquidity_lookback_minutes = liquidity_lookback_minutes
        self.market_close_buffer_minutes = market_close_buffer_minutes
        
        # 市场数据缓存
        self._market_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._execution_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # 参与率限制缓存
        self._participation_limits: Dict[str, Tuple[float, datetime]] = {}
        
        # 线程锁
        self._lock = threading.RLock()
        
        logger.info("Dynamic participation controller initialized")
    
    def update_market_data(self, market_data: MarketMicrostructure):
        """更新市场数据"""
        with self._lock:
            symbol = market_data.symbol
            self._market_data[symbol].append(market_data)
            
            # 清理过期数据
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=2)
            while (self._market_data[symbol] and 
                   self._market_data[symbol][0].timestamp < cutoff_time):
                self._market_data[symbol].popleft()
    
    def get_market_regime(self, symbol: str) -> MarketRegime:
        """判断市场状态"""
        with self._lock:
            if symbol not in self._market_data or len(self._market_data[symbol]) < 10:
                return MarketRegime.NORMAL
            
            recent_data = list(self._market_data[symbol])[-30:]  # 最近30个数据点
            
            # 计算波动率指标
            prices = [d.last_price for d in recent_data]
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(252 * 24 * 60)  # 年化波动率
            
            # 计算流动性指标
            spreads = [d.spread_bps for d in recent_data]
            avg_spread = np.mean(spreads)
            liquidity_scores = [d.liquidity_score for d in recent_data]
            avg_liquidity = np.mean(liquidity_scores)
            
            # 判断市场状态
            if avg_liquidity < 0.3:
                return MarketRegime.ILLIQUID
            elif volatility > 0.4:  # 40%年化波动率
                return MarketRegime.STRESSED
            elif volatility > 0.25:  # 25%年化波动率
                return MarketRegime.VOLATILE
            elif avg_spread < 5 and avg_liquidity > 0.7:  # 价差<5bp且流动性好
                return MarketRegime.QUIET
            else:
                return MarketRegime.NORMAL
    
    def calculate_optimal_participation_rate(self, 
                                           symbol: str,
                                           execution_params: ExecutionParameters) -> float:
        """计算最优参与率"""
        with self._lock:
            # 基础参与率（基于紧迫性）
            base_rates = {
                ExecutionUrgency.LOW: 0.05,      # 5%
                ExecutionUrgency.NORMAL: 0.10,   # 10%
                ExecutionUrgency.HIGH: 0.15,     # 15%
                ExecutionUrgency.URGENT: 0.25    # 25%
            }
            base_rate = base_rates.get(execution_params.urgency, 0.10)
            
            # 市场状态调整
            market_regime = self.get_market_regime(symbol)
            regime_adjustments = {
                MarketRegime.QUIET: 1.2,      # 安静市场可以更积极
                MarketRegime.NORMAL: 1.0,     # 正常市场无调整
                MarketRegime.VOLATILE: 0.7,   # 波动市场更保守
                MarketRegime.STRESSED: 0.5,   # 压力市场非常保守
                MarketRegime.ILLIQUID: 0.3    # 流动性不足时极度保守
            }
            adjusted_rate = base_rate * regime_adjustments.get(market_regime, 1.0)
            
            # 时间紧迫性调整
            remaining_minutes = (execution_params.end_time - datetime.now(timezone.utc)).total_seconds() / 60
            if remaining_minutes > 0:
                time_pressure = max(0.5, min(2.0, 60 / remaining_minutes))  # 时间压力因子
                adjusted_rate *= time_pressure
            
            # 波动率调整
            volatility_adjustment = self._get_volatility_adjustment(symbol)
            adjusted_rate *= volatility_adjustment
            
            # 应用限制
            final_rate = max(
                self.min_participation_rate,
                min(self.max_participation_rate, 
                    min(execution_params.max_participation_rate, adjusted_rate))
            )
            
            logger.debug(f"Participation rate for {symbol}: {final_rate:.3f} "
                        f"(base: {base_rate:.3f}, regime: {market_regime.value}, "
                        f"vol_adj: {volatility_adjustment:.3f})")
            
            return final_rate
    
    def _get_volatility_adjustment(self, symbol: str) -> float:
        """获取波动率调整系数"""
        if symbol not in self._market_data or len(self._market_data[symbol]) < 20:
            return 1.0
        
        # 计算短期波动率
        recent_data = list(self._market_data[symbol])[-20:]
        prices = [d.last_price for d in recent_data]
        returns = np.diff(np.log(prices))
        short_vol = np.std(returns)
        
        # 计算长期波动率
        longer_data = list(self._market_data[symbol])[-60:] if len(self._market_data[symbol]) >= 60 else recent_data
        prices_long = [d.last_price for d in longer_data]
        returns_long = np.diff(np.log(prices_long))
        long_vol = np.std(returns_long)
        
        if long_vol == 0:
            return 1.0
        
        # 波动率比率
        vol_ratio = short_vol / long_vol
        
        # 调整系数：波动率高时降低参与率
        if vol_ratio > 2.0:
            return 0.5  # 波动率过高，减半
        elif vol_ratio > 1.5:
            return 0.7  # 波动率较高，减少30%
        elif vol_ratio < 0.5:
            return 1.3  # 波动率很低，可以增加30%
        else:
            return 1.0  # 正常波动率
    
    def should_allow_new_positions(self, market: str = "NYSE") -> bool:
        """检查是否允许开新仓（考虑收盘时间）"""
        try:
            # 获取市场收盘时间（简化处理，假设NYSE 16:00 ET收盘）
            now = datetime.now(timezone.utc)
            
            # 转换为美东时间（简化处理）
            et_offset = timedelta(hours=-5)  # 标准时间偏移，忽略夏令时复杂性
            now_et = now + et_offset
            
            # 市场收盘时间
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            
            # 如果已经过了收盘时间，检查是否为第二天
            if now_et.time() < time(16, 0):
                # 今天还未收盘
                time_to_close = (market_close - now_et).total_seconds() / 60
            else:
                # 今天已收盘，计算到明天收盘的时间
                tomorrow_close = market_close + timedelta(days=1)
                time_to_close = (tomorrow_close - now_et).total_seconds() / 60
            
            # 收盘前30分钟（默认）不允许开新仓
            allow_new_positions = time_to_close > self.market_close_buffer_minutes
            
            if not allow_new_positions:
                logger.info(f"New positions blocked - {time_to_close:.1f} minutes to market close")
            
            return allow_new_positions
            
        except Exception as e:
            logger.error(f"Failed to check market close time: {e}")
            return True  # 出错时允许交易，避免阻塞
    
    def get_execution_slice_size(self, 
                                symbol: str,
                                execution_params: ExecutionParameters,
                                current_volume: float) -> float:
        """计算当前执行切片大小"""
        participation_rate = self.calculate_optimal_participation_rate(symbol, execution_params)
        
        # 基于当前成交量计算切片大小
        slice_size = current_volume * participation_rate
        
        # 应用最小/最大限制
        slice_size = max(execution_params.min_fill_size, slice_size)
        slice_size = min(execution_params.target_quantity, slice_size)
        
        return slice_size
    
    def estimate_market_impact(self, 
                              symbol: str,
                              quantity: float,
                              side: str) -> Dict[str, float]:
        """估算市场冲击"""
        with self._lock:
            if symbol not in self._market_data or not self._market_data[symbol]:
                return {'temporary_impact_bps': 0, 'permanent_impact_bps': 0}
            
            latest_data = self._market_data[symbol][-1]
            
            # 简化的市场冲击模型
            # 临时冲击 = f(订单大小/平均成交量, 价差)
            avg_volume = np.mean([d.volume for d in list(self._market_data[symbol])[-10:]])
            volume_ratio = quantity / max(avg_volume, 1)
            
            # 基础冲击（基点）
            base_impact = min(50, volume_ratio * 100)  # 最大50bp
            
            # 价差调整
            spread_adjustment = latest_data.spread_bps / 10  # 价差越大冲击越大
            
            temporary_impact = base_impact + spread_adjustment
            permanent_impact = temporary_impact * 0.3  # 永久冲击约为临时冲击的30%
            
            return {
                'temporary_impact_bps': temporary_impact,
                'permanent_impact_bps': permanent_impact,
                'total_cost_bps': temporary_impact + permanent_impact
            }
    
    def get_execution_schedule(self, 
                              execution_params: ExecutionParameters,
                              time_interval_minutes: int = 1) -> List[Dict[str, Any]]:
        """生成执行时间表"""
        schedule = []
        
        current_time = execution_params.start_time
        remaining_quantity = execution_params.target_quantity
        
        while current_time < execution_params.end_time and remaining_quantity > 0:
            # 计算这个时间片的目标数量
            time_remaining = (execution_params.end_time - current_time).total_seconds() / 60
            
            if time_remaining <= time_interval_minutes:
                # 最后一个时间片，执行剩余全部数量
                slice_quantity = remaining_quantity
            else:
                # 基于剩余时间和紧迫性分配数量
                if execution_params.urgency == ExecutionUrgency.URGENT:
                    # 紧急情况下前置更多数量
                    slice_quantity = remaining_quantity * (time_interval_minutes / time_remaining) * 1.5
                else:
                    # 正常情况下均匀分布
                    slice_quantity = remaining_quantity * (time_interval_minutes / time_remaining)
            
            slice_quantity = min(slice_quantity, remaining_quantity)
            
            schedule.append({
                'start_time': current_time,
                'end_time': min(current_time + timedelta(minutes=time_interval_minutes), 
                               execution_params.end_time),
                'target_quantity': slice_quantity,
                'cumulative_executed': execution_params.target_quantity - remaining_quantity
            })
            
            remaining_quantity -= slice_quantity
            current_time += timedelta(minutes=time_interval_minutes)
        
        return schedule
    
    def should_pause_execution(self, symbol: str) -> Tuple[bool, str]:
        """检查是否应该暂停执行"""
        market_regime = self.get_market_regime(symbol)
        
        # 在压力或流动性不足的市场中暂停
        if market_regime == MarketRegime.STRESSED:
            return True, "Market in stressed condition"
        
        if market_regime == MarketRegime.ILLIQUID:
            return True, "Insufficient liquidity"
        
        # 检查市场收盘时间
        if not self.should_allow_new_positions():
            return True, "Too close to market close"
        
        return False, ""
    
    def get_participation_analytics(self, symbol: str) -> Dict[str, Any]:
        """获取参与率分析数据"""
        with self._lock:
            if symbol not in self._market_data:
                return {}
            
            recent_data = list(self._market_data[symbol])[-100:]  # 最近100个数据点
            
            if not recent_data:
                return {}
            
            analytics = {
                'symbol': symbol,
                'data_points': len(recent_data),
                'market_regime': self.get_market_regime(symbol).value,
                'current_spread_bps': recent_data[-1].spread_bps if recent_data else 0,
                'avg_liquidity_score': np.mean([d.liquidity_score for d in recent_data]),
                'volatility_adjustment': self._get_volatility_adjustment(symbol),
                'recommended_base_participation': self.calculate_optimal_participation_rate(
                    symbol, 
                    ExecutionParameters(
                        symbol=symbol, side='BUY', target_quantity=1000,
                        max_participation_rate=0.2, urgency=ExecutionUrgency.NORMAL,
                        start_time=datetime.now(timezone.utc),
                        end_time=datetime.now(timezone.utc) + timedelta(hours=1)
                    )
                ),
                'market_close_buffer_active': not self.should_allow_new_positions(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return analytics


# 全局实例
_global_participation_controller: Optional[DynamicParticipationController] = None


def get_participation_controller() -> DynamicParticipationController:
    """获取全局动态参与率控制器"""
    global _global_participation_controller
    if _global_participation_controller is None:
        _global_participation_controller = DynamicParticipationController()
    return _global_participation_controller


if __name__ == "__main__":
    # 测试动态参与率控制
    logging.basicConfig(level=logging.INFO)
    
    controller = DynamicParticipationController()
    
    # 模拟市场数据
    market_data = MarketMicrostructure(
        symbol="AAPL",
        bid_price=150.0,
        ask_price=150.1,
        bid_size=1000,
        ask_size=1200,
        last_price=150.05,
        volume=50000,
        vwap=150.02,
        volatility=0.15,
        timestamp=datetime.now(timezone.utc)
    )
    
    controller.update_market_data(market_data)
    
    # 创建执行参数
    exec_params = ExecutionParameters(
        symbol="AAPL",
        side="BUY",
        target_quantity=10000,
        max_participation_rate=0.15,
        urgency=ExecutionUrgency.NORMAL,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc) + timedelta(hours=1)
    )
    
    # 测试参与率计算
    participation_rate = controller.calculate_optimal_participation_rate("AAPL", exec_params)
    print(f"Optimal participation rate: {participation_rate:.3f}")
    
    # 测试市场冲击估算
    impact = controller.estimate_market_impact("AAPL", 5000, "BUY")
    print(f"Market impact: {impact}")
    
    # 测试执行时间表
    schedule = controller.get_execution_schedule(exec_params, time_interval_minutes=5)
    print(f"Execution schedule ({len(schedule)} slices):")
    for i, slice_info in enumerate(schedule[:3]):  # 显示前3个切片
        print(f"  Slice {i+1}: {slice_info['target_quantity']:.0f} shares at {slice_info['start_time'].strftime('%H:%M')}")
    
    # 测试分析数据
    analytics = controller.get_participation_analytics("AAPL")
    print(f"Analytics: {analytics}")