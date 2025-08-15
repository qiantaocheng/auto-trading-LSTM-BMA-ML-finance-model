#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EOD 调度器 - 收盘后信号生成和移动止损更新
包含三个核心任务：
1. EOD 信号任务（收盘后 + 20 分钟）
2. EOD 移动止损任务（收盘后 + 20 分钟）
3. 次日开盘下单任务（开盘时/开盘后）
"""

import asyncio
import logging
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class EODConfig:
    """EOD 配置"""
    enabled: bool = True
    run_after_close_min: int = 20
    open_order_type: str = "LOO"  # LOO, MOO
    avoid_open_auction: bool = False
    limit_band_by_atr_mult: float = 0.5
    limit_band_floor_pct: float = 0.003
    limit_band_cap_pct: float = 0.015
    cancel_if_not_filled_minutes: int = 30
    atr_trailing_enabled: bool = True
    atr_trailing_period: int = 14
    atr_trailing_multiplier: float = 2.0
    atr_trailing_activate_after_R: float = 1.0

@dataclass
class OpeningPlan:
    """开盘计划"""
    symbol: str
    side: str
    quantity: int
    reference_price: float
    limit_price: float
    atr_value: float
    initial_stop: float
    entry_reason: str

class EODScheduler:
    """EOD 调度器"""
    
    def __init__(self, config: EODConfig):
        self.config = config
        self.logger = logging.getLogger("EODScheduler")
        
        # 组件依赖（外部注入）
        self.data_feed = None
        self.signal_engine = None
        self.order_executor = None
        self.position_manager = None
        
        # 状态管理
        self.pending_opening_plans: List[OpeningPlan] = []
        self.pending_closing_symbols: List[str] = []
        self.position_states: Dict[str, Dict[str, Any]] = {}  # 记录entry_price, initial_stop等
        
        # 任务管理
        from .task_lifecycle_manager import get_task_manager
        self.task_manager = get_task_manager()
        
        self.logger.info(f"EOD调度器初始化完成，配置: {config}")

    def set_dependencies(self, data_feed, signal_engine, order_executor, position_manager):
        """设置依赖组件"""
        self.data_feed = data_feed
        self.signal_engine = signal_engine
        self.order_executor = order_executor
        self.position_manager = position_manager
        self.logger.info("EOD调度器依赖组件设置完成")

    def get_market_close_time(self) -> datetime:
        """获取今日市场收盘时间（美东时间）"""
        # 简化版：假设美股16:00收盘
        today = datetime.now().date()
        return datetime.combine(today, dt_time(16, 0))

    def get_market_open_time(self) -> datetime:
        """获取次日市场开盘时间（美东时间）"""
        # 简化版：假设美股9:30开盘
        tomorrow = datetime.now().date() + timedelta(days=1)
        return datetime.combine(tomorrow, dt_time(9, 30))

    async def start_eod_tasks(self):
        """启动EOD任务调度"""
        if not self.config.enabled:
            self.logger.info("EOD模式未启用，跳过任务调度")
            return

        self.logger.info("启动EOD任务调度")
        
        # 计算EOD任务执行时间
        close_time = self.get_market_close_time()
        eod_run_time = close_time + timedelta(minutes=self.config.run_after_close_min)
        
        # 计算开盘任务执行时间
        open_time = self.get_market_open_time()
        
        # 调度EOD信号任务
        self.task_manager.create_task(
            self._schedule_eod_signal_task(eod_run_time),
            task_id="eod_signal_task",
            creator="eod_scheduler",
            description=f"EOD信号生成任务，执行时间: {eod_run_time}",
            group="eod_tasks",
            max_lifetime=3600  # 1小时
        )
        
        # 调度EOD移动止损任务
        self.task_manager.create_task(
            self._schedule_eod_trailing_task(eod_run_time),
            task_id="eod_trailing_task", 
            creator="eod_scheduler",
            description=f"EOD移动止损任务，执行时间: {eod_run_time}",
            group="eod_tasks",
            max_lifetime=3600
        )
        
        # 调度次日开盘任务
        self.task_manager.create_task(
            self._schedule_opening_task(open_time),
            task_id="opening_execution_task",
            creator="eod_scheduler", 
            description=f"次日开盘执行任务，执行时间: {open_time}",
            group="opening_tasks",
            max_lifetime=1800  # 30分钟
        )

    async def _schedule_eod_signal_task(self, run_time: datetime):
        """调度EOD信号任务"""
        # 等待到执行时间
        wait_seconds = (run_time - datetime.now()).total_seconds()
        if wait_seconds > 0:
            self.logger.info(f"等待{wait_seconds:.1f}秒执行EOD信号任务")
            await asyncio.sleep(wait_seconds)
        
        await self._execute_eod_signal_generation()

    async def _schedule_eod_trailing_task(self, run_time: datetime):
        """调度EOD移动止损任务"""
        # 等待到执行时间
        wait_seconds = (run_time - datetime.now()).total_seconds()
        if wait_seconds > 0:
            self.logger.info(f"等待{wait_seconds:.1f}秒执行EOD移动止损任务")
            await asyncio.sleep(wait_seconds)
        
        await self._execute_eod_trailing_stops()

    async def _schedule_opening_task(self, run_time: datetime):
        """调度开盘任务"""
        # 等待到执行时间
        wait_seconds = (run_time - datetime.now()).total_seconds()
        if wait_seconds > 0:
            self.logger.info(f"等待{wait_seconds:.1f}秒执行开盘任务")
            await asyncio.sleep(wait_seconds)
        
        await self._execute_opening_orders()

    async def _execute_eod_signal_generation(self):
        """执行EOD信号生成"""
        self.logger.info("开始执行EOD信号生成")
        
        try:
            if not self.data_feed or not self.signal_engine:
                self.logger.error("缺少必要组件：data_feed 或 signal_engine")
                return

            # 获取股票列表（这里需要从配置或文件读取）
            symbols = self._get_trading_universe()
            
            for symbol in symbols:
                try:
                    # 获取日线数据
                    daily_bars = await self.data_feed.fetch_daily_bars(symbol, lookback=200)
                    if daily_bars is None or len(daily_bars) < 100:
                        continue
                    
                    # 计算ATR
                    atr_value = self._calculate_atr(daily_bars, period=self.config.atr_trailing_period)
                    
                    # 运行信号引擎
                    signal = await self.signal_engine.on_daily_bar_close(symbol, daily_bars)
                    
                    if signal and signal.get('action') in ['BUY', 'SELL']:
                        # 生成开盘计划
                        plan = self._create_opening_plan(symbol, signal, daily_bars, atr_value)
                        if plan:
                            self.pending_opening_plans.append(plan)
                            self.logger.info(f"生成开盘计划: {symbol} {plan.side} {plan.quantity}股 @ {plan.limit_price}")
                    
                    elif signal and signal.get('action') == 'CLOSE':
                        # 加入平仓队列
                        self.pending_closing_symbols.append(symbol)
                        self.logger.info(f"加入平仓队列: {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"处理{symbol}信号时失败: {e}")
                    continue
            
            self.logger.info(f"EOD信号生成完成: {len(self.pending_opening_plans)}个开仓计划, {len(self.pending_closing_symbols)}个平仓计划")
            
        except Exception as e:
            self.logger.error(f"EOD信号生成失败: {e}")

    async def _execute_eod_trailing_stops(self):
        """执行EOD移动止损更新"""
        self.logger.info("开始执行EOD移动止损更新")
        
        try:
            if not self.config.atr_trailing_enabled:
                self.logger.info("ATR移动止损未启用，跳过")
                return
            
            if not self.order_executor:
                self.logger.error("缺少订单执行器")
                return

            # 获取当前持仓
            positions = await self._get_current_positions()
            
            for symbol, position_info in positions.items():
                try:
                    # 获取当前收盘价和ATR
                    daily_bars = await self.data_feed.fetch_daily_bars(symbol, lookback=60)
                    if daily_bars is None or len(daily_bars) == 0:
                        continue
                    
                    current_close = daily_bars['close'].iloc[-1]
                    atr_value = self._calculate_atr(daily_bars, period=self.config.atr_trailing_period)
                    
                    # 从position_states获取入场信息
                    if symbol not in self.position_states:
                        self.logger.warning(f"未找到{symbol}的入场信息，跳过移动止损")
                        continue
                    
                    state = self.position_states[symbol]
                    entry_price = state.get('entry_price')
                    initial_stop = state.get('initial_stop')
                    side = state.get('side')
                    
                    if not all([entry_price, initial_stop, side]):
                        self.logger.warning(f"{symbol}入场信息不完整，跳过移动止损")
                        continue
                    
                    # 更新移动止损
                    await self.order_executor.eod_update_trailing_stop(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        current_close=current_close,
                        atr_value=atr_value,
                        initial_stop=initial_stop,
                        activate_after_R=self.config.atr_trailing_activate_after_R,
                        atr_mult=self.config.atr_trailing_multiplier
                    )
                    
                except Exception as e:
                    self.logger.error(f"更新{symbol}移动止损失败: {e}")
                    continue
            
            self.logger.info(f"EOD移动止损更新完成，处理{len(positions)}个持仓")
            
        except Exception as e:
            self.logger.error(f"EOD移动止损更新失败: {e}")

    async def _execute_opening_orders(self):
        """执行开盘订单"""
        self.logger.info("开始执行开盘订单")
        
        try:
            if not self.order_executor:
                self.logger.error("缺少订单执行器")
                return

            # 处理平仓订单
            for symbol in self.pending_closing_symbols.copy():
                try:
                    await self._execute_closing_order(symbol)
                    self.pending_closing_symbols.remove(symbol)
                except Exception as e:
                    self.logger.error(f"平仓{symbol}失败: {e}")

            # 处理开仓订单
            for plan in self.pending_opening_plans.copy():
                try:
                    await self._execute_opening_plan(plan)
                    self.pending_opening_plans.remove(plan)
                except Exception as e:
                    self.logger.error(f"执行开仓计划{plan.symbol}失败: {e}")
            
            self.logger.info("开盘订单执行完成")
            
        except Exception as e:
            self.logger.error(f"开盘订单执行失败: {e}")

    def _get_trading_universe(self) -> List[str]:
        """获取交易股票池"""
        # 这里应该从配置文件或数据库读取
        # 暂时返回示例列表
        return ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

    def _calculate_atr(self, bars: pd.DataFrame, period: int = 14) -> float:
        """计算ATR"""
        try:
            high = bars['high']
            low = bars['low'] 
            close = bars['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return float(atr) if not pd.isna(atr) else 0.0
        except Exception as e:
            self.logger.error(f"计算ATR失败: {e}")
            return 0.0

    def _create_opening_plan(self, symbol: str, signal: Dict[str, Any], bars: pd.DataFrame, atr_value: float) -> Optional[OpeningPlan]:
        """创建开盘计划"""
        try:
            side = signal['action']
            reference_price = bars['close'].iloc[-1]  # 昨收
            
            # 计算价带
            band = self._calculate_price_band(reference_price, atr_value)
            
            if side == 'BUY':
                limit_price = reference_price + band
            else:
                limit_price = reference_price - band
            
            # 计算仓位大小（这里需要实现仓位管理逻辑）
            quantity = self._calculate_position_size(symbol, reference_price, signal.get('confidence', 1.0))
            
            # 计算初始止损
            initial_stop = self._calculate_initial_stop(reference_price, side)
            
            return OpeningPlan(
                symbol=symbol,
                side=side,
                quantity=quantity,
                reference_price=reference_price,
                limit_price=limit_price,
                atr_value=atr_value,
                initial_stop=initial_stop,
                entry_reason=signal.get('reason', 'EOD_Signal')
            )
            
        except Exception as e:
            self.logger.error(f"创建{symbol}开盘计划失败: {e}")
            return None

    def _calculate_price_band(self, reference_price: float, atr_value: float) -> float:
        """计算价格带宽"""
        # band = min( max(atr_mult*ATR, floor_pct*close), cap_pct*close )
        atr_band = self.config.limit_band_by_atr_mult * atr_value
        floor_band = self.config.limit_band_floor_pct * reference_price
        cap_band = self.config.limit_band_cap_pct * reference_price
        
        return min(max(atr_band, floor_band), cap_band)

    def _calculate_position_size(self, symbol: str, price: float, confidence: float) -> int:
        """计算仓位大小"""
        # 简化版：固定金额分配
        target_dollar_amount = 10000 * confidence  # 基础1万美元 * 信心度
        quantity = int(target_dollar_amount / price)
        return max(quantity, 1)  # 最小1股

    def _calculate_initial_stop(self, entry_price: float, side: str) -> float:
        """计算初始止损价（1%）"""
        stop_pct = 0.01  # 1%止损
        if side == 'BUY':
            return entry_price * (1 - stop_pct)
        else:
            return entry_price * (1 + stop_pct)

    async def _get_current_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取当前持仓"""
        # 这里应该从持仓管理器获取
        # 暂时返回空字典
        if self.position_manager:
            return await self.position_manager.get_all_positions()
        return {}

    async def _execute_opening_plan(self, plan: OpeningPlan):
        """执行开仓计划"""
        self.logger.info(f"执行开仓计划: {plan.symbol} {plan.side} {plan.quantity}股")
        
        try:
            if self.config.avoid_open_auction:
                # 开盘后用限价单
                trade = await self.order_executor.place_limit_rth(
                    symbol=plan.symbol,
                    side=plan.side,
                    quantity=plan.quantity,
                    limit_price=plan.limit_price,
                    cancel_after_min=self.config.cancel_if_not_filled_minutes
                )
            else:
                # 开盘单
                trade = await self.order_executor.place_open_order(
                    symbol=plan.symbol,
                    side=plan.side,
                    quantity=plan.quantity,
                    limit_price=plan.limit_price if self.config.open_order_type == "LOO" else None,
                    order_type=self.config.open_order_type
                )
            
            # 记录计划状态（用于后续回调）
            self.position_states[plan.symbol] = {
                'entry_price': plan.reference_price,  # 将在成交回报中更新为实际成交价
                'initial_stop': plan.initial_stop,
                'side': plan.side,
                'atr_value': plan.atr_value,
                'plan': plan
            }
            
            self.logger.info(f"开仓订单已提交: {plan.symbol} 订单ID: {getattr(trade.order, 'orderId', 'N/A')}")
            
        except Exception as e:
            self.logger.error(f"执行开仓计划失败 {plan.symbol}: {e}")
            raise

    async def _execute_closing_order(self, symbol: str):
        """执行平仓订单"""
        self.logger.info(f"执行平仓: {symbol}")
        
        # 这里需要实现平仓逻辑
        # 获取当前持仓方向和数量，然后反向下单
        
        # 清理状态
        if symbol in self.position_states:
            del self.position_states[symbol]

    def on_order_filled(self, symbol: str, fill_price: float, quantity: int, side: str):
        """订单成交回调"""
        if symbol in self.position_states:
            # 更新实际成交价
            self.position_states[symbol]['entry_price'] = fill_price
            self.logger.info(f"更新{symbol}成交价: {fill_price}")
            
            # 立即挂止损单
            initial_stop = self.position_states[symbol]['initial_stop']
            asyncio.create_task(self.order_executor.update_server_stop(symbol, initial_stop, quantity))

# 全局实例
_eod_scheduler: Optional[EODScheduler] = None

def get_eod_scheduler() -> Optional[EODScheduler]:
    """获取全局EOD调度器实例"""
    return _eod_scheduler

def create_eod_scheduler(config: EODConfig) -> EODScheduler:
    """创建EOD调度器"""
    global _eod_scheduler
    _eod_scheduler = EODScheduler(config)
    return _eod_scheduler