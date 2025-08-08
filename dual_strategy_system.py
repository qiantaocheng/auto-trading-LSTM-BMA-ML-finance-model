#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双策略并行管理系统 (DualStrategySystem)
实现A/B双策略并行管理，支持动态权重分配

核心功能:
1. 并行管理均值回归策略(A)和趋势跟踪策略(B)
2. 动态权重分配和资金管理
3. 策略间风险隔离和独立执行
4. 实时监控和状态同步
5. 统一的订单管理和持仓跟踪

Authors: AI Assistant
Version: 1.0
"""

import pandas as pd
import numpy as np
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# 导入策略基础类
try:
    from ibkr_trading_strategy_enhanced import EnhancedMeanReversionStrategy
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False

# 导入配比管理器
from allocation_manager import AllocationManager, AllocationConfig
from regime_detector import RegimeDetector

class StrategyType(Enum):
    """策略类型枚举"""
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"

class StrategyStatus(Enum):
    """策略状态枚举"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class StrategyState:
    """策略状态数据类"""
    strategy_id: str
    strategy_type: StrategyType
    status: StrategyStatus
    allocation: float
    capital_allocated: float
    positions: Dict[str, Any]
    pending_orders: List[Dict]
    daily_pnl: float
    total_pnl: float
    trade_count: int
    last_update: str
    error_message: str = ""

@dataclass
class SystemState:
    """系统状态数据类"""
    timestamp: str
    total_capital: float
    allocated_capital: float
    available_capital: float
    total_pnl: float
    daily_pnl: float
    strategy_states: Dict[str, StrategyState]
    regime_info: Dict
    allocation_config: AllocationConfig
    cooldown_active: bool
    system_status: str

class BaseStrategy:
    """策略基础类"""
    
    def __init__(self, strategy_id: str, strategy_type: StrategyType, config: Dict):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.config = config
        self.status = StrategyStatus.STOPPED
        self.logger = logging.getLogger(f"{__name__}.{strategy_id}")
        
        # 状态变量
        self.allocation = 0.0
        self.capital_allocated = 0.0
        self.positions = {}
        self.pending_orders = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.last_trade_time = None
        self.error_message = ""
        
        # 控制变量
        self.running = False
        self.paused = False
        
    async def start(self):
        """启动策略"""
        self.logger.info(f"启动策略 {self.strategy_id}")
        self.status = StrategyStatus.STARTING
        try:
            await self._initialize()
            self.running = True
            self.status = StrategyStatus.RUNNING
            self.logger.info(f"策略 {self.strategy_id} 已启动")
        except Exception as e:
            self.status = StrategyStatus.ERROR
            self.error_message = str(e)
            self.logger.error(f"策略 {self.strategy_id} 启动失败: {e}")
            raise
    
    async def stop(self):
        """停止策略"""
        self.logger.info(f"停止策略 {self.strategy_id}")
        self.running = False
        try:
            await self._cleanup()
            self.status = StrategyStatus.STOPPED
            self.logger.info(f"策略 {self.strategy_id} 已停止")
        except Exception as e:
            self.logger.error(f"策略 {self.strategy_id} 停止失败: {e}")
    
    async def pause(self):
        """暂停策略"""
        self.paused = True
        self.status = StrategyStatus.PAUSED
        self.logger.info(f"策略 {self.strategy_id} 已暂停")
    
    async def resume(self):
        """恢复策略"""
        self.paused = False
        self.status = StrategyStatus.RUNNING
        self.logger.info(f"策略 {self.strategy_id} 已恢复")
    
    def update_allocation(self, allocation: float, total_capital: float):
        """更新资金配比"""
        self.allocation = allocation
        self.capital_allocated = total_capital * allocation
        self.logger.info(f"策略 {self.strategy_id} 配比更新: {allocation:.1%} (${self.capital_allocated:,.2f})")
    
    def get_state(self) -> StrategyState:
        """获取策略状态"""
        return StrategyState(
            strategy_id=self.strategy_id,
            strategy_type=self.strategy_type,
            status=self.status,
            allocation=self.allocation,
            capital_allocated=self.capital_allocated,
            positions=self.positions.copy(),
            pending_orders=self.pending_orders.copy(),
            daily_pnl=self.daily_pnl,
            total_pnl=self.total_pnl,
            trade_count=self.trade_count,
            last_update=datetime.now().isoformat(),
            error_message=self.error_message
        )
    
    async def _initialize(self):
        """策略初始化（子类实现）"""
        pass
    
    async def _cleanup(self):
        """策略清理（子类实现）"""
        pass
    
    async def execute_trading_logic(self):
        """执行交易逻辑（子类实现）"""
        pass

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def __init__(self, strategy_id: str, config: Dict):
        super().__init__(strategy_id, StrategyType.MEAN_REVERSION, config)
        self.enhanced_strategy = None
    
    async def _initialize(self):
        """初始化均值回归策略"""
        if IBKR_AVAILABLE:
            # 创建增强版策略实例
            strategy_config = {
                **self.config,
                'enable_enhanced_mode': True,
                'strategy_type': 'mean_reversion'
            }
            self.enhanced_strategy = EnhancedMeanReversionStrategy(strategy_config)
            self.logger.info("均值回归策略初始化完成")
        else:
            self.logger.warning("IBKR不可用，使用模拟模式")
    
    async def execute_trading_logic(self):
        """执行均值回归交易逻辑"""
        if self.paused or not self.running:
            return
        
        try:
            if self.enhanced_strategy:
                # 执行增强版策略逻辑
                await self._execute_enhanced_logic()
            else:
                # 执行模拟逻辑
                await self._execute_simulation_logic()
                
        except Exception as e:
            self.logger.error(f"均值回归策略执行失败: {e}")
            self.error_message = str(e)
    
    async def _execute_enhanced_logic(self):
        """执行增强版策略逻辑"""
        try:
            # 获取策略状态
            status = self.enhanced_strategy.get_enhanced_status()
            
            # 更新策略状态
            self.positions = status.get('active_positions', {})
            self.pending_orders = status.get('pending_orders', [])
            
            # 计算PnL
            self._update_pnl()
            
        except Exception as e:
            self.logger.error(f"增强版策略逻辑执行失败: {e}")
    
    async def _execute_simulation_logic(self):
        """执行模拟策略逻辑"""
        # 模拟交易逻辑
        await asyncio.sleep(1)  # 模拟处理时间
        
        # 模拟生成一些随机收益
        random_return = np.random.normal(0.001, 0.01)  # 均值0.1%，标准差1%
        simulated_pnl = self.capital_allocated * random_return
        
        self.daily_pnl += simulated_pnl
        self.total_pnl += simulated_pnl
        
        if abs(simulated_pnl) > 0:
            self.trade_count += 1
    
    def _update_pnl(self):
        """更新PnL"""
        # 从持仓计算PnL
        total_position_pnl = 0
        for symbol, position in self.positions.items():
            if isinstance(position, dict):
                total_position_pnl += position.get('unrealized_pnl', 0)
        
        # 更新日内PnL（需要保存昨日总PnL进行计算）
        # 这里简化处理
        self.daily_pnl = total_position_pnl * 0.1  # 假设日内PnL是总PnL的10%
        self.total_pnl = total_position_pnl

class TrendFollowingStrategy(BaseStrategy):
    """趋势跟踪策略"""
    
    def __init__(self, strategy_id: str, config: Dict):
        super().__init__(strategy_id, StrategyType.TREND_FOLLOWING, config)
        self.enhanced_strategy = None
    
    async def _initialize(self):
        """初始化趋势跟踪策略"""
        if IBKR_AVAILABLE:
            # 创建增强版策略实例（配置为趋势跟踪模式）
            strategy_config = {
                **self.config,
                'enable_enhanced_mode': True,
                'strategy_type': 'trend_following',
                'adx_trend_threshold': 20,  # 更低的ADX阈值以捕捉趋势
                'momentum_lookback': 20     # 动量回看期
            }
            self.enhanced_strategy = EnhancedMeanReversionStrategy(strategy_config)
            self.logger.info("趋势跟踪策略初始化完成")
        else:
            self.logger.warning("IBKR不可用，使用模拟模式")
    
    async def execute_trading_logic(self):
        """执行趋势跟踪交易逻辑"""
        if self.paused or not self.running:
            return
        
        try:
            if self.enhanced_strategy:
                # 执行增强版策略逻辑
                await self._execute_enhanced_logic()
            else:
                # 执行模拟逻辑
                await self._execute_simulation_logic()
                
        except Exception as e:
            self.logger.error(f"趋势跟踪策略执行失败: {e}")
            self.error_message = str(e)
    
    async def _execute_enhanced_logic(self):
        """执行增强版策略逻辑"""
        try:
            # 获取策略状态
            status = self.enhanced_strategy.get_enhanced_status()
            
            # 更新策略状态
            self.positions = status.get('active_positions', {})
            self.pending_orders = status.get('pending_orders', [])
            
            # 计算PnL
            self._update_pnl()
            
        except Exception as e:
            self.logger.error(f"增强版策略逻辑执行失败: {e}")
    
    async def _execute_simulation_logic(self):
        """执行模拟策略逻辑"""
        # 模拟交易逻辑
        await asyncio.sleep(1)  # 模拟处理时间
        
        # 模拟生成一些随机收益（趋势跟踪通常波动更大）
        random_return = np.random.normal(0.002, 0.015)  # 均值0.2%，标准差1.5%
        simulated_pnl = self.capital_allocated * random_return
        
        self.daily_pnl += simulated_pnl
        self.total_pnl += simulated_pnl
        
        if abs(simulated_pnl) > 0:
            self.trade_count += 1
    
    def _update_pnl(self):
        """更新PnL"""
        # 从持仓计算PnL
        total_position_pnl = 0
        for symbol, position in self.positions.items():
            if isinstance(position, dict):
                total_position_pnl += position.get('unrealized_pnl', 0)
        
        # 更新日内PnL
        self.daily_pnl = total_position_pnl * 0.1
        self.total_pnl = total_position_pnl

class DualStrategySystem:
    """双策略并行管理系统"""
    
    def __init__(self, config: Dict):
        """
        初始化双策略系统
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 组件初始化
        self.regime_detector = RegimeDetector()
        self.allocation_manager = AllocationManager()
        
        # 策略实例
        self.strategies: Dict[str, BaseStrategy] = {}
        
        # 系统状态
        self.total_capital = config.get('total_capital', 100000.0)
        self.system_running = False
        self.last_rebalance_time = None
        
        # 创建双策略
        self._create_strategies()
        
        # 异步任务管理
        self.running_tasks = []
        
    def _create_strategies(self):
        """创建策略实例"""
        try:
            # 创建均值回归策略
            mean_reversion_id = "strategy_A_mean_reversion"
            self.strategies[mean_reversion_id] = MeanReversionStrategy(
                mean_reversion_id, 
                self.config
            )
            
            # 创建趋势跟踪策略
            trend_following_id = "strategy_B_trend_following"
            self.strategies[trend_following_id] = TrendFollowingStrategy(
                trend_following_id,
                self.config
            )
            
            self.logger.info("双策略实例创建完成")
            
        except Exception as e:
            self.logger.error(f"创建策略实例失败: {e}")
            raise
    
    async def start_system(self):
        """启动双策略系统"""
        try:
            self.logger.info("🚀 启动双策略并行管理系统")
            
            # 1. 检测市况并计算配比
            await self._update_allocation()
            
            # 2. 启动所有策略
            for strategy in self.strategies.values():
                await strategy.start()
            
            # 3. 启动系统主循环
            self.system_running = True
            
            # 创建并启动异步任务
            self.running_tasks = [
                asyncio.create_task(self._main_loop()),
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self._rebalancing_loop())
            ]
            
            self.logger.info("✅ 双策略系统启动完成")
            
        except Exception as e:
            self.logger.error(f"启动双策略系统失败: {e}")
            await self.stop_system()
            raise
    
    async def stop_system(self):
        """停止双策略系统"""
        try:
            self.logger.info("🛑 停止双策略系统")
            
            # 停止系统运行标志
            self.system_running = False
            
            # 取消所有异步任务
            for task in self.running_tasks:
                if not task.done():
                    task.cancel()
            
            # 等待任务完成
            if self.running_tasks:
                await asyncio.gather(*self.running_tasks, return_exceptions=True)
            
            # 停止所有策略
            for strategy in self.strategies.values():
                await strategy.stop()
            
            self.logger.info("✅ 双策略系统已停止")
            
        except Exception as e:
            self.logger.error(f"停止双策略系统失败: {e}")
    
    async def _main_loop(self):
        """系统主循环"""
        self.logger.info("启动系统主循环")
        
        while self.system_running:
            try:
                # 执行所有策略的交易逻辑
                tasks = []
                for strategy in self.strategies.values():
                    if strategy.status == StrategyStatus.RUNNING:
                        tasks.append(asyncio.create_task(strategy.execute_trading_logic()))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # 主循环间隔
                await asyncio.sleep(30)  # 30秒执行一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"系统主循环错误: {e}")
                await asyncio.sleep(60)  # 错误后等待更长时间
    
    async def _monitoring_loop(self):
        """监控循环"""
        self.logger.info("启动监控循环")
        
        while self.system_running:
            try:
                # 检查策略状态
                await self._check_strategy_health()
                
                # 检查风险控制
                await self._check_risk_controls()
                
                # 保存系统状态
                await self._save_system_state()
                
                # 监控循环间隔
                await asyncio.sleep(60)  # 1分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(120)
    
    async def _rebalancing_loop(self):
        """再平衡循环"""
        self.logger.info("启动再平衡循环")
        
        while self.system_running:
            try:
                # 检查是否需要再平衡
                if await self._should_rebalance():
                    await self._execute_rebalancing()
                
                # 再平衡循环间隔
                await asyncio.sleep(1800)  # 30分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"再平衡循环错误: {e}")
                await asyncio.sleep(3600)  # 错误后等待1小时
    
    async def _update_allocation(self):
        """更新资金配比"""
        try:
            # 1. 检测市况
            regime_result = self.regime_detector.detect_regime()
            
            # 2. 计算策略表现（如果有历史数据）
            strategy_performances = await self._calculate_strategy_performances()
            
            # 3. 生成配比配置
            allocation_config = self.allocation_manager.generate_allocation_config(
                regime_result, strategy_performances
            )
            
            # 4. 应用配比到策略
            await self._apply_allocation(allocation_config)
            
            # 5. 保存配置
            self.allocation_manager.save_allocation_config(allocation_config)
            
            self.logger.info(f"配比更新完成: A={allocation_config.final_allocation_A:.1%}, B={allocation_config.final_allocation_B:.1%}")
            
        except Exception as e:
            self.logger.error(f"更新配比失败: {e}")
    
    async def _calculate_strategy_performances(self):
        """计算策略表现"""
        # 这里应该从交易历史计算实际表现
        # 现在返回空字典，使用默认权重
        return {}
    
    async def _apply_allocation(self, allocation_config: AllocationConfig):
        """应用配比到策略"""
        try:
            strategy_list = list(self.strategies.values())
            
            if len(strategy_list) >= 2:
                # 策略A（均值回归）
                strategy_a = strategy_list[0]
                strategy_a.update_allocation(allocation_config.final_allocation_A, self.total_capital)
                
                # 策略B（趋势跟踪）
                strategy_b = strategy_list[1]
                strategy_b.update_allocation(allocation_config.final_allocation_B, self.total_capital)
                
                self.logger.info("配比应用完成")
            
        except Exception as e:
            self.logger.error(f"应用配比失败: {e}")
    
    async def _should_rebalance(self) -> bool:
        """检查是否需要再平衡"""
        now = datetime.now()
        
        # 每周五收盘后再平衡
        if now.weekday() == 4 and now.hour >= 16:  # 周五下午4点后
            if not self.last_rebalance_time or \
               (now - self.last_rebalance_time).days >= 1:
                return True
        
        # 配比偏离过大时再平衡
        current_allocation = await self._calculate_current_allocation()
        target_allocation = self.allocation_manager.get_current_allocation()
        
        if current_allocation and target_allocation:
            allocation_drift = abs(current_allocation[0] - target_allocation[0])
            if allocation_drift > 0.1:  # 偏离超过10%
                return True
        
        return False
    
    async def _calculate_current_allocation(self) -> Optional[Tuple[float, float]]:
        """计算当前实际配比"""
        try:
            total_allocated = 0
            strategy_allocations = []
            
            for strategy in self.strategies.values():
                total_allocated += strategy.capital_allocated
                strategy_allocations.append(strategy.capital_allocated)
            
            if total_allocated > 0:
                return tuple(alloc / total_allocated for alloc in strategy_allocations)
            
        except Exception as e:
            self.logger.error(f"计算当前配比失败: {e}")
        
        return None
    
    async def _execute_rebalancing(self):
        """执行再平衡"""
        try:
            self.logger.info("开始执行再平衡")
            
            # 1. 重新计算配比
            await self._update_allocation()
            
            # 2. 记录再平衡时间
            self.last_rebalance_time = datetime.now()
            
            self.logger.info("再平衡执行完成")
            
        except Exception as e:
            self.logger.error(f"执行再平衡失败: {e}")
    
    async def _check_strategy_health(self):
        """检查策略健康状态"""
        for strategy in self.strategies.values():
            if strategy.status == StrategyStatus.ERROR:
                self.logger.warning(f"策略 {strategy.strategy_id} 处于错误状态: {strategy.error_message}")
                # 可以在这里实现自动重启逻辑
    
    async def _check_risk_controls(self):
        """检查风险控制"""
        # 计算总PnL
        total_daily_pnl = sum(strategy.daily_pnl for strategy in self.strategies.values())
        total_pnl = sum(strategy.total_pnl for strategy in self.strategies.values())
        
        # 检查日内亏损限制
        daily_loss_limit = self.total_capital * 0.02  # 2%
        if total_daily_pnl <= -daily_loss_limit:
            self.logger.warning(f"触发日内亏损限制: {total_daily_pnl:.2f}")
            await self._trigger_cooldown("日内亏损超限")
        
        # 检查总回撤限制
        max_drawdown_limit = self.total_capital * 0.10  # 10%
        if total_pnl <= -max_drawdown_limit:
            self.logger.warning(f"触发总回撤限制: {total_pnl:.2f}")
            await self._trigger_cooldown("总回撤超限")
    
    async def _trigger_cooldown(self, reason: str):
        """触发冷静期"""
        self.logger.warning(f"触发冷静期: {reason}")
        
        # 暂停所有策略
        for strategy in self.strategies.values():
            await strategy.pause()
        
        # 更新配比管理器
        # 这里可以实现更复杂的冷静期逻辑
    
    async def _save_system_state(self):
        """保存系统状态"""
        try:
            # 计算系统级别的统计数据
            total_pnl = sum(strategy.total_pnl for strategy in self.strategies.values())
            daily_pnl = sum(strategy.daily_pnl for strategy in self.strategies.values())
            allocated_capital = sum(strategy.capital_allocated for strategy in self.strategies.values())
            
            # 获取策略状态
            strategy_states = {
                strategy_id: strategy.get_state() 
                for strategy_id, strategy in self.strategies.items()
            }
            
            # 构建系统状态
            system_state = SystemState(
                timestamp=datetime.now().isoformat(),
                total_capital=self.total_capital,
                allocated_capital=allocated_capital,
                available_capital=self.total_capital - allocated_capital,
                total_pnl=total_pnl,
                daily_pnl=daily_pnl,
                strategy_states=strategy_states,
                regime_info=self.regime_detector.current_regime or {},
                allocation_config=self.allocation_manager.current_config,
                cooldown_active=any(s.status == StrategyStatus.PAUSED for s in self.strategies.values()),
                system_status="running" if self.system_running else "stopped"
            )
            
            # 保存到文件
            state_file = "dual_strategy_system_state.json"
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(system_state), f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            self.logger.error(f"保存系统状态失败: {e}")
    
    def get_system_status(self) -> Dict:
        """获取系统状态摘要"""
        try:
            total_pnl = sum(strategy.total_pnl for strategy in self.strategies.values())
            daily_pnl = sum(strategy.daily_pnl for strategy in self.strategies.values())
            
            return {
                'system_running': self.system_running,
                'total_capital': self.total_capital,
                'total_pnl': total_pnl,
                'daily_pnl': daily_pnl,
                'strategy_count': len(self.strategies),
                'active_strategies': sum(1 for s in self.strategies.values() if s.status == StrategyStatus.RUNNING),
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取系统状态失败: {e}")
            return {'error': str(e)}

async def main():
    """测试双策略系统"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 配置
    config = {
        'total_capital': 100000.0,
        'enable_enhanced_mode': True,
        'enable_real_trading': False,  # 测试模式
        'ibkr_host': '127.0.0.1',
        'ibkr_port': 4002,
        'ibkr_client_id': 50310
    }
    
    # 创建双策略系统
    dual_system = DualStrategySystem(config)
    
    try:
        # 启动系统
        await dual_system.start_system()
        
        # 运行一段时间
        print("🔄 双策略系统运行中...")
        for i in range(10):
            await asyncio.sleep(10)
            status = dual_system.get_system_status()
            print(f"状态更新 {i+1}: PnL=${status.get('total_pnl', 0):.2f}, 日内PnL=${status.get('daily_pnl', 0):.2f}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    finally:
        # 停止系统
        await dual_system.stop_system()
        print("✅ 双策略系统测试完成")

if __name__ == "__main__":
    asyncio.run(main())