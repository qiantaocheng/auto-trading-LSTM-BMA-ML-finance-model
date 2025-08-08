#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
再平衡调度器 (RebalancingScheduler)
实现定期再平衡和冷静期管理

核心功能:
1. 周五收盘后再平衡任务
2. 盘前市况检测和配比调整
3. 日内亏损和回撤监控
4. 冷静期触发和管理
5. 动态加仓检测和执行

Authors: AI Assistant
Version: 1.0
"""

import asyncio
import schedule
import time
import threading
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Callable, Any
import logging
import json
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

# 导入相关模块
from regime_detector import RegimeDetector
from allocation_manager import AllocationManager, AllocationConfig, StrategyPerformance
from dual_strategy_system import DualStrategySystem

class TaskType(Enum):
    """任务类型枚举"""
    PRE_MARKET_REGIME_CHECK = "pre_market_regime_check"
    WEEKLY_REBALANCING = "weekly_rebalancing"
    MONTHLY_PERFORMANCE_REVIEW = "monthly_performance_review"
    DAILY_RISK_CHECK = "daily_risk_check"
    COOLDOWN_MONITOR = "cooldown_monitor"
    DYNAMIC_ALLOCATION_CHECK = "dynamic_allocation_check"

class ScheduleStatus(Enum):
    """调度状态枚举"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class ScheduledTask:
    """调度任务数据类"""
    task_id: str
    task_type: TaskType
    schedule_time: str
    last_run: Optional[str]
    next_run: Optional[str]
    run_count: int
    status: str
    error_message: str = ""

@dataclass
class RebalancingEvent:
    """再平衡事件数据类"""
    timestamp: str
    trigger_reason: str
    old_allocation: Dict[str, float]
    new_allocation: Dict[str, float]
    regime_info: Dict
    performance_data: Dict
    trades_executed: List[Dict]
    success: bool
    error_message: str = ""

class RebalancingScheduler:
    """再平衡调度器"""
    
    def __init__(self, 
                 dual_system: DualStrategySystem,
                 config: Dict = None):
        """
        初始化再平衡调度器
        
        Args:
            dual_system: 双策略系统实例
            config: 调度配置
        """
        self.dual_system = dual_system
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 组件初始化
        self.regime_detector = RegimeDetector()
        self.allocation_manager = AllocationManager()
        
        # 调度器状态
        self.status = ScheduleStatus.STOPPED
        self.scheduler_thread = None
        self.running = False
        
        # 任务记录
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.rebalancing_history: List[RebalancingEvent] = []
        
        # 冷静期状态
        self.cooldown_active = False
        self.cooldown_start_time = None
        self.cooldown_reason = ""
        self.cooldown_duration = timedelta(days=self.config.get('cooldown_days', 5))
        
        # 历史记录文件
        self.history_dir = "rebalancing_history"
        os.makedirs(self.history_dir, exist_ok=True)
        
        # 设置调度任务
        self._setup_scheduled_tasks()
    
    def _setup_scheduled_tasks(self):
        """设置调度任务"""
        try:
            # 清空现有调度
            schedule.clear()
            
            # 1. 盘前市况检测 (周一到周五 8:30)
            schedule.every().monday.at("08:30").do(self._run_task_safe, 
                                                   TaskType.PRE_MARKET_REGIME_CHECK)
            schedule.every().tuesday.at("08:30").do(self._run_task_safe, 
                                                    TaskType.PRE_MARKET_REGIME_CHECK)
            schedule.every().wednesday.at("08:30").do(self._run_task_safe, 
                                                      TaskType.PRE_MARKET_REGIME_CHECK)
            schedule.every().thursday.at("08:30").do(self._run_task_safe, 
                                                     TaskType.PRE_MARKET_REGIME_CHECK)
            schedule.every().friday.at("08:30").do(self._run_task_safe, 
                                                   TaskType.PRE_MARKET_REGIME_CHECK)
            
            # 2. 周度再平衡 (周五 16:30)
            schedule.every().friday.at("16:30").do(self._run_task_safe, 
                                                   TaskType.WEEKLY_REBALANCING)
            
            # 3. 月度表现回顾 (每月1日 18:00) - 改为每30天执行
            schedule.every(30).days.at("18:00").do(self._run_task_safe, 
                                                   TaskType.MONTHLY_PERFORMANCE_REVIEW)
            
            # 4. 日内风险检查 (每小时)
            schedule.every().hour.do(self._run_task_safe, 
                                     TaskType.DAILY_RISK_CHECK)
            
            # 5. 冷静期监控 (每10分钟)
            schedule.every(10).minutes.do(self._run_task_safe, 
                                          TaskType.COOLDOWN_MONITOR)
            
            # 6. 动态配比检查 (每日 12:00)
            schedule.every().day.at("12:00").do(self._run_task_safe, 
                                                TaskType.DYNAMIC_ALLOCATION_CHECK)
            
            # 初始化任务记录
            self._initialize_task_records()
            
            self.logger.info("调度任务设置完成")
            
        except Exception as e:
            self.logger.error(f"设置调度任务失败: {e}")
            raise
    
    def _initialize_task_records(self):
        """初始化任务记录"""
        for task_type in TaskType:
            task_id = f"task_{task_type.value}"
            self.scheduled_tasks[task_id] = ScheduledTask(
                task_id=task_id,
                task_type=task_type,
                schedule_time="varies",
                last_run=None,
                next_run=None,
                run_count=0,
                status="scheduled"
            )
    
    def start_scheduler(self):
        """启动调度器"""
        try:
            if self.status == ScheduleStatus.RUNNING:
                self.logger.warning("调度器已在运行中")
                return
            
            self.logger.info("🚀 启动再平衡调度器")
            
            self.running = True
            self.status = ScheduleStatus.RUNNING
            
            # 启动调度器线程
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            
            self.logger.info("✅ 再平衡调度器已启动")
            
        except Exception as e:
            self.logger.error(f"启动调度器失败: {e}")
            self.status = ScheduleStatus.ERROR
            raise
    
    def stop_scheduler(self):
        """停止调度器"""
        try:
            self.logger.info("🛑 停止再平衡调度器")
            
            self.running = False
            self.status = ScheduleStatus.STOPPED
            
            # 等待调度器线程结束
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=10)
            
            self.logger.info("✅ 再平衡调度器已停止")
            
        except Exception as e:
            self.logger.error(f"停止调度器失败: {e}")
    
    def _scheduler_loop(self):
        """调度器主循环"""
        self.logger.info("调度器主循环启动")
        
        while self.running:
            try:
                # 运行待执行的任务
                schedule.run_pending()
                
                # 更新任务状态
                self._update_task_status()
                
                # 循环间隔
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                self.logger.error(f"调度器循环错误: {e}")
                time.sleep(300)  # 错误后等待5分钟
        
        self.logger.info("调度器主循环结束")
    
    def _run_task_safe(self, task_type: TaskType):
        """安全执行任务"""
        task_id = f"task_{task_type.value}"
        
        try:
            self.logger.info(f"🔄 执行任务: {task_type.value}")
            
            # 更新任务状态
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                task.status = "running"
                task.run_count += 1
            
            # 执行具体任务
            success = self._execute_task(task_type)
            
            # 更新任务完成状态
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                task.status = "completed" if success else "error"
                task.last_run = datetime.now().isoformat()
                task.error_message = "" if success else "任务执行失败"
            
            if success:
                self.logger.info(f"✅ 任务完成: {task_type.value}")
            else:
                self.logger.error(f"❌ 任务失败: {task_type.value}")
                
        except Exception as e:
            self.logger.error(f"任务执行异常: {task_type.value} - {e}")
            
            # 记录错误
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                task.status = "error"
                task.error_message = str(e)
    
    def _execute_task(self, task_type: TaskType) -> bool:
        """执行具体任务"""
        try:
            if task_type == TaskType.PRE_MARKET_REGIME_CHECK:
                return self._pre_market_regime_check()
            
            elif task_type == TaskType.WEEKLY_REBALANCING:
                return asyncio.run(self._weekly_rebalancing())
            
            elif task_type == TaskType.MONTHLY_PERFORMANCE_REVIEW:
                return self._monthly_performance_review()
            
            elif task_type == TaskType.DAILY_RISK_CHECK:
                return self._daily_risk_check()
            
            elif task_type == TaskType.COOLDOWN_MONITOR:
                return self._cooldown_monitor()
            
            elif task_type == TaskType.DYNAMIC_ALLOCATION_CHECK:
                return self._dynamic_allocation_check()
            
            else:
                self.logger.warning(f"未知任务类型: {task_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"执行任务失败: {task_type.value} - {e}")
            return False
    
    def _pre_market_regime_check(self) -> bool:
        """盘前市况检测"""
        try:
            self.logger.info("执行盘前市况检测")
            
            # 1. 检测市况
            regime_result = self.regime_detector.detect_regime(force_update=True)
            
            # 2. 检查是否需要调整配比
            current_allocation = self.allocation_manager.get_current_allocation()
            regime_allocation = (
                regime_result['allocation']['mean_reversion_weight'],
                regime_result['allocation']['trend_following_weight']
            )
            
            # 3. 计算配比偏离度
            allocation_drift = abs(current_allocation[0] - regime_allocation[0])
            
            self.logger.info(f"市况: {regime_result['regime_type']['description']}")
            self.logger.info(f"当前配比: A={current_allocation[0]:.1%}, B={current_allocation[1]:.1%}")
            self.logger.info(f"市况配比: A={regime_allocation[0]:.1%}, B={regime_allocation[1]:.1%}")
            self.logger.info(f"配比偏离: {allocation_drift:.1%}")
            
            # 4. 如果偏离超过阈值，触发再平衡
            if allocation_drift > 0.05:  # 5%阈值
                self.logger.info("配比偏离过大，触发盘前再平衡")
                return asyncio.run(self._execute_rebalancing("盘前市况变化"))
            
            return True
            
        except Exception as e:
            self.logger.error(f"盘前市况检测失败: {e}")
            return False
    
    async def _weekly_rebalancing(self) -> bool:
        """周度再平衡"""
        try:
            self.logger.info("执行周度再平衡")
            
            # 检查冷静期
            if self.cooldown_active:
                self.logger.info("冷静期中，跳过周度再平衡")
                return True
            
            # 执行再平衡
            return await self._execute_rebalancing("周度定期再平衡")
            
        except Exception as e:
            self.logger.error(f"周度再平衡失败: {e}")
            return False
    
    def _monthly_performance_review(self) -> bool:
        """月度表现回顾"""
        try:
            self.logger.info("执行月度表现回顾")
            
            # 1. 计算过去30天的策略表现
            strategy_performances = self._calculate_monthly_performance()
            
            # 2. 更新Sharpe权重
            if strategy_performances:
                self.allocation_manager.strategy_performances = strategy_performances
                self.logger.info("月度表现权重已更新")
            
            # 3. 检查动态加仓条件
            boost_adjustments = self.allocation_manager.check_dynamic_boost(strategy_performances)
            
            if boost_adjustments:
                self.logger.info(f"检测到动态加仓机会: {boost_adjustments}")
                # 应用动态调整
                asyncio.run(self._apply_dynamic_adjustments(boost_adjustments))
            
            return True
            
        except Exception as e:
            self.logger.error(f"月度表现回顾失败: {e}")
            return False
    
    def _daily_risk_check(self) -> bool:
        """日内风险检查"""
        try:
            # 检查冷静期
            if self.cooldown_active:
                return True
            
            # 获取系统状态
            system_status = self.dual_system.get_system_status()
            
            # 检查日内亏损
            daily_pnl = system_status.get('daily_pnl', 0)
            total_capital = system_status.get('total_capital', 100000)
            
            daily_loss_limit = total_capital * 0.02  # 2%日内亏损限制
            
            if daily_pnl <= -daily_loss_limit:
                self.logger.warning(f"触发日内亏损限制: {daily_pnl:.2f}")
                self._trigger_cooldown("日内亏损超限")
                return True
            
            # 检查总回撤
            total_pnl = system_status.get('total_pnl', 0)
            drawdown_limit = total_capital * 0.10  # 10%回撤限制
            
            if total_pnl <= -drawdown_limit:
                self.logger.warning(f"触发总回撤限制: {total_pnl:.2f}")
                self._trigger_cooldown("总回撤超限")
                return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"日内风险检查失败: {e}")
            return False
    
    def _cooldown_monitor(self) -> bool:
        """冷静期监控"""
        try:
            if not self.cooldown_active:
                return True
            
            # 检查冷静期是否结束
            if self.cooldown_start_time:
                elapsed = datetime.now() - self.cooldown_start_time
                if elapsed >= self.cooldown_duration:
                    self._end_cooldown()
                    self.logger.info("冷静期结束，恢复正常交易")
            
            return True
            
        except Exception as e:
            self.logger.error(f"冷静期监控失败: {e}")
            return False
    
    def _dynamic_allocation_check(self) -> bool:
        """动态配比检查"""
        try:
            # 计算策略表现
            strategy_performances = self._calculate_recent_performance()
            
            # 检查动态加仓条件
            boost_adjustments = self.allocation_manager.check_dynamic_boost(strategy_performances)
            
            if boost_adjustments:
                self.logger.info(f"动态配比调整: {boost_adjustments}")
                asyncio.run(self._apply_dynamic_adjustments(boost_adjustments))
            
            return True
            
        except Exception as e:
            self.logger.error(f"动态配比检查失败: {e}")
            return False
    
    async def _execute_rebalancing(self, reason: str) -> bool:
        """执行再平衡"""
        try:
            self.logger.info(f"开始执行再平衡: {reason}")
            
            # 记录开始时的配比
            old_allocation = self.allocation_manager.get_current_allocation()
            
            # 1. 检测市况
            regime_result = self.regime_detector.detect_regime()
            
            # 2. 计算策略表现
            strategy_performances = self._calculate_recent_performance()
            
            # 3. 生成新配比
            allocation_config = self.allocation_manager.generate_allocation_config(
                regime_result, strategy_performances
            )
            
            # 4. 执行再平衡（这里应该调用双策略系统的再平衡方法）
            await self.dual_system._apply_allocation(allocation_config)
            
            # 5. 保存配置
            self.allocation_manager.save_allocation_config(allocation_config)
            
            # 6. 记录再平衡事件
            rebalancing_event = RebalancingEvent(
                timestamp=datetime.now().isoformat(),
                trigger_reason=reason,
                old_allocation={'A': old_allocation[0], 'B': old_allocation[1]},
                new_allocation={'A': allocation_config.final_allocation_A, 'B': allocation_config.final_allocation_B},
                regime_info=regime_result,
                performance_data={},  # TODO: 添加表现数据
                trades_executed=[],   # TODO: 记录执行的交易
                success=True
            )
            
            self.rebalancing_history.append(rebalancing_event)
            self._save_rebalancing_history()
            
            self.logger.info(f"再平衡完成: A={allocation_config.final_allocation_A:.1%}, B={allocation_config.final_allocation_B:.1%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"执行再平衡失败: {e}")
            
            # 记录失败事件
            failed_event = RebalancingEvent(
                timestamp=datetime.now().isoformat(),
                trigger_reason=reason,
                old_allocation={},
                new_allocation={},
                regime_info={},
                performance_data={},
                trades_executed=[],
                success=False,
                error_message=str(e)
            )
            
            self.rebalancing_history.append(failed_event)
            self._save_rebalancing_history()
            
            return False
    
    def _trigger_cooldown(self, reason: str):
        """触发冷静期"""
        try:
            self.logger.warning(f"触发冷静期: {reason}")
            
            self.cooldown_active = True
            self.cooldown_start_time = datetime.now()
            self.cooldown_reason = reason
            
            # 暂停双策略系统的交易
            asyncio.run(self._pause_trading())
            
            # 记录冷静期事件
            cooldown_file = os.path.join(self.history_dir, "cooldown_events.json")
            cooldown_event = {
                'timestamp': self.cooldown_start_time.isoformat(),
                'reason': reason,
                'duration_hours': self.cooldown_duration.total_seconds() / 3600
            }
            
            cooldown_events = []
            if os.path.exists(cooldown_file):
                try:
                    with open(cooldown_file, 'r', encoding='utf-8') as f:
                        cooldown_events = json.load(f)
                except:
                    cooldown_events = []
            
            cooldown_events.append(cooldown_event)
            
            with open(cooldown_file, 'w', encoding='utf-8') as f:
                json.dump(cooldown_events, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"触发冷静期失败: {e}")
    
    def _end_cooldown(self):
        """结束冷静期"""
        try:
            self.cooldown_active = False
            self.cooldown_start_time = None
            self.cooldown_reason = ""
            
            # 恢复双策略系统的交易
            asyncio.run(self._resume_trading())
            
        except Exception as e:
            self.logger.error(f"结束冷静期失败: {e}")
    
    async def _pause_trading(self):
        """暂停交易"""
        for strategy in self.dual_system.strategies.values():
            await strategy.pause()
    
    async def _resume_trading(self):
        """恢复交易"""
        for strategy in self.dual_system.strategies.values():
            await strategy.resume()
    
    async def _apply_dynamic_adjustments(self, boost_adjustments: Dict[str, float]):
        """应用动态调整"""
        try:
            # 获取当前配置
            current_config = self.allocation_manager.current_config
            
            if current_config:
                # 应用动态调整
                adjusted_config = self.allocation_manager.apply_dynamic_adjustments(
                    current_config, boost_adjustments
                )
                
                # 应用到双策略系统
                await self.dual_system._apply_allocation(adjusted_config)
                
                # 保存配置
                self.allocation_manager.save_allocation_config(adjusted_config)
                
                self.logger.info("动态调整已应用")
            
        except Exception as e:
            self.logger.error(f"应用动态调整失败: {e}")
    
    def _calculate_monthly_performance(self) -> Dict[str, StrategyPerformance]:
        """计算月度表现"""
        # 这里应该从实际交易历史计算表现
        # 现在返回模拟数据
        return {}
    
    def _calculate_recent_performance(self) -> Dict[str, StrategyPerformance]:
        """计算最近表现"""
        # 这里应该从实际交易历史计算表现
        # 现在返回模拟数据
        return {}
    
    def _update_task_status(self):
        """更新任务状态"""
        try:
            # 更新下次运行时间等信息
            for task in self.scheduled_tasks.values():
                if task.status != "running":
                    task.status = "scheduled"
        except Exception as e:
            self.logger.error(f"更新任务状态失败: {e}")
    
    def _save_rebalancing_history(self):
        """保存再平衡历史"""
        try:
            history_file = os.path.join(self.history_dir, "rebalancing_events.json")
            
            # 转换为可序列化格式
            history_data = [asdict(event) for event in self.rebalancing_history]
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"保存再平衡历史失败: {e}")
    
    def get_scheduler_status(self) -> Dict:
        """获取调度器状态"""
        return {
            'status': self.status.value,
            'running': self.running,
            'cooldown_active': self.cooldown_active,
            'cooldown_reason': self.cooldown_reason,
            'cooldown_start_time': self.cooldown_start_time.isoformat() if self.cooldown_start_time else None,
            'task_count': len(self.scheduled_tasks),
            'rebalancing_events': len(self.rebalancing_history),
            'next_rebalancing': self._get_next_rebalancing_time()
        }
    
    def _get_next_rebalancing_time(self) -> Optional[str]:
        """获取下次再平衡时间"""
        try:
            # 计算下个周五16:30
            now = datetime.now()
            days_ahead = 4 - now.weekday()  # 4是周五
            if days_ahead <= 0:  # 当前周的周五已过
                days_ahead += 7
            
            next_friday = now + timedelta(days=days_ahead)
            next_rebalancing = next_friday.replace(hour=16, minute=30, second=0, microsecond=0)
            
            return next_rebalancing.isoformat()
            
        except Exception as e:
            self.logger.error(f"计算下次再平衡时间失败: {e}")
            return None

def main():
    """测试再平衡调度器"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 模拟配置
    config = {
        'total_capital': 100000.0,
        'cooldown_days': 5
    }
    
    # 创建双策略系统（模拟）
    from dual_strategy_system import DualStrategySystem
    dual_system = DualStrategySystem(config)
    
    # 创建调度器
    scheduler = RebalancingScheduler(dual_system, config)
    
    try:
        # 启动调度器
        scheduler.start_scheduler()
        
        print("📅 再平衡调度器运行中...")
        print("状态:", scheduler.get_scheduler_status())
        
        # 运行一段时间
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断")
    finally:
        # 停止调度器
        scheduler.stop_scheduler()
        print("✅ 再平衡调度器测试完成")

if __name__ == "__main__":
    main()