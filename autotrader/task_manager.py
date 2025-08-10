#!/usr/bin/env python3
"""
任务管理器 - 可靠的异步任务生命周期管理
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Callable, Any, Set
from dataclasses import dataclass
from enum import Enum


class TaskState(Enum):
    """任务状态"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    RESTARTING = "RESTARTING"


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    task: Optional[asyncio.Task]
    state: TaskState
    target_func: Callable
    args: tuple
    kwargs: dict
    created_at: float
    last_restart: float
    restart_count: int
    max_restarts: int
    restart_delay: float
    last_error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()
        if self.last_restart == 0:
            self.last_restart = self.created_at


class TaskManager:
    """可靠的异步任务管理器"""
    
    def __init__(self, default_max_restarts: int = 5, default_restart_delay: float = 1.0):
        self.logger = logging.getLogger("TaskManager")
        self.tasks: Dict[str, TaskInfo] = {}
        self.default_max_restarts = default_max_restarts
        self.default_restart_delay = default_restart_delay
        
        # 监控任务
        self._monitor_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False
        
        # 统计
        self.stats = {
            'tasks_created': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_restarted': 0,
            'total_restarts': 0
        }
    
    async def start_monitoring(self):
        """启动任务监控"""
        if not self._monitor_task or self._monitor_task.done():
            self._stop_monitoring = False
            self._monitor_task = asyncio.create_task(self._monitor_tasks())
            self.logger.info("任务监控已启动")
    
    async def stop_monitoring(self):
        """停止任务监控"""
        self._stop_monitoring = True
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("任务监控已停止")
    
    async def _monitor_tasks(self):
        """监控任务状态"""
        check_interval = 2.0  # 2秒检查一次
        
        try:
            while not self._stop_monitoring:
                current_time = time.time()
                
                # 检查所有任务
                for task_id, task_info in list(self.tasks.items()):
                    try:
                        await self._check_task_status(task_id, task_info, current_time)
                    except Exception as e:
                        self.logger.error(f"检查任务状态失败 {task_id}: {e}")
                
                await asyncio.sleep(check_interval)
                
        except asyncio.CancelledError:
            self.logger.info("任务监控被取消")
        except Exception as e:
            self.logger.error(f"任务监控异常: {e}")
    
    async def _check_task_status(self, task_id: str, task_info: TaskInfo, current_time: float):
        """检查单个任务状态"""
        if not task_info.task:
            return
        
        # 检查任务是否完成或异常
        if task_info.task.done():
            try:
                # 获取任务结果（如果有异常会抛出）
                result = task_info.task.result()
                task_info.state = TaskState.COMPLETED
                self.stats['tasks_completed'] += 1
                self.logger.debug(f"任务完成: {task_id}")
                
            except asyncio.CancelledError:
                task_info.state = TaskState.CANCELLED
                self.logger.debug(f"任务被取消: {task_id}")
                
            except Exception as e:
                # 任务异常，考虑重启
                task_info.state = TaskState.FAILED
                task_info.last_error = str(e)
                self.stats['tasks_failed'] += 1
                
                self.logger.warning(f"任务异常 {task_id}: {e}")
                
                # 判断是否需要重启
                if self._should_restart_task(task_info, current_time):
                    await self._restart_task(task_id, task_info)
                else:
                    self.logger.error(f"任务 {task_id} 达到最大重试次数，停止重启")
    
    def _should_restart_task(self, task_info: TaskInfo, current_time: float) -> bool:
        """判断是否应该重启任务"""
        # 检查重启次数限制
        if task_info.restart_count >= task_info.max_restarts:
            return False
        
        # 检查重启间隔
        time_since_last_restart = current_time - task_info.last_restart
        if time_since_last_restart < task_info.restart_delay:
            return False
        
        return True
    
    async def _restart_task(self, task_id: str, task_info: TaskInfo):
        """重启任务"""
        try:
            task_info.state = TaskState.RESTARTING
            task_info.restart_count += 1
            task_info.last_restart = time.time()
            self.stats['tasks_restarted'] += 1
            self.stats['total_restarts'] += 1
            
            # 计算退避延迟
            backoff_delay = task_info.restart_delay * (1.5 ** task_info.restart_count)
            backoff_delay = min(backoff_delay, 30.0)  # 最大30秒
            
            self.logger.info(f"重启任务 {task_id} (第{task_info.restart_count}次), 延迟{backoff_delay:.1f}秒")
            
            await asyncio.sleep(backoff_delay)
            
            # 创建新任务
            new_task = asyncio.create_task(
                task_info.target_func(*task_info.args, **task_info.kwargs)
            )
            task_info.task = new_task
            task_info.state = TaskState.RUNNING
            
            self.logger.info(f"任务 {task_id} 重启成功")
            
        except Exception as e:
            self.logger.error(f"重启任务失败 {task_id}: {e}")
            task_info.state = TaskState.FAILED
    
    def create_task(self, task_id: str, coro_func: Callable, *args, 
                   max_restarts: Optional[int] = None, 
                   restart_delay: Optional[float] = None, **kwargs) -> bool:
        """创建可管理的任务"""
        if task_id in self.tasks:
            existing_task = self.tasks[task_id]
            if existing_task.state in [TaskState.RUNNING, TaskState.RESTARTING]:
                self.logger.warning(f"任务 {task_id} 已存在且正在运行")
                return False
        
        try:
            # 创建任务
            task = asyncio.create_task(coro_func(*args, **kwargs))
            
            # 创建任务信息
            task_info = TaskInfo(
                task_id=task_id,
                task=task,
                state=TaskState.RUNNING,
                target_func=coro_func,
                args=args,
                kwargs=kwargs,
                created_at=time.time(),
                last_restart=time.time(),
                restart_count=0,
                max_restarts=max_restarts or self.default_max_restarts,
                restart_delay=restart_delay or self.default_restart_delay
            )
            
            self.tasks[task_id] = task_info
            self.stats['tasks_created'] += 1
            
            self.logger.info(f"创建任务: {task_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建任务失败 {task_id}: {e}")
            return False
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id not in self.tasks:
            self.logger.warning(f"任务不存在: {task_id}")
            return False
        
        task_info = self.tasks[task_id]
        if task_info.task and not task_info.task.done():
            task_info.task.cancel()
            task_info.state = TaskState.CANCELLED
            self.logger.info(f"取消任务: {task_id}")
            return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskState]:
        """获取任务状态"""
        if task_id in self.tasks:
            return self.tasks[task_id].state
        return None
    
    def is_task_running(self, task_id: str) -> bool:
        """检查任务是否正在运行"""
        status = self.get_task_status(task_id)
        return status in [TaskState.RUNNING, TaskState.RESTARTING]
    
    def ensure_task_running(self, task_id: str, coro_func: Callable, *args, 
                           max_restarts: Optional[int] = None, 
                           restart_delay: Optional[float] = None, **kwargs) -> bool:
        """确保任务正在运行（如果不存在或已停止则创建/重启）"""
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            if task_info.state in [TaskState.RUNNING, TaskState.RESTARTING]:
                return True  # 任务已在运行
            
            # 任务已停止，重新创建
            self.logger.info(f"重新创建已停止的任务: {task_id}")
        
        return self.create_task(task_id, coro_func, *args, 
                               max_restarts=max_restarts, 
                               restart_delay=restart_delay, **kwargs)
    
    def get_all_tasks_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有任务状态"""
        result = {}
        current_time = time.time()
        
        for task_id, task_info in self.tasks.items():
            result[task_id] = {
                'state': task_info.state.value,
                'restart_count': task_info.restart_count,
                'max_restarts': task_info.max_restarts,
                'uptime': current_time - task_info.created_at,
                'last_error': task_info.last_error,
                'time_since_restart': current_time - task_info.last_restart
            }
        
        return result
    
    def cancel_task(self, task_id: str) -> bool:
        """取消指定任务"""
        if task_id not in self.tasks:
            self.logger.warning(f"尝试取消不存在的任务: {task_id}")
            return False
        
        task_info = self.tasks[task_id]
        
        if task_info.task and not task_info.task.done():
            task_info.task.cancel()
            task_info.state = TaskState.CANCELLED
            self.logger.info(f"任务已取消: {task_id}")
            return True
        else:
            self.logger.warning(f"任务无法取消（未运行或已完成）: {task_id}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        running_tasks = sum(1 for t in self.tasks.values() if t.state == TaskState.RUNNING)
        failed_tasks = sum(1 for t in self.tasks.values() if t.state == TaskState.FAILED)
        
        return {
            **self.stats,
            'current_running': running_tasks,
            'current_failed': failed_tasks,
            'total_managed': len(self.tasks)
        }
    
    async def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """清理已完成的老任务"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        to_remove = []
        for task_id, task_info in self.tasks.items():
            if (task_info.state in [TaskState.COMPLETED, TaskState.CANCELLED] and 
                task_info.created_at < cutoff_time):
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
            self.logger.debug(f"清理已完成任务: {task_id}")
        
        return len(to_remove)
    
    async def shutdown(self):
        """关闭任务管理器"""
        self.logger.info("关闭任务管理器...")
        
        # 停止监控
        await self.stop_monitoring()
        
        # 取消所有运行中的任务
        active_tasks = []
        for task_id, task_info in self.tasks.items():
            if task_info.task and not task_info.task.done():
                task_info.task.cancel()
                active_tasks.append(task_info.task)
        
        # 等待任务取消完成
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        
        self.logger.info("任务管理器已关闭")