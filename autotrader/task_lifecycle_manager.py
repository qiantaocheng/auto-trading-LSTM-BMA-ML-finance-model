#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务生命周期管理器 - 防止内存泄漏
统一管理所有asyncio任务的创建、监控和清理
"""

import asyncio
import time
import logging
import weakref
from typing import Dict, Set, Optional, Callable, Any, List
from dataclasses import dataclass
from threading import Lock
from collections import defaultdict
import traceback

@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    task: asyncio.Task
    created_at: float
    creator: str
    description: str
    cleanup_callback: Optional[Callable] = None
    max_lifetime: Optional[float] = None  # 最大生存时间（秒）

class TaskLifecycleManager:
    """任务生命周期管理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("TaskLifecycleManager")
        
        # 任务跟踪
        self._tasks: Dict[str, TaskInfo] = {}
        self._task_groups: Dict[str, Set[str]] = defaultdict(set)
        self._lock = Lock()
        
        # 统计
        self._stats = {
            'created': 0,
            'completed': 0,
            'cancelled': 0,
            'leaked': 0,
            'cleanup_errors': 0
        }
        
        # 配置
        self.max_task_lifetime = 3600.0  # 默认1小时最大生存时间
        self.cleanup_interval = 300.0   # 5分钟清理间隔
        self.warn_threshold = 100       # 任务数量警告阈值
        
        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_monitor()
    
    def create_task(self, coro, task_id: Optional[str] = None, 
                   creator: str = "unknown", description: str = "",
                   group: Optional[str] = None, max_lifetime: Optional[float] = None,
                   cleanup_callback: Optional[Callable] = None) -> asyncio.Task:
        """创建并注册任务"""
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000)}_{id(coro)}"
        
        # 创建任务
        task = asyncio.create_task(coro)
        
        # 注册任务信息
        task_info = TaskInfo(
            task_id=task_id,
            task=task,
            created_at=time.time(),
            creator=creator,
            description=description,
            cleanup_callback=cleanup_callback,
            max_lifetime=max_lifetime or self.max_task_lifetime
        )
        
        with self._lock:
            # 检查重复ID
            if task_id in self._tasks:
                self.logger.warning(f"任务ID重复: {task_id}，将覆盖")
                old_task = self._tasks[task_id].task
                if not old_task.done():
                    old_task.cancel()
            
            self._tasks[task_id] = task_info
            
            # 添加到组
            if group:
                self._task_groups[group].add(task_id)
            
            self._stats['created'] += 1
        
        # 添加完成回调
        task.add_done_callback(lambda t: self._on_task_done(task_id, t))
        
        self.logger.debug(f"创建任务: {task_id} ({creator}) - {description}")
        
        # 检查任务数量
        if len(self._tasks) > self.warn_threshold:
            self.logger.warning(f"活跃任务数量过多: {len(self._tasks)}")
        
        return task
    
    def _on_task_done(self, task_id: str, task: asyncio.Task):
        """任务完成回调"""
        with self._lock:
            task_info = self._tasks.pop(task_id, None)
            if not task_info:
                return
            
            # 从组中移除
            for group_tasks in self._task_groups.values():
                group_tasks.discard(task_id)
            
            # 更新统计
            if task.cancelled():
                self._stats['cancelled'] += 1
            elif task.exception():
                self._stats['completed'] += 1
                exception = task.exception()
                if not isinstance(exception, asyncio.CancelledError):
                    self.logger.warning(f"任务异常完成 {task_id}: {exception}")
            else:
                self._stats['completed'] += 1
            
            # 执行清理回调
            if task_info.cleanup_callback:
                try:
                    if asyncio.iscoroutinefunction(task_info.cleanup_callback):
                        # 异步回调需要在事件循环中执行
                        asyncio.create_task(task_info.cleanup_callback())
                    else:
                        task_info.cleanup_callback()
                except Exception as e:
                    self.logger.error(f"任务清理回调失败 {task_id}: {e}")
                    self._stats['cleanup_errors'] += 1
        
        self.logger.debug(f"任务完成: {task_id}")
    
    def cancel_task(self, task_id: str, reason: str = "") -> bool:
        """取消指定任务"""
        with self._lock:
            task_info = self._tasks.get(task_id)
            if not task_info or task_info.task.done():
                return False
            
            task_info.task.cancel()
            self.logger.info(f"取消任务: {task_id} - {reason}")
            return True
    
    def cancel_group(self, group: str, reason: str = "") -> int:
        """取消整个组的任务"""
        cancelled_count = 0
        with self._lock:
            task_ids = list(self._task_groups.get(group, set()))
        
        for task_id in task_ids:
            if self.cancel_task(task_id, f"{reason} (group: {group})"):
                cancelled_count += 1
        
        self.logger.info(f"取消任务组 {group}: {cancelled_count} 个任务")
        return cancelled_count
    
    def get_task(self, task_id: str) -> Optional[asyncio.Task]:
        """获取任务"""
        with self._lock:
            task_info = self._tasks.get(task_id)
            return task_info.task if task_info else None
    
    def list_tasks(self, group: Optional[str] = None, 
                  creator: Optional[str] = None) -> List[TaskInfo]:
        """列出任务"""
        with self._lock:
            tasks = []
            for task_info in self._tasks.values():
                if group and task_info.task_id not in self._task_groups.get(group, set()):
                    continue
                if creator and task_info.creator != creator:
                    continue
                tasks.append(task_info)
            return tasks.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            active_tasks = len(self._tasks)
            active_groups = {k: len(v) for k, v in self._task_groups.items() if v}
            
            # 计算任务年龄分布
            current_time = time.time()
            age_distribution = {'<1min': 0, '1-5min': 0, '5-30min': 0, '>30min': 0}
            
            for task_info in self._tasks.values():
                age = current_time - task_info.created_at
                if age < 60:
                    age_distribution['<1min'] += 1
                elif age < 300:
                    age_distribution['1-5min'] += 1
                elif age < 1800:
                    age_distribution['5-30min'] += 1
                else:
                    age_distribution['>30min'] += 1
            
            return {
                'active_tasks': active_tasks,
                'active_groups': active_groups,
                'age_distribution': age_distribution,
                'lifecycle_stats': self._stats.copy(),
                'memory_usage_estimate': active_tasks * 1024  # 粗略估算
            }
    
    def _start_cleanup_monitor(self):
        """启动清理监控任务"""
        async def cleanup_monitor():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self._periodic_cleanup()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"清理监控异常: {e}")
        
        try:
            self._cleanup_task = asyncio.create_task(cleanup_monitor())
        except RuntimeError:
            # 没有运行的事件循环，稍后再启动
            self.logger.debug("暂无事件循环，清理监控将稍后启动")
    
    async def _periodic_cleanup(self):
        """定期清理过期和异常任务"""
        current_time = time.time()
        expired_tasks = []
        leaked_tasks = []
        
        with self._lock:
            for task_id, task_info in list(self._tasks.items()):
                # 检查过期任务
                if current_time - task_info.created_at > task_info.max_lifetime:
                    expired_tasks.append(task_id)
                
                # 检查泄漏任务（已完成但未清理）
                if task_info.task.done() and task_id in self._tasks:
                    leaked_tasks.append(task_id)
        
        # 清理过期任务
        for task_id in expired_tasks:
            self.cancel_task(task_id, "任务超时")
        
        # 清理泄漏任务
        for task_id in leaked_tasks:
            with self._lock:
                self._tasks.pop(task_id, None)
            self._stats['leaked'] += 1
        
        if expired_tasks or leaked_tasks:
            self.logger.info(f"定期清理: 过期任务 {len(expired_tasks)} 个, 泄漏任务 {len(leaked_tasks)} 个")
    
    async def shutdown(self):
        """关闭管理器"""
        self.logger.info("开始关闭任务生命周期管理器...")
        
        # 停止清理监控
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有活跃任务
        active_tasks = []
        with self._lock:
            for task_info in self._tasks.values():
                if not task_info.task.done():
                    task_info.task.cancel()
                    active_tasks.append(task_info.task)
        
        # 等待任务完成
        if active_tasks:
            self.logger.info(f"等待 {len(active_tasks)} 个任务完成...")
            await asyncio.gather(*active_tasks, return_exceptions=True)
        
        # 清理状态
        with self._lock:
            self._tasks.clear()
            self._task_groups.clear()
        
        self.logger.info("任务生命周期管理器已关闭")
    
    def force_cleanup(self):
        """强制清理（同步方法）"""
        with self._lock:
            # 取消所有任务
            for task_info in self._tasks.values():
                if not task_info.task.done():
                    task_info.task.cancel()
            
            # 清理状态
            cancelled_count = len(self._tasks)
            self._tasks.clear()
            self._task_groups.clear()
            self._stats['cancelled'] += cancelled_count
        
        self.logger.warning(f"强制清理了 {cancelled_count} 个任务")

# 全局实例
_global_task_manager: Optional[TaskLifecycleManager] = None

def get_task_manager() -> TaskLifecycleManager:
    """获取全局任务生命周期管理器实例（兼容接口）"""
    global _global_task_manager
    if _global_task_manager is None:
        _global_task_manager = TaskLifecycleManager()
    return _global_task_manager

def create_managed_task(coro, task_id: Optional[str] = None, 
                       creator: str = "unknown", description: str = "",
                       group: Optional[str] = None, 
                       cleanup_callback: Optional[Callable] = None) -> asyncio.Task:
    """便捷函数：创建受管理的任务"""
    # 自动获取调用者信息
    if creator == "unknown":
        import inspect
        frame = inspect.currentframe().f_back
        creator = f"{frame.f_code.co_filename}:{frame.f_lineno}"
    
    manager = get_task_manager()
    return manager.create_task(coro, task_id, creator, description, group, cleanup_callback=cleanup_callback)

async def shutdown_task_manager():
    """关闭全局任务管理器"""
    global _global_task_manager
    if _global_task_manager:
        await _global_task_manager.shutdown()
        _global_task_manager = None
