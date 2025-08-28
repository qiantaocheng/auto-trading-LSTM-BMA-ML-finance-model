#!/usr/bin/env python3
"""
事件循环管理器
简化实现，支持GUI应用的事件循环管理
"""

import asyncio
import threading
import logging
from typing import Optional, Any, Callable
import queue
import time

logger = logging.getLogger(__name__)

class EventLoopManager:
    """事件循环管理器"""
    
    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        self.task_queue = queue.Queue()
        self._startup_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        
    def start(self) -> bool:
        """启动事件循环"""
        with self._lock:
            if self.is_running:
                logger.warning("事件循环已经在运行")
                return True
                
            try:
                # 重置事件
                self._startup_event.clear()
                self._shutdown_event.clear()
                
                # 创建新的事件循环
                self.loop = asyncio.new_event_loop()
                
                # 在新线程中运行事件循环
                self.thread = threading.Thread(target=self._run_loop, daemon=True)
                self.thread.start()
                
                # 等待线程实际启动（最多等待5秒）
                if self._startup_event.wait(timeout=5.0):
                    self.is_running = True
                    logger.info("事件循环管理器启动成功")
                    return True
                else:
                    logger.error("事件循环启动超时")
                    self._cleanup_failed_start()
                    return False
                    
            except Exception as e:
                logger.error(f"启动事件循环失败: {e}")
                self._cleanup_failed_start()
                return False
    
    def _cleanup_failed_start(self):
        """清理失败的启动状态"""
        self.is_running = False
        if self.loop and not self.loop.is_closed():
            self.loop.close()
        self.loop = None
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.thread = None
            
    def _run_loop(self):
        """在线程中运行事件循环"""
        try:
            asyncio.set_event_loop(self.loop)
            # 通知启动成功
            self._startup_event.set()
            logger.debug("事件循环线程启动完成")
            
            # 运行事件循环
            self.loop.run_forever()
            
        except Exception as e:
            logger.error(f"事件循环运行错误: {e}")
        finally:
            # 清理状态
            with self._lock:
                self.is_running = False
            self._shutdown_event.set()
            logger.debug("事件循环线程结束")
            
    def stop(self):
        """停止事件循环"""
        with self._lock:
            if not self.is_running or not self.loop:
                return
                
            try:
                # 停止事件循环
                self.loop.call_soon_threadsafe(self.loop.stop)
                
                # 等待线程结束
                if self.thread and self.thread.is_alive():
                    # 等待关闭事件
                    if self._shutdown_event.wait(timeout=5.0):
                        logger.debug("事件循环正常关闭")
                    else:
                        logger.warning("事件循环关闭超时")
                    
                    # 确保线程结束
                    self.thread.join(timeout=1.0)
                    if self.thread.is_alive():
                        logger.error("事件循环线程未能正常结束")
                
                # 清理资源
                if self.loop and not self.loop.is_closed():
                    self.loop.close()
                
                self.is_running = False
                self.loop = None
                self.thread = None
                logger.info("事件循环管理器已停止")
                
            except Exception as e:
                logger.error(f"停止事件循环失败: {e}")
                # 强制清理状态
                self.is_running = False
                self.loop = None
                self.thread = None
            
    def submit_task(self, coro_or_func: Any, *args, **kwargs) -> Optional[Any]:
        """提交任务到事件循环"""
        if not self.is_running or not self.loop:
            logger.warning("事件循环未运行，无法提交任务")
            return None
            
        try:
            if asyncio.iscoroutinefunction(coro_or_func):
                # 异步函数
                future = asyncio.run_coroutine_threadsafe(
                    coro_or_func(*args, **kwargs), self.loop
                )
                return future.result(timeout=30.0)
            else:
                # 同步函数
                future = self.loop.run_in_executor(
                    None, lambda: coro_or_func(*args, **kwargs)
                )
                return asyncio.run_coroutine_threadsafe(future, self.loop).result(timeout=30.0)
                
        except Exception as e:
            logger.error(f"提交任务失败: {e}")
            return None
            
    def submit_coroutine_nowait(self, coro) -> Optional[str]:
        """提交协程到事件循环，不等待结果"""
        if not self.is_running or not self.loop:
            logger.warning("事件循环未运行，无法提交协程")
            return None
            
        try:
            import uuid
            task_id = str(uuid.uuid4())
            # 使用call_soon_threadsafe安全地从其他线程调度协程
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            logger.debug(f"协程提交成功，任务ID: {task_id}")
            return task_id
        except Exception as e:
            logger.error(f"提交协程失败: {e}")
            return None
            
    def submit_coroutine(self, coro, timeout: float = 30.0) -> Optional[Any]:
        """提交协程到事件循环，等待结果"""
        if not self.is_running or not self.loop:
            logger.warning("事件循环未运行，无法提交协程")
            return None
            
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            result = future.result(timeout=timeout)
            logger.debug("协程执行完成")
            return result
        except Exception as e:
            logger.error(f"提交协程失败: {e}")
            return None
            
    def schedule_callback(self, callback: Callable, delay: float = 0):
        """调度回调函数"""
        if not self.is_running or not self.loop:
            logger.warning("事件循环未运行，无法调度回调")
            return
            
        try:
            if delay > 0:
                self.loop.call_later(delay, callback)
            else:
                self.loop.call_soon_threadsafe(callback)
        except Exception as e:
            logger.error(f"调度回调失败: {e}")
            
    def is_alive(self) -> bool:
        """检查事件循环是否活跃"""
        return self.is_running and self.loop is not None and not self.loop.is_closed()

# 全局事件循环管理器实例
_global_event_loop_manager: Optional[EventLoopManager] = None

def get_event_loop_manager() -> EventLoopManager:
    """获取全局事件循环管理器实例"""
    global _global_event_loop_manager
    
    if _global_event_loop_manager is None:
        _global_event_loop_manager = EventLoopManager()
        
    return _global_event_loop_manager

def init_event_loop_manager() -> EventLoopManager:
    """初始化并启动事件循环管理器"""
    manager = get_event_loop_manager()
    if not manager.is_running:
        manager.start()
    return manager

def cleanup_event_loop_manager():
    """清理事件循环管理器"""
    global _global_event_loop_manager
    
    if _global_event_loop_manager and _global_event_loop_manager.is_running:
        _global_event_loop_manager.stop()
        _global_event_loop_manager = None