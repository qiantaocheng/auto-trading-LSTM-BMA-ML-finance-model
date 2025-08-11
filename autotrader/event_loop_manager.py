#!/usr/bin/env python3
"""
线程安全的事件循环管理器
解决Tkinter GUI和asyncio事件循环的线程安全问题
"""

import asyncio
import threading
import queue
import time
import logging
import weakref
from typing import Optional, Callable, Any, Dict, Coroutine
from concurrent.futures import ThreadPoolExecutor, Future
import uuid

class ThreadSafeEventLoopManager:
    """线程安全的事件循环管理器"""
    
    def __init__(self, name: str = "TradingLoop"):
        self.name = name
        self.logger = logging.getLogger(f"EventLoop.{name}")
        
        # 核心组件
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._stopped = threading.Event()
        
        # 任务管理
        self._tasks: Dict[str, asyncio.Task] = {}
        self._pending_futures: Dict[str, Future] = {}
        
        # 线程安全的命令队列
        self._command_queue: queue.Queue = queue.Queue()
        self._response_queue: queue.Queue = queue.Queue()
        
        # 状态标志
        self._running = False
        self._error: Optional[Exception] = None
        
        # 统计信息
        self._stats = {
            'commands_processed': 0,
            'tasks_created': 0,
            'errors': 0,
            'start_time': None
        }
    
    def start(self, timeout: float = 10.0) -> bool:
        """启动事件循环（线程安全）"""
        if self._running:
            self.logger.warning("事件循环已在运行")
            return True
        
        self.logger.info("启动事件循环管理器...")
        
        # 创建并启动线程
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name=f"{self.name}-Thread",
            daemon=True
        )
        self._thread.start()
        
        # 等待启动完成
        if not self._started.wait(timeout):
            self.logger.error("事件循环启动超时")
            self._cleanup()
            return False
        
        if self._error:
            self.logger.error(f"事件循环启动失败: {self._error}")
            return False
        
        self._stats['start_time'] = time.time()
        self.logger.info("事件循环管理器启动成功")
        return True
    
    def _run_event_loop(self):
        """在独立线程中运行事件循环"""
        try:
            # 创建新的事件循环
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            # 设置异常处理器
            self._loop.set_exception_handler(self._exception_handler)
            
            # 添加命令处理器
            self._loop.call_soon(self._process_commands)
            
            # 标记启动完成
            self._running = True
            self._started.set()
            
            self.logger.info(f"事件循环已在线程 {threading.current_thread().name} 中启动")
            
            # 运行事件循环
            self._loop.run_forever()
            
        except Exception as e:
            self.logger.error(f"事件循环异常: {e}")
            self._error = e
            self._stats['errors'] += 1
            self._started.set()
        finally:
            self._cleanup_loop()
    
    def _process_commands(self):
        """处理来自其他线程的命令"""
        if not self._running:
            return
        
        try:
            # 非阻塞检查命令队列
            processed = 0
            while not self._command_queue.empty() and processed < 10:  # 限制每次处理数量
                try:
                    cmd = self._command_queue.get_nowait()
                    self._execute_command(cmd)
                    processed += 1
                    self._stats['commands_processed'] += 1
                except queue.Empty:
                    break
        except Exception as e:
            self.logger.error(f"命令处理异常: {e}")
            self._stats['errors'] += 1
        finally:
            # 继续调度下一次检查
            if self._running and self._loop:
                self._loop.call_later(0.05, self._process_commands)  # 50ms间隔
    
    def _execute_command(self, cmd: Dict[str, Any]):
        """执行命令"""
        cmd_type = cmd.get('type')
        cmd_id = cmd.get('id')
        
        try:
            if cmd_type == 'submit_coro':
                coro = cmd['coro']
                task = self._loop.create_task(coro)
                self._tasks[cmd_id] = task
                self._stats['tasks_created'] += 1
                
                # 设置完成回调
                def on_done(t):
                    try:
                        if t.cancelled():
                            result = {'status': 'cancelled'}
                        elif t.exception():
                            result = {'status': 'error', 'error': t.exception()}
                        else:
                            result = {'status': 'success', 'result': t.result()}
                        
                        self._response_queue.put({
                            'id': cmd_id,
                            **result
                        })
                    except Exception as e:
                        self._response_queue.put({
                            'id': cmd_id,
                            'status': 'error',
                            'error': e
                        })
                    finally:
                        # 清理任务引用
                        self._tasks.pop(cmd_id, None)
                
                task.add_done_callback(on_done)
                
            elif cmd_type == 'cancel_task':
                task_id = cmd['task_id']
                if task_id in self._tasks:
                    self._tasks[task_id].cancel()
                    
            elif cmd_type == 'stop':
                self._running = False
                # 取消所有任务
                for task in list(self._tasks.values()):
                    if not task.done():
                        task.cancel()
                self._loop.stop()
                
        except Exception as e:
            self.logger.error(f"执行命令失败: {e}")
            self._stats['errors'] += 1
            self._response_queue.put({
                'id': cmd_id,
                'status': 'error',
                'error': e
            })
    
    def submit_coroutine(self, coro: Coroutine, timeout: Optional[float] = None) -> Any:
        """线程安全地提交协程并等待结果"""
        if not self._running:
            raise RuntimeError("事件循环未运行")
        
        # 生成唯一ID
        cmd_id = str(uuid.uuid4())
        
        # 提交命令
        self._command_queue.put({
            'type': 'submit_coro',
            'id': cmd_id,
            'coro': coro
        })
        
        # 等待结果
        start_time = time.time()
        while True:
            try:
                # 检查响应队列
                try:
                    response = self._response_queue.get_nowait()
                    if response['id'] == cmd_id:
                        if response['status'] == 'success':
                            return response.get('result')
                        elif response['status'] == 'cancelled':
                            raise asyncio.CancelledError("协程被取消")
                        else:
                            raise response.get('error', Exception("未知错误"))
                except queue.Empty:
                    pass
                
                # 检查超时
                if timeout and (time.time() - start_time) > timeout:
                    # 取消任务
                    self._command_queue.put({
                        'type': 'cancel_task',
                        'id': f'cancel_{cmd_id}',
                        'task_id': cmd_id
                    })
                    raise TimeoutError(f"协程执行超时: {timeout}秒")
                
                time.sleep(0.01)  # 避免CPU过度使用
                
            except KeyboardInterrupt:
                # 用户中断，取消任务
                self._command_queue.put({
                    'type': 'cancel_task',
                    'id': f'cancel_{cmd_id}',
                    'task_id': cmd_id
                })
                raise
    
    def submit_coroutine_nowait(self, coro: Coroutine) -> str:
        """提交协程但不等待结果，返回任务ID"""
        if not self._running:
            raise RuntimeError("事件循环未运行")
        
        cmd_id = str(uuid.uuid4())
        self._command_queue.put({
            'type': 'submit_coro',
            'id': cmd_id,
            'coro': coro
        })
        
        return cmd_id
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """获取任务结果"""
        start_time = time.time()
        
        while True:
            try:
                response = self._response_queue.get_nowait()
                if response['id'] == task_id:
                    if response['status'] == 'success':
                        return response.get('result')
                    elif response['status'] == 'cancelled':
                        raise asyncio.CancelledError("任务被取消")
                    else:
                        raise response.get('error', Exception("未知错误"))
            except queue.Empty:
                pass
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"获取结果超时: {timeout}秒")
            
            time.sleep(0.01)
    
    def stop(self, timeout: float = 5.0):
        """停止事件循环"""
        if not self._running:
            return
        
        self.logger.info("停止事件循环管理器...")
        
        # 发送停止命令
        self._command_queue.put({'type': 'stop', 'id': 'stop'})
        
        # 等待线程结束
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)
            
            if self._thread.is_alive():
                self.logger.warning("事件循环线程未能正常结束")
        
        self._cleanup()
        self.logger.info("事件循环管理器已停止")
    
    def _cleanup(self):
        """清理资源"""
        self._running = False
        self._loop = None
        self._thread = None
        self._tasks.clear()
        self._stopped.set()
    
    def _cleanup_loop(self):
        """清理事件循环"""
        if self._loop:
            # 取消所有任务
            for task in self._tasks.values():
                if not task.done():
                    task.cancel()
            
            # 关闭循环
            if not self._loop.is_closed():
                self._loop.close()
    
    def _exception_handler(self, loop, context):
        """异常处理器"""
        self.logger.error(f"事件循环异常: {context}")
        self._stats['errors'] += 1
    
    def is_running(self) -> bool:
        """检查是否运行中"""
        return self._running and self._loop is not None and not self._loop.is_closed()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = time.time() - (self._stats['start_time'] or 0) if self._stats['start_time'] else 0
        
        return {
            'running': self._running,
            'thread_alive': self._thread.is_alive() if self._thread else False,
            'active_tasks': len(self._tasks),
            'pending_commands': self._command_queue.qsize(),
            'pending_responses': self._response_queue.qsize(),
            'thread_name': self._thread.name if self._thread else None,
            'uptime_seconds': uptime,
            **self._stats
        }
    
    def cancel_all_tasks(self):
        """取消所有任务"""
        for task_id in list(self._tasks.keys()):
            self._command_queue.put({
                'type': 'cancel_task',
                'id': f'cancel_all_{task_id}',
                'task_id': task_id
            })


# 全局事件循环管理器
_global_loop_manager: Optional[ThreadSafeEventLoopManager] = None

def get_event_loop_manager() -> ThreadSafeEventLoopManager:
    """获取全局事件循环管理器"""
    global _global_loop_manager
    if _global_loop_manager is None:
        _global_loop_manager = ThreadSafeEventLoopManager("GlobalTradingLoop")
    return _global_loop_manager

def ensure_event_loop_running() -> ThreadSafeEventLoopManager:
    """确保事件循环运行"""
    manager = get_event_loop_manager()
    if not manager.is_running():
        if not manager.start():
            raise RuntimeError("无法启动事件循环管理器")
    return manager

def safe_run_async(coro: Coroutine, timeout: Optional[float] = None) -> Any:
    """安全运行异步函数（自动处理事件循环）"""
    manager = ensure_event_loop_running()
    return manager.submit_coroutine(coro, timeout)

def shutdown_event_loop():
    """关闭全局事件循环"""
    global _global_loop_manager
    if _global_loop_manager:
        _global_loop_manager.stop()
        _global_loop_manager = None
