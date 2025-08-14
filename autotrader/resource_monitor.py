#!/usr/bin/env python3
"""
资源监控器，检测and预防资源泄漏
监控任务、connection、内存、线程等资源使use情况
"""

import asyncio
import psutil
import threading
import time
import gc
import weakref
import logging
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import traceback
import os

@dataclass
class ResourceInfo:
    """资源信息"""
    type: str
    id: str
    created_at: float
    size: Optional[int] = None
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ResourceMonitor:
    """资源监控器，检测and预防资源泄漏"""
    
    def __init__(self, enable_traceback: bool = False):
        self.logger = logging.getLogger("ResourceMonitor")
        self.enable_traceback = enable_traceback
        
        # 资源追踪（使use弱引use避免循环引use）
        self._tasks: weakref.WeakSet = weakref.WeakSet()
        self._connections: weakref.WeakSet = weakref.WeakSet()
        self._handlers: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._files: weakref.WeakSet = weakref.WeakSet()
        
        # 资源统计
        self._resource_stats = defaultdict(lambda: {
            'created': 0,
            'destroyed': 0,
            'active': 0,
            'peak': 0,
            'total_created': 0
        })
        
        # 内存监控
        self._process = psutil.Process()
        self._memory_history: deque = deque(maxlen=60)  # 保留60个数据点
        self._memory_limit_mb = 2048  # 2GB限制
        self._last_gc_time = 0
        self._gc_threshold = 300  # 5分钟
        
        # 性能监控
        self._cpu_history: deque = deque(maxlen=30)
        self._performance_alerts = []
        
        # 监控线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._monitor_interval = 30.0  # 30 seconds间隔
        
        # 警告回调
        self._warning_callbacks: List[Callable[[str, Dict], None]] = []
        
        # 清理策略
        self._cleanup_strategies = {
            'memory': self._cleanup_memory,
            'tasks': self._cleanup_tasks,
            'connections': self._cleanup_connections,
            'files': self._cleanup_files
        }
        
        # 统计信息
        self._start_time = time.time()
        self._total_warnings = 0
        self._total_cleanups = 0
    
    def start_monitoring(self, interval: float = 30.0):
        """start监控"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self.logger.warning("资源监控in运行")
            return
        
        self._monitor_interval = interval
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="ResourceMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info(f"Resource monitoring started, interval: {interval} seconds")
    
    def stop_monitoring(self):
        """停止监控"""
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("资源监控停止")
    
    def add_warning_callback(self, callback: Callable[[str, Dict], None]):
        """添加警告回调函数"""
        self._warning_callbacks.append(callback)
    
    def _monitor_loop(self):
        """监控循环"""
        while not self._stop_monitoring.is_set():
            try:
                self._check_all_resources()
                self._update_statistics()
                
            except Exception as e:
                self.logger.error(f"监控异常: {e}")
                if self.enable_traceback:
                    self.logger.error(traceback.format_exc())
            
            self._stop_monitoring.wait(self._monitor_interval)
    
    def _check_all_resources(self):
        """check所has资源"""
        # 1. check内存
        self._check_memory()
        
        # 2. checkCPU
        self._check_cpu()
        
        # 3. check任务
        self._check_tasks()
        
        # 4. checkconnection
        self._check_connections()
        
        # 5. check线程
        self._check_threads()
        
        # 6. check文件句柄
        self._check_file_handles()
        
        # 7. check磁盘空间
        self._check_disk_space()
    
    def _check_memory(self):
        """check内存使use"""
        try:
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self._memory_history.append({
                'timestamp': time.time(),
                'rss_mb': memory_mb,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': self._process.memory_percent()
            })
            
            # check内存增长趋势
            if len(self._memory_history) >= 10:
                recent = list(self._memory_history)[-10:]
                oldest = recent[0]['rss_mb']
                newest = recent[-1]['rss_mb']
                
                if oldest > 0:
                    growth_rate = (newest - oldest) / oldest
                    
                    if growth_rate > 0.2:  # 20%增长
                        self._trigger_warning("memory_growth", {
                            'growth_rate': growth_rate,
                            'current_mb': memory_mb,
                            'trend': 'increasing'
                        })
                        
                        # 触发内存清理
                        if growth_rate > 0.5:  # 50%增长触发强制清理
                            self._trigger_cleanup('memory')
            
            # check内存限制
            if memory_mb > self._memory_limit_mb:
                self._trigger_warning("memory_limit", {
                    'current_mb': memory_mb,
                    'limit_mb': self._memory_limit_mb
                })
                self._trigger_cleanup('memory')
            
            # 定期垃圾回收
            current_time = time.time()
            if current_time - self._last_gc_time > self._gc_threshold:
                self._force_garbage_collection()
                self._last_gc_time = current_time
        
        except Exception as e:
            self.logger.error(f"内存checkfailed: {e}")
    
    def _check_cpu(self):
        """checkCPU使use"""
        try:
            cpu_percent = self._process.cpu_percent()
            
            self._cpu_history.append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent
            })
            
            # checkCPU持续高使use
            if len(self._cpu_history) >= 5:
                recent_cpu = [item['cpu_percent'] for item in list(self._cpu_history)[-5:]]
                avg_cpu = sum(recent_cpu) / len(recent_cpu)
                
                if avg_cpu > 80:  # 80% CPU使use
                    self._trigger_warning("high_cpu", {
                        'average_cpu': avg_cpu,
                        'current_cpu': cpu_percent
                    })
        
        except Exception as e:
            self.logger.error(f"CPUcheckfailed: {e}")
    
    def _check_tasks(self):
        """check异步任务"""
        try:
            active_tasks = len(self._tasks)
            self._resource_stats['tasks']['active'] = active_tasks
            self._resource_stats['tasks']['peak'] = max(
                self._resource_stats['tasks']['peak'],
                active_tasks
            )
            
            if active_tasks > 100:
                self._trigger_warning("too_many_tasks", {
                    'active_tasks': active_tasks,
                    'limit': 100
                })
                self._trigger_cleanup('tasks')
            
            # check僵尸任务
            zombie_tasks = []
            for task in list(self._tasks):
                if hasattr(task, 'done') and task.done():
                    zombie_tasks.append(task)
            
            if zombie_tasks:
                self.logger.debug(f"发现 {len(zombie_tasks)} 个completed任务")
        
        except Exception as e:
            self.logger.error(f"任务checkfailed: {e}")
    
    def _check_connections(self):
        """checkconnection"""
        try:
            active_connections = len(self._connections)
            self._resource_stats['connections']['active'] = active_connections
            
            if active_connections > 50:
                self._trigger_warning("too_many_connections", {
                    'active_connections': active_connections,
                    'limit': 50
                })
                self._trigger_cleanup('connections')
        
        except Exception as e:
            self.logger.error(f"connectioncheckfailed: {e}")
    
    def _check_threads(self):
        """check线程状态"""
        try:
            threads = threading.enumerate()
            thread_count = len(threads)
            
            self._resource_stats['threads']['active'] = thread_count
            
            if thread_count > 50:
                self._trigger_warning("too_many_threads", {
                    'thread_count': thread_count,
                    'limit': 50
                })
                
                # 列出线程详情
                thread_info = []
                for thread in threads:
                    thread_info.append({
                        'name': thread.name,
                        'alive': thread.is_alive(),
                        'daemon': thread.daemon
                    })
                
                self.logger.warning(f"当before线程数: {thread_count}")
                for info in thread_info[:10]:  # 只显示before10个
                    self.logger.debug(f"  - {info}")
        
        except Exception as e:
            self.logger.error(f"线程checkfailed: {e}")
    
    def _check_file_handles(self):
        """check文件句柄"""
        try:
            open_files = self._process.open_files()
            num_files = len(open_files)
            
            self._resource_stats['files']['active'] = num_files
            
            if num_files > 100:
                self._trigger_warning("too_many_files", {
                    'open_files': num_files,
                    'limit': 100
                })
                
                # 列出文件
                file_info = []
                for file in open_files:
                    file_info.append({
                        'path': file.path,
                        'fd': file.fd
                    })
                
                self.logger.warning(f"打开文件数: {num_files}")
                for info in file_info[:10]:
                    self.logger.debug(f"  - {info}")
        
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass  # 某些系统not允许访问文件句柄信息
        except Exception as e:
            self.logger.error(f"文件句柄checkfailed: {e}")
    
    def _check_disk_space(self):
        """check磁盘空间"""
        try:
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 1.0:  # 少at1GB
                self._trigger_warning("low_disk_space", {
                    'free_gb': free_gb,
                    'total_gb': disk_usage.total / (1024**3),
                    'used_percent': (disk_usage.used / disk_usage.total) * 100
                })
        
        except Exception as e:
            self.logger.error(f"磁盘空间checkfailed: {e}")
    
    def _trigger_warning(self, warning_type: str, data: Dict[str, Any]):
        """触发警告"""
        self._total_warnings += 1
        warning_msg = f"资源警告 [{warning_type}]: {data}"
        
        self.logger.warning(warning_msg)
        
        # 调use回调函数
        for callback in self._warning_callbacks:
            try:
                callback(warning_type, data)
            except Exception as e:
                self.logger.error(f"警告回调failed: {e}")
        
        # 记录to性能警告历史
        self._performance_alerts.append({
            'timestamp': time.time(),
            'type': warning_type,
            'data': data
        })
        
        # 保持警告历史大小
        if len(self._performance_alerts) > 100:
            self._performance_alerts.pop(0)
    
    def _trigger_cleanup(self, resource_type: str):
        """触发清理"""
        self._total_cleanups += 1
        
        if resource_type in self._cleanup_strategies:
            try:
                self.logger.info(f"触发{resource_type}资源清理...")
                self._cleanup_strategies[resource_type]()
            except Exception as e:
                self.logger.error(f"{resource_type}清理failed: {e}")
    
    def _cleanup_memory(self):
        """内存清理"""
        # 强制垃圾回收
        self._force_garbage_collection()
        
        # 清理弱引use集合
        self._tasks = weakref.WeakSet([t for t in self._tasks if not getattr(t, 'done', lambda: True)()])
        
        # 清理内存历史
        if len(self._memory_history) > 30:
            # 保留最近30个数据点
            recent = list(self._memory_history)[-30:]
            self._memory_history.clear()
            self._memory_history.extend(recent)
    
    def _cleanup_tasks(self):
        """任务清理"""
        cancelled_count = 0
        
        for task in list(self._tasks):
            if hasattr(task, 'done'):
                if task.done():
                    continue  # completed任务会be弱引use自动清理
                elif hasattr(task, 'cancel'):
                    # 取消未completed任务
                    try:
                        task.cancel()
                        cancelled_count += 1
                    except Exception:
                        pass
        
        if cancelled_count > 0:
            self.logger.info(f"取消 {cancelled_count} 个任务")
    
    def _cleanup_connections(self):
        """connection清理"""
        closed_count = 0
        
        for conn in list(self._connections):
            if hasattr(conn, 'close'):
                try:
                    conn.close()
                    closed_count += 1
                except Exception:
                    pass
        
        if closed_count > 0:
            self.logger.info(f"关闭 {closed_count} 个connection")
    
    def _cleanup_files(self):
        """文件清理"""
        closed_count = 0
        
        for file in list(self._files):
            if hasattr(file, 'close'):
                try:
                    file.close()
                    closed_count += 1
                except Exception:
                    pass
        
        if closed_count > 0:
            self.logger.info(f"关闭 {closed_count} 个文件")
    
    def _force_garbage_collection(self):
        """强制垃圾回收"""
        before = len(gc.get_objects())
        
        # 执行全面垃圾回收
        for _ in range(3):
            collected = gc.collect()
            if collected == 0:
                break
        
        after = len(gc.get_objects())
        
        self.logger.debug(f"垃圾回收: {before} -> {after} for象 (减少 {before - after})")
    
    def _update_statistics(self):
        """updates统计信息"""
        # updates峰值统计
        for resource_type in self._resource_stats:
            current = self._resource_stats[resource_type]['active']
            self._resource_stats[resource_type]['peak'] = max(
                self._resource_stats[resource_type]['peak'],
                current
            )
    
    # 资源注册方法
    def register_task(self, task: asyncio.Task, metadata: Optional[Dict] = None):
        """注册任务"""
        self._tasks.add(task)
        self._resource_stats['tasks']['created'] += 1
        self._resource_stats['tasks']['total_created'] += 1
        
        if self.enable_traceback:
            # 记录创建堆栈
            task._resource_traceback = traceback.format_stack()
    
    def register_connection(self, conn: Any, metadata: Optional[Dict] = None):
        """注册connection"""
        self._connections.add(conn)
        self._resource_stats['connections']['created'] += 1
        self._resource_stats['connections']['total_created'] += 1
    
    def register_file(self, file: Any, metadata: Optional[Dict] = None):
        """注册文件"""
        self._files.add(file)
        self._resource_stats['files']['created'] += 1
        self._resource_stats['files']['total_created'] += 1
    
    def register_handler(self, handler_id: str, handler: Any):
        """注册处理器"""
        self._handlers[handler_id] = handler
        self._resource_stats['handlers']['created'] += 1
        self._resource_stats['handlers']['total_created'] += 1
    
    # retrieval统计信息
    def get_stats(self) -> Dict[str, Any]:
        """retrieval统计信息"""
        try:
            memory_info = self._process.memory_info()
            cpu_percent = self._process.cpu_percent()
            
            return {
                'uptime_seconds': time.time() - self._start_time,
                'memory': {
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024,
                    'percent': self._process.memory_percent(),
                    'limit_mb': self._memory_limit_mb
                },
                'cpu': {
                    'percent': cpu_percent,
                    'average_5min': self._get_average_cpu(300)  # 5分钟平均
                },
                'resources': {
                    'tasks': {
                        'active': len(self._tasks),
                        **self._resource_stats['tasks']
                    },
                    'connections': {
                        'active': len(self._connections),
                        **self._resource_stats['connections']
                    },
                    'files': {
                        'active': len(self._files),
                        **self._resource_stats['files']
                    },
                    'handlers': {
                        'active': len(self._handlers),
                        **self._resource_stats['handlers']
                    },
                    'threads': {
                        'active': threading.active_count(),
                        **self._resource_stats['threads']
                    }
                },
                'monitoring': {
                    'total_warnings': self._total_warnings,
                    'total_cleanups': self._total_cleanups,
                    'recent_alerts': len([a for a in self._performance_alerts 
                                        if time.time() - a['timestamp'] < 3600])  # 1小when内
                }
            }
        except Exception as e:
            self.logger.error(f"retrieval统计信息failed: {e}")
            return {}
    
    def _get_average_cpu(self, seconds: int) -> float:
        """retrieval指定when间内平均CPU使use率"""
        current_time = time.time()
        recent_cpu = [
            item['cpu_percent'] for item in self._cpu_history
            if current_time - item['timestamp'] <= seconds
        ]
        
        return sum(recent_cpu) / len(recent_cpu) if recent_cpu else 0.0
    
    def get_recent_alerts(self, hours: int = 1) -> List[Dict]:
        """retrieval最近警告"""
        cutoff_time = time.time() - (hours * 3600)
        return [
            alert for alert in self._performance_alerts
            if alert['timestamp'] >= cutoff_time
        ]
    
    def force_cleanup_all(self):
        """强制清理所has资源"""
        self.logger.info("starting强制清理所has资源...")
        
        for resource_type in self._cleanup_strategies:
            self._trigger_cleanup(resource_type)
        
        # 强制垃圾回收
        self._force_garbage_collection()
        
        self.logger.info("强制清理completed")


# 全局资源监控器
_global_resource_monitor: Optional[ResourceMonitor] = None

def get_resource_monitor() -> ResourceMonitor:
    """retrieval全局资源监控器"""
    global _global_resource_monitor
    if _global_resource_monitor is None:
        _global_resource_monitor = ResourceMonitor()
    return _global_resource_monitor

def start_global_monitoring(interval: float = 30.0):
    """start全局资源监控"""
    monitor = get_resource_monitor()
    monitor.start_monitoring(interval)

def stop_global_monitoring():
    """停止全局资源监控"""
    global _global_resource_monitor
    if _global_resource_monitor:
        _global_resource_monitor.stop_monitoring()

def register_task(task: asyncio.Task):
    """注册任务to全局监控"""
    monitor = get_resource_monitor()
    monitor.register_task(task)

def register_connection(conn: Any):
    """注册connectionto全局监控"""
    monitor = get_resource_monitor()
    monitor.register_connection(conn)
