#!/usr/bin/env python3
"""
资源监控器
监控系统资源使用情况，支持内存、CPU等监控
"""

import psutil
import logging
import threading
import time
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass
import queue

logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """资源指标"""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    memory_used_gb: float
    disk_usage_percent: float
    timestamp: float

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, 
                 check_interval: float = 5.0,
                 memory_threshold_percent: float = 85.0,
                 cpu_threshold_percent: float = 90.0):
        self.check_interval = check_interval
        self.memory_threshold = memory_threshold_percent
        self.cpu_threshold = cpu_threshold_percent
        
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_queue = queue.Queue(maxsize=100)
        self.alert_callbacks: list[Callable] = []
        
        self.current_metrics: Optional[ResourceMetrics] = None
        
    def start_monitoring(self):
        """开始资源监控"""
        if self.is_monitoring:
            logger.warning("资源监控已经在运行")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("资源监控器启动")
        
    def stop_monitoring(self):
        """停止资源监控"""
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10.0)
            if self.monitor_thread.is_alive():
                logger.warning("监控线程未能在超时时间内停止")
        
        # 清理资源
        self._cleanup_resources()
        
        logger.info("资源监控器停止")
    
    def _cleanup_resources(self):
        """清理资源"""
        # 清理队列
        while not self.metrics_queue.empty():
            try:
                self.metrics_queue.get_nowait()
            except queue.Empty:
                break
        
        # 清理回调（可选，根据需要）
        # self.clear_all_callbacks()
        
        # 重置当前指标
        self.current_metrics = None
        
        logger.debug("资源监控器资源清理完成")
    
    def _add_to_queue_safely(self, metrics: ResourceMetrics):
        """安全地添加指标到队列"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self.metrics_queue.put_nowait(metrics)
                return  # 成功添加，返回
            except queue.Full:
                # 队列满了，尝试移除最老的数据
                try:
                    self.metrics_queue.get_nowait()
                    # 继续尝试添加
                except queue.Empty:
                    # 队列在get_nowait()时变空了，直接添加
                    try:
                        self.metrics_queue.put_nowait(metrics)
                        return
                    except queue.Full:
                        # 仍然满，继续下一次尝试
                        continue
        
        # 如果所有尝试都失败，记录警告
        logger.warning("无法将指标添加到队列，队列操作失败")
        
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.current_metrics = metrics
                
                # 添加到队列，使用原子操作避免竞态条件
                self._add_to_queue_safely(metrics)
                        
                # 检查阈值
                self._check_thresholds(metrics)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"资源监控错误: {e}")
                time.sleep(self.check_interval)
                
    def _collect_metrics(self) -> ResourceMetrics:
        """收集资源指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1.0)
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_gb = memory.available / (1024**3)
        memory_used_gb = memory.used / (1024**3)
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_usage_percent = disk.percent
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_gb=memory_available_gb,
            memory_used_gb=memory_used_gb,
            disk_usage_percent=disk_usage_percent,
            timestamp=time.time()
        )
        
    def _check_thresholds(self, metrics: ResourceMetrics):
        """检查阈值并触发警报"""
        alerts = []
        
        if metrics.memory_percent > self.memory_threshold:
            alerts.append(f"内存使用率过高: {metrics.memory_percent:.1f}%")
            
        if metrics.cpu_percent > self.cpu_threshold:
            alerts.append(f"CPU使用率过高: {metrics.cpu_percent:.1f}%")
            
        if alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alerts, metrics)
                except Exception as e:
                    logger.error(f"警报回调失败: {e}")
                    
    def add_alert_callback(self, callback: Callable[[list, ResourceMetrics], None]):
        """添加警报回调"""
        if callback not in self.alert_callbacks:
            self.alert_callbacks.append(callback)
            logger.debug(f"Added alert callback: {callback.__name__ if hasattr(callback, '__name__') else str(callback)}")
        
    def remove_alert_callback(self, callback: Callable[[list, ResourceMetrics], None]):
        """移除警报回调"""
        try:
            self.alert_callbacks.remove(callback)
            logger.debug(f"Removed alert callback: {callback.__name__ if hasattr(callback, '__name__') else str(callback)}")
        except ValueError:
            logger.warning("尝试移除不存在的回调函数")
    
    def clear_all_callbacks(self):
        """清除所有回调函数"""
        callback_count = len(self.alert_callbacks)
        self.alert_callbacks.clear()
        logger.info(f"清除了 {callback_count} 个回调函数")
        
    def add_warning_callback(self, callback: Callable[[list, ResourceMetrics], None]):
        """添加警告回调（兼容性方法）"""
        self.add_alert_callback(callback)
        
    def remove_warning_callback(self, callback: Callable[[list, ResourceMetrics], None]):
        """移除警告回调（兼容性方法）"""
        self.remove_alert_callback(callback)
        
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """获取当前资源指标"""
        return self.current_metrics
        
    def get_metrics_history(self, max_count: int = 50) -> list[ResourceMetrics]:
        """获取历史指标"""
        history = []
        temp_queue = queue.Queue()
        
        # 从队列中取出数据
        while not self.metrics_queue.empty() and len(history) < max_count:
            try:
                metric = self.metrics_queue.get_nowait()
                history.append(metric)
                temp_queue.put(metric)
            except queue.Empty:
                break
                
        # 将数据放回队列
        while not temp_queue.empty():
            try:
                self.metrics_queue.put_nowait(temp_queue.get_nowait())
            except queue.Full:
                break
                
        return history[-max_count:] if history else []
        
    def get_resource_summary(self) -> Dict[str, Any]:
        """获取资源使用摘要"""
        if not self.current_metrics:
            return {"status": "监控未启动"}
            
        metrics = self.current_metrics
        
        return {
            "cpu_usage": f"{metrics.cpu_percent:.1f}%",
            "memory_usage": f"{metrics.memory_percent:.1f}%",
            "memory_available": f"{metrics.memory_available_gb:.2f}GB",
            "memory_used": f"{metrics.memory_used_gb:.2f}GB",
            "disk_usage": f"{metrics.disk_usage_percent:.1f}%",
            "status": "正常" if metrics.memory_percent < self.memory_threshold and metrics.cpu_percent < self.cpu_threshold else "警告",
            "last_update": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(metrics.timestamp))
        }

# 全局资源监控器实例
_global_resource_monitor: Optional[ResourceMonitor] = None

def get_resource_monitor() -> ResourceMonitor:
    """获取全局资源监控器实例"""
    global _global_resource_monitor
    
    if _global_resource_monitor is None:
        _global_resource_monitor = ResourceMonitor()
        
    return _global_resource_monitor

def init_resource_monitor() -> ResourceMonitor:
    """初始化并启动资源监控器"""
    monitor = get_resource_monitor()
    if not monitor.is_monitoring:
        monitor.start_monitoring()
    return monitor

def cleanup_resource_monitor():
    """清理资源监控器"""
    global _global_resource_monitor
    
    if _global_resource_monitor and _global_resource_monitor.is_monitoring:
        _global_resource_monitor.stop_monitoring()
        _global_resource_monitor = None