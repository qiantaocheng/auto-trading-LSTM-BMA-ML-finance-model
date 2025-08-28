#!/usr/bin/env python3
"""
=============================================================================
统一监控系统 - 整合版本
=============================================================================
整合以下监控功能:
- 实时系统监控 (原 realtime_monitoring_system.py)
- 性能监控 (整合自 performance_monitor.py) 
- 资源监控 (整合自 resource_monitor.py)
整合时间: 2025-08-20
=============================================================================

📈 实现关键交易指标的实时监控，包括：
- 订单延迟监控 (order_latency)
- 资金利用率监控 (capital_utilization)  
- 数据新鲜度监控 (data_freshness)
- 拒单率监控 (rejection_rate)
- 胜率监控 (win_rate)
- PnL跟踪 (pnl_tracking)
- 系统性能监控
- 资源使用监控
"""

import time
import logging
import sqlite3
import json
import threading
import asyncio
import psutil
import queue
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)

# =============================================================================
# 基础数据结构和枚举
# =============================================================================

class AlertLevel(Enum):
    """告警级别"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MetricType(Enum):
    """指标类型"""
    GAUGE = "GAUGE"          # 瞬时值
    COUNTER = "COUNTER"      # 累计计数
    HISTOGRAM = "HISTOGRAM"  # 分布统计
    RATE = "RATE"           # 速率

@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    """告警信息"""
    level: AlertLevel
    message: str
    metric_name: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# 性能监控 (从 performance_monitor.py 整合)
# =============================================================================

@dataclass
class PerformanceMetric:
    """性能指标"""
    operation: str
    duration_ms: float
    timestamp: float
    success: bool
    details: Dict = None

class TradingPerformanceMonitor:
    """交易性能监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger("PerformanceMonitor")
        self.metrics = deque(maxlen=1000)  # 保留最近1000条记录
        self.alerts = []
        
        # 性能阈值
        self.thresholds = {
            "order_validation": 100,  # 验证应在100ms内完成
            "order_submission": 500,  # 订单提交应在500ms内完成
            "price_fetch": 200,       # 价格获取应在200ms内完成
            "connection": 5000,       # 连接应在5秒内完成
        }
    
    async def monitor_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """监控操作执行时间"""
        start_time = time.time()
        success = True
        error = None
        
        try:
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # 记录性能指标
            metric = PerformanceMetric(
                operation=operation_name,
                duration_ms=duration_ms,
                timestamp=end_time,
                success=success,
                details={"error": error} if error else None
            )
            
            self.metrics.append(metric)
            
            # 检查性能阈值
            self._check_performance_threshold(metric)
    
    def _check_performance_threshold(self, metric: PerformanceMetric):
        """检查性能阈值"""
        threshold = self.thresholds.get(metric.operation)
        if threshold and metric.duration_ms > threshold:
            alert = f"性能警告: {metric.operation} 耗时 {metric.duration_ms:.1f}ms (阈值: {threshold}ms)"
            self.logger.warning(alert)
            self.alerts.append({
                "timestamp": metric.timestamp,
                "operation": metric.operation,
                "duration": metric.duration_ms,
                "threshold": threshold,
                "alert": alert
            })
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        if not self.metrics:
            return {"message": "暂无性能数据"}
        
        # 按操作类型分组
        by_operation = defaultdict(list)
        for metric in self.metrics:
            by_operation[metric.operation].append(metric.duration_ms)
        
        summary = {}
        for operation, durations in by_operation.items():
            summary[operation] = {
                "count": len(durations),
                "avg_ms": sum(durations) / len(durations),
                "max_ms": max(durations),
                "min_ms": min(durations),
                "success_rate": sum(1 for m in self.metrics if m.operation == operation and m.success) / len(durations)
            }
        
        return {
            "summary": summary,
            "total_operations": len(self.metrics),
            "recent_alerts": len([a for a in self.alerts if time.time() - a["timestamp"] < 300])  # 最近5分钟的警告
        }

# =============================================================================
# 资源监控 (从 resource_monitor.py 整合)
# =============================================================================

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
            
    def remove_alert_callback(self, callback: Callable[[list, ResourceMetrics], None]):
        """移除警报回调"""
        try:
            self.alert_callbacks.remove(callback)
        except ValueError:
            logger.warning("尝试移除不存在的回调函数")
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """获取当前资源指标"""
        return self.current_metrics
        
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

# =============================================================================
# 统一监控系统主类 (基于原 realtime_monitoring_system.py)
# =============================================================================

@dataclass 
class TradingMetrics:
    """交易指标"""
    timestamp: float
    
    # 订单相关指标
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    orders_canceled: int = 0
    
    # 延迟指标
    avg_order_latency_ms: float = 0.0
    max_order_latency_ms: float = 0.0
    
    # 资金相关
    capital_utilization: float = 0.0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # 数据质量
    data_freshness_score: float = 1.0
    price_update_delays: List[float] = field(default_factory=list)
    
    # 连接状态
    connection_status: str = "disconnected"
    reconnection_count: int = 0

@dataclass
class MonitoringConfig:
    """监控配置"""
    db_path: str = "monitoring.db"
    metrics_retention_hours: int = 24
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "order_latency_ms": 1000.0,
        "rejection_rate": 0.05,
        "data_freshness": 0.8,
        "capital_utilization": 0.95,
        "memory_usage": 85.0,
        "cpu_usage": 90.0
    })
    enable_db_storage: bool = True
    enable_alerts: bool = True

class UnifiedMonitoringSystem:
    """统一监控系统 - 整合版本"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger("UnifiedMonitoringSystem")
        
        # 监控组件
        self.performance_monitor = TradingPerformanceMonitor()
        self.resource_monitor = ResourceMonitor()
        
        # 指标存储
        self.current_metrics = TradingMetrics(timestamp=time.time())
        self.metrics_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        
        # 数据库
        self.db_connection: Optional[sqlite3.Connection] = None
        if self.config.enable_db_storage:
            self._init_database()
            
        # 线程安全
        self.metrics_lock = threading.Lock()
        
        self.logger.info("统一监控系统初始化完成")
    
    def _init_database(self):
        """初始化数据库"""
        try:
            self.db_connection = sqlite3.connect(
                self.config.db_path,
                check_same_thread=False
            )
            
            # 创建表结构
            self.db_connection.execute('''
                CREATE TABLE IF NOT EXISTS trading_metrics (
                    timestamp REAL PRIMARY KEY,
                    orders_submitted INTEGER,
                    orders_filled INTEGER,
                    orders_rejected INTEGER,
                    avg_order_latency_ms REAL,
                    capital_utilization REAL,
                    total_pnl REAL,
                    data_freshness_score REAL,
                    connection_status TEXT
                )
            ''')
            
            self.db_connection.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    timestamp REAL,
                    level TEXT,
                    metric_name TEXT,
                    message TEXT,
                    details TEXT
                )
            ''')
            
            self.db_connection.commit()
            self.logger.info(f"监控数据库已初始化: {self.config.db_path}")
            
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            self.db_connection = None
    
    def start_monitoring(self):
        """启动所有监控组件"""
        self.logger.info("启动统一监控系统...")
        
        # 启动资源监控
        self.resource_monitor.start_monitoring()
        
        # 添加资源监控告警回调
        self.resource_monitor.add_alert_callback(self._handle_resource_alerts)
        
        self.logger.info("统一监控系统启动完成")
    
    def stop_monitoring(self):
        """停止所有监控组件"""
        self.logger.info("停止统一监控系统...")
        
        # 停止资源监控
        self.resource_monitor.stop_monitoring()
        
        # 关闭数据库连接
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None
            
        self.logger.info("统一监控系统停止完成")
    
    def _handle_resource_alerts(self, alerts: List[str], metrics: ResourceMetrics):
        """处理资源告警"""
        for alert_msg in alerts:
            alert = Alert(
                level=AlertLevel.WARNING,
                message=alert_msg,
                metric_name="resource_usage",
                timestamp=time.time(),
                details={"metrics": metrics}
            )
            self._add_alert(alert)
    
    def record_order_event(self, event_type: str, latency_ms: float = 0.0):
        """记录订单事件"""
        with self.metrics_lock:
            if event_type == "submitted":
                self.current_metrics.orders_submitted += 1
            elif event_type == "filled":
                self.current_metrics.orders_filled += 1
            elif event_type == "rejected":
                self.current_metrics.orders_rejected += 1
            elif event_type == "canceled":
                self.current_metrics.orders_canceled += 1
            
            # 更新延迟指标
            if latency_ms > 0:
                if self.current_metrics.avg_order_latency_ms == 0:
                    self.current_metrics.avg_order_latency_ms = latency_ms
                else:
                    # 简单移动平均
                    self.current_metrics.avg_order_latency_ms = (
                        self.current_metrics.avg_order_latency_ms * 0.8 + latency_ms * 0.2
                    )
                
                self.current_metrics.max_order_latency_ms = max(
                    self.current_metrics.max_order_latency_ms, latency_ms
                )
                
                # 检查延迟告警
                if latency_ms > self.config.alert_thresholds["order_latency_ms"]:
                    alert = Alert(
                        level=AlertLevel.WARNING,
                        message=f"订单延迟过高: {latency_ms:.1f}ms",
                        metric_name="order_latency",
                        timestamp=time.time(),
                        details={"latency_ms": latency_ms}
                    )
                    self._add_alert(alert)
    
    def update_capital_metrics(self, total_pnl: float, unrealized_pnl: float, 
                              capital_utilization: float):
        """更新资金指标"""
        with self.metrics_lock:
            self.current_metrics.total_pnl = total_pnl
            self.current_metrics.unrealized_pnl = unrealized_pnl
            self.current_metrics.capital_utilization = capital_utilization
            
            # 检查资金利用率告警
            if capital_utilization > self.config.alert_thresholds["capital_utilization"]:
                alert = Alert(
                    level=AlertLevel.WARNING,
                    message=f"资金利用率过高: {capital_utilization:.1%}",
                    metric_name="capital_utilization",
                    timestamp=time.time(),
                    details={"utilization": capital_utilization}
                )
                self._add_alert(alert)
    
    def update_data_quality(self, freshness_score: float):
        """更新数据质量指标"""
        with self.metrics_lock:
            self.current_metrics.data_freshness_score = freshness_score
            
            # 检查数据新鲜度告警
            if freshness_score < self.config.alert_thresholds["data_freshness"]:
                alert = Alert(
                    level=AlertLevel.WARNING,
                    message=f"数据新鲜度低: {freshness_score:.2f}",
                    metric_name="data_freshness",
                    timestamp=time.time(),
                    details={"freshness_score": freshness_score}
                )
                self._add_alert(alert)
    
    def _add_alert(self, alert: Alert):
        """添加告警"""
        self.alerts.append(alert)
        
        # 记录日志
        if alert.level == AlertLevel.CRITICAL:
            self.logger.critical(alert.message)
        elif alert.level == AlertLevel.ERROR:
            self.logger.error(alert.message)
        elif alert.level == AlertLevel.WARNING:
            self.logger.warning(alert.message)
        else:
            self.logger.info(alert.message)
        
        # 存储到数据库
        if self.db_connection and self.config.enable_db_storage:
            try:
                self.db_connection.execute('''
                    INSERT INTO alerts (timestamp, level, metric_name, message, details)
                    VALUES (?, ?, ?, ?, ?)
                ''', (alert.timestamp, alert.level.value, alert.metric_name, 
                      alert.message, json.dumps(alert.details)))
                self.db_connection.commit()
            except Exception as e:
                self.logger.error(f"告警存储失败: {e}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        with self.metrics_lock:
            current = self.current_metrics
            
        # 计算胜率
        total_orders = current.orders_filled + current.orders_rejected
        win_rate = current.orders_filled / total_orders if total_orders > 0 else 0.0
        
        # 计算拒单率
        rejection_rate = current.orders_rejected / total_orders if total_orders > 0 else 0.0
        
        summary = {
            "timestamp": current.timestamp,
            "trading_metrics": {
                "orders_submitted": current.orders_submitted,
                "orders_filled": current.orders_filled,
                "orders_rejected": current.orders_rejected,
                "win_rate": f"{win_rate:.2%}",
                "rejection_rate": f"{rejection_rate:.2%}",
                "avg_order_latency_ms": f"{current.avg_order_latency_ms:.1f}ms",
                "max_order_latency_ms": f"{current.max_order_latency_ms:.1f}ms"
            },
            "financial_metrics": {
                "total_pnl": f"${current.total_pnl:.2f}",
                "unrealized_pnl": f"${current.unrealized_pnl:.2f}",
                "capital_utilization": f"{current.capital_utilization:.1%}"
            },
            "system_metrics": {
                "data_freshness_score": f"{current.data_freshness_score:.2f}",
                "connection_status": current.connection_status
            },
            "performance_metrics": self.performance_monitor.get_performance_summary(),
            "resource_metrics": self.resource_monitor.get_resource_summary(),
            "recent_alerts": len([a for a in self.alerts 
                                if time.time() - a.timestamp < 300])  # 最近5分钟
        }
        
        return summary
    
    def get_recent_alerts(self, minutes: int = 30) -> List[Dict]:
        """获取最近的告警"""
        cutoff_time = time.time() - (minutes * 60)
        recent_alerts = [
            {
                "timestamp": alert.timestamp,
                "level": alert.level.value,
                "metric_name": alert.metric_name,
                "message": alert.message,
                "details": alert.details
            }
            for alert in self.alerts
            if alert.timestamp > cutoff_time
        ]
        
        return sorted(recent_alerts, key=lambda x: x["timestamp"], reverse=True)
    
    # 性能监控接口代理
    async def monitor_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """监控操作执行时间 - 代理到性能监控器"""
        return await self.performance_monitor.monitor_operation(
            operation_name, operation_func, *args, **kwargs
        )

# =============================================================================
# 全局实例和工厂函数
# =============================================================================

# 全局统一监控系统实例
_unified_monitoring_system = None
_performance_monitor = None
_resource_monitor = None

def get_unified_monitoring_system(config: Optional[MonitoringConfig] = None) -> UnifiedMonitoringSystem:
    """获取统一监控系统单例"""
    global _unified_monitoring_system
    if _unified_monitoring_system is None:
        _unified_monitoring_system = UnifiedMonitoringSystem(config)
    return _unified_monitoring_system

def get_performance_monitor() -> TradingPerformanceMonitor:
    """获取性能监控器单例 - 向后兼容"""
    global _performance_monitor
    if _performance_monitor is None:
        monitoring_system = get_unified_monitoring_system()
        _performance_monitor = monitoring_system.performance_monitor
    return _performance_monitor

def get_resource_monitor() -> ResourceMonitor:
    """获取资源监控器单例 - 向后兼容"""
    global _resource_monitor
    if _resource_monitor is None:
        monitoring_system = get_unified_monitoring_system()
        _resource_monitor = monitoring_system.resource_monitor
    return _resource_monitor

def init_unified_monitoring(config: Optional[MonitoringConfig] = None) -> UnifiedMonitoringSystem:
    """初始化并启动统一监控系统"""
    monitoring_system = get_unified_monitoring_system(config)
    monitoring_system.start_monitoring()
    return monitoring_system

def cleanup_unified_monitoring():
    """清理统一监控系统"""
    global _unified_monitoring_system, _performance_monitor, _resource_monitor
    
    if _unified_monitoring_system:
        _unified_monitoring_system.stop_monitoring()
        _unified_monitoring_system = None
        _performance_monitor = None
        _resource_monitor = None

# 向后兼容的初始化函数
def init_resource_monitor() -> ResourceMonitor:
    """初始化并启动资源监控器 - 向后兼容"""
    return get_resource_monitor()

def cleanup_resource_monitor():
    """清理资源监控器 - 向后兼容"""
    # 统一监控系统会处理清理
    pass