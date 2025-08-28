#!/usr/bin/env python3
"""
增强监控告警系统 - 提供全面的系统可观测性
"""

import logging
import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"      # 计数器
    GAUGE = "gauge"          # 仪表盘
    HISTOGRAM = "histogram"   # 直方图
    TIMER = "timer"          # 计时器

@dataclass
class Alert:
    """告警数据结构"""
    id: str
    title: str
    message: str
    level: AlertLevel
    source: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    resolved: bool = False
    resolved_at: Optional[float] = None

@dataclass 
class Metric:
    """指标数据结构"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Optional[Dict[str, str]] = None
    description: Optional[str] = None

class PerformanceStats(NamedTuple):
    """性能统计"""
    total_requests: int
    success_rate: float
    avg_response_time: float
    error_count: int
    last_error_time: Optional[float]

class EnhancedMonitor:
    """增强监控器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 告警管理
        self._alerts: deque = deque(maxlen=1000)
        self._alert_history: List[Alert] = []
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._alert_cooldowns: Dict[str, float] = {}
        
        # 指标管理
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._metric_callbacks: List[Callable[[Metric], None]] = []
        
        # 性能监控
        self._performance_stats: Dict[str, PerformanceStats] = {}
        self._request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
        # 系统健康状态
        self._health_checks: Dict[str, Callable[[], bool]] = {}
        self._health_status: Dict[str, bool] = {}
        
        # 锁（优先初始化）
        self._lock = threading.RLock()
        
        # 监控配置
        self.alert_cooldown_seconds = self.config.get('alert_cooldown_seconds', 300)
        self.metric_retention_seconds = self.config.get('metric_retention_seconds', 3600)
        self.enable_file_logging = self.config.get('enable_file_logging', True)
        self.log_directory = Path(self.config.get('log_directory', 'logs'))
        
        # 创建日志目录
        if self.enable_file_logging:
            self.log_directory.mkdir(exist_ok=True)
            self._setup_file_logging()
        
        # 启动后台清理任务
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        logger.info("增强监控系统已启动")
    
    def _setup_file_logging(self):
        """设置文件日志"""
        try:
            # 创建专门的监控日志处理器
            monitor_log_file = self.log_directory / f"monitor_{datetime.now().strftime('%Y%m%d')}.log"
            monitor_handler = logging.FileHandler(monitor_log_file, encoding='utf-8')
            monitor_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - [MONITOR] - %(message)s')
            )
            
            # 创建专门的监控日志器
            self.monitor_logger = logging.getLogger('enhanced_monitor')
            self.monitor_logger.addHandler(monitor_handler)
            self.monitor_logger.setLevel(logging.DEBUG)
            
        except Exception as e:
            logger.error(f"设置监控文件日志失败: {e}")
    
    def emit_alert(self, title: str, message: str, level: AlertLevel, 
                  source: str = "system", metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        发送告警
        
        Args:
            title: 告警标题
            message: 告警消息  
            level: 告警级别
            source: 告警源
            metadata: 附加元数据
            
        Returns:
            告警ID
        """
        # 生成告警ID
        alert_id = f"{source}_{title}_{int(time.time())}"
        
        # 检查冷却期
        cooldown_key = f"{source}_{title}"
        current_time = time.time()
        
        if cooldown_key in self._alert_cooldowns:
            if current_time - self._alert_cooldowns[cooldown_key] < self.alert_cooldown_seconds:
                logger.debug(f"告警在冷却期内，跳过: {title}")
                return alert_id
        
        # 创建告警
        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            level=level,
            source=source,
            timestamp=current_time,
            metadata=metadata
        )
        
        with self._lock:
            self._alerts.append(alert)
            self._alert_history.append(alert)
            self._alert_cooldowns[cooldown_key] = current_time
        
        # 记录到日志
        log_message = f"[{source.upper()}] {title}: {message}"
        if level == AlertLevel.CRITICAL:
            logger.critical(log_message)
        elif level == AlertLevel.ERROR:
            logger.error(log_message)
        elif level == AlertLevel.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # 记录到监控日志
        if hasattr(self, 'monitor_logger'):
            self.monitor_logger.info(f"ALERT|{level.value}|{source}|{title}|{message}")
        
        # 调用告警回调
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
        
        return alert_id
    
    def record_metric(self, name: str, value: float, metric_type: MetricType,
                     labels: Optional[Dict[str, str]] = None, 
                     description: Optional[str] = None) -> None:
        """
        记录指标
        
        Args:
            name: 指标名称
            value: 指标值
            metric_type: 指标类型
            labels: 标签
            description: 描述
        """
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            labels=labels,
            description=description
        )
        
        with self._lock:
            self._metrics[name].append(metric)
        
        # 记录到监控日志
        if hasattr(self, 'monitor_logger'):
            labels_str = json.dumps(labels) if labels else "{}"
            self.monitor_logger.debug(f"METRIC|{name}|{value}|{metric_type.value}|{labels_str}")
        
        # 调用指标回调
        for callback in self._metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"指标回调执行失败: {e}")
    
    def track_performance(self, operation: str, duration: float, success: bool = True) -> None:
        """
        跟踪操作性能
        
        Args:
            operation: 操作名称
            duration: 执行时长（秒）
            success: 是否成功
        """
        current_time = time.time()
        
        with self._lock:
            # 更新请求时间历史
            self._request_times[operation].append(duration)
            
            # 计算统计信息
            times = list(self._request_times[operation])
            total_requests = len(times)
            avg_response_time = sum(times) / total_requests if total_requests > 0 else 0
            
            # 获取现有统计或创建新的
            if operation in self._performance_stats:
                old_stats = self._performance_stats[operation]
                success_count = old_stats.total_requests * old_stats.success_rate
                if success:
                    success_count += 1
                error_count = old_stats.error_count if success else old_stats.error_count + 1
                last_error_time = current_time if not success else old_stats.last_error_time
            else:
                success_count = 1 if success else 0
                error_count = 0 if success else 1
                last_error_time = current_time if not success else None
            
            # 更新统计
            self._performance_stats[operation] = PerformanceStats(
                total_requests=total_requests,
                success_rate=success_count / total_requests if total_requests > 0 else 0,
                avg_response_time=avg_response_time,
                error_count=error_count,
                last_error_time=last_error_time
            )
        
        # 记录指标
        self.record_metric(f"{operation}_duration", duration, MetricType.TIMER)
        self.record_metric(f"{operation}_success", 1 if success else 0, MetricType.COUNTER)
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """
        注册健康检查
        
        Args:
            name: 检查名称
            check_func: 检查函数，返回True表示健康
        """
        self._health_checks[name] = check_func
        logger.debug(f"已注册健康检查: {name}")
    
    def run_health_checks(self) -> Dict[str, bool]:
        """
        运行所有健康检查
        
        Returns:
            健康检查结果字典
        """
        results = {}
        
        for name, check_func in self._health_checks.items():
            try:
                is_healthy = check_func()
                results[name] = is_healthy
                
                # 如果状态发生变化，发送告警
                if name in self._health_status and self._health_status[name] != is_healthy:
                    if is_healthy:
                        self.emit_alert(
                            f"{name}健康检查恢复",
                            f"{name}组件已恢复正常",
                            AlertLevel.INFO,
                            "health_check"
                        )
                    else:
                        self.emit_alert(
                            f"{name}健康检查失败",
                            f"{name}组件检查失败",
                            AlertLevel.ERROR,
                            "health_check"
                        )
                
                self._health_status[name] = is_healthy
                
            except Exception as e:
                logger.error(f"健康检查执行失败 {name}: {e}")
                results[name] = False
                self._health_status[name] = False
        
        return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        获取系统整体健康状态
        
        Returns:
            系统健康状态信息
        """
        health_results = self.run_health_checks()
        
        total_checks = len(health_results)
        healthy_checks = sum(1 for is_healthy in health_results.values() if is_healthy)
        
        overall_health = healthy_checks == total_checks if total_checks > 0 else True
        health_score = healthy_checks / total_checks if total_checks > 0 else 1.0
        
        return {
            'overall_healthy': overall_health,
            'health_score': health_score,
            'total_checks': total_checks,
            'healthy_checks': healthy_checks,
            'check_results': health_results,
            'last_check_time': time.time()
        }
    
    def get_alerts(self, level: Optional[AlertLevel] = None, 
                  since: Optional[float] = None, limit: int = 50) -> List[Alert]:
        """
        获取告警列表
        
        Args:
            level: 过滤告警级别
            since: 时间戳过滤
            limit: 返回数量限制
            
        Returns:
            告警列表
        """
        alerts = list(self._alerts)
        
        # 过滤
        if level:
            alerts = [a for a in alerts if a.level == level]
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        # 排序并限制数量
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return alerts[:limit]
    
    def get_metrics(self, name: Optional[str] = None, 
                   since: Optional[float] = None) -> Dict[str, List[Metric]]:
        """
        获取指标数据
        
        Args:
            name: 指标名称过滤
            since: 时间戳过滤
            
        Returns:
            指标数据字典
        """
        result = {}
        
        metrics_to_get = [name] if name else self._metrics.keys()
        
        for metric_name in metrics_to_get:
            if metric_name in self._metrics:
                metrics = list(self._metrics[metric_name])
                if since:
                    metrics = [m for m in metrics if m.timestamp >= since]
                result[metric_name] = metrics
        
        return result
    
    def get_performance_stats(self) -> Dict[str, PerformanceStats]:
        """获取性能统计信息"""
        with self._lock:
            return self._performance_stats.copy()
    
    def export_monitoring_data(self, file_path: str) -> None:
        """
        导出监控数据到文件
        
        Args:
            file_path: 导出文件路径
        """
        try:
            export_data = {
                'timestamp': time.time(),
                'alerts': [asdict(alert) for alert in self.get_alerts(limit=100)],
                'metrics': {name: [asdict(m) for m in metrics[-20:]] 
                          for name, metrics in self._metrics.items()},
                'performance': {name: asdict(stats) 
                              for name, stats in self._performance_stats.items()},
                'health': self.get_system_health()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"监控数据已导出到: {file_path}")
            
        except Exception as e:
            logger.error(f"导出监控数据失败: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """添加告警回调函数"""
        self._alert_callbacks.append(callback)
    
    def add_metric_callback(self, callback: Callable[[Metric], None]) -> None:
        """添加指标回调函数"""
        self._metric_callbacks.append(callback)
    
    def _cleanup_worker(self) -> None:
        """后台清理工作线程"""
        while True:
            try:
                current_time = time.time()
                cutoff_time = current_time - self.metric_retention_seconds
                
                # 清理过期指标
                with self._lock:
                    for name, metrics in self._metrics.items():
                        # 转换为列表以避免在迭代时修改
                        metrics_list = list(metrics)
                        metrics.clear()
                        
                        # 只保留未过期的指标
                        for metric in metrics_list:
                            if metric.timestamp > cutoff_time:
                                metrics.append(metric)
                
                # 清理告警冷却
                expired_cooldowns = [
                    key for key, timestamp in self._alert_cooldowns.items()
                    if current_time - timestamp > self.alert_cooldown_seconds * 2
                ]
                for key in expired_cooldowns:
                    del self._alert_cooldowns[key]
                
                # 每分钟清理一次
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"监控清理任务失败: {e}")
                time.sleep(60)

# 全局监控器实例
_global_monitor: Optional[EnhancedMonitor] = None

def get_enhanced_monitor(config: Optional[Dict[str, Any]] = None) -> EnhancedMonitor:
    """获取全局监控器实例"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = EnhancedMonitor(config)
    return _global_monitor

def create_enhanced_monitor(config: Optional[Dict[str, Any]] = None) -> EnhancedMonitor:
    """创建新的监控器实例"""
    return EnhancedMonitor(config)