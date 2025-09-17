#!/usr/bin/env python3
"""
Unified Monitoring Interface - 统一监控接口
整合所有监控组件，提供简化的统一接口
"""

import logging
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

class MonitoringLevel(Enum):
    """监控级别"""
    MINIMAL = "minimal"      # 最小监控：仅关键指标
    STANDARD = "standard"    # 标准监控：常用指标
    DETAILED = "detailed"    # 详细监控：所有指标
    DEBUG = "debug"         # 调试监控：包含调试信息

class MetricCategory(Enum):
    """指标类别"""
    PERFORMANCE = "performance"      # 性能指标
    TRADING = "trading"             # 交易指标
    SYSTEM = "system"               # 系统指标
    ERROR = "error"                 # 错误指标
    CUSTOM = "custom"               # 自定义指标

@dataclass
class MetricData:
    """指标数据"""
    name: str
    value: Union[float, int, str]
    category: MetricCategory
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""

@dataclass 
class AlertRule:
    """告警规则"""
    metric_name: str
    condition: str  # e.g., ">", "<", ">=", "<=", "=="
    threshold: Union[float, int]
    severity: str = "warning"  # info, warning, error, critical
    enabled: bool = True
    cooldown_seconds: int = 300  # 告警冷却时间

class UnifiedMonitoringInterface:
    """
    统一监控接口
    整合多个监控组件，提供简化的统一接口
    """
    
    def __init__(self, level: MonitoringLevel = MonitoringLevel.STANDARD):
        self.logger = logging.getLogger("UnifiedMonitoring")
        self.level = level
        
        # 核心监控数据
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[Dict] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.last_alert_time: Dict[str, float] = {}
        
        # 监控组件
        self.monitoring_components = {}
        self.enabled_categories = set()
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 监控状态
        self.is_running = False
        self.start_time = time.time()
        
        # 根据监控级别配置
        self._configure_monitoring_level()
        
        # 初始化监控组件
        self._initialize_components()
        
        self.logger.info(f"Unified monitoring interface initialized with level: {level.value}")
    
    def _configure_monitoring_level(self):
        """根据监控级别配置启用的类别"""
        if self.level == MonitoringLevel.MINIMAL:
            self.enabled_categories = {MetricCategory.PERFORMANCE, MetricCategory.ERROR}
        elif self.level == MonitoringLevel.STANDARD:
            self.enabled_categories = {
                MetricCategory.PERFORMANCE, 
                MetricCategory.TRADING, 
                MetricCategory.ERROR
            }
        elif self.level == MonitoringLevel.DETAILED:
            self.enabled_categories = {
                MetricCategory.PERFORMANCE,
                MetricCategory.TRADING,
                MetricCategory.SYSTEM,
                MetricCategory.ERROR,
                MetricCategory.CUSTOM
            }
        else:  # DEBUG
            self.enabled_categories = set(MetricCategory)
    
    def _initialize_components(self):
        """初始化监控组件"""
        try:
            # 尝试加载各种监控组件
            self._load_performance_monitor()
            self._load_system_monitor()
            self._load_trading_monitor()
            
        except Exception as e:
            self.logger.warning(f"Some monitoring components failed to load: {e}")
    
    def _load_performance_monitor(self):
        """加载性能监控器"""
        try:
            from .unified_monitoring_system import TradingPerformanceMonitor as PerformanceMonitor
            self.monitoring_components['performance'] = PerformanceMonitor()
            self.logger.debug("Performance monitor loaded")
        except ImportError:
            self.logger.debug("Performance monitor not available")
    
    def _load_system_monitor(self):
        """加载系统监控器"""
        try:
            from .unified_monitoring_system import ResourceMonitor
            self.monitoring_components['system'] = ResourceMonitor()
            self.logger.debug("System monitor loaded")
        except ImportError:
            self.logger.debug("System monitor not available")
    
    def _load_trading_monitor(self):
        """加载交易监控器"""
        try:
            from .unified_monitoring_system import UnifiedMonitoringSystem as RealtimeMonitoringSystem
            self.monitoring_components['trading'] = RealtimeMonitoringSystem()
            self.logger.debug("Trading monitor loaded")
        except ImportError:
            self.logger.debug("Trading monitor not available")
    
    def record_metric(self, name: str, value: Union[float, int, str], 
                     category: MetricCategory = MetricCategory.CUSTOM,
                     labels: Optional[Dict[str, str]] = None,
                     description: str = ""):
        """
        记录指标
        
        Args:
            name: 指标名称
            value: 指标值
            category: 指标类别
            labels: 标签
            description: 描述
        """
        if category not in self.enabled_categories:
            return
        
        metric = MetricData(
            name=name,
            value=value,
            category=category,
            labels=labels or {},
            description=description
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            
            # 检查告警规则
            self._check_alert_rules(metric)
        
        self.logger.debug(f"Recorded metric: {name} = {value}")
    
    def _check_alert_rules(self, metric: MetricData):
        """检查告警规则"""
        rule = self.alert_rules.get(metric.name)
        if not rule or not rule.enabled:
            return
        
        # 检查冷却时间
        last_alert = self.last_alert_time.get(metric.name, 0)
        if time.time() - last_alert < rule.cooldown_seconds:
            return
        
        # 检查条件
        try:
            if isinstance(metric.value, (int, float)):
                if self._evaluate_condition(metric.value, rule.condition, rule.threshold):
                    self._trigger_alert(metric, rule)
        except Exception as e:
            self.logger.warning(f"Alert rule evaluation failed for {metric.name}: {e}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """评估告警条件"""
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        return False
    
    def _trigger_alert(self, metric: MetricData, rule: AlertRule):
        """触发告警"""
        alert = {
            'timestamp': time.time(),
            'metric_name': metric.name,
            'metric_value': metric.value,
            'rule_condition': f"{rule.condition} {rule.threshold}",
            'severity': rule.severity,
            'message': f"Metric {metric.name} ({metric.value}) {rule.condition} {rule.threshold}",
            'labels': metric.labels
        }
        
        with self._lock:
            self.alerts.append(alert)
            self.last_alert_time[metric.name] = time.time()
            
            # 限制告警历史长度
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-500:]
        
        # 记录告警日志
        level_map = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        log_level = level_map.get(rule.severity, logging.WARNING)
        self.logger.log(log_level, f"ALERT: {alert['message']}")
    
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self._lock:
            self.alert_rules[rule.metric_name] = rule
        self.logger.info(f"Added alert rule for {rule.metric_name}")
    
    def remove_alert_rule(self, metric_name: str):
        """移除告警规则"""
        with self._lock:
            if metric_name in self.alert_rules:
                del self.alert_rules[metric_name]
        self.logger.info(f"Removed alert rule for {metric_name}")
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[MetricData]:
        """获取指标历史"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            if name not in self.metrics:
                return []
            
            return [
                metric for metric in self.metrics[name]
                if metric.timestamp >= cutoff_time
            ]
    
    def get_latest_metrics(self) -> Dict[str, MetricData]:
        """获取最新指标值"""
        latest = {}
        
        with self._lock:
            for name, history in self.metrics.items():
                if history:
                    latest[name] = history[-1]
        
        return latest
    
    def get_recent_alerts(self, hours: int = 1) -> List[Dict]:
        """获取最近的告警"""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            return [
                alert for alert in self.alerts
                if alert['timestamp'] >= cutoff_time
            ]
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        with self._lock:
            recent_alerts = self.get_recent_alerts(24)  # 24小时内的告警
            
            summary = {
                'monitoring_level': self.level.value,
                'uptime_seconds': time.time() - self.start_time,
                'enabled_categories': [cat.value for cat in self.enabled_categories],
                'total_metrics': len(self.metrics),
                'active_alert_rules': len(self.alert_rules),
                'recent_alerts_24h': len(recent_alerts),
                'alert_summary': {
                    'critical': len([a for a in recent_alerts if a['severity'] == 'critical']),
                    'error': len([a for a in recent_alerts if a['severity'] == 'error']),
                    'warning': len([a for a in recent_alerts if a['severity'] == 'warning']),
                    'info': len([a for a in recent_alerts if a['severity'] == 'info'])
                },
                'available_components': list(self.monitoring_components.keys()),
                'metrics_summary': {}
            }
            
            # 添加指标摘要
            for name, history in self.metrics.items():
                if history:
                    latest = history[-1]
                    summary['metrics_summary'][name] = {
                        'latest_value': latest.value,
                        'category': latest.category.value,
                        'data_points': len(history),
                        'last_updated': latest.timestamp
                    }
            
            return summary
    
    def start_monitoring(self):
        """启动监控"""
        if self.is_running:
            self.logger.warning("Monitoring is already running")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # 启动可用的监控组件
        for name, component in self.monitoring_components.items():
            try:
                if hasattr(component, 'start'):
                    component.start()
                    self.logger.debug(f"Started monitoring component: {name}")
            except Exception as e:
                self.logger.warning(f"Failed to start monitoring component {name}: {e}")
        
        self.logger.info("Unified monitoring started")
    
    def stop_monitoring(self):
        """停止监控"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 停止所有监控组件
        for name, component in self.monitoring_components.items():
            try:
                if hasattr(component, 'stop'):
                    component.stop()
                    self.logger.debug(f"Stopped monitoring component: {name}")
            except Exception as e:
                self.logger.warning(f"Failed to stop monitoring component {name}: {e}")
        
        self.logger.info("Unified monitoring stopped")
    
    def export_metrics(self, filepath: Optional[str] = None) -> str:
        """导出指标数据"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/metrics_export_{timestamp}.json"
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'monitoring_summary': self.get_monitoring_summary(),
            'recent_alerts': self.get_recent_alerts(24),
            'metrics_data': {}
        }
        
        # 导出最近1小时的指标数据
        with self._lock:
            for name in self.metrics.keys():
                history = self.get_metric_history(name, 1)
                export_data['metrics_data'][name] = [
                    {
                        'timestamp': m.timestamp,
                        'value': m.value,
                        'category': m.category.value,
                        'labels': m.labels
                    }
                    for m in history
                ]
        
        # 写入文件
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Metrics exported to: {filepath}")
        return filepath
    
    # 便捷方法
    def record_performance_metric(self, name: str, duration_ms: float):
        """记录性能指标"""
        self.record_metric(name, duration_ms, MetricCategory.PERFORMANCE, description="Performance metric in milliseconds")
    
    def record_trading_metric(self, name: str, value: Union[float, int], labels: Optional[Dict] = None):
        """记录交易指标"""
        self.record_metric(name, value, MetricCategory.TRADING, labels, description="Trading metric")
    
    def record_system_metric(self, name: str, value: Union[float, int]):
        """记录系统指标"""
        self.record_metric(name, value, MetricCategory.SYSTEM, description="System metric")
    
    def record_error(self, error_type: str, count: int = 1):
        """记录错误"""
        self.record_metric(f"error_{error_type}", count, MetricCategory.ERROR, description="Error count")


# 全局单例
_global_monitoring_interface: Optional[UnifiedMonitoringInterface] = None
_monitoring_lock = threading.RLock()

def get_monitoring_interface(level: MonitoringLevel = MonitoringLevel.STANDARD) -> UnifiedMonitoringInterface:
    """获取全局监控接口"""
    global _global_monitoring_interface
    
    with _monitoring_lock:
        if _global_monitoring_interface is None:
            _global_monitoring_interface = UnifiedMonitoringInterface(level)
        return _global_monitoring_interface

def reset_monitoring_interface():
    """重置全局监控接口（用于测试）"""
    global _global_monitoring_interface
    with _monitoring_lock:
        if _global_monitoring_interface:
            _global_monitoring_interface.stop_monitoring()
        _global_monitoring_interface = None


# 便捷函数
def record_metric(name: str, value: Union[float, int, str], 
                 category: MetricCategory = MetricCategory.CUSTOM):
    """便捷的指标记录函数"""
    interface = get_monitoring_interface()
    interface.record_metric(name, value, category)

def record_performance(name: str, duration_ms: float):
    """便捷的性能指标记录函数"""
    interface = get_monitoring_interface()
    interface.record_performance_metric(name, duration_ms)

def record_trading(name: str, value: Union[float, int], labels: Optional[Dict] = None):
    """便捷的交易指标记录函数"""
    interface = get_monitoring_interface()
    interface.record_trading_metric(name, value, labels)

def add_alert(metric_name: str, condition: str, threshold: Union[float, int], 
              severity: str = "warning"):
    """便捷的告警规则添加函数"""
    interface = get_monitoring_interface()
    rule = AlertRule(
        metric_name=metric_name,
        condition=condition,
        threshold=threshold,
        severity=severity
    )
    interface.add_alert_rule(rule)


if __name__ == "__main__":
    # 测试统一监控接口
    logging.basicConfig(level=logging.INFO)
    
    # 创建监控接口
    monitor = UnifiedMonitoringInterface(MonitoringLevel.DETAILED)
    monitor.start_monitoring()
    
    # 测试指标记录
    monitor.record_performance_metric("test_latency", 25.5)
    monitor.record_trading_metric("order_count", 10, {"strategy": "momentum"})
    monitor.record_system_metric("cpu_usage", 75.2)
    
    # 测试告警规则
    monitor.add_alert_rule(AlertRule(
        metric_name="test_latency",
        condition=">",
        threshold=50.0,
        severity="warning"
    ))
    
    # 触发告警
    monitor.record_performance_metric("test_latency", 60.0)
    
    # 获取摘要
    summary = monitor.get_monitoring_summary()
    print(f"Monitoring summary: {json.dumps(summary, indent=2, default=str)}")
    
    # 导出指标
    export_file = monitor.export_metrics()
    print(f"Metrics exported to: {export_file}")
    
    monitor.stop_monitoring()
    print("Unified monitoring interface test completed")