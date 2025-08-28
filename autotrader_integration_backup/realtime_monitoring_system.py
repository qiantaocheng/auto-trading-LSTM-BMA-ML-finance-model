#!/usr/bin/env python3
"""
📈 P1级别修复：实时监控指标系统
=======================================

实现关键交易指标的实时监控，包括：
- 订单延迟监控 (order_latency)
- 资金利用率监控 (capital_utilization)  
- 数据新鲜度监控 (data_freshness)
- 拒单率监控 (rejection_rate)
- 胜率监控 (win_rate)
- PnL跟踪 (pnl_tracking)
"""

import time
import logging
import sqlite3
import json
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from collections import deque, defaultdict
import statistics
import asyncio

logger = logging.getLogger(__name__)


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
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    metric_name: str
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricCollector:
    """指标收集器"""
    
    def __init__(self, name: str, metric_type: MetricType, 
                 retention_seconds: int = 3600):
        self.name = name
        self.metric_type = metric_type
        self.retention_seconds = retention_seconds
        self._data_points = deque()
        self._lock = threading.RLock()
    
    def add_point(self, value: float, tags: Dict[str, str] = None, 
                  metadata: Dict[str, Any] = None):
        """添加数据点"""
        point = MetricPoint(
            name=self.name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self._lock:
            self._data_points.append(point)
            # 清理过期数据
            self._cleanup_old_points()
    
    def _cleanup_old_points(self):
        """清理过期数据点"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=self.retention_seconds)
        while self._data_points and self._data_points[0].timestamp < cutoff_time:
            self._data_points.popleft()
    
    def get_current_value(self) -> Optional[float]:
        """获取当前值"""
        with self._lock:
            if not self._data_points:
                return None
            return self._data_points[-1].value
    
    def get_statistics(self, window_seconds: int = 300) -> Dict[str, float]:
        """获取统计信息（默认5分钟窗口）"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        
        with self._lock:
            values = [p.value for p in self._data_points if p.timestamp >= cutoff_time]
            
            if not values:
                return {}
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                'p95': self._percentile(values, 0.95),
                'p99': self._percentile(values, 0.99)
            }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]


class RealtimeMonitoringSystem:
    """实时监控系统"""
    
    def __init__(self, db_path: str = "data/monitoring.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 指标收集器
        self._collectors: Dict[str, MetricCollector] = {}
        
        # 告警配置
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._active_alerts: Dict[str, Alert] = {}
        
        # 告警回调
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # 监控线程
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        self._lock = threading.RLock()
        
        self._init_database()
        self._init_standard_metrics()
        self._start_monitoring()
        
        logger.info("Realtime monitoring system initialized")
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 指标表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    tags TEXT,
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)
                )
            """)
            
            # 告警表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    metric_name TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_metric_name ON alerts(metric_name);
                CREATE INDEX IF NOT EXISTS idx_alerts_level ON alerts(level);
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)
                )
            """)
            
            conn.commit()
    
    def _init_standard_metrics(self):
        """初始化标准指标"""
        # 核心交易指标
        self.register_metric("order_latency_ms", MetricType.HISTOGRAM)
        self.register_metric("capital_utilization_pct", MetricType.GAUGE)  
        self.register_metric("data_freshness_seconds", MetricType.GAUGE)
        self.register_metric("rejection_rate_pct", MetricType.GAUGE)
        self.register_metric("win_rate_pct", MetricType.GAUGE)
        self.register_metric("unrealized_pnl_usd", MetricType.GAUGE)
        self.register_metric("realized_pnl_usd", MetricType.COUNTER)
        
        # 系统性能指标
        self.register_metric("orders_per_second", MetricType.RATE)
        self.register_metric("fills_per_second", MetricType.RATE)
        self.register_metric("memory_usage_mb", MetricType.GAUGE)
        self.register_metric("cpu_usage_pct", MetricType.GAUGE)
        
        # 配置默认告警阈值
        self.set_alert_rule("order_latency_ms", {
            'max_threshold': 1000,  # 1秒
            'level': AlertLevel.WARNING,
            'message_template': "Order latency too high: {value:.1f}ms > {threshold}ms"
        })
        
        self.set_alert_rule("capital_utilization_pct", {
            'max_threshold': 95,
            'level': AlertLevel.WARNING,
            'message_template': "Capital utilization high: {value:.1f}% > {threshold}%"
        })
        
        self.set_alert_rule("data_freshness_seconds", {
            'max_threshold': 300,  # 5分钟
            'level': AlertLevel.ERROR,
            'message_template': "Data staleness detected: {value:.1f}s > {threshold}s"
        })
        
        self.set_alert_rule("rejection_rate_pct", {
            'max_threshold': 5,  # 5%
            'level': AlertLevel.WARNING,
            'message_template': "High rejection rate: {value:.1f}% > {threshold}%"
        })
    
    def register_metric(self, name: str, metric_type: MetricType, 
                       retention_seconds: int = 3600):
        """注册新指标"""
        with self._lock:
            if name in self._collectors:
                logger.warning(f"Metric {name} already registered")
                return
            
            self._collectors[name] = MetricCollector(name, metric_type, retention_seconds)
            logger.info(f"Registered metric: {name} ({metric_type.value})")
    
    def record_metric(self, name: str, value: float, 
                     tags: Dict[str, str] = None, 
                     metadata: Dict[str, Any] = None):
        """记录指标值"""
        if name not in self._collectors:
            logger.warning(f"Unknown metric: {name}")
            return
        
        self._collectors[name].add_point(value, tags, metadata)
        
        # 异步保存到数据库
        threading.Thread(
            target=self._save_metric_to_db,
            args=(name, value, tags, metadata),
            daemon=True
        ).start()
        
        # 检查告警
        self._check_alerts(name, value)
    
    def _save_metric_to_db(self, name: str, value: float,
                          tags: Dict[str, str] = None,
                          metadata: Dict[str, Any] = None):
        """保存指标到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO metrics (name, value, timestamp, tags, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    name, value, datetime.now(timezone.utc).isoformat(),
                    json.dumps(tags) if tags else None,
                    json.dumps(metadata) if metadata else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save metric {name}: {e}")
    
    def set_alert_rule(self, metric_name: str, rule_config: Dict[str, Any]):
        """设置告警规则"""
        with self._lock:
            self._alert_rules[metric_name] = rule_config
            logger.info(f"Alert rule set for {metric_name}: {rule_config}")
    
    def _check_alerts(self, metric_name: str, value: float):
        """检查告警条件"""
        if metric_name not in self._alert_rules:
            return
        
        rule = self._alert_rules[metric_name]
        
        # 检查最大阈值
        if 'max_threshold' in rule and value > rule['max_threshold']:
            self._trigger_alert(metric_name, value, rule['max_threshold'], rule)
        
        # 检查最小阈值
        elif 'min_threshold' in rule and value < rule['min_threshold']:
            self._trigger_alert(metric_name, value, rule['min_threshold'], rule)
    
    def _trigger_alert(self, metric_name: str, value: float, threshold: float, rule: Dict[str, Any]):
        """触发告警"""
        alert_key = f"{metric_name}_{rule.get('level', 'WARNING')}"
        
        # 避免重复告警（5分钟内）
        if alert_key in self._active_alerts:
            last_alert = self._active_alerts[alert_key]
            if (datetime.now(timezone.utc) - last_alert.timestamp).total_seconds() < 300:
                return
        
        # 创建告警
        alert = Alert(
            alert_id=f"{int(time.time())}_{alert_key}",
            metric_name=metric_name,
            level=AlertLevel(rule.get('level', 'WARNING')),
            message=rule.get('message_template', 'Alert: {metric} = {value}').format(
                metric=metric_name, value=value, threshold=threshold
            ),
            value=value,
            threshold=threshold,
            timestamp=datetime.now(timezone.utc)
        )
        
        with self._lock:
            self._active_alerts[alert_key] = alert
            
            # 保存到数据库
            self._save_alert_to_db(alert)
            
            # 触发回调
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"ALERT [{alert.level.value}] {alert.message}")
    
    def _save_alert_to_db(self, alert: Alert):
        """保存告警到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO alerts 
                    (alert_id, metric_name, level, message, value, threshold, timestamp, resolved, resolved_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id, alert.metric_name, alert.level.value, alert.message,
                    alert.value, alert.threshold, alert.timestamp.isoformat(),
                    alert.resolved, alert.resolved_at.isoformat() if alert.resolved_at else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调"""
        with self._lock:
            self._alert_callbacks.append(callback)
    
    def get_metric_value(self, name: str) -> Optional[float]:
        """获取指标当前值"""
        if name not in self._collectors:
            return None
        return self._collectors[name].get_current_value()
    
    def get_metric_statistics(self, name: str, window_seconds: int = 300) -> Dict[str, float]:
        """获取指标统计信息"""
        if name not in self._collectors:
            return {}
        return self._collectors[name].get_statistics(window_seconds)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表盘数据"""
        dashboard = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics': {},
            'alerts': {
                'active': len([a for a in self._active_alerts.values() if not a.resolved]),
                'critical': len([a for a in self._active_alerts.values() 
                               if a.level == AlertLevel.CRITICAL and not a.resolved])
            }
        }
        
        # 获取所有指标的当前值和统计
        for name, collector in self._collectors.items():
            current_value = collector.get_current_value()
            stats = collector.get_statistics(300)  # 5分钟窗口
            
            dashboard['metrics'][name] = {
                'current_value': current_value,
                'statistics': stats
            }
        
        return dashboard
    
    def _start_monitoring(self):
        """启动监控线程"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="MonitoringLoop",
            daemon=True
        )
        self._monitoring_thread.start()
    
    def _monitoring_loop(self):
        """监控循环"""
        while not self._stop_monitoring.wait(10):  # 10秒检查一次
            try:
                self._collect_system_metrics()
                self._cleanup_old_data()
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("cpu_usage_pct", cpu_percent)
            
            # 内存使用
            memory_info = psutil.virtual_memory()
            memory_mb = (memory_info.total - memory_info.available) / 1024 / 1024
            self.record_metric("memory_usage_mb", memory_mb)
            
        except ImportError:
            # psutil不可用，跳过系统指标
            pass
    
    def _cleanup_old_data(self):
        """清理旧数据"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 清理旧指标数据
                cursor.execute("DELETE FROM metrics WHERE timestamp < ?", 
                             (cutoff_time.isoformat(),))
                
                # 清理旧告警数据
                cursor.execute("DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE", 
                             (cutoff_time.isoformat(),))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def stop_monitoring(self):
        """停止监控"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Monitoring system stopped")


# 专用监控函数
class TradingMetrics:
    """交易指标记录器"""
    
    def __init__(self, monitoring_system: RealtimeMonitoringSystem):
        self.monitoring = monitoring_system
        self._order_start_times: Dict[str, float] = {}
        
    def start_order_timing(self, order_id: str):
        """开始订单计时"""
        self._order_start_times[order_id] = time.time()
    
    def end_order_timing(self, order_id: str, success: bool = True):
        """结束订单计时"""
        if order_id in self._order_start_times:
            latency_ms = (time.time() - self._order_start_times[order_id]) * 1000
            self.monitoring.record_metric("order_latency_ms", latency_ms, 
                                        tags={'success': str(success)})
            del self._order_start_times[order_id]
    
    def record_capital_utilization(self, used_capital: float, total_capital: float):
        """记录资金利用率"""
        utilization_pct = (used_capital / total_capital) * 100 if total_capital > 0 else 0
        self.monitoring.record_metric("capital_utilization_pct", utilization_pct)
    
    def record_data_freshness(self, data_age_seconds: float, symbol: str = None):
        """记录数据新鲜度"""
        tags = {'symbol': symbol} if symbol else {}
        self.monitoring.record_metric("data_freshness_seconds", data_age_seconds, tags=tags)
    
    def record_order_rejection(self, total_orders: int, rejected_orders: int):
        """记录拒单率"""
        rejection_rate = (rejected_orders / total_orders) * 100 if total_orders > 0 else 0
        self.monitoring.record_metric("rejection_rate_pct", rejection_rate)
    
    def record_trade_result(self, pnl: float, is_win: bool):
        """记录交易结果"""
        self.monitoring.record_metric("realized_pnl_usd", pnl)
        # 计算胜率需要累计数据，这里简化处理
        win_value = 100.0 if is_win else 0.0
        self.monitoring.record_metric("win_rate_pct", win_value, tags={'trade_result': str(is_win)})


# 全局实例
_global_monitoring: Optional[RealtimeMonitoringSystem] = None
_global_trading_metrics: Optional[TradingMetrics] = None


def get_monitoring_system() -> RealtimeMonitoringSystem:
    """获取全局监控系统"""
    global _global_monitoring
    if _global_monitoring is None:
        _global_monitoring = RealtimeMonitoringSystem()
    return _global_monitoring


def get_trading_metrics() -> TradingMetrics:
    """获取交易指标记录器"""
    global _global_trading_metrics
    if _global_trading_metrics is None:
        monitoring = get_monitoring_system()
        _global_trading_metrics = TradingMetrics(monitoring)
    return _global_trading_metrics


if __name__ == "__main__":
    # 测试监控系统
    logging.basicConfig(level=logging.INFO)
    
    monitoring = RealtimeMonitoringSystem()
    trading_metrics = TradingMetrics(monitoring)
    
    # 添加简单的告警回调
    def alert_callback(alert: Alert):
        print(f"🚨 ALERT: {alert.message}")
    
    monitoring.add_alert_callback(alert_callback)
    
    # 模拟一些指标数据
    trading_metrics.record_capital_utilization(85000, 100000)  # 85%使用率
    trading_metrics.record_data_freshness(120, "AAPL")  # 2分钟延迟
    trading_metrics.record_order_rejection(100, 3)  # 3%拒单率
    
    # 模拟订单延迟
    trading_metrics.start_order_timing("ORDER_001")
    time.sleep(0.1)  # 100ms
    trading_metrics.end_order_timing("ORDER_001", True)
    
    # 获取仪表盘数据
    dashboard = monitoring.get_dashboard_data()
    print("Dashboard:", json.dumps(dashboard, indent=2, default=str))
    
    # 等待一下让监控线程运行
    time.sleep(2)
    
    monitoring.stop_monitoring()