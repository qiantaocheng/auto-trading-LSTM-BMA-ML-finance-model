#!/usr/bin/env python3
"""
交易性能监控器
监控订单执行延迟和系统性能
"""

import time
import asyncio
import logging
from typing import Dict, List
from dataclasses import dataclass
from collections import defaultdict, deque

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

# 全局监控器实例
_performance_monitor = TradingPerformanceMonitor()

def get_performance_monitor():
    """获取性能监控器"""
    return _performance_monitor
