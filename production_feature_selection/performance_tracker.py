
import psutil
import time
from collections import defaultdict

class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self):
        self.performance_data = []
        self.baseline_metrics = None
        self.current_session_metrics = defaultdict(list)
    
    def start_tracking(self, operation_name):
        """开始跟踪某个操作"""
        return PerformanceContext(self, operation_name)
    
    def record_metrics(self, operation_name, metrics):
        """记录性能指标"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation_name,
            "metrics": metrics
        }
        
        self.performance_data.append(record)
        self.current_session_metrics[operation_name].append(metrics)
    
    def set_baseline(self, baseline_data):
        """设置基线性能数据"""
        self.baseline_metrics = baseline_data
        self.logger.info("基线性能数据已设置")
    
    def get_performance_summary(self, days=7):
        """获取性能总结"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_data = [
            record for record in self.performance_data
            if datetime.fromisoformat(record["timestamp"]) > cutoff_date
        ]
        
        if not recent_data:
            return {"status": "no_recent_data"}
        
        # 按操作类型分组
        operation_metrics = defaultdict(list)
        for record in recent_data:
            operation_metrics[record["operation"]].append(record["metrics"])
        
        summary = {}
        for operation, metrics_list in operation_metrics.items():
            summary[operation] = self._calculate_operation_summary(metrics_list)
        
        # 与基线对比
        if self.baseline_metrics:
            summary["vs_baseline"] = self._compare_with_baseline(summary)
        
        return summary
    
    def _calculate_operation_summary(self, metrics_list):
        """计算操作的性能总结"""
        if not metrics_list:
            return {}
        
        # 提取各个指标
        times = [m.get("execution_time", 0) for m in metrics_list]
        memory_peaks = [m.get("peak_memory_mb", 0) for m in metrics_list]
        feature_counts = [m.get("feature_count", 0) for m in metrics_list]
        
        return {
            "avg_execution_time": np.mean(times),
            "min_execution_time": min(times),
            "max_execution_time": max(times),
            "avg_memory_usage": np.mean(memory_peaks),
            "avg_feature_count": np.mean(feature_counts),
            "total_operations": len(metrics_list)
        }
    
    def _compare_with_baseline(self, current_summary):
        """与基线对比"""
        comparison = {}
        
        for operation, current_metrics in current_summary.items():
            if operation in self.baseline_metrics:
                baseline = self.baseline_metrics[operation]
                
                time_improvement = (
                    baseline.get("avg_execution_time", 0) - 
                    current_metrics.get("avg_execution_time", 0)
                ) / baseline.get("avg_execution_time", 1)
                
                memory_improvement = (
                    baseline.get("avg_memory_usage", 0) - 
                    current_metrics.get("avg_memory_usage", 0)
                ) / baseline.get("avg_memory_usage", 1)
                
                comparison[operation] = {
                    "time_improvement_percent": time_improvement * 100,
                    "memory_improvement_percent": memory_improvement * 100,
                    "feature_reduction": baseline.get("avg_feature_count", 0) - current_metrics.get("avg_feature_count", 0)
                }
        
        return comparison
    
    def generate_performance_report(self):
        """生成性能报告"""
        summary = self.get_performance_summary()
        
        report = {
            "report_date": datetime.now().isoformat(),
            "summary": summary,
            "recommendations": self._generate_recommendations(summary),
            "charts_data": self._prepare_charts_data()
        }
        
        # 保存报告
        filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.manager.base_path / "reports" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _generate_recommendations(self, summary):
        """生成优化建议"""
        recommendations = []
        
        if "vs_baseline" in summary:
            baseline_comparison = summary["vs_baseline"]
            
            for operation, comparison in baseline_comparison.items():
                time_improvement = comparison.get("time_improvement_percent", 0)
                memory_improvement = comparison.get("memory_improvement_percent", 0)
                
                if time_improvement < 10:  # 时间改善不足10%
                    recommendations.append(f"{operation}: 时间优化效果有限，建议检查参数配置")
                
                if memory_improvement < 0:  # 内存使用增加
                    recommendations.append(f"{operation}: 内存使用增加，建议优化内存管理")
                
                if time_improvement > 50:  # 时间改善超过50%
                    recommendations.append(f"{operation}: 性能优化显著，可考虑进一步推广")
        
        return recommendations

class PerformanceContext:
    """性能跟踪上下文管理器"""
    
    def __init__(self, tracker, operation_name):
        self.tracker = tracker
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
        self.process = psutil.Process()
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        metrics = {
            "execution_time": end_time - self.start_time,
            "peak_memory_mb": max(self.start_memory, end_memory),
            "memory_change_mb": end_memory - self.start_memory
        }
        
        self.tracker.record_metrics(self.operation_name, metrics)

# 使用示例
def track_model_performance():
    """跟踪模型性能示例"""
    
    tracker = PerformanceTracker()
    
    # 跟踪特征创建性能
    with tracker.start_tracking("feature_creation") as ctx:
        # 这里是特征创建代码
        feature_data = model.create_traditional_features()
        ctx.metrics["feature_count"] = len(feature_data.columns) - 3
    
    # 跟踪模型训练性能
    with tracker.start_tracking("model_training") as ctx:
        # 这里是模型训练代码
        training_results = model.train_enhanced_models()
        ctx.metrics["model_count"] = len(training_results)
    
    # 生成性能报告
    report = tracker.generate_performance_report()
    return report
