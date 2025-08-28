#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳健特征选择系统生产部署计划
包含集成、监控、调优和性能跟踪的完整方案
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

class ProductionDeploymentManager:
    """
    生产部署管理器
    负责稳健特征选择系统的生产环境集成和管理
    """
    
    def __init__(self, base_path="production_feature_selection"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.base_path / "configs").mkdir(exist_ok=True)
        (self.base_path / "models").mkdir(exist_ok=True)
        (self.base_path / "logs").mkdir(exist_ok=True)
        (self.base_path / "reports").mkdir(exist_ok=True)
        (self.base_path / "monitoring").mkdir(exist_ok=True)
        (self.base_path / "backups").mkdir(exist_ok=True)
        
        self.logger = self._setup_logging()
        self.deployment_config = self._create_deployment_config()
        
    def _setup_logging(self):
        """设置生产环境日志"""
        logger = logging.getLogger('ProductionFeatureSelection')
        logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(
            self.base_path / "logs" / f"feature_selection_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _create_deployment_config(self):
        """创建部署配置"""
        config = {
            "version": "1.0.0",
            "deployment_date": datetime.now().isoformat(),
            
            # 阶段性部署配置
            "deployment_stages": {
                "stage1_testing": {
                    "enabled": True,
                    "parallel_mode": True,  # 与原系统并行运行
                    "comparison_enabled": True,
                    "rollback_threshold": 0.05  # 性能下降5%则回滚
                },
                "stage2_gradual": {
                    "enabled": False,
                    "traffic_percentage": 30,  # 30%流量使用新系统
                    "monitoring_period_days": 30
                },
                "stage3_full": {
                    "enabled": False,
                    "full_replacement": True
                }
            },
            
            # 特征选择参数（生产环境）
            "feature_selection_params": {
                "target_features": 16,
                "ic_window": 126,  # 6个月
                "min_ic_mean": 0.008,  # 稍微提高标准
                "min_ic_ir": 0.25,     # 稍微提高标准
                "max_correlation": 0.55,  # 稍微降低相关性
                "reselection_period": 90,  # 3个月重选
                "emergency_fallback": True
            },
            
            # 监控配置
            "monitoring": {
                "performance_check_frequency": "daily",
                "feature_health_check": "weekly",
                "full_reselection_check": "monthly",
                "alert_thresholds": {
                    "ic_degradation": 0.2,  # IC下降20%报警
                    "prediction_error_increase": 0.15,  # 预测误差增加15%报警
                    "feature_correlation_spike": 0.8,  # 特征相关性超过80%报警
                    "processing_time_increase": 2.0  # 处理时间增加100%报警
                }
            },
            
            # 性能跟踪配置
            "performance_tracking": {
                "metrics": [
                    "training_time",
                    "prediction_time", 
                    "memory_usage",
                    "feature_count",
                    "ic_quality",
                    "prediction_accuracy",
                    "model_stability"
                ],
                "baseline_period_days": 30,
                "comparison_period_days": 7
            }
        }
        
        # 保存配置
        config_path = self.base_path / "configs" / "deployment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return config

class Stage1TestingDeployment:
    """
    第一阶段：测试环境部署
    与原系统并行运行，进行性能对比
    """
    
    def __init__(self, manager):
        self.manager = manager
        self.logger = manager.logger
        
    def deploy_parallel_testing(self):
        """部署并行测试系统"""
        self.logger.info("🚀 开始第一阶段部署：并行测试")
        
        deployment_code = '''
# 第一阶段部署代码
def create_parallel_testing_setup():
    """创建并行测试环境"""
    
    # 1. 导入必要模块
    from bma_robust_feature_production import create_enhanced_bma_model
    from robust_feature_selection import RobustFeatureSelector
    
    class ParallelTestingManager:
        def __init__(self, original_model):
            self.original_model = original_model
            self.enhanced_model = create_enhanced_bma_model(original_model)
            self.comparison_results = []
            
        def run_parallel_analysis(self, *args, **kwargs):
            """并行运行原版和增强版，对比结果"""
            
            start_time = time.time()
            
            # 运行原版模型
            original_start = time.time()
            try:
                original_result = self.original_model.run_complete_analysis(*args, **kwargs)
                original_time = time.time() - original_start
                original_success = True
            except Exception as e:
                original_result = {"error": str(e)}
                original_time = time.time() - original_start
                original_success = False
            
            # 运行增强版模型
            enhanced_start = time.time()
            try:
                enhanced_result = self.enhanced_model.run_complete_analysis(*args, **kwargs)
                enhanced_time = time.time() - enhanced_start
                enhanced_success = True
            except Exception as e:
                enhanced_result = {"error": str(e)}
                enhanced_time = time.time() - enhanced_start
                enhanced_success = False
            
            # 性能对比
            comparison = self._compare_results(
                original_result, enhanced_result,
                original_time, enhanced_time,
                original_success, enhanced_success
            )
            
            self.comparison_results.append({
                "timestamp": datetime.now().isoformat(),
                "args": str(args),
                "kwargs": str(kwargs),
                "comparison": comparison
            })
            
            # 根据配置返回结果
            if enhanced_success and comparison["performance_improvement"] > 0:
                return enhanced_result, comparison
            else:
                return original_result, comparison
        
        def _compare_results(self, orig, enh, orig_time, enh_time, orig_success, enh_success):
            """对比两个模型的结果"""
            comparison = {
                "original_success": orig_success,
                "enhanced_success": enh_success,
                "original_time": orig_time,
                "enhanced_time": enh_time,
                "time_improvement": (orig_time - enh_time) / orig_time if orig_time > 0 else 0,
                "performance_improvement": 0
            }
            
            if orig_success and enh_success:
                # 比较预测质量
                if "predictions" in orig and "predictions" in enh:
                    orig_pred = orig["predictions"]
                    enh_pred = enh["predictions"]
                    
                    if orig_pred is not None and enh_pred is not None:
                        # 计算预测一致性
                        if len(orig_pred) == len(enh_pred):
                            correlation = np.corrcoef(orig_pred, enh_pred)[0, 1]
                            comparison["prediction_correlation"] = correlation
                            
                            # 如果相关性高且时间改善，认为是改进
                            if correlation > 0.9 and comparison["time_improvement"] > 0:
                                comparison["performance_improvement"] = comparison["time_improvement"]
            
            return comparison
        
        def get_performance_summary(self):
            """获取性能总结"""
            if not self.comparison_results:
                return {"status": "no_data"}
            
            improvements = [r["comparison"]["time_improvement"] for r in self.comparison_results]
            successes = [r["comparison"]["enhanced_success"] for r in self.comparison_results]
            
            return {
                "total_runs": len(self.comparison_results),
                "success_rate": sum(successes) / len(successes),
                "avg_time_improvement": np.mean(improvements),
                "min_time_improvement": min(improvements),
                "max_time_improvement": max(improvements),
                "recommendation": "proceed" if np.mean(improvements) > 0.1 else "investigate"
            }
    
    return ParallelTestingManager

# 使用示例
if __name__ == "__main__":
    # 创建并行测试管理器
    original_bma = UltraEnhancedQuantitativeModel()
    parallel_manager = create_parallel_testing_setup()(original_bma)
    
    # 运行测试
    result, comparison = parallel_manager.run_parallel_analysis(
        tickers=['AAPL', 'MSFT'],
        start_date='2024-01-01',
        end_date='2024-12-01'
    )
    
    # 查看性能总结
    summary = parallel_manager.get_performance_summary()
    print(f"性能改进: {summary}")
'''
        
        # 保存部署代码
        code_path = self.manager.base_path / "stage1_parallel_testing.py"
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(deployment_code)
        
        self.logger.info(f"✅ 第一阶段部署代码已生成: {code_path}")
        
        return deployment_code

class Stage2GradualDeployment:
    """
    第二阶段：渐进式部署
    部分流量使用新系统，逐步扩大范围
    """
    
    def __init__(self, manager):
        self.manager = manager
        self.logger = manager.logger
    
    def create_gradual_deployment(self):
        """创建渐进式部署系统"""
        self.logger.info("🎯 准备第二阶段部署：渐进式替换")
        
        gradual_code = '''
class GradualDeploymentManager:
    """渐进式部署管理器"""
    
    def __init__(self, original_model, traffic_percentage=30):
        self.original_model = original_model
        self.enhanced_model = create_enhanced_bma_model(original_model)
        self.traffic_percentage = traffic_percentage
        self.deployment_metrics = []
        
    def route_request(self, request_hash, *args, **kwargs):
        """根据流量百分比路由请求"""
        
        # 使用请求哈希决定路由
        if hash(request_hash) % 100 < self.traffic_percentage:
            # 使用增强版模型
            try:
                result = self.enhanced_model.run_complete_analysis(*args, **kwargs)
                self._record_metrics("enhanced", True, result)
                return result
            except Exception as e:
                self.logger.error(f"增强版模型失败，回退到原版: {e}")
                result = self.original_model.run_complete_analysis(*args, **kwargs)
                self._record_metrics("enhanced", False, result)
                return result
        else:
            # 使用原版模型
            result = self.original_model.run_complete_analysis(*args, **kwargs)
            self._record_metrics("original", True, result)
            return result
    
    def _record_metrics(self, model_type, success, result):
        """记录部署指标"""
        self.deployment_metrics.append({
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "success": success,
            "has_predictions": "predictions" in result and result["predictions"] is not None
        })
    
    def get_deployment_health(self):
        """获取部署健康状况"""
        if not self.deployment_metrics:
            return {"status": "no_data"}
        
        recent_metrics = [m for m in self.deployment_metrics 
                         if datetime.fromisoformat(m["timestamp"]) > datetime.now() - timedelta(days=7)]
        
        enhanced_metrics = [m for m in recent_metrics if m["model_type"] == "enhanced"]
        original_metrics = [m for m in recent_metrics if m["model_type"] == "original"]
        
        enhanced_success = sum(m["success"] for m in enhanced_metrics) / len(enhanced_metrics) if enhanced_metrics else 0
        original_success = sum(m["success"] for m in original_metrics) / len(original_metrics) if original_metrics else 0
        
        return {
            "enhanced_success_rate": enhanced_success,
            "original_success_rate": original_success,
            "enhanced_requests": len(enhanced_metrics),
            "original_requests": len(original_metrics),
            "health_status": "good" if enhanced_success >= original_success * 0.95 else "needs_attention"
        }
'''
        
        # 保存渐进部署代码
        code_path = self.manager.base_path / "stage2_gradual_deployment.py"
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(gradual_code)
        
        self.logger.info(f"✅ 第二阶段部署代码已生成: {code_path}")

class ParameterOptimizer:
    """
    参数调优器
    根据实际数据和性能反馈调优特征选择参数
    """
    
    def __init__(self, manager):
        self.manager = manager
        self.logger = manager.logger
    
    def create_parameter_tuning_system(self):
        """创建参数调优系统"""
        self.logger.info("🎛️ 创建参数调优系统")
        
        tuning_code = '''
class AdaptiveParameterTuner:
    """自适应参数调优器"""
    
    def __init__(self, base_config):
        self.base_config = base_config.copy()
        self.tuning_history = []
        self.current_params = base_config.copy()
        
    def evaluate_current_parameters(self, model, test_data):
        """评估当前参数的效果"""
        
        # 运行特征选择
        selector = RobustFeatureSelector(**self.current_params)
        
        try:
            X = test_data["features"]
            y = test_data["targets"]
            dates = test_data["dates"]
            
            X_selected = selector.fit_transform(X, y, dates)
            
            # 计算评估指标
            report = selector.get_feature_report()
            selected_stats = report[report['selected']]
            
            evaluation = {
                "selected_feature_count": len(selected_stats),
                "avg_ic": selected_stats['ic_mean'].mean(),
                "avg_ic_ir": selected_stats['ic_ir'].mean(),
                "compression_ratio": len(selected_stats) / len(report),
                "max_correlation": self._calculate_max_correlation(X_selected),
                "timestamp": datetime.now().isoformat(),
                "parameters": self.current_params.copy()
            }
            
            self.tuning_history.append(evaluation)
            return evaluation
            
        except Exception as e:
            self.logger.error(f"参数评估失败: {e}")
            return None
    
    def suggest_parameter_adjustments(self):
        """基于历史数据建议参数调整"""
        
        if len(self.tuning_history) < 3:
            return {"message": "需要更多历史数据进行调优"}
        
        recent_evals = self.tuning_history[-5:]  # 最近5次评估
        
        suggestions = []
        
        # 分析特征数量趋势
        feature_counts = [e["selected_feature_count"] for e in recent_evals]
        avg_features = np.mean(feature_counts)
        
        if avg_features < self.current_params["target_features"] * 0.7:
            suggestions.append({
                "parameter": "min_ic_mean",
                "current": self.current_params["min_ic_mean"],
                "suggested": self.current_params["min_ic_mean"] * 0.8,
                "reason": "选择特征过少，降低IC阈值"
            })
        elif avg_features > self.current_params["target_features"] * 1.2:
            suggestions.append({
                "parameter": "min_ic_ir", 
                "current": self.current_params["min_ic_ir"],
                "suggested": self.current_params["min_ic_ir"] * 1.2,
                "reason": "选择特征过多，提高IC_IR阈值"
            })
        
        # 分析IC质量趋势
        ic_values = [e["avg_ic"] for e in recent_evals]
        if np.mean(ic_values) < 0.01:
            suggestions.append({
                "parameter": "ic_window",
                "current": self.current_params["ic_window"],
                "suggested": max(60, self.current_params["ic_window"] - 30),
                "reason": "IC质量较低，缩短评估窗口"
            })
        
        return suggestions
    
    def apply_parameter_adjustment(self, adjustments):
        """应用参数调整"""
        for adj in adjustments:
            if adj["parameter"] in self.current_params:
                old_value = self.current_params[adj["parameter"]]
                self.current_params[adj["parameter"]] = adj["suggested"]
                self.logger.info(f"参数调整: {adj['parameter']} {old_value} -> {adj['suggested']} ({adj['reason']})")
        
        return self.current_params
    
    def _calculate_max_correlation(self, X):
        """计算特征间最大相关性"""
        if len(X.columns) <= 1:
            return 0.0
        
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        return upper_triangle.max().max()

# 使用示例
def run_parameter_optimization():
    """运行参数优化流程"""
    
    base_params = {
        "target_features": 16,
        "ic_window": 126,
        "min_ic_mean": 0.008,
        "min_ic_ir": 0.25,
        "max_correlation": 0.55
    }
    
    tuner = AdaptiveParameterTuner(base_params)
    
    # 每周运行一次参数评估
    # 这里应该用实际的测试数据
    test_data = load_recent_market_data()  # 需要实现
    
    evaluation = tuner.evaluate_current_parameters(model, test_data)
    suggestions = tuner.suggest_parameter_adjustments()
    
    if suggestions:
        tuner.apply_parameter_adjustment(suggestions)
        save_updated_parameters(tuner.current_params)
'''
        
        # 保存调优代码
        code_path = self.manager.base_path / "parameter_optimizer.py"
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(tuning_code)
        
        self.logger.info(f"✅ 参数调优系统已生成: {code_path}")

class MonitoringSystem:
    """
    监控系统
    实现6-12个月的定期特征"体检"
    """
    
    def __init__(self, manager):
        self.manager = manager
        self.logger = manager.logger
    
    def create_monitoring_system(self):
        """创建监控系统"""
        self.logger.info("📊 创建监控系统")
        
        monitoring_code = '''
import schedule
import time
from datetime import datetime, timedelta

class FeatureHealthMonitor:
    """特征健康监控器"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.health_history = []
        self.alert_thresholds = {
            "ic_degradation": 0.2,
            "feature_count_change": 0.3,
            "correlation_spike": 0.8,
            "performance_drop": 0.15
        }
    
    def daily_health_check(self):
        """每日健康检查"""
        self.logger.info("🔍 执行每日特征健康检查")
        
        try:
            # 获取当前模型状态
            status = self.model_manager.get_feature_selection_status()
            
            # 检查基本状态
            health_report = {
                "timestamp": datetime.now().isoformat(),
                "feature_count": status.get("selected_features_count", 0),
                "last_selection_age_days": self._calculate_selection_age(status),
                "selector_available": status.get("selector_available", False),
                "health_score": 0.0
            }
            
            # 计算健康分数
            health_score = 1.0
            
            if not status.get("selector_available", False):
                health_score -= 0.5
                health_report["alerts"] = ["特征选择器不可用"]
            
            if health_report["last_selection_age_days"] > 120:  # 4个月
                health_score -= 0.3
                health_report.setdefault("alerts", []).append("特征选择过期")
            
            health_report["health_score"] = health_score
            self.health_history.append(health_report)
            
            # 保存报告
            self._save_health_report(health_report)
            
            # 触发警报
            if health_score < 0.7:
                self._send_alert(health_report)
            
        except Exception as e:
            self.logger.error(f"每日健康检查失败: {e}")
    
    def weekly_feature_analysis(self):
        """每周特征分析"""
        self.logger.info("📈 执行每周特征分析")
        
        try:
            # 获取最近一周的模型表现数据
            recent_performance = self._get_recent_performance()
            
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "analysis_period": "weekly",
                "performance_metrics": recent_performance,
                "recommendations": []
            }
            
            # 分析性能趋势
            if recent_performance:
                ic_trend = recent_performance.get("ic_trend", 0)
                if ic_trend < -0.1:  # IC下降超过10%
                    analysis["recommendations"].append("考虑重新进行特征选择")
                
                processing_time_trend = recent_performance.get("processing_time_trend", 0)
                if processing_time_trend > 0.2:  # 处理时间增加20%
                    analysis["recommendations"].append("检查系统资源使用情况")
            
            # 保存分析结果
            self._save_analysis_report(analysis)
            
        except Exception as e:
            self.logger.error(f"每周特征分析失败: {e}")
    
    def monthly_comprehensive_review(self):
        """每月综合评估"""
        self.logger.info("🔬 执行每月综合评估")
        
        try:
            # 获取过去一个月的所有数据
            monthly_data = self._get_monthly_data()
            
            comprehensive_review = {
                "timestamp": datetime.now().isoformat(),
                "review_period": "monthly",
                "summary": self._generate_monthly_summary(monthly_data),
                "feature_stability": self._analyze_feature_stability(monthly_data),
                "performance_comparison": self._compare_monthly_performance(monthly_data),
                "reselection_recommendation": False
            }
            
            # 决定是否需要重新选择特征
            if self._should_trigger_reselection(comprehensive_review):
                comprehensive_review["reselection_recommendation"] = True
                self.logger.warning("🚨 建议进行特征重新选择")
            
            # 保存综合评估
            self._save_comprehensive_review(comprehensive_review)
            
        except Exception as e:
            self.logger.error(f"每月综合评估失败: {e}")
    
    def setup_scheduled_monitoring(self):
        """设置定时监控任务"""
        
        # 每日健康检查（工作日早上9点）
        schedule.every().monday.at("09:00").do(self.daily_health_check)
        schedule.every().tuesday.at("09:00").do(self.daily_health_check)
        schedule.every().wednesday.at("09:00").do(self.daily_health_check)
        schedule.every().thursday.at("09:00").do(self.daily_health_check)
        schedule.every().friday.at("09:00").do(self.daily_health_check)
        
        # 每周特征分析（周日晚上8点）
        schedule.every().sunday.at("20:00").do(self.weekly_feature_analysis)
        
        # 每月综合评估（每月1号早上8点）
        schedule.every().month.do(self.monthly_comprehensive_review)
        
        self.logger.info("✅ 定时监控任务已设置")
        
        # 启动监控循环
        while True:
            schedule.run_pending()
            time.sleep(3600)  # 每小时检查一次
    
    def _calculate_selection_age(self, status):
        """计算特征选择的年龄（天数）"""
        if not status.get("last_selection_date"):
            return 999  # 如果没有记录，返回很大的数字
        
        last_date = datetime.fromisoformat(status["last_selection_date"])
        return (datetime.now() - last_date).days
    
    def _save_health_report(self, report):
        """保存健康报告"""
        filename = f"health_report_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = self.manager.base_path / "monitoring" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def _send_alert(self, health_report):
        """发送警报"""
        alert_message = f"""
        🚨 特征选择系统健康警报
        
        时间: {health_report['timestamp']}
        健康分数: {health_report['health_score']:.2f}
        问题: {health_report.get('alerts', [])}
        
        建议立即检查系统状态。
        """
        
        self.logger.warning(alert_message)
        # 这里可以集成邮件、钉钉、微信等通知系统
'''
        
        # 保存监控代码
        code_path = self.manager.base_path / "monitoring_system.py"
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(monitoring_code)
        
        self.logger.info(f"✅ 监控系统已生成: {code_path}")

class PerformanceTracker:
    """
    性能跟踪器
    监控计算效率和预测质量提升
    """
    
    def __init__(self, manager):
        self.manager = manager
        self.logger = manager.logger
    
    def create_performance_tracking(self):
        """创建性能跟踪系统"""
        self.logger.info("⚡ 创建性能跟踪系统")
        
        tracking_code = '''
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
'''
        
        # 保存跟踪代码
        code_path = self.manager.base_path / "performance_tracker.py"
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(tracking_code)
        
        self.logger.info(f"✅ 性能跟踪系统已生成: {code_path}")

def main():
    """主要部署流程"""
    print("🚀 稳健特征选择系统生产部署计划")
    print("=" * 60)
    
    # 创建部署管理器
    manager = ProductionDeploymentManager()
    
    print("1. 创建第一阶段部署（并行测试）...")
    stage1 = Stage1TestingDeployment(manager)
    stage1.deploy_parallel_testing()
    
    print("2. 创建第二阶段部署（渐进式替换）...")
    stage2 = Stage2GradualDeployment(manager)
    stage2.create_gradual_deployment()
    
    print("3. 创建参数调优系统...")
    optimizer = ParameterOptimizer(manager)
    optimizer.create_parameter_tuning_system()
    
    print("4. 创建监控系统...")
    monitor = MonitoringSystem(manager)
    monitor.create_monitoring_system()
    
    print("5. 创建性能跟踪系统...")
    tracker = PerformanceTracker(manager)
    tracker.create_performance_tracking()
    
    print("\n✅ 生产部署计划创建完成！")
    print(f"📁 所有文件已保存到: {manager.base_path}")
    print("\n📋 下一步操作：")
    print("1. 审查生成的部署代码")
    print("2. 根据实际环境调整配置参数")
    print("3. 在测试环境运行第一阶段部署")
    print("4. 监控性能指标并逐步推进")

if __name__ == "__main__":
    main()
