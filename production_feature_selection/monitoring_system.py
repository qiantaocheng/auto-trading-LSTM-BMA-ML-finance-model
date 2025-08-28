
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
