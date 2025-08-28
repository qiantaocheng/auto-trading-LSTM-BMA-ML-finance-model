
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
