
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
