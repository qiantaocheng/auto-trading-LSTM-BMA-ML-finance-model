
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
