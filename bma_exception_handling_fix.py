
import traceback
from typing import Any, Dict, Optional
from contextlib import contextmanager

class BMAExceptionHandler:
    """BMA异常处理器"""
    
    def __init__(self, logger):
        self.logger = logger
        self.error_counts = {}
        self.max_retries = 3
        
    @contextmanager
    def safe_execution(self, operation_name: str, fallback_result: Any = None):
        """安全执行上下文管理器"""
        try:
            self.logger.debug(f"开始执行: {operation_name}")
            yield
            self.logger.debug(f"成功完成: {operation_name}")
            
        except Exception as e:
            self.logger.error(f"操作失败: {operation_name} - {e}")
            self.logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
            
            # 记录错误统计
            self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
            
            # 如果有回退结果，返回回退结果而不是抛出异常
            if fallback_result is not None:
                self.logger.warning(f"使用回退结果: {operation_name}")
                return fallback_result
            else:
                raise

    def safe_model_training(self, model_name: str, train_func, *args, **kwargs):
        """安全的模型训练"""
        with self.safe_execution(f"{model_name}模型训练", fallback_result=None):
            return train_func(*args, **kwargs)
    
    def safe_prediction(self, model_name: str, predict_func, *args, **kwargs):
        """安全的模型预测"""
        fallback_prediction = {'prediction': 0.0, 'confidence': 0.0, 'model_type': 'fallback'}
        
        with self.safe_execution(f"{model_name}预测", fallback_result=fallback_prediction):
            return predict_func(*args, **kwargs)
    
    def safe_feature_generation(self, feature_name: str, feature_func, *args, **kwargs):
        """安全的特征生成"""
        with self.safe_execution(f"{feature_name}特征生成", fallback_result=pd.DataFrame()):
            return feature_func(*args, **kwargs)

# 增强的训练方法 - 带完善异常处理
def robust_train_enhanced_models(self, feature_data: pd.DataFrame, 
                               current_ticker: str = None) -> Dict[str, Any]:
    """稳健的增强模型训练 - 完善异常处理"""
    
    exception_handler = BMAExceptionHandler(logger)
    training_results = {
        'model_performance': {},
        'oof_predictions': {},
        'stacking': {},
        'v5_enhancements': {},
        'errors': [],
        'warnings': []
    }
    
    try:
        # 1. 数据验证阶段
        with exception_handler.safe_execution("数据验证"):
            if feature_data.empty:
                raise ValueError("特征数据为空")
            
            if 'target' not in feature_data.columns:
                raise ValueError("缺少目标变量列")
            
            # 检查数据质量
            null_ratio = feature_data.isnull().sum().sum() / (feature_data.shape[0] * feature_data.shape[1])
            if null_ratio > 0.5:
                training_results['warnings'].append(f"数据缺失率过高: {null_ratio:.1%}")
        
        # 2. Alpha引擎训练
        if hasattr(self, 'alpha_engine') and self.alpha_engine:
            alpha_result = exception_handler.safe_model_training(
                "Alpha引擎", 
                self._safe_alpha_training,
                feature_data
            )
            if alpha_result:
                training_results['alpha_strategies'] = alpha_result
        
        # 3. LTR训练
        if hasattr(self, 'ltr_bma') and self.ltr_bma:
            ltr_result = exception_handler.safe_model_training(
                "LTR",
                self._safe_ltr_training,
                feature_data
            )
            if ltr_result:
                training_results['learning_to_rank'] = ltr_result
        
        # 4. Regime-aware训练
        if hasattr(self, 'regime_trainer') and self.regime_trainer:
            regime_result = exception_handler.safe_model_training(
                "Regime-aware",
                self._safe_regime_training,
                feature_data
            )
            if regime_result:
                training_results['regime_aware'] = regime_result
        
        # 5. 传统模型训练
        traditional_result = exception_handler.safe_model_training(
            "传统模型",
            self._safe_traditional_training,
            feature_data, current_ticker
        )
        if traditional_result:
            training_results['traditional_models'] = traditional_result
        
        # 6. V5增强功能摘要
        if any(key in training_results for key in ['traditional_models', 'learning_to_rank']):
            try:
                v5_summary = self._generate_v5_enhancement_summary(
                    training_results.get('traditional_models', {}).get('model_performance', {}),
                    np.ones(len(feature_data)),  # 默认权重
                    {'embargo': 5, 'gap': 5}    # 默认CV配置
                )
                training_results['v5_enhancements'] = v5_summary
            except Exception as e:
                training_results['warnings'].append(f"V5摘要生成失败: {e}")
        
        # 7. 记录统计信息
        training_results['training_statistics'] = {
            'total_samples': len(feature_data),
            'feature_count': len([col for col in feature_data.columns if col not in ['ticker', 'date', 'target']]),
            'successful_models': len([k for k, v in training_results.items() if v and k not in ['errors', 'warnings']]),
            'error_count': len(exception_handler.error_counts),
            'warning_count': len(training_results['warnings'])
        }
        
        logger.info(f"训练完成: {training_results['training_statistics']}")
        
        return training_results
        
    except Exception as e:
        logger.error(f"训练过程发生严重错误: {e}")
        training_results['errors'].append({
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc()
        })
        
        # 即使失败也返回部分结果
        return training_results

def _safe_alpha_training(self, feature_data):
    """安全的Alpha策略训练"""
    try:
        # Alpha策略训练逻辑
        if not hasattr(self.alpha_engine, 'compute_all_alphas'):
            raise AttributeError("Alpha引擎缺少compute_all_alphas方法")
        
        alpha_features = self.alpha_engine.compute_all_alphas(feature_data)
        return {'alpha_features': alpha_features, 'alpha_count': len(alpha_features.columns)}
        
    except Exception as e:
        logger.warning(f"Alpha训练失败: {e}")
        return None

def _safe_ltr_training(self, feature_data):
    """安全的LTR训练"""
    try:
        # 检查LTR训练条件
        if 'date' not in feature_data.columns:
            raise ValueError("LTR训练需要date列")
        
        unique_dates = feature_data['date'].nunique()
        if unique_dates < 3:
            raise ValueError(f"LTR训练需要至少3个不同日期，当前只有{unique_dates}个")
        
        # LTR训练逻辑
        # ... 实际LTR训练代码
        
        return {'ltr_models': [], 'ltr_performance': {}}
        
    except Exception as e:
        logger.warning(f"LTR训练失败: {e}")
        return None

def _safe_regime_training(self, feature_data):
    """安全的Regime训练"""
    try:
        # Regime训练逻辑
        return {'regime_models': {}, 'regime_performance': {}}
    except Exception as e:
        logger.warning(f"Regime训练失败: {e}")
        return None

def _safe_traditional_training(self, feature_data, current_ticker):
    """安全的传统模型训练"""
    try:
        # 传统模型训练逻辑
        return {'model_performance': {}, 'oof_predictions': {}}
    except Exception as e:
        logger.warning(f"传统模型训练失败: {e}")
        return None
