"""
统一结果对象和装饰器框架
解决Mixed Success Logging问题
"""
import logging
import functools
from typing import Any, Optional, Union, Dict, List
from dataclasses import dataclass
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)

class ResultStatus(Enum):
    SUCCESS = "success"
    WARNING = "warning"  
    FAILURE = "failure"
    FALLBACK = "fallback"

@dataclass
class OperationResult:
    """统一结果对象，带强制success语义"""
    status: ResultStatus
    data: Any = None
    message: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_success(self) -> bool:
        return self.status == ResultStatus.SUCCESS
    
    @property
    def is_failure(self) -> bool:
        return self.status == ResultStatus.FAILURE
    
    @property
    def is_warning(self) -> bool:
        return self.status == ResultStatus.WARNING
    
    @property
    def is_fallback(self) -> bool:
        return self.status == ResultStatus.FALLBACK
    
    def log_result(self, logger_instance: logging.Logger = None):
        """强制只有满足success条件才允许打印✅OK"""
        if logger_instance is None:
            logger_instance = logger
            
        if self.is_success:
            logger_instance.info(f"✅ [OK] {self.message}")
        elif self.is_warning:
            logger_instance.warning(f"⚠️ [WARN] {self.message}")
        elif self.is_fallback:
            logger_instance.warning(f"🔄 [FALLBACK] {self.message}")
        else:
            logger_instance.error(f"❌ [ERROR] {self.message}")

def validate_result_logging(validate_success_condition=None):
    """装饰器：强制结果验证和统一日志格式"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 如果函数直接返回OperationResult，使用它
                if isinstance(result, OperationResult):
                    result.log_result()
                    return result
                
                # 否则，根据验证条件包装结果
                if validate_success_condition:
                    is_success = validate_success_condition(result)
                else:
                    # 默认验证逻辑
                    is_success = _default_success_validation(result)
                
                if is_success:
                    op_result = OperationResult(
                        status=ResultStatus.SUCCESS,
                        data=result,
                        message=f"{func.__name__} 执行成功"
                    )
                else:
                    op_result = OperationResult(
                        status=ResultStatus.FAILURE,
                        data=result,
                        message=f"{func.__name__} 执行失败或返回无效结果"
                    )
                
                op_result.log_result()
                return op_result
                
            except Exception as e:
                error_result = OperationResult(
                    status=ResultStatus.FAILURE,
                    data=None,
                    message=f"{func.__name__} 执行异常: {str(e)}",
                    metadata={"exception": str(e)}
                )
                error_result.log_result()
                return error_result
        
        return wrapper
    return decorator

def _default_success_validation(result) -> bool:
    """默认成功验证逻辑"""
    if result is None:
        return False
    if isinstance(result, pd.DataFrame):
        return not result.empty and result.shape[1] > 0
    if isinstance(result, (list, tuple)):
        return len(result) > 0
    if isinstance(result, dict):
        return len(result) > 0
    return True

# 预定义验证条件
def alpha_signals_validation(result) -> bool:
    """Alpha信号特定验证"""
    if result is None:
        return False
    if isinstance(result, pd.DataFrame):
        return not result.empty and result.shape[1] > 0 and result.shape[0] > 10
    return False

def feature_integration_validation(result) -> bool:
    """特征集成验证"""
    if result is None:
        return False
    if isinstance(result, pd.DataFrame):
        return not result.empty and result.shape[1] >= 3  # 至少3个特征
    return False

def model_training_validation(result) -> bool:
    """模型训练验证"""
    if result is None:
        return False
    if hasattr(result, 'score') or hasattr(result, 'predict'):
        return True
    if isinstance(result, dict) and 'model' in result:
        return result['model'] is not None
    return False
