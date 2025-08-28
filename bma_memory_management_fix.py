
import gc
import psutil
from functools import wraps
from typing import Any, Callable

class MemoryManager:
    """内存管理器 - 预防内存泄漏"""
    
    def __init__(self, memory_threshold: float = 80.0, auto_cleanup: bool = True):
        self.memory_threshold = memory_threshold  # 内存使用率阈值(%)
        self.auto_cleanup = auto_cleanup
        self.logger = logging.getLogger(__name__)
        
    def get_memory_usage(self) -> float:
        """获取当前内存使用率"""
        return psutil.virtual_memory().percent
    
    def check_memory_pressure(self) -> bool:
        """检查内存压力"""
        usage = self.get_memory_usage()
        return usage > self.memory_threshold
    
    def force_cleanup(self):
        """强制内存清理"""
        gc.collect()
        self.logger.debug(f"强制内存清理完成, 当前使用率: {self.get_memory_usage():.1f}%")
    
    def memory_safe_wrapper(self, func: Callable) -> Callable:
        """内存安全装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 训练前检查
            initial_memory = self.get_memory_usage()
            if initial_memory > self.memory_threshold:
                self.logger.warning(f"内存使用率过高: {initial_memory:.1f}%, 执行清理")
                self.force_cleanup()
            
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 训练后检查
                final_memory = self.get_memory_usage()
                memory_increase = final_memory - initial_memory
                
                if memory_increase > 20:  # 内存增长超过20%
                    self.logger.warning(f"内存增长异常: +{memory_increase:.1f}%")
                    
                if self.auto_cleanup and final_memory > self.memory_threshold:
                    self.force_cleanup()
                
                return result
                
            except MemoryError as e:
                self.logger.error(f"内存不足错误: {e}")
                self.force_cleanup()
                raise
            except Exception as e:
                # 异常时也要清理内存
                if self.auto_cleanup:
                    self.force_cleanup()
                raise
            finally:
                # 确保大对象被释放
                if 'args' in locals():
                    for arg in args:
                        if hasattr(arg, '__del__'):
                            del arg
                
        return wrapper

# 内存安全的训练方法
def memory_safe_train_enhanced_models(self, feature_data: pd.DataFrame, 
                                    current_ticker: str = None) -> Dict[str, Any]:
    """内存安全的增强模型训练"""
    
    memory_manager = MemoryManager(memory_threshold=75.0)
    
    @memory_manager.memory_safe_wrapper
    def _safe_training():
        # 原有的训练逻辑
        return self._original_train_enhanced_models(feature_data, current_ticker)
    
    try:
        # 训练前内存状态
        initial_memory = memory_manager.get_memory_usage()
        logger.info(f"开始训练 - 初始内存使用: {initial_memory:.1f}%")
        
        # 数据预处理优化
        if hasattr(feature_data, 'memory_usage'):
            data_memory = feature_data.memory_usage(deep=True).sum() / 1024**2  # MB
            logger.info(f"特征数据大小: {data_memory:.1f}MB")
            
            # 如果数据太大，进行分批处理
            if data_memory > 500:  # 超过500MB
                logger.warning("数据量过大，启用分批处理模式")
                return self._batch_training_mode(feature_data, current_ticker)
        
        # 安全训练
        result = _safe_training()
        
        # 训练后清理
        final_memory = memory_manager.get_memory_usage()
        logger.info(f"训练完成 - 最终内存使用: {final_memory:.1f}%")
        
        return result
        
    except Exception as e:
        logger.error(f"内存安全训练失败: {e}")
        memory_manager.force_cleanup()
        raise

def _batch_training_mode(self, feature_data: pd.DataFrame, current_ticker: str):
    """分批训练模式 - 处理大数据集"""
    logger.info("启用分批训练模式")
    
    # 按ticker分批处理
    unique_tickers = feature_data['ticker'].unique() if 'ticker' in feature_data.columns else [current_ticker]
    batch_size = max(1, len(unique_tickers) // 4)  # 分4批
    
    batch_results = []
    
    for i in range(0, len(unique_tickers), batch_size):
        batch_tickers = unique_tickers[i:i+batch_size]
        
        if 'ticker' in feature_data.columns:
            batch_data = feature_data[feature_data['ticker'].isin(batch_tickers)]
        else:
            # 按行分批
            start_idx = i * len(feature_data) // len(unique_tickers)
            end_idx = min((i + batch_size) * len(feature_data) // len(unique_tickers), len(feature_data))
            batch_data = feature_data.iloc[start_idx:end_idx]
        
        logger.info(f"处理批次 {i//batch_size + 1}: {len(batch_data)} 样本")
        
        # 训练单个批次
        batch_result = self._original_train_enhanced_models(batch_data, current_ticker)
        batch_results.append(batch_result)
        
        # 批次间清理内存
        del batch_data
        gc.collect()
    
    # 合并批次结果
    return self._merge_batch_results(batch_results)
