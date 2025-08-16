#!/usr/bin/env python3
"""
安全内存管理策略 - 防止过度释放导致结果失败
智能识别关键对象，只清理安全的临时变量
"""

import pandas as pd
import numpy as np
import gc
import logging
import weakref
from typing import Dict, List, Any, Set
import pickle
import os

logger = logging.getLogger(__name__)

class SafeMemoryManager:
    """安全内存管理器 - 保护关键结果不被误删"""
    
    def __init__(self):
        # 保护的对象类型（不允许清理）
        self.protected_attributes = {
            'raw_data',           # 原始数据
            'feature_data',       # 特征数据  
            'latest_ticker_predictions',  # 最新预测
            'portfolio_weights',  # 投资组合权重
            'traditional_models', # 训练好的模型
            'alpha_engine',       # Alpha引擎
            'ltr_bma',           # LTR模型
            'target_engineer',   # 目标工程器
            'risk_model_results', # 风险模型结果
            'bucket_models',     # 桶模型
            'ranker_models',     # 排序模型
            'isotonic_calibrators' # 校准器
        }
        
        # 可安全清理的临时对象
        self.safe_to_clean = {
            '_temp_',            # 临时变量前缀
            'df_copy',           # DataFrame副本
            'batch_',            # 批处理变量
            'chunk_',            # 分块变量
            'intermediate_',     # 中间结果
            '_cache_',           # 缓存变量
            'temp_features',     # 临时特征
            'temp_data'          # 临时数据
        }
        
        # 内存使用阈值
        self.memory_warning_threshold = 2048  # 2GB
        self.memory_critical_threshold = 4096  # 4GB
        
        # 保存关键对象的备份信息
        self.object_registry = {}
        self.backup_enabled = True
        
    def assess_memory_safety(self, obj, obj_name: str) -> str:
        """评估对象的内存清理安全性"""
        
        # 1. 检查保护属性
        for protected in self.protected_attributes:
            if protected in obj_name:
                return "PROTECTED"  # 绝对不允许清理
        
        # 2. 检查安全清理列表
        for safe in self.safe_to_clean:
            if safe in obj_name:
                return "SAFE"  # 可以安全清理
        
        # 3. 检查对象类型
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            if obj_name.endswith('_result'):
                return "PROTECTED"  # 结果对象保护
            elif obj_name.startswith('temp_') or obj_name.endswith('_temp'):
                return "SAFE"  # 临时对象可清理
            elif len(obj) < 100:  # 小对象通常是临时的
                return "SAFE"
            else:
                return "CAUTIOUS"  # 需要谨慎处理
        
        # 4. 检查对象大小和重要性
        try:
            obj_size = self._estimate_object_size(obj)
            if obj_size > 100:  # 100MB以上的大对象
                if any(keyword in obj_name.lower() for keyword in ['model', 'result', 'prediction', 'weight']):
                    return "PROTECTED"  # 大型重要对象
                else:
                    return "CAUTIOUS"  # 大型临时对象
        except:
            pass
        
        return "CAUTIOUS"  # 默认谨慎处理
    
    def _estimate_object_size(self, obj) -> float:
        """估算对象大小(MB)"""
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.memory_usage(deep=True).sum() / 1024 / 1024
            elif hasattr(obj, '__sizeof__'):
                return obj.__sizeof__() / 1024 / 1024
            else:
                # 使用pickle估算
                return len(pickle.dumps(obj, protocol=-1)) / 1024 / 1024
        except:
            return 0.0
    
    def safe_cleanup_object(self, parent_obj, obj_name: str, force: bool = False) -> bool:
        """安全清理指定对象"""
        if not hasattr(parent_obj, obj_name):
            return False
        
        obj = getattr(parent_obj, obj_name)
        safety_level = self.assess_memory_safety(obj, obj_name)
        
        if safety_level == "PROTECTED" and not force:
            logger.debug(f"对象 {obj_name} 受保护，跳过清理")
            return False
        
        elif safety_level == "SAFE":
            try:
                # 安全清理
                delattr(parent_obj, obj_name)
                logger.debug(f"✅ 安全清理对象: {obj_name}")
                return True
            except Exception as e:
                logger.warning(f"清理对象 {obj_name} 失败: {e}")
                return False
        
        elif safety_level == "CAUTIOUS":
            if force:
                # 创建备份（如果启用）
                if self.backup_enabled:
                    self._create_backup(obj, obj_name)
                
                try:
                    delattr(parent_obj, obj_name)
                    logger.info(f"⚠️ 谨慎清理对象: {obj_name} (已备份)")
                    return True
                except Exception as e:
                    logger.warning(f"清理对象 {obj_name} 失败: {e}")
                    return False
            else:
                logger.debug(f"对象 {obj_name} 需要谨慎处理，跳过清理 (使用force=True强制)")
                return False
        
        return False
    
    def _create_backup(self, obj, obj_name: str):
        """为重要对象创建备份"""
        try:
            backup_info = {
                'name': obj_name,
                'type': type(obj).__name__,
                'size_mb': self._estimate_object_size(obj),
                'timestamp': pd.Timestamp.now()
            }
            
            # 只备份元信息，不备份实际数据（避免内存翻倍）
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                backup_info.update({
                    'shape': getattr(obj, 'shape', None),
                    'columns': getattr(obj, 'columns', None).tolist() if hasattr(obj, 'columns') else None,
                    'index_type': type(getattr(obj, 'index', None)).__name__
                })
            
            self.object_registry[obj_name] = backup_info
            logger.debug(f"对象 {obj_name} 备份信息已记录")
            
        except Exception as e:
            logger.warning(f"创建备份信息失败 {obj_name}: {e}")
    
    def smart_memory_cleanup(self, parent_obj, exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """智能内存清理 - 只清理安全的对象"""
        exclude_patterns = exclude_patterns or []
        
        cleanup_stats = {
            'cleaned_objects': [],
            'protected_objects': [],
            'failed_cleanups': [],
            'memory_freed_mb': 0
        }
        
        # 获取当前内存使用
        initial_memory = self._get_current_memory()
        
        # 获取所有属性
        all_attributes = [attr for attr in dir(parent_obj) 
                         if not attr.startswith('__') and hasattr(parent_obj, attr)]
        
        # 按安全级别分类
        safe_objects = []
        cautious_objects = []
        protected_objects = []
        
        for attr_name in all_attributes:
            # 跳过排除的模式
            if any(pattern in attr_name for pattern in exclude_patterns):
                continue
            
            try:
                attr_value = getattr(parent_obj, attr_name)
                
                # 跳过方法和属性
                if callable(attr_value) or isinstance(attr_value, property):
                    continue
                
                safety_level = self.assess_memory_safety(attr_value, attr_name)
                
                if safety_level == "SAFE":
                    safe_objects.append(attr_name)
                elif safety_level == "CAUTIOUS":
                    cautious_objects.append(attr_name)
                else:  # PROTECTED
                    protected_objects.append(attr_name)
                    
            except Exception as e:
                logger.debug(f"评估对象 {attr_name} 时出错: {e}")
        
        # 1. 优先清理安全对象
        for obj_name in safe_objects:
            if self.safe_cleanup_object(parent_obj, obj_name):
                cleanup_stats['cleaned_objects'].append(obj_name)
        
        # 2. 检查内存使用，决定是否清理谨慎对象
        current_memory = self._get_current_memory()
        if current_memory > self.memory_warning_threshold:
            logger.info(f"内存使用 {current_memory:.1f}MB 超过警告阈值，清理谨慎对象")
            
            for obj_name in cautious_objects[:5]:  # 最多清理5个谨慎对象
                if self.safe_cleanup_object(parent_obj, obj_name, force=True):
                    cleanup_stats['cleaned_objects'].append(obj_name)
        
        # 3. 记录保护对象
        cleanup_stats['protected_objects'] = protected_objects
        
        # 4. 计算释放的内存
        final_memory = self._get_current_memory()
        cleanup_stats['memory_freed_mb'] = max(0, initial_memory - final_memory)
        
        # 5. 执行垃圾回收
        for _ in range(2):
            collected = gc.collect()
            if collected == 0:
                break
        
        logger.info(f"智能内存清理完成: 清理{len(cleanup_stats['cleaned_objects'])}个对象, "
                   f"保护{len(cleanup_stats['protected_objects'])}个对象, "
                   f"释放{cleanup_stats['memory_freed_mb']:.1f}MB内存")
        
        return cleanup_stats
    
    def _get_current_memory(self) -> float:
        """获取当前内存使用(MB)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def emergency_cleanup(self, parent_obj) -> Dict[str, Any]:
        """紧急内存清理 - 在内存严重不足时使用"""
        logger.warning("执行紧急内存清理")
        
        # 强制清理所有非关键对象
        cleanup_stats = self.smart_memory_cleanup(
            parent_obj, 
            exclude_patterns=['engine', 'model', 'optimizer', 'results']
        )
        
        # 强制垃圾回收
        for _ in range(5):
            collected = gc.collect()
            if collected == 0:
                break
        
        return cleanup_stats
    
    def get_memory_report(self, parent_obj) -> Dict[str, Any]:
        """获取内存使用报告"""
        current_memory = self._get_current_memory()
        
        # 分析对象大小
        object_sizes = {}
        total_estimated = 0
        
        for attr_name in dir(parent_obj):
            if not attr_name.startswith('__'):
                try:
                    attr_value = getattr(parent_obj, attr_name)
                    if not callable(attr_value):
                        size_mb = self._estimate_object_size(attr_value)
                        if size_mb > 1:  # 只报告1MB以上的对象
                            object_sizes[attr_name] = size_mb
                            total_estimated += size_mb
                except:
                    pass
        
        # 排序
        sorted_objects = sorted(object_sizes.items(), key=lambda x: x[1], reverse=True)
        
        report = {
            'current_memory_mb': current_memory,
            'estimated_object_memory_mb': total_estimated,
            'large_objects': sorted_objects[:10],  # 前10个最大对象
            'memory_status': self._get_memory_status(current_memory),
            'cleanup_recommendations': self._get_cleanup_recommendations(sorted_objects)
        }
        
        return report
    
    def _get_memory_status(self, current_memory: float) -> str:
        """获取内存状态"""
        if current_memory > self.memory_critical_threshold:
            return "CRITICAL"
        elif current_memory > self.memory_warning_threshold:
            return "WARNING"
        else:
            return "NORMAL"
    
    def _get_cleanup_recommendations(self, sorted_objects: List) -> List[str]:
        """获取清理建议"""
        recommendations = []
        
        for obj_name, size_mb in sorted_objects[:5]:
            safety_level = self.assess_memory_safety(None, obj_name)
            if safety_level in ["SAFE", "CAUTIOUS"]:
                recommendations.append(f"考虑清理 {obj_name} ({size_mb:.1f}MB)")
        
        return recommendations


def create_safe_memory_manager() -> SafeMemoryManager:
    """创建安全内存管理器"""
    return SafeMemoryManager()


# 使用示例
if __name__ == "__main__":
    # 测试安全内存管理
    manager = SafeMemoryManager()
    
    # 创建测试对象
    class TestModel:
        def __init__(self):
            self.raw_data = pd.DataFrame({'test': range(1000)})  # 保护对象
            self.temp_data = pd.DataFrame({'temp': range(500)})   # 临时对象
            self.model_results = {'accuracy': 0.95}              # 保护对象
            self.temp_features = np.random.randn(100, 10)        # 临时对象
    
    test_obj = TestModel()
    
    # 获取内存报告
    report = manager.get_memory_report(test_obj)
    print("内存报告:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # 执行安全清理
    cleanup_stats = manager.smart_memory_cleanup(test_obj)
    print(f"\n清理统计: {cleanup_stats}")
    
    print("安全内存管理测试完成")