#!/usr/bin/env python3
"""
内存优化的分批训练器
解决2800股票训练内存不足问题
"""

import gc
import psutil
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryOptimizedTrainer:
    """内存优化的分批训练器"""
    
    def __init__(self, 
                 batch_size: int = 400,
                 memory_limit_gb: float = 3.0,
                 enable_gc_aggressive: bool = True,
                 cache_dir: str = "cache/training",
                 force_retrain: bool = False):
        """
        初始化内存优化训练器
        
        Args:
            batch_size: 每批训练股票数量
            memory_limit_gb: 内存使用限制(GB)
            enable_gc_aggressive: 启用激进垃圾回收
            cache_dir: 缓存目录
            force_retrain: 强制重新训练，忽略缓存
        """
        self.batch_size = batch_size
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.enable_gc_aggressive = enable_gc_aggressive
        self.force_retrain = force_retrain
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.current_batch = 0
        self.total_batches = 0
        self.batch_results = []
        self.training_start_time = None
        
        # 内存监控
        self.memory_warnings = 0
        self.gc_collections = 0
        
    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'system_used_gb': (memory.total - memory.available) / (1024**3),
            'system_available_gb': memory.available / (1024**3),
            'system_percent': memory.percent,
            'process_rss_gb': process_memory.rss / (1024**3),
            'process_vms_gb': process_memory.vms / (1024**3)
        }
    
    def check_memory_safety(self) -> bool:
        """检查内存是否安全"""
        memory_info = self.get_memory_usage()
        current_usage_bytes = memory_info['process_rss_gb'] * (1024**3)
        
        if current_usage_bytes > self.memory_limit_bytes * 0.8:
            self.memory_warnings += 1
            logger.warning(f"内存使用接近限制: {memory_info['process_rss_gb']:.2f}GB / {self.memory_limit_bytes/(1024**3):.2f}GB")
            return False
        return True
    
    def force_garbage_collection(self):
        """强制垃圾回收"""
        if self.enable_gc_aggressive:
            for _ in range(3):
                collected = gc.collect()
                self.gc_collections += collected
            
            # 强制释放numpy数组
            try:
                import numpy as np
                np.seterr(all='ignore')
            except:
                pass
    
    def split_universe_into_batches(self, universe: List[str]) -> List[List[str]]:
        """将股票池分割为批次"""
        batches = []
        for i in range(0, len(universe), self.batch_size):
            batch = universe[i:i + self.batch_size]
            batches.append(batch)
        
        self.total_batches = len(batches)
        logger.info(f"股票池分割为 {self.total_batches} 批次，每批 {self.batch_size} 股票")
        return batches
    
    def save_batch_result(self, batch_idx: int, result: Dict[str, Any]):
        """保存批次结果到缓存"""
        cache_file = self.cache_dir / f"batch_{batch_idx:03d}_result.json"
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                serializable_result[key] = value.to_dict()
            else:
                serializable_result[key] = value
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"批次 {batch_idx} 结果已保存: {cache_file}")
    
    def load_batch_result(self, batch_idx: int) -> Optional[Dict[str, Any]]:
        """从缓存加载批次结果"""
        cache_file = self.cache_dir / f"batch_{batch_idx:03d}_result.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            logger.debug(f"批次 {batch_idx} 结果已加载: {cache_file}")
            return result
        except Exception as e:
            logger.error(f"加载批次结果失败: {e}")
            return None
    
    def train_batch(self, 
                   batch_tickers: List[str], 
                   batch_idx: int,
                   model_trainer_func: callable,
                   **kwargs) -> Dict[str, Any]:
        """训练单个批次"""
        logger.info(f"开始训练批次 {batch_idx + 1}/{self.total_batches} ({len(batch_tickers)} 股票)")
        
        # 检查缓存 (除非强制重新训练)
        if not self.force_retrain:
            cached_result = self.load_batch_result(batch_idx)
            if cached_result is not None:
                logger.info(f"使用缓存的批次 {batch_idx} 结果")
                return cached_result
        else:
            logger.info(f"强制重新训练批次 {batch_idx}")
        
        # 内存安全检查
        if not self.check_memory_safety():
            self.force_garbage_collection()
            time.sleep(1)  # 给GC时间清理
        
        batch_start_time = time.time()
        
        try:
            # 调用实际的模型训练函数
            result = model_trainer_func(batch_tickers, **kwargs)
            
            # 添加批次元数据
            result['batch_metadata'] = {
                'batch_idx': batch_idx,
                'tickers_count': len(batch_tickers),
                'training_time_seconds': time.time() - batch_start_time,
                'memory_usage': self.get_memory_usage()
            }
            
            # 保存结果
            self.save_batch_result(batch_idx, result)
            
            # 强制清理内存
            self.force_garbage_collection()
            
            logger.info(f"批次 {batch_idx + 1} 训练完成，用时 {time.time() - batch_start_time:.1f}秒")
            return result
            
        except Exception as e:
            logger.error(f"批次 {batch_idx} 训练失败: {e}")
            # 即使失败也要清理内存
            self.force_garbage_collection()
            raise
    
    def combine_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并所有批次结果"""
        logger.info("开始合并批次结果...")
        
        combined_result = {
            'predictions': {},
            'model_performance': {},
            'feature_importance': {},
            'batch_summary': [],
            'total_training_time': 0,
            'total_stocks': 0
        }
        
        for batch_result in batch_results:
            if not batch_result:
                continue
                
            # 合并预测结果
            if 'predictions' in batch_result:
                combined_result['predictions'].update(batch_result['predictions'])
            
            # 合并模型性能
            if 'model_performance' in batch_result:
                for model_name, performance in batch_result['model_performance'].items():
                    if model_name not in combined_result['model_performance']:
                        combined_result['model_performance'][model_name] = []
                    combined_result['model_performance'][model_name].append(performance)
            
            # 合并特征重要性 - 修复：正确处理字典类型的特征重要性
            if 'feature_importance' in batch_result:
                for ticker, importance_data in batch_result['feature_importance'].items():
                    if isinstance(importance_data, dict):
                        # 如果是字典，直接合并到combined_result中
                        combined_result['feature_importance'][ticker] = importance_data
                    else:
                        # 如果是数值，按原来的逻辑处理
                        if ticker not in combined_result['feature_importance']:
                            combined_result['feature_importance'][ticker] = []
                        combined_result['feature_importance'][ticker].append(importance_data)
            
            # 收集批次摘要 - 改进元数据处理
            if 'batch_metadata' in batch_result:
                metadata = batch_result['batch_metadata']
                combined_result['batch_summary'].append(metadata)
                
                # 安全地累加训练时间和股票数
                training_time = metadata.get('training_time_seconds', 0)
                tickers_count = metadata.get('tickers_count', 0)
                
                if isinstance(training_time, (int, float)):
                    combined_result['total_training_time'] += training_time
                if isinstance(tickers_count, (int, float)):
                    combined_result['total_stocks'] += tickers_count
        
        # 计算平均性能指标 - 改进类型处理
        for model_name, performances in combined_result['model_performance'].items():
            if performances:
                # 确保performances是数值列表
                numeric_performances = []
                for perf in performances:
                    if isinstance(perf, (int, float)) and not np.isnan(perf):
                        numeric_performances.append(float(perf))
                    elif isinstance(perf, dict):
                        # 尝试多个可能的数值键
                        for key in ['confidence', 'score', 'accuracy', 'value']:
                            if key in perf and isinstance(perf[key], (int, float)) and not np.isnan(perf[key]):
                                numeric_performances.append(float(perf[key]))
                                break
                        else:
                            numeric_performances.append(0.5)  # 默认值
                    else:
                        numeric_performances.append(0.5)  # 默认值
                
                if numeric_performances:
                    combined_result['model_performance'][model_name] = {
                        'mean_score': float(np.mean(numeric_performances)),
                        'std_score': float(np.std(numeric_performances)),
                        'count': len(numeric_performances),
                        'scores': numeric_performances
                    }
                else:
                    # 如果没有有效数值，提供默认值
                    combined_result['model_performance'][model_name] = {
                        'mean_score': 0.5,
                        'std_score': 0.0,
                        'count': 0,
                        'scores': []
                    }
        
        # 计算平均特征重要性 - 修复：只处理数值类型的特征重要性
        for ticker, importances in combined_result['feature_importance'].items():
            if isinstance(importances, list):
                # 只处理数值列表
                numeric_importances = []
                for imp in importances:
                    if isinstance(imp, (int, float)):
                        numeric_importances.append(imp)
                    else:
                        numeric_importances.append(0.0)  # 默认值
                
                if numeric_importances:
                    combined_result['feature_importance'][ticker] = {
                        'mean_importance': np.mean(numeric_importances),
                        'std_importance': np.std(numeric_importances),
                        'importances': numeric_importances
                    }
            # 如果已经是字典，保持不变
        
        logger.info(f"合并完成: {combined_result['total_stocks']} 股票，总用时 {combined_result['total_training_time']:.1f}秒")
        return combined_result
    
    def train_universe(self, 
                      universe: List[str],
                      model_trainer_func: callable,
                      **kwargs) -> Dict[str, Any]:
        """训练整个股票池"""
        self.training_start_time = time.time()
        logger.info(f"开始内存优化训练: {len(universe)} 股票，批次大小 {self.batch_size}")
        
        # 分割为批次
        batches = self.split_universe_into_batches(universe)
        
        # 逐批训练
        self.batch_results = []
        for batch_idx, batch_tickers in enumerate(batches):
            self.current_batch = batch_idx
            
            try:
                batch_result = self.train_batch(
                    batch_tickers, 
                    batch_idx, 
                    model_trainer_func,
                    **kwargs
                )
                self.batch_results.append(batch_result)
                
                # 显示进度
                progress = (batch_idx + 1) / len(batches) * 100
                elapsed_time = time.time() - self.training_start_time
                estimated_total = elapsed_time / (batch_idx + 1) * len(batches)
                remaining_time = estimated_total - elapsed_time
                
                logger.info(f"进度: {progress:.1f}% ({batch_idx + 1}/{len(batches)})，"
                           f"已用时 {elapsed_time/60:.1f}分钟，预计剩余 {remaining_time/60:.1f}分钟")
                
            except Exception as e:
                logger.error(f"批次 {batch_idx} 训练失败，跳过: {e}")
                self.batch_results.append(None)
                continue
        
        # 合并结果
        final_result = self.combine_batch_results(self.batch_results)
        
        # 添加训练统计信息
        final_result['training_statistics'] = {
            'total_time_minutes': (time.time() - self.training_start_time) / 60,
            'total_batches': len(batches),
            'successful_batches': len([r for r in self.batch_results if r is not None]),
            'failed_batches': len([r for r in self.batch_results if r is None]),
            'memory_warnings': self.memory_warnings,
            'gc_collections': self.gc_collections,
            'final_memory_usage': self.get_memory_usage()
        }
        
        logger.info(f"训练完成! 总用时 {final_result['training_statistics']['total_time_minutes']:.1f}分钟")
        return final_result


def create_memory_optimized_trainer(**kwargs) -> MemoryOptimizedTrainer:
    """创建内存优化训练器的工厂函数"""
    return MemoryOptimizedTrainer(**kwargs)