#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Out-of-Sample Manager
真实样本外管理器 - 管理和维护真实的OOS预测历史，用于BMA权重优化

核心功能:
1. 收集和存储各个模型在真实OOS折上的预测结果
2. 维护历史性能记录，避免使用模拟或泄露数据
3. 为BMA权重更新提供可靠的历史OOS数据
4. 支持时间窗口和最小折数筛选
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
import json
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class OOSFoldRecord:
    """单个OOS折记录"""
    fold_id: str
    timestamp: datetime
    model_predictions: Dict[str, pd.Series]  # model_name -> predictions
    actuals: pd.Series
    fold_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """计算并存储基本指标"""
        if not self.fold_metrics and len(self.model_predictions) > 0 and len(self.actuals) > 0:
            from scipy.stats import pearsonr, spearmanr
            
            for model_name, predictions in self.model_predictions.items():
                try:
                    # 确保索引对齐
                    aligned_actuals = self.actuals.loc[predictions.index] if hasattr(self.actuals, 'loc') else self.actuals
                    
                    if len(predictions) > 10:  # 最小样本要求
                        ic, _ = pearsonr(aligned_actuals, predictions)
                        rank_ic, _ = spearmanr(aligned_actuals, predictions)
                        mse = np.mean((aligned_actuals - predictions) ** 2)
                        
                        self.fold_metrics[f"{model_name}_ic"] = ic if not np.isnan(ic) else 0.0
                        self.fold_metrics[f"{model_name}_rank_ic"] = rank_ic if not np.isnan(rank_ic) else 0.0
                        self.fold_metrics[f"{model_name}_mse"] = mse if not np.isnan(mse) else 1.0
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate metrics for {model_name}: {e}")
                    self.fold_metrics[f"{model_name}_ic"] = 0.0
                    self.fold_metrics[f"{model_name}_rank_ic"] = 0.0
                    self.fold_metrics[f"{model_name}_mse"] = 1.0

class RealOOSManager:
    """真实样本外管理器"""
    
    def __init__(self, storage_path: str = "cache/real_oos_history", 
                 max_history_days: int = 180, 
                 auto_cleanup: bool = True):
        """
        初始化真实OOS管理器
        
        Args:
            storage_path: 存储路径
            max_history_days: 最大历史记录天数
            auto_cleanup: 是否自动清理过期数据
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_history_days = max_history_days
        self.auto_cleanup = auto_cleanup
        
        # 内存缓存
        self.fold_history: List[OOSFoldRecord] = []
        self.model_performance_cache: Dict[str, Dict] = {}
        
        # 加载历史数据
        self._load_history()
        
        logger.info(f"RealOOSManager初始化完成，历史记录: {len(self.fold_history)}个折")
    
    def add_fold_predictions(self, fold_id: str, 
                           model_predictions: Dict[str, pd.Series],
                           actuals: pd.Series,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        添加一个折的预测结果
        
        Args:
            fold_id: 折ID，唯一标识符
            model_predictions: 模型预测字典 {model_name: predictions}
            actuals: 真实标签
            metadata: 元数据
            
        Returns:
            bool: 是否成功添加
        """
        try:
            # 检查是否已存在
            if any(record.fold_id == fold_id for record in self.fold_history):
                logger.warning(f"折 {fold_id} 已存在，跳过添加")
                return False
            
            # 验证数据有效性
            if not model_predictions or len(actuals) == 0:
                logger.warning(f"折 {fold_id} 数据无效，跳过添加")
                return False
            
            # 验证预测和实际值的对齐
            for model_name, predictions in model_predictions.items():
                if len(predictions) != len(actuals):
                    logger.warning(f"模型 {model_name} 预测长度与实际值不匹配")
                    continue
            
            # 创建记录
            fold_record = OOSFoldRecord(
                fold_id=fold_id,
                timestamp=datetime.now(),
                model_predictions=model_predictions.copy(),
                actuals=actuals.copy(),
                metadata=metadata or {}
            )
            
            # 添加到历史
            self.fold_history.append(fold_record)
            
            # 异步持久化
            self._save_fold_record(fold_record)
            
            # 更新性能缓存
            self._update_performance_cache(fold_record)
            
            # 自动清理
            if self.auto_cleanup:
                self._cleanup_expired_data()
            
            logger.info(f"成功添加折 {fold_id}，包含 {len(model_predictions)} 个模型，"
                       f"{len(actuals)} 个样本")
            
            return True
            
        except Exception as e:
            logger.error(f"添加折 {fold_id} 失败: {e}")
            return False
    
    def get_bma_update_data(self, min_folds: int = 3, 
                          lookback_days: int = 30,
                          models: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        获取用于BMA权重更新的OOS数据
        
        Args:
            min_folds: 最小折数要求
            lookback_days: 回看天数
            models: 指定模型列表，None则使用所有模型
            
        Returns:
            DataFrame: 包含target和各模型预测列的数据，如果数据不足返回None
        """
        try:
            # 筛选时间范围内的折
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_folds = [
                fold for fold in self.fold_history 
                if fold.timestamp >= cutoff_date
            ]
            
            if len(recent_folds) < min_folds:
                logger.warning(f"OOS折数不足: {len(recent_folds)} < {min_folds}")
                return None
            
            # 收集所有模型名称
            all_models = set()
            for fold in recent_folds:
                all_models.update(fold.model_predictions.keys())
            
            if models is not None:
                all_models = all_models.intersection(set(models))
            
            if not all_models:
                logger.warning("没有可用的模型数据")
                return None
            
            # 构建BMA更新数据
            all_data = []
            
            for fold in recent_folds:
                # 为每个折创建一个包含所有模型预测的数据帧
                fold_data_dict = {'target': fold.actuals}
                
                for model_name in all_models:
                    if model_name in fold.model_predictions:
                        predictions = fold.model_predictions[model_name]
                        # 确保索引对齐
                        if hasattr(predictions, 'index') and hasattr(fold.actuals, 'index'):
                            common_idx = predictions.index.intersection(fold.actuals.index)
                            if len(common_idx) > 0:
                                fold_data_dict[f'{model_name}_pred'] = predictions.loc[common_idx]
                                fold_data_dict['target'] = fold.actuals.loc[common_idx]
                        else:
                            fold_data_dict[f'{model_name}_pred'] = predictions
                
                if len(fold_data_dict) > 1:  # 至少有target + 一个模型
                    fold_data = pd.DataFrame(fold_data_dict)
                    fold_data['fold_id'] = fold.fold_id
                    fold_data['timestamp'] = fold.timestamp
                    all_data.append(fold_data)
            
            if not all_data:
                logger.warning("无法构建BMA更新数据")
                return None
            
            # 合并所有数据
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # 透视表格式：每行是一个样本，列包含target和各模型预测
            pivot_data = {}
            pivot_data['target'] = combined_data['target']
            
            for model_name in all_models:
                pred_col = f'{model_name}_pred'
                if pred_col in combined_data.columns:
                    pivot_data[pred_col] = combined_data[pred_col]
            
            result = pd.DataFrame(pivot_data)
            result = result.dropna()  # 移除任何包含NaN的行
            
            logger.info(f"构建BMA更新数据成功: {len(result)} 样本, "
                       f"{len(all_models)} 模型, {len(recent_folds)} 折")
            
            return result
            
        except Exception as e:
            logger.error(f"获取BMA更新数据失败: {e}")
            return None
    
    def get_model_performance_summary(self, lookback_days: int = 30) -> Dict[str, Dict[str, float]]:
        """
        获取模型性能摘要
        
        Args:
            lookback_days: 回看天数
            
        Returns:
            Dict: {model_name: {metric_name: value}}
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            recent_folds = [
                fold for fold in self.fold_history 
                if fold.timestamp >= cutoff_date
            ]
            
            if not recent_folds:
                return {}
            
            # 收集各模型指标
            model_metrics = {}
            
            for fold in recent_folds:
                for metric_name, value in fold.fold_metrics.items():
                    if '_' in metric_name:
                        model_name, metric_type = metric_name.rsplit('_', 1)
                        
                        if model_name not in model_metrics:
                            model_metrics[model_name] = {}
                        
                        if metric_type not in model_metrics[model_name]:
                            model_metrics[model_name][metric_type] = []
                        
                        model_metrics[model_name][metric_type].append(value)
            
            # 计算汇总统计
            summary = {}
            for model_name, metrics in model_metrics.items():
                summary[model_name] = {}
                
                for metric_type, values in metrics.items():
                    if values:
                        summary[model_name][f'mean_{metric_type}'] = np.mean(values)
                        summary[model_name][f'std_{metric_type}'] = np.std(values)
                        summary[model_name][f'n_folds_{metric_type}'] = len(values)
            
            return summary
            
        except Exception as e:
            logger.error(f"获取性能摘要失败: {e}")
            return {}
    
    def get_fold_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取折历史记录
        
        Args:
            limit: 限制返回数量
            
        Returns:
            List: 历史记录列表
        """
        try:
            # 按时间排序
            sorted_folds = sorted(self.fold_history, key=lambda x: x.timestamp, reverse=True)
            
            if limit:
                sorted_folds = sorted_folds[:limit]
            
            history = []
            for fold in sorted_folds:
                record = {
                    'fold_id': fold.fold_id,
                    'timestamp': fold.timestamp.isoformat(),
                    'models': list(fold.model_predictions.keys()),
                    'sample_count': len(fold.actuals),
                    'metrics': fold.fold_metrics,
                    'metadata': fold.metadata
                }
                history.append(record)
            
            return history
            
        except Exception as e:
            logger.error(f"获取折历史失败: {e}")
            return []
    
    def cleanup_old_data(self, days_to_keep: Optional[int] = None) -> int:
        """
        清理过期数据
        
        Args:
            days_to_keep: 保留天数，None则使用默认值
            
        Returns:
            int: 清理的记录数
        """
        try:
            days_to_keep = days_to_keep or self.max_history_days
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # 筛选要保留的记录
            original_count = len(self.fold_history)
            self.fold_history = [
                fold for fold in self.fold_history
                if fold.timestamp >= cutoff_date
            ]
            cleaned_count = original_count - len(self.fold_history)
            
            # 清理文件存储
            self._cleanup_storage_files(cutoff_date)
            
            # 重建性能缓存
            self._rebuild_performance_cache()
            
            if cleaned_count > 0:
                logger.info(f"清理了 {cleaned_count} 个过期记录")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理数据失败: {e}")
            return 0
    
    def _load_history(self) -> None:
        """从存储中加载历史数据"""
        try:
            history_file = self.storage_path / "fold_history.pkl"
            if history_file.exists():
                with open(history_file, 'rb') as f:
                    self.fold_history = pickle.load(f)
                logger.info(f"加载了 {len(self.fold_history)} 个历史折记录")
            else:
                self.fold_history = []
                logger.info("初始化空的历史记录")
                
            # 重建缓存
            self._rebuild_performance_cache()
            
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
            self.fold_history = []
    
    def _save_fold_record(self, fold_record: OOSFoldRecord) -> None:
        """保存单个折记录"""
        try:
            # 保存到主历史文件
            history_file = self.storage_path / "fold_history.pkl"
            with open(history_file, 'wb') as f:
                pickle.dump(self.fold_history, f)
            
            # 保存单个折的详细数据
            fold_file = self.storage_path / f"fold_{fold_record.fold_id}.pkl"
            with open(fold_file, 'wb') as f:
                pickle.dump(fold_record, f)
                
        except Exception as e:
            logger.warning(f"保存折记录失败: {e}")
    
    def _update_performance_cache(self, fold_record: OOSFoldRecord) -> None:
        """更新性能缓存"""
        try:
            for model_name in fold_record.model_predictions.keys():
                if model_name not in self.model_performance_cache:
                    self.model_performance_cache[model_name] = {
                        'recent_performance': [],
                        'summary_stats': {}
                    }
                
                # 添加到recent_performance
                performance_entry = {
                    'timestamp': fold_record.timestamp,
                    'fold_id': fold_record.fold_id,
                    'ic': fold_record.fold_metrics.get(f'{model_name}_ic', 0.0),
                    'rank_ic': fold_record.fold_metrics.get(f'{model_name}_rank_ic', 0.0),
                    'mse': fold_record.fold_metrics.get(f'{model_name}_mse', 1.0)
                }
                
                self.model_performance_cache[model_name]['recent_performance'].append(performance_entry)
                
                # 保持最近50个记录
                if len(self.model_performance_cache[model_name]['recent_performance']) > 50:
                    self.model_performance_cache[model_name]['recent_performance'] = \
                        self.model_performance_cache[model_name]['recent_performance'][-50:]
                
        except Exception as e:
            logger.warning(f"更新性能缓存失败: {e}")
    
    def _rebuild_performance_cache(self) -> None:
        """重建性能缓存"""
        try:
            self.model_performance_cache.clear()
            
            for fold_record in self.fold_history:
                self._update_performance_cache(fold_record)
                
            logger.debug("性能缓存重建完成")
            
        except Exception as e:
            logger.error(f"重建性能缓存失败: {e}")
    
    def _cleanup_expired_data(self) -> None:
        """自动清理过期数据"""
        if len(self.fold_history) > 100:  # 如果记录太多，触发清理
            self.cleanup_old_data()
    
    def _cleanup_storage_files(self, cutoff_date: datetime) -> None:
        """清理存储文件"""
        try:
            for file_path in self.storage_path.glob("fold_*.pkl"):
                try:
                    # 检查文件修改时间
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"清理存储文件失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        try:
            status = {
                'total_folds': len(self.fold_history),
                'storage_path': str(self.storage_path),
                'max_history_days': self.max_history_days,
                'models_tracked': len(self.model_performance_cache),
                'oldest_record': None,
                'newest_record': None
            }
            
            if self.fold_history:
                timestamps = [fold.timestamp for fold in self.fold_history]
                status['oldest_record'] = min(timestamps).isoformat()
                status['newest_record'] = max(timestamps).isoformat()
            
            return status
            
        except Exception as e:
            logger.error(f"获取状态失败: {e}")
            return {'status': 'error', 'error': str(e)}