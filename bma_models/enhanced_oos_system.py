#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced OOS (Out-of-Sample) System for BMA Enhanced
基于BMA Enhanced模型特性的增强样本外系统

This module integrates with the BMA Enhanced system to provide:
1. Proper temporal validation with purged time series cross-validation
2. Real OOS predictions from rolling windows
3. BMA weight updates based on true OOS performance
4. Production-ready validation gates
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
import pickle
import json
import hashlib
from scipy import stats

logger = logging.getLogger(__name__)

@dataclass
class OOSConfig:
    """Enhanced OOS系统配置"""
    # 时间分割参数
    cv_n_splits: int = 5
    cv_gap_days: int = 11  # CRITICAL FIX: 统一为11天gap
    embargo_days: int = 11  # CRITICAL FIX: 统一为11天embargo
    test_size_ratio: float = 0.2  # 测试集比例
    
    # 滚动窗口参数
    rolling_window_months: int = 24  # 与Walk-Forward一致
    step_size_days: int = 30
    min_train_samples: int = 1000
    
    # OOS预测管理
    max_oos_history: int = 50  # 保留最近50个OOS预测
    min_samples_for_weight_update: int = 200
    
    # 验证阈值
    min_oos_ic: float = 0.01  # OOS IC最低要求
    stability_threshold: float = 0.5  # 稳定性阈值
    
    # 缓存配置
    cache_dir: str = "cache/oos_system"
    enable_caching: bool = True

@dataclass
class OOSPrediction:
    """单个OOS预测记录"""
    timestamp: datetime
    fold_id: str
    model_name: str
    tickers: List[str]
    predictions: pd.Series
    actuals: pd.Series
    feature_hash: str
    model_version: str
    cv_config: Dict[str, Any]
    
    def calculate_metrics(self) -> Dict[str, float]:
        """计算OOS预测指标"""
        # 对齐预测和实际值
        aligned_pred, aligned_actual = self._align_data()
        
        if len(aligned_pred) < 10:
            return {'valid': False, 'n_samples': len(aligned_pred)}
        
        # ✅ FIXED: 优化IC计算稳健性
        try:
            # Spearman相关系数（已经是rank correlation）
            ic, ic_pvalue = stats.spearmanr(aligned_pred, aligned_actual)
            
            # ✅ FIXED: 移除冗余计算，RankIC与SpearmanIC相同
            # 添加Pearson IC作为补充指标
            pearson_ic, pearson_pvalue = stats.pearsonr(aligned_pred, aligned_actual)
            
            # 处理异常值
            ic = ic if not (np.isnan(ic) or np.isinf(ic)) else 0.0
            ic_pvalue = ic_pvalue if not (np.isnan(ic_pvalue) or np.isinf(ic_pvalue)) else 1.0
            pearson_ic = pearson_ic if not (np.isnan(pearson_ic) or np.isinf(pearson_ic)) else 0.0
            
        except Exception as e:
            logger.warning(f"IC计算异常: {e}")
            ic, ic_pvalue, pearson_ic = 0.0, 1.0, 0.0
        
        # ✅ FIXED: 增强指标计算稳健性
        try:
            mse = np.mean((aligned_pred - aligned_actual) ** 2)
            mae = np.mean(np.abs(aligned_pred - aligned_actual))
            
            # 稳健命中率计算
            pred_median = np.median(aligned_pred)
            actual_median = np.median(aligned_actual)
            hit_rate = np.mean((aligned_pred > pred_median) == (aligned_actual > actual_median))
            
        except Exception as e:
            logger.warning(f"指标计算异常: {e}")
            mse, mae, hit_rate = float('inf'), float('inf'), 0.5
        
        metrics = {
            'valid': True,
            'ic': ic,  # Spearman相关系数
            'ic_pvalue': ic_pvalue,
            'rankic': ic,  # ✅ FIXED: 与ic相同，避免混淆
            'pearson_ic': pearson_ic,  # ✅ ADDED: Pearson相关系数
            'mse': mse,
            'mae': mae,
            'n_samples': len(aligned_pred),
            'hit_rate': hit_rate
        }
        
        return metrics
    
    def _align_data(self) -> Tuple[pd.Series, pd.Series]:
        """对齐预测和实际值数据"""
        # 确保索引对齐
        common_index = self.predictions.index.intersection(self.actuals.index)
        
        pred_aligned = self.predictions.loc[common_index]
        actual_aligned = self.actuals.loc[common_index]
        
        # 移除NaN值
        valid_mask = ~(pred_aligned.isna() | actual_aligned.isna())
        
        return pred_aligned[valid_mask], actual_aligned[valid_mask]

class EnhancedOOSSystem:
    """
    Enhanced OOS System for BMA Enhanced
    为BMA Enhanced量身定制的增强样本外系统
    """
    
    def __init__(self, config: Optional[OOSConfig] = None):
        """初始化Enhanced OOS System"""
        self.config = config or OOSConfig()
        
        # 创建缓存目录
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # OOS预测历史
        self.oos_history: deque = deque(maxlen=self.config.max_oos_history)
        
        # 当前批次的OOS预测
        self.current_batch_predictions: Dict[str, OOSPrediction] = {}
        
        # BMA权重更新历史
        self.weight_update_history: List[Dict[str, Any]] = []
        
        # 加载历史数据
        self._load_oos_cache()
        
        logger.info(f"Enhanced OOS System初始化完成，历史预测: {len(self.oos_history)}个")
    
    def integrate_with_bma_cv(self, 
                             feature_data: pd.DataFrame,
                             target_data: pd.Series,
                             models: Dict[str, Any],
                             bma_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        与BMA Enhanced的CV流程集成
        
        Args:
            feature_data: 特征数据
            target_data: 目标变量
            models: BMA模型字典
            bma_config: BMA配置
            
        Returns:
            包含OOS结果的字典
        """
        logger.info("开始与BMA Enhanced CV流程集成")
        
        try:
            # 1. 准备时间分割
            cv_splits = self._create_temporal_splits(feature_data)
            
            # 2. 执行OOS验证
            oos_results = self._execute_oos_validation(
                feature_data, target_data, models, cv_splits, bma_config
            )
            
            # 3. 更新BMA权重（基于真实OOS性能）
            weight_update_result = self._update_bma_weights(oos_results, models)
            
            # 4. 生产就绪验证
            readiness_check = self._validate_oos_readiness(oos_results)
            
            integration_result = {
                'success': True,
                'oos_validation': oos_results,
                'weight_update': weight_update_result,
                'readiness_check': readiness_check,
                'cv_splits_used': len(cv_splits),
                'total_oos_samples': sum(len(split['test_idx']) for split in cv_splits)
            }
            
            logger.info(f"BMA Enhanced OOS集成完成: {integration_result['total_oos_samples']}个OOS样本")
            return integration_result
            
        except Exception as e:
            logger.error(f"BMA Enhanced OOS集成失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}
    
    def _create_temporal_splits(self, feature_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """创建时间感知的训练/测试分割"""
        if 'date' not in feature_data.columns:
            raise ValueError("特征数据必须包含'date'列")
        
        # 按日期排序
        data_sorted = feature_data.sort_values('date').reset_index(drop=True)
        dates = pd.to_datetime(data_sorted['date'])
        
        splits = []
        n_samples = len(data_sorted)
        
        # 使用BMA Enhanced的时间分割逻辑
        train_size = int(n_samples * (1 - self.config.test_size_ratio))
        
        # 使用滚动窗口方式创建CV分割
        window_size = n_samples // self.config.cv_n_splits
        test_size = int(window_size * self.config.test_size_ratio)
        
        for fold in range(self.config.cv_n_splits):
            # 计算fold的起始位置
            fold_start = fold * window_size
            
            if fold_start + window_size >= n_samples:
                break
            
            # 训练集结束位置
            train_end = fold_start + window_size - test_size - self.config.cv_gap_days
            
            if train_end <= fold_start or train_end >= n_samples:
                continue
            
            # 训练集索引
            train_idx = list(range(fold_start, train_end))
            
            # 测试集索引（考虑gap和embargo）
            test_start = train_end + self.config.cv_gap_days
            test_end = min(test_start + test_size, n_samples)
            
            if test_start >= n_samples or test_end <= test_start:
                continue
                
            test_idx = list(range(test_start, test_end))
            
            # 验证时间顺序
            train_max_date = dates.iloc[train_idx].max()
            test_min_date = dates.iloc[test_idx].min()
            
            if train_max_date >= test_min_date:
                logger.warning(f"Fold {fold}: 时间顺序验证失败，跳过")
                continue
            
            split_info = {
                'fold_id': f"fold_{fold}",
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_date_range': (dates.iloc[train_idx].min(), dates.iloc[train_idx].max()),
                'test_date_range': (dates.iloc[test_idx].min(), dates.iloc[test_idx].max()),
                'gap_days': (test_min_date - train_max_date).days,
                'embargo_satisfied': (test_min_date - train_max_date).days >= self.config.embargo_days
            }
            
            splits.append(split_info)
        
        logger.info(f"创建了{len(splits)}个时间感知CV分割")
        return splits
    
    def _execute_oos_validation(self, 
                               feature_data: pd.DataFrame,
                               target_data: pd.Series,
                               models: Dict[str, Any],
                               cv_splits: List[Dict[str, Any]],
                               bma_config: Dict[str, Any]) -> Dict[str, Any]:
        """执行OOS验证"""
        oos_predictions = {}
        model_performance = {}
        
        for split in cv_splits:
            fold_id = split['fold_id']
            train_idx = split['train_idx']
            test_idx = split['test_idx']
            
            logger.info(f"执行{fold_id}: 训练样本{len(train_idx)}, 测试样本{len(test_idx)}")
            
            # 准备训练和测试数据
            X_train = feature_data.iloc[train_idx].drop(columns=['date'], errors='ignore')
            X_test = feature_data.iloc[test_idx].drop(columns=['date'], errors='ignore')
            y_train = target_data.iloc[train_idx]
            y_test = target_data.iloc[test_idx]
            
            fold_predictions = {}
            
            # 对每个模型进行训练和预测
            for model_name, model in models.items():
                try:
                    # 训练模型
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train)
                    
                    # OOS预测
                    if hasattr(model, 'predict'):
                        y_pred = model.predict(X_test)
                    else:
                        logger.warning(f"{model_name}没有predict方法，跳过")
                        continue
                    
                    # 创建OOS预测记录
                    oos_pred = OOSPrediction(
                        timestamp=datetime.now(),
                        fold_id=fold_id,
                        model_name=model_name,
                        tickers=feature_data.get('ticker', ['MIXED']).unique().tolist(),
                        predictions=pd.Series(y_pred, index=X_test.index),
                        actuals=y_test,
                        feature_hash=self._compute_feature_hash(X_train),
                        model_version=bma_config.get('version', 'unknown'),
                        cv_config=asdict(self.config)
                    )
                    
                    # 计算指标
                    metrics = oos_pred.calculate_metrics()
                    
                    fold_predictions[model_name] = {
                        'predictions': y_pred,
                        'metrics': metrics,
                        'oos_record': oos_pred
                    }
                    
                    logger.info(f"{fold_id}-{model_name}: OOS IC={metrics.get('ic', 0):.4f}")
                    
                except Exception as e:
                    logger.error(f"{fold_id}-{model_name}训练失败: {e}")
            
            oos_predictions[fold_id] = fold_predictions
        
        # 汇总模型性能
        for model_name in models.keys():
            model_ics = []
            for fold_pred in oos_predictions.values():
                if model_name in fold_pred:
                    ic = fold_pred[model_name]['metrics'].get('ic', 0)
                    if not np.isnan(ic):
                        model_ics.append(ic)
            
            if model_ics:
                model_performance[model_name] = {
                    'mean_oos_ic': np.mean(model_ics),
                    'std_oos_ic': np.std(model_ics),
                    'consistency': np.mean([ic > 0 for ic in model_ics]),
                    'folds_evaluated': len(model_ics)
                }
        
        # 保存OOS预测到历史
        for fold_predictions in oos_predictions.values():
            for model_pred in fold_predictions.values():
                if 'oos_record' in model_pred:
                    self.oos_history.append(model_pred['oos_record'])
        
        return {
            'predictions_by_fold': oos_predictions,
            'model_performance': model_performance,
            'validation_summary': self._summarize_oos_validation(model_performance)
        }
    
    def _update_bma_weights(self, 
                           oos_results: Dict[str, Any],
                           models: Dict[str, Any]) -> Dict[str, Any]:
        """基于OOS性能更新BMA权重"""
        model_performance = oos_results.get('model_performance', {})
        
        if not model_performance:
            logger.warning("没有有效的OOS性能数据，使用等权重")
            equal_weight = 1.0 / len(models)
            return {
                'method': 'equal_weight',
                'weights': {model_name: equal_weight for model_name in models.keys()},
                'reason': 'insufficient_oos_data'
            }
        
        # 基于OOS IC计算权重
        oos_ics = {}
        for model_name, perf in model_performance.items():
            oos_ic = perf.get('mean_oos_ic', 0)
            consistency = perf.get('consistency', 0)
            
            # 结合IC和稳定性
            adjusted_ic = oos_ic * consistency
            oos_ics[model_name] = max(adjusted_ic, 0.001)  # 避免负权重
        
        # 计算softmax权重
        ic_values = np.array(list(oos_ics.values()))
        softmax_weights = np.exp(ic_values) / np.sum(np.exp(ic_values))
        
        updated_weights = {}
        for i, model_name in enumerate(oos_ics.keys()):
            updated_weights[model_name] = float(softmax_weights[i])
        
        # 记录权重更新
        weight_update = {
            'timestamp': datetime.now(),
            'method': 'oos_ic_softmax',
            'weights': updated_weights,
            'base_ics': oos_ics,
            'total_oos_samples': sum(perf.get('folds_evaluated', 0) * 100 for perf in model_performance.values())
        }
        
        self.weight_update_history.append(weight_update)
        
        logger.info(f"BMA权重已更新: {updated_weights}")
        return weight_update
    
    def _validate_oos_readiness(self, oos_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证OOS系统生产就绪性"""
        model_performance = oos_results.get('model_performance', {})
        
        readiness_checks = {
            'sufficient_models': len(model_performance) >= 2,
            'minimum_oos_ic': False,
            'stability_check': False,
            'coverage_check': False
        }
        
        if model_performance:
            # 检查最低IC要求
            max_ic = max(perf.get('mean_oos_ic', 0) for perf in model_performance.values())
            readiness_checks['minimum_oos_ic'] = max_ic >= self.config.min_oos_ic
            
            # 检查稳定性
            avg_consistency = np.mean([perf.get('consistency', 0) for perf in model_performance.values()])
            readiness_checks['stability_check'] = avg_consistency >= self.config.stability_threshold
            
            # 检查覆盖度
            min_folds = min(perf.get('folds_evaluated', 0) for perf in model_performance.values())
            readiness_checks['coverage_check'] = min_folds >= 3
        
        overall_ready = all(readiness_checks.values())
        
        return {
            'ready_for_production': overall_ready,
            'checks': readiness_checks,
            'summary': f"{'✅ 通过' if overall_ready else '❌ 未通过'} OOS生产就绪验证"
        }
    
    def _compute_feature_hash(self, features: pd.DataFrame) -> str:
        """计算特征数据的哈希值"""
        feature_str = f"{features.shape}_{list(features.columns)}_{features.dtypes.to_dict()}"
        return hashlib.md5(feature_str.encode()).hexdigest()[:16]
    
    def _summarize_oos_validation(self, model_performance: Dict[str, Any]) -> Dict[str, Any]:
        """总结OOS验证结果"""
        if not model_performance:
            return {'status': 'failed', 'reason': 'no_valid_models'}
        
        all_ics = []
        for perf in model_performance.values():
            ic = perf.get('mean_oos_ic', 0)
            if not np.isnan(ic):
                all_ics.append(ic)
        
        return {
            'status': 'success',
            'models_evaluated': len(model_performance),
            'mean_oos_ic': np.mean(all_ics) if all_ics else 0,
            'best_model_ic': max(all_ics) if all_ics else 0,
            'positive_ic_models': sum(1 for ic in all_ics if ic > 0),
            'total_oos_history': len(self.oos_history)
        }
    
    def _load_oos_cache(self):
        """加载OOS缓存数据"""
        if not self.config.enable_caching:
            return
        
        cache_file = self.cache_dir / "oos_history.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.oos_history = deque(cached_data.get('oos_history', []), 
                                           maxlen=self.config.max_oos_history)
                    self.weight_update_history = cached_data.get('weight_history', [])
                
                logger.info(f"加载OOS缓存: {len(self.oos_history)}个预测记录")
            except Exception as e:
                logger.warning(f"加载OOS缓存失败: {e}")
    
    def save_oos_cache(self):
        """保存OOS缓存数据"""
        if not self.config.enable_caching:
            return
        
        cache_data = {
            'oos_history': list(self.oos_history),
            'weight_history': self.weight_update_history,
            'last_updated': datetime.now().isoformat()
        }
        
        cache_file = self.cache_dir / "oos_history.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"OOS缓存已保存: {len(self.oos_history)}个记录")
        except Exception as e:
            logger.error(f"保存OOS缓存失败: {e}")
    
    def get_oos_performance_report(self) -> Dict[str, Any]:
        """生成OOS性能报告"""
        if not self.oos_history:
            return {'status': 'no_data'}
        
        # 按模型分组分析
        model_stats = {}
        for oos_pred in self.oos_history:
            model_name = oos_pred.model_name
            metrics = oos_pred.calculate_metrics()
            
            if model_name not in model_stats:
                model_stats[model_name] = []
            
            if metrics.get('valid', False):
                model_stats[model_name].append(metrics)
        
        # 生成统计报告
        report = {
            'total_oos_predictions': len(self.oos_history),
            'unique_models': len(model_stats),
            'model_performance': {}
        }
        
        for model_name, stats_list in model_stats.items():
            if stats_list:
                ics = [s['ic'] for s in stats_list]
                report['model_performance'][model_name] = {
                    'predictions_count': len(stats_list),
                    'mean_ic': np.mean(ics),
                    'std_ic': np.std(ics),
                    'best_ic': max(ics),
                    'consistency_rate': sum(1 for ic in ics if ic > 0) / len(ics)
                }
        
        return report

def create_enhanced_oos_system(config: Optional[Dict[str, Any]] = None) -> EnhancedOOSSystem:
    """创建Enhanced OOS System实例"""
    if config:
        oos_config = OOSConfig(**config)
    else:
        oos_config = OOSConfig()
    
    return EnhancedOOSSystem(oos_config)