"""
T+10预测参数优化器 - 针对T+10预测调优所有时间参数
T+10 Parameter Optimizer - Optimize all time parameters for T+10 predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from scipy.optimize import minimize_scalar, minimize
import itertools
from concurrent.futures import ThreadPoolExecutor
import warnings

logger = logging.getLogger(__name__)

@dataclass
class T10OptimizationConfig:
    """T+10预测优化配置"""
    # 预测目标
    prediction_horizon: int = 10  # T+10预测
    
    # Gap/Embargo优化范围
    min_gap_days: int = 10
    max_gap_days: int = 25
    gap_step: int = 1
    
    min_embargo_days: int = 10  
    max_embargo_days: int = 25
    embargo_step: int = 1
    
    # 特征滞后优化
    min_feature_lag: int = 1
    max_feature_lag: int = 5
    lag_step: int = 1
    
    # CV参数优化
    min_cv_splits: int = 3
    max_cv_splits: int = 8
    
    # 窗口参数
    min_rolling_window_months: int = 12
    max_rolling_window_months: int = 36
    window_step_months: int = 6
    
    # 优化目标
    primary_metric: str = 'ic'  # 'ic', 'sharpe', 'returns', 'stability'
    secondary_metrics: List[str] = None
    
    # 约束条件
    min_samples_per_fold: int = 500
    max_optimization_time_minutes: int = 60
    
    # 稳健性设置
    stability_weight: float = 0.3  # 稳定性权重
    performance_weight: float = 0.7  # 性能权重

@dataclass
class OptimizationResult:
    """优化结果"""
    optimal_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict]
    optimization_summary: Dict[str, Any]
    recommendations: List[str]

class T10ParameterOptimizer:
    """T+10预测参数优化器"""
    
    def __init__(self, config: Optional[T10OptimizationConfig] = None):
        self.config = config or T10OptimizationConfig()
        self.optimization_history: List[Dict] = []
        self.best_params: Optional[Dict] = None
        self.evaluation_cache: Dict = {}
        
        if self.config.secondary_metrics is None:
            self.config.secondary_metrics = ['sharpe', 'stability', 'drawdown']
        
        logger.info(f"初始化T+10参数优化器 - 预测期:{self.config.prediction_horizon}天")
    
    def optimize_gap_embargo(self, 
                           data: pd.DataFrame,
                           target_col: str = 'target',
                           date_col: str = 'date') -> Dict[str, Any]:
        """优化Gap和Embargo参数"""
        logger.info("开始Gap/Embargo参数优化...")
        
        best_score = -np.inf
        best_params = {}
        all_results = []
        
        # 参数网格搜索
        gap_range = range(self.config.min_gap_days, 
                         self.config.max_gap_days + 1, 
                         self.config.gap_step)
        embargo_range = range(self.config.min_embargo_days,
                             self.config.max_embargo_days + 1,
                             self.config.embargo_step)
        
        total_combinations = len(list(gap_range)) * len(list(embargo_range))
        logger.info(f"搜索空间: {total_combinations} 个参数组合")
        
        for gap_days, embargo_days in itertools.product(gap_range, embargo_range):
            try:
                # 确保合理的参数约束
                if embargo_days < gap_days - 5:  # embargo不应太小于gap
                    continue
                    
                params = {
                    'gap_days': gap_days,
                    'embargo_days': embargo_days,
                    'prediction_horizon': self.config.prediction_horizon
                }
                
                # 评估参数组合
                score, metrics = self._evaluate_temporal_params(data, params, target_col, date_col)
                
                result = {
                    'params': params.copy(),
                    'score': score,
                    'metrics': metrics,
                    'evaluation_time': datetime.now()
                }
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    logger.info(f"新最优参数: gap={gap_days}, embargo={embargo_days}, score={score:.4f}")
                    
            except Exception as e:
                logger.warning(f"参数评估失败 gap={gap_days}, embargo={embargo_days}: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'optimization_type': 'gap_embargo'
        }
    
    def optimize_feature_lags(self, 
                            data: pd.DataFrame,
                            feature_cols: List[str],
                            target_col: str = 'target') -> Dict[str, Any]:
        """优化特征滞后参数"""
        logger.info(f"开始特征滞后优化 - {len(feature_cols)}个特征")
        
        # 针对每个特征优化滞后
        feature_optimal_lags = {}
        
        for feature in feature_cols[:10]:  # 限制特征数量避免过长优化
            logger.debug(f"优化特征滞后: {feature}")
            
            best_lag = self.config.min_feature_lag
            best_ic = -np.inf
            
            for lag in range(self.config.min_feature_lag, 
                           self.config.max_feature_lag + 1, 
                           self.config.lag_step):
                try:
                    # 创建滞后特征
                    lagged_feature = data[feature].shift(lag)
                    target = data[target_col].shift(-self.config.prediction_horizon)
                    
                    # 计算IC
                    valid_idx = ~(lagged_feature.isna() | target.isna())
                    if valid_idx.sum() < 100:
                        continue
                        
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(lagged_feature[valid_idx], target[valid_idx])
                    
                    if abs(ic) > abs(best_ic):
                        best_ic = ic
                        best_lag = lag
                        
                except Exception as e:
                    logger.debug(f"滞后评估失败 {feature} lag={lag}: {e}")
                    continue
            
            feature_optimal_lags[feature] = {
                'optimal_lag': best_lag,
                'best_ic': best_ic
            }
        
        return {
            'feature_lags': feature_optimal_lags,
            'global_recommendations': self._get_lag_recommendations(feature_optimal_lags)
        }
    
    def optimize_cv_parameters(self, 
                             data: pd.DataFrame,
                             base_params: Dict[str, Any]) -> Dict[str, Any]:
        """优化交叉验证参数"""
        logger.info("开始CV参数优化...")
        
        best_score = -np.inf
        best_cv_params = {}
        all_results = []
        
        # CV折数优化
        for n_splits in range(self.config.min_cv_splits, self.config.max_cv_splits + 1):
            # 滚动窗口优化
            for window_months in range(self.config.min_rolling_window_months,
                                     self.config.max_rolling_window_months + 1,
                                     self.config.window_step_months):
                try:
                    cv_params = {
                        'n_splits': n_splits,
                        'rolling_window_months': window_months,
                        **base_params
                    }
                    
                    # 评估CV配置
                    score, stability = self._evaluate_cv_config(data, cv_params)
                    
                    result = {
                        'cv_params': cv_params.copy(),
                        'score': score,
                        'stability': stability
                    }
                    all_results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_cv_params = cv_params.copy()
                        
                except Exception as e:
                    logger.debug(f"CV参数评估失败: {e}")
                    continue
        
        return {
            'best_cv_params': best_cv_params,
            'best_score': best_score,
            'all_results': all_results
        }
    
    def comprehensive_optimization(self, 
                                 data: pd.DataFrame,
                                 feature_cols: List[str],
                                 target_col: str = 'target',
                                 date_col: str = 'date') -> OptimizationResult:
        """综合参数优化"""
        logger.info("开始T+10预测综合参数优化...")
        
        optimization_start = datetime.now()
        all_results = []
        
        try:
            # 阶段1: Gap/Embargo优化
            logger.info("=== 阶段1: Gap/Embargo优化 ===")
            gap_embargo_result = self.optimize_gap_embargo(data, target_col, date_col)
            all_results.append(gap_embargo_result)
            
            # 阶段2: 特征滞后优化
            logger.info("=== 阶段2: 特征滞后优化 ===")
            lag_result = self.optimize_feature_lags(data, feature_cols, target_col)
            all_results.append(lag_result)
            
            # 阶段3: CV参数优化
            logger.info("=== 阶段3: CV参数优化 ===")
            cv_result = self.optimize_cv_parameters(data, gap_embargo_result['best_params'])
            all_results.append(cv_result)
            
            # 综合最优参数
            optimal_params = self._combine_optimal_params(gap_embargo_result, lag_result, cv_result)
            
            # 最终验证
            final_score, final_metrics = self._final_validation(data, optimal_params, target_col)
            
            # 生成建议
            recommendations = self._generate_recommendations(optimal_params, final_metrics)
            
            optimization_time = (datetime.now() - optimization_start).total_seconds() / 60
            
            result = OptimizationResult(
                optimal_params=optimal_params,
                best_score=final_score,
                all_results=all_results,
                optimization_summary={
                    'optimization_time_minutes': optimization_time,
                    'total_evaluations': sum(len(r.get('all_results', [])) for r in all_results),
                    'final_metrics': final_metrics,
                    'prediction_horizon': self.config.prediction_horizon
                },
                recommendations=recommendations
            )
            
            self.best_params = optimal_params
            logger.info(f"T+10参数优化完成 - 耗时{optimization_time:.1f}分钟")
            
            return result
            
        except Exception as e:
            logger.error(f"综合优化失败: {e}")
            raise
    
    def get_optimized_config_dict(self) -> Dict[str, Any]:
        """获取优化后的配置字典，用于更新现有系统"""
        if not self.best_params:
            logger.warning("尚未运行优化，返回默认配置")
            return self._get_default_config()
        
        config_dict = {
            # 时间参数
            'cv_gap_days': self.best_params.get('gap_days', 15),
            'embargo_days': self.best_params.get('embargo_days', 15),
            'prediction_horizon': self.config.prediction_horizon,
            
            # CV参数
            'cv_n_splits': self.best_params.get('n_splits', 5),
            'rolling_window_months': self.best_params.get('rolling_window_months', 24),
            
            # 安全参数
            'min_samples_per_fold': self.config.min_samples_per_fold,
            'strict_temporal_validation': True,
            
            # 优化元信息
            'optimization_timestamp': datetime.now().isoformat(),
            'optimized_for': f'T+{self.config.prediction_horizon}_predictions'
        }
        
        return config_dict
    
    # 辅助方法
    def _evaluate_temporal_params(self, data: pd.DataFrame, params: Dict, 
                                target_col: str, date_col: str) -> Tuple[float, Dict]:
        """评估时间参数"""
        try:
            # 简化的评估：检查数据可用性和基本IC
            gap_days = params['gap_days']
            horizon = params['prediction_horizon']
            
            # 创建特征和目标
            feature = data.select_dtypes(include=[np.number]).iloc[:, 0]  # 第一个数值列作为示例
            target = data[target_col].shift(-horizon) if target_col in data.columns else feature.shift(-horizon)
            
            # 应用gap
            lagged_feature = feature.shift(gap_days)
            
            # 计算有效样本
            valid_mask = ~(lagged_feature.isna() | target.isna())
            n_valid = valid_mask.sum()
            
            if n_valid < self.config.min_samples_per_fold:
                return -1.0, {'n_samples': n_valid, 'error': 'insufficient_samples'}
            
            # 计算IC
            from scipy.stats import spearmanr
            ic, pvalue = spearmanr(lagged_feature[valid_mask], target[valid_mask])
            
            # 计算稳定性指标（简化）
            stability = 1.0 / (1.0 + abs(ic - 0.1))  # 适中的IC更稳定
            
            # 综合评分
            score = (self.config.performance_weight * abs(ic) + 
                    self.config.stability_weight * stability)
            
            metrics = {
                'ic': ic,
                'pvalue': pvalue,
                'stability': stability,
                'n_samples': n_valid
            }
            
            return score, metrics
            
        except Exception as e:
            logger.debug(f"参数评估出错: {e}")
            return -1.0, {'error': str(e)}
    
    def _evaluate_cv_config(self, data: pd.DataFrame, cv_params: Dict) -> Tuple[float, float]:
        """评估CV配置"""
        try:
            n_splits = cv_params['n_splits']
            window_months = cv_params['rolling_window_months']
            
            # 估算可用数据
            if len(data) < n_splits * self.config.min_samples_per_fold:
                return -1.0, 0.0
            
            # 简化的稳定性评估
            samples_per_fold = len(data) // n_splits
            stability = min(1.0, samples_per_fold / self.config.min_samples_per_fold)
            
            # 窗口大小适中性评分
            window_score = 1.0 - abs(window_months - 24) / 24  # 24个月为最优
            
            score = 0.6 * stability + 0.4 * max(0, window_score)
            
            return score, stability
            
        except Exception as e:
            logger.debug(f"CV配置评估出错: {e}")
            return -1.0, 0.0
    
    def _get_lag_recommendations(self, feature_lags: Dict) -> Dict[str, Any]:
        """获取滞后建议"""
        if not feature_lags:
            return {'recommended_global_lag': self.config.min_feature_lag}
        
        # 统计最优滞后分布
        optimal_lags = [info['optimal_lag'] for info in feature_lags.values()]
        most_common_lag = max(set(optimal_lags), key=optimal_lags.count)
        
        # 计算平均IC
        average_ic = np.mean([abs(info['best_ic']) for info in feature_lags.values()])
        
        return {
            'recommended_global_lag': most_common_lag,
            'lag_distribution': dict(zip(*np.unique(optimal_lags, return_counts=True))),
            'average_abs_ic': average_ic,
            'feature_specific_lags': len(set(optimal_lags)) > 1
        }
    
    def _combine_optimal_params(self, gap_result: Dict, lag_result: Dict, cv_result: Dict) -> Dict:
        """合并最优参数"""
        combined = {}
        
        # 时间参数
        if 'best_params' in gap_result:
            combined.update(gap_result['best_params'])
        
        # 滞后参数
        if 'global_recommendations' in lag_result:
            combined['recommended_lag'] = lag_result['global_recommendations']['recommended_global_lag']
        
        # CV参数  
        if 'best_cv_params' in cv_result:
            combined.update(cv_result['best_cv_params'])
        
        # 确保关键参数存在
        combined.setdefault('gap_days', 15)
        combined.setdefault('embargo_days', 15) 
        combined.setdefault('n_splits', 5)
        combined.setdefault('rolling_window_months', 24)
        
        return combined
    
    def _final_validation(self, data: pd.DataFrame, params: Dict, target_col: str) -> Tuple[float, Dict]:
        """最终验证"""
        try:
            # 使用最优参数进行最终评估
            score, metrics = self._evaluate_temporal_params(
                data, params, target_col, 'date'
            )
            
            # 添加参数合理性检查
            gap_days = params.get('gap_days', 15)
            embargo_days = params.get('embargo_days', 15)
            
            # 安全性检查
            safety_score = 1.0
            if gap_days < 10:  # T+10预测最小安全gap
                safety_score *= 0.5
            if embargo_days < gap_days:
                safety_score *= 0.8
            
            final_score = score * safety_score
            metrics['safety_score'] = safety_score
            metrics['final_score'] = final_score
            
            return final_score, metrics
            
        except Exception as e:
            logger.error(f"最终验证失败: {e}")
            return 0.0, {'error': str(e)}
    
    def _generate_recommendations(self, params: Dict, metrics: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        gap_days = params.get('gap_days', 15)
        embargo_days = params.get('embargo_days', 15)
        
        # Gap/Embargo建议
        if gap_days >= 15:
            recommendations.append(f"[OK] Gap设置安全: {gap_days}天 >= T+10最小要求15天")
        else:
            recommendations.append(f"[WARN] 建议增加Gap至15天以上 (当前{gap_days}天)")
        
        if embargo_days >= gap_days:
            recommendations.append(f"[OK] Embargo设置合理: {embargo_days}天")
        else:
            recommendations.append(f"[WARN] 建议Embargo不低于Gap: {embargo_days}天 vs {gap_days}天")
        
        # 性能建议
        ic = metrics.get('ic', 0)
        if abs(ic) > 0.05:
            recommendations.append(f"[OK] IC表现良好: {ic:.4f}")
        elif abs(ic) > 0.02:
            recommendations.append(f"[WARN] IC偏低，考虑特征工程优化: {ic:.4f}")
        else:
            recommendations.append(f"[ALERT] IC过低，需要重新审视模型: {ic:.4f}")
        
        # 样本数量建议
        n_samples = metrics.get('n_samples', 0)
        if n_samples < 1000:
            recommendations.append(f"[WARN] 有效样本不足，建议增加数据或调整参数: {n_samples}")
        
        return recommendations
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'cv_gap_days': 15,
            'embargo_days': 15,
            'prediction_horizon': 10,
            'cv_n_splits': 5,
            'rolling_window_months': 24,
            'optimization_status': 'default_config'
        }