"""
Time-Safe BMA Weight Calculation Module
======================================
确保BMA权重计算严格基于历史数据，防止前瞻偏差

关键原则：
1. 权重计算只能使用T-1及之前的历史数据
2. IC计算必须基于历史out-of-sample预测
3. 滚动窗口更新权重，确保时间一致性
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from scipy.stats import spearmanr
from collections import defaultdict

logger = logging.getLogger(__name__)


class TimeSafeBMAWeightCalculator:
    """
    时间安全的BMA权重计算器
    
    确保所有权重计算严格遵循时间序列原则：
    - 使用historical-only数据计算IC
    - 滚动窗口更新权重
    - 严格的时间验证
    """
    
    def __init__(self, 
                 lookback_days: int = 252,
                 min_history_days: int = 63,
                 rebalance_frequency: int = 21,
                 ic_shrinkage_factor: float = 0.8):
        """
        初始化时间安全BMA权重计算器
        
        Parameters:
        -----------
        lookback_days : int
            权重计算的历史回望天数（约1年）
        min_history_days : int
            计算权重所需的最小历史天数（约3个月）
        rebalance_frequency : int
            权重重新计算频率（天数）
        ic_shrinkage_factor : float
            IC收缩因子，防止过拟合
        """
        self.lookback_days = lookback_days
        self.min_history_days = min_history_days
        self.rebalance_frequency = rebalance_frequency
        self.ic_shrinkage_factor = ic_shrinkage_factor
        
        # 权重历史记录
        self.weight_history = {}
        self.ic_history = {}
        self.last_rebalance_date = None
        
        logger.info(f"时间安全BMA权重计算器初始化: "
                   f"回望={lookback_days}天, 最小历史={min_history_days}天, "
                   f"重平衡频率={rebalance_frequency}天")
    
    def calculate_time_safe_weights(self, 
                                   oof_predictions: Dict[str, pd.Series],
                                   targets: pd.Series,
                                   current_date: pd.Timestamp,
                                   force_rebalance: bool = False) -> Dict[str, float]:
        """
        计算时间安全的BMA权重
        
        Parameters:
        -----------
        oof_predictions : Dict[str, pd.Series]
            各模型的OOF预测，index必须包含date信息
        targets : pd.Series
            目标变量，index必须包含date信息
        current_date : pd.Timestamp
            当前日期（权重计算的截止日期）
        force_rebalance : bool
            是否强制重新计算权重
            
        Returns:
        --------
        Dict[str, float] : 时间安全的BMA权重
        """
        logger.info(f"🕒 开始计算时间安全BMA权重 (截止日期: {current_date.strftime('%Y-%m-%d')})")
        
        # 检查是否需要重新计算权重
        if not force_rebalance and self._should_use_cached_weights(current_date):
            logger.info("使用缓存的权重")
            return self.weight_history.get(current_date, {})
        
        # 验证输入数据的时间安全性
        self._validate_time_safety(oof_predictions, targets, current_date)
        
        # 过滤历史数据（严格T-1截止）
        historical_predictions = self._filter_historical_data(oof_predictions, current_date)
        historical_targets = self._filter_historical_data({'targets': targets}, current_date)['targets']
        
        # 检查数据充足性
        if not self._has_sufficient_history(historical_predictions, historical_targets):
            logger.warning("历史数据不足，使用均等权重")
            return self._get_equal_weights(list(oof_predictions.keys()))
        
        # 计算历史IC指标
        historical_metrics = self._calculate_historical_metrics(historical_predictions, historical_targets)
        
        # 计算时间安全的BMA权重
        safe_weights = self._compute_safe_bma_weights(historical_metrics, current_date)
        
        # 缓存权重
        self.weight_history[current_date] = safe_weights
        self.last_rebalance_date = current_date
        
        logger.info(f"权重计算完成: {dict(safe_weights)}")
        return safe_weights
    
    def _validate_time_safety(self, 
                             oof_predictions: Dict[str, pd.Series],
                             targets: pd.Series,
                             current_date: pd.Timestamp) -> None:
        """验证输入数据的时间安全性"""
        
        # 检查所有数据都严格早于current_date
        for model_name, predictions in oof_predictions.items():
            if hasattr(predictions.index, 'get_level_values'):
                # MultiIndex情况
                try:
                    dates = pd.to_datetime(predictions.index.get_level_values('date'))
                except:
                    dates = pd.to_datetime(predictions.index.get_level_values(0))  # 假设第一层是日期
            else:
                dates = pd.to_datetime(predictions.index)
            
            latest_date = dates.max()
            if latest_date >= current_date:
                raise ValueError(f"模型 {model_name} 包含当期或未来数据: "
                               f"最新日期 {latest_date} >= 当前日期 {current_date}")
        
        # 检查目标数据
        if hasattr(targets.index, 'get_level_values'):
            try:
                target_dates = pd.to_datetime(targets.index.get_level_values('date'))
            except:
                target_dates = pd.to_datetime(targets.index.get_level_values(0))
        else:
            target_dates = pd.to_datetime(targets.index)
        
        latest_target_date = target_dates.max()
        if latest_target_date >= current_date:
            raise ValueError(f"目标数据包含当期或未来信息: "
                           f"最新日期 {latest_target_date} >= 当前日期 {current_date}")
        
        logger.info("✅ 时间安全性验证通过")
    
    def _filter_historical_data(self, 
                               data_dict: Dict[str, pd.Series],
                               current_date: pd.Timestamp) -> Dict[str, pd.Series]:
        """过滤出严格的历史数据（T-1截止）"""
        
        # 计算历史数据的时间窗口
        end_date = current_date - timedelta(days=1)  # T-1截止
        start_date = end_date - timedelta(days=self.lookback_days)
        
        filtered_data = {}
        
        for key, series in data_dict.items():
            try:
                if hasattr(series.index, 'get_level_values'):
                    # MultiIndex处理
                    try:
                        dates = pd.to_datetime(series.index.get_level_values('date'))
                    except:
                        dates = pd.to_datetime(series.index.get_level_values(0))
                    
                    # 筛选时间范围
                    mask = (dates >= start_date) & (dates <= end_date)
                    filtered_series = series[mask]
                else:
                    # 普通Index处理
                    dates = pd.to_datetime(series.index)
                    mask = (dates >= start_date) & (dates <= end_date)
                    filtered_series = series[mask]
                
                filtered_data[key] = filtered_series
                logger.info(f"{key}: 过滤后数据量 {len(filtered_series)} "
                           f"(时间范围: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})")
                
            except Exception as e:
                logger.error(f"过滤历史数据失败 {key}: {e}")
                filtered_data[key] = pd.Series(dtype=float)
        
        return filtered_data
    
    def _has_sufficient_history(self, 
                               predictions: Dict[str, pd.Series],
                               targets: pd.Series) -> bool:
        """检查是否有足够的历史数据"""
        
        if len(targets) < self.min_history_days:
            logger.warning(f"目标数据不足: {len(targets)} < {self.min_history_days}")
            return False
        
        for model_name, pred in predictions.items():
            if len(pred) < self.min_history_days:
                logger.warning(f"模型 {model_name} 历史数据不足: {len(pred)} < {self.min_history_days}")
                return False
        
        return True
    
    def _calculate_historical_metrics(self, 
                                    predictions: Dict[str, pd.Series],
                                    targets: pd.Series) -> Dict[str, Dict]:
        """计算基于历史数据的IC指标"""
        
        metrics = {}
        
        for model_name, pred in predictions.items():
            try:
                # 对齐预测和目标数据
                aligned_pred, aligned_target = self._align_series(pred, targets)
                
                if len(aligned_pred) < 10:
                    logger.warning(f"模型 {model_name} 对齐后数据不足")
                    metrics[model_name] = self._get_default_metrics()
                    continue
                
                # 计算IC (使用Spearman相关系数)
                ic_corr, ic_pvalue = spearmanr(aligned_target, aligned_pred)
                ic = ic_corr if not np.isnan(ic_corr) else 0.0
                
                # 计算t统计量
                n = len(aligned_pred)
                t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-8)) if abs(ic) < 0.99 else 0
                
                # 计算滚动IC的稳定性 (ICIR近似)
                ic_std = self._calculate_rolling_ic_std(aligned_pred, aligned_target)
                icir = ic / (ic_std + 1e-8)
                
                # 应用IC收缩
                ic_shrunk = ic * self.ic_shrinkage_factor
                
                metrics[model_name] = {
                    'ic_raw': ic,
                    'ic_shrunk': ic_shrunk,
                    'ic_pvalue': ic_pvalue,
                    't_stat': t_stat,
                    'icir': icir,
                    'sample_count': len(aligned_pred)
                }
                
                logger.info(f"{model_name} 历史指标: IC={ic:.4f}→{ic_shrunk:.4f}, "
                           f"t={t_stat:.2f}, ICIR={icir:.4f}")
                
            except Exception as e:
                logger.error(f"计算 {model_name} 历史指标失败: {e}")
                metrics[model_name] = self._get_default_metrics()
        
        return metrics
    
    def _align_series(self, pred: pd.Series, target: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """对齐两个Series（处理MultiIndex）"""
        
        try:
            # 尝试直接对齐
            if hasattr(pred.index, 'names') and hasattr(target.index, 'names'):
                # 都是MultiIndex
                common_index = pred.index.intersection(target.index)
                if len(common_index) > 0:
                    aligned_pred = pred.reindex(common_index).dropna()
                    aligned_target = target.reindex(common_index).dropna()
                    
                    # 再次对齐以确保无缺失值
                    final_index = aligned_pred.index.intersection(aligned_target.index)
                    return aligned_pred.reindex(final_index), aligned_target.reindex(final_index)
            
            # 简单对齐（如果Index结构不匹配）
            min_len = min(len(pred), len(target))
            aligned_pred = pred.iloc[:min_len].dropna()
            aligned_target = target.iloc[:min_len].dropna()
            
            # 确保长度一致
            final_len = min(len(aligned_pred), len(aligned_target))
            return aligned_pred.iloc[:final_len], aligned_target.iloc[:final_len]
            
        except Exception as e:
            logger.error(f"序列对齐失败: {e}")
            return pd.Series(dtype=float), pd.Series(dtype=float)
    
    def _calculate_rolling_ic_std(self, pred: pd.Series, target: pd.Series, window: int = 63) -> float:
        """计算滚动IC的标准差（用于ICIR计算）"""
        try:
            if len(pred) < window:
                return 1.0  # 默认值
            
            rolling_ics = []
            for i in range(window, len(pred)):
                window_pred = pred.iloc[i-window:i]
                window_target = target.iloc[i-window:i]
                
                if len(window_pred) == len(window_target) and len(window_pred) > 5:
                    ic_corr, _ = spearmanr(window_target, window_pred)
                    if not np.isnan(ic_corr):
                        rolling_ics.append(ic_corr)
            
            return np.std(rolling_ics) if len(rolling_ics) > 1 else 1.0
            
        except Exception as e:
            logger.error(f"计算滚动IC标准差失败: {e}")
            return 1.0
    
    def _compute_safe_bma_weights(self, 
                                 metrics: Dict[str, Dict],
                                 current_date: pd.Timestamp) -> Dict[str, float]:
        """计算安全的BMA权重"""
        
        # 过滤掉无效模型
        valid_models = {name: m for name, m in metrics.items() 
                       if m['sample_count'] >= 10 and abs(m['t_stat']) >= 1.0}
        
        if not valid_models:
            logger.warning("没有有效模型，使用均等权重")
            return self._get_equal_weights(list(metrics.keys()))
        
        # 计算原始权重（基于收缩后的IC和ICIR）
        raw_weights = {}
        for model_name, m in valid_models.items():
            # 权重 = IC_shrunk × ICIR × max(0, IC_shrunk)
            weight = m['ic_shrunk'] * m['icir'] * max(0, m['ic_shrunk'])
            raw_weights[model_name] = weight
        
        # 标准化权重
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            logger.warning("总权重非正，使用均等权重")
            return self._get_equal_weights(list(valid_models.keys()))
        
        normalized_weights = {name: w / total_weight for name, w in raw_weights.items()}
        
        # 应用权重约束（单模型不超过50%）
        constrained_weights = self._apply_weight_constraints(normalized_weights)
        
        # EMA平滑（如果有历史权重）
        smoothed_weights = self._apply_ema_smoothing(constrained_weights, current_date)
        
        return smoothed_weights
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用权重约束"""
        max_weight = 0.5  # 单模型最大权重50%
        
        # 截断过大权重
        constrained = {}
        total_excess = 0
        
        for name, weight in weights.items():
            if weight > max_weight:
                constrained[name] = max_weight
                total_excess += weight - max_weight
            else:
                constrained[name] = weight
        
        # 重新分配超额权重
        if total_excess > 0:
            eligible_models = [name for name, w in constrained.items() if w < max_weight]
            if eligible_models:
                redistribution = total_excess / len(eligible_models)
                for name in eligible_models:
                    constrained[name] = min(constrained[name] + redistribution, max_weight)
        
        # 重新标准化
        total = sum(constrained.values())
        if total > 0:
            constrained = {name: w / total for name, w in constrained.items()}
        
        return constrained
    
    def _apply_ema_smoothing(self, 
                            current_weights: Dict[str, float],
                            current_date: pd.Timestamp,
                            alpha: float = 0.3) -> Dict[str, float]:
        """应用EMA平滑到权重序列"""
        
        if not self.weight_history:
            return current_weights
        
        # 寻找最近的历史权重
        previous_weights = {}
        for date in sorted(self.weight_history.keys(), reverse=True):
            if date < current_date:
                previous_weights = self.weight_history[date]
                break
        
        if not previous_weights:
            return current_weights
        
        # EMA平滑: w_new = α × w_current + (1-α) × w_previous
        smoothed = {}
        all_models = set(current_weights.keys()) | set(previous_weights.keys())
        
        for model in all_models:
            current = current_weights.get(model, 0.0)
            previous = previous_weights.get(model, 0.0)
            smoothed[model] = alpha * current + (1 - alpha) * previous
        
        # 标准化并过滤小权重
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {name: w / total for name, w in smoothed.items() if w / total >= 0.01}
        
        logger.info(f"EMA平滑应用: α={alpha}")
        return smoothed
    
    def _should_use_cached_weights(self, current_date: pd.Timestamp) -> bool:
        """检查是否应该使用缓存权重"""
        if self.last_rebalance_date is None:
            return False
        
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        return days_since_rebalance < self.rebalance_frequency
    
    def _get_equal_weights(self, model_names: List[str]) -> Dict[str, float]:
        """获取均等权重"""
        if not model_names:
            return {}
        
        weight = 1.0 / len(model_names)
        return {name: weight for name in model_names}
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """获取默认指标"""
        return {
            'ic_raw': 0.0,
            'ic_shrunk': 0.0,
            'ic_pvalue': 1.0,
            't_stat': 0.0,
            'icir': 0.0,
            'sample_count': 0
        }
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """获取权重统计信息"""
        if not self.weight_history:
            return {}
        
        latest_date = max(self.weight_history.keys())
        latest_weights = self.weight_history[latest_date]
        
        return {
            'latest_date': latest_date,
            'latest_weights': latest_weights,
            'total_rebalances': len(self.weight_history),
            'weight_concentration': max(latest_weights.values()) if latest_weights else 0,
            'active_models': len([w for w in latest_weights.values() if w > 0.01])
        }


def create_time_safe_bma_calculator(lookback_days: int = 252,
                                   min_history_days: int = 63,
                                   rebalance_frequency: int = 21) -> TimeSafeBMAWeightCalculator:
    """
    创建时间安全的BMA权重计算器
    
    Returns:
    --------
    TimeSafeBMAWeightCalculator : 配置好的权重计算器
    """
    return TimeSafeBMAWeightCalculator(
        lookback_days=lookback_days,
        min_history_days=min_history_days,
        rebalance_frequency=rebalance_frequency
    )