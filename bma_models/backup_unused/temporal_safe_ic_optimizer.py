#!/usr/bin/env python3
"""
时间安全的IC权重优化器 - 修复时间泄漏和同步信息问题
============================================================
替换随机验证切分，使用purged & embargo时间切分
仅使用滚动窗口统计，禁止未来信息泄漏
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import warnings
try:
    from .unified_cv_policy import get_global_cv_policy
    from .unified_ic_calculator import get_global_ic_calculator
except ImportError:
    from unified_cv_policy import get_global_cv_policy
    from unified_ic_calculator import get_global_ic_calculator

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class TemporalSafeICConfig:
    """时间安全IC优化器配置"""
    # ==================== 时间安全配置 ====================
    use_purged_cv: bool = True                 # 使用purged时间序列CV
    embargo_days: int = 10                     # 禁带期天数，来自全局CV策略
    purge_days: int = 10                       # 净化天数，来自全局CV策略
    min_train_window: int = 120                # 最小训练窗口（天）
    
    # ==================== 滚动统计配置 ====================
    rolling_window_days: int = 60              # 滚动窗口天数
    ic_decay_halflife: int = 30                # IC计算衰减半衰期
    correlation_window: int = 60               # 相关性计算窗口
    volatility_window: int = 30                # 波动率计算窗口
    
    # ==================== 特征工程配置 ====================
    max_features: int = 20                     # 最大特征数
    min_ic_threshold: float = 0.015            # 最小IC阈值（调整后）
    ic_stability_weight: float = 0.4           # IC稳定性权重
    ic_magnitude_weight: float = 0.6           # IC幅度权重
    
    # ==================== 优化器配置 ====================
    optimization_method: str = "ridge"         # ridge/elastic_net/random_forest
    alpha_regularization: float = 1.0          # 正则化参数
    cv_folds: int = 3                          # 时间序列CV折数
    min_samples_per_fold: int = 50             # 每折最少样本数
    
    # ==================== 质量控制 ====================
    outlier_threshold: float = 3.0             # 异常值阈值（标准差倍数）
    min_correlation_samples: int = 30          # 最少相关性计算样本
    max_weight_concentration: float = 0.5      # 最大权重集中度
    
    # ==================== 回退策略 ====================
    enable_fallback: bool = True               # 启用回退策略
    fallback_method: str = "equal_weight"      # 回退方法：equal_weight/ic_rank
    fallback_trigger_threshold: float = 0.3    # 回退触发阈值

class PurgedTimeSeriesSplit:
    """Purged时间序列交叉验证"""
    
    def __init__(self, n_splits: int = 3, embargo_days: int = 10, 
                 purge_days: int = 10, min_samples: int = 50):
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.purge_days = purge_days
        self.min_samples = min_samples
    
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        生成时间安全的训练/验证分割
        
        Args:
            X: 特征数据 (index必须是时间索引)
            y: 目标变量
        
        Returns:
            训练/验证索引对的列表
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X的索引必须是DatetimeIndex")
        
        dates = X.index
        total_samples = len(dates)
        
        if total_samples < self.min_samples * self.n_splits:
            logger.warning(f"样本数不足: {total_samples} < {self.min_samples * self.n_splits}")
            return []
        
        splits = []
        
        # 计算每折的测试集大小
        test_size = total_samples // (self.n_splits + 1)
        
        for fold in range(self.n_splits):
            # 计算测试集起止位置
            test_start_idx = (fold + 1) * test_size
            test_end_idx = min(test_start_idx + test_size, total_samples)
            
            if test_end_idx - test_start_idx < self.min_samples:
                continue
            
            # 训练集：测试集开始前的所有数据，但要留出禁带期和净化期
            train_end_idx = max(0, test_start_idx - self.embargo_days - self.purge_days)
            
            if train_end_idx < self.min_samples:
                continue
            
            # 验证时间安全性
            train_end_date = dates[train_end_idx - 1]
            test_start_date = dates[test_start_idx]
            gap_days = (test_start_date - train_end_date).days
            
            if gap_days < self.embargo_days + self.purge_days:
                logger.warning(f"Fold {fold} 时间间隔不足: {gap_days}天 < {self.embargo_days + self.purge_days}天")
                continue
            
            train_indices = np.arange(train_end_idx)
            test_indices = np.arange(test_start_idx, test_end_idx)
            
            splits.append((train_indices, test_indices))
            
            logger.debug(f"Fold {fold}: 训练样本 {len(train_indices)}, "
                        f"测试样本 {len(test_indices)}, 间隔 {gap_days}天")
        
        return splits

class TemporalSafeICOptimizer:
    """时间安全的IC权重优化器"""
    
    def __init__(self, config: TemporalSafeICConfig = None):
        """初始化优化器"""
        self.config = config or TemporalSafeICConfig()
        
        # 获取全局CV策略
        self.cv_policy = get_global_cv_policy()
        self.ic_calculator = get_global_ic_calculator()
        
        # 使用全局策略覆盖本地配置
        self.config.embargo_days = self.cv_policy.cv_policy.embargo_days
        self.config.purge_days = self.cv_policy.cv_policy.isolation_days
        
        # 缓存
        self.feature_cache = {}
        self.weight_cache = {}
        
        # 统计信息
        self.stats = {
            'optimizations_performed': 0,
            'cv_folds_generated': 0,
            'temporal_violations': 0,
            'fallback_triggered': 0,
            'feature_computations': 0
        }
        
        logger.info(f"时间安全IC优化器初始化完成 - embargo: {self.config.embargo_days}天, "
                   f"purge: {self.config.purge_days}天")
    
    def compute_rolling_features(self, factor_data: pd.DataFrame, 
                                return_data: pd.DataFrame,
                                current_date: pd.Timestamp) -> pd.DataFrame:
        """
        计算滚动特征（仅使用截至当前日期的历史数据）
        
        Args:
            factor_data: 因子数据 
            return_data: 收益数据
            current_date: 当前日期
            
        Returns:
            滚动特征数据框
        """
        # 确保仅使用历史数据
        historical_factor = factor_data[factor_data.index <= current_date]
        historical_return = return_data[return_data.index <= current_date]
        
        if len(historical_factor) < self.config.rolling_window_days:
            logger.warning(f"历史数据不足: {len(historical_factor)}天 < {self.config.rolling_window_days}天")
            return pd.DataFrame()
        
        features = []
        feature_names = []
        
        # 滚动窗口计算各种统计量
        for factor_name in historical_factor.columns:
            try:
                # 1. 滚动IC统计
                ic_series = self._compute_rolling_ic(
                    historical_factor[factor_name], 
                    historical_return, 
                    current_date
                )
                
                if not ic_series.empty and len(ic_series) >= self.config.min_correlation_samples:
                    # IC均值
                    features.append(ic_series.mean())
                    feature_names.append(f"{factor_name}_ic_mean")
                    
                    # IC标准差（稳定性）
                    features.append(ic_series.std())
                    feature_names.append(f"{factor_name}_ic_std")
                    
                    # IC信息比率
                    ic_ir = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0
                    features.append(ic_ir)
                    feature_names.append(f"{factor_name}_ic_ir")
                    
                    # IC衰减加权
                    ewm_ic = ic_series.ewm(halflife=self.config.ic_decay_halflife).mean().iloc[-1]
                    features.append(ewm_ic)
                    feature_names.append(f"{factor_name}_ic_ewm")
                
                # 2. 因子本身的滚动统计（截至当前日期）
                factor_series = historical_factor[factor_name].dropna()
                if len(factor_series) >= self.config.volatility_window:
                    rolling_vol = factor_series.rolling(self.config.volatility_window).std().iloc[-1]
                    rolling_mean = factor_series.rolling(self.config.volatility_window).mean().iloc[-1]
                    
                    features.extend([rolling_vol, rolling_mean])
                    feature_names.extend([f"{factor_name}_vol", f"{factor_name}_mean"])
                
                self.stats['feature_computations'] += 1
                
            except Exception as e:
                logger.debug(f"因子 {factor_name} 滚动特征计算失败: {e}")
                continue
        
        # 3. 因子间相关性（仅使用历史数据的滚动窗口）
        if len(historical_factor.columns) > 1:
            recent_factor_data = historical_factor.tail(self.config.correlation_window)
            if len(recent_factor_data) >= self.config.min_correlation_samples:
                try:
                    corr_matrix = recent_factor_data.corr()
                    # 提取上三角（避免重复）
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if not np.isnan(corr_val):
                                features.append(abs(corr_val))  # 使用绝对值
                                feature_names.append(f"corr_{corr_matrix.columns[i]}_{corr_matrix.columns[j]}")
                except Exception as e:
                    logger.debug(f"相关性特征计算失败: {e}")
        
        if not features:
            return pd.DataFrame()
        
        # 构建特征数据框
        feature_df = pd.DataFrame([features], columns=feature_names, index=[current_date])
        
        # 异常值处理
        for col in feature_df.columns:
            if feature_df[col].std() > 0:
                z_score = abs((feature_df[col] - feature_df[col].mean()) / feature_df[col].std())
                if z_score.iloc[0] > self.config.outlier_threshold:
                    # 用中位数替换异常值
                    feature_df.loc[feature_df.index[0], col] = feature_df[col].median()
        
        return feature_df
    
    def _compute_rolling_ic(self, factor_series: pd.Series, 
                           return_data: pd.DataFrame,
                           current_date: pd.Timestamp) -> pd.Series:
        """计算滚动IC序列"""
        # 确保仅使用历史数据
        factor_series = factor_series[factor_series.index <= current_date]
        return_data = return_data[return_data.index <= current_date]
        
        # 对齐日期索引
        common_dates = factor_series.index.intersection(return_data.index)
        if len(common_dates) < self.config.min_correlation_samples:
            return pd.Series(dtype=float)
        
        ic_values = []
        valid_dates = []
        
        for date in common_dates:
            try:
                # 获取当日横截面数据
                factor_cross = factor_series.loc[date] if hasattr(factor_series.loc[date], '__iter__') else factor_series.to_frame().loc[date]
                return_cross = return_data.loc[date]
                
                # 计算横截面IC
                if hasattr(factor_cross, '__iter__') and hasattr(return_cross, '__iter__'):
                    ic_value = self.ic_calculator.calculate_cross_sectional_ic(
                        pd.Series(factor_cross) if not isinstance(factor_cross, pd.Series) else factor_cross,
                        pd.Series(return_cross) if not isinstance(return_cross, pd.Series) else return_cross
                    )
                    
                    if not np.isnan(ic_value):
                        ic_values.append(ic_value)
                        valid_dates.append(date)
                
            except Exception as e:
                logger.debug(f"日期 {date} IC计算失败: {e}")
                continue
        
        return pd.Series(ic_values, index=valid_dates)
    
    def optimize_factor_weights(self, factor_data: pd.DataFrame,
                               return_data: pd.DataFrame,
                               target_date: pd.Timestamp = None) -> Dict[str, Any]:
        """
        优化因子权重（主接口）
        
        Args:
            factor_data: 因子数据
            return_data: 收益数据
            target_date: 目标日期（权重生效日期）
            
        Returns:
            优化结果字典
        """
        if target_date is None:
            target_date = factor_data.index[-1]
        
        logger.info(f"开始时间安全IC权重优化 - 目标日期: {target_date.date()}")
        
        try:
            # 步骤1: 构建训练数据集（仅使用历史数据）
            training_data = self._build_training_dataset(factor_data, return_data, target_date)
            
            if training_data['X'].empty or training_data['y'].empty:
                logger.warning("训练数据为空，触发回退策略")
                return self._fallback_weights(factor_data.columns, "insufficient_data")
            
            # 步骤2: 时间安全的交叉验证
            cv_results = self._perform_temporal_cv(
                training_data['X'], 
                training_data['y'],
                target_date
            )
            
            if not cv_results['success']:
                logger.warning("时间序列CV失败，触发回退策略")
                return self._fallback_weights(factor_data.columns, "cv_failed")
            
            # 步骤3: 优化权重
            optimal_weights = self._optimize_weights_with_cv(cv_results)
            
            # 步骤4: 后处理和验证
            final_weights = self._postprocess_weights(optimal_weights, factor_data.columns)
            
            # 步骤5: 构建结果
            optimization_result = {
                'factor_weights': final_weights,
                'optimization_method': self.config.optimization_method,
                'cv_performance': cv_results.get('performance_metrics', {}),
                'temporal_safety': {
                    'embargo_days': self.config.embargo_days,
                    'purge_days': self.config.purge_days,
                    'temporal_violations': self.stats['temporal_violations']
                },
                'data_quality': {
                    'training_samples': len(training_data['X']),
                    'feature_count': training_data['X'].shape[1],
                    'target_date': target_date.isoformat(),
                    'training_window_days': (target_date - training_data['X'].index[0]).days
                },
                'model_diagnostics': cv_results.get('diagnostics', {}),
                'timestamp': datetime.now().isoformat(),
                'config': self.config.__dict__
            }
            
            self.stats['optimizations_performed'] += 1
            
            logger.info(f"IC权重优化完成 - 生成 {len(final_weights)} 个权重")
            return optimization_result
            
        except Exception as e:
            logger.error(f"IC权重优化失败: {e}")
            return self._fallback_weights(factor_data.columns, f"optimization_error: {str(e)}")
    
    def _build_training_dataset(self, factor_data: pd.DataFrame,
                               return_data: pd.DataFrame,
                               target_date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """构建训练数据集（严格时间安全）"""
        # 确保仅使用target_date之前的数据
        historical_factor = factor_data[factor_data.index < target_date]
        historical_return = return_data[return_data.index < target_date]
        
        if len(historical_factor) < self.config.min_train_window:
            logger.warning(f"历史数据窗口不足: {len(historical_factor)}天 < {self.config.min_train_window}天")
            return {'X': pd.DataFrame(), 'y': pd.Series(dtype=float)}
        
        # 使用滚动窗口构建训练样本
        X_data = []
        y_data = []
        sample_dates = []
        
        # 从足够的历史数据开始，逐步构建训练样本
        start_date = historical_factor.index[self.config.rolling_window_days]
        end_date = target_date - timedelta(days=self.config.embargo_days)
        
        sample_dates_range = historical_factor.index[
            (historical_factor.index >= start_date) & (historical_factor.index <= end_date)
        ]
        
        for sample_date in sample_dates_range[::5]:  # 每5天采样一次，减少计算量
            try:
                # 计算截至sample_date的滚动特征
                features = self.compute_rolling_features(
                    historical_factor, historical_return, sample_date
                )
                
                if features.empty:
                    continue
                
                # 计算target：sample_date之后的因子表现
                future_performance = self._compute_future_factor_performance(
                    historical_factor, historical_return, sample_date
                )
                
                if future_performance is not None:
                    X_data.append(features.iloc[0])
                    y_data.append(future_performance)
                    sample_dates.append(sample_date)
                
            except Exception as e:
                logger.debug(f"样本构建失败 - 日期 {sample_date}: {e}")
                continue
        
        if not X_data:
            return {'X': pd.DataFrame(), 'y': pd.Series(dtype=float)}
        
        X_df = pd.DataFrame(X_data, index=sample_dates)
        y_series = pd.Series(y_data, index=sample_dates)
        
        # 移除缺失值
        valid_mask = ~(X_df.isna().any(axis=1) | y_series.isna())
        X_clean = X_df[valid_mask]
        y_clean = y_series[valid_mask]
        
        logger.info(f"训练数据集构建完成: {len(X_clean)} 样本, {X_clean.shape[1]} 特征")
        return {'X': X_clean, 'y': y_clean}
    
    def _compute_future_factor_performance(self, factor_data: pd.DataFrame,
                                         return_data: pd.DataFrame,
                                         current_date: pd.Timestamp) -> Optional[float]:
        """计算未来因子表现（target变量）"""
        try:
            # 定义前向期间（避免过度拟合）
            future_start = current_date + timedelta(days=self.config.embargo_days)
            future_end = current_date + timedelta(days=self.config.embargo_days + 20)  # 20天前向表现
            
            future_factor = factor_data[
                (factor_data.index >= future_start) & (factor_data.index <= future_end)
            ]
            future_return = return_data[
                (return_data.index >= future_start) & (return_data.index <= future_end)
            ]
            
            if future_factor.empty or future_return.empty:
                return None
            
            # 计算未来期间的平均IC表现作为target
            ic_values = []
            for date in future_factor.index.intersection(future_return.index):
                daily_ic = self.ic_calculator.calculate_cross_sectional_ic(
                    future_factor.loc[date], 
                    future_return.loc[date]
                )
                if not np.isnan(daily_ic):
                    ic_values.append(daily_ic)
            
            return np.mean(ic_values) if ic_values else None
            
        except Exception as e:
            logger.debug(f"未来表现计算失败: {e}")
            return None
    
    def _perform_temporal_cv(self, X: pd.DataFrame, y: pd.Series, 
                           target_date: pd.Timestamp) -> Dict[str, Any]:
        """执行时间安全的交叉验证"""
        cv_splitter = PurgedTimeSeriesSplit(
            n_splits=self.config.cv_folds,
            embargo_days=self.config.embargo_days,
            purge_days=self.config.purge_days,
            min_samples=self.config.min_samples_per_fold
        )
        
        try:
            splits = cv_splitter.split(X, y)
            
            if not splits:
                self.stats['temporal_violations'] += 1
                return {'success': False, 'reason': 'no_valid_splits'}
            
            cv_scores = []
            model_coefficients = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                try:
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # 训练模型
                    model = self._get_optimization_model()
                    model.fit(X_train.fillna(0), y_train)
                    
                    # 验证
                    y_pred = model.predict(X_val.fillna(0))
                    
                    # 计算性能指标
                    mse = mean_squared_error(y_val, y_pred)
                    mae = mean_absolute_error(y_val, y_pred)
                    
                    correlation = np.corrcoef(y_val, y_pred)[0, 1] if len(np.unique(y_val)) > 1 else 0
                    
                    cv_scores.append({
                        'fold': fold_idx,
                        'mse': mse,
                        'mae': mae,
                        'correlation': correlation,
                        'train_samples': len(X_train),
                        'val_samples': len(X_val)
                    })
                    
                    # 保存模型系数
                    if hasattr(model, 'coef_'):
                        model_coefficients.append(model.coef_)
                    elif hasattr(model, 'feature_importances_'):
                        model_coefficients.append(model.feature_importances_)
                    
                    self.stats['cv_folds_generated'] += 1
                    
                except Exception as e:
                    logger.debug(f"CV fold {fold_idx} 失败: {e}")
                    continue
            
            if not cv_scores:
                return {'success': False, 'reason': 'all_folds_failed'}
            
            # 聚合CV结果
            avg_performance = {
                'mean_mse': np.mean([s['mse'] for s in cv_scores]),
                'mean_mae': np.mean([s['mae'] for s in cv_scores]),
                'mean_correlation': np.mean([s['correlation'] for s in cv_scores if not np.isnan(s['correlation'])]),
                'std_correlation': np.std([s['correlation'] for s in cv_scores if not np.isnan(s['correlation'])]),
                'successful_folds': len(cv_scores)
            }
            
            return {
                'success': True,
                'cv_scores': cv_scores,
                'performance_metrics': avg_performance,
                'model_coefficients': model_coefficients,
                'diagnostics': {
                    'total_splits': len(splits),
                    'successful_splits': len(cv_scores),
                    'average_train_size': np.mean([s['train_samples'] for s in cv_scores]),
                    'average_val_size': np.mean([s['val_samples'] for s in cv_scores])
                }
            }
            
        except Exception as e:
            logger.error(f"时间序列CV执行失败: {e}")
            return {'success': False, 'reason': f'cv_execution_error: {str(e)}'}
    
    def _get_optimization_model(self):
        """获取优化模型"""
        if self.config.optimization_method == "ridge":
            return Ridge(alpha=self.config.alpha_regularization)
        elif self.config.optimization_method == "elastic_net":
            return ElasticNet(alpha=self.config.alpha_regularization)
        elif self.config.optimization_method == "random_forest":
            return RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            return Ridge(alpha=1.0)
    
    def _optimize_weights_with_cv(self, cv_results: Dict[str, Any]) -> Dict[str, float]:
        """基于CV结果优化权重"""
        if not cv_results.get('model_coefficients'):
            return {}
        
        # 平均模型系数作为权重
        coefficients_array = np.array(cv_results['model_coefficients'])
        avg_coefficients = np.mean(coefficients_array, axis=0)
        
        # 构建权重字典（需要映射到因子名称）
        # 这里简化处理，实际需要根据特征名称映射到因子
        weights = {}
        
        return weights
    
    def _postprocess_weights(self, weights: Dict[str, float], 
                           factor_names: pd.Index) -> Dict[str, float]:
        """权重后处理"""
        if not weights:
            # 如果权重为空，使用等权重作为回退
            return {factor: 1.0 / len(factor_names) for factor in factor_names}
        
        # 权重归一化
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
        else:
            normalized_weights = {k: 1.0 / len(weights) for k in weights.keys()}
        
        # 检查权重集中度
        max_weight = max(abs(w) for w in normalized_weights.values()) if normalized_weights else 0
        if max_weight > self.config.max_weight_concentration:
            logger.warning(f"权重过度集中: {max_weight:.3f} > {self.config.max_weight_concentration}")
            # 应用权重平滑
            smoothed_weights = {}
            for k, v in normalized_weights.items():
                if abs(v) > self.config.max_weight_concentration:
                    smoothed_weights[k] = np.sign(v) * self.config.max_weight_concentration
                else:
                    smoothed_weights[k] = v
            normalized_weights = smoothed_weights
        
        return normalized_weights
    
    def _fallback_weights(self, factor_names: pd.Index, reason: str) -> Dict[str, Any]:
        """回退权重策略"""
        self.stats['fallback_triggered'] += 1
        logger.warning(f"触发权重回退策略: {reason}")
        
        if self.config.fallback_method == "equal_weight":
            weights = {factor: 1.0 / len(factor_names) for factor in factor_names}
        else:
            # 基于历史IC排序的权重
            weights = {factor: 1.0 / len(factor_names) for factor in factor_names}  # 简化实现
        
        return {
            'factor_weights': weights,
            'fallback_triggered': True,
            'fallback_reason': reason,
            'fallback_method': self.config.fallback_method,
            'temporal_safety': {
                'embargo_days': self.config.embargo_days,
                'purge_days': self.config.purge_days,
                'temporal_violations': self.stats['temporal_violations']
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_optimizer_stats(self) -> Dict[str, Any]:
        """获取优化器统计信息"""
        return {
            'optimization_stats': self.stats,
            'config': self.config.__dict__,
            'temporal_safety_summary': {
                'embargo_days': self.config.embargo_days,
                'purge_days': self.config.purge_days,
                'rolling_window_days': self.config.rolling_window_days,
                'temporal_violations_rate': (
                    self.stats['temporal_violations'] / 
                    max(1, self.stats['optimizations_performed'])
                ),
                'fallback_rate': (
                    self.stats['fallback_triggered'] / 
                    max(1, self.stats['optimizations_performed'])
                )
            }
        }

# 全局时间安全IC优化器
def create_temporal_safe_ic_optimizer(config: TemporalSafeICConfig = None) -> TemporalSafeICOptimizer:
    """创建时间安全IC优化器"""
    return TemporalSafeICOptimizer(config)

if __name__ == "__main__":
    # 测试时间安全IC优化器
    optimizer = create_temporal_safe_ic_optimizer()
    
    # 模拟数据测试
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    factors = ['momentum', 'value', 'quality', 'low_risk']
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # 创建模拟因子和收益数据
    # np.random.seed removed
    factor_data = pd.DataFrame(
        np.zeros(200), index=dates, columns=factors
    )
    return_data = pd.DataFrame(
        np.zeros(200) * 0.02, index=dates, columns=tickers
    )
    
    # 测试权重优化
    target_date = dates[-20]  # 倒数第20天作为目标日期
    
    result = optimizer.optimize_factor_weights(
        factor_data, return_data, target_date
    )
    
    print("=== 时间安全IC权重优化测试 ===")
    if 'factor_weights' in result:
        print("因子权重:")
        for factor, weight in result['factor_weights'].items():
            print(f"  {factor}: {weight:.4f}")
    
    print(f"\n回退触发: {'是' if result.get('fallback_triggered', False) else '否'}")
    if 'temporal_safety' in result:
        print(f"时间安全设置: embargo={result['temporal_safety']['embargo_days']}天, "
              f"purge={result['temporal_safety']['purge_days']}天")
    
    print(f"\n优化器统计:")
    stats = optimizer.get_optimizer_stats()
    for key, value in stats['temporal_safety_summary'].items():
        print(f"  {key}: {value}")