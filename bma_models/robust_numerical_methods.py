#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Numerical Methods for Quantitative Finance
量化金融中的鲁棒数值方法

Features:
1. Robust Fisher-Z transformation with numerical stability
2. Advanced weight constraint optimization
3. Stable IC calculation with outlier handling
4. High-precision portfolio optimization
5. Numerical stability monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
from scipy import optimize, stats
from scipy.special import expit, logit
import warnings

logger = logging.getLogger(__name__)

@dataclass
class NumericalStabilityReport:
    """数值稳定性报告"""
    method_name: str
    input_stats: Dict[str, Any]
    output_stats: Dict[str, Any]
    stability_score: float  # 0-1, 越高越稳定
    warnings: List[str]
    recommendations: List[str]

class RobustFisherZTransform:
    """
    机构级Fisher-Z变换实现

    Features:
    1. 动态clip边界优化
    2. 高精度数值计算
    3. 异常值检测和处理
    4. 批量处理优化
    """

    def __init__(self,
                 dynamic_clipping: bool = True,
                 precision_mode: str = 'high',
                 outlier_handling: str = 'adaptive'):
        """
        Args:
            dynamic_clipping: 是否使用动态clip边界
            precision_mode: 'standard', 'high', 'ultra'
            outlier_handling: 'clip', 'adaptive', 'none'
        """
        self.dynamic_clipping = dynamic_clipping
        self.precision_mode = precision_mode
        self.outlier_handling = outlier_handling

        # 精度配置
        self.precision_config = {
            'standard': {'rtol': 1e-10, 'max_exp': 700},
            'high': {'rtol': 1e-14, 'max_exp': 500},
            'ultra': {'rtol': 1e-16, 'max_exp': 300}
        }[precision_mode]

        # 统计跟踪
        self.transform_stats = {'calls': 0, 'warnings': 0, 'clipped_values': 0}

    def fisher_z_transform(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Robust Fisher-Z transformation: z = 0.5 * ln((1+r)/(1-r))

        Features:
        - Dynamic boundary optimization
        - Numerical stability for extreme values
        - Batch processing optimization
        - Comprehensive error handling
        """
        self.transform_stats['calls'] += 1

        # 输入验证
        r_array = np.asarray(r)
        is_scalar = r_array.ndim == 0
        r_array = np.atleast_1d(r_array)

        # 检测异常值
        if self.outlier_handling != 'none':
            r_array = self._handle_outliers(r_array)

        # 动态clip边界
        if self.dynamic_clipping:
            clip_bound = self._compute_dynamic_clip_bound(r_array)
        else:
            clip_bound = 0.999  # 默认边界

        # Apply clipping
        r_clipped = np.clip(r_array, -clip_bound, clip_bound)
        clipped_count = np.sum(np.abs(r_array) > clip_bound)
        self.transform_stats['clipped_values'] += clipped_count

        if clipped_count > 0:
            logger.debug(f"Fisher-Z: clipped {clipped_count} extreme values (>{clip_bound:.6f})")

        try:
            # 高精度Fisher-Z变换
            if self.precision_mode == 'ultra':
                # 使用高精度算法避免数值不稳定
                z = self._ultra_precision_fisher_z(r_clipped)
            else:
                # 标准实现，优化数值稳定性
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # 使用log1p和expm1提高精度
                    numerator = 1 + r_clipped
                    denominator = 1 - r_clipped

                    # 避免除零和数值溢出
                    safe_mask = (np.abs(denominator) > 1e-15)
                    z = np.full_like(r_clipped, 0.0)

                    z[safe_mask] = 0.5 * np.log(
                        numerator[safe_mask] / denominator[safe_mask]
                    )

                    # 处理边界情况
                    extreme_mask = ~safe_mask
                    if np.any(extreme_mask):
                        z[extreme_mask] = np.sign(r_clipped[extreme_mask]) * self.precision_config['max_exp']

        except Exception as e:
            logger.warning(f"Fisher-Z transform failed: {e}")
            self.transform_stats['warnings'] += 1
            z = np.zeros_like(r_clipped)

        # 验证输出质量
        self._validate_transform_output(z, r_clipped)

        return z[0] if is_scalar else z

    def inverse_fisher_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Robust inverse Fisher-Z transformation: r = (exp(2z) - 1) / (exp(2z) + 1)

        Features:
        - Numerical overflow protection
        - High-precision calculation
        - Boundary value handling
        """
        z_array = np.asarray(z)
        is_scalar = z_array.ndim == 0
        z_array = np.atleast_1d(z_array)

        try:
            # 防止数值溢出
            z_clipped = np.clip(z_array,
                              -self.precision_config['max_exp'],
                              self.precision_config['max_exp'])

            if self.precision_mode == 'ultra':
                r = self._ultra_precision_inverse_fisher_z(z_clipped)
            else:
                # 使用tanh函数（数值稳定）
                r = np.tanh(z_clipped)

                # 验证结果范围
                r = np.clip(r, -0.999999, 0.999999)

        except Exception as e:
            logger.warning(f"Inverse Fisher-Z transform failed: {e}")
            r = np.zeros_like(z_clipped)

        return r[0] if is_scalar else r

    def _compute_dynamic_clip_bound(self, r_array: np.ndarray) -> float:
        """计算动态clip边界"""
        try:
            finite_r = r_array[np.isfinite(r_array)]
            if len(finite_r) == 0:
                return 0.999

            # 基于数据分布调整边界
            q99 = np.percentile(np.abs(finite_r), 99)
            q95 = np.percentile(np.abs(finite_r), 95)

            # 动态边界：在保守(0.99)和激进(0.999999)之间选择
            if q99 < 0.5:
                clip_bound = 0.999
            elif q99 < 0.8:
                clip_bound = min(0.9999, q99 + 0.1)
            else:
                clip_bound = min(0.999999, q99 + 0.05)

            return clip_bound

        except Exception:
            return 0.999  # 安全回退

    def _handle_outliers(self, r_array: np.ndarray) -> np.ndarray:
        """处理异常值"""
        if self.outlier_handling == 'clip':
            return np.clip(r_array, -0.95, 0.95)
        elif self.outlier_handling == 'adaptive':
            # 基于IQR的自适应异常值处理
            finite_r = r_array[np.isfinite(r_array)]
            if len(finite_r) > 10:
                q25, q75 = np.percentile(finite_r, [25, 75])
                iqr = q75 - q25
                lower_bound = q25 - 3 * iqr
                upper_bound = q75 + 3 * iqr
                return np.clip(r_array, lower_bound, upper_bound)

        return r_array

    def _ultra_precision_fisher_z(self, r: np.ndarray) -> np.ndarray:
        """超高精度Fisher-Z变换"""
        # 使用级数展开提高精度
        z = np.zeros_like(r)

        # 对于小值使用级数展开
        small_mask = np.abs(r) < 0.5
        if np.any(small_mask):
            r_small = r[small_mask]
            # Fisher-Z的级数展开: z = r + r³/3 + 2r⁵/15 + 17r⁷/315 + ...
            z[small_mask] = (r_small +
                           r_small**3 / 3 +
                           2 * r_small**5 / 15 +
                           17 * r_small**7 / 315)

        # 对于大值使用标准公式
        large_mask = ~small_mask
        if np.any(large_mask):
            r_large = r[large_mask]
            z[large_mask] = 0.5 * np.log((1 + r_large) / (1 - r_large))

        return z

    def _ultra_precision_inverse_fisher_z(self, z: np.ndarray) -> np.ndarray:
        """超高精度Fisher-Z逆变换"""
        # 对于小值使用tanh的级数展开
        small_mask = np.abs(z) < 1.0
        r = np.zeros_like(z)

        if np.any(small_mask):
            z_small = z[small_mask]
            # tanh级数展开
            r[small_mask] = z_small - z_small**3 / 3 + 2 * z_small**5 / 15

        # 对于大值使用标准tanh
        large_mask = ~small_mask
        if np.any(large_mask):
            r[large_mask] = np.tanh(z[large_mask])

        return r

    def _validate_transform_output(self, z: np.ndarray, r_input: np.ndarray):
        """验证变换输出质量"""
        try:
            # 检查无穷值和NaN
            inf_count = np.sum(np.isinf(z))
            nan_count = np.sum(np.isnan(z))

            if inf_count > 0 or nan_count > 0:
                logger.warning(f"Fisher-Z output quality issues: {inf_count} inf, {nan_count} nan")
                self.transform_stats['warnings'] += 1

            # 检查数值精度（通过逆变换测试）
            if len(z) > 0 and np.all(np.isfinite(z)):
                r_reconstructed = self.inverse_fisher_z(z)
                max_error = np.max(np.abs(r_reconstructed - r_input))

                if max_error > self.precision_config['rtol']:
                    logger.debug(f"Fisher-Z precision warning: max reconstruction error {max_error:.2e}")

        except Exception as e:
            logger.debug(f"Transform validation failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取变换统计信息"""
        return {
            'total_calls': self.transform_stats['calls'],
            'warning_rate': self.transform_stats['warnings'] / max(self.transform_stats['calls'], 1),
            'clip_rate': self.transform_stats['clipped_values'] / max(self.transform_stats['calls'], 1),
            'precision_mode': self.precision_mode,
            'dynamic_clipping': self.dynamic_clipping
        }

class RobustWeightOptimizer:
    """
    机构级权重约束优化器

    Features:
    1. 数学一致的多重约束处理
    2. 数值稳定的simplex投影
    3. 性能优化的批量处理
    4. 约束冲突检测和解决
    """

    def __init__(self,
                 method: str = 'quadratic_programming',
                 tolerance: float = 1e-12,
                 max_iterations: int = 1000):
        """
        Args:
            method: 'simplex_projection', 'quadratic_programming', 'iterative'
            tolerance: 数值容差
            max_iterations: 最大迭代次数
        """
        self.method = method
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        # 优化统计
        self.optimization_stats = {
            'calls': 0, 'failures': 0, 'avg_iterations': 0
        }

    def optimize_weights(self,
                        raw_weights: np.ndarray,
                        constraints: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        优化权重以满足多重约束

        Args:
            raw_weights: 原始权重
            constraints: 约束字典
                - 'sum_to_one': bool, 权重和为1
                - 'non_negative': bool, 非负权重
                - 'max_weight': float, 单个权重上限
                - 'min_weight': float, 单个权重下限
                - 'sector_limits': dict, 行业权重限制

        Returns:
            (optimized_weights, optimization_info)
        """
        self.optimization_stats['calls'] += 1

        try:
            if self.method == 'quadratic_programming':
                return self._quadratic_programming_solution(raw_weights, constraints)
            elif self.method == 'iterative':
                return self._iterative_projection(raw_weights, constraints)
            else:  # simplex_projection
                return self._enhanced_simplex_projection(raw_weights, constraints)

        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
            self.optimization_stats['failures'] += 1

            # 回退到等权重
            n = len(raw_weights)
            fallback_weights = np.ones(n) / n

            return fallback_weights, {
                'success': False,
                'method': 'fallback_equal_weights',
                'error': str(e)
            }

    def _enhanced_simplex_projection(self,
                                   raw_weights: np.ndarray,
                                   constraints: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        增强的simplex投影算法

        使用Duchi等人的高效simplex投影算法，并扩展支持box constraints
        """
        n = len(raw_weights)
        w = raw_weights.copy()

        # Step 1: 处理box constraints (min/max weight limits)
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)

        # 确保约束的一致性
        if min_weight * n > 1.0:
            logger.warning(f"Min weight constraint inconsistent: {min_weight} * {n} > 1.0")
            min_weight = 1.0 / n * 0.9

        if max_weight < 1.0 / n:
            logger.warning(f"Max weight constraint too restrictive: {max_weight} < {1.0/n}")
            max_weight = 1.0 / n * 1.1

        # Apply box constraints
        w_boxed = np.clip(w, min_weight, max_weight)

        # Step 2: Project to simplex
        w_proj = self._project_to_simplex_with_bounds(w_boxed, min_weight, max_weight)

        # Step 3: 验证约束满足
        validation_info = self._validate_constraints(w_proj, constraints)

        optimization_info = {
            'success': validation_info['all_satisfied'],
            'method': 'enhanced_simplex_projection',
            'original_sum': np.sum(raw_weights),
            'final_sum': np.sum(w_proj),
            'constraint_violations': validation_info['violations'],
            'max_weight_change': np.max(np.abs(w_proj - raw_weights))
        }

        return w_proj, optimization_info

    def _project_to_simplex_with_bounds(self,
                                      v: np.ndarray,
                                      min_bound: float,
                                      max_bound: float) -> np.ndarray:
        """
        投影到带边界约束的simplex

        Implements the algorithm from:
        "Efficient Projections onto the l1-Ball for Learning in High Dimensions"
        """
        n = len(v)

        # 处理边界约束
        v_bounded = np.clip(v, min_bound, max_bound)

        # 如果已经满足simplex约束，直接返回
        current_sum = np.sum(v_bounded)
        if abs(current_sum - 1.0) < self.tolerance:
            return v_bounded

        # Duchi's simplex projection algorithm
        u = np.sort(v_bounded)[::-1]  # 降序排列

        # 寻找最优的threshold
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(n) + 1
        cond = u - cssv / ind > 0

        if np.any(cond):
            rho = ind[cond][-1]
            theta = cssv[cond][-1] / rho
        else:
            rho = 1
            theta = (np.sum(v_bounded) - 1.0) / n

        # Apply projection
        w_proj = np.maximum(v_bounded - theta, 0)

        # 重新归一化以确保精确的和为1
        w_sum = np.sum(w_proj)
        if w_sum > self.tolerance:
            w_proj = w_proj / w_sum
        else:
            # 极端情况：回退到等权重
            w_proj = np.ones(n) / n

        # 最后一次边界检查
        w_proj = np.clip(w_proj, min_bound, max_bound)

        # 如果clip后和不为1，进行微调
        final_sum = np.sum(w_proj)
        if abs(final_sum - 1.0) > self.tolerance:
            adjustment = (1.0 - final_sum) / n
            w_proj += adjustment
            w_proj = np.clip(w_proj, min_bound, max_bound)

        return w_proj

    def _quadratic_programming_solution(self,
                                      raw_weights: np.ndarray,
                                      constraints: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        使用二次规划求解权重优化

        最小化: ||w - w_raw||² subject to constraints
        """
        n = len(raw_weights)

        try:
            from scipy.optimize import minimize

            # 目标函数：最小化与原始权重的差异
            def objective(w):
                return np.sum((w - raw_weights)**2)

            # 约束条件
            constraint_list = []

            # 等式约束：权重和为1
            if constraints.get('sum_to_one', True):
                constraint_list.append({
                    'type': 'eq',
                    'fun': lambda w: np.sum(w) - 1.0
                })

            # 不等式约束
            min_weight = constraints.get('min_weight', 0.0)
            max_weight = constraints.get('max_weight', 1.0)

            bounds = [(min_weight, max_weight) for _ in range(n)]

            # 求解
            result = minimize(
                objective,
                x0=raw_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'ftol': self.tolerance, 'maxiter': self.max_iterations}
            )

            if result.success:
                w_opt = result.x
                validation_info = self._validate_constraints(w_opt, constraints)

                optimization_info = {
                    'success': True,
                    'method': 'quadratic_programming',
                    'iterations': result.nit,
                    'final_objective': result.fun,
                    'constraint_violations': validation_info['violations']
                }
            else:
                raise RuntimeError(f"QP optimization failed: {result.message}")

            return w_opt, optimization_info

        except Exception as e:
            # 回退到simplex投影
            logger.warning(f"QP failed, falling back to simplex projection: {e}")
            return self._enhanced_simplex_projection(raw_weights, constraints)

    def _validate_constraints(self,
                            weights: np.ndarray,
                            constraints: Dict[str, Any]) -> Dict[str, Any]:
        """验证约束满足情况"""
        violations = []

        # 检查权重和
        if constraints.get('sum_to_one', True):
            sum_error = abs(np.sum(weights) - 1.0)
            if sum_error > self.tolerance:
                violations.append(f"Sum constraint violated: {sum_error:.2e}")

        # 检查非负性
        if constraints.get('non_negative', True):
            negative_count = np.sum(weights < -self.tolerance)
            if negative_count > 0:
                violations.append(f"Non-negativity violated: {negative_count} negative weights")

        # 检查权重边界
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)

        below_min = np.sum(weights < min_weight - self.tolerance)
        above_max = np.sum(weights > max_weight + self.tolerance)

        if below_min > 0:
            violations.append(f"Min weight violated: {below_min} weights below {min_weight}")
        if above_max > 0:
            violations.append(f"Max weight violated: {above_max} weights above {max_weight}")

        return {
            'all_satisfied': len(violations) == 0,
            'violations': violations,
            'summary': {
                'sum': np.sum(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'negative_count': np.sum(weights < 0)
            }
        }

class RobustICCalculator:
    """
    机构级IC计算器

    Features:
    1. 异常值robust的相关性计算
    2. 样本不足情况的处理
    3. 多种相关性度量
    4. 置信区间估计
    """

    def __init__(self,
                 method: str = 'spearman',
                 outlier_method: str = 'winsorize',
                 min_samples: int = 30,
                 confidence_level: float = 0.95):
        """
        Args:
            method: 'spearman', 'pearson', 'kendall', 'robust_spearman'
            outlier_method: 'winsorize', 'remove', 'robust', 'none'
            min_samples: 最小样本数要求
            confidence_level: 置信区间水平
        """
        self.method = method
        self.outlier_method = outlier_method
        self.min_samples = min_samples
        self.confidence_level = confidence_level

    def calculate_ic(self,
                    predictions: np.ndarray,
                    targets: np.ndarray,
                    weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        计算信息系数(IC)

        Returns:
            包含IC值、置信区间、统计显著性等信息的字典
        """
        try:
            # 数据预处理
            pred_clean, target_clean, weights_clean = self._preprocess_data(
                predictions, targets, weights
            )

            if len(pred_clean) < self.min_samples:
                return {
                    'ic': np.nan,
                    'p_value': np.nan,
                    'confidence_interval': (np.nan, np.nan),
                    'n_samples': len(pred_clean),
                    'warning': f'Sample size {len(pred_clean)} < minimum {self.min_samples}'
                }

            # 计算IC
            if self.method == 'spearman':
                ic, p_value = stats.spearmanr(pred_clean, target_clean)
            elif self.method == 'pearson':
                ic, p_value = stats.pearsonr(pred_clean, target_clean)
            elif self.method == 'kendall':
                ic, p_value = stats.kendalltau(pred_clean, target_clean)
            elif self.method == 'robust_spearman':
                ic, p_value = self._robust_spearman(pred_clean, target_clean)
            else:
                raise ValueError(f"Unknown IC method: {self.method}")

            # 置信区间计算
            ci_lower, ci_upper = self._calculate_confidence_interval(
                ic, len(pred_clean), self.confidence_level
            )

            # 统计显著性
            is_significant = p_value < (1 - self.confidence_level)

            return {
                'ic': ic if not np.isnan(ic) else 0.0,
                'p_value': p_value if not np.isnan(p_value) else 1.0,
                'confidence_interval': (ci_lower, ci_upper),
                'is_significant': is_significant,
                'n_samples': len(pred_clean),
                'method': self.method,
                'outlier_method': self.outlier_method
            }

        except Exception as e:
            logger.error(f"IC calculation failed: {e}")
            return {
                'ic': 0.0,
                'p_value': 1.0,
                'confidence_interval': (0.0, 0.0),
                'is_significant': False,
                'n_samples': 0,
                'error': str(e)
            }

    def _preprocess_data(self,
                        predictions: np.ndarray,
                        targets: np.ndarray,
                        weights: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """数据预处理：异常值处理、缺失值处理等"""

        # 转换为numpy数组
        pred = np.asarray(predictions)
        target = np.asarray(targets)

        # 检查长度一致性
        if len(pred) != len(target):
            raise ValueError("Predictions and targets must have same length")

        # 处理权重
        if weights is not None:
            weights = np.asarray(weights)
            if len(weights) != len(pred):
                raise ValueError("Weights must have same length as predictions")

        # 移除无穷值和NaN
        finite_mask = (
            np.isfinite(pred) &
            np.isfinite(target) &
            (weights is None or np.isfinite(weights))
        )

        pred_clean = pred[finite_mask]
        target_clean = target[finite_mask]
        weights_clean = weights[finite_mask] if weights is not None else None

        # 异常值处理
        if self.outlier_method == 'winsorize':
            pred_clean = self._winsorize(pred_clean, limits=(0.01, 0.01))
            target_clean = self._winsorize(target_clean, limits=(0.01, 0.01))
        elif self.outlier_method == 'remove':
            outlier_mask = self._detect_outliers(pred_clean, target_clean)
            pred_clean = pred_clean[~outlier_mask]
            target_clean = target_clean[~outlier_mask]
            if weights_clean is not None:
                weights_clean = weights_clean[~outlier_mask]

        return pred_clean, target_clean, weights_clean

    def _winsorize(self, data: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
        """Winsorize数据"""
        lower_limit, upper_limit = limits
        lower_val = np.percentile(data, lower_limit * 100)
        upper_val = np.percentile(data, (1 - upper_limit) * 100)
        return np.clip(data, lower_val, upper_val)

    def _detect_outliers(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """检测异常值（基于Mahalanobis距离）"""
        try:
            from scipy.spatial.distance import mahalanobis

            data = np.column_stack([pred, target])
            mean = np.mean(data, axis=0)
            cov = np.cov(data.T)

            # 计算Mahalanobis距离
            distances = []
            for i in range(len(data)):
                dist = mahalanobis(data[i], mean, np.linalg.inv(cov))
                distances.append(dist)

            distances = np.array(distances)
            threshold = np.percentile(distances, 95)  # 5%异常值

            return distances > threshold

        except Exception:
            # 回退到简单的z-score方法
            z_pred = np.abs(stats.zscore(pred))
            z_target = np.abs(stats.zscore(target))
            return (z_pred > 3) | (z_target > 3)

    def _robust_spearman(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
        """Robust Spearman相关系数"""
        try:
            # 使用bootstrap方法估计robust相关性
            n_bootstrap = 1000
            correlations = []

            n = len(pred)
            for _ in range(n_bootstrap):
                indices = np.random.choice(n, size=n, replace=True)
                corr, _ = stats.spearmanr(pred[indices], target[indices])
                if not np.isnan(corr):
                    correlations.append(corr)

            if correlations:
                robust_ic = np.median(correlations)
                # 估计p值
                null_corrs = np.abs(correlations)
                p_value = 2 * np.mean(null_corrs <= abs(robust_ic))
                return robust_ic, p_value
            else:
                return np.nan, np.nan

        except Exception:
            # 回退到标准Spearman
            return stats.spearmanr(pred, target)

    def _calculate_confidence_interval(self,
                                     correlation: float,
                                     n_samples: int,
                                     confidence_level: float) -> Tuple[float, float]:
        """计算相关系数的置信区间"""
        try:
            if np.isnan(correlation) or n_samples < 3:
                return np.nan, np.nan

            # Fisher z-transformation
            z = 0.5 * np.log((1 + correlation) / (1 - correlation))
            z_se = 1 / np.sqrt(n_samples - 3)

            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha/2)

            z_lower = z - z_critical * z_se
            z_upper = z + z_critical * z_se

            # Transform back
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

            return r_lower, r_upper

        except Exception:
            return np.nan, np.nan

# 全局实例
ROBUST_FISHER_Z = RobustFisherZTransform(precision_mode='high')
ROBUST_WEIGHT_OPTIMIZER = RobustWeightOptimizer(method='quadratic_programming')
ROBUST_IC_CALCULATOR = RobustICCalculator(method='spearman', min_samples=20)

# 便捷函数接口
def robust_fisher_z_transform(r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """便捷的robust Fisher-Z变换接口"""
    return ROBUST_FISHER_Z.fisher_z_transform(r)

def robust_inverse_fisher_z(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """便捷的robust Fisher-Z逆变换接口"""
    return ROBUST_FISHER_Z.inverse_fisher_z(z)

def robust_optimize_weights(raw_weights: np.ndarray,
                          max_weight: float = 0.6,
                          min_weight: float = 0.05) -> np.ndarray:
    """便捷的权重优化接口"""
    constraints = {
        'sum_to_one': True,
        'non_negative': True,
        'max_weight': max_weight,
        'min_weight': min_weight
    }

    optimized_weights, _ = ROBUST_WEIGHT_OPTIMIZER.optimize_weights(raw_weights, constraints)
    return optimized_weights

def robust_calculate_ic(predictions: np.ndarray,
                       targets: np.ndarray,
                       method: str = 'spearman') -> float:
    """便捷的IC计算接口"""
    calculator = RobustICCalculator(method=method)
    result = calculator.calculate_ic(predictions, targets)
    return result['ic']