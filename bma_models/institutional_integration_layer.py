#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Institutional Integration Layer for BMA Enhanced Model
专业量化机构级别的BMA模型集成层

Features:
1. 无缝集成到现有BMA系统
2. T+10 Excel输出完整性验证
3. 实时数值稳定性监控
4. 机构级别的风险控制
5. 性能和质量追踪
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import warnings

# 导入我们的robust模块
from .institutional_validation_framework import (
    InstitutionalT10Validator, validate_t10_output, T10ValidationResult
)
from .robust_numerical_methods import (
    RobustFisherZTransform, RobustWeightOptimizer, RobustICCalculator,
    robust_fisher_z_transform, robust_inverse_fisher_z, robust_optimize_weights
)

logger = logging.getLogger(__name__)

@dataclass
class ExcelOutputValidation:
    """Excel输出验证结果"""
    is_valid: bool
    t10_alignment_confirmed: bool
    ticker_mapping_verified: bool
    prediction_quality_score: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    export_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InstitutionalMetrics:
    """机构级别指标"""
    sharpe_ratio: float
    max_drawdown: float
    information_ratio: float
    tracking_error: float
    beta_stability: float
    regime_consistency: float

class InstitutionalBMAIntegration:
    """
    机构级BMA集成层

    将所有改进无缝集成到现有的量化模型_bma_ultra_enhanced.py中
    """

    def __init__(self,
                 enable_robust_numerics: bool = True,
                 enable_t10_validation: bool = True,
                 enable_excel_verification: bool = True,
                 monitoring_level: str = 'institutional'):
        """
        Args:
            enable_robust_numerics: 启用数值稳定性改进
            enable_t10_validation: 启用T+10验证
            enable_excel_verification: 启用Excel输出验证
            monitoring_level: 'basic', 'standard', 'institutional'
        """
        self.enable_robust_numerics = enable_robust_numerics
        self.enable_t10_validation = enable_t10_validation
        self.enable_excel_verification = enable_excel_verification
        self.monitoring_level = monitoring_level

        # 初始化组件
        if enable_robust_numerics:
            self.fisher_z = RobustFisherZTransform(precision_mode='high')
            self.weight_optimizer = RobustWeightOptimizer(method='quadratic_programming')
            self.ic_calculator = RobustICCalculator(method='spearman')

        if enable_t10_validation:
            self.t10_validator = InstitutionalT10Validator()

        # 性能监控
        self.performance_tracker = InstitutionalPerformanceTracker()

        # 集成点统计
        self.integration_stats = {
            'fisher_z_calls': 0,
            'weight_optimizations': 0,
            'ic_calculations': 0,
            't10_validations': 0,
            'excel_exports': 0,
            'numerical_warnings': 0
        }

    # ===== 核心集成方法：直接替换现有系统中的关键函数 =====

    def enhanced_fisher_z_transform(self, r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        替换现有的_fisher_z_transform函数

        集成点：量化模型_bma_ultra_enhanced.py line ~5828
        """
        # 形状和约束断言
        if isinstance(r, np.ndarray):
            assert r.ndim <= 2, f"Fisher-Z input must be 1D or 2D, got shape {r.shape}"
            assert np.all(np.abs(r) <= 1.0), "Correlation values must be in [-1, 1]"
            if r.size > 0:
                assert not np.all(np.isnan(r)), "Input contains all NaN values"

        if not self.enable_robust_numerics:
            # 回退到简单实现
            r_clipped = np.clip(np.asarray(r), -0.999, 0.999)
            return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))

        self.integration_stats['fisher_z_calls'] += 1

        try:
            result = self.fisher_z.fisher_z_transform(r)

            # 输出形状验证
            if isinstance(r, np.ndarray):
                assert result.shape == r.shape, f"Shape mismatch: input {r.shape}, output {result.shape}"

            # 质量监控
            self._monitor_fisher_z_quality(r, result)

            return result

        except (AssertionError, ValueError) as e:
            logger.error(f"Fisher-Z shape/constraint assertion failed: {e}")
            logger.warning("Falling back to basic implementation")
            self.integration_stats['numerical_warnings'] += 1
            r_clipped = np.clip(np.asarray(r), -0.999, 0.999)
            return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))
        except Exception as e:
            logger.warning(f"Robust Fisher-Z failed, fallback to simple: {e}")
            self.integration_stats['numerical_warnings'] += 1
            r_clipped = np.clip(np.asarray(r), -0.999, 0.999)
            return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))

    def enhanced_inverse_fisher_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        替换现有的_inverse_fisher_z函数

        集成点：量化模型_bma_ultra_enhanced.py line ~5835
        """
        if not self.enable_robust_numerics:
            return np.tanh(z)

        try:
            return self.fisher_z.inverse_fisher_z(z)
        except Exception as e:
            logger.warning(f"Robust inverse Fisher-Z failed, fallback: {e}")
            return np.tanh(z)

    def enhanced_weight_optimization(self,
                                   raw_weights: np.ndarray,
                                   meta_cfg: Dict[str, Any]) -> np.ndarray:
        """
        替换现有的权重约束逻辑

        集成点：量化模型_bma_ultra_enhanced.py line ~5952-5978 (权重约束部分)
        """
        # 形状和约束断言
        assert isinstance(raw_weights, np.ndarray), "Weights must be numpy array"
        assert raw_weights.ndim == 1, f"Weights must be 1D, got shape {raw_weights.shape}"
        assert raw_weights.size > 0, "Weights array cannot be empty"
        assert not np.all(np.isnan(raw_weights)), "Weights contain all NaN values"

        if not self.enable_robust_numerics:
            # 回退到现有的simplex投影
            return self._fallback_weight_constraints(raw_weights, meta_cfg)

        self.integration_stats['weight_optimizations'] += 1

        try:
            # 验证配置约束
            max_weight = meta_cfg.get('cap', 0.6)
            min_weight = meta_cfg.get('weight_floor', 0.05)
            assert 0 < min_weight < max_weight <= 1.0, f"Invalid weight bounds: min={min_weight}, max={max_weight}"

            # 构建约束
            constraints = {
                'sum_to_one': True,
                'non_negative': True,
                'max_weight': max_weight,
                'min_weight': min_weight
            }

            optimized_weights, opt_info = self.weight_optimizer.optimize_weights(
                raw_weights, constraints
            )

            # 验证输出
            assert optimized_weights.shape == raw_weights.shape, "Weight shape mismatch after optimization"
            assert np.abs(np.sum(optimized_weights) - 1.0) < 1e-6, "Weights don't sum to 1"
            assert np.all(optimized_weights >= 0), "Negative weights detected"
            assert np.all(optimized_weights <= max_weight + 1e-6), f"Weight exceeds max: {np.max(optimized_weights)}"

            # 监控优化质量
            self._monitor_weight_optimization(raw_weights, optimized_weights, opt_info)

            return optimized_weights

        except (AssertionError, ValueError) as e:
            logger.error(f"Weight optimization assertion failed: {e}")
            logger.warning("Falling back to basic weight constraints")
            self.integration_stats['numerical_warnings'] += 1
            return self._fallback_weight_constraints(raw_weights, meta_cfg)
        except Exception as e:
            logger.warning(f"Enhanced weight optimization failed, fallback: {e}")
            self.integration_stats['numerical_warnings'] += 1
            return self._fallback_weight_constraints(raw_weights, meta_cfg)

    def enhanced_ic_calculation(self,
                              predictions: np.ndarray,
                              targets: np.ndarray,
                              method: str = 'spearman') -> Dict[str, Any]:
        """
        替换现有的IC计算逻辑

        集成点：量化模型_bma_ultra_enhanced.py line ~5809-5814 (IC计算部分)
        """
        if not self.enable_robust_numerics:
            # 回退到scipy.stats
            from scipy.stats import spearmanr
            try:
                ic, p_value = spearmanr(predictions, targets)
                return {'ic': ic if not np.isnan(ic) else 0.0, 'p_value': p_value}
            except:
                return {'ic': 0.0, 'p_value': 1.0}

        self.integration_stats['ic_calculations'] += 1

        try:
            result = self.ic_calculator.calculate_ic(predictions, targets)

            # 监控IC质量
            self._monitor_ic_calculation(predictions, targets, result)

            return result

        except Exception as e:
            logger.warning(f"Enhanced IC calculation failed, fallback: {e}")
            from scipy.stats import spearmanr
            try:
                ic, p_value = spearmanr(predictions, targets)
                return {'ic': ic if not np.isnan(ic) else 0.0, 'p_value': p_value}
            except:
                return {'ic': 0.0, 'p_value': 1.0}

    # ===== T+10输出验证 =====

    def validate_t10_predictions(self,
                                predictions: Union[pd.Series, np.ndarray],
                                feature_data: pd.DataFrame,
                                tickers: List[str],
                                current_date: datetime = None) -> T10ValidationResult:
        """
        完整的T+10预测验证

        集成点：在生成Excel输出之前调用
        """
        # T+10验证的形状和约束断言
        if isinstance(predictions, pd.Series):
            predictions_array = predictions.values
        else:
            predictions_array = np.asarray(predictions)

        assert predictions_array.size > 0, "Predictions array is empty"
        assert len(predictions_array) == len(feature_data), f"Length mismatch: predictions={len(predictions_array)}, features={len(feature_data)}"
        assert len(tickers) == len(predictions_array), f"Ticker count mismatch: {len(tickers)} vs {len(predictions_array)}"

        if not self.enable_t10_validation:
            # 创建简单的通过验证
            from .institutional_validation_framework import T10ValidationResult
            return T10ValidationResult(
                is_valid=True,
                confidence_score=0.9,
                issues=[],
                warnings=[]
            )

        self.integration_stats['t10_validations'] += 1

        if current_date is None:
            current_date = datetime.now()

        try:
            model_metadata = {
                'model_version': 'bma_ultra_enhanced_v2',
                'prediction_horizon': 10,
                'feature_lag': 1,
                'tickers': tickers,
                'validation_timestamp': current_date
            }

            result = self.t10_validator.validate_t10_predictions(
                predictions, feature_data, current_date, model_metadata
            )

            # 记录验证结果
            self._log_validation_result(result)

            return result

        except Exception as e:
            logger.error(f"T+10 validation failed: {e}")
            from .institutional_validation_framework import T10ValidationResult
            return T10ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                issues=[f"Validation failed: {str(e)}"],
                warnings=[]
            )

    def validate_excel_output(self,
                            excel_path: str,
                            predictions_dict: Dict[str, float],
                            feature_data: pd.DataFrame,
                            tickers: List[str]) -> ExcelOutputValidation:
        """
        验证Excel输出的完整性和正确性

        集成点：在Excel文件生成后立即调用
        """
        # Excel验证的形状和约束断言
        assert isinstance(excel_path, str) and excel_path, "Excel path must be non-empty string"
        assert isinstance(predictions_dict, dict), "Predictions must be a dictionary"
        assert len(predictions_dict) > 0, "Predictions dictionary is empty"
        assert isinstance(feature_data, pd.DataFrame), "Feature data must be DataFrame"
        assert not feature_data.empty, "Feature data is empty"
        assert isinstance(tickers, list) and len(tickers) > 0, "Tickers must be non-empty list"

        if not self.enable_excel_verification:
            return ExcelOutputValidation(
                is_valid=True,
                t10_alignment_confirmed=True,
                ticker_mapping_verified=True,
                prediction_quality_score=0.9
            )

        self.integration_stats['excel_exports'] += 1

        try:
            # 验证文件存在
            import os
            assert os.path.exists(excel_path), f"Excel file not found: {excel_path}"

            # 读取生成的Excel文件
            excel_data = self._load_excel_output(excel_path)

            # 验证组件
            validation_result = ExcelOutputValidation(is_valid=True, t10_alignment_confirmed=True,
                                                    ticker_mapping_verified=True, prediction_quality_score=0.0)

            # 1. T+10时间对齐验证
            t10_result = self._verify_t10_alignment(excel_data, feature_data)
            validation_result.t10_alignment_confirmed = t10_result['valid']
            if not t10_result['valid']:
                validation_result.issues.extend(t10_result['issues'])

            # 2. Ticker映射验证
            ticker_result = self._verify_ticker_mapping(excel_data, predictions_dict, tickers)
            validation_result.ticker_mapping_verified = ticker_result['valid']
            if not ticker_result['valid']:
                validation_result.issues.extend(ticker_result['issues'])

            # 3. 预测质量评分
            quality_score = self._assess_prediction_quality(excel_data, predictions_dict)
            validation_result.prediction_quality_score = quality_score['score']

            # 4. 综合验证判决
            validation_result.is_valid = (
                validation_result.t10_alignment_confirmed and
                validation_result.ticker_mapping_verified and
                validation_result.prediction_quality_score >= 0.8
            )

            # 5. 生成改进建议
            if not validation_result.is_valid:
                validation_result.recommendations = self._generate_excel_recommendations(validation_result)

            # 6. 记录导出元数据
            validation_result.export_metadata = {
                'export_timestamp': datetime.now(),
                'excel_path': excel_path,
                'total_predictions': len(predictions_dict),
                'total_tickers': len(tickers),
                'file_size_mb': Path(excel_path).stat().st_size / (1024*1024) if Path(excel_path).exists() else 0
            }

            return validation_result

        except Exception as e:
            logger.error(f"Excel output validation failed: {e}")
            return ExcelOutputValidation(
                is_valid=False,
                t10_alignment_confirmed=False,
                ticker_mapping_verified=False,
                prediction_quality_score=0.0,
                issues=[f"Excel validation failed: {str(e)}"]
            )

    # ===== 监控和质量保证 =====

    def _monitor_fisher_z_quality(self, input_r: Union[float, np.ndarray], output_z: Union[float, np.ndarray]):
        """监控Fisher-Z变换质量"""
        try:
            if self.monitoring_level == 'institutional':
                # 检查极端值比例
                r_array = np.asarray(input_r)
                extreme_ratio = np.mean(np.abs(r_array) > 0.95)

                if extreme_ratio > 0.1:  # 超过10%为极端值
                    logger.warning(f"High extreme correlation ratio in Fisher-Z: {extreme_ratio:.1%}")

                # 检查变换的数值稳定性
                z_array = np.asarray(output_z)
                inf_ratio = np.mean(np.isinf(z_array))
                if inf_ratio > 0:
                    logger.warning(f"Fisher-Z produced {inf_ratio:.1%} infinite values")

        except Exception as e:
            logger.debug(f"Fisher-Z monitoring failed: {e}")

    def _monitor_weight_optimization(self, raw_weights: np.ndarray,
                                   optimized_weights: np.ndarray,
                                   opt_info: Dict[str, Any]):
        """监控权重优化质量"""
        try:
            if self.monitoring_level in ['standard', 'institutional']:
                # 监控权重变化幅度
                weight_change = np.mean(np.abs(optimized_weights - raw_weights))
                if weight_change > 0.3:  # 平均权重变化超过30%
                    logger.warning(f"Large weight adjustment: mean change {weight_change:.1%}")

                # 监控约束满足情况
                if 'constraint_violations' in opt_info and opt_info['constraint_violations']:
                    logger.warning(f"Weight optimization violations: {opt_info['constraint_violations']}")

                # 监控多样化程度
                hhi = np.sum(optimized_weights**2)
                effective_stocks = 1.0 / hhi
                if effective_stocks < 5:  # 有效股票数少于5只
                    logger.warning(f"Low diversification: effective stocks {effective_stocks:.1f}")

        except Exception as e:
            logger.debug(f"Weight optimization monitoring failed: {e}")

    def _monitor_ic_calculation(self, predictions: np.ndarray,
                              targets: np.ndarray,
                              result: Dict[str, Any]):
        """监控IC计算质量"""
        try:
            if self.monitoring_level == 'institutional':
                ic_value = result.get('ic', 0.0)
                p_value = result.get('p_value', 1.0)
                n_samples = result.get('n_samples', 0)

                # IC质量监控
                if abs(ic_value) > 0.5:  # IC过高可能有问题
                    logger.warning(f"Unusually high IC: {ic_value:.3f}")

                if p_value > 0.05 and abs(ic_value) > 0.1:  # IC不显著但值较大
                    logger.info(f"High IC but not significant: IC={ic_value:.3f}, p={p_value:.3f}")

                if n_samples < 50:  # 样本数不足
                    logger.warning(f"Low sample count for IC: {n_samples}")

        except Exception as e:
            logger.debug(f"IC calculation monitoring failed: {e}")

    def _verify_t10_alignment(self, excel_data: Dict[str, pd.DataFrame],
                            feature_data: pd.DataFrame) -> Dict[str, Any]:
        """验证T+10时间对齐"""
        try:
            issues = []

            # 检查预测时间范围
            if '推荐列表' in excel_data:
                predictions_df = excel_data['推荐列表']

                # 验证预测是否真的是T+10
                if isinstance(feature_data.index, pd.MultiIndex) and 'date' in feature_data.index.names:
                    latest_feature_date = feature_data.index.get_level_values('date').max()
                    expected_prediction_date = latest_feature_date + timedelta(days=10)

                    # 检查是否有预测日期信息
                    if 'prediction_date' in predictions_df.columns:
                        pred_dates = pd.to_datetime(predictions_df['prediction_date'])
                        date_diff = (pred_dates.max() - expected_prediction_date).days

                        if abs(date_diff) > 2:  # 允许2天误差
                            issues.append(f"T+10 alignment issue: expected {expected_prediction_date}, got {pred_dates.max()}")
                    else:
                        # 如果没有预测日期列，发出警告
                        issues.append("Excel输出缺少prediction_date列，无法验证T+10对齐")

            return {
                'valid': len(issues) == 0,
                'issues': issues
            }

        except Exception as e:
            return {
                'valid': False,
                'issues': [f"T+10 alignment verification failed: {str(e)}"]
            }

    def _verify_ticker_mapping(self, excel_data: Dict[str, pd.DataFrame],
                             predictions_dict: Dict[str, float],
                             tickers: List[str]) -> Dict[str, Any]:
        """验证ticker映射正确性"""
        try:
            issues = []

            if '推荐列表' in excel_data:
                excel_tickers = set(excel_data['推荐列表']['ticker'].tolist())
                prediction_tickers = set(predictions_dict.keys())
                input_tickers = set(tickers)

                # 检查ticker一致性
                missing_in_excel = prediction_tickers - excel_tickers
                extra_in_excel = excel_tickers - prediction_tickers

                if missing_in_excel:
                    issues.append(f"Missing tickers in Excel: {list(missing_in_excel)[:5]}")

                if extra_in_excel:
                    issues.append(f"Extra tickers in Excel: {list(extra_in_excel)[:5]}")

                # 检查预测值映射
                if '预测值' in excel_data['推荐列表'].columns:
                    for _, row in excel_data['推荐列表'].head(10).iterrows():  # 检查前10个
                        ticker = row['ticker']
                        excel_pred = row['预测值']
                        original_pred = predictions_dict.get(ticker)

                        if original_pred is not None:
                            diff = abs(excel_pred - original_pred)
                            if diff > 1e-6:  # 数值精度检查
                                issues.append(f"Prediction mismatch for {ticker}: {excel_pred} vs {original_pred}")

            return {
                'valid': len(issues) == 0,
                'issues': issues
            }

        except Exception as e:
            return {
                'valid': False,
                'issues': [f"Ticker mapping verification failed: {str(e)}"]
            }

    def _assess_prediction_quality(self, excel_data: Dict[str, pd.DataFrame],
                                 predictions_dict: Dict[str, float]) -> Dict[str, Any]:
        """评估预测质量"""
        try:
            if '推荐列表' not in excel_data:
                return {'score': 0.0}

            predictions_df = excel_data['推荐列表']
            if '预测值' not in predictions_df.columns:
                return {'score': 0.5}

            pred_values = predictions_df['预测值'].values
            finite_pred = pred_values[np.isfinite(pred_values)]

            if len(finite_pred) == 0:
                return {'score': 0.0}

            # 质量评分组件
            score_components = []

            # 1. 预测值的方差（应该有足够的区分度）
            pred_std = np.std(finite_pred)
            variance_score = min(1.0, pred_std / 0.05)  # 标准差5%为满分
            score_components.append(('variance', variance_score, 0.3))

            # 2. 异常值比例（不应过多）
            extreme_ratio = np.mean(np.abs(finite_pred) > 0.5)
            extreme_score = max(0.0, 1.0 - extreme_ratio * 10)  # 超过10%严重扣分
            score_components.append(('extreme_values', extreme_score, 0.2))

            # 3. 多空平衡（不应过度偏向）
            positive_ratio = np.mean(finite_pred > 0)
            balance_score = 1.0 - 2 * abs(positive_ratio - 0.5)  # 50/50为最佳
            score_components.append(('balance', balance_score, 0.2))

            # 4. 排序合理性（排序应该与预测值一致）
            if '排名' in predictions_df.columns:
                ranks = predictions_df['排名'].values
                rank_corr = np.corrcoef(ranks, -pred_values)[0, 1]  # 负相关（排名越小预测越高）
                ranking_score = max(0.0, rank_corr)
            else:
                ranking_score = 0.8  # 没有排名信息时给中等分
            score_components.append(('ranking', ranking_score, 0.3))

            # 综合评分
            total_score = sum(score * weight for _, score, weight in score_components)

            return {
                'score': total_score,
                'components': dict((name, score) for name, score, _ in score_components)
            }

        except Exception as e:
            logger.warning(f"Prediction quality assessment failed: {e}")
            return {'score': 0.5}

    def _load_excel_output(self, excel_path: str) -> Dict[str, pd.DataFrame]:
        """加载Excel输出文件"""
        try:
            excel_file = pd.ExcelFile(excel_path)
            excel_data = {}

            for sheet_name in excel_file.sheet_names:
                excel_data[sheet_name] = pd.read_excel(excel_path, sheet_name=sheet_name)

            return excel_data

        except Exception as e:
            logger.error(f"Failed to load Excel output: {e}")
            return {}

    def _generate_excel_recommendations(self, validation_result: ExcelOutputValidation) -> List[str]:
        """生成Excel输出改进建议"""
        recommendations = []

        if not validation_result.t10_alignment_confirmed:
            recommendations.append("检查T+10时间对齐逻辑，确保预测日期正确")

        if not validation_result.ticker_mapping_verified:
            recommendations.append("验证ticker映射逻辑，确保股票代码一致性")

        if validation_result.prediction_quality_score < 0.8:
            recommendations.append("改进预测质量：检查特征工程和模型训练")

        if validation_result.prediction_quality_score < 0.5:
            recommendations.append("预测质量严重不足，建议重新训练模型")

        return recommendations

    def _fallback_weight_constraints(self, raw_weights: np.ndarray, meta_cfg: Dict[str, Any]) -> np.ndarray:
        """权重约束的回退实现（使用现有逻辑）"""
        try:
            # 简化的simplex投影
            cap = meta_cfg.get('cap', 0.6)
            w_floor = meta_cfg.get('weight_floor', 0.05)

            # 基本约束
            w = np.maximum(raw_weights, 0.0)
            w = np.minimum(w, cap)
            w = np.maximum(w, w_floor)

            # 归一化
            w_sum = np.sum(w)
            if w_sum > 0:
                w = w / w_sum
            else:
                w = np.ones(len(w)) / len(w)

            return w

        except Exception:
            # 最终回退
            return np.ones(len(raw_weights)) / len(raw_weights)

    def _log_validation_result(self, result: T10ValidationResult):
        """记录验证结果"""
        try:
            if result.is_valid:
                logger.info(f"✅ T+10 validation passed (confidence: {result.confidence_score:.3f})")
            else:
                logger.warning(f"❌ T+10 validation failed ({len(result.issues)} issues)")
                for issue in result.issues[:3]:  # 只记录前3个问题
                    logger.warning(f"  • {issue}")

        except Exception as e:
            logger.debug(f"Validation result logging failed: {e}")

    def get_integration_stats(self) -> Dict[str, Any]:
        """获取集成统计信息"""
        return {
            'integration_stats': self.integration_stats.copy(),
            'robust_numerics_enabled': self.enable_robust_numerics,
            't10_validation_enabled': self.enable_t10_validation,
            'excel_verification_enabled': self.enable_excel_verification,
            'monitoring_level': self.monitoring_level
        }

class InstitutionalPerformanceTracker:
    """机构级性能追踪器"""

    def __init__(self):
        self.metrics_history = []
        self.performance_alerts = []

    def record_performance(self, metrics: InstitutionalMetrics):
        """记录性能指标"""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })

        # 性能警报检查
        self._check_performance_alerts(metrics)

    def _check_performance_alerts(self, metrics: InstitutionalMetrics):
        """检查性能警报"""
        alerts = []

        if metrics.sharpe_ratio < 0.5:
            alerts.append(f"Low Sharpe ratio: {metrics.sharpe_ratio:.2f}")

        if metrics.max_drawdown > 0.2:
            alerts.append(f"High max drawdown: {metrics.max_drawdown:.1%}")

        if metrics.information_ratio < 0.3:
            alerts.append(f"Low information ratio: {metrics.information_ratio:.2f}")

        for alert in alerts:
            self.performance_alerts.append({
                'timestamp': datetime.now(),
                'alert': alert,
                'severity': 'warning'
            })

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {'message': 'No performance data available'}

        recent_metrics = self.metrics_history[-10:]  # 最近10次

        return {
            'total_records': len(self.metrics_history),
            'recent_avg_sharpe': np.mean([m['metrics'].sharpe_ratio for m in recent_metrics]),
            'recent_avg_max_dd': np.mean([m['metrics'].max_drawdown for m in recent_metrics]),
            'recent_alerts': len([a for a in self.performance_alerts[-20:] if a['severity'] == 'warning']),
            'last_update': recent_metrics[-1]['timestamp'] if recent_metrics else None
        }

# 全局集成实例
INSTITUTIONAL_INTEGRATION = InstitutionalBMAIntegration()

# 便捷接口函数
def integrate_fisher_z_transform(r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """集成的Fisher-Z变换接口"""
    return INSTITUTIONAL_INTEGRATION.enhanced_fisher_z_transform(r)

def integrate_weight_optimization(raw_weights: np.ndarray, meta_cfg: Dict[str, Any]) -> np.ndarray:
    """集成的权重优化接口"""
    return INSTITUTIONAL_INTEGRATION.enhanced_weight_optimization(raw_weights, meta_cfg)

def integrate_t10_validation(predictions: Union[pd.Series, np.ndarray],
                           feature_data: pd.DataFrame,
                           tickers: List[str]) -> T10ValidationResult:
    """集成的T+10验证接口"""
    return INSTITUTIONAL_INTEGRATION.validate_t10_predictions(predictions, feature_data, tickers)

def integrate_excel_validation(excel_path: str,
                             predictions_dict: Dict[str, float],
                             feature_data: pd.DataFrame,
                             tickers: List[str]) -> ExcelOutputValidation:
    """集成的Excel验证接口"""
    return INSTITUTIONAL_INTEGRATION.validate_excel_output(excel_path, predictions_dict, feature_data, tickers)