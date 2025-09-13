#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间安全验证器 - 防止数据泄漏和look-ahead bias
专为第二层EWA stacking系统设计
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

@dataclass
class TimeValidationResult:
    """时间验证结果"""
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

class TemporalSafetyValidator:
    """
    时间安全验证器

    核心功能：
    1. 验证OOF预测的时间一致性
    2. 检测look-ahead bias
    3. 确保预测-实际对齐
    4. 验证时间序列的单调性和完整性
    """

    def __init__(self,
                 tolerance_seconds: int = 86400,  # 1天容忍度
                 min_gap_days: int = 1,           # 最小时间间隔
                 strict_mode: bool = True,       # 严格模式
                 production_mode: bool = False,   # 生产模式 - 硬失败
                 allow_config_override: bool = True):  # 允许配置覆盖
        self.tolerance_seconds = tolerance_seconds
        self.min_gap_days = min_gap_days
        self.strict_mode = strict_mode
        self.production_mode = production_mode
        self.allow_config_override = allow_config_override

        # 从环境变量或配置文件读取生产模式设置
        if allow_config_override:
            import os
            self.production_mode = os.environ.get('BMA_PRODUCTION_MODE', '').lower() == 'true' or production_mode

        # 验证历史
        self.validation_history = []
        self.last_validation_time = None

        if self.production_mode:
            logger.warning("⚠️ PRODUCTION MODE ENABLED: Time isolation failures will be HARD FAILURES")

    def validate_oof_predictions(self,
                                oof_predictions: Dict[str, np.ndarray],
                                dates: pd.Series,
                                cv_split_info: Optional[Dict] = None) -> TimeValidationResult:
        """
        验证OOF预测的时间安全性

        Args:
            oof_predictions: 各模型的OOF预测
            dates: 对应的日期序列
            cv_split_info: CV分割信息(可选)

        Returns:
            时间验证结果
        """
        result = TimeValidationResult(
            is_valid=True,
            warnings=[],
            errors=[],
            metadata={}
        )

        try:
            # 1. 基础验证
            self._validate_basic_structure(oof_predictions, dates, result)

            # 2. 时间序列完整性验证
            self._validate_time_series_integrity(dates, result)

            # 3. OOF时间一致性验证
            self._validate_oof_temporal_consistency(oof_predictions, dates, result)

            # 4. Look-ahead bias检测
            self._detect_lookahead_bias(oof_predictions, dates, cv_split_info, result)

            # 5. 预测值合理性检查
            self._validate_prediction_reasonableness(oof_predictions, result)

            # 更新元数据
            result.metadata.update({
                'n_predictions': {k: len(v) for k, v in oof_predictions.items()},
                'date_range': (dates.min(), dates.max()),
                'n_dates': len(dates),
                'validation_timestamp': datetime.now()
            })

        except Exception as e:
            result.is_valid = False
            result.errors.append(f"验证过程异常: {e}")
            logger.error(f"OOF预测验证失败: {e}")

        # 记录验证历史
        self.validation_history.append(result)
        self.last_validation_time = datetime.now()

        # 生产模式下的硬失败处理
        if self.production_mode:
            if result.has_errors:
                error_msg = f"PRODUCTION MODE: Time isolation validation FAILED with {len(result.errors)} errors:\n"
                for err in result.errors:
                    error_msg += f"  - {err}\n"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if self.strict_mode and result.has_warnings:
                warning_msg = f"PRODUCTION MODE (strict): {len(result.warnings)} warnings treated as errors:\n"
                for warn in result.warnings:
                    warning_msg += f"  - {warn}\n"
                logger.error(warning_msg)
                raise ValueError(warning_msg)

        return result

    def _validate_basic_structure(self,
                                 oof_predictions: Dict[str, np.ndarray],
                                 dates: pd.Series,
                                 result: TimeValidationResult):
        """基础结构验证"""

        # 检查预测数据不为空
        if not oof_predictions:
            result.errors.append("OOF预测数据为空")
            result.is_valid = False
            return

        # 检查日期序列
        if dates is None or len(dates) == 0:
            result.errors.append("日期序列为空")
            result.is_valid = False
            return

        # 检查各模型预测长度一致性
        pred_lengths = [len(preds) for preds in oof_predictions.values()]
        if len(set(pred_lengths)) > 1:
            result.errors.append(f"各模型预测长度不一致: {pred_lengths}")
            result.is_valid = False

        # 检查预测长度与日期长度一致性
        expected_length = len(dates)
        for model_name, preds in oof_predictions.items():
            if len(preds) != expected_length:
                result.errors.append(
                    f"模型 {model_name} 预测长度 {len(preds)} 与日期长度 {expected_length} 不匹配"
                )
                result.is_valid = False

    def _validate_time_series_integrity(self,
                                      dates: pd.Series,
                                      result: TimeValidationResult):
        """时间序列完整性验证"""

        try:
            # 转换为datetime
            if not pd.api.types.is_datetime64_any_dtype(dates):
                dates_converted = pd.to_datetime(dates, errors='coerce')
                if dates_converted.isna().any():
                    result.errors.append("存在无法解析的日期值")
                    result.is_valid = False
                    return
                dates = dates_converted

            # 检查单调性
            if not dates.is_monotonic_increasing:
                result.warnings.append("日期序列不是单调递增的")
                if self.strict_mode:
                    result.errors.append("严格模式下要求日期序列单调递增")
                    result.is_valid = False

            # 检查重复日期
            duplicates = dates.duplicated().sum()
            if duplicates > 0:
                result.warnings.append(f"存在 {duplicates} 个重复日期")
                if self.strict_mode:
                    result.errors.append("严格模式下不允许重复日期")
                    result.is_valid = False

            # 检查时间间隔合理性
            if len(dates) > 1:
                time_diffs = dates.diff().dropna()
                min_diff = time_diffs.min()
                max_diff = time_diffs.max()

                if min_diff < pd.Timedelta(days=self.min_gap_days):
                    result.warnings.append(f"存在过小的时间间隔: {min_diff}")

                # 检查异常大的时间间隔
                median_diff = time_diffs.median()
                large_gaps = time_diffs[time_diffs > median_diff * 5]
                if len(large_gaps) > 0:
                    result.warnings.append(f"存在 {len(large_gaps)} 个异常大的时间间隔")

        except Exception as e:
            result.errors.append(f"时间序列完整性验证异常: {e}")
            result.is_valid = False

    def _validate_oof_temporal_consistency(self,
                                         oof_predictions: Dict[str, np.ndarray],
                                         dates: pd.Series,
                                         result: TimeValidationResult):
        """OOF时间一致性验证"""

        # 检查预测值的时间依赖性
        for model_name, predictions in oof_predictions.items():
            try:
                # 检查是否存在异常的预测值模式
                finite_preds = predictions[np.isfinite(predictions)]

                if len(finite_preds) == 0:
                    result.errors.append(f"模型 {model_name} 的所有预测值都不是有限值")
                    result.is_valid = False
                    continue

                # 检查预测值的变化幅度
                if len(finite_preds) > 1:
                    pred_changes = np.abs(np.diff(finite_preds))
                    median_change = np.median(pred_changes)
                    extreme_changes = pred_changes > median_change * 10

                    if np.any(extreme_changes):
                        n_extreme = np.sum(extreme_changes)
                        result.warnings.append(
                            f"模型 {model_name} 存在 {n_extreme} 个异常大的预测值变化"
                        )

                # 检查预测值的分布合理性
                pred_std = np.std(finite_preds)
                pred_mean = np.mean(finite_preds)

                # 检查是否所有预测值都相同（可能的错误）
                if pred_std < 1e-10:
                    result.warnings.append(f"模型 {model_name} 的预测值几乎没有变化（std={pred_std:.2e}）")

                # 检查极端预测值
                extreme_threshold = 5 * pred_std
                extreme_preds = np.abs(finite_preds - pred_mean) > extreme_threshold
                if np.any(extreme_preds):
                    n_extreme = np.sum(extreme_preds)
                    result.warnings.append(
                        f"模型 {model_name} 存在 {n_extreme} 个极端预测值"
                    )

            except Exception as e:
                result.warnings.append(f"模型 {model_name} 时间一致性验证异常: {e}")

    def _detect_lookahead_bias(self,
                              oof_predictions: Dict[str, np.ndarray],
                              dates: pd.Series,
                              cv_split_info: Optional[Dict],
                              result: TimeValidationResult):
        """检测look-ahead bias"""

        # 如果有CV分割信息，进行更严格的检查
        if cv_split_info is not None:
            try:
                self._validate_cv_temporal_splits(cv_split_info, dates, result)
            except Exception as e:
                result.warnings.append(f"CV时间分割验证异常: {e}")

        # 通用look-ahead bias检测
        try:
            # 检查预测值与未来日期的相关性
            for model_name, predictions in oof_predictions.items():
                if len(predictions) < 10:  # 数据太少，跳过检查
                    continue

                # 简单的未来信息泄漏检测：
                # 检查预测值是否与后续几天的预测值过度相关
                finite_mask = np.isfinite(predictions)
                if np.sum(finite_mask) < 5:
                    continue

                finite_preds = predictions[finite_mask]

                # 计算滞后相关性
                for lag in [1, 2, 3]:
                    if len(finite_preds) > lag:
                        lagged_corr = np.corrcoef(
                            finite_preds[:-lag],
                            finite_preds[lag:]
                        )[0, 1]

                        if not np.isnan(lagged_corr) and lagged_corr > 0.95:
                            result.warnings.append(
                                f"模型 {model_name} 存在疑似look-ahead bias: "
                                f"lag-{lag}相关性 = {lagged_corr:.4f}"
                            )

        except Exception as e:
            result.warnings.append(f"Look-ahead bias检测异常: {e}")

    def _validate_cv_temporal_splits(self,
                                   cv_split_info: Dict,
                                   dates: pd.Series,
                                   result: TimeValidationResult):
        """验证CV时间分割的正确性"""

        if 'splits' not in cv_split_info:
            result.warnings.append("CV分割信息不包含splits字段")
            return

        try:
            for fold_idx, (train_idx, val_idx) in enumerate(cv_split_info['splits']):
                # 检查训练集和验证集的时间顺序
                train_dates = dates.iloc[train_idx]
                val_dates = dates.iloc[val_idx]

                train_max = train_dates.max()
                val_min = val_dates.min()

                # 验证集的日期应该晚于训练集
                if val_min <= train_max:
                    result.errors.append(
                        f"Fold {fold_idx}: 验证集日期 {val_min} 不晚于训练集最大日期 {train_max}"
                    )
                    result.is_valid = False

                # 检查时间间隔是否合理
                time_gap = val_min - train_max
                if time_gap < pd.Timedelta(days=self.min_gap_days):
                    result.warnings.append(
                        f"Fold {fold_idx}: 时间间隔过小 {time_gap}"
                    )

        except Exception as e:
            result.warnings.append(f"CV时间分割验证异常: {e}")

    def _validate_prediction_reasonableness(self,
                                          oof_predictions: Dict[str, np.ndarray],
                                          result: TimeValidationResult):
        """验证预测值的合理性"""

        for model_name, predictions in oof_predictions.items():
            try:
                # 基础统计检查
                finite_preds = predictions[np.isfinite(predictions)]

                if len(finite_preds) == 0:
                    continue

                pred_min = np.min(finite_preds)
                pred_max = np.max(finite_preds)
                pred_std = np.std(finite_preds)
                pred_mean = np.mean(finite_preds)

                # 检查预测值范围是否合理（对于收益预测）
                if pred_min < -0.5 or pred_max > 0.5:
                    result.warnings.append(
                        f"模型 {model_name} 预测值范围异常: [{pred_min:.4f}, {pred_max:.4f}]"
                    )

                # 检查是否存在异常多的NaN值
                nan_ratio = np.mean(~np.isfinite(predictions))
                if nan_ratio > 0.1:  # 超过10%的NaN值
                    result.warnings.append(
                        f"模型 {model_name} 包含 {nan_ratio:.2%} 的NaN预测值"
                    )

                # 检查预测值的合理统计特征
                if pred_std < 1e-6:
                    result.warnings.append(
                        f"模型 {model_name} 预测值标准差过小: {pred_std:.2e}"
                    )

                if abs(pred_mean) > 0.1:  # 平均预测不应该偏离0太远
                    result.warnings.append(
                        f"模型 {model_name} 平均预测值异常: {pred_mean:.4f}"
                    )

            except Exception as e:
                result.warnings.append(f"模型 {model_name} 合理性检查异常: {e}")

    def set_production_mode(self, enabled: bool = True):
        """动态设置生产模式"""
        self.production_mode = enabled
        if enabled:
            logger.warning("⚠️ PRODUCTION MODE ENABLED: Time isolation failures will be HARD FAILURES")
        else:
            logger.info("Production mode disabled, validation will generate warnings only")

    def get_validation_summary(self) -> Dict[str, Any]:
        """获取验证历史摘要"""
        if not self.validation_history:
            return {"message": "No validation history"}

        total = len(self.validation_history)
        valid = sum(1 for r in self.validation_history if r.is_valid)
        with_errors = sum(1 for r in self.validation_history if r.has_errors)
        with_warnings = sum(1 for r in self.validation_history if r.has_warnings)

        return {
            "total_validations": total,
            "valid": valid,
            "with_errors": with_errors,
            "with_warnings": with_warnings,
            "success_rate": valid / total if total > 0 else 0,
            "production_mode": self.production_mode,
            "last_validation": self.last_validation_time.isoformat() if self.last_validation_time else None
        }

    def validate_prediction_timing(self,
                                  prediction_timestamp: pd.Timestamp,
                                  data_timestamp: pd.Timestamp,
                                  target_timestamp: pd.Timestamp) -> TimeValidationResult:
        """
        验证预测时间的合理性

        Args:
            prediction_timestamp: 预测生成时间
            data_timestamp: 数据时间戳
            target_timestamp: 目标时间戳

        Returns:
            验证结果
        """
        result = TimeValidationResult(
            is_valid=True,
            warnings=[],
            errors=[],
            metadata={}
        )

        try:
            # 检查时间顺序: data_timestamp <= prediction_timestamp < target_timestamp
            if data_timestamp > prediction_timestamp:
                result.errors.append(
                    f"数据时间 {data_timestamp} 晚于预测时间 {prediction_timestamp}"
                )
                result.is_valid = False

            if prediction_timestamp >= target_timestamp:
                result.errors.append(
                    f"预测时间 {prediction_timestamp} 不早于目标时间 {target_timestamp}"
                )
                result.is_valid = False

            # 检查时间间隔合理性
            prediction_gap = prediction_timestamp - data_timestamp
            forecast_horizon = target_timestamp - prediction_timestamp

            if prediction_gap > pd.Timedelta(days=7):  # 预测与数据间隔过大
                result.warnings.append(
                    f"预测时间与数据时间间隔过大: {prediction_gap}"
                )

            if forecast_horizon > pd.Timedelta(days=30):  # 预测期过长
                result.warnings.append(
                    f"预测期过长: {forecast_horizon}"
                )

            result.metadata.update({
                'prediction_gap': prediction_gap,
                'forecast_horizon': forecast_horizon,
                'validation_time': datetime.now()
            })

        except Exception as e:
            result.errors.append(f"预测时间验证异常: {e}")
            result.is_valid = False

        return result

    def get_validation_summary(self) -> Dict[str, Any]:
        """获取验证历史摘要"""

        if not self.validation_history:
            return {"message": "暂无验证历史"}

        recent_validations = self.validation_history[-10:]  # 最近10次

        summary = {
            'total_validations': len(self.validation_history),
            'recent_validations': len(recent_validations),
            'last_validation': self.last_validation_time,
            'success_rate': np.mean([v.is_valid for v in recent_validations]),
            'common_warnings': {},
            'common_errors': {}
        }

        # 统计常见警告和错误
        all_warnings = []
        all_errors = []

        for validation in recent_validations:
            all_warnings.extend(validation.warnings)
            all_errors.extend(validation.errors)

        from collections import Counter
        summary['common_warnings'] = dict(Counter(all_warnings).most_common(5))
        summary['common_errors'] = dict(Counter(all_errors).most_common(5))

        return summary


# 工具函数
def create_temporal_validator(strict_mode: bool = True) -> TemporalSafetyValidator:
    """创建时间安全验证器的便捷函数"""
    return TemporalSafetyValidator(
        tolerance_seconds=86400,  # 1天
        min_gap_days=1,
        strict_mode=strict_mode
    )

def validate_stacking_data_safety(
    oof_predictions: Dict[str, np.ndarray],
    dates: pd.Series,
    cv_info: Optional[Dict] = None,
    strict_mode: bool = True
) -> TimeValidationResult:
    """验证stacking数据时间安全性的便捷函数"""

    validator = create_temporal_validator(strict_mode)
    return validator.validate_oof_predictions(oof_predictions, dates, cv_info)


if __name__ == "__main__":
    # 测试时间安全验证器
    import numpy as np

    # 创建测试数据
    np.random.seed(42)
    n_samples = 100
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

    # 模拟OOF预测数据
    oof_preds = {
        'model_a': np.random.randn(n_samples) * 0.02,
        'model_b': np.random.randn(n_samples) * 0.015,
        'model_c': np.random.randn(n_samples) * 0.025
    }

    # 引入一些问题来测试验证器
    oof_preds['model_a'][50] = np.nan  # NaN值
    oof_preds['model_b'][-1] = 0.8     # 极端值

    print("=== 时间安全验证器测试 ===")

    # 验证
    result = validate_stacking_data_safety(oof_preds, dates, strict_mode=True)

    print(f"验证结果: {'通过' if result.is_valid else '失败'}")
    print(f"警告数量: {len(result.warnings)}")
    print(f"错误数量: {len(result.errors)}")

    if result.warnings:
        print("\n警告:")
        for warning in result.warnings[:5]:  # 显示前5个
            print(f"  - {warning}")

    if result.errors:
        print("\n错误:")
        for error in result.errors:
            print(f"  - {error}")

    print(f"\n元数据: {result.metadata}")

# 添加实用工具函数
def set_global_production_mode(enabled: bool = True):
    """设置全局生产模式"""
    global _global_validator
    if _global_validator is None:
        _global_validator = TemporalSafetyValidator(production_mode=enabled)
    else:
        _global_validator.set_production_mode(enabled)
    return _global_validator

def get_global_validator() -> TemporalSafetyValidator:
    """获取全局验证器实例"""
    global _global_validator
    if _global_validator is None:
        import os
        production_mode = os.environ.get('BMA_PRODUCTION_MODE', '').lower() == 'true'
        _global_validator = TemporalSafetyValidator(production_mode=production_mode)
    return _global_validator