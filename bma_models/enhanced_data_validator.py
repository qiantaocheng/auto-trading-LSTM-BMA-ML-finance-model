#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化数据验证机制 - 确保数据质量和时间序列安全性
Enhanced Data Validator for quality and temporal safety

验证内容：
- 数据完整性和质量
- 时间序列一致性
- 前视偏误检测
- 数据泄漏预防
- 统计异常检测
- MultiIndex格式验证
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """数据验证错误"""
    pass

class TemporalSafetyError(Exception):
    """时间序列安全性错误"""
    pass

class DataLeakageError(Exception):
    """数据泄漏错误"""
    pass

class EnhancedDataValidator:
    """
    强化数据验证器

    功能：
    1. 全面数据质量检查
    2. 时间序列安全性验证
    3. 前视偏误检测
    4. 数据泄漏预防
    5. 统计异常检测
    """

    def __init__(self,
                 min_samples: int = 100,
                 min_coverage_rate: float = 0.8,
                 max_nan_rate: float = 0.2,
                 temporal_safety: bool = True,
                 leakage_detection: bool = True):
        """
        初始化数据验证器

        Args:
            min_samples: 最小样本数
            min_coverage_rate: 最小数据覆盖率
            max_nan_rate: 最大NaN比例
            temporal_safety: 是否启用时间序列安全性检查
            leakage_detection: 是否启用数据泄漏检测
        """
        self.min_samples = min_samples
        self.min_coverage_rate = min_coverage_rate
        self.max_nan_rate = max_nan_rate
        self.temporal_safety = temporal_safety
        self.leakage_detection = leakage_detection

        # 验证结果存储
        self.validation_results = {}
        self.warnings_list = []
        self.errors_list = []

        logger.info(f"Enhanced数据验证器初始化完成")
        logger.info(f"  最小样本数: {min_samples}")
        logger.info(f"  最小覆盖率: {min_coverage_rate:.1%}")
        logger.info(f"  最大NaN率: {max_nan_rate:.1%}")
        logger.info(f"  时间安全性: {temporal_safety}")
        logger.info(f"  泄漏检测: {leakage_detection}")

    def validate_multiindex_structure(self, data: Union[pd.DataFrame, pd.Series], name: str) -> Dict[str, Any]:
        """
        验证MultiIndex结构

        Args:
            data: 待验证数据
            name: 数据名称

        Returns:
            验证结果字典

        Raises:
            ValidationError: 结构验证失败
        """
        logger.info(f"验证MultiIndex结构: {name}")

        result = {
            'name': name,
            'type': type(data).__name__,
            'index_type': type(data.index).__name__,
            'is_multiindex': False,
            'levels': 0,
            'names': [],
            'duplicates': 0,
            'sorted': False,
            'valid': False
        }

        try:
            # 基本格式检查
            if not isinstance(data.index, pd.MultiIndex):
                raise ValidationError(f"{name}: 必须具有MultiIndex，当前: {type(data.index)}")

            result['is_multiindex'] = True
            result['levels'] = data.index.nlevels
            result['names'] = list(data.index.names)

            # 验证层数
            if data.index.nlevels != 2:
                raise ValidationError(f"{name}: MultiIndex必须是2层，当前: {data.index.nlevels}")

            # 验证层名称
            expected_names = ['date', 'ticker']
            if result['names'] != expected_names:
                raise ValidationError(f"{name}: 索引名称必须是{expected_names}，当前: {result['names']}")

            # 验证日期层
            date_level = data.index.get_level_values('date')
            if not pd.api.types.is_datetime64_any_dtype(date_level):
                raise ValidationError(f"{name}: date层必须是datetime类型，当前: {date_level.dtype}")

            # 检查重复
            result['duplicates'] = data.index.duplicated().sum()
            if result['duplicates'] > 0:
                if result['duplicates'] > len(data) * 0.01:  # 超过1%
                    raise ValidationError(f"{name}: 重复索引过多: {result['duplicates']} ({result['duplicates']/len(data):.1%})")
                else:
                    self.warnings_list.append(f"{name}: 存在少量重复索引: {result['duplicates']}")

            # 检查排序
            result['sorted'] = data.index.is_monotonic_increasing

            result['valid'] = True
            logger.info(f"{name} MultiIndex验证通过")

        except Exception as e:
            result['error'] = str(e)
            self.errors_list.append(f"{name} MultiIndex验证失败: {e}")
            if isinstance(e, ValidationError):
                raise
            else:
                raise ValidationError(f"{name} MultiIndex验证异常: {e}")

        self.validation_results[f"{name}_multiindex"] = result
        return result

    def validate_data_quality(self, data: Union[pd.DataFrame, pd.Series], name: str) -> Dict[str, Any]:
        """
        验证数据质量

        Args:
            data: 待验证数据
            name: 数据名称

        Returns:
            数据质量报告

        Raises:
            ValidationError: 数据质量不达标
        """
        logger.info(f"验证数据质量: {name}")

        result = {
            'name': name,
            'shape': getattr(data, 'shape', len(data)),
            'total_elements': 0,
            'valid_elements': 0,
            'nan_count': 0,
            'inf_count': 0,
            'zero_count': 0,
            'coverage_rate': 0.0,
            'nan_rate': 0.0,
            'inf_rate': 0.0,
            'zero_rate': 0.0,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'valid': False
        }

        try:
            # 基本统计
            if isinstance(data, pd.DataFrame):
                result['total_elements'] = data.size
                result['nan_count'] = int(data.isna().sum().sum())

                # 处理数值型数据
                numeric_data = data.select_dtypes(include=[np.number])
                if not numeric_data.empty:
                    values = numeric_data.values
                    result['inf_count'] = int(np.isinf(values).sum())
                    result['zero_count'] = int((values == 0).sum())

                    valid_mask = ~(np.isnan(values) | np.isinf(values))
                    if valid_mask.any():
                        valid_values = values[valid_mask]
                        result['mean'] = float(np.mean(valid_values))
                        result['std'] = float(np.std(valid_values))
                        result['min'] = float(np.min(valid_values))
                        result['max'] = float(np.max(valid_values))

            else:  # Series
                result['total_elements'] = len(data)
                result['nan_count'] = int(data.isna().sum())

                if pd.api.types.is_numeric_dtype(data):
                    values = data.values
                    result['inf_count'] = int(np.isinf(values).sum())
                    result['zero_count'] = int((values == 0).sum())

                    valid_mask = ~(np.isnan(values) | np.isinf(values))
                    if valid_mask.any():
                        valid_values = values[valid_mask]
                        result['mean'] = float(np.mean(valid_values))
                        result['std'] = float(np.std(valid_values))
                        result['min'] = float(np.min(valid_values))
                        result['max'] = float(np.max(valid_values))

            # 计算比例
            if result['total_elements'] > 0:
                result['nan_rate'] = result['nan_count'] / result['total_elements']
                result['inf_rate'] = result['inf_count'] / result['total_elements']
                result['zero_rate'] = result['zero_count'] / result['total_elements']
                result['valid_elements'] = result['total_elements'] - result['nan_count'] - result['inf_count']
                result['coverage_rate'] = result['valid_elements'] / result['total_elements']

            # 质量检查
            if result['total_elements'] < self.min_samples:
                raise ValidationError(f"{name}: 样本数不足 {result['total_elements']} < {self.min_samples}")

            if result['coverage_rate'] < self.min_coverage_rate:
                raise ValidationError(f"{name}: 数据覆盖率过低 {result['coverage_rate']:.2%} < {self.min_coverage_rate:.2%}")

            if result['nan_rate'] > self.max_nan_rate:
                raise ValidationError(f"{name}: NaN比例过高 {result['nan_rate']:.2%} > {self.max_nan_rate:.2%}")

            if result['inf_count'] > 0:
                raise ValidationError(f"{name}: 包含无穷值 {result['inf_count']} 个")

            # 警告检查
            if result['zero_rate'] > 0.5:
                self.warnings_list.append(f"{name}: 零值比例较高 {result['zero_rate']:.2%}")

            if result['std'] is not None and result['std'] < 1e-10:
                self.warnings_list.append(f"{name}: 标准差过小，可能缺乏变异性 {result['std']:.2e}")

            result['valid'] = True
            logger.info(f"{name} 数据质量验证通过: 覆盖率={result['coverage_rate']:.2%}")

        except Exception as e:
            result['error'] = str(e)
            self.errors_list.append(f"{name} 数据质量验证失败: {e}")
            if isinstance(e, ValidationError):
                raise
            else:
                raise ValidationError(f"{name} 数据质量验证异常: {e}")

        self.validation_results[f"{name}_quality"] = result
        return result

    def validate_temporal_safety(self, data: Union[pd.DataFrame, pd.Series], name: str) -> Dict[str, Any]:
        """
        验证时间序列安全性

        Args:
            data: 时间序列数据
            name: 数据名称

        Returns:
            时间安全性报告

        Raises:
            TemporalSafetyError: 时间安全性检查失败
        """
        if not self.temporal_safety:
            return {'skipped': True}

        logger.info(f"验证时间序列安全性: {name}")

        result = {
            'name': name,
            'date_range': None,
            'trading_days': 0,
            'gaps': [],
            'weekends_included': False,
            'future_dates': 0,
            'time_consistency': True,
            'monotonic': False,
            'valid': False
        }

        try:
            # 获取日期序列
            if isinstance(data.index, pd.MultiIndex) and 'date' in data.index.names:
                dates = data.index.get_level_values('date')
            elif pd.api.types.is_datetime64_any_dtype(data.index):
                dates = data.index
            else:
                self.warnings_list.append(f"{name}: 无法提取日期信息进行时间安全性检查")
                return result

            unique_dates = pd.Series(dates.unique()).sort_values()
            result['date_range'] = (unique_dates.min(), unique_dates.max())
            result['trading_days'] = len(unique_dates)

            # 检查未来日期
            today = pd.Timestamp.now().normalize()
            future_dates = unique_dates[unique_dates > today]
            result['future_dates'] = len(future_dates)

            if result['future_dates'] > 0:
                raise TemporalSafetyError(f"{name}: 包含未来日期 {result['future_dates']} 个")

            # 检查周末数据
            weekdays = unique_dates.dt.dayofweek
            weekend_count = ((weekdays == 5) | (weekdays == 6)).sum()
            result['weekends_included'] = weekend_count > 0

            if result['weekends_included']:
                self.warnings_list.append(f"{name}: 包含周末数据 {weekend_count} 天")

            # 检查时间间隔
            if len(unique_dates) > 1:
                gaps = unique_dates.diff().dropna()
                large_gaps = gaps[gaps > pd.Timedelta(days=7)]
                result['gaps'] = [gap.days for gap in large_gaps]

                if len(result['gaps']) > 0:
                    self.warnings_list.append(f"{name}: 存在大时间间隔: {result['gaps']} 天")

            # 检查单调性
            result['monotonic'] = unique_dates.is_monotonic_increasing

            if not result['monotonic']:
                self.warnings_list.append(f"{name}: 日期序列非单调递增")

            result['valid'] = True
            logger.info(f"{name} 时间安全性验证通过: {result['trading_days']} 交易日")

        except Exception as e:
            result['error'] = str(e)
            self.errors_list.append(f"{name} 时间安全性验证失败: {e}")
            if isinstance(e, TemporalSafetyError):
                raise
            else:
                raise TemporalSafetyError(f"{name} 时间安全性验证异常: {e}")

        self.validation_results[f"{name}_temporal"] = result
        return result

    def detect_data_leakage(self, features: pd.DataFrame, target: pd.Series, name: str = "dataset") -> Dict[str, Any]:
        """
        检测数据泄漏

        Args:
            features: 特征数据
            target: 目标变量
            name: 数据集名称

        Returns:
            泄漏检测报告

        Raises:
            DataLeakageError: 检测到数据泄漏
        """
        if not self.leakage_detection:
            return {'skipped': True}

        logger.info(f"检测数据泄漏: {name}")

        result = {
            'name': name,
            'perfect_correlations': [],
            'high_correlations': [],
            'suspicious_patterns': [],
            'feature_target_correlation': {},
            'valid': False
        }

        try:
            # 检查特征与目标的相关性
            if isinstance(features, pd.DataFrame):
                for col in features.columns:
                    if pd.api.types.is_numeric_dtype(features[col]) and pd.api.types.is_numeric_dtype(target):
                        # 使用共同的有效索引
                        common_idx = features[col].dropna().index.intersection(target.dropna().index)
                        if len(common_idx) > 10:
                            feat_vals = features.loc[common_idx, col]
                            targ_vals = target.loc[common_idx]

                            corr = np.corrcoef(feat_vals, targ_vals)[0, 1]
                            if not np.isnan(corr):
                                result['feature_target_correlation'][col] = corr

                                # 检查完美相关性（数据泄漏）
                                if abs(corr) > 0.99:
                                    result['perfect_correlations'].append((col, corr))

                                # 检查高相关性（可疑）
                                elif abs(corr) > 0.8:
                                    result['high_correlations'].append((col, corr))

            # 检查可疑模式
            for col in features.columns:
                # 检查是否包含"未来"、"forward"等关键词
                suspicious_keywords = ['future', 'forward', 'next', 'lead', 'fwd', 'ahead']
                if any(keyword in col.lower() for keyword in suspicious_keywords):
                    result['suspicious_patterns'].append(f"列名包含可疑关键词: {col}")

            # 错误报告
            if result['perfect_correlations']:
                raise DataLeakageError(f"{name}: 检测到完美相关性（数据泄漏）: {result['perfect_correlations']}")

            if len(result['high_correlations']) > len(features.columns) * 0.3:
                self.warnings_list.append(f"{name}: 高相关性特征过多: {len(result['high_correlations'])}")

            if result['suspicious_patterns']:
                self.warnings_list.append(f"{name}: 可疑特征名称: {result['suspicious_patterns']}")

            result['valid'] = True
            logger.info(f"{name} 数据泄漏检测通过")

        except Exception as e:
            result['error'] = str(e)
            self.errors_list.append(f"{name} 数据泄漏检测失败: {e}")
            if isinstance(e, DataLeakageError):
                raise
            else:
                raise DataLeakageError(f"{name} 数据泄漏检测异常: {e}")

        self.validation_results[f"{name}_leakage"] = result
        return result

    def validate_alignment_consistency(self, datasets: Dict[str, Union[pd.DataFrame, pd.Series]]) -> Dict[str, Any]:
        """
        验证多个数据集的对齐一致性

        Args:
            datasets: 数据集字典

        Returns:
            对齐一致性报告

        Raises:
            ValidationError: 对齐不一致
        """
        logger.info(f"验证对齐一致性: {list(datasets.keys())}")

        result = {
            'datasets': list(datasets.keys()),
            'index_consistency': True,
            'shape_consistency': True,
            'length_consistency': True,
            'index_intersection_rate': 0.0,
            'shapes': {},
            'lengths': {},
            'valid': False
        }

        try:
            # 收集形状和长度信息
            indices = []
            for name, data in datasets.items():
                result['shapes'][name] = getattr(data, 'shape', len(data))
                result['lengths'][name] = len(data)
                indices.append(data.index)

            # 检查长度一致性
            lengths = list(result['lengths'].values())
            if len(set(lengths)) > 1:
                result['length_consistency'] = False
                raise ValidationError(f"数据集长度不一致: {result['lengths']}")

            # 检查索引一致性
            if len(indices) > 1:
                first_index = indices[0]
                for i, idx in enumerate(indices[1:], 1):
                    if not idx.equals(first_index):
                        result['index_consistency'] = False
                        self.warnings_list.append(f"索引不一致: {list(datasets.keys())[0]} vs {list(datasets.keys())[i]}")

                # 计算索引交集率
                intersection = first_index
                for idx in indices[1:]:
                    intersection = intersection.intersection(idx)

                if len(first_index) > 0:
                    result['index_intersection_rate'] = len(intersection) / len(first_index)
                else:
                    result['index_intersection_rate'] = 0.0

                if result['index_intersection_rate'] < 0.95:
                    raise ValidationError(f"索引交集率过低: {result['index_intersection_rate']:.2%}")

            result['valid'] = True
            logger.info(f"对齐一致性验证通过: 交集率={result['index_intersection_rate']:.2%}")

        except Exception as e:
            result['error'] = str(e)
            self.errors_list.append(f"对齐一致性验证失败: {e}")
            if isinstance(e, ValidationError):
                raise
            else:
                raise ValidationError(f"对齐一致性验证异常: {e}")

        self.validation_results['alignment_consistency'] = result
        return result

    def comprehensive_validation(self,
                                oof_predictions: Dict[str, pd.Series],
                                target: pd.Series,
                                features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        综合验证

        Args:
            oof_predictions: OOF预测字典
            target: 目标变量
            features: 特征数据（可选）

        Returns:
            综合验证报告

        Raises:
            ValidationError: 验证失败
        """
        logger.info("开始综合数据验证")

        # 清空之前的结果
        self.validation_results.clear()
        self.warnings_list.clear()
        self.errors_list.clear()

        validation_summary = {
            'timestamp': datetime.now(),
            'validation_passed': False,
            'total_checks': 0,
            'passed_checks': 0,
            'failed_checks': 0,
            'warnings_count': 0,
            'errors_count': 0,
            'datasets_validated': list(oof_predictions.keys()) + ['target'],
            'validation_details': {}
        }

        try:
            # 1. 验证OOF预测
            for name, pred in oof_predictions.items():
                logger.info(f"验证OOF预测: {name}")

                # MultiIndex结构验证
                self.validate_multiindex_structure(pred, f"oof_{name}")
                validation_summary['total_checks'] += 1

                # 数据质量验证
                self.validate_data_quality(pred, f"oof_{name}")
                validation_summary['total_checks'] += 1

                # 时间安全性验证
                self.validate_temporal_safety(pred, f"oof_{name}")
                validation_summary['total_checks'] += 1

            # 2. 验证目标变量
            logger.info("验证目标变量")

            self.validate_multiindex_structure(target, "target")
            validation_summary['total_checks'] += 1

            self.validate_data_quality(target, "target")
            validation_summary['total_checks'] += 1

            self.validate_temporal_safety(target, "target")
            validation_summary['total_checks'] += 1

            # 3. 验证对齐一致性
            all_datasets = {**oof_predictions, 'target': target}
            self.validate_alignment_consistency(all_datasets)
            validation_summary['total_checks'] += 1

            # 4. 数据泄漏检测（如果提供特征）
            if features is not None:
                self.detect_data_leakage(features, target, "features_target")
                validation_summary['total_checks'] += 1

            # 统计结果
            passed_validations = sum(1 for result in self.validation_results.values()
                                   if result.get('valid', False))
            failed_validations = validation_summary['total_checks'] - passed_validations

            validation_summary.update({
                'validation_passed': failed_validations == 0,
                'passed_checks': passed_validations,
                'failed_checks': failed_validations,
                'warnings_count': len(self.warnings_list),
                'errors_count': len(self.errors_list),
                'validation_details': self.validation_results.copy(),
                'warnings': self.warnings_list.copy(),
                'errors': self.errors_list.copy()
            })

            # 记录结果
            logger.info(f"综合验证完成: {passed_validations}/{validation_summary['total_checks']} 通过")
            logger.info(f"警告: {len(self.warnings_list)} 个")
            logger.info(f"错误: {len(self.errors_list)} 个")

            if not validation_summary['validation_passed']:
                raise ValidationError(f"验证失败: {failed_validations} 个检查未通过")

        except Exception as e:
            validation_summary['validation_error'] = str(e)
            validation_summary['validation_passed'] = False
            logger.error(f"综合验证失败: {e}")
            raise

        return validation_summary

def create_enhanced_validator(**kwargs) -> EnhancedDataValidator:
    """
    创建强化数据验证器的便捷函数

    Args:
        **kwargs: 验证器参数

    Returns:
        EnhancedDataValidator实例
    """
    return EnhancedDataValidator(**kwargs)

# 测试函数
def test_enhanced_validator():
    """测试强化数据验证器"""
    import numpy as np

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])

    # 创建OOF预测
    oof_predictions = {
        'elastic_net': pd.Series(np.random.normal(0, 0.02, len(index)), index=index),
        'xgboost': pd.Series(np.random.normal(0, 0.02, len(index)), index=index),
        'catboost': pd.Series(np.random.normal(0, 0.02, len(index)), index=index)
    }

    # 创建目标变量
    target = pd.Series(np.random.normal(0, 0.03, len(index)), index=index)

    # 测试验证器
    validator = create_enhanced_validator(
        min_samples=200,
        min_coverage_rate=0.9,
        temporal_safety=True,
        leakage_detection=True
    )

    try:
        validation_report = validator.comprehensive_validation(oof_predictions, target)
        print("✅ 强化数据验证器测试成功")
        print(f"验证通过: {validation_report['validation_passed']}")
        print(f"检查项: {validation_report['passed_checks']}/{validation_report['total_checks']}")
        print(f"警告数: {validation_report['warnings_count']}")
        print(f"错误数: {validation_report['errors_count']}")

        return validation_report

    except Exception as e:
        print(f"❌ 强化数据验证器测试失败: {e}")
        return None

if __name__ == "__main__":
    test_enhanced_validator()