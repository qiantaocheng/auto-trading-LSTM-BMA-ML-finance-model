#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化数据对齐器 - 确保第一层到第二层数据对齐成功
Simplified Data Aligner for robust first-to-second layer alignment

设计原则：
- 单一职责，功能明确
- 统一的索引处理标准
- 强化的数据验证
- 简化的错误处理
- 零容忍数据泄漏
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Union, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """数据验证错误"""
    pass

class IndexAlignmentError(Exception):
    """索引对齐错误"""
    pass

class SimplifiedDataAligner:
    """
    简化数据对齐器

    核心功能：
    1. 统一的MultiIndex标准化
    2. 强化的数据验证
    3. 简化的对齐逻辑
    4. 零fallback设计（快速失败）
    """

    def __init__(self, strict_mode: bool = True, allow_partial_align: bool = False):
        """
        初始化数据对齐器

        Args:
            strict_mode: 严格模式，不允许任何数据不一致
            allow_partial_align: 是否允许部分对齐（交集模式）
        """
        self.strict_mode = strict_mode
        self.allow_partial_align = allow_partial_align
        self.validation_stats = {}

        logger.info(f"SimplifiedDataAligner初始化: strict={strict_mode}, partial_align={allow_partial_align}")

    def _validate_multiindex(self, index: pd.Index, name: str) -> pd.MultiIndex:
        """
        验证并标准化MultiIndex

        Args:
            index: 待验证的索引
            name: 数据名称（用于错误报告）

        Returns:
            标准化的MultiIndex

        Raises:
            DataValidationError: 索引格式不正确
        """
        if not isinstance(index, pd.MultiIndex):
            raise DataValidationError(f"{name}: 必须是MultiIndex，当前类型: {type(index)}")

        if index.nlevels != 2:
            raise DataValidationError(f"{name}: MultiIndex必须是2层，当前层数: {index.nlevels}")

        expected_names = ['date', 'ticker']
        if list(index.names) != expected_names:
            logger.warning(f"{name}: 索引名称不标准 {index.names} -> {expected_names}")
            # 自动修正索引名称
            index = index.set_names(expected_names)

        # 验证日期格式
        date_level = index.get_level_values('date')
        if not pd.api.types.is_datetime64_any_dtype(date_level):
            try:
                date_level = pd.to_datetime(date_level)
                ticker_level = index.get_level_values('ticker')
                index = pd.MultiIndex.from_arrays([date_level, ticker_level], names=expected_names)
                logger.info(f"{name}: 日期列已转换为datetime类型")
            except Exception as e:
                raise DataValidationError(f"{name}: 日期格式转换失败: {e}")

        # 检查重复索引
        if index.duplicated().any():
            dup_count = index.duplicated().sum()
            if self.strict_mode:
                raise DataValidationError(f"{name}: 存在重复索引 {dup_count} 个")
            else:
                logger.warning(f"{name}: 存在重复索引 {dup_count} 个，将保留第一个")

        return index

    def _validate_data_quality(self, data: Union[pd.Series, pd.DataFrame], name: str) -> Dict[str, Any]:
        """
        验证数据质量

        Args:
            data: 待验证的数据
            name: 数据名称

        Returns:
            数据质量报告
        """
        stats = {
            'name': name,
            'type': type(data).__name__,
            'shape': getattr(data, 'shape', len(data)),
            'nan_count': 0,
            'inf_count': 0,
            'zero_count': 0,
            'coverage_rate': 0.0
        }

        if hasattr(data, 'isna'):
            stats['nan_count'] = data.isna().sum()
            if isinstance(data, pd.DataFrame):
                stats['nan_count'] = int(stats['nan_count'].sum())
            else:
                stats['nan_count'] = int(stats['nan_count'])

        if hasattr(data, 'values'):
            values = data.values
            if np.issubdtype(values.dtype, np.number):
                stats['inf_count'] = int(np.isinf(values).sum())
                stats['zero_count'] = int((values == 0).sum())

                total_elements = values.size
                valid_elements = total_elements - stats['nan_count'] - stats['inf_count']
                stats['coverage_rate'] = valid_elements / total_elements if total_elements > 0 else 0.0

        # 严格模式验证
        if self.strict_mode:
            if stats['coverage_rate'] < 0.8:
                raise DataValidationError(f"{name}: 数据覆盖率过低 {stats['coverage_rate']:.2%}")

            if stats['inf_count'] > 0:
                raise DataValidationError(f"{name}: 包含无穷值 {stats['inf_count']} 个")

        self.validation_stats[name] = stats
        logger.info(f"{name} 数据质量: 覆盖率={stats['coverage_rate']:.2%}, NaN={stats['nan_count']}, Inf={stats['inf_count']}")

        return stats

    def _standardize_prediction_columns(self, oof_predictions: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        标准化预测列名

        Args:
            oof_predictions: 原始预测字典

        Returns:
            标准化后的预测字典
        """
        # 标准列名映射
        column_mapping = {
            'elastic_net': 'pred_elastic',
            'elasticnet': 'pred_elastic',
            'elastic': 'pred_elastic',
            'xgboost': 'pred_xgb',
            'xgb': 'pred_xgb',
            'catboost': 'pred_catboost',
            'cat': 'pred_catboost',
            'lightgbm': 'pred_lgb',
            'lgb': 'pred_lgb'
        }

        standardized = {}
        for key, pred in oof_predictions.items():
            # 标准化键名
            std_key = column_mapping.get(key.lower(), key)

            # 确保是Series类型
            if not isinstance(pred, pd.Series):
                if hasattr(pred, 'values') and hasattr(pred, '__len__'):
                    # DataFrame的单列转Series
                    if isinstance(pred, pd.DataFrame) and pred.shape[1] == 1:
                        pred = pred.iloc[:, 0]
                    else:
                        pred = pd.Series(pred)
                else:
                    raise DataValidationError(f"预测 {key} 不是有效的Series类型: {type(pred)}")

            standardized[std_key] = pred

        logger.info(f"列名标准化完成: {list(oof_predictions.keys())} -> {list(standardized.keys())}")
        return standardized

    def align_first_to_second_layer(self,
                                   oof_predictions: Dict[str, pd.Series],
                                   target: pd.Series,
                                   target_column_name: str = 'ret_fwd_5d') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        第一层到第二层数据对齐（简化版）

        Args:
            oof_predictions: 第一层OOF预测字典
            target: 目标变量Series
            target_column_name: 目标变量列名

        Returns:
            Tuple[对齐后的DataFrame, 对齐报告]

        Raises:
            DataValidationError: 数据验证失败
            IndexAlignmentError: 索引对齐失败
        """
        logger.info("开始简化版第一层到第二层数据对齐")

        # Step 1: 输入验证
        if not oof_predictions:
            raise DataValidationError("OOF预测为空")

        if not isinstance(target, pd.Series):
            raise DataValidationError(f"目标变量必须是Series，当前类型: {type(target)}")

        # Step 2: 标准化预测列名
        standardized_preds = self._standardize_prediction_columns(oof_predictions)

        # Step 3: 获取基准索引（使用第一个预测的索引）
        first_pred_name = list(standardized_preds.keys())[0]
        first_pred = standardized_preds[first_pred_name]

        # 验证基准索引
        base_index = self._validate_multiindex(first_pred.index, first_pred_name)

        # Step 4: 验证所有预测的索引一致性
        aligned_preds = {}
        for name, pred in standardized_preds.items():
            # 验证索引
            pred_index = self._validate_multiindex(pred.index, name)

            # 检查索引是否一致
            if not pred_index.equals(base_index):
                if self.allow_partial_align:
                    # 使用交集对齐
                    common_index = base_index.intersection(pred_index)
                    if len(common_index) == 0:
                        raise IndexAlignmentError(f"预测 {name} 与基准索引无交集")

                    pred = pred.reindex(common_index)
                    logger.info(f"{name}: 使用交集对齐，样本数: {len(common_index)}")
                else:
                    # 严格模式：要求索引完全一致
                    raise IndexAlignmentError(f"预测 {name} 索引与基准不一致")

            # 验证数据质量
            self._validate_data_quality(pred, name)
            aligned_preds[name] = pred

        # Step 5: 对齐目标变量
        target_index = self._validate_multiindex(target.index, 'target')

        if self.allow_partial_align:
            # 找到所有预测和目标的交集
            all_indices = [pred.index for pred in aligned_preds.values()] + [target.index]
            final_index = all_indices[0]
            for idx in all_indices[1:]:
                final_index = final_index.intersection(idx)

            if len(final_index) == 0:
                raise IndexAlignmentError("预测和目标变量无公共索引")

            # 重新对齐所有数据到最终索引
            for name in aligned_preds:
                aligned_preds[name] = aligned_preds[name].reindex(final_index)
            target = target.reindex(final_index)

            logger.info(f"使用交集对齐，最终样本数: {len(final_index)}")
        else:
            # 严格模式：要求所有索引完全一致
            if not target_index.equals(base_index):
                raise IndexAlignmentError("目标变量索引与预测索引不一致")
            final_index = base_index

        # Step 6: 验证目标变量数据质量
        self._validate_data_quality(target, 'target')

        # Step 7: 构建最终DataFrame
        stacker_data = pd.DataFrame(index=final_index)

        # 添加预测列
        for name, pred in aligned_preds.items():
            stacker_data[name] = pred

        # 添加目标变量
        stacker_data[target_column_name] = target

        # Step 8: 最终验证
        if stacker_data.empty:
            raise DataValidationError("对齐后数据为空")

        if stacker_data.isna().all().any():
            na_cols = stacker_data.columns[stacker_data.isna().all()].tolist()
            raise DataValidationError(f"列 {na_cols} 全部为NaN")

        # 生成对齐报告
        alignment_report = {
            'success': True,
            'method': 'simplified_intersection' if self.allow_partial_align else 'strict_equality',
            'original_samples': len(base_index),
            'final_samples': len(final_index),
            'sample_retention_rate': len(final_index) / len(base_index),
            'predictions_aligned': list(aligned_preds.keys()),
            'target_column': target_column_name,
            'validation_stats': self.validation_stats.copy(),
            'index_names': final_index.names,
            'date_range': (final_index.get_level_values('date').min(),
                          final_index.get_level_values('date').max()),
            'unique_tickers': len(final_index.get_level_values('ticker').unique())
        }

        logger.info(f"数据对齐成功: {len(final_index)} 样本, 保留率: {alignment_report['sample_retention_rate']:.2%}")
        logger.info(f"预测列: {list(aligned_preds.keys())}")
        logger.info(f"目标列: {target_column_name}")

        return stacker_data, alignment_report

    def validate_ridge_input(self, stacker_data: pd.DataFrame) -> bool:
        """
        验证Ridge Stacker输入数据格式

        Args:
            stacker_data: 待验证的stacker数据

        Returns:
            验证是否通过

        Raises:
            DataValidationError: 验证失败
        """
        logger.info("验证Ridge Stacker输入数据格式")

        # 检查基本格式
        if not isinstance(stacker_data, pd.DataFrame):
            raise DataValidationError(f"输入必须是DataFrame，当前类型: {type(stacker_data)}")

        if not isinstance(stacker_data.index, pd.MultiIndex):
            raise DataValidationError("输入必须有MultiIndex")

        if stacker_data.index.names != ['date', 'ticker']:
            raise DataValidationError(f"索引名称必须是['date', 'ticker']，当前: {stacker_data.index.names}")

        # 检查必需的预测列
        required_pred_cols = ['pred_catboost', 'pred_elastic', 'pred_xgb']
        missing_cols = [col for col in required_pred_cols if col not in stacker_data.columns]
        if missing_cols:
            raise DataValidationError(f"缺少必需的预测列: {missing_cols}")

        # 检查目标变量列
        target_cols = [col for col in stacker_data.columns if col.startswith('ret_fwd')]
        if not target_cols:
            raise DataValidationError("缺少目标变量列（ret_fwd_*）")

        # 检查数据完整性
        pred_data = stacker_data[required_pred_cols]
        if pred_data.isna().all().any():
            na_cols = pred_data.columns[pred_data.isna().all()].tolist()
            raise DataValidationError(f"预测列 {na_cols} 全部为NaN")

        # 统计信息
        total_samples = len(stacker_data)
        valid_samples = len(stacker_data.dropna(subset=required_pred_cols))
        coverage_rate = valid_samples / total_samples if total_samples > 0 else 0.0

        logger.info(f"Ridge输入验证通过: {total_samples} 样本, 有效率: {coverage_rate:.2%}")

        return True

def create_simple_aligner(strict_mode: bool = True) -> SimplifiedDataAligner:
    """
    创建简化数据对齐器的便捷函数

    Args:
        strict_mode: 是否使用严格模式

    Returns:
        SimplifiedDataAligner实例
    """
    return SimplifiedDataAligner(
        strict_mode=strict_mode,
        allow_partial_align=not strict_mode  # 非严格模式允许部分对齐
    )

# 测试函数
def test_simplified_aligner():
    """测试简化数据对齐器"""
    import numpy as np
    from datetime import datetime, timedelta

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=10, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    # 创建MultiIndex
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])

    # 创建模拟OOF预测
    oof_predictions = {
        'elastic_net': pd.Series(np.random.normal(0, 0.02, len(index)), index=index),
        'xgboost': pd.Series(np.random.normal(0, 0.02, len(index)), index=index),
        'catboost': pd.Series(np.random.normal(0, 0.02, len(index)), index=index)
    }

    # 创建目标变量
    target = pd.Series(np.random.normal(0, 0.03, len(index)), index=index)

    # 测试对齐器
    aligner = create_simple_aligner(strict_mode=True)

    try:
        stacker_data, report = aligner.align_first_to_second_layer(oof_predictions, target)
        print("✅ 简化对齐器测试成功")
        print(f"对齐后数据形状: {stacker_data.shape}")
        print(f"列名: {list(stacker_data.columns)}")
        print(f"样本保留率: {report['sample_retention_rate']:.2%}")

        # 验证Ridge输入
        aligner.validate_ridge_input(stacker_data)
        print("✅ Ridge输入验证通过")

        return stacker_data, report

    except Exception as e:
        print(f"❌ 简化对齐器测试失败: {e}")
        return None, None

if __name__ == "__main__":
    test_simplified_aligner()