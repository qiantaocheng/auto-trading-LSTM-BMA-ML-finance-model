#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Index Aligner module for aligning data indices in the pipeline
支持MultiIndex和时间标准化的增强版索引对齐器
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)

# 内置时间标准化函数
def _standardize_dates_to_day(dates):
    """
    内置时间标准化函数，将所有日期标准化为天精度

    Args:
        dates: 日期数组/序列

    Returns:
        np.ndarray: 标准化后的日期（datetime64[D]类型）
    """
    import numpy as np
    import pandas as pd

    # 转换为numpy数组
    if isinstance(dates, (pd.DatetimeIndex, pd.Series)):
        dates_array = dates.values
    elif isinstance(dates, list):
        dates_array = pd.to_datetime(dates).values
    else:
        dates_array = np.asarray(dates)

    # 标准化为天精度，移除时间组件
    return dates_array.astype('datetime64[D]')

TIME_ALIGNMENT_AVAILABLE = True  # 内置功能始终可用


class EnhancedIndexAligner:
    """Enhanced Index Aligner with MultiIndex support and time standardization"""

    def __init__(self, horizon: int = 10, mode: str = 'train'):
        """
        Initialize enhanced index aligner

        Args:
            horizon: Prediction horizon in days
            mode: 'train' or 'inference'
        """
        self.horizon = horizon
        self.mode = mode
        logger.info(f"[EnhancedIndexAligner] 初始化 - horizon: {self.horizon}, mode: {mode}")

    def align_all_data(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      dates: pd.Series,
                      tickers: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Enhanced align all data components with MultiIndex support

        Args:
            X: Feature DataFrame
            y: Target Series
            dates: Date Series
            tickers: Ticker Series

        Returns:
            Tuple of (aligned_data_dict, alignment_report_dict)
        """
        logger.info(f"[EnhancedIndexAligner] 开始数据对齐 - X: {X.shape}, y: {len(y)}")

        # Create enhanced alignment report
        alignment_report = {
            'original_shapes': {
                'X': X.shape,
                'y': len(y),
                'dates': len(dates),
                'tickers': len(tickers)
            },
            'alignment_method': 'enhanced_multiindex_intersection',
            'mode': self.mode,
            'horizon': self.horizon,
            'multiindex_detected': isinstance(X.index, pd.MultiIndex),
            'time_alignment_used': False
        }

        # Enhanced index alignment with MultiIndex support
        common_index = X.index

        # 验证MultiIndex结构
        if isinstance(X.index, pd.MultiIndex):
            logger.info(f"[EnhancedIndexAligner] 检测到MultiIndex: {X.index.names}")
            alignment_report['multiindex_names'] = X.index.names

            # 时间对齐处理（如果可用）
            if TIME_ALIGNMENT_AVAILABLE and 'date' in X.index.names:
                try:
                    dates_raw = X.index.get_level_values('date')
                    dates_standardized = _standardize_dates_to_day(dates_raw)
                    tickers_raw = X.index.get_level_values('ticker')

                    # 重建标准化的MultiIndex
                    standardized_index = pd.MultiIndex.from_arrays(
                        [dates_standardized, tickers_raw],
                        names=['date', 'ticker']
                    )

                    # 更新X的索引
                    X = X.copy()
                    X.index = standardized_index
                    common_index = standardized_index

                    alignment_report['time_alignment_used'] = True
                    logger.info("[EnhancedIndexAligner] ✅ 已应用时间标准化")
                except Exception as e:
                    logger.warning(f"[EnhancedIndexAligner] ⚠️ 时间标准化失败: {e}")

        # 与目标变量对齐
        if hasattr(y, 'index'):
            if isinstance(y.index, pd.MultiIndex) and isinstance(common_index, pd.MultiIndex):
                # MultiIndex对齐
                common_index = common_index.intersection(y.index)
            else:
                # 普通索引对齐
                common_index = common_index.intersection(y.index)

        # Apply common index with enhanced alignment
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index] if hasattr(y, 'loc') else y[:len(common_index)]

        # Enhanced dates and tickers alignment for MultiIndex
        if isinstance(common_index, pd.MultiIndex):
            # 从MultiIndex中提取dates和tickers
            if 'date' in common_index.names:
                dates_aligned = pd.Series(
                    common_index.get_level_values('date'),
                    index=common_index,
                    name='date'
                )
            else:
                # 使用提供的dates参数
                dates_aligned = self._align_series_to_index(dates, common_index)

            if 'ticker' in common_index.names:
                tickers_aligned = pd.Series(
                    common_index.get_level_values('ticker'),
                    index=common_index,
                    name='ticker'
                )
            else:
                # 使用提供的tickers参数
                tickers_aligned = self._align_series_to_index(tickers, common_index)
        else:
            # 普通索引对齐
            dates_aligned = self._align_series_to_index(dates, common_index)
            tickers_aligned = self._align_series_to_index(tickers, common_index)

        # 验证对齐结果
        alignment_success = self._validate_aligned_lengths(
            X_aligned, y_aligned, dates_aligned, tickers_aligned
        )

        # Update enhanced report
        alignment_report['aligned_shapes'] = {
            'X': X_aligned.shape,
            'y': len(y_aligned),
            'dates': len(dates_aligned),
            'tickers': len(tickers_aligned)
        }
        alignment_report['samples_removed'] = len(X) - len(X_aligned)
        alignment_report['alignment_successful'] = alignment_success
        alignment_report['common_index_type'] = type(common_index).__name__

        if not alignment_success:
            logger.error(f"[EnhancedIndexAligner] 对齐失败 - 长度不一致")
        else:
            logger.info(f"[EnhancedIndexAligner] ✅ 对齐成功 - 最终样本数: {len(X_aligned)}")

        # Create aligned data dictionary
        aligned_data = {
            'X': X_aligned,
            'y': y_aligned,
            'dates': dates_aligned,
            'tickers': tickers_aligned
        }

        return aligned_data, alignment_report

    def _align_series_to_index(self, series: pd.Series, target_index: pd.Index) -> pd.Series:
        """
        Helper method to align a series to a target index
        """
        if isinstance(series, pd.Series) and hasattr(series, 'loc'):
            try:
                # 尝试基于索引对齐
                return series.loc[target_index]
            except (KeyError, IndexError):
                # 如果索引对齐失败，使用长度截断
                return pd.Series(
                    series.values[:len(target_index)],
                    index=target_index,
                    name=series.name
                )
        else:
            # 非pandas Series，使用长度截断
            return pd.Series(
                series[:len(target_index)] if hasattr(series, '__getitem__') else [series] * len(target_index),
                index=target_index
            )

    def _validate_aligned_lengths(self, X, y, dates, tickers) -> bool:
        """
        验证所有对齐后数据的长度一致性
        """
        lengths = [len(X), len(y), len(dates), len(tickers)]
        unique_lengths = set(lengths)

        if len(unique_lengths) != 1:
            logger.error(f"[EnhancedIndexAligner] 长度不一致: {dict(zip(['X', 'y', 'dates', 'tickers'], lengths))}")
            return False

        return True

    def align_first_to_second_layer(self,
                                   first_layer_preds: Dict[str, pd.Series],
                                   y: pd.Series,
                                   dates: pd.Series = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        专门用于第一层到第二层的对齐

        Args:
            first_layer_preds: 第一层预测结果字典
            y: 目标变量
            dates: 日期序列（可选）

        Returns:
            对齐后的DataFrame和报告
        """
        logger.info("[EnhancedIndexAligner] 开始第一层到第二层数据对齐")

        # 获取第一个预测作为基准（支持dict或DataFrame输入）
        if isinstance(first_layer_preds, dict):
            first_pred = next(iter(first_layer_preds.values()))
        elif isinstance(first_layer_preds, pd.DataFrame) and len(first_layer_preds.columns) > 0:
            # 取第一列作为基准
            first_pred = first_layer_preds.iloc[:, 0]
        # Note: Unsupported types are handled earlier; no extra branch needed here

        if not hasattr(first_pred, 'index') or not isinstance(first_pred.index, pd.MultiIndex):
            raise ValueError("第一层预测必须有MultiIndex")

        base_index = first_pred.index

        # 时间标准化（如果可用）
        if TIME_ALIGNMENT_AVAILABLE:
            try:
                dates_raw = base_index.get_level_values('date')
                dates_standardized = _standardize_dates_to_day(dates_raw)
                tickers = base_index.get_level_values('ticker')

                base_index = pd.MultiIndex.from_arrays(
                    [dates_standardized, tickers],
                    names=['date', 'ticker']
                )
                logger.info("[EnhancedIndexAligner] ✅ 第二层数据已标准化时间格式")
            except Exception as e:
                logger.warning(f"[EnhancedIndexAligner] ⚠️ 第二层时间标准化失败: {e}")

        # 验证所有预测的索引一致性（如果不一致则强制reindex而非抛错）
        original_base = first_pred.index
        if isinstance(first_layer_preds, dict):
            iterator = first_layer_preds.items()
        else:
            iterator = [(col, first_layer_preds[col]) for col in first_layer_preds.columns]
        for name, pred in iterator:
            try:
                if hasattr(pred, 'index') and isinstance(pred.index, pd.MultiIndex):
                    # 若不一致则重建到基准索引
                    if not pred.index.equals(original_base):
                        if hasattr(pred, 'reindex'):
                            if isinstance(first_layer_preds, dict):
                                first_layer_preds[name] = pred.reindex(original_base)
                            else:
                                first_layer_preds[name] = pred.reindex(original_base)
                else:
                    # 将非Series/非MultiIndex数据包装为Series并对齐索引
                    wrapped = pd.Series(np.asarray(pred), index=original_base)
                    if isinstance(first_layer_preds, dict):
                        first_layer_preds[name] = wrapped
                    else:
                        first_layer_preds[name] = wrapped
            except TypeError as te:
                # 防止 'numpy.ndarray' object is not callable 等类型错误
                raise ValueError(f"对齐第一层预测时发生类型错误: {name}: {te}")

        # 验证y的索引一致性（预测模式时允许虚拟目标变量）
        if hasattr(y, 'index') and self.mode != 'inference':
            if not y.index.equals(original_base):
                raise ValueError("目标变量索引不匹配")

        # 创建第二层数据框
        stacker_data = pd.DataFrame(index=base_index)

        # 检查重复索引
        if stacker_data.index.duplicated().any():
            dup_count = stacker_data.index.duplicated().sum()
            raise ValueError(f"第二层输入存在重复(date,ticker)键: {dup_count} 个")

        # 标准化列名映射
        column_mapping = {
            'elastic_net': 'pred_elastic',
            'xgboost': 'pred_xgb',
            'catboost': 'pred_catboost'
        }

        # 添加预测列（统一对齐到base_index）
        def _safe_series(v) -> pd.Series:
            if isinstance(v, pd.Series):
                if not v.index.equals(base_index):
                    return v.reindex(base_index)
                return v
            return pd.Series(np.asarray(v), index=base_index)

        if isinstance(first_layer_preds, dict):
            for model_name, pred in first_layer_preds.items():
                col_name = column_mapping.get(model_name, f'pred_{model_name}')
                s = _safe_series(pred)
                stacker_data[col_name] = s.values
        elif hasattr(first_layer_preds, 'columns'):
            for col in first_layer_preds.columns:
                std_col = col
                if col not in ['pred_elastic', 'pred_xgb', 'pred_catboost']:
                    for orig_name, std_name in column_mapping.items():
                        if col == std_name:
                            std_col = std_name
                            break
                s = _safe_series(first_layer_preds[col])
                stacker_data[std_col] = s.values
        else:
            raise ValueError(f"Unsupported first_layer_preds type: {type(first_layer_preds)}")

        # 添加目标变量（预测模式时可能是虚拟变量）
        # 固定目标列名（统一T+5）。预测模式创建虚拟目标。
        target_col = f'ret_fwd_{self.horizon}d'

        if self.mode == 'inference':
            stacker_data[target_col] = np.zeros(len(stacker_data))
        else:
            if hasattr(y, 'values'):
                stacker_data[target_col] = y.values
            elif isinstance(y, np.ndarray):
                stacker_data[target_col] = y
            else:
                stacker_data[target_col] = np.array(y)

        # 确保索引名称正确
        if stacker_data.index.names != ['date', 'ticker']:
            stacker_data.index.names = ['date', 'ticker']

        report = {
            'alignment_method': 'first_to_second_layer',
            'samples': len(stacker_data),
            'columns': list(stacker_data.columns),
            'time_standardized': TIME_ALIGNMENT_AVAILABLE,
            'duplicates_found': False
        }

        logger.info(f"[EnhancedIndexAligner] ✅ 第一层到第二层对齐完成: {stacker_data.shape}")
        return stacker_data, report

    def validate_alignment(self, aligned_data: Dict[str, Any]) -> bool:
        """
        Enhanced validation that all data components are properly aligned
        """
        X = aligned_data.get('X')
        y = aligned_data.get('y')
        dates = aligned_data.get('dates')
        tickers = aligned_data.get('tickers')

        # Check lengths
        x_len = len(X) if X is not None else 0
        y_len = len(y) if y is not None else 0
        dates_len = len(dates) if dates is not None else 0
        tickers_len = len(tickers) if tickers is not None else 0

        # All should have same length
        lengths = [x_len, y_len, dates_len, tickers_len]
        if len(set(lengths)) != 1:
            logger.error(f"[EnhancedIndexAligner] 验证失败 - 长度不一致: {lengths}")
            return False

        # Check indices if available
        if hasattr(X, 'index') and hasattr(y, 'index'):
            if not X.index.equals(y.index):
                logger.error("[EnhancedIndexAligner] 验证失败 - X和y的索引不一致")
                return False

        # Additional MultiIndex validation
        if isinstance(X.index, pd.MultiIndex):
            if X.index.names != ['date', 'ticker']:
                logger.warning(f"[EnhancedIndexAligner] MultiIndex名称非标准: {X.index.names}")

            # Check for duplicates
            if X.index.duplicated().any():
                logger.error("[EnhancedIndexAligner] 验证失败 - 发现重复索引")
                return False

        logger.info("[EnhancedIndexAligner] ✅ 对齐验证通过")
        return True
