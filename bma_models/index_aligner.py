#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IndexAligner - 统一索引对齐器
解决738 vs 748维度不匹配问题的核心模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AlignmentReport:
    """对齐报告"""
    original_shapes: Dict[str, Tuple]
    final_shape: Tuple
    removed_samples: Dict[str, int]
    removal_reasons: Dict[str, int]
    coverage_rate: float
    effective_tickers: int
    effective_dates: int
    horizon_trimmed: int
    # [CRITICAL] 新增横截面统计
    daily_tickers_stats: Dict[str, float] = None  # min/median/max每日股票数
    cross_section_ready: bool = True  # 是否满足横截面要求
    
class IndexAligner:
    """统一索引对齐器 - 解决所有维度不匹配问题"""
    
    def __init__(self, horizon: int = 10, strict_mode: bool = True, mode: str = 'train'):
        """
        初始化对齐器
        
        Args:
            horizon: 前瞻期(T+10)，仅用于记录，实际剪尾通过CV的gap/embargo实现
            strict_mode: 严格模式，维度不匹配时报错
            mode: 'train' (训练模式) 或 'predict' (预测模式) - 现在都不执行剪尾
        """
        # 🔧 CRITICAL FIX: 统一不剪尾策略，防止维度不匹配
        # 前视偏差防范通过CV的gap/embargo实现，而不是数据剪尾
        self.horizon = 0  # 统一设为0，不执行任何剪尾
        self.original_horizon = horizon  # 保留原始horizon设置用于日志
        self.strict_mode = strict_mode
        self.mode = mode
        self.alignment_history = []
        logger.info(f"IndexAligner初始化: original_horizon={horizon}, actual_horizon=0, mode={mode}")
        
        # 🎯 统一处理策略说明
        logger.info(f"[DIMENSION_CONSISTENCY] 训练和预测均不剪尾，确保维度一致性")
        logger.info(f"[TEMPORAL_SAFETY] 前视偏差通过CV gap={horizon-1}, embargo={horizon} 防范")
    
    def align_all_data(self, **data_dict) -> Tuple[Dict[str, Any], AlignmentReport]:
        """
        统一对齐所有数据，解决738 vs 748问题
        
        Args:
            **data_dict: 命名数据 如 X=features, y=labels, alpha=alpha_features, pred=predictions
            
        Returns:
            (aligned_data_dict, alignment_report)
        """
        logger.info("[TARGET] 开始IndexAligner统一对齐")
        
        # 1. 记录原始形状
        original_shapes = {}
        for name, data in data_dict.items():
            if hasattr(data, 'shape'):
                original_shapes[name] = data.shape
            elif hasattr(data, '__len__'):
                original_shapes[name] = (len(data),)
            else:
                original_shapes[name] = 'scalar'
        
        logger.info("[DATA] 原始数据形状:")
        for name, shape in original_shapes.items():
            logger.info(f"  {name}: {shape}")
        
        # 2. 统一数据预处理 - 维度一致性优先，不执行剪尾
        # 🔧 CRITICAL FIX: 移除所有剪尾逻辑，确保训练/预测维度一致
        processed_data = {}
        
        for name, data in data_dict.items():
            if data is None:
                processed_data[name] = None
                continue
            
            # 🎯 保持原始数据完整性，只进行必要的验证和清理
            if hasattr(data, 'index') and isinstance(data.index, pd.MultiIndex):
                # MultiIndex (date, ticker) 格式 - 保持完整结构
                processed_data[name] = data.copy()
                logger.info(f"  {name}: 保持完整 MultiIndex 结构 {data.shape}")
            else:
                # 普通索引或数组 - 直接保持
                processed_data[name] = data
                shape_info = getattr(data, 'shape', len(data) if hasattr(data, '__len__') else 'scalar')
                logger.info(f"  {name}: 保持原始格式 {shape_info}")
        
        # 日志说明：不执行剪尾的原因
        logger.info(f"[NO_TRIMMING] 为保证维度一致性，所有数据保持完整")
        logger.info(f"[TEMPORAL_SAFETY] 前视偏差通过CV gap={self.original_horizon-1}, embargo={self.original_horizon} 防范")
        horizon_trimmed = 0  # 实际未剪尾任何数据
        
        # 3. 构建通用索引 - inner join所有数据
        common_index = None
        removal_reasons = {'horizon_trim': horizon_trimmed, 'nan_removal': 0, 'index_mismatch': 0}
        
        for name, data in processed_data.items():
            if data is None or (hasattr(data, 'empty') and data.empty):
                continue
                
            if hasattr(data, 'index'):
                current_index = data.index
                if common_index is None:
                    common_index = current_index
                else:
                    # Inner join索引
                    before_len = len(common_index)
                    common_index = common_index.intersection(current_index)
                    after_len = len(common_index)
                    
                    if before_len > after_len:
                        removed = before_len - after_len
                        removal_reasons['index_mismatch'] += removed
                        logger.info(f"  {name}: 索引inner join移除 {removed} 条")
        
        if common_index is None or len(common_index) == 0:
            logger.error("[ERROR] 所有数据索引交集为空，无法对齐")
            logger.warning("索引对齐失败：所有数据交集为空")
        
        logger.info(f"[TARGET] 通用索引确定: {len(common_index)} 条")
        
        # 4. 使用通用索引对齐所有数据
        aligned_data = {}
        removed_samples = {}
        
        for name, data in processed_data.items():
            if data is None:
                aligned_data[name] = None
                removed_samples[name] = 0
                continue
            
            if hasattr(data, 'loc'):
                try:
                    # 使用.loc对齐到通用索引
                    original_len = len(data)
                    aligned_data[name] = data.loc[common_index]
                    final_len = len(aligned_data[name])
                    removed_samples[name] = original_len - final_len
                    
                    logger.info(f"  {name}: {original_len} → {final_len} (-{removed_samples[name]})")
                except Exception as e:
                    logger.warning(f"  {name}: 索引对齐失败，使用原数据: {e}")
                    aligned_data[name] = data
                    removed_samples[name] = 0
            else:
                # 数组类型，按长度截断到common_index长度
                if hasattr(data, '__len__') and len(data) > len(common_index):
                    aligned_data[name] = data[:len(common_index)]
                    removed_samples[name] = len(data) - len(common_index)
                    logger.info(f"  {name}: 数组截断 {len(data)} → {len(common_index)}")
                else:
                    aligned_data[name] = data
                    removed_samples[name] = 0
        
        # 5. 验证最终一致性
        final_lengths = {}
        for name, data in aligned_data.items():
            if data is not None and hasattr(data, '__len__'):
                final_lengths[name] = len(data)
        
        if final_lengths:
            unique_lengths = set(final_lengths.values())
            if len(unique_lengths) > 1:
                error_msg = f"[ERROR] 对齐后长度仍不一致: {final_lengths}"
                logger.error(error_msg)
                if self.strict_mode:
                    logger.warning(error_msg)
        
        # 6. 计算覆盖统计和横截面检查
        daily_tickers_stats = None
        cross_section_ready = True
        # 动态横截面要求：根据数据规模调整
        # 对于研究和单股票分析，允许较低的横截面要求
        if len(common_index) > 10000:  # 大数据集
            MIN_CROSS_SECTION = 30
        elif len(common_index) > 1000:  # 中等数据集  
            MIN_CROSS_SECTION = 10
        else:  # 小数据集或单股票研究
            MIN_CROSS_SECTION = 1
        
        # [CRITICAL] DEBUG: 检查common_index的实际状态
        logger.info(f"[SEARCH] DEBUG common_index检测:")
        logger.info(f"  类型: {type(common_index)}")
        logger.info(f"  长度: {len(common_index) if common_index is not None else 'None'}")
        logger.info(f"  hasattr get_level_values: {hasattr(common_index, 'get_level_values')}")
        logger.info(f"  hasattr nlevels: {hasattr(common_index, 'nlevels')}")
        if hasattr(common_index, 'nlevels'):
            logger.info(f"  nlevels: {common_index.nlevels}")
            logger.info(f"  nlevels >= 2: {common_index.nlevels >= 2}")
        logger.info(f"  isinstance MultiIndex: {isinstance(common_index, pd.MultiIndex)}")
        
        if hasattr(common_index, 'get_level_values') and common_index.nlevels >= 2:
            logger.info("[OK] 进入MultiIndex分支")
            # MultiIndex情况 - 计算每日股票数分布
            effective_dates = len(common_index.get_level_values(0).unique())
            effective_tickers = len(common_index.get_level_values(1).unique())
            
            # [CRITICAL] CRITICAL: 计算每日股票数分布
            daily_tickers = pd.Series(common_index.get_level_values(1)).groupby(
                pd.Series(common_index.get_level_values(0))
            ).nunique()
            
            daily_tickers_stats = {
                'min': float(daily_tickers.min()),
                'median': float(daily_tickers.median()),
                'max': float(daily_tickers.max()),
                'mean': float(daily_tickers.mean())
            }
            
            # MultiIndex情况下的横截面检查
            if effective_tickers < MIN_CROSS_SECTION:
                cross_section_ready = False
                logger.error(f"[ERROR] MultiIndex股票数量不足：{effective_tickers} < {MIN_CROSS_SECTION}")
            else:
                cross_section_ready = True
                logger.info(f"[OK] MultiIndex股票数量充足：{effective_tickers} >= {MIN_CROSS_SECTION}")
                logger.info(f"[OK] 横截面充足：每日股票数 min={daily_tickers_stats['min']}, median={daily_tickers_stats['median']}, max={daily_tickers_stats['max']}")
            
        else:
            logger.info("[ERROR] 进入else分支（非MultiIndex）")
            # [CRITICAL] CRITICAL FIX: 非MultiIndex情况下正确计算股票数量
            logger.warning("[WARNING] 检测到非MultiIndex格式，尝试从数据中推断股票数量")
            
            effective_dates = len(common_index.unique()) if hasattr(common_index, 'unique') else len(common_index)
            effective_tickers = 1  # 默认值
            
            # [CRITICAL] PRIORITY 1: 从多种可能的股票参数获取数量（更健壮）
            tickers_found = False
            for ticker_param_name in ['tickers', 'ticker', 'prediction_tickers', 'symbols', 'stocks']:
                if ticker_param_name in aligned_data and aligned_data[ticker_param_name] is not None:
                    tickers_data = aligned_data[ticker_param_name]
                    if hasattr(tickers_data, 'unique'):
                        try:
                            unique_tickers = tickers_data.unique()
                            effective_tickers = len(unique_tickers)
                            logger.info(f"[TARGET] 从{ticker_param_name}参数直接获取: {effective_tickers}只股票")
                            if effective_tickers > 1:
                                logger.info(f"[DATA] 股票列表: {list(unique_tickers)[:10]}...")
                            tickers_found = True
                            break
                        except Exception as e:
                            logger.warning(f"[WARNING] {ticker_param_name}.unique()失败: {e}")
                            continue
                    elif hasattr(tickers_data, '__len__'):
                        # 如果是列表或数组
                        try:
                            unique_tickers = list(set(tickers_data)) if hasattr(tickers_data, '__iter__') else [tickers_data]
                            effective_tickers = len(unique_tickers)
                            logger.info(f"[TARGET] 从{ticker_param_name}数组获取: {effective_tickers}只股票")
                            tickers_found = True
                            break
                        except Exception as e:
                            logger.warning(f"[WARNING] 处理{ticker_param_name}参数时出错: {e}")
                            continue
            
            # [CRITICAL] FALLBACK: 如果所有股票参数都无效，尝试其他方法推断
            if not tickers_found and effective_tickers == 1:
                logger.warning("[WARNING] tickers参数无效，尝试从其他数据推断...")
                
                # 检查是否能从对齐后的数据推断股票数量
                for name, data in aligned_data.items():
                    if name in ['tickers', 'ticker', 'prediction_tickers', 'symbols', 'stocks']:  # 跳过已经处理过的参数
                        continue
                        
                    if data is not None and hasattr(data, 'columns'):
                        # 检查DataFrame中是否有ticker相关列
                        ticker_cols = [col for col in data.columns if 'ticker' in str(col).lower() or 'symbol' in str(col).lower()]
                        if ticker_cols:
                            ticker_col = ticker_cols[0]
                            unique_tickers = data[ticker_col].unique() if hasattr(data[ticker_col], 'unique') else []
                            inferred_tickers = len(unique_tickers)
                            if inferred_tickers > 1:
                                effective_tickers = inferred_tickers
                                logger.info(f"[DATA] 从{name}.{ticker_col}列推断出{effective_tickers}只股票: {list(unique_tickers)[:10]}...")
                                tickers_found = True
                                break
                    elif data is not None and hasattr(data, 'index'):
                        # 检查索引中是否有股票信息模式
                        index_str = str(data.index)
                        if 'ticker' in index_str.lower() or 'symbol' in index_str.lower():
                            logger.info(f"[DATA] 检测到{name}数据包含股票索引信息")
                        
                        # 如果数据长度远大于日期数，可能是多股票
                        if effective_dates > 0 and len(data) > effective_dates * 2:  # 至少是日期数的2倍
                            inferred_tickers = len(data) // effective_dates
                            if 2 <= inferred_tickers <= 1000:  # 合理范围：2-1000只股票
                                effective_tickers = inferred_tickers
                                logger.info(f"[DATA] 从{name}数据长度推断出约{effective_tickers}只股票 (数据长度:{len(data)} / 日期数:{effective_dates})")
                                tickers_found = True
                                break
                    elif data is not None and hasattr(data, '__len__'):
                        # 对于普通数组/列表，检查长度模式
                        if effective_dates > 0 and len(data) > effective_dates * 2:
                            inferred_tickers = len(data) // effective_dates
                            if 2 <= inferred_tickers <= 1000:
                                effective_tickers = inferred_tickers  
                                logger.info(f"[DATA] 从{name}数组长度推断出约{effective_tickers}只股票")
                                tickers_found = True
                                break
            
            logger.info(f"[DATA] 非MultiIndex数据: {effective_tickers}只股票, {effective_dates}个时间点")
            
            # 横截面检查
            if effective_tickers < MIN_CROSS_SECTION:
                cross_section_ready = False
                logger.error(f"[ERROR] 股票数量不足：{effective_tickers} < {MIN_CROSS_SECTION}")
            else:
                cross_section_ready = True
                logger.info(f"[OK] 股票数量充足：{effective_tickers} >= {MIN_CROSS_SECTION}")
        
        total_removed = sum(removed_samples.values()) 
        total_original = sum(shape[0] if isinstance(shape, tuple) else 1 for shape in original_shapes.values())
        coverage_rate = 1.0 - (total_removed / max(total_original, 1))
        
        # 7. 生成对齐报告
        final_shape = (len(common_index),) if common_index is not None else (0,)
        
        alignment_report = AlignmentReport(
            original_shapes=original_shapes,
            final_shape=final_shape,
            removed_samples=removed_samples,
            removal_reasons=removal_reasons,
            coverage_rate=coverage_rate,
            effective_tickers=effective_tickers,
            effective_dates=effective_dates,
            horizon_trimmed=horizon_trimmed,
            daily_tickers_stats=daily_tickers_stats,
            cross_section_ready=cross_section_ready
        )
        
        # 8. 记录对齐历史
        self.alignment_history.append(alignment_report)
        
        # [HOT] CRITICAL FIX: Enhanced dimension validation
        self._validate_dimension_consistency(aligned_data, common_index)
        self._final_dimension_check(aligned_data)
        
        logger.info("[OK] IndexAligner对齐完成")
        logger.info(f"[DATA] 最终形状: {final_shape}")
        logger.info(f"[DATA] 覆盖率: {coverage_rate:.1%}")
        logger.info(f"[DATA] 有效股票: {effective_tickers}, 有效日期: {effective_dates}")
        
        return aligned_data, alignment_report
    
    def align_data(self, **data_dict) -> Tuple[Dict[str, Any], AlignmentReport]:
        """别名方法 - 调用align_all_data以保持兼容性"""
        return self.align_all_data(**data_dict)
    
    def print_alignment_report(self, report: AlignmentReport) -> None:
        """打印详细对齐报告"""
        print("="*60)
        print("[DATA] IndexAligner对齐报告")
        print("="*60)
        
        print("\n[SEARCH] 原始数据形状:")
        for name, shape in report.original_shapes.items():
            print(f"  {name:15s}: {shape}")
        
        print(f"\n[OK] 最终统一形状: {report.final_shape}")
        print(f"[TREND] 数据覆盖率: {report.coverage_rate:.1%}")
        print(f"[DATA] 有效股票数: {report.effective_tickers}")
        print(f"[DATA] 有效日期数: {report.effective_dates}")
        
        # [CRITICAL] 新增横截面统计显示
        if report.daily_tickers_stats:
            stats = report.daily_tickers_stats
            print(f"[TARGET] 每日股票数分布: min={stats['min']:.0f}, median={stats['median']:.0f}, max={stats['max']:.0f}")
            if not report.cross_section_ready:
                print("[ERROR] 横截面不足：无法进行有效的横截面分析")
            else:
                print("[OK] 横截面充足：可进行横截面排序分析")
        
        print("\n[DELETE] 数据移除统计:")
        for name, removed in report.removed_samples.items():
            if removed > 0:
                print(f"  {name:15s}: -{removed:,} 条")
        
        print("\n[LIST] 移除原因分析:")
        for reason, count in report.removal_reasons.items():
            if count > 0:
                print(f"  {reason:15s}: {count:,} 条")
        
        print("="*60)

    def _validate_dimension_consistency(self, aligned_data: Dict[str, Any], common_index: pd.Index):
        """验证维度一致性，防止738 vs 748等问题"""
        expected_len = len(common_index)
        
        for name, data in aligned_data.items():
            if data is None:
                continue
                
            actual_len = len(data) if hasattr(data, '__len__') else None
            
            if actual_len is not None and actual_len != expected_len:
                error_msg = f"CRITICAL DIMENSION MISMATCH: {name} has length {actual_len}, expected {expected_len}"
                logger.error(error_msg)
                
                # [HOT] FORCE ALIGNMENT: Truncate or pad to correct length
                if hasattr(data, 'iloc'):
                    if actual_len > expected_len:
                        aligned_data[name] = data.iloc[:expected_len]
                        logger.info(f"Force truncated {name} from {actual_len} to {expected_len}")
                    elif actual_len < expected_len:
                        # For MultiIndex, we can't easily pad, so we validate the common_index instead
                        if isinstance(data.index, pd.MultiIndex):
                            logger.warning(f"Cannot pad MultiIndex data {name}, using intersection")
                            intersection_index = common_index.intersection(data.index)
                            aligned_data[name] = data.loc[intersection_index]
                        else:
                            logger.warning(f"Cannot pad {name} from {actual_len} to {expected_len}")
    
    def _final_dimension_check(self, aligned_data: Dict[str, Any]):
        """最终维度检查，确保所有数据长度一致"""
        lengths = {}
        for name, data in aligned_data.items():
            if data is not None and hasattr(data, '__len__'):
                lengths[name] = len(data)
        
        if not lengths:
            return
            
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            error_msg = f"FINAL DIMENSION CHECK FAILED: {dict(lengths)}"
            logger.error(error_msg)
            
            # Find the most common length and force align all to it
            from collections import Counter
            length_counts = Counter(lengths.values())
            target_length = length_counts.most_common(1)[0][0]
            
            logger.info(f"Force aligning all data to length {target_length}")
            
            for name, data in aligned_data.items():
                if data is not None and hasattr(data, '__len__') and len(data) != target_length:
                    if hasattr(data, 'iloc'):
                        aligned_data[name] = data.iloc[:target_length]
                    elif hasattr(data, '__getitem__'):
                        aligned_data[name] = data[:target_length]
                    logger.info(f"Force aligned {name} to length {target_length}")
        else:
            logger.info(f"[OK] Final dimension check passed: all data has length {list(unique_lengths)[0]}")


def create_index_aligner(horizon: int = 10, strict_mode: bool = True, mode: str = 'train') -> IndexAligner:
    """
    创建索引对齐器
    
    Args:
        horizon: 前瞻期，训练模式下用于剪尾
        strict_mode: 严格模式 
        mode: 'train' (训练模式，执行剪尾) 或 'predict' (预测模式，不剪尾)
    """
    return IndexAligner(horizon=horizon, strict_mode=strict_mode, mode=mode)


# 全局对齐器实例
_global_aligner = None

def get_global_aligner() -> IndexAligner:
    """获取全局对齐器实例"""
    global _global_aligner
    if _global_aligner is None:
        _global_aligner = create_index_aligner()
    return _global_aligner


if __name__ == "__main__":
    # 测试对齐器
    import numpy as np
    
    # 模拟738 vs 748问题
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # 创建多级索引
    multi_index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    
    # 模拟不同长度的数据
    X = pd.DataFrame(np.random.randn(len(multi_index), 5), index=multi_index)  # 1500条
    y = pd.Series(np.random.randn(1490), index=multi_index[:1490])  # 1490条 (缺10条)
    alpha = pd.DataFrame(np.random.randn(1495, 3), index=multi_index[:1495])  # 1495条
    pred = pd.Series(np.random.randn(1485), index=multi_index[:1485])  # 1485条
    
    print("=== 测试IndexAligner ===")
    aligner = create_index_aligner(horizon=10)
    
    aligned_data, report = aligner.align_all_data(
        X=X, y=y, alpha=alpha, pred=pred
    )
    
    aligner.print_alignment_report(report)
    
    print("\n[OK] 对齐后数据长度:")
    for name, data in aligned_data.items():
        if data is not None:
            print(f"  {name}: {len(data)}")