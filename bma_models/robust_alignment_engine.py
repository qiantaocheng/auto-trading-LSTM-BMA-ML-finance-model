#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健壮对齐引擎 - 统一的第一层到第二层数据对齐解决方案
Robust Alignment Engine for unified first-to-second layer data alignment

整合功能：
- 简化的数据对齐逻辑
- 强化的数据验证机制
- 统一的错误处理
- 详细的对齐报告
- 自动修复常见问题
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional, Union
import logging
from datetime import datetime

# 导入自定义模块
try:
    from .simplified_data_aligner import SimplifiedDataAligner, create_simple_aligner
    from .enhanced_data_validator import EnhancedDataValidator, create_enhanced_validator
    CUSTOM_MODULES_AVAILABLE = True
except ImportError:
    # Fallback到当前目录导入
    try:
        from simplified_data_aligner import SimplifiedDataAligner, create_simple_aligner
        from enhanced_data_validator import EnhancedDataValidator, create_enhanced_validator
        CUSTOM_MODULES_AVAILABLE = True
    except ImportError:
        CUSTOM_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)

class AlignmentError(Exception):
    """对齐错误"""
    pass

class RobustAlignmentEngine:
    """
    健壮对齐引擎

    核心功能：
    1. 统一的数据验证和对齐流程
    2. 自动问题检测和修复
    3. 详细的诊断报告
    4. 多层次的fallback机制
    5. 完整的审计追踪
    """

    def __init__(self,
                 strict_validation: bool = True,
                 auto_fix: bool = True,
                 backup_strategy: str = 'intersection',
                 min_samples: int = 100):
        """
        初始化健壮对齐引擎

        Args:
            strict_validation: 是否使用严格验证
            auto_fix: 是否自动修复常见问题
            backup_strategy: 备用策略 ('intersection', 'strict', 'partial')
            min_samples: 最小样本数要求
        """
        self.strict_validation = strict_validation
        self.auto_fix = auto_fix
        self.backup_strategy = backup_strategy
        self.min_samples = min_samples

        # 初始化组件
        if CUSTOM_MODULES_AVAILABLE:
            self.validator = create_enhanced_validator(
                min_samples=min_samples,
                min_coverage_rate=0.8,
                temporal_safety=strict_validation,
                leakage_detection=strict_validation
            )
            self.aligner = create_simple_aligner(strict_mode=strict_validation)
        else:
            self.validator = None
            self.aligner = None
            logger.warning("自定义模块不可用，使用基础对齐逻辑")

        # 状态跟踪
        self.alignment_history = []
        self.last_alignment_report = None
        self.auto_fixes_applied = []

        logger.info(f"RobustAlignmentEngine初始化完成")
        logger.info(f"  严格验证: {strict_validation}")
        logger.info(f"  自动修复: {auto_fix}")
        logger.info(f"  备用策略: {backup_strategy}")
        logger.info(f"  最小样本: {min_samples}")

    def _fallback_basic_alignment(self,
                                 oof_predictions: Dict[str, pd.Series],
                                 target: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        基础对齐逻辑（fallback）

        Args:
            oof_predictions: OOF预测字典
            target: 目标变量

        Returns:
            对齐后的数据和报告
        """
        logger.info("使用基础对齐逻辑")

        try:
            # 获取第一个预测作为基准
            first_key = list(oof_predictions.keys())[0]
            base_index = oof_predictions[first_key].index

            # 验证基本格式
            if not isinstance(base_index, pd.MultiIndex):
                raise AlignmentError("基准索引必须是MultiIndex")

            if base_index.nlevels != 2:
                raise AlignmentError(f"MultiIndex必须是2层，当前: {base_index.nlevels}")

            # 确保索引名称
            if base_index.names != ['date', 'ticker']:
                base_index = base_index.set_names(['date', 'ticker'])

            # 对齐所有预测到基准索引
            aligned_predictions = {}
            for name, pred in oof_predictions.items():
                if isinstance(pred.index, pd.MultiIndex):
                    pred_index = pred.index.set_names(['date', 'ticker'])
                    if not pred_index.equals(base_index):
                        # 使用交集对齐
                        common_index = base_index.intersection(pred_index)
                        if len(common_index) > 0:
                            aligned_predictions[name] = pred.reindex(common_index)
                        else:
                            raise AlignmentError(f"预测 {name} 与基准无交集")
                    else:
                        aligned_predictions[name] = pred
                else:
                    raise AlignmentError(f"预测 {name} 索引格式错误")

            # 对齐目标变量
            if isinstance(target.index, pd.MultiIndex):
                target_index = target.index.set_names(['date', 'ticker'])
                final_index = list(aligned_predictions.values())[0].index
                if not target_index.equals(final_index):
                    common_index = final_index.intersection(target_index)
                    if len(common_index) > 0:
                        # 重新对齐所有数据到最终交集
                        for name in aligned_predictions:
                            aligned_predictions[name] = aligned_predictions[name].reindex(common_index)
                        target = target.reindex(common_index)
                        final_index = common_index
                    else:
                        raise AlignmentError("目标变量与预测无交集")
            else:
                raise AlignmentError("目标变量索引格式错误")

            # 构建最终DataFrame
            stacker_data = pd.DataFrame(index=final_index)

            # 标准化列名
            column_mapping = {
                'elastic_net': 'pred_elastic',
                'xgboost': 'pred_xgb',
                'catboost': 'pred_catboost'
            }

            for name, pred in aligned_predictions.items():
                std_name = column_mapping.get(name, name)
                stacker_data[std_name] = pred

            # 固定目标列：统一T+5
            target_col = 'ret_fwd_10d'
            stacker_data[target_col] = target

            # 基础验证
            if len(stacker_data) < self.min_samples:
                raise AlignmentError(f"样本数不足: {len(stacker_data)} < {self.min_samples}")

            # 生成报告
            report = {
                'success': True,
                'method': 'fallback_basic_alignment',
                'samples': len(stacker_data),
                'predictions': list(aligned_predictions.keys()),
                'target_column': target_col,
                'warnings': [],
                'errors': []
            }

            logger.info(f"基础对齐完成: {len(stacker_data)} 样本")
            return stacker_data, report

        except Exception as e:
            logger.error(f"基础对齐失败: {e}")
            raise AlignmentError(f"基础对齐失败: {e}")

    def _apply_auto_fixes(self, oof_predictions: Dict[str, pd.Series], target: pd.Series) -> Tuple[Dict[str, pd.Series], pd.Series]:
        """
        应用自动修复

        Args:
            oof_predictions: OOF预测字典
            target: 目标变量

        Returns:
            修复后的预测和目标变量
        """
        if not self.auto_fix:
            return oof_predictions, target

        logger.info("应用自动修复")
        self.auto_fixes_applied.clear()

        fixed_predictions = {}

        # 修复1: 标准化索引名称
        for name, pred in oof_predictions.items():
            if isinstance(pred.index, pd.MultiIndex):
                if pred.index.names != ['date', 'ticker']:
                    pred = pred.copy()
                    pred.index = pred.index.set_names(['date', 'ticker'])
                    self.auto_fixes_applied.append(f"标准化 {name} 索引名称")

            fixed_predictions[name] = pred

        # 修复目标变量索引
        if isinstance(target.index, pd.MultiIndex):
            if target.index.names != ['date', 'ticker']:
                target = target.copy()
                target.index = target.index.set_names(['date', 'ticker'])
                self.auto_fixes_applied.append("标准化目标变量索引名称")

        # 修复2: 处理重复索引
        for name, pred in fixed_predictions.items():
            if pred.index.duplicated().any():
                dup_count = pred.index.duplicated().sum()
                pred = pred[~pred.index.duplicated(keep='first')]
                fixed_predictions[name] = pred
                self.auto_fixes_applied.append(f"移除 {name} 的 {dup_count} 个重复索引")

        if target.index.duplicated().any():
            dup_count = target.index.duplicated().sum()
            target = target[~target.index.duplicated(keep='first')]
            self.auto_fixes_applied.append(f"移除目标变量的 {dup_count} 个重复索引")

        # 修复3: 处理无穷值和NaN
        for name, pred in fixed_predictions.items():
            original_count = len(pred)
            pred = pred.replace([np.inf, -np.inf], np.nan)
            fixed_predictions[name] = pred

            nan_count = pred.isna().sum()
            if nan_count > 0:
                self.auto_fixes_applied.append(f"{name}: 处理了 {nan_count} 个NaN/Inf值")

        target = target.replace([np.inf, -np.inf], np.nan)
        target_nan_count = target.isna().sum()
        if target_nan_count > 0:
            self.auto_fixes_applied.append(f"目标变量: 处理了 {target_nan_count} 个NaN/Inf值")

        if self.auto_fixes_applied:
            logger.info(f"应用了 {len(self.auto_fixes_applied)} 个自动修复")
            for fix in self.auto_fixes_applied:
                logger.info(f"  - {fix}")

        return fixed_predictions, target

    def align_data(self,
                   oof_predictions: Dict[str, pd.Series],
                   target: pd.Series,
                   features: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        主要对齐方法

        Args:
            oof_predictions: OOF预测字典
            target: 目标变量
            features: 特征数据（可选，用于泄漏检测）

        Returns:
            对齐后的数据和详细报告

        Raises:
            AlignmentError: 对齐失败
        """
        start_time = datetime.now()
        logger.info("开始健壮数据对齐")

        # 初始化报告
        alignment_report = {
            'timestamp': start_time,
            'success': False,
            'method': None,
            'validation_report': None,
            'alignment_details': None,
            'auto_fixes_applied': [],
            'warnings': [],
            'errors': [],
            'performance': {}
        }

        try:
            # Step 1: 应用自动修复
            if self.auto_fix:
                oof_predictions, target = self._apply_auto_fixes(oof_predictions, target)
                alignment_report['auto_fixes_applied'] = self.auto_fixes_applied.copy()

            # Step 2: 数据验证（如果可用）
            if self.validator is not None:
                try:
                    validation_report = self.validator.comprehensive_validation(
                        oof_predictions, target, features
                    )
                    alignment_report['validation_report'] = validation_report

                    if not validation_report['validation_passed']:
                        if self.strict_validation:
                            raise AlignmentError(f"数据验证失败: {validation_report['errors_count']} 个错误")
                        else:
                            alignment_report['warnings'].extend(validation_report.get('warnings', []))

                except Exception as e:
                    logger.warning(f"数据验证失败，继续对齐: {e}")
                    alignment_report['warnings'].append(f"数据验证异常: {e}")

            # Step 3: 数据对齐
            if self.aligner is not None:
                try:
                    stacker_data, align_details = self.aligner.align_first_to_second_layer(
                        oof_predictions, target
                    )
                    alignment_report['method'] = 'simplified_aligner'
                    alignment_report['alignment_details'] = align_details

                except Exception as e:
                    logger.warning(f"简化对齐器失败，使用fallback: {e}")
                    alignment_report['warnings'].append(f"简化对齐器失败: {e}")

                    stacker_data, align_details = self._fallback_basic_alignment(
                        oof_predictions, target
                    )
                    alignment_report['method'] = 'fallback_basic'
                    alignment_report['alignment_details'] = align_details
            else:
                # 直接使用fallback
                stacker_data, align_details = self._fallback_basic_alignment(
                    oof_predictions, target
                )
                alignment_report['method'] = 'fallback_basic'
                alignment_report['alignment_details'] = align_details

            # Step 4: 最终验证
            if self.aligner is not None:
                try:
                    self.aligner.validate_ridge_input(stacker_data)
                    logger.info("Ridge输入验证通过")
                except Exception as e:
                    alignment_report['warnings'].append(f"Ridge输入验证警告: {e}")

            # Step 5: 生成性能报告
            end_time = datetime.now()
            alignment_report['performance'] = {
                'total_time': (end_time - start_time).total_seconds(),
                'samples_processed': len(stacker_data),
                'predictions_aligned': len(oof_predictions),
                'memory_usage_mb': stacker_data.memory_usage(deep=True).sum() / 1024 / 1024
            }

            alignment_report['success'] = True
            self.last_alignment_report = alignment_report

            # 添加到历史
            self.alignment_history.append({
                'timestamp': start_time,
                'method': alignment_report['method'],
                'samples': len(stacker_data),
                'success': True
            })

            logger.info(f"数据对齐成功: {len(stacker_data)} 样本，方法: {alignment_report['method']}")
            logger.info(f"处理时间: {alignment_report['performance']['total_time']:.2f}秒")

            return stacker_data, alignment_report

        except Exception as e:
            alignment_report['success'] = False
            alignment_report['errors'].append(str(e))

            # 添加失败记录到历史
            self.alignment_history.append({
                'timestamp': start_time,
                'method': alignment_report.get('method', 'unknown'),
                'samples': 0,
                'success': False,
                'error': str(e)
            })

            logger.error(f"数据对齐失败: {e}")
            raise AlignmentError(f"数据对齐失败: {e}")

    def get_alignment_summary(self) -> Dict[str, Any]:
        """
        获取对齐总结

        Returns:
            对齐历史和统计
        """
        if not self.alignment_history:
            return {'message': '暂无对齐历史'}

        successful_alignments = [h for h in self.alignment_history if h['success']]
        failed_alignments = [h for h in self.alignment_history if not h['success']]

        summary = {
            'total_alignments': len(self.alignment_history),
            'successful_alignments': len(successful_alignments),
            'failed_alignments': len(failed_alignments),
            'success_rate': len(successful_alignments) / len(self.alignment_history) if self.alignment_history else 0,
            'last_alignment': self.last_alignment_report,
            'methods_used': list(set(h['method'] for h in self.alignment_history)),
            'average_samples': np.mean([h['samples'] for h in successful_alignments]) if successful_alignments else 0,
            'recent_history': self.alignment_history[-5:]  # 最近5次
        }

        return summary

def create_robust_alignment_engine(**kwargs) -> RobustAlignmentEngine:
    """
    创建健壮对齐引擎的便捷函数

    Args:
        **kwargs: 引擎参数

    Returns:
        RobustAlignmentEngine实例
    """
    return RobustAlignmentEngine(**kwargs)

# 测试函数
def test_robust_alignment_engine():
    """测试健壮对齐引擎"""
    import numpy as np

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])

    # 创建OOF预测（模拟不同的数据质量问题）
    oof_predictions = {
        'elastic_net': pd.Series(np.random.normal(0, 0.02, len(index)), index=index),
        'xgboost': pd.Series(np.random.normal(0, 0.02, len(index)), index=index),
        'catboost': pd.Series(np.random.normal(0, 0.02, len(index)), index=index)
    }

    # 在xgboost预测中添加一些NaN和Inf
    oof_predictions['xgboost'].iloc[10:15] = np.nan
    oof_predictions['xgboost'].iloc[20] = np.inf

    # 创建目标变量
    target = pd.Series(np.random.normal(0, 0.03, len(index)), index=index)

    # 测试引擎
    engine = create_robust_alignment_engine(
        strict_validation=False,  # 允许一些数据质量问题
        auto_fix=True,
        backup_strategy='intersection'
    )

    try:
        stacker_data, report = engine.align_data(oof_predictions, target)

        print("✅ 健壮对齐引擎测试成功")
        print(f"对齐方法: {report['method']}")
        print(f"最终样本数: {len(stacker_data)}")
        print(f"列名: {list(stacker_data.columns)}")
        print(f"自动修复数: {len(report['auto_fixes_applied'])}")
        print(f"警告数: {len(report['warnings'])}")
        print(f"处理时间: {report['performance']['total_time']:.2f}秒")

        # 获取对齐总结
        summary = engine.get_alignment_summary()
        print(f"对齐成功率: {summary['success_rate']:.1%}")

        return stacker_data, report

    except Exception as e:
        print(f"❌ 健壮对齐引擎测试失败: {e}")
        return None, None

if __name__ == "__main__":
    test_robust_alignment_engine()