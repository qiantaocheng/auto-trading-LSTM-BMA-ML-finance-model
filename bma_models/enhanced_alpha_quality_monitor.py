#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Alpha Factor Quality Monitor with Comprehensive Data Checks
每个Alpha因子计算都进行全面的数据质量监控
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from scipy import stats
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    missing_ratio: float = 0.0
    infinite_ratio: float = 0.0
    outlier_ratio: float = 0.0
    zero_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    coverage_ratio: float = 0.0
    stability_score: float = 0.0
    distribution_score: float = 0.0
    correlation_stability: float = 0.0
    time_consistency: float = 0.0

@dataclass
class AlphaFactorQualityReport:
    """Alpha因子质量报告"""
    factor_name: str
    timestamp: datetime
    input_quality: DataQualityMetrics
    output_quality: DataQualityMetrics
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class EnhancedAlphaQualityMonitor:
    """
    增强版Alpha因子质量监控器
    对每个Alpha因子计算进行全面的数据质量检查
    """
    
    def __init__(self, 
                 strict_mode: bool = True,
                 alert_threshold: Dict[str, float] = None,
                 log_dir: str = "logs/alpha_quality"):
        """
        初始化质量监控器
        
        Args:
            strict_mode: 严格模式，发现问题时是否抛出异常
            alert_threshold: 各项指标的警报阈值
            log_dir: 日志目录
        """
        self.strict_mode = strict_mode
        self.alert_threshold = alert_threshold or self._get_default_thresholds()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 监控历史
        self.quality_history = []
        self.alert_history = []
        
        # 性能统计
        self.factor_performance = {}
        
    def _get_default_thresholds(self) -> Dict[str, float]:
        """获取默认警报阈值"""
        return {
            'missing_ratio': 0.2,      # 缺失率超过20%警报
            'infinite_ratio': 0.01,     # 无穷值超过1%警报
            'outlier_ratio': 0.1,       # 异常值超过10%警报
            'zero_ratio': 0.5,          # 零值超过50%警报
            'duplicate_ratio': 0.3,     # 重复值超过30%警报
            'coverage_ratio': 0.7,      # 覆盖率低于70%警报
            'stability_score': 0.6,     # 稳定性低于0.6警报
            'distribution_score': 0.5,  # 分布得分低于0.5警报
            'correlation_stability': 0.3, # 相关性稳定性低于0.3警报
            'time_consistency': 0.7     # 时间一致性低于0.7警报
        }
    
    def monitor_alpha_calculation(self, 
                                 factor_name: str,
                                 input_data: pd.DataFrame,
                                 output_data: pd.Series,
                                 calculation_func: callable = None,
                                 **kwargs) -> AlphaFactorQualityReport:
        """
        监控单个Alpha因子计算
        
        Args:
            factor_name: 因子名称
            input_data: 输入数据
            output_data: 输出的因子值
            calculation_func: 计算函数（可选，用于性能分析）
            **kwargs: 其他参数
            
        Returns:
            质量报告
        """
        report = AlphaFactorQualityReport(
            factor_name=factor_name,
            timestamp=datetime.now(),
            input_quality=DataQualityMetrics(),
            output_quality=DataQualityMetrics()
        )
        
        # 1. 检查输入数据质量
        logger.info(f"[{factor_name}] 开始数据质量检查...")
        self._check_input_quality(input_data, report)
        
        # 2. 检查输出数据质量
        self._check_output_quality(output_data, report)
        
        # 3. 检查数据一致性
        self._check_data_consistency(input_data, output_data, report)
        
        # 4. 检查时间序列特性
        self._check_time_series_properties(input_data, output_data, report)
        
        # 5. 检查分布特性
        self._check_distribution_properties(output_data, report)
        
        # 6. 生成警报和建议
        self._generate_alerts_and_recommendations(report)
        
        # 7. 记录报告
        self._log_report(report)
        
        # 8. 严格模式下的异常处理
        if self.strict_mode and report.errors:
            error_msg = f"[{factor_name}] 数据质量检查失败: {'; '.join(report.errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return report
    
    def _check_input_quality(self, data: pd.DataFrame, report: AlphaFactorQualityReport):
        """检查输入数据质量"""
        if data is None or data.empty:
            report.errors.append("输入数据为空")
            return
        
        metrics = report.input_quality
        total_size = data.size
        
        # 检查缺失值
        metrics.missing_ratio = data.isna().sum().sum() / total_size
        if metrics.missing_ratio > self.alert_threshold['missing_ratio']:
            report.warnings.append(f"输入数据缺失率过高: {metrics.missing_ratio:.2%}")
        
        # 检查无穷值
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            metrics.infinite_ratio = np.isinf(numeric_data).sum().sum() / numeric_data.size
            if metrics.infinite_ratio > self.alert_threshold['infinite_ratio']:
                report.errors.append(f"输入数据包含无穷值: {metrics.infinite_ratio:.2%}")
        
        # 检查重复值（按时间和股票）
        if 'date' in data.columns and 'ticker' in data.columns:
            dup_count = data.duplicated(subset=['date', 'ticker']).sum()
            metrics.duplicate_ratio = dup_count / len(data)
            if metrics.duplicate_ratio > 0:
                report.warnings.append(f"输入数据存在重复: {metrics.duplicate_ratio:.2%}")
    
    def _check_output_quality(self, data: pd.Series, report: AlphaFactorQualityReport):
        """检查输出数据质量"""
        if data is None or data.empty:
            report.errors.append("输出数据为空")
            return
        
        metrics = report.output_quality
        total_size = len(data)
        
        # 检查缺失值
        metrics.missing_ratio = data.isna().sum() / total_size
        if metrics.missing_ratio > self.alert_threshold['missing_ratio']:
            report.warnings.append(f"输出因子缺失率过高: {metrics.missing_ratio:.2%}")
        
        # 检查无穷值
        metrics.infinite_ratio = np.isinf(data).sum() / total_size
        if metrics.infinite_ratio > self.alert_threshold['infinite_ratio']:
            report.errors.append(f"输出因子包含无穷值: {metrics.infinite_ratio:.2%}")
        
        # 检查零值
        metrics.zero_ratio = (data == 0).sum() / total_size
        if metrics.zero_ratio > self.alert_threshold['zero_ratio']:
            report.warnings.append(f"输出因子零值过多: {metrics.zero_ratio:.2%}")
        
        # 检查异常值（使用IQR方法）
        valid_data = data.dropna()
        if len(valid_data) > 10:
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((valid_data < (Q1 - 3 * IQR)) | (valid_data > (Q3 + 3 * IQR))).sum()
            metrics.outlier_ratio = outliers / len(valid_data)
            if metrics.outlier_ratio > self.alert_threshold['outlier_ratio']:
                report.warnings.append(f"输出因子异常值过多: {metrics.outlier_ratio:.2%}")
    
    def _check_data_consistency(self, input_data: pd.DataFrame, output_data: pd.Series, report: AlphaFactorQualityReport):
        """检查输入输出数据一致性"""
        # 检查长度一致性
        if len(input_data) != len(output_data):
            report.errors.append(f"输入输出长度不一致: 输入{len(input_data)}, 输出{len(output_data)}")
        
        # 检查索引一致性
        if not input_data.index.equals(output_data.index):
            report.warnings.append("输入输出索引不完全一致")
            
        # 检查覆盖率
        if len(output_data.dropna()) > 0:
            report.output_quality.coverage_ratio = len(output_data.dropna()) / len(input_data)
            if report.output_quality.coverage_ratio < self.alert_threshold['coverage_ratio']:
                report.warnings.append(f"因子覆盖率不足: {report.output_quality.coverage_ratio:.2%}")
    
    def _check_time_series_properties(self, input_data: pd.DataFrame, output_data: pd.Series, report: AlphaFactorQualityReport):
        """检查时间序列特性"""
        if 'date' not in input_data.columns:
            return
        
        # 按日期分组检查
        date_groups = input_data.groupby('date').size()
        
        # 检查时间连续性
        dates = pd.to_datetime(date_groups.index)
        date_diffs = dates.to_series().diff()
        if date_diffs.max() > pd.Timedelta(days=10):
            report.warnings.append(f"时间序列存在较大断裂: 最大间隔{date_diffs.max().days}天")
        
        # 检查每个时间点的数据量稳定性
        std_ratio = date_groups.std() / date_groups.mean() if date_groups.mean() > 0 else 0
        report.output_quality.time_consistency = 1 - min(std_ratio, 1)
        if report.output_quality.time_consistency < self.alert_threshold['time_consistency']:
            report.warnings.append(f"时间序列数据量不稳定: 一致性{report.output_quality.time_consistency:.2f}")
    
    def _check_distribution_properties(self, output_data: pd.Series, report: AlphaFactorQualityReport):
        """检查分布特性"""
        valid_data = output_data.dropna()
        if len(valid_data) < 30:
            return
        
        # 检查正态性
        _, p_value = stats.normaltest(valid_data)
        is_normal = p_value > 0.05
        
        # 检查偏度和峰度
        skewness = stats.skew(valid_data)
        kurtosis = stats.kurtosis(valid_data)
        
        # 计算分布得分
        distribution_score = 1.0
        if abs(skewness) > 2:
            distribution_score *= 0.8
            report.warnings.append(f"因子分布偏度过大: {skewness:.2f}")
        if abs(kurtosis) > 7:
            distribution_score *= 0.8
            report.warnings.append(f"因子分布峰度过大: {kurtosis:.2f}")
        
        report.output_quality.distribution_score = distribution_score
        
        # 记录性能指标
        report.performance_metrics['skewness'] = skewness
        report.performance_metrics['kurtosis'] = kurtosis
        report.performance_metrics['is_normal'] = is_normal
    
    def _generate_alerts_and_recommendations(self, report: AlphaFactorQualityReport):
        """生成警报和建议"""
        # 根据检查结果生成建议
        if report.input_quality.missing_ratio > 0.1:
            report.recommendations.append("建议增强数据清洗流程，减少缺失值")
        
        if report.output_quality.outlier_ratio > 0.05:
            report.recommendations.append("建议应用Winsorization或其他异常值处理方法")
        
        if report.output_quality.zero_ratio > 0.3:
            report.recommendations.append("建议检查因子计算逻辑，过多零值可能影响信号质量")
        
        if report.output_quality.distribution_score < 0.7:
            report.recommendations.append("建议对因子进行标准化或正态化处理")
        
        if report.output_quality.coverage_ratio < 0.8:
            report.recommendations.append("建议优化数据获取流程，提高因子覆盖率")
    
    def _log_report(self, report: AlphaFactorQualityReport):
        """记录质量报告"""
        # 添加到历史记录
        self.quality_history.append(report)
        
        # 记录到文件
        log_file = self.log_dir / f"alpha_quality_{report.factor_name}_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        report_dict = {
            'factor_name': report.factor_name,
            'timestamp': report.timestamp.isoformat(),
            'input_quality': report.input_quality.__dict__,
            'output_quality': report.output_quality.__dict__,
            'warnings': report.warnings,
            'errors': report.errors,
            'recommendations': report.recommendations,
            'performance_metrics': report.performance_metrics
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
        # 记录警报
        if report.errors or report.warnings:
            self.alert_history.append({
                'factor_name': report.factor_name,
                'timestamp': report.timestamp,
                'errors': report.errors,
                'warnings': report.warnings
            })
    
    def get_factor_statistics(self, factor_name: str = None) -> Dict[str, Any]:
        """
        获取因子质量统计
        
        Args:
            factor_name: 因子名称，None表示所有因子
            
        Returns:
            统计信息
        """
        if factor_name:
            reports = [r for r in self.quality_history if r.factor_name == factor_name]
        else:
            reports = self.quality_history
        
        if not reports:
            return {}
        
        stats = {
            'total_checks': len(reports),
            'error_count': sum(1 for r in reports if r.errors),
            'warning_count': sum(1 for r in reports if r.warnings),
            'avg_missing_ratio': np.mean([r.output_quality.missing_ratio for r in reports]),
            'avg_coverage_ratio': np.mean([r.output_quality.coverage_ratio for r in reports]),
            'avg_distribution_score': np.mean([r.output_quality.distribution_score for r in reports]),
            'recent_alerts': self.alert_history[-10:] if self.alert_history else []
        }
        
        return stats
    
    def generate_summary_report(self, output_file: str = None) -> pd.DataFrame:
        """
        生成汇总报告
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            汇总DataFrame
        """
        if not self.quality_history:
            logger.warning("没有质量检查记录")
            return pd.DataFrame()
        
        # 构建汇总数据
        summary_data = []
        for report in self.quality_history:
            summary_data.append({
                'factor_name': report.factor_name,
                'timestamp': report.timestamp,
                'input_missing_ratio': report.input_quality.missing_ratio,
                'output_missing_ratio': report.output_quality.missing_ratio,
                'coverage_ratio': report.output_quality.coverage_ratio,
                'outlier_ratio': report.output_quality.outlier_ratio,
                'distribution_score': report.output_quality.distribution_score,
                'error_count': len(report.errors),
                'warning_count': len(report.warnings),
                'has_issues': len(report.errors) > 0 or len(report.warnings) > 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存到文件
        if output_file:
            if output_file.endswith('.csv'):
                summary_df.to_csv(output_file, index=False)
            elif output_file.endswith('.xlsx'):
                summary_df.to_excel(output_file, index=False)
            logger.info(f"质量汇总报告已保存到: {output_file}")
        
        return summary_df

# 便捷函数
def create_quality_monitor(strict_mode: bool = False) -> EnhancedAlphaQualityMonitor:
    """创建质量监控器实例"""
    return EnhancedAlphaQualityMonitor(strict_mode=strict_mode)

def monitor_alpha_factor(factor_name: str, 
                        input_data: pd.DataFrame,
                        output_data: pd.Series,
                        monitor: EnhancedAlphaQualityMonitor = None) -> AlphaFactorQualityReport:
    """
    监控单个Alpha因子
    
    Args:
        factor_name: 因子名称
        input_data: 输入数据
        output_data: 输出因子值
        monitor: 监控器实例，None则创建新实例
        
    Returns:
        质量报告
    """
    if monitor is None:
        monitor = create_quality_monitor()
    
    return monitor.monitor_alpha_calculation(factor_name, input_data, output_data)