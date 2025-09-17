#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Factor Quality Monitoring System
实时监控每个alpha因子的计算质量和数据完整性
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class AlphaFactorQualityMonitor:
    """
    Alpha因子质量监控系统
    
    功能:
    1. 实时监控每个因子计算
    2. 数据质量检查
    3. 异常值检测
    4. 覆盖率统计
    5. 性能追踪
    """
    
    def __init__(self, save_reports: bool = True, report_dir: str = "cache/factor_quality"):
        self.save_reports = save_reports
        self.report_dir = report_dir
        self.factor_stats = {}
        self.quality_issues = []
        self.computation_times = {}
        
        if save_reports:
            os.makedirs(report_dir, exist_ok=True)
    
    def monitor_factor_computation(self, factor_name: str, factor_data: pd.Series, 
                                  computation_time: float = None) -> Dict[str, Any]:
        """
        监控单个因子的计算质量
        
        Args:
            factor_name: 因子名称
            factor_data: 因子数据
            computation_time: 计算耗时
            
        Returns:
            质量报告字典
        """
        logger.info(f"🔍 [MONITOR] Analyzing {factor_name}...")
        
        quality_report = {
            'factor_name': factor_name,
            'timestamp': datetime.now().isoformat(),
            'data_points': len(factor_data),
            'computation_time': computation_time
        }
        
        # 1. 基础统计
        quality_report['statistics'] = {
            'mean': float(factor_data.mean()),
            'std': float(factor_data.std()),
            'min': float(factor_data.min()),
            'max': float(factor_data.max()),
            'median': float(factor_data.median()),
            'skew': float(factor_data.skew()),
            'kurtosis': float(factor_data.kurtosis())
        }
        
        # 2. 数据质量指标
        quality_report['data_quality'] = {
            'non_zero_count': int((factor_data != 0).sum()),
            'non_zero_ratio': float((factor_data != 0).sum() / len(factor_data)),
            'nan_count': int(factor_data.isna().sum()),
            'nan_ratio': float(factor_data.isna().sum() / len(factor_data)),
            'inf_count': int(np.isinf(factor_data).sum()),
            'unique_values': int(factor_data.nunique()),
            'unique_ratio': float(factor_data.nunique() / len(factor_data))
        }
        
        # 3. 覆盖率评估
        coverage = quality_report['data_quality']['non_zero_ratio'] * 100
        quality_report['coverage'] = {
            'percentage': coverage,
            'status': self._get_coverage_status(coverage)
        }
        
        # 4. 异常值检测
        outliers = self._detect_outliers(factor_data)
        quality_report['outliers'] = {
            'count': len(outliers),
            'percentage': float(len(outliers) / len(factor_data) * 100),
            'values': outliers[:10].tolist() if len(outliers) > 0 else []
        }
        
        # 5. 分布健康度
        quality_report['distribution_health'] = self._assess_distribution_health(factor_data)
        
        # 6. 时间序列特性（如果是时间序列数据）
        if isinstance(factor_data.index, pd.MultiIndex):
            quality_report['temporal_properties'] = self._analyze_temporal_properties(factor_data)
        
        # 7. 质量评分
        quality_score = self._calculate_quality_score(quality_report)
        quality_report['quality_score'] = quality_score
        
        # 记录问题
        if quality_score['overall'] < 70:
            self.quality_issues.append({
                'factor': factor_name,
                'score': quality_score['overall'],
                'issues': quality_score['issues']
            })
        
        # 保存统计
        self.factor_stats[factor_name] = quality_report
        
        # 记录计算时间
        if computation_time:
            self.computation_times[factor_name] = computation_time
        
        # 实时日志输出
        self._log_factor_quality(factor_name, quality_report)
        
        return quality_report
    
    def _get_coverage_status(self, coverage: float) -> str:
        """评估覆盖率状态"""
        if coverage >= 90:
            return "EXCELLENT"
        elif coverage >= 70:
            return "GOOD"
        elif coverage >= 50:
            return "FAIR"
        elif coverage >= 30:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _detect_outliers(self, data: pd.Series, n_std: float = 3) -> pd.Series:
        """检测异常值（超过n个标准差）"""
        mean = data.mean()
        std = data.std()
        if std == 0:
            return pd.Series([])
        outliers = data[np.abs(data - mean) > n_std * std]
        return outliers
    
    def _assess_distribution_health(self, data: pd.Series) -> Dict[str, Any]:
        """评估数据分布健康度"""
        health = {
            'is_constant': data.std() == 0,
            'is_binary': data.nunique() == 2,
            'is_highly_skewed': abs(data.skew()) > 2,
            'has_fat_tails': data.kurtosis() > 3,
            'variance_ratio': float(data.var() / (data.mean()**2 + 1e-10))
        }
        
        # 健康评分
        health_score = 100
        if health['is_constant']:
            health_score -= 50
        if health['is_highly_skewed']:
            health_score -= 20
        if health['has_fat_tails']:
            health_score -= 10
        
        health['score'] = health_score
        return health
    
    def _analyze_temporal_properties(self, data: pd.Series) -> Dict[str, Any]:
        """分析时间序列特性"""
        temporal = {}
        
        try:
            # 按日期分组分析
            dates = data.index.get_level_values(0)
            unique_dates = dates.unique()
            
            temporal['date_coverage'] = {
                'unique_dates': len(unique_dates),
                'first_date': str(unique_dates.min()),
                'last_date': str(unique_dates.max())
            }
            
            # 每日数据点数量
            daily_counts = data.groupby(level=0).count()
            temporal['daily_distribution'] = {
                'mean_points_per_day': float(daily_counts.mean()),
                'std_points_per_day': float(daily_counts.std()),
                'min_points_per_day': int(daily_counts.min()),
                'max_points_per_day': int(daily_counts.max())
            }
            
        except Exception as e:
            logger.warning(f"Temporal analysis failed: {e}")
            temporal = {'error': str(e)}
        
        return temporal
    
    def _calculate_quality_score(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """计算综合质量分数"""
        score_components = {
            'coverage': min(report['coverage']['percentage'], 100) * 0.3,
            'uniqueness': min(report['data_quality']['unique_ratio'] * 100, 100) * 0.2,
            'distribution': report['distribution_health']['score'] * 0.25,
            'outliers': max(100 - report['outliers']['percentage'] * 10, 0) * 0.15,
            'completeness': (1 - report['data_quality']['nan_ratio']) * 100 * 0.1
        }
        
        overall_score = sum(score_components.values())
        
        # 识别问题
        issues = []
        if report['coverage']['percentage'] < 50:
            issues.append(f"Low coverage: {report['coverage']['percentage']:.1f}%")
        if report['data_quality']['unique_ratio'] < 0.1:
            issues.append(f"Low uniqueness: {report['data_quality']['unique_ratio']:.2%}")
        if report['distribution_health']['is_constant']:
            issues.append("Constant values detected")
        if report['outliers']['percentage'] > 5:
            issues.append(f"High outlier ratio: {report['outliers']['percentage']:.1f}%")
        
        return {
            'overall': overall_score,
            'components': score_components,
            'issues': issues
        }
    
    def _log_factor_quality(self, factor_name: str, report: Dict[str, Any]):
        """输出因子质量日志"""
        score = report['quality_score']['overall']
        coverage = report['coverage']['percentage']
        
        # 选择合适的emoji
        if score >= 90:
            emoji = "🌟"
            level = "EXCELLENT"
        elif score >= 70:
            emoji = "✅"
            level = "GOOD"
        elif score >= 50:
            emoji = "⚠️"
            level = "WARNING"
        else:
            emoji = "❌"
            level = "CRITICAL"
        
        logger.info(f"   {emoji} {factor_name}: Score={score:.1f}, Coverage={coverage:.1f}%, Status={level}")
        
        # 输出具体问题
        if report['quality_score']['issues']:
            for issue in report['quality_score']['issues']:
                logger.warning(f"      ⚠️ {issue}")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """获取汇总报告"""
        if not self.factor_stats:
            return {'error': 'No factors monitored yet'}
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_factors': len(self.factor_stats),
            'total_computation_time': sum(self.computation_times.values()),
            'average_quality_score': np.mean([s['quality_score']['overall'] 
                                             for s in self.factor_stats.values()]),
            'factors_with_issues': len(self.quality_issues),
            'quality_distribution': self._get_quality_distribution(),
            'top_issues': self.quality_issues[:5],
            'performance_metrics': self._get_performance_metrics()
        }
        
        return summary
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """获取质量分布"""
        distribution = {
            'excellent': 0,
            'good': 0,
            'fair': 0,
            'poor': 0,
            'critical': 0
        }
        
        for stats in self.factor_stats.values():
            score = stats['quality_score']['overall']
            if score >= 90:
                distribution['excellent'] += 1
            elif score >= 70:
                distribution['good'] += 1
            elif score >= 50:
                distribution['fair'] += 1
            elif score >= 30:
                distribution['poor'] += 1
            else:
                distribution['critical'] += 1
        
        return distribution
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.computation_times:
            return {}
        
        times = list(self.computation_times.values())
        return {
            'total_time': sum(times),
            'average_time': np.mean(times),
            'fastest_factor': min(self.computation_times.items(), key=lambda x: x[1]),
            'slowest_factor': max(self.computation_times.items(), key=lambda x: x[1])
        }
    
    def save_report(self, filename: str = None):
        """保存质量报告"""
        if not self.save_reports:
            return
        
        if filename is None:
            filename = f"factor_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.report_dir, filename)
        
        report = {
            'summary': self.get_summary_report(),
            'factor_details': self.factor_stats,
            'quality_issues': self.quality_issues,
            'computation_times': self.computation_times
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Quality report saved to {filepath}")
    
    def print_summary(self):
        """打印质量摘要"""
        summary = self.get_summary_report()
        
        print("\n" + "="*60)
        print("ALPHA FACTOR QUALITY SUMMARY")
        print("="*60)
        print(f"Total Factors: {summary['total_factors']}")
        print(f"Average Quality Score: {summary['average_quality_score']:.1f}")
        print(f"Factors with Issues: {summary['factors_with_issues']}")
        print(f"Total Computation Time: {summary['total_computation_time']:.3f}s")
        
        print("\nQuality Distribution:")
        for level, count in summary['quality_distribution'].items():
            print(f"  {level.upper()}: {count}")
        
        if summary['top_issues']:
            print("\nTop Issues:")
            for issue in summary['top_issues']:
                print(f"  - {issue['factor']}: {issue['issues']}")
        
        print("="*60)


# 全局监控实例
factor_monitor = AlphaFactorQualityMonitor()

def monitor_factor(factor_name: str, factor_data: pd.Series, computation_time: float = None) -> Dict[str, Any]:
    """便捷函数：监控单个因子"""
    return factor_monitor.monitor_factor_computation(factor_name, factor_data, computation_time)