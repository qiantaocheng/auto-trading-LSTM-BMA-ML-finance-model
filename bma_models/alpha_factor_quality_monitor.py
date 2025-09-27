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
        total_count = len(factor_data)
        quality_report['data_quality'] = {
            'non_zero_count': int((factor_data != 0).sum()),
            'non_zero_ratio': float((factor_data != 0).sum() / total_count) if total_count > 0 else 0.0,
            'nan_count': int(factor_data.isna().sum()),
            'nan_ratio': float(factor_data.isna().sum() / total_count) if total_count > 0 else 0.0,
            'inf_count': int(np.isinf(factor_data).sum()),
            'unique_values': int(factor_data.nunique()),
            'unique_ratio': float(factor_data.nunique() / total_count) if total_count > 0 else 0.0
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
            'percentage': float(len(outliers) / total_count * 100) if total_count > 0 else 0.0,
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
            'is_constant': bool(data.std() == 0),
            'is_binary': bool(data.nunique() == 2),
            'is_highly_skewed': bool(abs(data.skew()) > 2),
            'has_fat_tails': bool(data.kurtosis() > 3),
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
    
    def monitor_sentiment_factor(self, sentiment_data: pd.DataFrame,
                               news_count_by_ticker: Dict[str, int] = None,
                               processing_time: float = None) -> Dict[str, Any]:
        """
        专门监控情感因子的数据质量

        Args:
            sentiment_data: 情感因子数据 (MultiIndex: date, ticker)
            news_count_by_ticker: 每只股票的新闻数量统计
            processing_time: 处理耗时

        Returns:
            情感因子专用质量报告
        """
        logger.info("📰 [SENTIMENT MONITOR] Analyzing sentiment factor quality...")

        # 基础监控
        if 'sentiment_score' in sentiment_data.columns:
            base_report = self.monitor_factor_computation(
                'sentiment_score',
                sentiment_data['sentiment_score'],
                processing_time
            )
        else:
            logger.warning("sentiment_score column not found in data")
            return {}

        # 情感特有的质量检查
        sentiment_quality = {
            'timestamp': datetime.now().isoformat(),
            'base_quality': base_report,
            'sentiment_specific': {}
        }

        # 1. 新闻覆盖率检查
        if news_count_by_ticker:
            sentiment_quality['sentiment_specific']['news_coverage'] = self._analyze_news_coverage(news_count_by_ticker)

        # 2. 情感分布检查
        sentiment_quality['sentiment_specific']['sentiment_distribution'] = self._analyze_sentiment_distribution(sentiment_data['sentiment_score'])

        # 3. 横截面标准化验证
        sentiment_quality['sentiment_specific']['cross_sectional_standardization'] = self._verify_cross_sectional_standardization(sentiment_data)

        # 4. 时间序列一致性检查
        sentiment_quality['sentiment_specific']['temporal_consistency'] = self._check_temporal_consistency(sentiment_data)

        # 5. 计算情感因子专用评分
        sentiment_score = self._calculate_sentiment_quality_score(sentiment_quality)
        sentiment_quality['sentiment_quality_score'] = sentiment_score

        # 6. 生成报告和建议
        sentiment_quality['recommendations'] = self._generate_sentiment_recommendations(sentiment_quality)

        # 记录到系统
        self.factor_stats['sentiment_score_detailed'] = sentiment_quality

        # 输出专用日志
        self._log_sentiment_quality(sentiment_quality)

        return sentiment_quality

    def _analyze_news_coverage(self, news_count: Dict[str, int]) -> Dict[str, Any]:
        """分析新闻覆盖情况"""
        if not news_count:
            return {'status': 'NO_DATA', 'total_tickers': 0, 'tickers_with_news': 0}

        total_tickers = len(news_count)
        tickers_with_news = sum(1 for count in news_count.values() if count > 0)
        news_counts = list(news_count.values())

        return {
            'total_tickers': total_tickers,
            'tickers_with_news': tickers_with_news,
            'coverage_ratio': tickers_with_news / total_tickers if total_tickers > 0 else 0.0,
            'avg_news_per_ticker': np.mean(news_counts),
            'median_news_per_ticker': np.median(news_counts),
            'max_news_per_ticker': max(news_counts) if news_counts else 0,
            'min_news_per_ticker': min(news_counts) if news_counts else 0,
            'status': 'GOOD' if tickers_with_news / total_tickers > 0.7 else 'POOR'
        }

    def _analyze_sentiment_distribution(self, sentiment_scores: pd.Series) -> Dict[str, Any]:
        """分析情感分数分布特性"""
        clean_scores = sentiment_scores.dropna()

        if len(clean_scores) == 0:
            return {'status': 'NO_DATA'}

        # 分布特性
        distribution = {
            'range': {
                'min': float(clean_scores.min()),
                'max': float(clean_scores.max()),
                'span': float(clean_scores.max() - clean_scores.min())
            },
            'central_tendency': {
                'mean': float(clean_scores.mean()),
                'median': float(clean_scores.median()),
                'mode_exists': len(clean_scores.mode()) > 0
            },
            'spread': {
                'std': float(clean_scores.std()),
                'var': float(clean_scores.var()),
                'iqr': float(clean_scores.quantile(0.75) - clean_scores.quantile(0.25))
            },
            'shape': {
                'skewness': float(clean_scores.skew()),
                'kurtosis': float(clean_scores.kurtosis()),
                'is_normal_like': bool(abs(clean_scores.skew()) < 1 and abs(clean_scores.kurtosis()) < 3)
            }
        }

        # 情感特有检查
        distribution['sentiment_characteristics'] = {
            'has_positive_sentiment': bool((clean_scores > 0).any()),
            'has_negative_sentiment': bool((clean_scores < 0).any()),
            'positive_ratio': float((clean_scores > 0).mean()),
            'negative_ratio': float((clean_scores < 0).mean()),
            'neutral_ratio': float((clean_scores == 0).mean()),
            'extreme_positive_ratio': float((clean_scores > 2).mean()),
            'extreme_negative_ratio': float((clean_scores < -2).mean())
        }

        return distribution

    def _verify_cross_sectional_standardization(self, sentiment_data: pd.DataFrame) -> Dict[str, Any]:
        """验证横截面标准化是否正确"""
        if 'sentiment_score' not in sentiment_data.columns:
            return {'status': 'NO_SENTIMENT_COLUMN'}

        standardization_check = {
            'daily_means': [],
            'daily_stds': [],
            'dates_checked': 0,
            'standardization_quality': 'UNKNOWN'
        }

        try:
            # 检查每日的均值和标准差
            for date in sentiment_data.index.get_level_values('date').unique()[:10]:  # 检查前10天
                daily_scores = sentiment_data.loc[date, 'sentiment_score'].dropna()
                if len(daily_scores) > 1:
                    daily_mean = daily_scores.mean()
                    daily_std = daily_scores.std()
                    standardization_check['daily_means'].append(float(daily_mean))
                    standardization_check['daily_stds'].append(float(daily_std))
                    standardization_check['dates_checked'] += 1

            if standardization_check['daily_means']:
                avg_mean = np.mean(standardization_check['daily_means'])
                avg_std = np.mean(standardization_check['daily_stds'])

                # 评估标准化质量
                if abs(avg_mean) < 0.1 and abs(avg_std - 1.0) < 0.2:
                    standardization_check['standardization_quality'] = 'EXCELLENT'
                elif abs(avg_mean) < 0.5 and abs(avg_std - 1.0) < 0.5:
                    standardization_check['standardization_quality'] = 'GOOD'
                else:
                    standardization_check['standardization_quality'] = 'POOR'

                standardization_check['average_daily_mean'] = float(avg_mean)
                standardization_check['average_daily_std'] = float(avg_std)

        except Exception as e:
            logger.warning(f"Cross-sectional standardization check failed: {e}")
            standardization_check['error'] = str(e)

        return standardization_check

    def _check_temporal_consistency(self, sentiment_data: pd.DataFrame) -> Dict[str, Any]:
        """检查时间序列一致性"""
        if 'sentiment_score' not in sentiment_data.columns:
            return {'status': 'NO_SENTIMENT_COLUMN'}

        consistency = {
            'date_gaps': [],
            'ticker_coverage_consistency': {},
            'temporal_stability': 'UNKNOWN'
        }

        try:
            # 检查日期连续性
            dates = sorted(sentiment_data.index.get_level_values('date').unique())
            if len(dates) > 1:
                for i in range(1, len(dates)):
                    gap_days = (dates[i] - dates[i-1]).days
                    if gap_days > 1:  # 假设工作日
                        consistency['date_gaps'].append({
                            'from': str(dates[i-1]),
                            'to': str(dates[i]),
                            'gap_days': gap_days
                        })

            # 检查每日股票覆盖一致性
            daily_ticker_counts = sentiment_data.groupby(level='date').size()
            consistency['ticker_coverage_consistency'] = {
                'mean_tickers_per_day': float(daily_ticker_counts.mean()),
                'std_tickers_per_day': float(daily_ticker_counts.std()),
                'min_tickers_per_day': int(daily_ticker_counts.min()),
                'max_tickers_per_day': int(daily_ticker_counts.max()),
                'coverage_stability': 'STABLE' if daily_ticker_counts.std() / daily_ticker_counts.mean() < 0.1 else 'UNSTABLE'
            }

        except Exception as e:
            logger.warning(f"Temporal consistency check failed: {e}")
            consistency['error'] = str(e)

        return consistency

    def _calculate_sentiment_quality_score(self, sentiment_quality: Dict[str, Any]) -> Dict[str, Any]:
        """计算情感因子专用质量评分"""
        base_score = sentiment_quality['base_quality']['quality_score']['overall']

        # 情感特有评分组件
        sentiment_components = {
            'base_quality': base_score * 0.4,
            'news_coverage': 0,
            'sentiment_distribution': 0,
            'standardization': 0,
            'temporal_consistency': 0
        }

        # 新闻覆盖评分
        news_cov = sentiment_quality['sentiment_specific'].get('news_coverage', {})
        if news_cov.get('status') == 'GOOD':
            sentiment_components['news_coverage'] = 25
        elif news_cov.get('coverage_ratio', 0) > 0.5:
            sentiment_components['news_coverage'] = 15
        else:
            sentiment_components['news_coverage'] = 5

        # 标准化质量评分
        std_check = sentiment_quality['sentiment_specific'].get('cross_sectional_standardization', {})
        if std_check.get('standardization_quality') == 'EXCELLENT':
            sentiment_components['standardization'] = 20
        elif std_check.get('standardization_quality') == 'GOOD':
            sentiment_components['standardization'] = 15
        else:
            sentiment_components['standardization'] = 5

        # 时间一致性评分
        temp_check = sentiment_quality['sentiment_specific'].get('temporal_consistency', {})
        if temp_check.get('ticker_coverage_consistency', {}).get('coverage_stability') == 'STABLE':
            sentiment_components['temporal_consistency'] = 15
        else:
            sentiment_components['temporal_consistency'] = 8

        overall_sentiment_score = sum(sentiment_components.values())

        return {
            'overall': overall_sentiment_score,
            'components': sentiment_components,
            'grade': self._get_sentiment_grade(overall_sentiment_score)
        }

    def _get_sentiment_grade(self, score: float) -> str:
        """获取情感因子质量等级"""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B+'
        elif score >= 60:
            return 'B'
        elif score >= 50:
            return 'C'
        else:
            return 'D'

    def _generate_sentiment_recommendations(self, sentiment_quality: Dict[str, Any]) -> List[str]:
        """生成情感因子改进建议"""
        recommendations = []

        # 基于各项检查结果生成建议
        base_score = sentiment_quality['base_quality']['quality_score']['overall']
        if base_score < 70:
            recommendations.append("Consider improving base data quality (coverage, uniqueness)")

        news_cov = sentiment_quality['sentiment_specific'].get('news_coverage', {})
        if news_cov.get('coverage_ratio', 0) < 0.5:
            recommendations.append("Increase news coverage - consider expanding news sources or date range")

        std_check = sentiment_quality['sentiment_specific'].get('cross_sectional_standardization', {})
        if std_check.get('standardization_quality') != 'EXCELLENT':
            recommendations.append("Review cross-sectional standardization - daily means should be ~0, std ~1")

        sent_dist = sentiment_quality['sentiment_specific'].get('sentiment_distribution', {})
        if sent_dist.get('sentiment_characteristics', {}).get('neutral_ratio', 0) > 0.8:
            recommendations.append("High neutral sentiment ratio - verify news sentiment analysis is working")

        temp_check = sentiment_quality['sentiment_specific'].get('temporal_consistency', {})
        if temp_check.get('ticker_coverage_consistency', {}).get('coverage_stability') != 'STABLE':
            recommendations.append("Inconsistent daily ticker coverage - check data pipeline stability")

        if not recommendations:
            recommendations.append("Sentiment factor quality is good - maintain current pipeline")

        return recommendations

    def _log_sentiment_quality(self, sentiment_quality: Dict[str, Any]):
        """输出情感因子质量专用日志"""
        score = sentiment_quality['sentiment_quality_score']['overall']
        grade = sentiment_quality['sentiment_quality_score']['grade']

        # 获取关键指标
        base_score = sentiment_quality['base_quality']['quality_score']['overall']
        news_coverage = sentiment_quality['sentiment_specific'].get('news_coverage', {}).get('coverage_ratio', 0)
        std_quality = sentiment_quality['sentiment_specific'].get('cross_sectional_standardization', {}).get('standardization_quality', 'UNKNOWN')

        logger.info("=" * 80)
        logger.info(f"📰 SENTIMENT FACTOR QUALITY REPORT")
        logger.info("=" * 80)
        logger.info(f"Overall Sentiment Score: {score:.1f}/100 (Grade: {grade})")
        logger.info(f"Base Factor Quality: {base_score:.1f}/100")
        logger.info(f"News Coverage: {news_coverage:.1%}")
        logger.info(f"Standardization: {std_quality}")

        # 显示建议
        recommendations = sentiment_quality.get('recommendations', [])
        if recommendations:
            logger.info("\n📋 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")

        logger.info("=" * 80)

        # 保存详细报告
        if self.save_reports:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.report_dir, f"sentiment_quality_report_{timestamp}.json")
            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(sentiment_quality, f, ensure_ascii=False, indent=2)
                logger.info(f"📄 Detailed report saved: {report_file}")
            except Exception as e:
                logger.warning(f"Failed to save sentiment quality report: {e}")

    def _log_factor_quality(self, factor_name: str, report: Dict[str, Any]):
        """输出因子质量日志"""
        score = report['quality_score']['overall']
        coverage = report['coverage']['percentage']

        # 选择合适的emoji
        if score >= 90:
            emoji = "EXCELLENT"
            level = "EXCELLENT"
        elif score >= 70:
            emoji = "GOOD"
            level = "GOOD"
        elif score >= 50:
            emoji = "WARNING"
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