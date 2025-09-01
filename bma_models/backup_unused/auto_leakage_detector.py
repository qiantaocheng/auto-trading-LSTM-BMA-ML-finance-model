"""
自动泄露检测系统 - 实时监控数据泄露风险
Auto Leakage Detection System - Real-time monitoring for data leakage risks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

@dataclass
class LeakageDetectionConfig:
    """泄露检测配置"""
    # 时间间隔检测
    min_safe_gap_days: int = 15  # 更新为15天以支持T+10预测
    max_acceptable_gap_days: int = 30  # 最大合理间隔
    
    # 统计泄露检测
    max_ic_threshold: float = 0.8  # IC过高可能存在泄露
    min_pvalue_threshold: float = 0.001  # p值过低可能存在泄露
    
    # 前瞻偏差检测  
    future_info_tolerance: float = 0.05  # 允许的未来信息比例
    
    # 监控阈值
    continuous_high_ic_count: int = 5  # 连续高IC次数阈值
    anomaly_detection_window: int = 20  # 异常检测窗口
    
    # 严格模式
    strict_mode: bool = True  # 严格模式下更敏感的检测

@dataclass 
class LeakageAlert:
    """泄露警报"""
    timestamp: datetime
    alert_type: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    description: str
    affected_features: List[str]
    recommended_action: str
    metrics: Dict[str, float]

class AutoLeakageDetector:
    """自动泄露检测器"""
    
    def __init__(self, config: Optional[LeakageDetectionConfig] = None):
        self.config = config or LeakageDetectionConfig()
        self.alerts: List[LeakageAlert] = []
        self.detection_history: List[Dict] = []
        self.feature_monitoring: Dict[str, List[float]] = {}
        
        logger.info(f"初始化自动泄露检测器 - 最小安全间隔: {self.config.min_safe_gap_days}天")
    
    def detect_temporal_leakage(self, 
                               train_dates: pd.Series, 
                               test_dates: pd.Series,
                               feature_dates: Optional[pd.Series] = None) -> Dict[str, Any]:
        """检测时间泄露"""
        logger.debug("开始时间泄露检测...")
        
        result = {
            'has_leakage': False,
            'gap_days': 0,
            'alerts': [],
            'recommendations': []
        }
        
        try:
            # 计算实际时间间隔
            if len(train_dates) == 0 or len(test_dates) == 0:
                result['alerts'].append("训练集或测试集日期为空")
                return result
                
            train_max = train_dates.max()
            test_min = test_dates.min()
            
            if pd.isna(train_max) or pd.isna(test_min):
                result['alerts'].append("日期数据包含NaN值")
                return result
                
            gap_days = (test_min - train_max).days
            result['gap_days'] = gap_days
            
            # 检测时间间隔是否足够
            if gap_days < self.config.min_safe_gap_days:
                result['has_leakage'] = True
                severity = 'CRITICAL' if gap_days <= 0 else 'HIGH'
                
                alert = LeakageAlert(
                    timestamp=datetime.now(),
                    alert_type='TEMPORAL_LEAKAGE',
                    severity=severity,
                    description=f"时间间隔不足: {gap_days}天 < {self.config.min_safe_gap_days}天",
                    affected_features=[],
                    recommended_action=f"增加时间间隔至{self.config.min_safe_gap_days}天以上",
                    metrics={'gap_days': gap_days, 'required_gap': self.config.min_safe_gap_days}
                )
                self.alerts.append(alert)
                result['alerts'].append(alert.description)
                
            # 检测特征日期泄露
            if feature_dates is not None and len(feature_dates) > 0:
                feature_leakage = self._detect_feature_date_leakage(feature_dates, test_dates)
                if feature_leakage['has_leakage']:
                    result['has_leakage'] = True
                    result['alerts'].extend(feature_leakage['alerts'])
                    
            logger.info(f"时间泄露检测完成: gap={gap_days}天, 泄露={'是' if result['has_leakage'] else '否'}")
            
        except Exception as e:
            logger.error(f"时间泄露检测出错: {e}")
            result['alerts'].append(f"检测失败: {e}")
            
        return result
    
    def detect_statistical_leakage(self, 
                                  predictions: pd.Series, 
                                  actuals: pd.Series,
                                  feature_name: str = "unknown") -> Dict[str, Any]:
        """检测统计泄露 - 通过异常高的预测性能识别"""
        logger.debug(f"开始统计泄露检测: {feature_name}")
        
        result = {
            'has_leakage': False,
            'ic': 0.0,
            'pvalue': 1.0,
            'alerts': [],
            'severity': 'LOW'
        }
        
        try:
            # 数据对齐和清理
            aligned_pred, aligned_actual = self._align_and_clean_data(predictions, actuals)
            
            if len(aligned_pred) < 10:
                result['alerts'].append(f"样本数量不足: {len(aligned_pred)} < 10")
                return result
            
            # 计算IC和p值
            ic, pvalue = stats.spearmanr(aligned_pred, aligned_actual)
            result['ic'] = ic if not np.isnan(ic) else 0.0
            result['pvalue'] = pvalue if not np.isnan(pvalue) else 1.0
            
            # 更新特征监控历史
            if feature_name not in self.feature_monitoring:
                self.feature_monitoring[feature_name] = []
            self.feature_monitoring[feature_name].append(result['ic'])
            
            # 检测异常高IC
            if abs(result['ic']) > self.config.max_ic_threshold:
                result['has_leakage'] = True
                result['severity'] = 'HIGH'
                
                alert = LeakageAlert(
                    timestamp=datetime.now(),
                    alert_type='STATISTICAL_LEAKAGE',
                    severity='HIGH',
                    description=f"异常高IC: {result['ic']:.4f} > {self.config.max_ic_threshold}",
                    affected_features=[feature_name],
                    recommended_action="检查特征构建逻辑，确认无未来信息泄露",
                    metrics={'ic': result['ic'], 'pvalue': result['pvalue']}
                )
                self.alerts.append(alert)
                result['alerts'].append(alert.description)
            
            # 检测异常低p值
            elif result['pvalue'] < self.config.min_pvalue_threshold:
                result['has_leakage'] = True
                result['severity'] = 'MEDIUM'
                result['alerts'].append(f"p值异常低: {result['pvalue']:.6f}")
            
            # 检测连续异常IC
            continuous_alert = self._check_continuous_anomaly(feature_name)
            if continuous_alert:
                result['has_leakage'] = True
                result['alerts'].extend(continuous_alert)
                
        except Exception as e:
            logger.error(f"统计泄露检测出错: {e}")
            result['alerts'].append(f"检测失败: {e}")
            
        return result
    
    def detect_feature_leakage(self, 
                             features: pd.DataFrame, 
                             target_dates: pd.Series,
                             feature_creation_logic: Optional[Dict] = None) -> Dict[str, Any]:
        """检测特征泄露 - 分析特征是否包含未来信息"""
        logger.debug("开始特征泄露检测...")
        
        result = {
            'has_leakage': False,
            'leaked_features': [],
            'alerts': [],
            'feature_analysis': {}
        }
        
        try:
            for col in features.columns:
                col_analysis = self._analyze_feature_column(features[col], target_dates, col)
                result['feature_analysis'][col] = col_analysis
                
                if col_analysis['suspected_leakage']:
                    result['has_leakage'] = True
                    result['leaked_features'].append(col)
                    result['alerts'].append(f"特征'{col}': {col_analysis['reason']}")
                    
        except Exception as e:
            logger.error(f"特征泄露检测出错: {e}")
            result['alerts'].append(f"检测失败: {e}")
            
        return result
    
    def comprehensive_leakage_scan(self, 
                                  train_data: pd.DataFrame,
                                  test_data: pd.DataFrame,
                                  predictions: pd.Series,
                                  actuals: pd.Series) -> Dict[str, Any]:
        """综合泄露扫描"""
        logger.info("开始综合泄露扫描...")
        
        scan_result = {
            'overall_risk': 'LOW',
            'total_alerts': 0,
            'temporal_check': {},
            'statistical_check': {},
            'feature_check': {},
            'recommendations': [],
            'critical_issues': []
        }
        
        try:
            # 1. 时间泄露检测
            if 'date' in train_data.columns and 'date' in test_data.columns:
                scan_result['temporal_check'] = self.detect_temporal_leakage(
                    train_data['date'], test_data['date']
                )
            
            # 2. 统计泄露检测
            scan_result['statistical_check'] = self.detect_statistical_leakage(
                predictions, actuals, "model_predictions"
            )
            
            # 3. 特征泄露检测
            feature_cols = [col for col in train_data.columns if col != 'date']
            if feature_cols:
                feature_subset = train_data[feature_cols[:10]]  # 限制前10个特征避免过长
                target_dates = test_data.get('date', pd.Series())
                scan_result['feature_check'] = self.detect_feature_leakage(
                    feature_subset, target_dates
                )
            
            # 综合风险评估
            scan_result = self._assess_overall_risk(scan_result)
            
            logger.info(f"综合泄露扫描完成 - 总体风险: {scan_result['overall_risk']}")
            
        except Exception as e:
            logger.error(f"综合泄露扫描出错: {e}")
            scan_result['critical_issues'].append(f"扫描失败: {e}")
            scan_result['overall_risk'] = 'UNKNOWN'
            
        return scan_result
    
    def get_leakage_report(self) -> str:
        """生成泄露检测报告"""
        report = ["=== 自动泄露检测报告 ===\n"]
        
        if not self.alerts:
            report.append("[OK] 未发现数据泄露风险\n")
        else:
            report.append(f"[ALERT] 发现 {len(self.alerts)} 个潜在问题:\n")
            
            for i, alert in enumerate(self.alerts[-10:], 1):  # 显示最近10个警报
                report.append(f"{i}. [{alert.severity}] {alert.alert_type}")
                report.append(f"   描述: {alert.description}")
                report.append(f"   建议: {alert.recommended_action}")
                report.append("")
        
        # 添加监控统计
        if self.feature_monitoring:
            report.append("=== 特征监控统计 ===")
            for feature, ics in self.feature_monitoring.items():
                recent_ic = ics[-1] if ics else 0
                avg_ic = np.mean(ics) if ics else 0
                report.append(f"{feature}: 最近IC={recent_ic:.4f}, 平均IC={avg_ic:.4f}")
        
        return "\n".join(report)
    
    # 辅助方法
    def _detect_feature_date_leakage(self, feature_dates: pd.Series, test_dates: pd.Series) -> Dict:
        """检测特征日期泄露"""
        result = {'has_leakage': False, 'alerts': []}
        
        try:
            if len(feature_dates) == 0 or len(test_dates) == 0:
                return result
                
            # 检查是否有特征日期晚于测试日期
            test_min = test_dates.min()
            future_features = feature_dates[feature_dates >= test_min]
            
            if len(future_features) > 0:
                future_ratio = len(future_features) / len(feature_dates)
                if future_ratio > self.config.future_info_tolerance:
                    result['has_leakage'] = True
                    result['alerts'].append(f"特征包含{future_ratio:.2%}的未来信息")
                    
        except Exception as e:
            logger.warning(f"特征日期泄露检测出错: {e}")
            
        return result
    
    def _align_and_clean_data(self, predictions: pd.Series, actuals: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """对齐和清理数据"""
        # 找到共同索引
        common_idx = predictions.index.intersection(actuals.index)
        
        if len(common_idx) == 0:
            return np.array([]), np.array([])
        
        aligned_pred = predictions.loc[common_idx].dropna()
        aligned_actual = actuals.loc[common_idx].dropna()
        
        # 再次找到共同的非NaN索引
        final_common_idx = aligned_pred.index.intersection(aligned_actual.index)
        
        return (aligned_pred.loc[final_common_idx].values, 
                aligned_actual.loc[final_common_idx].values)
    
    def _check_continuous_anomaly(self, feature_name: str) -> List[str]:
        """检查连续异常"""
        alerts = []
        
        if feature_name in self.feature_monitoring:
            recent_ics = self.feature_monitoring[feature_name][-self.config.continuous_high_ic_count:]
            
            if len(recent_ics) >= self.config.continuous_high_ic_count:
                high_ic_count = sum(1 for ic in recent_ics if abs(ic) > self.config.max_ic_threshold * 0.8)
                
                if high_ic_count >= self.config.continuous_high_ic_count * 0.8:
                    alerts.append(f"连续高IC警报: {feature_name}")
                    
        return alerts
    
    def _analyze_feature_column(self, feature: pd.Series, target_dates: pd.Series, col_name: str) -> Dict:
        """分析特征列是否存在泄露"""
        analysis = {
            'suspected_leakage': False,
            'reason': '',
            'confidence': 0.0
        }
        
        try:
            # 检查特征值的时间相关性
            if len(target_dates) > 0 and len(feature) > 0:
                # 简单的统计检查：异常高的方差或偏度
                feature_clean = feature.dropna()
                if len(feature_clean) > 10:
                    skewness = abs(stats.skew(feature_clean))
                    if skewness > 3:  # 极度偏斜可能是泄露信号
                        analysis['suspected_leakage'] = True
                        analysis['reason'] = f"特征分布极度偏斜 (偏度={skewness:.2f})"
                        analysis['confidence'] = min(0.7, skewness / 5)
                        
        except Exception as e:
            logger.debug(f"特征分析出错 {col_name}: {e}")
            
        return analysis
    
    def _assess_overall_risk(self, scan_result: Dict) -> Dict:
        """评估总体风险"""
        critical_count = 0
        high_count = 0
        medium_count = 0
        
        # 统计不同严重程度的问题
        for check_name, check_result in scan_result.items():
            if isinstance(check_result, dict) and check_result.get('has_leakage'):
                if 'CRITICAL' in str(check_result.get('alerts', [])):
                    critical_count += 1
                elif 'HIGH' in str(check_result.get('alerts', [])):
                    high_count += 1
                else:
                    medium_count += 1
        
        scan_result['total_alerts'] = critical_count + high_count + medium_count
        
        # 确定总体风险级别
        if critical_count > 0:
            scan_result['overall_risk'] = 'CRITICAL'
            scan_result['recommendations'].append("立即停止模型使用，修复严重泄露问题")
        elif high_count > 0:
            scan_result['overall_risk'] = 'HIGH'  
            scan_result['recommendations'].append("建议暂停模型使用，调查高风险泄露")
        elif medium_count > 0:
            scan_result['overall_risk'] = 'MEDIUM'
            scan_result['recommendations'].append("密切监控，考虑调整参数")
        else:
            scan_result['overall_risk'] = 'LOW'
            scan_result['recommendations'].append("继续监控，定期检查")
            
        return scan_result