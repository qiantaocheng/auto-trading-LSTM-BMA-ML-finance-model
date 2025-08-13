#!/usr/bin/env python3
"""
模型实时监控和异常检测系统
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """告警信息"""
    timestamp: datetime
    level: str  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component: str
    message: str
    metrics: Dict[str, float]
    suggested_action: str

@dataclass
class ModelMetrics:
    """模型指标"""
    timestamp: datetime
    prediction_count: int
    prediction_mean: float
    prediction_std: float
    prediction_range: Tuple[float, float]
    feature_drift_score: float
    prediction_drift_score: float
    outlier_ratio: float
    processing_time_ms: float

class ModelMonitor:
    """模型实时监控系统"""
    
    def __init__(self, window_size: int = 1000, alert_thresholds: Optional[Dict] = None):
        """
        初始化监控系统
        
        Args:
            window_size: 滑动窗口大小
            alert_thresholds: 告警阈值配置
        """
        self.window_size = window_size
        
        # 默认告警阈值
        self.alert_thresholds = {
            'prediction_drift_high': 0.3,
            'prediction_drift_critical': 0.5,
            'feature_drift_high': 0.2,
            'feature_drift_critical': 0.4,
            'outlier_ratio_high': 0.15,
            'outlier_ratio_critical': 0.25,
            'processing_time_high': 5000,  # ms
            'processing_time_critical': 10000,  # ms
            'prediction_std_low': 0.001,  # 预测方差过低
            'prediction_std_high': 1.0    # 预测方差过高
        }
        
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)
        
        # 历史数据存储
        self.prediction_history = deque(maxlen=window_size)
        self.feature_history = deque(maxlen=window_size)
        self.metrics_history = deque(maxlen=window_size) 
        self.alerts = deque(maxlen=1000)
        
        # 基线统计（用于漂移检测）
        self.baseline_stats = {
            'prediction_mean': None,
            'prediction_std': None,
            'feature_means': None,
            'feature_stds': None
        }
        
        # 异常检测模型
        self.outlier_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.outlier_detector_fitted = False
        
        # 性能计数器
        self.counters = {
            'total_predictions': 0,
            'total_alerts': 0,
            'drift_detections': 0,
            'outlier_detections': 0,
            'processing_errors': 0
        }
        
        logger.info("模型监控系统初始化完成")
    
    def set_baseline(self, baseline_predictions: np.ndarray, 
                     baseline_features: Optional[np.ndarray] = None):
        """
        设置基线统计
        
        Args:
            baseline_predictions: 基线预测值
            baseline_features: 基线特征值
        """
        self.baseline_stats['prediction_mean'] = np.mean(baseline_predictions)
        self.baseline_stats['prediction_std'] = np.std(baseline_predictions)
        
        if baseline_features is not None:
            self.baseline_stats['feature_means'] = np.mean(baseline_features, axis=0)
            self.baseline_stats['feature_stds'] = np.std(baseline_features, axis=0)
            
            # 训练异常检测模型
            try:
                self.outlier_detector.fit(baseline_features)
                self.outlier_detector_fitted = True
                logger.info("异常检测模型训练完成")
            except Exception as e:
                logger.warning(f"异常检测模型训练失败: {e}")
        
        logger.info("基线统计设置完成")
    
    def detect_prediction_drift(self, current_predictions: np.ndarray) -> float:
        """
        检测预测漂移
        
        Args:
            current_predictions: 当前预测值
            
        Returns:
            漂移分数 (0-1, 越高越严重)
        """
        if (self.baseline_stats['prediction_mean'] is None or 
            self.baseline_stats['prediction_std'] is None):
            return 0.0
        
        try:
            # 使用Kolmogorov-Smirnov检验
            baseline_samples = np.random.normal(
                self.baseline_stats['prediction_mean'],
                self.baseline_stats['prediction_std'],
                len(current_predictions)
            )
            
            ks_stat, p_value = stats.ks_2samp(baseline_samples, current_predictions)
            
            # 转换为0-1范围的漂移分数
            drift_score = min(ks_stat * 2, 1.0)  # KS统计量通常0-0.5
            
            return drift_score
            
        except Exception as e:
            logger.warning(f"预测漂移检测失败: {e}")
            return 0.0
    
    def detect_feature_drift(self, current_features: np.ndarray) -> float:
        """
        检测特征漂移
        
        Args:
            current_features: 当前特征值
            
        Returns:
            漂移分数 (0-1, 越高越严重)
        """
        if (self.baseline_stats['feature_means'] is None or 
            self.baseline_stats['feature_stds'] is None):
            return 0.0
        
        try:
            current_means = np.mean(current_features, axis=0)
            baseline_means = self.baseline_stats['feature_means']
            baseline_stds = self.baseline_stats['feature_stds']
            
            # 计算标准化差异
            normalized_diffs = np.abs(current_means - baseline_means) / (baseline_stds + 1e-8)
            
            # 平均漂移分数
            drift_score = np.mean(np.clip(normalized_diffs / 3, 0, 1))  # 3个标准差为最大
            
            return drift_score
            
        except Exception as e:
            logger.warning(f"特征漂移检测失败: {e}")
            return 0.0
    
    def detect_outliers(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        检测异常值
        
        Args:
            features: 特征数据
            
        Returns:
            (outlier_mask, outlier_ratio) 异常值掩码和比例
        """
        if not self.outlier_detector_fitted:
            return np.zeros(len(features), dtype=bool), 0.0
        
        try:
            outlier_scores = self.outlier_detector.decision_function(features)
            outlier_mask = self.outlier_detector.predict(features) == -1
            outlier_ratio = np.mean(outlier_mask)
            
            return outlier_mask, outlier_ratio
            
        except Exception as e:
            logger.warning(f"异常值检测失败: {e}")
            return np.zeros(len(features), dtype=bool), 0.0
    
    def monitor_predictions(self, predictions: np.ndarray, 
                          features: Optional[np.ndarray] = None,
                          processing_time_ms: float = 0.0) -> List[Alert]:
        """
        监控预测结果并生成告警
        
        Args:
            predictions: 预测值
            features: 特征值（可选）
            processing_time_ms: 处理时间（毫秒）
            
        Returns:
            告警列表
        """
        start_time = datetime.now()
        alerts = []
        
        try:
            # 更新计数器
            self.counters['total_predictions'] += len(predictions)
            
            # 基本统计
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            pred_min, pred_max = np.min(predictions), np.max(predictions)
            
            # 漂移检测
            prediction_drift = self.detect_prediction_drift(predictions)
            feature_drift = 0.0
            outlier_ratio = 0.0
            
            if features is not None:
                feature_drift = self.detect_feature_drift(features)
                _, outlier_ratio = self.detect_outliers(features)
                self.feature_history.extend(features.tolist())
            
            # 记录历史
            self.prediction_history.extend(predictions.tolist())
            
            # 创建指标对象
            metrics = ModelMetrics(
                timestamp=start_time,
                prediction_count=len(predictions),
                prediction_mean=pred_mean,
                prediction_std=pred_std,
                prediction_range=(pred_min, pred_max),
                feature_drift_score=feature_drift,
                prediction_drift_score=prediction_drift,
                outlier_ratio=outlier_ratio,
                processing_time_ms=processing_time_ms
            )
            
            self.metrics_history.append(metrics)
            
            # 生成告警
            alerts.extend(self._check_drift_alerts(prediction_drift, feature_drift))
            alerts.extend(self._check_outlier_alerts(outlier_ratio))
            alerts.extend(self._check_performance_alerts(pred_std, processing_time_ms))
            alerts.extend(self._check_distribution_alerts(predictions))
            
            # 记录告警
            for alert in alerts:
                self.alerts.append(alert)
                self.counters['total_alerts'] += 1
            
            return alerts
            
        except Exception as e:
            self.counters['processing_errors'] += 1
            error_alert = Alert(
                timestamp=datetime.now(),
                level='ERROR',
                component='ModelMonitor',
                message=f"监控处理失败: {e}",
                metrics={'error_count': self.counters['processing_errors']},
                suggested_action='检查监控系统配置和输入数据'
            )
            return [error_alert]
    
    def _check_drift_alerts(self, prediction_drift: float, feature_drift: float) -> List[Alert]:
        """检查漂移告警"""
        alerts = []
        
        # 预测漂移告警
        if prediction_drift >= self.alert_thresholds['prediction_drift_critical']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level='CRITICAL',
                component='PredictionDrift',
                message=f'严重预测漂移检测: {prediction_drift:.3f}',
                metrics={'drift_score': prediction_drift},
                suggested_action='立即重新训练模型或检查数据质量'
            ))
            self.counters['drift_detections'] += 1
        elif prediction_drift >= self.alert_thresholds['prediction_drift_high']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level='WARNING',
                component='PredictionDrift',
                message=f'预测漂移警告: {prediction_drift:.3f}',
                metrics={'drift_score': prediction_drift},
                suggested_action='考虑重新训练模型'
            ))
        
        # 特征漂移告警
        if feature_drift >= self.alert_thresholds['feature_drift_critical']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level='CRITICAL',
                component='FeatureDrift',
                message=f'严重特征漂移检测: {feature_drift:.3f}',
                metrics={'drift_score': feature_drift},
                suggested_action='检查数据源和特征工程流程'
            ))
        elif feature_drift >= self.alert_thresholds['feature_drift_high']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level='WARNING',
                component='FeatureDrift',
                message=f'特征漂移警告: {feature_drift:.3f}',
                metrics={'drift_score': feature_drift},
                suggested_action='监控特征分布变化'
            ))
        
        return alerts
    
    def _check_outlier_alerts(self, outlier_ratio: float) -> List[Alert]:
        """检查异常值告警"""
        alerts = []
        
        if outlier_ratio >= self.alert_thresholds['outlier_ratio_critical']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level='CRITICAL',
                component='OutlierDetection',
                message=f'严重异常值比例: {outlier_ratio:.3f}',
                metrics={'outlier_ratio': outlier_ratio},
                suggested_action='检查数据质量和预处理流程'
            ))
            self.counters['outlier_detections'] += 1
        elif outlier_ratio >= self.alert_thresholds['outlier_ratio_high']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level='WARNING',
                component='OutlierDetection',
                message=f'异常值比例较高: {outlier_ratio:.3f}',
                metrics={'outlier_ratio': outlier_ratio},
                suggested_action='监控数据异常'
            ))
        
        return alerts
    
    def _check_performance_alerts(self, pred_std: float, processing_time: float) -> List[Alert]:
        """检查性能告警"""
        alerts = []
        
        # 处理时间告警
        if processing_time >= self.alert_thresholds['processing_time_critical']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level='CRITICAL',
                component='Performance',
                message=f'处理时间过长: {processing_time:.1f}ms',
                metrics={'processing_time_ms': processing_time},
                suggested_action='优化模型或增加计算资源'
            ))
        elif processing_time >= self.alert_thresholds['processing_time_high']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level='WARNING',
                component='Performance',
                message=f'处理时间较长: {processing_time:.1f}ms',
                metrics={'processing_time_ms': processing_time},
                suggested_action='监控系统性能'
            ))
        
        # 预测方差告警
        if pred_std <= self.alert_thresholds['prediction_std_low']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level='WARNING',
                component='PredictionVariance',
                message=f'预测方差过低: {pred_std:.4f}',
                metrics={'prediction_std': pred_std},
                suggested_action='检查模型是否过度平滑或缺乏多样性'
            ))
        elif pred_std >= self.alert_thresholds['prediction_std_high']:
            alerts.append(Alert(
                timestamp=datetime.now(),
                level='WARNING',
                component='PredictionVariance',
                message=f'预测方差过高: {pred_std:.4f}',
                metrics={'prediction_std': pred_std},
                suggested_action='检查模型稳定性和输入数据质量'
            ))
        
        return alerts
    
    def _check_distribution_alerts(self, predictions: np.ndarray) -> List[Alert]:
        """检查分布异常告警"""
        alerts = []
        
        try:
            # 检查是否有NaN或无穷值
            nan_count = np.isnan(predictions).sum()
            inf_count = np.isinf(predictions).sum()
            
            if nan_count > 0:
                alerts.append(Alert(
                    timestamp=datetime.now(),
                    level='ERROR',
                    component='PredictionQuality',
                    message=f'预测包含NaN值: {nan_count}个',
                    metrics={'nan_count': nan_count},
                    suggested_action='检查模型输入和计算流程'
                ))
            
            if inf_count > 0:
                alerts.append(Alert(
                    timestamp=datetime.now(),
                    level='ERROR',
                    component='PredictionQuality',
                    message=f'预测包含无穷值: {inf_count}个',
                    metrics={'inf_count': inf_count},
                    suggested_action='检查数值计算稳定性'
                ))
                
        except Exception as e:
            logger.warning(f"分布检查失败: {e}")
        
        return alerts
    
    def get_recent_alerts(self, hours: int = 24, level: Optional[str] = None) -> List[Alert]:
        """获取最近的告警"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alerts 
            if alert.timestamp >= cutoff_time
        ]
        
        if level:
            recent_alerts = [
                alert for alert in recent_alerts
                if alert.level == level
            ]
        
        return sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控总结"""
        recent_metrics = list(self.metrics_history)[-100:] if self.metrics_history else []
        recent_alerts = self.get_recent_alerts(hours=24)
        
        summary = {
            'counters': self.counters.copy(),
            'recent_24h_alerts': len(recent_alerts),
            'alert_breakdown': {},
            'current_health': 'HEALTHY',
            'recommendations': []
        }
        
        # 告警分类统计
        for alert in recent_alerts:
            summary['alert_breakdown'][alert.level] = summary['alert_breakdown'].get(alert.level, 0) + 1
        
        # 健康状态评估
        critical_alerts = summary['alert_breakdown'].get('CRITICAL', 0)
        warning_alerts = summary['alert_breakdown'].get('WARNING', 0)
        
        if critical_alerts > 0:
            summary['current_health'] = 'CRITICAL'
            summary['recommendations'].append('存在严重问题，需要立即处理')
        elif warning_alerts > 5:
            summary['current_health'] = 'WARNING'
            summary['recommendations'].append('存在多个警告，建议检查系统状态')
        elif warning_alerts > 0:
            summary['current_health'] = 'CAUTION'
            summary['recommendations'].append('存在警告，建议持续监控')
        
        # 性能统计
        if recent_metrics:
            summary['avg_processing_time'] = np.mean([m.processing_time_ms for m in recent_metrics])
            summary['avg_prediction_std'] = np.mean([m.prediction_std for m in recent_metrics])
            summary['avg_drift_score'] = np.mean([m.prediction_drift_score for m in recent_metrics])
        
        return summary
    
    def export_monitoring_data(self, output_file: str):
        """导出监控数据"""
        data = {
            'metrics_history': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'prediction_count': m.prediction_count,
                    'prediction_mean': m.prediction_mean,
                    'prediction_std': m.prediction_std,
                    'feature_drift_score': m.feature_drift_score,
                    'prediction_drift_score': m.prediction_drift_score,
                    'outlier_ratio': m.outlier_ratio,
                    'processing_time_ms': m.processing_time_ms
                }
                for m in self.metrics_history
            ],
            'alerts_history': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'level': a.level,
                    'component': a.component,
                    'message': a.message,
                    'metrics': a.metrics,
                    'suggested_action': a.suggested_action
                }
                for a in self.alerts
            ],
            'counters': self.counters,
            'export_timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"监控数据已导出到: {output_file}")


def example_usage():
    """示例用法"""
    print("🔍 模型监控系统示例")
    
    # 初始化监控系统
    monitor = ModelMonitor(window_size=500)
    
    # 设置基线（模拟历史数据）
    baseline_predictions = np.random.normal(0, 0.1, 1000)
    baseline_features = np.random.randn(1000, 10)
    
    monitor.set_baseline(baseline_predictions, baseline_features)
    
    # 模拟正常预测
    print("\n📈 监控正常预测...")
    normal_predictions = np.random.normal(0.02, 0.12, 100)  # 轻微漂移
    normal_features = np.random.randn(100, 10) + 0.1        # 轻微特征漂移
    
    alerts = monitor.monitor_predictions(
        predictions=normal_predictions,
        features=normal_features,
        processing_time_ms=50
    )
    
    print(f"正常预测告警数: {len(alerts)}")
    for alert in alerts:
        print(f"  {alert.level}: {alert.message}")
    
    # 模拟异常预测
    print("\n🚨 监控异常预测...")
    abnormal_predictions = np.random.normal(0.5, 0.3, 100)  # 严重漂移
    abnormal_features = np.random.randn(100, 10) + 2        # 严重特征漂移
    
    alerts = monitor.monitor_predictions(
        predictions=abnormal_predictions,
        features=abnormal_features,
        processing_time_ms=8000  # 处理时间过长
    )
    
    print(f"异常预测告警数: {len(alerts)}")
    for alert in alerts:
        print(f"  {alert.level}: {alert.message}")
    
    # 获取监控总结
    print("\n📊 监控总结:")
    summary = monitor.get_monitoring_summary()
    print(f"  系统健康状态: {summary['current_health']}")
    print(f"  24小时告警数: {summary['recent_24h_alerts']}")
    print(f"  总预测数: {summary['counters']['total_predictions']}")
    
    return monitor


if __name__ == "__main__":
    example_usage()
