#!/usr/bin/env python3
"""
Knowledge Retention System for BMA Enhanced
Implements feature importance monitoring, model distillation, and knowledge transfer
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import pickle
import json
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings

logger = logging.getLogger(__name__)

class DriftSeverity(Enum):
    """Drift severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class KnowledgeRetentionConfig:
    """Configuration for knowledge retention system"""
    # Feature importance monitoring
    importance_history_window: int = 10  # Number of recent models to compare
    kl_divergence_threshold: float = 0.3  # KL divergence threshold for drift
    js_distance_threshold: float = 0.2   # Jensen-Shannon distance threshold
    
    # Stability monitoring
    rank_correlation_threshold: float = 0.6  # Minimum rank correlation for stability
    top_features_count: int = 20  # Number of top features to track
    
    # Drift detection
    drift_detection_window: int = 5  # Rolling window for drift detection
    consecutive_drift_threshold: int = 3  # Consecutive periods for drift alert
    
    # Model distillation
    enable_model_distillation: bool = True
    distillation_temperature: float = 3.0  # Temperature for knowledge distillation
    student_model_complexity: float = 0.7  # Relative complexity of student model
    
    # Knowledge transfer
    enable_transfer_learning: bool = True
    transfer_learning_weight: float = 0.3  # Weight for transferred knowledge
    min_transfer_similarity: float = 0.5   # Minimum similarity for knowledge transfer
    
    # Storage settings
    max_history_retention: int = 50  # Maximum number of historical snapshots
    cache_compression: bool = True   # Enable compression for storage

@dataclass
class FeatureImportanceSnapshot:
    """Snapshot of feature importance at a point in time"""
    timestamp: datetime
    importance_dict: Dict[str, float]
    model_type: str
    model_performance: Dict[str, float]
    feature_count: int
    model_hash: str = ""

@dataclass
class DriftAlert:
    """Drift detection alert"""
    timestamp: datetime
    drift_type: str
    severity: DriftSeverity
    metric_value: float
    threshold: float
    affected_features: List[str]
    recommendation: str

class KnowledgeRetentionSystem:
    """
    Knowledge Retention System
    
    Key features:
    1. Feature importance tracking and drift detection
    2. Model distillation for knowledge transfer
    3. Performance degradation monitoring
    4. Automated alerting and recommendations
    5. Historical knowledge preservation
    """
    
    def __init__(self, config: KnowledgeRetentionConfig = None, cache_dir: str = "cache/knowledge_retention"):
        self.config = config or KnowledgeRetentionConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for different model types
        self.importance_history = {
            'lightgbm': [],
            'bma': [],
            'ensemble': []
        }
        
        # Drift detection state
        self.drift_alerts = []
        self.drift_detection_state = {}
        
        # Knowledge distillation artifacts
        self.teacher_models = {}
        self.distillation_history = []
        
        logger.info("KnowledgeRetentionSystem initialized")
        logger.info(f"Config: KL threshold={self.config.kl_divergence_threshold}, "
                   f"History window={self.config.importance_history_window}")
    
    def record_feature_importance(self, importance_dict: Dict[str, float],
                                model_type: str,
                                model_performance: Dict[str, float] = None,
                                model_hash: str = "",
                                additional_metadata: Dict[str, Any] = None) -> None:
        """Record feature importance snapshot"""
        
        if not importance_dict:
            logger.warning(f"Empty importance dictionary for {model_type}")
            return
        
        # Create snapshot
        snapshot = FeatureImportanceSnapshot(
            timestamp=datetime.now(),
            importance_dict=importance_dict.copy(),
            model_type=model_type,
            model_performance=model_performance or {},
            feature_count=len(importance_dict),
            model_hash=model_hash
        )
        
        # Add to history
        if model_type not in self.importance_history:
            self.importance_history[model_type] = []
        
        self.importance_history[model_type].append(snapshot)
        
        # Maintain history window
        if len(self.importance_history[model_type]) > self.config.max_history_retention:
            self.importance_history[model_type] = self.importance_history[model_type][-self.config.max_history_retention:]
        
        # Perform drift detection
        drift_analysis = self.detect_feature_drift(model_type)
        
        if drift_analysis['alerts']:
            for alert in drift_analysis['alerts']:
                self.drift_alerts.append(alert)
                self._log_drift_alert(alert)
        
        # Save snapshot
        self._save_snapshot(snapshot, additional_metadata)
        
        logger.info(f"Recorded feature importance for {model_type}: {len(importance_dict)} features")
    
    def detect_feature_drift(self, model_type: str) -> Dict[str, Any]:
        """Detect feature importance drift using multiple metrics"""
        
        history = self.importance_history.get(model_type, [])
        if len(history) < 2:
            return {'status': 'insufficient_history', 'alerts': []}
        
        # Get recent snapshots for comparison
        current_snapshot = history[-1]
        comparison_snapshots = history[-self.config.importance_history_window:]
        
        alerts = []
        drift_metrics = {}
        
        # Compare with each historical snapshot
        for i, historical_snapshot in enumerate(comparison_snapshots[:-1]):  # Exclude current
            drift_result = self._compare_importance_distributions(
                current_snapshot, historical_snapshot
            )
            
            # Store metrics
            periods_back = len(comparison_snapshots) - 1 - i
            drift_metrics[f'periods_back_{periods_back}'] = drift_result
            
            # Check for significant drift
            if drift_result['kl_divergence'] > self.config.kl_divergence_threshold:
                severity = self._determine_drift_severity(drift_result['kl_divergence'], self.config.kl_divergence_threshold)
                
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    drift_type='feature_importance_kl',
                    severity=severity,
                    metric_value=drift_result['kl_divergence'],
                    threshold=self.config.kl_divergence_threshold,
                    affected_features=drift_result['top_changed_features'],
                    recommendation=self._get_drift_recommendation(severity, 'kl_divergence')
                )
                alerts.append(alert)
            
            # Check Jensen-Shannon distance
            if drift_result['js_distance'] > self.config.js_distance_threshold:
                severity = self._determine_drift_severity(drift_result['js_distance'], self.config.js_distance_threshold)
                
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    drift_type='feature_importance_js',
                    severity=severity,
                    metric_value=drift_result['js_distance'],
                    threshold=self.config.js_distance_threshold,
                    affected_features=drift_result['top_changed_features'],
                    recommendation=self._get_drift_recommendation(severity, 'js_distance')
                )
                alerts.append(alert)
            
            # Check rank correlation
            if drift_result['rank_correlation'] < self.config.rank_correlation_threshold:
                severity = DriftSeverity.HIGH if drift_result['rank_correlation'] < 0.3 else DriftSeverity.MEDIUM
                
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    drift_type='feature_ranking',
                    severity=severity,
                    metric_value=drift_result['rank_correlation'],
                    threshold=self.config.rank_correlation_threshold,
                    affected_features=drift_result['top_changed_features'],
                    recommendation=self._get_drift_recommendation(severity, 'rank_correlation')
                )
                alerts.append(alert)
        
        # Update drift detection state
        self.drift_detection_state[model_type] = {
            'last_check': datetime.now(),
            'drift_metrics': drift_metrics,
            'alert_count': len(alerts)
        }
        
        return {
            'status': 'completed',
            'alerts': alerts,
            'drift_metrics': drift_metrics,
            'model_type': model_type,
            'snapshots_compared': len(comparison_snapshots) - 1
        }
    
    def _compare_importance_distributions(self, current: FeatureImportanceSnapshot,
                                        historical: FeatureImportanceSnapshot) -> Dict[str, Any]:
        """Compare two feature importance distributions"""
        
        # Get common features
        current_features = set(current.importance_dict.keys())
        historical_features = set(historical.importance_dict.keys())
        common_features = current_features & historical_features
        
        if len(common_features) < 5:  # Need minimum features for comparison
            return {
                'status': 'insufficient_overlap',
                'common_features': len(common_features),
                'kl_divergence': 0.0,
                'js_distance': 0.0,
                'rank_correlation': 1.0,
                'top_changed_features': []
            }
        
        # Extract values for common features
        current_values = np.array([current.importance_dict[f] for f in common_features])
        historical_values = np.array([historical.importance_dict[f] for f in common_features])
        
        # Normalize to probability distributions
        current_dist = current_values / current_values.sum()
        historical_dist = historical_values / historical_values.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        current_dist = current_dist + epsilon
        historical_dist = historical_dist + epsilon
        
        # Renormalize after adding epsilon
        current_dist = current_dist / current_dist.sum()
        historical_dist = historical_dist / historical_dist.sum()
        
        # Calculate KL divergence
        kl_div = np.sum(current_dist * np.log(current_dist / historical_dist))
        
        # Calculate Jensen-Shannon distance
        js_dist = jensenshannon(current_dist, historical_dist)
        
        # Calculate rank correlation
        current_ranks = stats.rankdata(-current_values)  # Negative for descending order
        historical_ranks = stats.rankdata(-historical_values)
        rank_corr, _ = stats.spearmanr(current_ranks, historical_ranks)
        rank_corr = rank_corr if not np.isnan(rank_corr) else 0.0
        
        # Find top changed features
        importance_changes = {}
        for feature in common_features:
            current_imp = current.importance_dict[feature]
            historical_imp = historical.importance_dict[feature]
            
            # Relative change
            if historical_imp > 0:
                rel_change = abs(current_imp - historical_imp) / historical_imp
            else:
                rel_change = abs(current_imp)  # Historical was zero
            
            importance_changes[feature] = rel_change
        
        # Get top N changed features
        top_changed = sorted(importance_changes.items(), key=lambda x: x[1], reverse=True)[:10]
        top_changed_features = [f[0] for f in top_changed]
        
        return {
            'status': 'success',
            'common_features': len(common_features),
            'kl_divergence': kl_div,
            'js_distance': js_dist,
            'rank_correlation': rank_corr,
            'top_changed_features': top_changed_features,
            'importance_changes': dict(top_changed[:5])  # Top 5 for logging
        }
    
    def _determine_drift_severity(self, metric_value: float, threshold: float) -> DriftSeverity:
        """Determine drift severity based on metric value"""
        ratio = metric_value / threshold
        
        if ratio >= 3.0:
            return DriftSeverity.CRITICAL
        elif ratio >= 2.0:
            return DriftSeverity.HIGH
        elif ratio >= 1.5:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    def _get_drift_recommendation(self, severity: DriftSeverity, metric_type: str) -> str:
        """Get recommendation based on drift severity and type"""
        base_recommendations = {
            DriftSeverity.LOW: "Monitor closely in next few periods",
            DriftSeverity.MEDIUM: "Consider feature engineering review or model retraining",
            DriftSeverity.HIGH: "Recommend full model retraining with feature analysis",
            DriftSeverity.CRITICAL: "Immediate action required: Emergency retraining recommended"
        }
        
        metric_specific = {
            'kl_divergence': "Feature importance distribution has changed significantly",
            'js_distance': "Feature importance patterns show substantial shift", 
            'rank_correlation': "Feature ranking order has changed dramatically"
        }
        
        base_rec = base_recommendations.get(severity, "Review model performance")
        metric_rec = metric_specific.get(metric_type, "")
        
        return f"{metric_rec}. {base_rec}."
    
    def create_teacher_model_snapshot(self, model: Any, model_type: str,
                                    performance_metrics: Dict[str, float],
                                    feature_importance: Dict[str, float]) -> str:
        """Create a snapshot of a well-performing model for distillation"""
        
        if not self.config.enable_model_distillation:
            logger.info("Model distillation disabled")
            return ""
        
        # Generate snapshot ID
        snapshot_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create teacher model record
        teacher_record = {
            'snapshot_id': snapshot_id,
            'model_type': model_type,
            'timestamp': datetime.now(),
            'performance_metrics': performance_metrics.copy(),
            'feature_importance': feature_importance.copy(),
            'distillation_ready': True
        }
        
        # Store model if possible
        try:
            model_path = self.cache_dir / f"teacher_model_{snapshot_id}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            teacher_record['model_path'] = str(model_path)
            logger.info(f"Teacher model saved: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to save teacher model: {e}")
            teacher_record['model_path'] = None
        
        # Store record
        self.teacher_models[snapshot_id] = teacher_record
        
        # Maintain storage limits
        if len(self.teacher_models) > self.config.max_history_retention // 2:  # Keep fewer teacher models
            # Remove oldest teacher models
            oldest_snapshots = sorted(self.teacher_models.keys(), 
                                    key=lambda x: self.teacher_models[x]['timestamp'])[:5]
            for old_id in oldest_snapshots:
                self._remove_teacher_model(old_id)
        
        logger.info(f"Created teacher model snapshot: {snapshot_id}")
        return snapshot_id
    
    def perform_knowledge_distillation(self, student_model_params: Dict[str, Any],
                                     teacher_snapshot_id: str,
                                     training_data: pd.DataFrame,
                                     training_target: pd.Series) -> Dict[str, Any]:
        """Perform knowledge distillation from teacher to student model"""
        
        if not self.config.enable_model_distillation:
            return {'status': 'disabled', 'reason': 'Knowledge distillation disabled'}
        
        if teacher_snapshot_id not in self.teacher_models:
            return {'status': 'error', 'reason': f'Teacher model {teacher_snapshot_id} not found'}
        
        teacher_record = self.teacher_models[teacher_snapshot_id]
        
        try:
            # Load teacher model
            if teacher_record.get('model_path') and Path(teacher_record['model_path']).exists():
                with open(teacher_record['model_path'], 'rb') as f:
                    teacher_model = pickle.load(f)
            else:
                return {'status': 'error', 'reason': 'Teacher model file not found'}
            
            # Get teacher predictions (soft targets)
            teacher_predictions = teacher_model.predict(training_data)
            
            # Apply temperature scaling for knowledge distillation
            temperature = self.config.distillation_temperature
            soft_targets = teacher_predictions / temperature
            
            # Combine hard and soft targets
            distillation_weight = self.config.transfer_learning_weight
            combined_targets = (1 - distillation_weight) * training_target + distillation_weight * soft_targets
            
            # Create student model with reduced complexity
            student_params = self._adjust_student_complexity(student_model_params)
            
            # Train student model (this would be implemented based on the specific model type)
            distillation_result = self._train_student_model(
                student_params, training_data, combined_targets, teacher_predictions
            )
            
            # Record distillation
            distillation_record = {
                'timestamp': datetime.now(),
                'teacher_snapshot_id': teacher_snapshot_id,
                'student_params': student_params,
                'distillation_weight': distillation_weight,
                'temperature': temperature,
                'result': distillation_result
            }
            
            self.distillation_history.append(distillation_record)
            
            logger.info(f"Knowledge distillation completed: {teacher_snapshot_id}")
            return {'status': 'success', 'distillation_record': distillation_record}
            
        except Exception as e:
            logger.error(f"Knowledge distillation failed: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def _adjust_student_complexity(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust student model complexity relative to teacher"""
        student_params = base_params.copy()
        complexity_factor = self.config.student_model_complexity
        
        # Reduce model complexity based on the factor
        if 'num_leaves' in student_params:
            student_params['num_leaves'] = max(10, int(student_params['num_leaves'] * complexity_factor))
        
        if 'max_depth' in student_params and student_params['max_depth'] > 0:
            student_params['max_depth'] = max(3, int(student_params['max_depth'] * complexity_factor))
        
        if 'n_estimators' in student_params:
            student_params['n_estimators'] = max(50, int(student_params['n_estimators'] * complexity_factor))
        
        return student_params
    
    def _train_student_model(self, params: Dict, data: pd.DataFrame, 
                           targets: pd.Series, teacher_preds: np.ndarray) -> Dict[str, Any]:
        """Train student model with knowledge distillation (placeholder implementation)"""
        # This is a placeholder - actual implementation would depend on the specific model type
        # For LightGBM, this would involve creating a custom loss function that combines
        # MSE loss with distillation loss
        
        return {
            'student_model': None,  # Placeholder
            'training_loss': 0.0,
            'distillation_loss': 0.0,
            'validation_metrics': {}
        }
    
    def get_drift_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """Get summary of recent drift detection results"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_alerts = [alert for alert in self.drift_alerts if alert.timestamp >= cutoff_date]
        
        if not recent_alerts:
            return {
                'status': 'no_alerts',
                'period_days': days_back,
                'total_alerts': 0
            }
        
        # Categorize alerts
        alert_by_severity = {}
        alert_by_type = {}
        
        for alert in recent_alerts:
            severity = alert.severity.value
            alert_type = alert.drift_type
            
            alert_by_severity[severity] = alert_by_severity.get(severity, 0) + 1
            alert_by_type[alert_type] = alert_by_type.get(alert_type, 0) + 1
        
        # Find most affected features
        affected_features = {}
        for alert in recent_alerts:
            for feature in alert.affected_features[:5]:  # Top 5 from each alert
                affected_features[feature] = affected_features.get(feature, 0) + 1
        
        top_affected_features = sorted(affected_features.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'status': 'summary_available',
            'period_days': days_back,
            'total_alerts': len(recent_alerts),
            'alerts_by_severity': alert_by_severity,
            'alerts_by_type': alert_by_type,
            'top_affected_features': top_affected_features,
            'most_recent_alert': recent_alerts[-1].timestamp if recent_alerts else None,
            'critical_alerts': sum(1 for a in recent_alerts if a.severity == DriftSeverity.CRITICAL)
        }
    
    def get_knowledge_retention_report(self) -> Dict[str, Any]:
        """Generate comprehensive knowledge retention report"""
        report = {
            'timestamp': datetime.now(),
            'system_status': 'active',
            'configuration': {
                'kl_threshold': self.config.kl_divergence_threshold,
                'js_threshold': self.config.js_distance_threshold,
                'rank_correlation_threshold': self.config.rank_correlation_threshold,
                'history_window': self.config.importance_history_window,
                'distillation_enabled': self.config.enable_model_distillation
            }
        }
        
        # Model tracking summary
        model_summaries = {}
        for model_type, history in self.importance_history.items():
            if history:
                latest = history[-1]
                model_summaries[model_type] = {
                    'snapshots_recorded': len(history),
                    'latest_timestamp': latest.timestamp,
                    'latest_feature_count': latest.feature_count,
                    'latest_performance': latest.model_performance,
                    'drift_alerts_24h': len([a for a in self.drift_alerts 
                                           if a.timestamp >= datetime.now() - timedelta(days=1)
                                           and model_type in str(a)])
                }
        
        report['model_tracking'] = model_summaries
        
        # Drift analysis summary
        report['drift_analysis'] = self.get_drift_summary(days_back=7)  # Last week
        
        # Teacher model status
        report['knowledge_distillation'] = {
            'teacher_models_available': len(self.teacher_models),
            'distillation_runs': len(self.distillation_history),
            'last_distillation': self.distillation_history[-1]['timestamp'] if self.distillation_history else None
        }
        
        # System health
        total_alerts = len(self.drift_alerts)
        critical_alerts_7d = len([a for a in self.drift_alerts 
                                if a.timestamp >= datetime.now() - timedelta(days=7)
                                and a.severity == DriftSeverity.CRITICAL])
        
        if critical_alerts_7d > 3:
            health_status = 'critical'
        elif critical_alerts_7d > 1:
            health_status = 'warning'
        elif total_alerts > 0:
            health_status = 'monitoring'
        else:
            health_status = 'healthy'
        
        report['system_health'] = {
            'status': health_status,
            'total_alerts': total_alerts,
            'critical_alerts_7d': critical_alerts_7d,
            'last_alert': self.drift_alerts[-1].timestamp if self.drift_alerts else None
        }
        
        return report
    
    def _save_snapshot(self, snapshot: FeatureImportanceSnapshot, 
                      metadata: Dict[str, Any] = None) -> None:
        """Save snapshot to disk"""
        try:
            snapshot_data = {
                'timestamp': snapshot.timestamp.isoformat(),
                'importance_dict': snapshot.importance_dict,
                'model_type': snapshot.model_type,
                'model_performance': snapshot.model_performance,
                'feature_count': snapshot.feature_count,
                'model_hash': snapshot.model_hash,
                'metadata': metadata or {}
            }
            
            filename = f"importance_snapshot_{snapshot.model_type}_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.cache_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(snapshot_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save snapshot: {e}")
    
    def _remove_teacher_model(self, snapshot_id: str) -> None:
        """Remove teacher model and clean up files"""
        if snapshot_id in self.teacher_models:
            teacher_record = self.teacher_models[snapshot_id]
            
            # Remove model file
            if teacher_record.get('model_path'):
                try:
                    Path(teacher_record['model_path']).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to remove teacher model file: {e}")
            
            # Remove from memory
            del self.teacher_models[snapshot_id]
            logger.info(f"Removed teacher model: {snapshot_id}")
    
    def _log_drift_alert(self, alert: DriftAlert) -> None:
        """Log drift alert with appropriate severity"""
        severity_map = {
            DriftSeverity.LOW: logger.info,
            DriftSeverity.MEDIUM: logger.warning,
            DriftSeverity.HIGH: logger.warning,
            DriftSeverity.CRITICAL: logger.error
        }
        
        log_func = severity_map.get(alert.severity, logger.info)
        
        log_func(f"DRIFT ALERT [{alert.severity.value.upper()}]: "
                f"{alert.drift_type} = {alert.metric_value:.4f} "
                f"(threshold: {alert.threshold:.4f}). "
                f"Affected features: {', '.join(alert.affected_features[:3])}...")
        
        log_func(f"Recommendation: {alert.recommendation}")


# Helper function for external usage
def calculate_feature_stability_score(importance_history: List[Dict[str, float]], 
                                    window: int = 5) -> float:
    """Calculate overall feature stability score"""
    if len(importance_history) < 2:
        return 1.0
    
    recent_history = importance_history[-window:]
    if len(recent_history) < 2:
        return 1.0
    
    # Calculate pairwise correlations
    correlations = []
    for i in range(len(recent_history) - 1):
        curr_imp = recent_history[i+1]
        prev_imp = recent_history[i]
        
        # Find common features
        common_features = set(curr_imp.keys()) & set(prev_imp.keys())
        if len(common_features) < 5:
            continue
        
        # Get values for common features
        curr_values = [curr_imp[f] for f in common_features]
        prev_values = [prev_imp[f] for f in common_features]
        
        # Calculate correlation
        corr, _ = stats.pearsonr(curr_values, prev_values)
        if not np.isnan(corr):
            correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0.0