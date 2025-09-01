#!/usr/bin/env python3
"""
Production Readiness System with Quantitative Go/No-Go Gates
Implements specific thresholds for model deployment decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class DeploymentDecision(Enum):
    """Deployment decision types"""
    GO = "go"              # Deploy to production
    NO_GO = "no_go"        # Don't deploy
    SHADOW = "shadow"      # Deploy in shadow mode
    CONDITIONAL = "conditional"  # Deploy with conditions

@dataclass
class ProductionGates:
    """Production readiness gates with specific thresholds"""
    
    # IC-based gates
    min_ic_improvement: float = 0.02         # Minimum IC improvement vs baseline
    min_absolute_ic: float = 0.01            # Minimum absolute IC
    min_rank_ic: float = 0.015               # Minimum Rank IC
    ic_stability_threshold: float = 0.7      # IC stability ratio (good periods / total)
    
    # QLIKE/MSE gates  
    max_qlike_improvement: float = 0.08      # Maximum QLIKE improvement (lower is better)
    max_rmse_improvement: float = 0.10       # Maximum RMSE improvement (lower is better)
    
    # OR logic gate configuration (Fix 6 implementation)
    use_or_logic: bool = True                # Enable OR logic: IC>=0.02 OR QLIKE>=8%
    
    # Training efficiency gates
    max_training_time_multiplier: float = 1.5  # Max training time vs baseline
    min_convergence_quality: float = 0.8        # Convergence quality metric
    
    # Risk gates
    max_turnover_increase: float = 0.15      # Maximum turnover increase
    max_drawdown_increase: float = 0.05      # Maximum drawdown increase
    min_sharpe_ratio: float = 0.5            # Minimum Sharpe ratio
    
    # Stability gates
    min_feature_stability: float = 0.6       # Feature importance stability
    max_prediction_drift: float = 0.1        # Maximum prediction drift
    min_model_consistency: float = 0.8       # Cross-validation consistency
    
    # Business gates
    min_capacity_retention: float = 0.9      # Minimum capacity retention
    max_implementation_complexity: int = 3    # Max complexity score (1-5)

@dataclass
class ValidationMetrics:
    """Validation metrics for production readiness"""
    # Performance metrics
    ic_current: float = 0.0
    ic_baseline: float = 0.0
    rank_ic_current: float = 0.0
    rank_ic_baseline: float = 0.0
    qlike_current: float = 0.0
    qlike_baseline: float = 0.0
    rmse_current: float = 0.0
    rmse_baseline: float = 0.0
    
    # Stability metrics
    ic_stability: float = 0.0
    feature_stability: float = 0.0
    model_consistency: float = 0.0
    prediction_drift: float = 0.0
    
    # Efficiency metrics
    training_time_ratio: float = 1.0
    convergence_quality: float = 0.0
    
    # Risk metrics
    turnover_ratio: float = 1.0
    max_drawdown_ratio: float = 1.0
    sharpe_ratio: float = 0.0
    
    # Business metrics
    capacity_retention: float = 1.0
    implementation_complexity: int = 1

class ProductionReadinessSystem:
    """
    Production Readiness System with Quantitative Gates
    
    Implements comprehensive go/no-go decision framework with:
    1. Quantitative performance thresholds
    2. Risk and stability gates
    3. Business impact assessment
    4. Shadow mode recommendations
    """
    
    def __init__(self, gates: ProductionGates = None):
        self.gates = gates or ProductionGates()
        self.validation_history = []
        
        # Rollback safety tracking (Fix)
        self.consecutive_failures = 0
        self.last_stable_model = None
        self.max_consecutive_failures = 3  # Threshold for auto-rollback warning
        
        logger.info("ProductionReadinessSystem initialized")
        logger.info(f"Gates: IC≥{self.gates.min_ic_improvement:.3f}, "
                   f"QLIKE≤{self.gates.max_qlike_improvement:.3f}, "
                   f"Training≤{self.gates.max_training_time_multiplier:.1f}x")
        logger.info(f"Rollback safety: {self.max_consecutive_failures} consecutive failures trigger warning")
    
    def evaluate_model_readiness(self, current_metrics: ValidationMetrics, 
                               baseline_metrics: Optional[ValidationMetrics] = None,
                               model_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive model readiness evaluation
        Returns deployment decision with detailed reasoning
        """
        # Set baseline if not provided
        if baseline_metrics is None:
            baseline_metrics = self._get_default_baseline()
        
        # Calculate derived metrics
        metrics = self._calculate_derived_metrics(current_metrics, baseline_metrics)
        
        # Evaluate each gate category
        performance_result = self._evaluate_performance_gates(metrics)
        stability_result = self._evaluate_stability_gates(metrics)
        risk_result = self._evaluate_risk_gates(metrics)
        efficiency_result = self._evaluate_efficiency_gates(metrics)
        business_result = self._evaluate_business_gates(metrics, model_metadata or {})
        
        # Overall decision logic
        overall_decision = self._make_deployment_decision(
            performance_result, stability_result, risk_result, 
            efficiency_result, business_result
        )
        
        # 🔧 CRITICAL FIX: Ensure consistent gate counting across all reporting
        gates_passed_count = overall_decision['gates_passed']
        total_gates_count = overall_decision['total_gates']
        
        # Compile comprehensive result with consistent gate counting
        result = {
            'decision': overall_decision['decision'],
            'confidence': overall_decision['confidence'],
            'reasoning': overall_decision['reasoning'],
            'gates': {
                'performance': performance_result,
                'stability': stability_result,
                'risk': risk_result,
                'efficiency': efficiency_result,
                'business': business_result
            },
            # 🔧 FIX: Add explicit gate counting to result for consistent reporting
            'gates_passed': gates_passed_count,
            'total_gates': total_gates_count,
            'overall_score': overall_decision['overall_score'],
            'metrics': metrics,
            'recommendations': self._generate_recommendations(overall_decision, metrics),
            'shadow_conditions': self._get_shadow_conditions(overall_decision),
            'timestamp': pd.Timestamp.now()
        }
        
        # Store for history
        self.validation_history.append(result)
        
        # Update rollback safety tracking (Fix)
        self._update_rollback_tracking(result)
        
        # Log decision
        self._log_decision(result)
        
        return result
    
    def _get_default_baseline(self) -> ValidationMetrics:
        """Get default baseline metrics for comparison"""
        return ValidationMetrics(
            ic_current=0.005,
            ic_baseline=0.005,
            rank_ic_current=0.01,
            rank_ic_baseline=0.01,
            qlike_current=1.0,
            qlike_baseline=1.0,
            rmse_current=0.1,
            rmse_baseline=0.1,
            sharpe_ratio=0.3
        )
    
    def _calculate_derived_metrics(self, current: ValidationMetrics, 
                                 baseline: ValidationMetrics) -> Dict[str, float]:
        """Calculate derived metrics for gate evaluation"""
        return {
            # Performance improvements
            'ic_improvement': current.ic_current - baseline.ic_baseline,
            'ic_improvement_pct': (current.ic_current - baseline.ic_baseline) / max(abs(baseline.ic_baseline), 1e-6),
            'rank_ic_improvement': current.rank_ic_current - baseline.rank_ic_baseline,
            'qlike_improvement': (baseline.qlike_baseline - current.qlike_current) / max(baseline.qlike_baseline, 1e-6),
            'rmse_improvement': (baseline.rmse_baseline - current.rmse_current) / max(baseline.rmse_baseline, 1e-6),
            
            # Absolute values
            'absolute_ic': abs(current.ic_current),
            'absolute_rank_ic': abs(current.rank_ic_current),
            
            # Stability metrics
            'ic_stability': current.ic_stability,
            'feature_stability': current.feature_stability,
            'model_consistency': current.model_consistency,
            'prediction_drift': current.prediction_drift,
            
            # Risk metrics
            'training_time_ratio': current.training_time_ratio,
            'convergence_quality': current.convergence_quality,
            'turnover_ratio': current.turnover_ratio,
            'drawdown_ratio': current.max_drawdown_ratio,
            'sharpe_ratio': current.sharpe_ratio,
            
            # Business metrics
            'capacity_retention': current.capacity_retention,
            'implementation_complexity': current.implementation_complexity
        }
    
    def _evaluate_performance_gates(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate performance-related gates with OR logic (Fix 6)"""
        gates_passed = []
        gates_failed = []
        
        # OR logic implementation: IC improvement ≥ 0.02 OR QLIKE improvement ≥ 8%
        ic_gate_pass = metrics['ic_improvement'] >= self.gates.min_ic_improvement
        qlike_gate_pass = metrics.get('qlike_improvement', 0) >= self.gates.max_qlike_improvement
        
        if self.gates.use_or_logic and (ic_gate_pass or qlike_gate_pass):
            # OR logic: pass if either IC or QLIKE gate passes
            gates_passed.append(f"OR Gate PASS: IC={metrics['ic_improvement']:.4f} ≥ {self.gates.min_ic_improvement:.4f} OR QLIKE={metrics.get('qlike_improvement', 0):.3f} ≥ {self.gates.max_qlike_improvement:.3f}")
        else:
            # Traditional AND logic
            if ic_gate_pass:
                gates_passed.append(f"IC improvement: {metrics['ic_improvement']:.4f} ≥ {self.gates.min_ic_improvement:.4f}")
            else:
                gates_failed.append(f"IC improvement: {metrics['ic_improvement']:.4f} < {self.gates.min_ic_improvement:.4f}")
        
        # Absolute IC gate
        if metrics['absolute_ic'] >= self.gates.min_absolute_ic:
            gates_passed.append(f"Absolute IC: {metrics['absolute_ic']:.4f} ≥ {self.gates.min_absolute_ic:.4f}")
        else:
            gates_failed.append(f"Absolute IC: {metrics['absolute_ic']:.4f} < {self.gates.min_absolute_ic:.4f}")
        
        # Rank IC gate
        if metrics['absolute_rank_ic'] >= self.gates.min_rank_ic:
            gates_passed.append(f"Rank IC: {metrics['absolute_rank_ic']:.4f} ≥ {self.gates.min_rank_ic:.4f}")
        else:
            gates_failed.append(f"Rank IC: {metrics['absolute_rank_ic']:.4f} < {self.gates.min_rank_ic:.4f}")
        
        # QLIKE improvement gate (optional - only if we have improvement data)
        if metrics.get('qlike_improvement', 0) > 0:
            if metrics['qlike_improvement'] >= self.gates.max_qlike_improvement:
                gates_passed.append(f"QLIKE improvement: {metrics['qlike_improvement']:.3f} ≥ {self.gates.max_qlike_improvement:.3f}")
            else:
                gates_failed.append(f"QLIKE improvement: {metrics['qlike_improvement']:.3f} < {self.gates.max_qlike_improvement:.3f}")
        
        return {
            'status': 'PASS' if not gates_failed else 'FAIL',
            'gates_passed': gates_passed,
            'gates_failed': gates_failed,
            'score': len(gates_passed) / (len(gates_passed) + len(gates_failed)) if gates_passed or gates_failed else 0.5
        }
    
    def _evaluate_stability_gates(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate stability-related gates"""
        gates_passed = []
        gates_failed = []
        
        # IC stability gate
        if metrics['ic_stability'] >= self.gates.ic_stability_threshold:
            gates_passed.append(f"IC stability: {metrics['ic_stability']:.3f} ≥ {self.gates.ic_stability_threshold:.3f}")
        else:
            gates_failed.append(f"IC stability: {metrics['ic_stability']:.3f} < {self.gates.ic_stability_threshold:.3f}")
        
        # Feature stability gate
        if metrics['feature_stability'] >= self.gates.min_feature_stability:
            gates_passed.append(f"Feature stability: {metrics['feature_stability']:.3f} ≥ {self.gates.min_feature_stability:.3f}")
        else:
            gates_failed.append(f"Feature stability: {metrics['feature_stability']:.3f} < {self.gates.min_feature_stability:.3f}")
        
        # Model consistency gate
        if metrics['model_consistency'] >= self.gates.min_model_consistency:
            gates_passed.append(f"Model consistency: {metrics['model_consistency']:.3f} ≥ {self.gates.min_model_consistency:.3f}")
        else:
            gates_failed.append(f"Model consistency: {metrics['model_consistency']:.3f} < {self.gates.min_model_consistency:.3f}")
        
        # Prediction drift gate
        if metrics['prediction_drift'] <= self.gates.max_prediction_drift:
            gates_passed.append(f"Prediction drift: {metrics['prediction_drift']:.3f} ≤ {self.gates.max_prediction_drift:.3f}")
        else:
            gates_failed.append(f"Prediction drift: {metrics['prediction_drift']:.3f} > {self.gates.max_prediction_drift:.3f}")
        
        return {
            'status': 'PASS' if not gates_failed else 'FAIL',
            'gates_passed': gates_passed,
            'gates_failed': gates_failed,
            'score': len(gates_passed) / (len(gates_passed) + len(gates_failed)) if gates_passed or gates_failed else 0.5
        }
    
    def _evaluate_risk_gates(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate risk-related gates"""
        gates_passed = []
        gates_failed = []
        
        # Sharpe ratio gate
        if metrics['sharpe_ratio'] >= self.gates.min_sharpe_ratio:
            gates_passed.append(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f} ≥ {self.gates.min_sharpe_ratio:.3f}")
        else:
            gates_failed.append(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f} < {self.gates.min_sharpe_ratio:.3f}")
        
        # Turnover gate
        if metrics['turnover_ratio'] <= (1 + self.gates.max_turnover_increase):
            gates_passed.append(f"Turnover ratio: {metrics['turnover_ratio']:.3f} ≤ {1 + self.gates.max_turnover_increase:.3f}")
        else:
            gates_failed.append(f"Turnover ratio: {metrics['turnover_ratio']:.3f} > {1 + self.gates.max_turnover_increase:.3f}")
        
        # Drawdown gate
        if metrics['drawdown_ratio'] <= (1 + self.gates.max_drawdown_increase):
            gates_passed.append(f"Drawdown ratio: {metrics['drawdown_ratio']:.3f} ≤ {1 + self.gates.max_drawdown_increase:.3f}")
        else:
            gates_failed.append(f"Drawdown ratio: {metrics['drawdown_ratio']:.3f} > {1 + self.gates.max_drawdown_increase:.3f}")
        
        return {
            'status': 'PASS' if not gates_failed else 'FAIL',
            'gates_passed': gates_passed,
            'gates_failed': gates_failed,
            'score': len(gates_passed) / (len(gates_passed) + len(gates_failed)) if gates_passed or gates_failed else 0.5
        }
    
    def _evaluate_efficiency_gates(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate efficiency-related gates"""
        gates_passed = []
        gates_failed = []
        
        # Training time gate
        if metrics['training_time_ratio'] <= self.gates.max_training_time_multiplier:
            gates_passed.append(f"Training time ratio: {metrics['training_time_ratio']:.2f} ≤ {self.gates.max_training_time_multiplier:.2f}")
        else:
            gates_failed.append(f"Training time ratio: {metrics['training_time_ratio']:.2f} > {self.gates.max_training_time_multiplier:.2f}")
        
        # Convergence quality gate
        if metrics['convergence_quality'] >= self.gates.min_convergence_quality:
            gates_passed.append(f"Convergence quality: {metrics['convergence_quality']:.3f} ≥ {self.gates.min_convergence_quality:.3f}")
        else:
            gates_failed.append(f"Convergence quality: {metrics['convergence_quality']:.3f} < {self.gates.min_convergence_quality:.3f}")
        
        return {
            'status': 'PASS' if not gates_failed else 'FAIL',
            'gates_passed': gates_passed,
            'gates_failed': gates_failed,
            'score': len(gates_passed) / (len(gates_passed) + len(gates_failed)) if gates_passed or gates_failed else 0.5
        }
    
    def _evaluate_business_gates(self, metrics: Dict[str, float], 
                               metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate business-related gates"""
        gates_passed = []
        gates_failed = []
        
        # Capacity retention gate
        if metrics['capacity_retention'] >= self.gates.min_capacity_retention:
            gates_passed.append(f"Capacity retention: {metrics['capacity_retention']:.3f} ≥ {self.gates.min_capacity_retention:.3f}")
        else:
            gates_failed.append(f"Capacity retention: {metrics['capacity_retention']:.3f} < {self.gates.min_capacity_retention:.3f}")
        
        # Implementation complexity gate
        complexity = metrics.get('implementation_complexity', 1)
        if complexity <= self.gates.max_implementation_complexity:
            gates_passed.append(f"Implementation complexity: {complexity} ≤ {self.gates.max_implementation_complexity}")
        else:
            gates_failed.append(f"Implementation complexity: {complexity} > {self.gates.max_implementation_complexity}")
        
        return {
            'status': 'PASS' if not gates_failed else 'FAIL',
            'gates_passed': gates_passed,
            'gates_failed': gates_failed,
            'score': len(gates_passed) / (len(gates_passed) + len(gates_failed)) if gates_passed or gates_failed else 0.5
        }
    
    def _make_deployment_decision(self, performance: Dict, stability: Dict, 
                                risk: Dict, efficiency: Dict, business: Dict) -> Dict[str, Any]:
        """Make overall deployment decision based on gate results"""
        gate_results = [performance, stability, risk, efficiency, business]
        gate_names = ['Performance', 'Stability', 'Risk', 'Efficiency', 'Business']
        
        # Count passes and calculate overall score
        passed_gates = sum(1 for gate in gate_results if gate['status'] == 'PASS')
        total_gates = len(gate_results)
        overall_score = sum(gate['score'] for gate in gate_results) / total_gates
        
        # Decision logic
        if passed_gates == total_gates:
            # All gates pass - GO
            decision = DeploymentDecision.GO
            confidence = 'high'
            reasoning = f"All {total_gates} gate categories passed. Ready for production deployment."
        
        elif passed_gates >= 4:
            # Most gates pass - CONDITIONAL GO
            failed_gate = next(i for i, gate in enumerate(gate_results) if gate['status'] == 'FAIL')
            decision = DeploymentDecision.CONDITIONAL
            confidence = 'medium'
            reasoning = f"{passed_gates}/{total_gates} gates passed. {gate_names[failed_gate]} gate failed - deploy with monitoring."
        
        elif passed_gates >= 3 or (passed_gates >= 2 and performance['status'] == 'PASS'):
            # Some gates pass, including performance - SHADOW
            decision = DeploymentDecision.SHADOW
            confidence = 'medium'
            reasoning = f"{passed_gates}/{total_gates} gates passed. Deploy in shadow mode for validation."
        
        else:
            # Too many failures - NO GO
            decision = DeploymentDecision.NO_GO
            confidence = 'high'
            reasoning = f"Only {passed_gates}/{total_gates} gates passed. Not ready for deployment."
        
        return {
            'decision': decision,
            'confidence': confidence,
            'reasoning': reasoning,
            'overall_score': overall_score,
            'gates_passed': passed_gates,
            'total_gates': total_gates
        }
    
    def _generate_recommendations(self, decision_result: Dict, metrics: Dict) -> List[str]:
        """Generate actionable recommendations based on decision"""
        recommendations = []
        
        if decision_result['decision'] == DeploymentDecision.GO:
            recommendations.append("✅ Model ready for production deployment")
            recommendations.append("🔄 Continue monitoring performance metrics")
            
        elif decision_result['decision'] == DeploymentDecision.CONDITIONAL:
            recommendations.append("⚠️  Deploy with enhanced monitoring")
            recommendations.append("📊 Set up real-time alerting for failed gate metrics")
            recommendations.append("🔄 Plan bi-weekly performance review")
            
        elif decision_result['decision'] == DeploymentDecision.SHADOW:
            recommendations.append("👥 Deploy in shadow mode alongside current model")
            recommendations.append("📈 Monitor comparative performance for 2-4 weeks")
            recommendations.append("🎯 Focus on improving failed gate metrics")
            
        else:  # NO_GO
            recommendations.append("❌ Do not deploy - address critical issues first")
            
            # Specific recommendations based on metrics
            if metrics['ic_improvement'] < self.gates.min_ic_improvement:
                recommendations.append("📈 Improve IC: Add more predictive features or optimize training")
            
            if metrics['ic_stability'] < self.gates.ic_stability_threshold:
                recommendations.append("🎯 Improve IC stability: Consider regime-aware modeling")
            
            if metrics['training_time_ratio'] > self.gates.max_training_time_multiplier:
                recommendations.append("⚡ Optimize training efficiency: Reduce model complexity")
        
        return recommendations
    
    def _get_shadow_conditions(self, decision_result: Dict) -> Optional[Dict[str, Any]]:
        """Get conditions for shadow mode deployment"""
        if decision_result['decision'] != DeploymentDecision.SHADOW:
            return None
        
        return {
            'monitoring_period_days': 21,  # 3 weeks
            'success_criteria': {
                'min_ic_improvement': self.gates.min_ic_improvement,
                'max_drawdown_ratio': 1 + self.gates.max_drawdown_increase,
                'min_consistency': self.gates.min_model_consistency
            },
            'escalation_conditions': {
                'ic_degradation_threshold': -0.01,
                'max_consecutive_bad_days': 5,
                'max_prediction_error': 0.2
            }
        }
    
    def _log_decision(self, result: Dict[str, Any]) -> None:
        """Log deployment decision with summary"""
        decision = result['decision'].value if hasattr(result['decision'], 'value') else str(result['decision'])
        confidence = result['confidence']
        
        # 🔧 CRITICAL FIX: Use consistent gate counting from result, not recalculated
        overall_score = result.get('overall_score', 0.0)
        gates_passed = result.get('gates_passed', 0)
        total_gates = result.get('total_gates', 5)
        
        # Fallback calculation only if not already in result (should not happen with fix)
        if overall_score == 0.0:
            gate_scores = [gate_result['score'] for gate_result in result['gates'].values()]
            overall_score = sum(gate_scores) / len(gate_scores) if gate_scores else 0.0
        
        # Double-check gate counting consistency (validation)
        recalc_gates_passed = sum(1 for gate_result in result['gates'].values() if gate_result['status'] == 'PASS')
        recalc_total_gates = len(result['gates'])
        
        if gates_passed != recalc_gates_passed or total_gates != recalc_total_gates:
            logger.warning(f"🔧 Gate counting inconsistency detected and fixed: "
                          f"reported={gates_passed}/{total_gates}, actual={recalc_gates_passed}/{recalc_total_gates}")
            gates_passed = recalc_gates_passed
            total_gates = recalc_total_gates
        
        logger.info("=== PRODUCTION READINESS DECISION ===")
        logger.info(f"Decision: {decision.upper()} (confidence: {confidence})")
        logger.info(f"Overall score: {overall_score:.3f}")
        logger.info(f"Reasoning: {result['reasoning']}")
        
        # 🔧 FIX: Use consistent gate counting 
        logger.info(f"Gate pass summary: {gates_passed}/{total_gates} gates passed")
        
        for gate_name, gate_result in result['gates'].items():
            status = gate_result['status']
            gate_score = gate_result['score']
            logger.info(f"{gate_name.capitalize()} gate: {status} ({gate_score:.3f})")
    
    def get_historical_decisions(self, last_n: int = 10) -> pd.DataFrame:
        """Get historical deployment decisions"""
        if not self.validation_history:
            return pd.DataFrame()
        
        recent_history = self.validation_history[-last_n:]
        
        records = []
        for result in recent_history:
            decision_value = result['decision'].value if hasattr(result['decision'], 'value') else str(result['decision'])
            
            record = {
                'timestamp': result['timestamp'],
                'decision': decision_value,
                'confidence': result['confidence'],
                'overall_score': result['gates']['performance']['score'],
                'performance_gate': result['gates']['performance']['status'],
                'stability_gate': result['gates']['stability']['status'],
                'risk_gate': result['gates']['risk']['status'],
                'efficiency_gate': result['gates']['efficiency']['status'],
                'business_gate': result['gates']['business']['status']
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def update_gates(self, **kwargs) -> None:
        """Update gate thresholds dynamically"""
        for key, value in kwargs.items():
            if hasattr(self.gates, key):
                old_value = getattr(self.gates, key)
                setattr(self.gates, key, value)
                logger.info(f"Updated gate {key}: {old_value} -> {value}")
            else:
                logger.warning(f"Unknown gate parameter: {key}")
    
    def _update_rollback_tracking(self, result: Dict[str, Any]) -> None:
        """Update rollback safety tracking (Fix)"""
        decision = result['decision']
        
        if decision in [DeploymentDecision.GO, DeploymentDecision.CONDITIONAL]:
            # Success: reset failure counter and update stable model
            self.consecutive_failures = 0
            self.last_stable_model = {
                'timestamp': result['timestamp'],
                'metrics': result['metrics'],
                'decision': decision,
                'confidence': result['confidence']
            }
            logger.debug(f"Updated stable model reference: {decision.value}")
            
        elif decision == DeploymentDecision.NO_GO:
            # Failure: increment counter and check threshold
            self.consecutive_failures += 1
            logger.warning(f"Production gate failure #{self.consecutive_failures}")
            
            if self.consecutive_failures >= self.max_consecutive_failures:
                self._trigger_rollback_warning()
        
        # Shadow mode doesn't affect failure counting (neutral)
    
    def _trigger_rollback_warning(self) -> None:
        """Trigger rollback warning after consecutive failures (Fix)"""
        logger.error(f"🚨 ROLLBACK SAFETY ALERT: {self.consecutive_failures} consecutive NO_GO decisions")
        logger.error(f"Consider rolling back to last stable model or investigating systemic issues")
        
        if self.last_stable_model:
            stable_time = self.last_stable_model['timestamp']
            logger.error(f"Last stable model: {stable_time}, decision: {self.last_stable_model['decision'].value}")
            logger.error(f"Rollback recommendation: Revert to model state from {stable_time}")
        else:
            logger.error(f"No stable model reference available - manual intervention required")
        
        # Generate rollback operation suggestions
        rollback_ops = self._generate_rollback_operations()
        for op in rollback_ops:
            logger.error(f"Rollback action: {op}")
    
    def _generate_rollback_operations(self) -> List[str]:
        """Generate concrete rollback operation suggestions (Fix)"""
        operations = [
            "1. Call incremental_trainer.rollback_model(steps=3) to undo recent updates",
            "2. Reset to last stable checkpoint from cache/incremental_training/",
            "3. Force full rebuild with TrainingType.FULL_REBUILD",
            "4. Review feature importance drift in knowledge_retention_system",
            "5. Check for data quality issues in recent training batches"
        ]
        
        if self.last_stable_model:
            stable_time = self.last_stable_model['timestamp'].strftime('%Y%m%d_%H%M%S')
            operations.append(f"6. Restore model state from {stable_time} checkpoint")
        
        return operations
    
    def get_rollback_status(self) -> Dict[str, Any]:
        """Get current rollback safety status (Fix)"""
        return {
            'consecutive_failures': self.consecutive_failures,
            'failure_threshold': self.max_consecutive_failures,
            'at_risk': self.consecutive_failures >= self.max_consecutive_failures - 1,
            'last_stable_model': self.last_stable_model,
            'rollback_needed': self.consecutive_failures >= self.max_consecutive_failures,
            'recent_decisions': [h['decision'].value if hasattr(h['decision'], 'value') else str(h['decision']) 
                               for h in self.validation_history[-5:]]
        }


# Helper function to calculate IC stability
def calculate_ic_stability(ic_series: pd.Series, window: int = 60) -> float:
    """Calculate IC stability as the ratio of positive IC periods"""
    if len(ic_series) < window:
        return 0.0
    
    rolling_ic = ic_series.rolling(window=window, min_periods=window//2).mean()
    positive_periods = (rolling_ic > 0).sum()
    total_periods = rolling_ic.notna().sum()
    
    return positive_periods / total_periods if total_periods > 0 else 0.0


# Helper function to calculate feature stability
def calculate_feature_stability(importance_history: List[Dict[str, float]], 
                              top_n: int = 20) -> float:
    """Calculate feature importance stability using rank correlation"""
    if len(importance_history) < 2:
        return 1.0
    
    # Get top features from each period
    feature_ranks = []
    for importance_dict in importance_history:
        # Sort by importance and get top N
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:top_n]]
        feature_ranks.append(top_features)
    
    # Calculate pairwise rank correlations
    correlations = []
    for i in range(len(feature_ranks) - 1):
        # Create rank dictionaries
        rank1 = {f: i for i, f in enumerate(feature_ranks[i])}
        rank2 = {f: i for i, f in enumerate(feature_ranks[i + 1])}
        
        # Find common features
        common_features = set(rank1.keys()) & set(rank2.keys())
        if len(common_features) < 5:  # Need minimum overlap
            continue
        
        # Calculate rank correlation
        ranks1 = [rank1[f] for f in common_features]
        ranks2 = [rank2[f] for f in common_features]
        
        correlation = stats.spearmanr(ranks1, ranks2)[0]
        if not np.isnan(correlation):
            correlations.append(correlation)
    
    return np.mean(correlations) if correlations else 0.0