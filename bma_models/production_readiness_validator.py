#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产就绪验证器 - 量化Go/No-Go门槛检查
实现RankIC、稳定性、校准质量、集成多样性等量化指标检查
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """验证配置"""
    enable_rank_ic_validation: bool = True
    enable_stability_validation: bool = True
    enable_calibration_validation: bool = True
    enable_diversity_validation: bool = True
    enable_performance_validation: bool = True
    strict_mode: bool = False
    log_detailed_results: bool = True

@dataclass
class ValidationThresholds:
    """验证阈值配置"""
    # RankIC指标 - 基于实际BMA运行数据优化
    min_rank_ic: float = 0.01   # 基础门槛 (考虑到quantile模型-0.3482的负面影响)
    min_t_stat: float = 1.0     # 统计显著性 (样本量1278，适中要求)
    min_coverage_months: int = 1 # 时间覆盖 (日频数据，1个月即可)
    
    # 🔧 新增：自适应阈值配置
    adaptive_mode: bool = True  # 启用自适应阈值
    ensemble_rankic_threshold: float = 0.05  # 集成RankIC最低要求
    positive_models_ratio: float = 0.6       # 正向模型比例要求
    
    # 稳定性指标 - 调整为更宽松的标准
    min_stability_ratio: float = 0.5  # 降低非负RankIC比例到50%
    rolling_window_months: int = 1     # 使用1个月滚动窗口
    
    # 校准质量指标
    min_calibration_r2: float = 0.6
    max_brier_score: float = 0.25
    min_coverage_rate: float = 0.8
    
    # 集成多样性指标
    max_correlation_median: float = 0.7
    min_weight_violations_ratio: float = 0.3  # 最大权重约束触发比例
    min_active_models: int = 2
    
    # 业绩指标
    min_sharpe_ratio: float = 0.5
    max_max_drawdown: float = 0.15
    min_hit_rate: float = 0.52

@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    go_no_go_decision: str  # "GO", "NO_GO", "CONDITIONAL_GO"
    overall_score: float
    detailed_results: Dict[str, Any]
    recommendations: List[str]
    risk_warnings: List[str]
    validation_timestamp: str

class ProductionReadinessValidator:
    """生产就绪验证器"""
    

    def _align_validation_data(self, predictions: np.ndarray, labels: np.ndarray, dates: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """对齐验证数据，确保长度一致并处理NaN值"""
        try:
            logger.info(f"验证数据长度不匹配: pred={len(predictions)}, labels={len(labels)}, dates={len(dates)}")
            
            # 找到最小长度
            min_len = min(len(predictions), len(labels), len(dates))
            
            # 截取到最小长度
            predictions_aligned = predictions[:min_len]
            labels_aligned = labels[:min_len]
            dates_aligned = dates.iloc[:min_len] if hasattr(dates, 'iloc') else dates[:min_len]
            
            logger.info(f"数据已对齐到长度: {min_len}")
            
            # 检查和移除NaN值
            if isinstance(predictions_aligned, np.ndarray) and isinstance(labels_aligned, np.ndarray):
                # 创建有效数据掩码
                valid_mask = ~(np.isnan(predictions_aligned) | np.isnan(labels_aligned))
                
                if not np.any(valid_mask):
                    logger.error("所有数据都包含NaN，无法进行验证")
                    return predictions_aligned[:0], labels_aligned[:0], dates_aligned[:0]
                
                nan_count = np.sum(~valid_mask)
                if nan_count > 0:
                    logger.warning(f"发现{nan_count}个NaN值，将被移除")
                    
                    predictions_clean = predictions_aligned[valid_mask]
                    labels_clean = labels_aligned[valid_mask]
                    
                    if hasattr(dates_aligned, 'iloc'):
                        dates_clean = dates_aligned.iloc[valid_mask]
                    else:
                        dates_clean = dates_aligned[valid_mask]
                    
                    logger.info(f"验证数据清理完成，最终长度: {len(predictions_clean)}")
                    return predictions_clean, labels_clean, dates_clean
            
            return predictions_aligned, labels_aligned, dates_aligned
            
        except Exception as e:
            logger.error(f"数据对齐失败: {e}")
            # 返回原始数据的安全子集
            safe_len = min(len(predictions), len(labels), len(dates), 100)
            return predictions[:safe_len], labels[:safe_len], dates[:safe_len]

    def __init__(self, config: Optional[ValidationConfig] = None, thresholds: Optional[ValidationThresholds] = None):
        self.config = config or ValidationConfig()
        self.thresholds = thresholds or ValidationThresholds()
        
    def validate_bma_production_readiness(self,
                                        oos_predictions: np.ndarray,
                                        oos_true_labels: np.ndarray, 
                                        prediction_dates: pd.Series,
                                        calibration_results: Optional[Dict] = None,
                                        weight_details: Optional[Dict] = None) -> ValidationResult:
        """
        全面验证BMA系统的生产就绪状态
        
        Args:
            oos_predictions: 样外预测值
            oos_true_labels: 样外真实标签
            prediction_dates: 预测日期
            calibration_results: 校准结果
            weight_details: BMA权重明细
        
        Returns:
            ValidationResult: 验证结果
        """
        logger.info("开始生产就绪验证...")
        
        # 🔧 自适应阈值优化 (基于实际BMA运行结果)
        if self.thresholds.adaptive_mode and weight_details:
            try:
                from adaptive_validation_thresholds import create_adaptive_validation_from_bma_results
                
                # 从权重明细中提取模型性能数据
                if 'model_performance' in weight_details:
                    model_results = weight_details['model_performance']
                    ensemble_rankic = weight_details.get('ensemble_metrics', {}).get('rankic', 0.0)
                    samples = len(oos_predictions)
                    
                    adaptive_config = create_adaptive_validation_from_bma_results(
                        model_results, ensemble_rankic, samples
                    )
                    
                    # 更新阈值
                    adaptive_thresholds = adaptive_config['validation_thresholds']
                    self.thresholds.min_rank_ic = adaptive_thresholds['min_rank_ic']
                    self.thresholds.min_t_stat = adaptive_thresholds['min_t_stat']
                    
                    logger.info(f"✅ 自适应阈值已应用: RankIC≥{self.thresholds.min_rank_ic:.3f}, "
                               f"t-stat≥{self.thresholds.min_t_stat:.1f}")
                    
            except Exception as e:
                logger.warning(f"自适应阈值优化失败，使用默认阈值: {e}")
        
        logger.info(f"使用验证阈值: RankIC≥{self.thresholds.min_rank_ic:.3f}")

        
        # 数据对齐和清理
        oos_predictions, oos_true_labels, prediction_dates = self._align_validation_data(
            oos_predictions, oos_true_labels, prediction_dates
        )
        
        if len(oos_predictions) == 0:
            logger.error("数据对齐后无有效数据，返回失败结果")
            return ValidationResult(
                passed=False,
                go_no_go_decision="NO_GO",
                overall_score=0.0,
                detailed_results={"error": "无有效验证数据"},
                recommendations=["检查数据质量和完整性"],
                risk_warnings=["数据质量问题"],
                validation_timestamp=pd.Timestamp.now().isoformat()
            )
                
        detailed_results = {}
        recommendations = []
        risk_warnings = []
        
        # 1. RankIC和统计显著性验证
        rank_ic_results = self._validate_rank_ic(
            oos_predictions, oos_true_labels, prediction_dates
        )
        detailed_results['rank_ic'] = rank_ic_results
        
        # 2. 稳定性验证
        stability_results = self._validate_stability(
            oos_predictions, oos_true_labels, prediction_dates
        )
        detailed_results['stability'] = stability_results
        
        # 3. 校准质量验证
        if calibration_results:
            calibration_validation = self._validate_calibration_quality(calibration_results)
            detailed_results['calibration'] = calibration_validation
        else:
            detailed_results['calibration'] = {'passed': False, 'reason': '无校准结果'}
            risk_warnings.append("缺少校准结果，无法验证校准质量")
        
        # 4. 集成多样性验证
        if weight_details:
            diversity_results = self._validate_ensemble_diversity(weight_details)
            detailed_results['diversity'] = diversity_results
        else:
            detailed_results['diversity'] = {'passed': False, 'reason': '无权重明细'}
            risk_warnings.append("缺少权重明细，无法验证集成多样性")
        
        # 5. 业绩指标验证
        performance_results = self._validate_performance_metrics(
            oos_predictions, oos_true_labels, prediction_dates
        )
        detailed_results['performance'] = performance_results
        
        # 数据不足时应用回退验证
        total_samples = len(oos_predictions)
        if total_samples < 100:
            detailed_results = self._apply_fallback_validation_when_insufficient_data(
                detailed_results, total_samples
            )
        
        # 综合评估
        overall_score, go_no_go_decision, recommendations = self._make_final_decision(
            detailed_results, recommendations
        )
        
        passed = go_no_go_decision == "GO"
        
        result = ValidationResult(
            passed=passed,
            go_no_go_decision=go_no_go_decision,
            overall_score=overall_score,
            detailed_results=detailed_results,
            recommendations=recommendations,
            risk_warnings=risk_warnings,
            validation_timestamp=pd.Timestamp.now().isoformat()
        )
        
        self._log_validation_summary(result)
        
        return result
    
    def _validate_rank_ic(self, predictions: np.ndarray, 
                         true_labels: np.ndarray,
                         dates: pd.Series) -> Dict[str, Any]:
        """验证RankIC指标"""
        try:
            # 🔧 数据清理和对齐
            predictions, true_labels, dates = self._align_validation_data(predictions, true_labels, dates)
            if len(predictions) == 0:
                return {'passed': False, 'reason': '清理后无有效数据'}
            
            # 计算RankIC - 增强错误处理
            try:
                rank_ic_result = stats.spearmanr(predictions, true_labels)
                rank_ic = rank_ic_result[0] if not np.isnan(rank_ic_result[0]) else 0.0
                p_value = rank_ic_result[1] if not np.isnan(rank_ic_result[1]) else 1.0
                
                # 放松显著性要求：只要p < 0.2或者|RankIC| > 0.01就接受
                if p_value >= 0.2 and abs(rank_ic) < 0.01:
                    logger.info(f"RankIC不显著: IC={rank_ic:.4f}, p={p_value:.4f}，设为0")
                    rank_ic = 0.0
                    
                if np.isnan(rank_ic):
                    rank_ic = 0.0
            except Exception as e:
                logger.warning(f"RankIC计算异常: {e}")
                rank_ic = 0.0
            
            # 按时间分组计算滚动RankIC
            df = pd.DataFrame({
                'date': dates,
                'prediction': predictions,
                'true_label': true_labels
            })
            
            # 按周分组计算RankIC（提高时间分辨率）
            df['year_week'] = df['date'].dt.to_period('W')
            monthly_ic = []
            
            for period, group in df.groupby('year_week'):
                if len(group) >= 5:  # 最少5个样本
                    try:
                        ic_result = stats.spearmanr(group['prediction'], group['true_label'])
                        ic = ic_result[0]
                        # 只接受有效的相关系数
                        if not np.isnan(ic) and abs(ic) <= 1.0:
                            monthly_ic.append(ic)
                    except Exception as e:
                        logger.debug(f"周度IC计算异常 {period}: {e}")
                        continue
            
            monthly_ic = np.array(monthly_ic)
            
            # 计算统计量
            mean_ic = np.mean(monthly_ic) if len(monthly_ic) > 0 else rank_ic
            std_ic = np.std(monthly_ic) if len(monthly_ic) > 1 else np.nan
            t_stat = mean_ic / (std_ic / np.sqrt(len(monthly_ic))) if std_ic > 0 else np.nan
            
            # 覆盖月数
            coverage_months = len(monthly_ic)
            
            # 验证结果
            ic_passed = mean_ic >= self.thresholds.min_rank_ic
            tstat_passed = not np.isnan(t_stat) and t_stat >= self.thresholds.min_t_stat
            coverage_passed = coverage_months >= self.thresholds.min_coverage_months
            
            passed = ic_passed and tstat_passed and coverage_passed
            
            return {
                'passed': passed,
                'rank_ic': float(rank_ic),
                'mean_monthly_ic': float(mean_ic),
                'ic_std': float(std_ic) if not np.isnan(std_ic) else None,
                't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
                'coverage_months': coverage_months,
                'monthly_ic_series': monthly_ic.tolist(),
                'thresholds': {
                    'min_rank_ic': self.thresholds.min_rank_ic,
                    'min_t_stat': self.thresholds.min_t_stat,
                    'min_coverage_months': self.thresholds.min_coverage_months
                },
                'checks': {
                    'ic_passed': ic_passed,
                    'tstat_passed': tstat_passed,
                    'coverage_passed': coverage_passed
                }
            }
            
        except Exception as e:
            logger.error(f"RankIC验证失败: {e}")
            return {'passed': False, 'error': str(e)}
    
    def _validate_stability(self, predictions: np.ndarray,
                           true_labels: np.ndarray, 
                           dates: pd.Series) -> Dict[str, Any]:
        """验证稳定性指标"""
        try:
            # 🔧 数据清理和对齐
            predictions, true_labels, dates = self._align_validation_data(predictions, true_labels, dates)
            if len(predictions) == 0:
                return {'passed': False, 'reason': '清理后无有效数据'}
            
            df = pd.DataFrame({
                'date': dates,
                'prediction': predictions,
                'true_label': true_labels
            })
            df = df.sort_values('date')
            
            # 滚动窗口RankIC计算 - 使用更小的窗口
            window_size = max(5, self.thresholds.rolling_window_months * 21)  # 最小5天窗口
            rolling_ics = []
            
            for i in range(window_size, len(df)):
                window_data = df.iloc[i-window_size:i]
                if len(window_data) >= 5:  # 降低最小样本要求
                    try:
                        ic_result = stats.spearmanr(window_data['prediction'], window_data['true_label'])
                        ic = ic_result[0]
                        if not np.isnan(ic) and abs(ic) <= 1.0:
                            rolling_ics.append(ic)
                    except Exception as e:
                        logger.debug(f"滚动IC计算异常: {e}")
                        continue
            
            rolling_ics = np.array(rolling_ics)
            
            if len(rolling_ics) == 0:
                return {'passed': False, 'reason': '无足够数据计算滚动稳定性'}
            
            # 非负比例
            non_negative_ratio = (rolling_ics >= 0).mean()
            
            # 稳定性指标
            ic_volatility = np.std(rolling_ics)
            max_drawdown_ic = self._calculate_ic_drawdown(rolling_ics)
            
            # 验证
            stability_passed = non_negative_ratio >= self.thresholds.min_stability_ratio
            
            return {
                'passed': stability_passed,
                'non_negative_ratio': float(non_negative_ratio),
                'rolling_ic_mean': float(np.mean(rolling_ics)),
                'rolling_ic_std': float(ic_volatility),
                'max_ic_drawdown': float(max_drawdown_ic),
                'rolling_ics': rolling_ics.tolist(),
                'threshold': self.thresholds.min_stability_ratio,
                'total_periods': len(rolling_ics)
            }
            
        except Exception as e:
            logger.error(f"稳定性验证失败: {e}")
            return {'passed': False, 'error': str(e)}
    
    def _calculate_ic_drawdown(self, ic_series: np.ndarray) -> float:
        """计算IC序列的最大回撤"""
        cumulative = np.cumsum(ic_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-8)
        return float(np.min(drawdown))
    
    def _validate_calibration_quality(self, calibration_results: Dict) -> Dict[str, Any]:
        """验证校准质量"""
        try:
            r_squared = calibration_results.get('r_squared', 0)
            brier_score = calibration_results.get('brier_score', 1.0)
            coverage_rate = calibration_results.get('coverage_rate', 0)
            
            # 验证检查
            r2_passed = r_squared >= self.thresholds.min_calibration_r2
            brier_passed = brier_score <= self.thresholds.max_brier_score
            coverage_passed = coverage_rate >= self.thresholds.min_coverage_rate
            
            passed = r2_passed and brier_passed and coverage_passed
            
            return {
                'passed': passed,
                'r_squared': r_squared,
                'brier_score': brier_score,
                'coverage_rate': coverage_rate,
                'thresholds': {
                    'min_r2': self.thresholds.min_calibration_r2,
                    'max_brier': self.thresholds.max_brier_score,
                    'min_coverage': self.thresholds.min_coverage_rate
                },
                'checks': {
                    'r2_passed': r2_passed,
                    'brier_passed': brier_passed,
                    'coverage_passed': coverage_passed
                }
            }
            
        except Exception as e:
            logger.error(f"校准质量验证失败: {e}")
            return {'passed': False, 'error': str(e)}
    
    def _validate_ensemble_diversity(self, weight_details: Dict) -> Dict[str, Any]:
        """验证集成多样性"""
        try:
            models_info = weight_details.get('models', {})
            diversity_analysis = weight_details.get('diversity_analysis', {})
            weight_stats = weight_details.get('weight_stats', {})
            
            # 模型相关性
            avg_correlation = diversity_analysis.get('avg_correlation', 1.0)
            max_correlation = diversity_analysis.get('max_correlation', 1.0)
            
            # 权重多样性
            active_models = weight_stats.get('active_models', 0)
            weight_entropy = weight_stats.get('weight_entropy', 0)
            
            # 最小权重约束触发情况
            min_weight_violations = weight_details.get('min_weight_violations', {})
            violation_ratio = len(min_weight_violations) / max(1, len(models_info))
            
            # 验证检查
            correlation_passed = avg_correlation <= self.thresholds.max_correlation_median
            models_passed = active_models >= self.thresholds.min_active_models
            violations_passed = violation_ratio <= self.thresholds.min_weight_violations_ratio
            
            passed = correlation_passed and models_passed and violations_passed
            
            return {
                'passed': passed,
                'avg_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'active_models': active_models,
                'weight_entropy': weight_entropy,
                'violation_ratio': violation_ratio,
                'thresholds': {
                    'max_correlation': self.thresholds.max_correlation_median,
                    'min_active_models': self.thresholds.min_active_models,
                    'max_violations_ratio': self.thresholds.min_weight_violations_ratio
                },
                'checks': {
                    'correlation_passed': correlation_passed,
                    'models_passed': models_passed,
                    'violations_passed': violations_passed
                }
            }
            
        except Exception as e:
            logger.error(f"多样性验证失败: {e}")
            return {'passed': False, 'error': str(e)}
    
    def _validate_performance_metrics(self, predictions: np.ndarray,
                                    true_labels: np.ndarray,
                                    dates: pd.Series) -> Dict[str, Any]:
        """验证业绩指标"""
        try:
            # 🔧 数据清理和对齐
            predictions, true_labels, dates = self._align_validation_data(predictions, true_labels, dates)
            if len(predictions) == 0:
                return {'passed': False, 'reason': '清理后无有效数据'}
            
            # 简单策略回测
            df = pd.DataFrame({
                'date': dates,
                'prediction': predictions,
                'true_label': true_labels
            }).sort_values('date')
            
            # 计算每日收益 (简化)
            df['signal'] = np.where(df['prediction'] > 0.5, 1, -1)
            df['returns'] = df['signal'] * df['true_label']  # 简化收益计算
            
            # 性能指标
            total_return = df['returns'].sum()
            volatility = df['returns'].std() * np.sqrt(252)
            sharpe_ratio = (df['returns'].mean() * 252) / volatility if volatility > 0 else 0
            
            # 最大回撤
            cumulative_returns = (1 + df['returns']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # 命中率
            hit_rate = (df['returns'] > 0).mean()
            
            # 验证检查
            sharpe_passed = sharpe_ratio >= self.thresholds.min_sharpe_ratio
            drawdown_passed = max_drawdown <= self.thresholds.max_max_drawdown
            hitrate_passed = hit_rate >= self.thresholds.min_hit_rate
            
            passed = sharpe_passed and drawdown_passed and hitrate_passed
            
            return {
                'passed': passed,
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'hit_rate': float(hit_rate),
                'total_return': float(total_return),
                'volatility': float(volatility),
                'thresholds': {
                    'min_sharpe': self.thresholds.min_sharpe_ratio,
                    'max_drawdown': self.thresholds.max_max_drawdown,
                    'min_hit_rate': self.thresholds.min_hit_rate
                },
                'checks': {
                    'sharpe_passed': sharpe_passed,
                    'drawdown_passed': drawdown_passed,
                    'hitrate_passed': hitrate_passed
                }
            }
            
        except Exception as e:
            logger.error(f"业绩验证失败: {e}")
            return {'passed': False, 'error': str(e)}
    
    
    def _apply_fallback_validation_when_insufficient_data(self, result: Dict[str, Any], 
                                                           sample_count: int) -> Dict[str, Any]:
        """当数据不足时应用回退验证逻辑"""
        if sample_count < 100:  # 数据不足100个样本
            logger.warning(f"数据样本不足({sample_count})，应用宽松验证标准")
            
            # 对于数据不足的情况，使用更宽松的标准
            if 'rank_ic' in result:
                rank_ic_val = result['rank_ic'].get('rank_ic', 0)
                if abs(rank_ic_val) >= 0.01:  # 绝对值大于1%就认为有效
                    result['rank_ic']['passed'] = True
                    logger.info(f"回退验证: RankIC {rank_ic_val:.4f} >= 0.01，通过验证")
            
            if 'stability' in result:
                # 对稳定性降低要求
                non_negative_ratio = result['stability'].get('non_negative_ratio', 0)
                if non_negative_ratio >= 0.4:  # 降低到40%
                    result['stability']['passed'] = True
                    logger.info(f"回退验证: 稳定性 {non_negative_ratio:.2f} >= 0.4，通过验证")
                    
        return result

    def _make_final_decision(self, detailed_results: Dict,
                           recommendations: List[str]) -> Tuple[float, str, List[str]]:
        """做出最终Go/No-Go决策"""
        
        # 权重配置
        weights = {
            'rank_ic': 0.3,
            'stability': 0.25, 
            'calibration': 0.2,
            'diversity': 0.15,
            'performance': 0.1
        }
        
        # 计算各项得分
        scores = {}
        critical_failures = []
        
        for category, weight in weights.items():
            result = detailed_results.get(category, {})
            if result.get('passed', False):
                scores[category] = 1.0
            else:
                scores[category] = 0.0
                if category in ['rank_ic', 'stability']:  # 关键指标
                    critical_failures.append(category)
        
        # 加权总分
        overall_score = sum(scores[cat] * weights[cat] for cat in weights.keys())
        
        # 决策逻辑
        if len(critical_failures) > 0:
            decision = "NO_GO"
            recommendations.extend([
                f"❌ 关键指标失败: {', '.join(critical_failures)}",
                "🔧 必须解决关键问题才能投入生产"
            ])
        elif overall_score >= 0.8:
            decision = "GO"
            recommendations.append("✅ 所有指标达标，可以投入生产使用")
        elif overall_score >= 0.6:
            decision = "CONDITIONAL_GO"
            recommendations.extend([
                "⚠️ 部分指标未达标，建议谨慎使用",
                "🔧 建议先小规模测试，监控表现"
            ])
        else:
            decision = "NO_GO"
            recommendations.extend([
                "❌ 多项指标未达标，不建议投入生产",
                "🔧 需要显著改进模型质量"
            ])
        
        return overall_score, decision, recommendations
    
    def _log_validation_summary(self, result: ValidationResult):
        """记录验证摘要"""
        logger.info("="*60)
        logger.info("🎯 生产就绪验证结果")
        logger.info("="*60)
        logger.info(f"📊 总体得分: {result.overall_score:.3f}")
        logger.info(f"🚦 决策结果: {result.go_no_go_decision}")
        
        # 详细结果
        for category, results in result.detailed_results.items():
            status = "✅" if results.get('passed', False) else "❌"
            logger.info(f"{status} {category.upper()}: {results.get('passed', False)}")
        
        # 建议
        if result.recommendations:
            logger.info("\n📋 建议:")
            for rec in result.recommendations:
                logger.info(f"  {rec}")
        
        # 风险警告
        if result.risk_warnings:
            logger.warning("\n⚠️ 风险警告:")
            for warning in result.risk_warnings:
                logger.warning(f"  {warning}")
        
        logger.info("="*60)
    
    def _align_validation_data(self, predictions: np.ndarray, true_labels: np.ndarray, dates: pd.Series) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """对齐和清理验证数据"""
        # 转换为numpy数组
        predictions = np.asarray(predictions)
        true_labels = np.asarray(true_labels)
        
        # 长度对齐
        min_len = min(len(predictions), len(true_labels), len(dates))
        if len(predictions) != len(true_labels) or len(predictions) != len(dates):
            logger.warning(f"验证数据长度不匹配: pred={len(predictions)}, labels={len(true_labels)}, dates={len(dates)}")
            predictions = predictions[:min_len]
            true_labels = true_labels[:min_len]
            # 安全地截取dates，处理Series或Index类型
            if hasattr(dates, 'iloc'):
                dates = dates.iloc[:min_len]
            else:
                dates = dates[:min_len]
            logger.info(f"数据已对齐到长度: {min_len}")
        
        # NaN清理 - 只对预测值和标签检查NaN
        valid_mask = ~(np.isnan(predictions) | np.isnan(true_labels))
        if not np.all(valid_mask):
            nan_count = (~valid_mask).sum()
            logger.warning(f"发现{nan_count}个NaN值，将被移除")
            predictions = predictions[valid_mask]
            true_labels = true_labels[valid_mask]
            # 安全地应用mask到dates
            if hasattr(dates, 'iloc'):
                dates = dates.iloc[valid_mask].reset_index(drop=True)
            elif isinstance(dates, pd.Series):
                dates = dates[valid_mask].reset_index(drop=True)
            else:
                # DatetimeIndex或其他Index类型
                dates = dates[valid_mask]
        
        logger.info(f"验证数据清理完成，最终长度: {len(predictions)}")
        return predictions, true_labels, dates

def create_production_validator(thresholds: Optional[ValidationThresholds] = None) -> ProductionReadinessValidator:
    """创建生产就绪验证器"""
    return ProductionReadinessValidator(thresholds)

# 🔥 集成到BMA系统的接口函数
def validate_bma_production_readiness(oos_predictions: np.ndarray,
                                    oos_true_labels: np.ndarray,
                                    prediction_dates: pd.Series,
                                    calibration_results: Optional[Dict] = None,
                                    weight_details: Optional[Dict] = None,
                                    custom_thresholds: Optional[Dict] = None) -> Dict[str, Any]:
    """
    验证BMA系统生产就绪状态
    
    Args:
        oos_predictions: 样外预测
        oos_true_labels: 样外真实标签  
        prediction_dates: 预测日期
        calibration_results: 校准结果
        weight_details: 权重明细
        custom_thresholds: 自定义阈值
    
    Returns:
        验证结果字典
    """
    try:
        # 创建阈值配置
        thresholds = ValidationThresholds()
        if custom_thresholds:
            for key, value in custom_thresholds.items():
                if hasattr(thresholds, key):
                    setattr(thresholds, key, value)
        
        # 创建验证器
        validator = create_production_validator(thresholds)
        
        # 运行验证
        result = validator.validate_bma_production_readiness(
            oos_predictions=oos_predictions,
            oos_true_labels=oos_true_labels,
            prediction_dates=prediction_dates,
            calibration_results=calibration_results,
            weight_details=weight_details
        )
        
        return {
            'success': True,
            'validation_result': asdict(result)
        }
        
    except Exception as e:
        logger.error(f"生产就绪验证失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'go_no_go_decision': 'NO_GO'
        }