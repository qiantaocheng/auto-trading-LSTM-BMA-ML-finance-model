#!/usr/bin/env python3
"""
严格OOS校准系统 - 消除乐观偏差的回退路径
================================================
仅允许严格OOS校准，禁止"用测试当训练"的回退策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss
from datetime import datetime, timedelta
import warnings
try:
    from .unified_cv_policy import get_global_cv_policy
except ImportError:
    from unified_cv_policy import get_global_cv_policy

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class StrictCalibrationConfig:
    """严格校准配置"""
    # ==================== 基本校准配置 ====================
    calibration_method: str = "isotonic"       # isotonic/platt/none
    min_folds_required: int = 3                # 最少CV折数（严格要求）
    min_calibration_samples: int = 50          # 最少校准样本数
    
    # ==================== 严格OOS配置 ====================
    allow_full_sample_fallback: bool = False  # 禁止全样本回退
    strict_oos_only: bool = True               # 仅允许严格OOS
    require_temporal_gap: bool = True          # 要求时间间隔
    min_temporal_gap_days: int = 10            # 最小时间间隔天数
    
    # ==================== 质量控制 ====================
    max_calibration_error: float = 0.1        # 最大校准误差
    min_reliability_score: float = 0.7        # 最小可靠性分数
    calibration_stability_threshold: float = 0.05  # 校准稳定性阈值
    
    # ==================== 风险控制 ====================
    enable_risk_monitoring: bool = True       # 启用风险监控
    max_prediction_shift: float = 0.2         # 最大预测偏移
    outlier_percentile: float = 0.95          # 异常值百分位
    
    # ==================== 回退策略 ====================
    no_calibration_fallback: bool = True      # 无校准回退（返回原始预测）
    log_calibration_failures: bool = True     # 记录校准失败
    fail_fast_on_insufficient_data: bool = True  # 数据不足时快速失败

class CalibrationResult:
    """校准结果容器"""
    
    def __init__(self, success: bool = False):
        self.success = success
        self.calibrated_predictions: Optional[np.ndarray] = None
        self.calibration_model: Optional[Any] = None
        self.calibration_curve: Optional[Dict[str, np.ndarray]] = None
        self.performance_metrics: Dict[str, float] = {}
        self.risk_metrics: Dict[str, float] = {}
        self.temporal_safety: Dict[str, Any] = {}
        self.error_message: Optional[str] = None
        self.fallback_applied: bool = False
        self.raw_predictions: Optional[np.ndarray] = None

class StrictOOSCalibrator:
    """严格OOS校准器"""
    
    def __init__(self, config: StrictCalibrationConfig = None):
        """初始化校准器"""
        self.config = config or StrictCalibrationConfig()
        
        # 获取全局CV策略
        self.cv_policy = get_global_cv_policy()
        
        # 统计信息
        self.stats = {
            'calibrations_attempted': 0,
            'calibrations_succeeded': 0,
            'calibrations_failed': 0,
            'fallback_triggered': 0,
            'temporal_violations': 0,
            'insufficient_data_events': 0
        }
        
        # 校准历史记录
        self.calibration_history = []
        
        logger.info(f"严格OOS校准器初始化完成 - 方法: {self.config.calibration_method}")
    
    def calibrate_predictions(self, cv_results: Dict[str, Any],
                            raw_predictions: np.ndarray,
                            true_labels: Optional[np.ndarray] = None,
                            prediction_dates: Optional[pd.DatetimeIndex] = None) -> CalibrationResult:
        """
        校准预测结果（主接口）
        
        Args:
            cv_results: CV交叉验证结果
            raw_predictions: 原始预测值
            true_labels: 真实标签（可选，用于验证）
            prediction_dates: 预测日期（用于时间安全验证）
        
        Returns:
            校准结果对象
        """
        self.stats['calibrations_attempted'] += 1
        
        logger.info(f"开始严格OOS校准 - 样本数: {len(raw_predictions)}")
        
        # 创建校准结果对象
        result = CalibrationResult()
        result.raw_predictions = raw_predictions.copy()
        
        try:
            # 步骤1: 验证CV结果的时间安全性
            temporal_validation = self._validate_temporal_safety(cv_results, prediction_dates)
            result.temporal_safety = temporal_validation
            
            if not temporal_validation['is_safe']:
                self.stats['temporal_violations'] += 1
                return self._handle_calibration_failure(
                    result, f"时间安全验证失败: {temporal_validation['reason']}"
                )
            
            # 步骤2: 检查数据充分性
            data_sufficiency = self._check_data_sufficiency(cv_results)
            
            if not data_sufficiency['sufficient']:
                self.stats['insufficient_data_events'] += 1
                return self._handle_calibration_failure(
                    result, f"数据不足: {data_sufficiency['reason']}"
                )
            
            # 步骤3: 构建严格OOS校准数据
            calibration_data = self._build_oos_calibration_data(cv_results)
            
            if calibration_data['X_cal'].empty or calibration_data['y_cal'].empty:
                return self._handle_calibration_failure(
                    result, "无法构建OOS校准数据"
                )
            
            # 步骤4: 训练校准模型
            calibration_model = self._train_calibration_model(
                calibration_data['X_cal'], 
                calibration_data['y_cal']
            )
            
            if calibration_model is None:
                return self._handle_calibration_failure(
                    result, "校准模型训练失败"
                )
            
            # 步骤5: 应用校准
            calibrated_preds = self._apply_calibration(calibration_model, raw_predictions)
            
            # 步骤6: 验证校准质量
            quality_metrics = self._validate_calibration_quality(
                calibrated_preds, raw_predictions, calibration_data
            )
            
            if not quality_metrics['acceptable']:
                return self._handle_calibration_failure(
                    result, f"校准质量不达标: {quality_metrics['reason']}"
                )
            
            # 步骤7: 构建成功结果
            result.success = True
            result.calibrated_predictions = calibrated_preds
            result.calibration_model = calibration_model
            result.performance_metrics = quality_metrics['metrics']
            result.risk_metrics = self._compute_risk_metrics(
                calibrated_preds, raw_predictions
            )
            result.calibration_curve = self._compute_calibration_curve(
                calibration_data['X_cal'], calibration_data['y_cal'], calibration_model
            )
            
            self.stats['calibrations_succeeded'] += 1
            
            # 记录成功的校准历史
            self._record_calibration_history(result, cv_results)
            
            logger.info(f"校准成功完成 - Brier Score: {result.performance_metrics.get('brier_score', 'N/A'):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"校准过程异常: {e}")
            return self._handle_calibration_failure(result, f"校准异常: {str(e)}")
    
    def _validate_temporal_safety(self, cv_results: Dict[str, Any],
                                 prediction_dates: Optional[pd.DatetimeIndex]) -> Dict[str, Any]:
        """验证时间安全性"""
        validation_result = {
            'is_safe': True,
            'reason': None,
            'gap_days': None,
            'temporal_order_ok': True
        }
        
        try:
            # 检查CV结果是否包含时间信息
            if 'cv_scores' not in cv_results:
                validation_result.update({
                    'is_safe': False,
                    'reason': 'CV结果缺少时间信息'
                })
                return validation_result
            
            cv_scores = cv_results['cv_scores']
            
            # 检查是否有足够的fold
            if len(cv_scores) < self.config.min_folds_required:
                validation_result.update({
                    'is_safe': False,
                    'reason': f'CV折数不足: {len(cv_scores)} < {self.config.min_folds_required}'
                })
                return validation_result
            
            # 检查每个fold的时间间隔（如果有时间信息）
            if prediction_dates is not None and self.config.require_temporal_gap:
                # 假设最后一个fold代表最新的训练-验证切分
                last_fold = cv_scores[-1]
                
                if 'train_end_date' in last_fold and 'val_start_date' in last_fold:
                    train_end = pd.to_datetime(last_fold['train_end_date'])
                    val_start = pd.to_datetime(last_fold['val_start_date'])
                    
                    gap_days = (val_start - train_end).days
                    validation_result['gap_days'] = gap_days
                    
                    if gap_days < self.config.min_temporal_gap_days:
                        validation_result.update({
                            'is_safe': False,
                            'reason': f'时间间隔不足: {gap_days}天 < {self.config.min_temporal_gap_days}天'
                        })
                        return validation_result
            
            logger.debug("时间安全验证通过")
            return validation_result
            
        except Exception as e:
            logger.debug(f"时间安全验证异常: {e}")
            validation_result.update({
                'is_safe': False,
                'reason': f'时间安全验证异常: {str(e)}'
            })
            return validation_result
    
    def _check_data_sufficiency(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """检查数据充分性"""
        sufficiency_result = {
            'sufficient': True,
            'reason': None,
            'total_samples': 0,
            'validation_samples': 0
        }
        
        try:
            cv_scores = cv_results.get('cv_scores', [])
            
            if not cv_scores:
                sufficiency_result.update({
                    'sufficient': False,
                    'reason': 'CV结果为空'
                })
                return sufficiency_result
            
            # 计算总的验证样本数
            total_val_samples = sum(score.get('val_samples', 0) for score in cv_scores)
            total_train_samples = sum(score.get('train_samples', 0) for score in cv_scores)
            
            sufficiency_result.update({
                'total_samples': total_train_samples,
                'validation_samples': total_val_samples
            })
            
            if total_val_samples < self.config.min_calibration_samples:
                sufficiency_result.update({
                    'sufficient': False,
                    'reason': f'验证样本不足: {total_val_samples} < {self.config.min_calibration_samples}'
                })
                return sufficiency_result
            
            # 检查每个fold的最小样本数
            min_fold_samples = min(score.get('val_samples', 0) for score in cv_scores)
            
            if min_fold_samples < 10:  # 每折最少10个样本
                sufficiency_result.update({
                    'sufficient': False,
                    'reason': f'单折样本数过少: {min_fold_samples} < 10'
                })
                return sufficiency_result
            
            logger.debug(f"数据充分性检查通过 - 验证样本: {total_val_samples}")
            return sufficiency_result
            
        except Exception as e:
            logger.debug(f"数据充分性检查异常: {e}")
            sufficiency_result.update({
                'sufficient': False,
                'reason': f'数据充分性检查异常: {str(e)}'
            })
            return sufficiency_result
    
    def _build_oos_calibration_data(self, cv_results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """构建严格OOS校准数据"""
        try:
            cv_scores = cv_results.get('cv_scores', [])
            
            # 收集所有fold的OOS预测和真实值
            all_predictions = []
            all_true_values = []
            all_fold_indices = []
            
            for fold_idx, fold_score in enumerate(cv_scores):
                # 从fold结果中提取OOS预测（如果存在）
                if 'oos_predictions' in fold_score and 'oos_true_values' in fold_score:
                    fold_preds = np.array(fold_score['oos_predictions'])
                    fold_trues = np.array(fold_score['oos_true_values'])
                    
                    if len(fold_preds) == len(fold_trues) and len(fold_preds) > 0:
                        all_predictions.extend(fold_preds)
                        all_true_values.extend(fold_trues)
                        all_fold_indices.extend([fold_idx] * len(fold_preds))
            
            if not all_predictions:
                logger.warning("无法从CV结果中提取OOS预测数据")
                return {'X_cal': pd.DataFrame(), 'y_cal': pd.Series(dtype=float)}
            
            # 转换为DataFrame和Series
            X_cal = pd.DataFrame({
                'prediction': all_predictions,
                'fold_idx': all_fold_indices
            })
            y_cal = pd.Series(all_true_values)
            
            logger.debug(f"构建OOS校准数据完成 - 样本数: {len(X_cal)}")
            
            return {'X_cal': X_cal, 'y_cal': y_cal}
            
        except Exception as e:
            logger.error(f"构建OOS校准数据失败: {e}")
            return {'X_cal': pd.DataFrame(), 'y_cal': pd.Series(dtype=float)}
    
    def _train_calibration_model(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> Optional[Any]:
        """训练校准模型"""
        try:
            if self.config.calibration_method == "none":
                return None
            
            if len(X_cal) < self.config.min_calibration_samples:
                logger.warning(f"校准样本不足: {len(X_cal)} < {self.config.min_calibration_samples}")
                return None
            
            predictions = X_cal['prediction'].values
            true_values = y_cal.values
            
            if self.config.calibration_method == "isotonic":
                # 等渗回归校准
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(predictions, true_values)
                return calibrator
                
            elif self.config.calibration_method == "platt":
                # Platt校准（逻辑回归）
                # 需要将回归问题转换为分类问题
                binary_labels = (true_values > np.median(true_values)).astype(int)
                calibrator = LogisticRegression()
                calibrator.fit(predictions.reshape(-1, 1), binary_labels)
                return calibrator
            
            else:
                logger.warning(f"不支持的校准方法: {self.config.calibration_method}")
                return None
                
        except Exception as e:
            logger.error(f"校准模型训练失败: {e}")
            return None
    
    def _apply_calibration(self, calibration_model: Any, 
                          raw_predictions: np.ndarray) -> np.ndarray:
        """应用校准"""
        try:
            if calibration_model is None:
                return raw_predictions
            
            if self.config.calibration_method == "isotonic":
                return calibration_model.predict(raw_predictions)
                
            elif self.config.calibration_method == "platt":
                # Platt校准返回概率
                probabilities = calibration_model.predict_proba(raw_predictions.reshape(-1, 1))
                return probabilities[:, 1]  # 返回正类概率
            
            else:
                return raw_predictions
                
        except Exception as e:
            logger.error(f"应用校准失败: {e}")
            return raw_predictions
    
    def _validate_calibration_quality(self, calibrated_preds: np.ndarray,
                                    raw_predictions: np.ndarray,
                                    calibration_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """验证校准质量"""
        quality_result = {
            'acceptable': True,
            'reason': None,
            'metrics': {}
        }
        
        try:
            # 基本质量检查
            if np.any(np.isnan(calibrated_preds)) or np.any(np.isinf(calibrated_preds)):
                quality_result.update({
                    'acceptable': False,
                    'reason': '校准结果包含NaN或Inf值'
                })
                return quality_result
            
            # 计算校准指标（使用OOS数据）
            if not calibration_data['y_cal'].empty:
                try:
                    # 在OOS校准数据上评估校准质量
                    cal_preds = calibration_data['X_cal']['prediction'].values
                    cal_true = calibration_data['y_cal'].values
                    
                    # Brier分数（越小越好）
                    brier_score = brier_score_loss(
                        (cal_true > np.median(cal_true)).astype(int),
                        cal_preds
                    )
                    
                    # 校准误差（平均预测 vs 平均真实值）
                    calibration_error = abs(np.mean(cal_preds) - np.mean(cal_true))
                    
                    quality_result['metrics'] = {
                        'brier_score': float(brier_score),
                        'calibration_error': float(calibration_error),
                        'prediction_std': float(np.std(calibrated_preds)),
                        'prediction_range': float(np.ptp(calibrated_preds))
                    }
                    
                    # 质量阈值检查
                    if calibration_error > self.config.max_calibration_error:
                        quality_result.update({
                            'acceptable': False,
                            'reason': f'校准误差过大: {calibration_error:.4f} > {self.config.max_calibration_error}'
                        })
                        return quality_result
                    
                except Exception as e:
                    logger.debug(f"校准质量指标计算失败: {e}")
                    quality_result['metrics'] = {
                        'brier_score': np.nan,
                        'calibration_error': np.nan,
                        'prediction_std': float(np.std(calibrated_preds)),
                        'prediction_range': float(np.ptp(calibrated_preds))
                    }
            
            # 检查预测变化合理性
            pred_shift = np.mean(np.abs(calibrated_preds - raw_predictions))
            if pred_shift > self.config.max_prediction_shift:
                quality_result.update({
                    'acceptable': False,
                    'reason': f'预测偏移过大: {pred_shift:.4f} > {self.config.max_prediction_shift}'
                })
                return quality_result
            
            quality_result['metrics']['prediction_shift'] = float(pred_shift)
            
            logger.debug(f"校准质量验证通过 - 偏移: {pred_shift:.4f}")
            return quality_result
            
        except Exception as e:
            logger.error(f"校准质量验证异常: {e}")
            quality_result.update({
                'acceptable': False,
                'reason': f'校准质量验证异常: {str(e)}'
            })
            return quality_result
    
    def _compute_risk_metrics(self, calibrated_preds: np.ndarray,
                            raw_predictions: np.ndarray) -> Dict[str, float]:
        """计算风险指标"""
        try:
            risk_metrics = {}
            
            # 预测分布变化
            risk_metrics['prediction_shift_mean'] = float(np.mean(calibrated_preds - raw_predictions))
            risk_metrics['prediction_shift_std'] = float(np.std(calibrated_preds - raw_predictions))
            
            # 异常值检测
            shift_diff = np.abs(calibrated_preds - raw_predictions)
            risk_metrics['outlier_threshold'] = float(np.percentile(shift_diff, self.config.outlier_percentile * 100))
            risk_metrics['outlier_count'] = int(np.sum(shift_diff > risk_metrics['outlier_threshold']))
            risk_metrics['outlier_rate'] = float(risk_metrics['outlier_count'] / len(calibrated_preds))
            
            # 稳定性指标
            risk_metrics['prediction_volatility'] = float(np.std(calibrated_preds) / (np.mean(np.abs(calibrated_preds)) + 1e-8))
            
            return risk_metrics
            
        except Exception as e:
            logger.debug(f"风险指标计算失败: {e}")
            return {}
    
    def _compute_calibration_curve(self, X_cal: pd.DataFrame, y_cal: pd.Series,
                                 calibration_model: Any) -> Dict[str, np.ndarray]:
        """计算校准曲线"""
        try:
            if calibration_model is None or X_cal.empty:
                return {}
            
            predictions = X_cal['prediction'].values
            true_values = y_cal.values
            
            # 分bin计算校准曲线
            n_bins = min(10, len(predictions) // 5)  # 自适应bin数量
            
            bin_boundaries = np.linspace(predictions.min(), predictions.max(), n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_centers = []
            bin_true_means = []
            bin_pred_means = []
            bin_counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
                if bin_upper == predictions.max():  # 包含最大值
                    in_bin |= (predictions == bin_upper)
                
                if np.sum(in_bin) > 0:
                    bin_centers.append((bin_lower + bin_upper) / 2)
                    bin_true_means.append(np.mean(true_values[in_bin]))
                    bin_pred_means.append(np.mean(predictions[in_bin]))
                    bin_counts.append(np.sum(in_bin))
            
            return {
                'bin_centers': np.array(bin_centers),
                'bin_true_means': np.array(bin_true_means),
                'bin_pred_means': np.array(bin_pred_means),
                'bin_counts': np.array(bin_counts)
            }
            
        except Exception as e:
            logger.debug(f"校准曲线计算失败: {e}")
            return {}
    
    def _handle_calibration_failure(self, result: CalibrationResult, 
                                  error_message: str) -> CalibrationResult:
        """处理校准失败"""
        self.stats['calibrations_failed'] += 1
        
        result.success = False
        result.error_message = error_message
        
        if self.config.log_calibration_failures:
            logger.warning(f"校准失败: {error_message}")
        
        # 根据配置决定回退策略
        if self.config.no_calibration_fallback and result.raw_predictions is not None:
            # 回退到原始预测，但标记为回退
            result.calibrated_predictions = result.raw_predictions.copy()
            result.fallback_applied = True
            self.stats['fallback_triggered'] += 1
            
            logger.info("应用无校准回退策略 - 返回原始预测")
        
        # 记录失败历史
        self._record_calibration_history(result, {})
        
        return result
    
    def _record_calibration_history(self, result: CalibrationResult, 
                                  cv_results: Dict[str, Any]):
        """记录校准历史"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'success': result.success,
            'method': self.config.calibration_method,
            'error_message': result.error_message,
            'fallback_applied': result.fallback_applied,
            'performance_metrics': result.performance_metrics,
            'risk_metrics': result.risk_metrics,
            'temporal_safety': result.temporal_safety
        }
        
        self.calibration_history.append(history_entry)
        
        # 保持历史记录大小
        if len(self.calibration_history) > 100:
            self.calibration_history = self.calibration_history[-50:]
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """获取校准统计信息"""
        total_attempts = self.stats['calibrations_attempted']
        
        return {
            'calibration_stats': self.stats,
            'success_rate': (
                self.stats['calibrations_succeeded'] / max(1, total_attempts)
            ),
            'failure_rate': (
                self.stats['calibrations_failed'] / max(1, total_attempts)
            ),
            'fallback_rate': (
                self.stats['fallback_triggered'] / max(1, total_attempts)
            ),
            'temporal_violation_rate': (
                self.stats['temporal_violations'] / max(1, total_attempts)
            ),
            'config': self.config.__dict__,
            'recent_calibrations': self.calibration_history[-10:] if self.calibration_history else []
        }

# 全局严格OOS校准器
def create_strict_oos_calibrator(config: StrictCalibrationConfig = None) -> StrictOOSCalibrator:
    """创建严格OOS校准器"""
    return StrictOOSCalibrator(config)

if __name__ == "__main__":
    # 测试严格OOS校准器
    calibrator = create_strict_oos_calibrator()
    
    # 模拟CV结果
    mock_cv_results = {
        'cv_scores': [
            {
                'fold': 0,
                'oos_predictions': np.zeros(30),
                'oos_true_values': np.zeros(30),
                'val_samples': 30,
                'train_samples': 100
            },
            {
                'fold': 1,
                'oos_predictions': np.zeros(25),
                'oos_true_values': np.zeros(25),
                'val_samples': 25,
                'train_samples': 105
            },
            {
                'fold': 2,
                'oos_predictions': np.zeros(35),
                'oos_true_values': np.zeros(35),
                'val_samples': 35,
                'train_samples': 95
            }
        ]
    }
    
    # 模拟原始预测
    raw_predictions = np.zeros(100)
    
    # 测试校准
    result = calibrator.calibrate_predictions(mock_cv_results, raw_predictions)
    
    print("=== 严格OOS校准测试 ===")
    print(f"校准成功: {'是' if result.success else '否'}")
    if not result.success:
        print(f"失败原因: {result.error_message}")
    
    print(f"回退应用: {'是' if result.fallback_applied else '否'}")
    
    if result.performance_metrics:
        print(f"性能指标:")
        for key, value in result.performance_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    print(f"\n校准器统计:")
    stats = calibrator.get_calibration_stats()
    print(f"  成功率: {stats['success_rate']:.2%}")
    print(f"  失败率: {stats['failure_rate']:.2%}")
    print(f"  回退率: {stats['fallback_rate']:.2%}")