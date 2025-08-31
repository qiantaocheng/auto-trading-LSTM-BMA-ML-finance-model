#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一校准系统 - 合并高级校准和预测标定功能
实现等张回归、分桶校准、阈值扫描和实时标定
"""

import numpy as np
import pandas as pd
import pickle
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import deque
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class UnifiedCalibrationConfig:
    """统一校准配置"""
    # 基础配置
    method: str = 'isotonic'  # isotonic, platt, ridge, bins, linear
    
    # 历史数据配置
    lookback_days: int = 252
    min_observations: int = 100
    
    # 校准参数
    n_bins: int = 10
    min_samples_per_bin: int = 50
    confidence_level: float = 0.95
    
    # 更新频率
    update_frequency_days: int = 30
    force_refit_days: int = 90
    
    # 阈值配置
    threshold_scan_range: Tuple[float, float] = (0.1, 0.9)
    threshold_scan_steps: int = 20  # 简化步数
    
    # 输出调整
    output_scale: float = 1.0
    apply_volatility_scaling: bool = True
    target_volatility: float = 0.02
    
    # 数据过滤
    score_bounds: Tuple[float, float] = (-5.0, 5.0)
    return_bounds: Tuple[float, float] = (-0.2, 0.2)
    
@dataclass 
class ThresholdConfig:
    """简化的阈值配置"""
    buy_threshold: float = 0.6
    sell_threshold: float = 0.4
    confidence_threshold: float = 0.7
    version: str = "1.0.0"
    created_date: str = ""
    
@dataclass
class CalibrationResult:
    """校准结果"""
    calibrated_probs: np.ndarray
    brier_score: float
    log_loss_score: float
    r_squared: float
    threshold_config: ThresholdConfig
    method_used: str
    calibration_date: str

class UnifiedCalibrationSystem:
    """统一校准系统 - 合并高级校准和实时标定功能"""
    
    def __init__(self, config: Optional[UnifiedCalibrationConfig] = None):
        self.config = config or UnifiedCalibrationConfig()
        self.calibrator = None
        self.threshold_config = None
        
        # 实时数据存储
        self._calibration_data = deque(maxlen=500)
        self._last_fit_time = None
        self._is_fitted = False
        
        # 缓存
        self.cache_dir = Path("cache/calibration")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'n_observations': 0,
            'last_r2_score': 0.0,
            'last_fit_time': None
        }
        
        self._lock = threading.RLock()
        logger.info(f"Unified calibration system initialized with method: {self.config.method}")
    
    def add_observation(self, 
                       scores: Union[pd.Series, Dict[str, float]],
                       realized_returns: Union[pd.Series, Dict[str, float]],
                       timestamp: datetime = None):
        """添加实时观测数据"""
        timestamp = timestamp or datetime.now()
        
        # 数据格式转换和对齐
        if isinstance(scores, dict):
            scores = pd.Series(scores)
        if isinstance(realized_returns, dict):
            realized_returns = pd.Series(realized_returns)
        
        common_stocks = scores.index.intersection(realized_returns.index)
        if len(common_stocks) == 0:
            return
        
        aligned_scores = scores.loc[common_stocks]
        aligned_returns = realized_returns.loc[common_stocks]
        
        # 数据过滤
        score_mask = ((aligned_scores >= self.config.score_bounds[0]) & 
                     (aligned_scores <= self.config.score_bounds[1]))
        return_mask = ((aligned_returns >= self.config.return_bounds[0]) & 
                      (aligned_returns <= self.config.return_bounds[1]))
        valid_mask = score_mask & return_mask & aligned_scores.notna() & aligned_returns.notna()
        
        if valid_mask.sum() == 0:
            return
        
        # 存储观测
        with self._lock:
            self._calibration_data.append({
                'timestamp': timestamp,
                'scores': aligned_scores[valid_mask],
                'returns': aligned_returns[valid_mask]
            })
            
            self.stats['n_observations'] = sum(len(obs['scores']) for obs in self._calibration_data)
            
            # 检查是否需要重新拟合
            if self._should_refit():
                self.fit_calibrator_from_history()
    
    def fit_calibrator(self, 
                      predictions: np.ndarray, 
                      true_labels: np.ndarray,
                      validation_set: bool = True) -> CalibrationResult:
        """拟合校准器 - 批量训练模式"""
        try:
            if not validation_set:
                raise ValueError("校准器只能用验证/历史数据拟合")
        
            logger.info(f"开始校准器训练，方法: {self.config.method}")
            
            # 数据清理和对齐
            predictions, true_labels = self._clean_and_align_data(predictions, true_labels)
            
            if len(predictions) < self.config.min_observations:
                logger.warning(f"样本数量不足: {len(predictions)} < {self.config.min_observations}")
            
            # 归一化预测值
            predictions_norm = self._normalize_predictions(predictions)
            
            # 选择校准方法
            calibrated_probs = self._fit_by_method(predictions_norm, true_labels)
            
            # 计算校准指标
            brier_score = self._safe_brier_score(true_labels, calibrated_probs)
            log_loss_score = self._safe_log_loss(true_labels, calibrated_probs)
            r_squared = self._compute_r_squared(true_labels, calibrated_probs)
            
            # 阈值优化
            threshold_config = self._optimize_thresholds(calibrated_probs, true_labels)
            
            result = CalibrationResult(
                calibrated_probs=calibrated_probs,
                brier_score=brier_score,
                log_loss_score=log_loss_score,
                r_squared=r_squared,
                threshold_config=threshold_config,
                method_used=self.config.method,
                calibration_date=datetime.now().isoformat()
            )
            
            # 保存结果
            self._save_calibration_artifacts(result)
            
            logger.info(f"校准完成 - Brier: {brier_score:.4f}, R²: {r_squared:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"校准失败: {e}")
            return self._create_fallback_result(predictions, true_labels)
    
    def calibrate_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """实时校准预测"""
        if not self._is_fitted or self.calibrator is None:
            logger.warning("校准器未训练，返回原始预测")
            return predictions
        
        try:
            predictions_norm = self._normalize_predictions(predictions)
            
            if self.config.method == 'isotonic':
                return self.calibrator.transform(predictions_norm)
            elif self.config.method == 'linear':
                return self.calibrator.predict(predictions_norm.reshape(-1, 1))
            elif self.config.method == 'ridge':
                return self.calibrator.predict(predictions_norm.reshape(-1, 1))
            else:
                return predictions_norm
                
        except Exception as e:
            logger.error(f"预测校准失败: {e}")
            return predictions
    
    def _fit_by_method(self, predictions: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
        """根据方法拟合校准器"""
        if self.config.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            return self.calibrator.fit_transform(predictions, true_labels)
            
        elif self.config.method == 'linear':
            self.calibrator = LinearRegression()
            self.calibrator.fit(predictions.reshape(-1, 1), true_labels)
            return self.calibrator.predict(predictions.reshape(-1, 1))
            
        elif self.config.method == 'ridge':
            self.calibrator = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            self.calibrator.fit(predictions.reshape(-1, 1), true_labels)
            return self.calibrator.predict(predictions.reshape(-1, 1))
            
        elif self.config.method == 'bins':
            return self._fit_binned_calibration(predictions, true_labels)
            
        else:
            raise ValueError(f"不支持的校准方法: {self.config.method}")
    
    def _normalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """标准化预测值到[0,1]范围"""
        if len(predictions) == 0:
            return predictions
            
        if np.all((predictions >= 0) & (predictions <= 1)):
            return predictions
            
        # 使用sigmoid变换
        predictions = np.clip(predictions, -10, 10)
        return 1 / (1 + np.exp(-predictions))
    
    def _clean_and_align_data(self, predictions: np.ndarray, true_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """清理和对齐数据"""
        # 移除NaN值
        valid_mask = ~(np.isnan(predictions) | np.isnan(true_labels))
        predictions = predictions[valid_mask]
        true_labels = true_labels[valid_mask]
        
        # 确保长度一致
        min_len = min(len(predictions), len(true_labels))
        return predictions[:min_len], true_labels[:min_len]
    
    def _safe_brier_score(self, true_labels: np.ndarray, predictions: np.ndarray) -> float:
        """安全计算Brier分数"""
        try:
            unique_labels = np.unique(true_labels[~np.isnan(true_labels)])
            
            if len(unique_labels) == 2 and set(unique_labels).issubset({0, 1}):
                return brier_score_loss(true_labels, predictions)
            else:
                # 回归情况 - 使用MSE
                mse = np.mean((true_labels - predictions) ** 2)
                return min(1.0, mse / max(np.var(true_labels), 1e-8))
        except Exception:
            return 0.25
    
    def _safe_log_loss(self, true_labels: np.ndarray, predictions: np.ndarray) -> float:
        """安全计算对数损失"""
        try:
            unique_labels = np.unique(true_labels[~np.isnan(true_labels)])
            if len(unique_labels) == 2 and set(unique_labels).issubset({0, 1}):
                predictions_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
                return log_loss(true_labels, predictions_clipped)
            else:
                # 回归情况
                return float(np.mean((true_labels - predictions) ** 2))
        except Exception:
            return 1.0
    
    def _compute_r_squared(self, true_labels: np.ndarray, predictions: np.ndarray) -> float:
        """计算R²"""
        try:
            ss_res = np.sum((true_labels - predictions) ** 2)
            ss_tot = np.sum((true_labels - np.mean(true_labels)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-8))
        except Exception:
            return 0.0
    
    def _optimize_thresholds(self, predictions: np.ndarray, true_labels: np.ndarray) -> ThresholdConfig:
        """简化的阈值优化"""
        try:
            # 使用简单的分位数方法
            buy_threshold = np.percentile(predictions, 70)
            sell_threshold = np.percentile(predictions, 30)
            
            return ThresholdConfig(
                buy_threshold=float(buy_threshold),
                sell_threshold=float(sell_threshold),
                confidence_threshold=0.7,
                version="1.0.0",
                created_date=datetime.now().isoformat()
            )
        except Exception:
            return ThresholdConfig()
    
    def _fit_binned_calibration(self, predictions: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
        """分桶校准"""
        try:
            n_bins = min(self.config.n_bins, len(predictions) // 10)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibrated = np.zeros_like(predictions)
            
            for i in range(n_bins):
                bin_mask = (predictions >= bin_lowers[i]) & (predictions < bin_uppers[i])
                if i == n_bins - 1:  # 最后一个bin包含上边界
                    bin_mask = (predictions >= bin_lowers[i]) & (predictions <= bin_uppers[i])
                
                if np.sum(bin_mask) > 0:
                    calibrated[bin_mask] = np.mean(true_labels[bin_mask])
                else:
                    calibrated[bin_mask] = (bin_lowers[i] + bin_uppers[i]) / 2
            
            return calibrated
        except Exception as e:
            logger.error(f"分桶校准失败: {e}")
            return predictions
    
    def fit_calibrator_from_history(self):
        """从历史数据拟合校准器"""
        with self._lock:
            if len(self._calibration_data) < 10:
                return
            
            # 收集历史数据
            all_scores = []
            all_returns = []
            
            for obs in self._calibration_data:
                all_scores.extend(obs['scores'].values)
                all_returns.extend(obs['returns'].values)
            
            if len(all_scores) < self.config.min_observations:
                return
            
            scores_array = np.array(all_scores)
            returns_array = np.array(all_returns)
            
            # 拟合校准器
            try:
                self.fit_calibrator(scores_array, returns_array)
                self._is_fitted = True
                self._last_fit_time = datetime.now()
                logger.info(f"从历史数据更新校准器，样本数: {len(all_scores)}")
            except Exception as e:
                logger.error(f"历史数据校准失败: {e}")
    
    def _should_refit(self) -> bool:
        """判断是否需要重新拟合"""
        if not self._is_fitted:
            return True
        
        if self._last_fit_time is None:
            return True
        
        days_since_fit = (datetime.now() - self._last_fit_time).days
        return days_since_fit >= self.config.update_frequency_days
    
    def _save_calibration_artifacts(self, result: CalibrationResult):
        """保存校准结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存校准器
            if self.calibrator is not None:
                calibrator_path = self.cache_dir / f"calibrator_{timestamp}.pkl"
                with open(calibrator_path, 'wb') as f:
                    pickle.dump(self.calibrator, f)
            
            # 保存配置
            config_path = self.cache_dir / f"threshold_config_{timestamp}.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(result.threshold_config), f, indent=2)
            
            logger.info(f"校准结果已保存到 {self.cache_dir}")
            
        except Exception as e:
            logger.error(f"保存校准结果失败: {e}")
    
    def _create_fallback_result(self, predictions: np.ndarray, true_labels: np.ndarray) -> CalibrationResult:
        """创建回退校准结果"""
        return CalibrationResult(
            calibrated_probs=predictions,
            brier_score=0.25,
            log_loss_score=1.0,
            r_squared=0.0,
            threshold_config=ThresholdConfig(),
            method_used="fallback",
            calibration_date=datetime.now().isoformat()
        )

# 便利函数
def get_unified_calibrator(config: Optional[UnifiedCalibrationConfig] = None) -> UnifiedCalibrationSystem:
    """获取统一校准器实例"""
    return UnifiedCalibrationSystem(config)

def create_calibration_config(method: str = 'isotonic', 
                             lookback_days: int = 252) -> UnifiedCalibrationConfig:
    """创建校准配置"""
    return UnifiedCalibrationConfig(method=method, lookback_days=lookback_days)
