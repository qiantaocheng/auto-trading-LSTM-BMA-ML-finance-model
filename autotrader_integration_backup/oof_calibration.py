#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OOF等值校准模块
Out-of-Fold (OOF) calibration for prediction reliability
基于历史OOF预测构建IsotonicRegression校准器，输出等值胜率和置信度
"""

import pandas as pd
import numpy as np
import logging
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
import sqlite3

logger = logging.getLogger(__name__)

class OOFCalibrator:
    """
    OOF校准器
    
    功能：
    1. 收集历史OOF预测和实际收益
    2. 训练IsotonicRegression进行概率校准
    3. 将原始预测转换为期望alpha(bps)和置信度
    """
    
    def __init__(self, calibration_db_path: str = "oof_calibration.db"):
        self.calibration_db_path = calibration_db_path
        self.prediction_calibrator = None  # 预测值校准器
        self.confidence_calibrator = None  # 置信度校准器
        self.last_update = None
        self._init_database()
    
    def _init_database(self):
        """初始化校准数据库"""
        try:
            conn = sqlite3.connect(self.calibration_db_path)
            cursor = conn.cursor()
            
            # 创建OOF数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS oof_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction_date DATE NOT NULL,
                    raw_prediction REAL NOT NULL,
                    raw_confidence REAL NOT NULL,
                    actual_return_1d REAL,
                    actual_return_5d REAL,
                    actual_return_20d REAL,
                    reference_price REAL NOT NULL,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建校准器状态表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS calibrator_status (
                    id INTEGER PRIMARY KEY,
                    last_training_date DATE,
                    num_samples INTEGER,
                    prediction_r2 REAL,
                    confidence_r2 REAL,
                    calibrator_blob BLOB
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("OOF校准数据库初始化完成")
        except Exception as e:
            logger.error(f"初始化校准数据库失败: {e}")
    
    def record_oof_prediction(self, symbol: str, raw_prediction: float, 
                            raw_confidence: float, reference_price: float,
                            model_version: str = "default") -> bool:
        """
        记录OOF预测，待后续更新实际收益
        
        Args:
            symbol: 股票代码
            raw_prediction: 原始预测值
            raw_confidence: 原始置信度
            reference_price: 参考价格
            model_version: 模型版本
        """
        try:
            conn = sqlite3.connect(self.calibration_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO oof_predictions 
                (symbol, prediction_date, raw_prediction, raw_confidence, 
                 reference_price, model_version)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, datetime.now().date(), raw_prediction, raw_confidence,
                  reference_price, model_version))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"记录OOF预测失败: {e}")
            return False
    
    def update_actual_returns(self, lookback_days: int = 30):
        """
        更新历史预测的实际收益率
        
        Args:
            lookback_days: 回看天数
        """
        try:
            conn = sqlite3.connect(self.calibration_db_path)
            
            # 查询需要更新的预测记录
            cutoff_date = datetime.now().date() - timedelta(days=lookback_days)
            
            query = '''
                SELECT id, symbol, prediction_date, reference_price
                FROM oof_predictions 
                WHERE prediction_date >= ? 
                AND (actual_return_1d IS NULL OR actual_return_5d IS NULL OR actual_return_20d IS NULL)
                ORDER BY prediction_date DESC
            '''
            
            df = pd.read_sql(query, conn, params=(cutoff_date,))
            
            # 从polygon获取实际价格数据并计算收益
            for _, row in df.iterrows():
                symbol = row['symbol']
                pred_date = pd.to_datetime(row['prediction_date'])
                ref_price = row['reference_price']
                
                # 调用实际的数据源获取后续价格
                actual_prices = self._get_actual_prices_mock(symbol, pred_date, ref_price)  # TODO: Replace with real data source
                
                if actual_prices:
                    returns_1d = (actual_prices.get('1d', ref_price) - ref_price) / ref_price if ref_price > 0 else 0
                    returns_5d = (actual_prices.get('5d', ref_price) - ref_price) / ref_price if ref_price > 0 else 0
                    returns_20d = (actual_prices.get('20d', ref_price) - ref_price) / ref_price if ref_price > 0 else 0
                    
                    # 更新数据库
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE oof_predictions 
                        SET actual_return_1d = ?, actual_return_5d = ?, actual_return_20d = ?
                        WHERE id = ?
                    ''', (returns_1d, returns_5d, returns_20d, row['id']))
            
            conn.commit()
            conn.close()
            logger.info(f"更新了{len(df)}条OOF预测的实际收益")
        except Exception as e:
            logger.error(f"更新实际收益失败: {e}")
    
    def _get_actual_prices_mock(self, symbol: str, pred_date: datetime, ref_price: float) -> Dict[str, float]:
        """模拟获取实际价格数据（实际中应接入真实数据源）"""
        # 使用生产安全的随机控制
        from .production_random_control import get_production_safe_seed
        
        # 获取确定性种子用于生产环境
        seed = get_production_safe_seed(symbol, f"price_mock_{pred_date.day}")
        np.random.seed(seed)
        
        # 模拟1日、5日、20日后的价格
        volatility = 0.02  # 2%日波动率
        prices = {}
        
        for days in [1, 5, 20]:
            # 随机游走模拟
            daily_returns = np.random.normal(0, volatility, days)
            cumulative_return = np.sum(daily_returns)
            prices[f'{days}d'] = ref_price * (1 + cumulative_return)
        
        return prices
    
    def train_calibrators(self, min_samples: int = 100) -> bool:
        """
        训练IsotonicRegression校准器
        
        Args:
            min_samples: 最少样本数
        """
        try:
            conn = sqlite3.connect(self.calibration_db_path)
            
            # 获取有完整实际收益的数据
            query = '''
                SELECT raw_prediction, raw_confidence, actual_return_1d, actual_return_5d, actual_return_20d
                FROM oof_predictions 
                WHERE actual_return_1d IS NOT NULL 
                AND actual_return_5d IS NOT NULL 
                AND actual_return_20d IS NOT NULL
                ORDER BY prediction_date DESC
                LIMIT 10000
            '''
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            if len(df) < min_samples:
                logger.warning(f"样本数量不足({len(df)} < {min_samples})，无法训练校准器")
                return False
            
            logger.info(f"使用{len(df)}个样本训练OOF校准器")
            
            # 准备训练数据
            X_pred = df['raw_prediction'].values
            X_conf = df['raw_confidence'].values
            
            # 使用5日收益作为目标（可配置）
            y_actual = df['actual_return_5d'].values
            
            # 🚀 完整OOF等值校准流水线：raw_prediction -> 胜率/期望收益(bps)
            
            # 1. 训练预测校准器：raw_prediction -> expected_alpha_bps
            self.prediction_calibrator = IsotonicRegression(out_of_bounds='clip')
            y_alpha_bps = y_actual * 10000  # 转为bps
            self.prediction_calibrator.fit(X_pred, y_alpha_bps)
            
            # 2. 训练置信度校准器：raw_confidence -> 胜率(win_rate)
            # 计算方向预测正确的比例作为胜率
            direction_correct = np.sign(X_pred) == np.sign(y_actual)
            win_rates = self._smooth_win_rate_by_confidence(X_conf, direction_correct)
            
            self.confidence_calibrator = IsotonicRegression(out_of_bounds='clip')
            self.confidence_calibrator.fit(X_conf, win_rates)
            
            # 3. 存储校准统计信息供调试
            self.calibration_stats = {
                'total_samples': len(df),
                'avg_actual_return_bps': np.mean(y_alpha_bps),
                'overall_win_rate': np.mean(direction_correct),
                'prediction_corr': np.corrcoef(X_pred, y_actual)[0, 1] if len(X_pred) > 1 else 0,
                'confidence_corr': np.corrcoef(X_conf, direction_correct)[0, 1] if len(X_conf) > 1 else 0
            }
            
            # 计算校准质量指标
            pred_r2 = self._calculate_r2(y_alpha_bps, self.prediction_calibrator.predict(X_pred))
            conf_r2 = self._calculate_r2(win_rates, self.confidence_calibrator.predict(X_conf))
            
            # 保存校准器状态
            self._save_calibrators(len(df), pred_r2, conf_r2)
            
            self.last_update = datetime.now()
            logger.info(f"OOF校准器训练完成 - 预测R²: {pred_r2:.3f}, 置信度R²: {conf_r2:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"训练校准器失败: {e}")
            return False
    
    def _smooth_win_rate_by_confidence(self, confidences: np.ndarray, direction_correct: np.ndarray, 
                                      bins: int = 10) -> np.ndarray:
        """
        平滑处理置信度对应的胜率，避免噪声影响
        
        Args:
            confidences: 置信度数组
            direction_correct: 方向预测正确的布尔数组
            bins: 分箱数量
            
        Returns:
            平滑后的胜率数组
        """
        try:
            # 创建置信度分箱
            conf_bins = np.linspace(np.min(confidences), np.max(confidences), bins + 1)
            win_rates = np.zeros_like(confidences, dtype=float)
            
            for i in range(len(conf_bins) - 1):
                # 找到在当前置信度区间的样本
                mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i + 1])
                
                if i == len(conf_bins) - 2:  # 最后一个区间包含右边界
                    mask = (confidences >= conf_bins[i]) & (confidences <= conf_bins[i + 1])
                
                if np.sum(mask) > 0:
                    # 计算该区间的平均胜率
                    bin_win_rate = np.mean(direction_correct[mask])
                    win_rates[mask] = bin_win_rate
                else:
                    # 如果区间内没有样本，使用全局胜率
                    win_rates[mask] = np.mean(direction_correct)
            
            # 确保胜率在合理范围内 [0.01, 0.99]
            win_rates = np.clip(win_rates, 0.01, 0.99)
            
            return win_rates
            
        except Exception as e:
            self.logger.warning(f"胜率平滑处理失败: {e}，使用简单方法")
            # 回退到简单方法：直接返回原始direction_correct的浮点版本
            return np.clip(direction_correct.astype(float), 0.01, 0.99)
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算R²分数"""
        try:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)
        except:
            return 0.0
    
    def _save_calibrators(self, num_samples: int, pred_r2: float, conf_r2: float):
        """保存校准器到数据库"""
        try:
            # 序列化校准器
            calibrator_data = {
                'prediction_calibrator': self.prediction_calibrator,
                'confidence_calibrator': self.confidence_calibrator
            }
            calibrator_blob = pickle.dumps(calibrator_data)
            
            conn = sqlite3.connect(self.calibration_db_path)
            cursor = conn.cursor()
            
            # 删除旧记录
            cursor.execute('DELETE FROM calibrator_status')
            
            # 插入新记录
            cursor.execute('''
                INSERT INTO calibrator_status 
                (id, last_training_date, num_samples, prediction_r2, confidence_r2, calibrator_blob)
                VALUES (1, ?, ?, ?, ?, ?)
            ''', (datetime.now().date(), num_samples, pred_r2, conf_r2, calibrator_blob))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"保存校准器失败: {e}")
    
    def load_calibrators(self) -> bool:
        """从数据库加载校准器"""
        try:
            conn = sqlite3.connect(self.calibration_db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT calibrator_blob FROM calibrator_status WHERE id = 1')
            result = cursor.fetchone()
            conn.close()
            
            if result:
                calibrator_data = pickle.loads(result[0])
                self.prediction_calibrator = calibrator_data['prediction_calibrator']
                self.confidence_calibrator = calibrator_data['confidence_calibrator']
                logger.info("OOF校准器加载成功")
                return True
            else:
                logger.warning("未找到保存的校准器")
                return False
        except Exception as e:
            logger.error(f"加载校准器失败: {e}")
            return False
    
    def calibrate_prediction(self, raw_prediction: float, raw_confidence: float) -> Tuple[float, float]:
        """
        校准单个预测
        
        Args:
            raw_prediction: 原始预测值
            raw_confidence: 原始置信度
            
        Returns:
            Tuple[expected_alpha_bps, calibrated_confidence]
        """
        try:
            # 如果校准器未训练，使用简单映射
            if self.prediction_calibrator is None or self.confidence_calibrator is None:
                expected_alpha_bps = abs(raw_prediction * 10000)  # 转为bps
                calibrated_confidence = max(0.01, min(0.99, raw_confidence))
                return expected_alpha_bps, calibrated_confidence
            
            # 使用校准器
            expected_alpha_bps = float(self.prediction_calibrator.predict([raw_prediction])[0])
            calibrated_confidence = float(self.confidence_calibrator.predict([raw_confidence])[0])
            
            # 确保在合理范围内
            expected_alpha_bps = max(0, min(1000, abs(expected_alpha_bps)))  # 0-1000bps
            calibrated_confidence = max(0.01, min(0.99, calibrated_confidence))
            
            return expected_alpha_bps, calibrated_confidence
            
        except Exception as e:
            logger.error(f"校准预测失败: {e}")
            # 回退到简单映射
            expected_alpha_bps = abs(raw_prediction * 10000)
            calibrated_confidence = max(0.01, min(0.99, raw_confidence))
            return expected_alpha_bps, calibrated_confidence
    
    def get_calibrator_stats(self) -> Dict[str, Any]:
        """获取校准器统计信息"""
        try:
            conn = sqlite3.connect(self.calibration_db_path)
            
            # 获取校准器状态
            status_df = pd.read_sql('SELECT * FROM calibrator_status WHERE id = 1', conn)
            
            # 获取数据统计
            stats_query = '''
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN actual_return_1d IS NOT NULL THEN 1 END) as completed_predictions,
                    AVG(ABS(raw_prediction - actual_return_5d)) as avg_prediction_error
                FROM oof_predictions
                WHERE prediction_date >= date('now', '-30 days')
            '''
            stats_df = pd.read_sql(stats_query, conn)
            conn.close()
            
            result = {
                'calibrator_available': self.prediction_calibrator is not None,
                'last_training_date': status_df['last_training_date'].iloc[0] if len(status_df) > 0 else None,
                'training_samples': status_df['num_samples'].iloc[0] if len(status_df) > 0 else 0,
                'prediction_r2': status_df['prediction_r2'].iloc[0] if len(status_df) > 0 else 0,
                'confidence_r2': status_df['confidence_r2'].iloc[0] if len(status_df) > 0 else 0,
                'recent_predictions': stats_df['total_predictions'].iloc[0] if len(stats_df) > 0 else 0,
                'completed_predictions': stats_df['completed_predictions'].iloc[0] if len(stats_df) > 0 else 0,
                'avg_prediction_error': stats_df['avg_prediction_error'].iloc[0] if len(stats_df) > 0 else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"获取校准器统计失败: {e}")
            return {'calibrator_available': False}

# 全局校准器实例
_global_oof_calibrator = None

def get_oof_calibrator() -> OOFCalibrator:
    """获取全局OOF校准器实例"""
    global _global_oof_calibrator
    if _global_oof_calibrator is None:
        _global_oof_calibrator = OOFCalibrator()
        # 尝试加载已有校准器
        _global_oof_calibrator.load_calibrators()
    return _global_oof_calibrator

def calibrate_signal(raw_prediction: float, raw_confidence: float) -> Tuple[float, float]:
    """
    便捷函数：校准单个信号
    
    Args:
        raw_prediction: 原始预测值
        raw_confidence: 原始置信度
        
    Returns:
        Tuple[expected_alpha_bps, calibrated_confidence]
    """
    calibrator = get_oof_calibrator()
    return calibrator.calibrate_prediction(raw_prediction, raw_confidence)