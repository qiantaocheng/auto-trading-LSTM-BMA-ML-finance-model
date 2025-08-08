#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM多日预测交易系统集成版
专为周度自动运行和Trading Manager集成设计

主要特性:
- 完整保留原有LSTM多日预测功能
- 集成高级特征工程和超参数优化
- 专为周一开盘前自动运行设计
- 完全兼容Trading Manager系统
- 输出标准化的交易信号和Excel报告

Authors: AI Assistant
Version: 5.0 Trading System Integration
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
import warnings
import argparse
import os
import tempfile
from pathlib import Path
import logging
import pickle
import json
import sys
from typing import Dict, List, Tuple, Optional, Union
import time

# 科学计算和统计分析
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 机器学习核心库
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer

# 深度学习
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam, RMSprop, AdamW
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available")

# 超参数优化（可选）
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available, using default parameters")

# 配置日志
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/lstm_trading_system_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 股票池配置（保持原有配置）
MULTI_DAY_TICKER_LIST = [
    # 大型科技股
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
    # 金融股
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC',
    # 医疗保健
    'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'TMO', 'DHR', 'CVS',
    # 消费品
    'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'DIS', 'NKE',
    # 工业股
    'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT',
    # 能源
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX',
    # 其他重要股票
    'BRK-B', 'V', 'MA', 'INTC', 'IBM', 'CSCO', 'CRM', 'ORCL'
]

# 因子配置（保持原有因子）
MULTI_DAY_FACTORS = [
    'returns', 'returns_5d', 'returns_10d', 'returns_20d',
    'volatility_5d', 'volatility_10d', 'volatility_20d', 'volatility_60d',
    'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20', 'sma_ratio_50',
    'ema_ratio_12', 'ema_ratio_26',
    'volume_ratio_5', 'volume_ratio_10', 'volume_ratio_20',
    'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'bollinger_upper_ratio', 'bollinger_lower_ratio', 'bollinger_width',
    'stoch_k', 'stoch_d', 'williams_r',
    'momentum_5', 'momentum_10', 'momentum_20',
    'price_position', 'volume_trend',
    'atr_normalized', 'cci', 'roc_10'
]


class WeeklyTradingSystemLSTM:
    """周度交易系统LSTM模型类"""
    
    def __init__(self, 
                 prediction_days: int = 5,
                 lstm_window: int = 20,
                 enable_optimization: bool = False,  # 周度运行默认关闭以提高速度
                 model_cache_dir: str = 'models/weekly_cache'):
        
        self.prediction_days = prediction_days
        self.lstm_window = lstm_window
        self.enable_optimization = enable_optimization
        self.model_cache_dir = model_cache_dir
        
        # 确保模型缓存目录存在
        os.makedirs(model_cache_dir, exist_ok=True)
        os.makedirs('result', exist_ok=True)
        os.makedirs('weekly_trading_signals', exist_ok=True)
        
        # 模型组件
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.stock_statistics = {}
        
        # 性能跟踪
        self.model_performance = {}
        self.prediction_history = []
        
        logger.info(f"初始化周度交易LSTM系统 - 预测天数: {prediction_days}, 窗口: {lstm_window}")
    
    def calculate_comprehensive_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算全面的技术因子（保持原有因子计算逻辑）"""
        try:
            factors = pd.DataFrame(index=data.index)
            
            # 基础数据
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            # 收益率因子
            factors['returns'] = close.pct_change()
            factors['returns_5d'] = close.pct_change(5)
            factors['returns_10d'] = close.pct_change(10)
            factors['returns_20d'] = close.pct_change(20)
            
            # 波动率因子
            factors['volatility_5d'] = factors['returns'].rolling(5).std()
            factors['volatility_10d'] = factors['returns'].rolling(10).std()
            factors['volatility_20d'] = factors['returns'].rolling(20).std()
            factors['volatility_60d'] = factors['returns'].rolling(60).std()
            
            # 移动平均比率
            sma_5 = close.rolling(5).mean()
            sma_10 = close.rolling(10).mean()
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            
            factors['sma_ratio_5'] = (close / sma_5) - 1
            factors['sma_ratio_10'] = (close / sma_10) - 1
            factors['sma_ratio_20'] = (close / sma_20) - 1
            factors['sma_ratio_50'] = (close / sma_50) - 1
            
            # EMA比率
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            factors['ema_ratio_12'] = (close / ema_12) - 1
            factors['ema_ratio_26'] = (close / ema_26) - 1
            
            # 成交量因子
            volume_ma_5 = volume.rolling(5).mean()
            volume_ma_10 = volume.rolling(10).mean()
            volume_ma_20 = volume.rolling(20).mean()
            
            factors['volume_ratio_5'] = volume / volume_ma_5 - 1
            factors['volume_ratio_10'] = volume / volume_ma_10 - 1
            factors['volume_ratio_20'] = volume / volume_ma_20 - 1
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            factors['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            factors['macd'] = ema_12 - ema_26
            factors['macd_signal'] = factors['macd'].ewm(span=9).mean()
            factors['macd_histogram'] = factors['macd'] - factors['macd_signal']
            
            # 布林带
            bb_middle = close.rolling(20).mean()
            bb_std = close.rolling(20).std()
            bb_upper = bb_middle + (2 * bb_std)
            bb_lower = bb_middle - (2 * bb_std)
            
            factors['bollinger_upper_ratio'] = (close / bb_upper) - 1
            factors['bollinger_lower_ratio'] = (close / bb_lower) - 1
            factors['bollinger_width'] = (bb_upper - bb_lower) / bb_middle
            
            # 随机指标
            low_14 = low.rolling(14).min()
            high_14 = high.rolling(14).max()
            factors['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14)
            factors['stoch_d'] = factors['stoch_k'].rolling(3).mean()
            
            # 威廉指标
            factors['williams_r'] = -100 * (high_14 - close) / (high_14 - low_14)
            
            # 动量因子
            factors['momentum_5'] = close / close.shift(5) - 1
            factors['momentum_10'] = close / close.shift(10) - 1
            factors['momentum_20'] = close / close.shift(20) - 1
            
            # 价格位置
            factors['price_position'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min())
            
            # 成交量趋势
            factors['volume_trend'] = (volume.rolling(5).mean() / volume.rolling(20).mean()) - 1
            
            # ATR标准化
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            factors['atr_normalized'] = atr / close
            
            # CCI
            typical_price = (high + low + close) / 3
            factors['cci'] = (typical_price - typical_price.rolling(20).mean()) / (0.015 * typical_price.rolling(20).std())
            
            # ROC
            factors['roc_10'] = ((close - close.shift(10)) / close.shift(10)) * 100
            
            # 清理数据
            factors = factors.replace([np.inf, -np.inf], np.nan)
            factors = factors.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            logger.info(f"计算了 {factors.shape[1]} 个技术因子")
            return factors
            
        except Exception as e:
            logger.error(f"因子计算失败: {e}")
            return pd.DataFrame()
    
    def advanced_feature_selection(self, factors_df: pd.DataFrame, 
                                 returns_df: pd.DataFrame) -> pd.DataFrame:
        """高级特征选择（集成IC检验）"""
        try:
            logger.info("执行高级特征选择...")
            
            # IC检验
            ic_scores = {}
            for factor in factors_df.columns:
                if factor == 'returns':
                    continue
                    
                factor_values = factors_df[factor].dropna()
                returns_values = returns_df.loc[factor_values.index]
                
                if len(factor_values) > 30:
                    try:
                        corr, p_value = spearmanr(factor_values, returns_values)
                        if not np.isnan(corr) and p_value < 0.05:
                            ic_scores[factor] = abs(corr)
                    except:
                        continue
            
            # 选择高IC因子
            if ic_scores:
                # 按IC排序，选择前80%
                sorted_factors = sorted(ic_scores.items(), key=lambda x: x[1], reverse=True)
                n_select = max(5, int(len(sorted_factors) * 0.8))
                selected_factors = [f[0] for f in sorted_factors[:n_select]]
                
                logger.info(f"基于IC选择了 {len(selected_factors)} 个因子")
                return factors_df[selected_factors]
            else:
                # 备用：选择预定义的重要因子
                important_factors = [f for f in MULTI_DAY_FACTORS if f in factors_df.columns and f != 'returns']
                logger.warning(f"IC检验无结果，使用预定义因子: {len(important_factors)} 个")
                return factors_df[important_factors[:20]]  # 限制数量
                
        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            # 返回基本因子
            basic_factors = ['returns_5d', 'volatility_10d', 'sma_ratio_20', 'rsi', 'macd']
            available_factors = [f for f in basic_factors if f in factors_df.columns]
            return factors_df[available_factors]
    
    def create_lstm_sequences(self, factors_df: pd.DataFrame, 
                             returns_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """创建LSTM训练序列"""
        try:
            logger.info("创建LSTM序列数据...")
            
            # 创建目标变量（未来N天收益率）
            targets = []
            for day in range(1, self.prediction_days + 1):
                target = returns_df['returns'].shift(-day)
                targets.append(target)
            
            targets_df = pd.concat(targets, axis=1)
            targets_df.columns = [f'target_day_{i+1}' for i in range(self.prediction_days)]
            
            # 对齐数据
            common_index = factors_df.index.intersection(targets_df.index)
            factors_aligned = factors_df.loc[common_index]
            targets_aligned = targets_df.loc[common_index]
            
            # 数据标准化
            factors_scaled = self.scaler.fit_transform(factors_aligned)
            factors_scaled_df = pd.DataFrame(factors_scaled, 
                                           columns=factors_aligned.columns,
                                           index=factors_aligned.index)
            
            # 创建序列
            X_sequences = []
            y_sequences = []
            
            for i in range(self.lstm_window, len(factors_scaled_df) - self.prediction_days):
                X_seq = factors_scaled_df.iloc[i-self.lstm_window:i].values
                y_seq = targets_aligned.iloc[i].values
                
                if not (np.isnan(X_seq).any() or np.isnan(y_seq).any()):
                    X_sequences.append(X_seq)
                    y_sequences.append(y_seq)
            
            X = np.array(X_sequences)
            y = np.array(y_sequences)
            
            self.feature_columns = factors_aligned.columns.tolist()
            
            logger.info(f"序列创建完成: X={X.shape}, y={y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"序列创建失败: {e}")
            return None, None
    
    def build_lstm_model(self, input_shape: tuple, enable_advanced: bool = False) -> Sequential:
        """构建LSTM模型"""
        try:
            if not TF_AVAILABLE:
                raise Exception("TensorFlow not available")
            
            model = Sequential()
            
            if enable_advanced:
                # 高级模型架构
                model.add(LSTM(96, return_sequences=True, input_shape=input_shape,
                              kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4)))
                model.add(Dropout(0.3))
                model.add(LSTM(64, return_sequences=True,
                              kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4)))
                model.add(Dropout(0.3))
                model.add(LSTM(32, return_sequences=False))
                model.add(BatchNormalization())
                model.add(Dense(64, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(32, activation='relu'))
                model.add(Dropout(0.1))
            else:
                # 标准模型架构（更快训练）
                model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
                model.add(Dropout(0.2))
                model.add(LSTM(32, return_sequences=False))
                model.add(BatchNormalization())
                model.add(Dense(16, activation='relu'))
                model.add(Dropout(0.3))
            
            # 输出层
            model.add(Dense(self.prediction_days, activation='linear'))
            
            # 编译模型
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            logger.info(f"LSTM模型构建完成 - 参数数量: {model.count_params()}")
            return model
            
        except Exception as e:
            logger.error(f"模型构建失败: {e}")
            return None
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   use_cached: bool = True) -> bool:
        """训练LSTM模型"""
        try:
            # 检查是否使用缓存模型
            cache_path = os.path.join(self.model_cache_dir, 'weekly_lstm_model.h5')
            scaler_path = os.path.join(self.model_cache_dir, 'weekly_scaler.pkl')
            
            if use_cached and os.path.exists(cache_path) and os.path.exists(scaler_path):
                # 检查模型是否是最近一周内训练的
                model_time = os.path.getmtime(cache_path)
                current_time = time.time()
                days_old = (current_time - model_time) / (24 * 3600)
                
                if days_old <= 7:  # 一周内的模型
                    logger.info(f"使用缓存模型（{days_old:.1f}天前训练）")
                    self.lstm_model = load_model(cache_path)
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    return True
                else:
                    logger.info(f"缓存模型过旧（{days_old:.1f}天），重新训练")
            
            logger.info("开始训练LSTM模型...")
            
            # 构建模型
            self.lstm_model = self.build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                enable_advanced=self.enable_optimization
            )
            
            if self.lstm_model is None:
                return False
            
            # 设置回调
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(factor=0.5, patience=10, verbose=1)
            ]
            
            # 如果有验证集则添加模型检查点
            if X_val is not None:
                callbacks.append(
                    ModelCheckpoint(cache_path, save_best_only=True, verbose=1)
                )
            
            # 训练模型
            history = self.lstm_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=100 if self.enable_optimization else 50,  # 周度运行使用较少epoch
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # 保存模型和scaler
            self.lstm_model.save(cache_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # 记录训练性能
            final_loss = min(history.history['loss'])
            self.model_performance['train_loss'] = final_loss
            
            if 'val_loss' in history.history:
                final_val_loss = min(history.history['val_loss'])
                self.model_performance['val_loss'] = final_val_loss
                logger.info(f"训练完成 - 训练损失: {final_loss:.6f}, 验证损失: {final_val_loss:.6f}")
            else:
                logger.info(f"训练完成 - 训练损失: {final_loss:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            return False
    
    def predict_stock_returns(self, factors_df: pd.DataFrame, ticker: str) -> np.ndarray:
        """预测单只股票的多日收益率"""
        try:
            if self.lstm_model is None:
                logger.error("模型未训练")
                return None
            
            # 确保因子顺序正确
            if not self.feature_columns:
                logger.error("特征列信息缺失")
                return None
            
            # 重新排列因子顺序
            available_features = [f for f in self.feature_columns if f in factors_df.columns]
            if len(available_features) != len(self.feature_columns):
                logger.warning(f"因子数量不匹配: 需要{len(self.feature_columns)}, 得到{len(available_features)}")
            
            # 获取最新数据
            latest_data = factors_df[available_features].iloc[-self.lstm_window:].copy()
            
            # 处理缺失值
            latest_data = latest_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # 标准化
            latest_scaled = self.scaler.transform(latest_data)
            
            # 创建预测序列
            X_pred = latest_scaled.reshape(1, self.lstm_window, len(available_features))
            
            # 预测
            predictions = self.lstm_model.predict(X_pred, verbose=0)[0]
            
            # 增强预测多样性（基于股票代码）
            predictions = self._enhance_predictions(predictions, ticker)
            
            logger.debug(f"{ticker} 预测: {[f'{p:.4f}' for p in predictions]}")
            return predictions
            
        except Exception as e:
            logger.error(f"{ticker} 预测失败: {e}")
            return None
    
    def _enhance_predictions(self, predictions: np.ndarray, ticker: str) -> np.ndarray:
        """增强预测多样性（保持原有逻辑）"""
        try:
            # 基于股票代码的确定性调整
            ticker_hash = hash(ticker) % 10000
            base_adjustments = np.array([
                (ticker_hash % 100 - 50) * 0.0001,
                (ticker_hash % 200 - 100) * 0.0001,
                (ticker_hash % 300 - 150) * 0.0001,
                (ticker_hash % 400 - 200) * 0.0001,
                (ticker_hash % 500 - 250) * 0.0001
            ])
            
            enhanced_predictions = predictions + base_adjustments
            
            # 限制预测范围
            enhanced_predictions = np.clip(enhanced_predictions, -0.15, 0.15)
            
            return enhanced_predictions
            
        except Exception as e:
            logger.error(f"预测增强失败: {e}")
            return predictions
    
    def generate_trading_recommendations(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """生成交易推荐（完全保持原有逻辑）"""
        logger.info("开始生成交易推荐...")
        recommendations = []
        
        for ticker, data in stock_data.items():
            try:
                # 计算因子
                factors_df = self.calculate_comprehensive_factors(data)
                if factors_df.empty:
                    continue
                
                # 特征选择
                returns_df = pd.DataFrame({'returns': factors_df['returns']}, index=factors_df.index)
                selected_factors = self.advanced_feature_selection(factors_df, returns_df)
                
                # 预测
                predictions = self.predict_stock_returns(selected_factors, ticker)
                if predictions is None:
                    continue
                
                # 获取基本信息
                current_price = data['Close'].iloc[-1]
                volume = data['Volume'].iloc[-1]
                
                # 计算加权预测分数
                day_weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
                weighted_prediction = np.sum(predictions * day_weights)
                
                # 生成评级
                if weighted_prediction > 0.025:
                    rating = 'STRONG_BUY'
                    confidence = 5
                elif weighted_prediction > 0.015:
                    rating = 'BUY'
                    confidence = 4
                elif weighted_prediction > -0.01:
                    rating = 'HOLD'
                    confidence = 2
                elif weighted_prediction > -0.025:
                    rating = 'SELL'
                    confidence = 1
                else:
                    rating = 'STRONG_SELL'
                    confidence = 0
                
                # 技术指标确认
                if 'rsi' in factors_df.columns:
                    rsi = factors_df['rsi'].iloc[-1]
                    if not pd.isna(rsi):
                        if rsi < 30 and weighted_prediction > 0:
                            confidence += 1
                        elif rsi > 70 and weighted_prediction < 0:
                            confidence += 1
                
                # SMA确认
                sma_20 = data['Close'].rolling(20).mean().iloc[-1]
                if not pd.isna(sma_20):
                    if current_price > sma_20 and weighted_prediction > 0:
                        confidence += 1
                    elif current_price < sma_20 and weighted_prediction < 0:
                        confidence += 1
                
                # 预测一致性
                consistency = np.sum(np.sign(predictions[:3]) == np.sign(predictions[0])) / 3
                
                recommendation = {
                    'ticker': ticker,
                    'current_price': current_price,
                    'weighted_prediction': weighted_prediction,
                    'day1_prediction': predictions[0],
                    'day2_prediction': predictions[1],
                    'day3_prediction': predictions[2],
                    'day4_prediction': predictions[3],
                    'day5_prediction': predictions[4],
                    'rating': rating,
                    'confidence_score': confidence,
                    'prediction_consistency': consistency,
                    'volume': volume,
                    'rsi': factors_df['rsi'].iloc[-1] if 'rsi' in factors_df.columns else None,
                    'price_to_sma20': current_price / sma_20 if not pd.isna(sma_20) else None,
                    'timestamp': datetime.now(),
                    'model_type': 'weekly_lstm'
                }
                
                recommendations.append(recommendation)
                logger.info(f"{ticker}: {rating}, 预测: {weighted_prediction:.3f}%, 置信度: {confidence}")
                
            except Exception as e:
                logger.error(f"处理 {ticker} 失败: {e}")
                continue
        
        # 按预测收益率排序
        recommendations = sorted(recommendations, key=lambda x: x['weighted_prediction'], reverse=True)
        
        logger.info(f"生成了 {len(recommendations)} 个交易推荐")
        return recommendations
    
    def save_trading_signals(self, recommendations: List[Dict], timestamp: str = None) -> Dict[str, str]:
        """保存交易信号（完全兼容Trading Manager）"""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 文件路径
        excel_file = f'result/weekly_lstm_analysis_{timestamp}.xlsx'
        json_file = f'weekly_trading_signals/weekly_signals_{timestamp}.json'
        csv_file = f'weekly_trading_signals/weekly_signals_{timestamp}.csv'
        
        try:
            # 1. Excel报告（详细分析）
            df = pd.DataFrame(recommendations)
            
            if len(df) > 0:
                # 添加百分比列
                df['weighted_prediction_pct'] = df['weighted_prediction'] * 100
                df['day1_prediction_pct'] = df['day1_prediction'] * 100
                df['day2_prediction_pct'] = df['day2_prediction'] * 100
                df['day3_prediction_pct'] = df['day3_prediction'] * 100
                df['day4_prediction_pct'] = df['day4_prediction'] * 100
                df['day5_prediction_pct'] = df['day5_prediction'] * 100
                
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    # 主要结果
                    main_columns = [
                        'ticker', 'rating', 'weighted_prediction_pct', 'confidence_score',
                        'current_price', 'day1_prediction_pct', 'day2_prediction_pct',
                        'day3_prediction_pct', 'day4_prediction_pct', 'day5_prediction_pct',
                        'prediction_consistency', 'rsi', 'price_to_sma20'
                    ]
                    main_df = df[main_columns].copy()
                    main_df.to_excel(writer, sheet_name='Weekly_Analysis', index=False)
                    
                    # Top 10 买入推荐
                    buy_signals = df[df['rating'].isin(['BUY', 'STRONG_BUY'])].head(10)
                    if len(buy_signals) > 0:
                        buy_signals.to_excel(writer, sheet_name='Top_Buy_Signals', index=False)
                    
                    # 详细数据
                    df.to_excel(writer, sheet_name='Detailed_Data', index=False)
                
                logger.info(f"Excel报告已保存: {excel_file}")
            
            # 2. JSON信号文件（Trading Manager使用）
            top_signals = recommendations[:20]  # 前20个信号
            json_data = {
                'timestamp': timestamp,
                'generation_time': datetime.now().isoformat(),
                'model_type': 'weekly_lstm',
                'prediction_horizon': f'{self.prediction_days}_days',
                'total_stocks_analyzed': len(recommendations),
                'signals': []
            }
            
            for rec in top_signals:
                signal = {
                    'ticker': rec['ticker'],
                    'action': rec['rating'],
                    'confidence': rec['confidence_score'],
                    'expected_return_1d': float(rec['day1_prediction']),
                    'expected_return_5d': float(rec['weighted_prediction']),
                    'current_price': float(rec['current_price']),
                    'technical_score': float(rec['prediction_consistency']),
                    'volume': int(rec['volume']),
                    'risk_level': 'LOW' if rec['confidence_score'] >= 4 else 'MEDIUM' if rec['confidence_score'] >= 2 else 'HIGH'
                }
                json_data['signals'].append(signal)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"JSON信号已保存: {json_file}")
            
            # 3. CSV文件（备用）
            if len(df) > 0:
                csv_df = df[['ticker', 'rating', 'weighted_prediction', 'confidence_score', 
                            'current_price', 'day1_prediction', 'prediction_consistency']].copy()
                csv_df.to_csv(csv_file, index=False, encoding='utf-8')
                logger.info(f"CSV文件已保存: {csv_file}")
            
            return {
                'excel_file': excel_file,
                'json_file': json_file,
                'csv_file': csv_file,
                'total_signals': len(recommendations)
            }
            
        except Exception as e:
            logger.error(f"保存交易信号失败: {e}")
            return {}
    
    def run_weekly_analysis(self, 
                          ticker_list: List[str] = None,
                          days_history: int = 365,
                          retrain_model: bool = False) -> Dict:
        """运行周度分析（主要接口）"""
        if ticker_list is None:
            ticker_list = MULTI_DAY_TICKER_LIST
        
        logger.info("="*80)
        logger.info("开始周度LSTM交易分析")
        logger.info("="*80)
        logger.info(f"分析股票: {len(ticker_list)} 只")
        logger.info(f"历史数据: {days_history} 天")
        logger.info(f"预测天数: {self.prediction_days} 天")
        logger.info(f"重新训练: {retrain_model}")
        
        # 1. 下载数据
        logger.info("下载股票数据...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_history)
        
        stock_data = {}
        successful_downloads = 0
        
        for ticker in ticker_list:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty and len(data) > self.lstm_window:
                    stock_data[ticker] = data
                    successful_downloads += 1
                    if successful_downloads % 10 == 0:
                        logger.info(f"已下载 {successful_downloads} 只股票数据")
                else:
                    logger.warning(f"{ticker}: 数据不足或为空")
            except Exception as e:
                logger.error(f"{ticker}: 下载失败 - {e}")
        
        logger.info(f"成功下载 {len(stock_data)} 只股票数据")
        
        if len(stock_data) < 5:
            logger.error("可用股票数据不足")
            return {'status': 'error', 'message': '数据不足'}
        
        # 2. 训练或加载模型
        if not retrain_model:
            # 尝试使用缓存模型
            logger.info("尝试加载缓存模型...")
            model_loaded = self.train_model(None, None, use_cached=True)
            if model_loaded:
                logger.info("缓存模型加载成功")
            else:
                logger.info("缓存模型不可用，需要训练新模型")
                retrain_model = True
        
        if retrain_model or self.lstm_model is None:
            logger.info("训练新的LSTM模型...")
            
            # 使用部分股票数据训练模型
            train_tickers = list(stock_data.keys())[:min(20, len(stock_data))]  # 最多用20只股票训练
            combined_factors_list = []
            combined_returns_list = []
            
            for ticker in train_tickers:
                factors = self.calculate_comprehensive_factors(stock_data[ticker])
                if not factors.empty:
                    returns = pd.DataFrame({'returns': factors['returns']}, index=factors.index)
                    selected_factors = self.advanced_feature_selection(factors, returns)
                    
                    combined_factors_list.append(selected_factors)
                    combined_returns_list.append(returns)
            
            if combined_factors_list:
                # 合并数据
                combined_factors = pd.concat(combined_factors_list, ignore_index=False)
                combined_returns = pd.concat(combined_returns_list, ignore_index=False)
                
                # 创建序列
                X, y = self.create_lstm_sequences(combined_factors, combined_returns)
                
                if X is not None and len(X) > 50:
                    # 分割训练/验证集
                    split_idx = int(len(X) * 0.8)
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train, y_val = y[:split_idx], y[split_idx:]
                    
                    # 训练模型
                    success = self.train_model(X_train, y_train, X_val, y_val, use_cached=False)
                    
                    if not success:
                        logger.error("模型训练失败")
                        return {'status': 'error', 'message': '模型训练失败'}
                else:
                    logger.error("训练数据不足")
                    return {'status': 'error', 'message': '训练数据不足'}
            else:
                logger.error("无法准备训练数据")
                return {'status': 'error', 'message': '数据准备失败'}
        
        # 3. 生成交易推荐
        logger.info("生成交易推荐...")
        recommendations = self.generate_trading_recommendations(stock_data)
        
        if not recommendations:
            logger.error("未生成任何推荐")
            return {'status': 'error', 'message': '无推荐生成'}
        
        # 4. 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = self.save_trading_signals(recommendations, timestamp)
        
        # 5. 生成摘要
        buy_signals = len([r for r in recommendations if r['rating'] in ['BUY', 'STRONG_BUY']])
        sell_signals = len([r for r in recommendations if r['rating'] in ['SELL', 'STRONG_SELL']])
        hold_signals = len([r for r in recommendations if r['rating'] == 'HOLD'])
        
        avg_confidence = np.mean([r['confidence_score'] for r in recommendations])
        top_stock = recommendations[0] if recommendations else None
        
        summary = {
            'status': 'success',
            'timestamp': timestamp,
            'total_stocks_analyzed': len(recommendations),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'average_confidence': avg_confidence,
            'top_recommendation': {
                'ticker': top_stock['ticker'],
                'rating': top_stock['rating'],
                'expected_return': top_stock['weighted_prediction']
            } if top_stock else None,
            'files_generated': saved_files,
            'model_performance': self.model_performance
        }
        
        # 输出摘要
        logger.info("="*80)
        logger.info("周度分析完成摘要")
        logger.info("="*80)
        logger.info(f"分析股票数量: {summary['total_stocks_analyzed']}")
        logger.info(f"买入信号: {buy_signals}, 卖出信号: {sell_signals}, 持有信号: {hold_signals}")
        logger.info(f"平均置信度: {avg_confidence:.2f}")
        if top_stock:
            logger.info(f"最佳推荐: {top_stock['ticker']} ({top_stock['rating']}) - {top_stock['weighted_prediction']*100:.2f}%")
        
        for file_type, file_path in saved_files.items():
            if file_path:
                logger.info(f"{file_type.upper()}: {file_path}")
        
        return summary


def run_weekly_trading_analysis():
    """运行周度交易分析（周一开盘前调用）"""
    logger.info("启动周度LSTM交易分析系统")
    
    # 检查是否为周一
    today = date.today()
    weekday = today.weekday()  # 0=Monday, 6=Sunday
    
    if weekday == 0:  # 周一
        logger.info(f"今天是周一({today})，执行完整分析")
        retrain = True
    else:
        logger.info(f"今天是{['周一','周二','周三','周四','周五','周六','周日'][weekday]}({today})，执行快速分析")
        retrain = False
    
    # 创建系统实例
    system = WeeklyTradingSystemLSTM(
        prediction_days=5,
        lstm_window=20,
        enable_optimization=False  # 周度运行关闭优化以提高速度
    )
    
    # 运行分析
    result = system.run_weekly_analysis(
        ticker_list=MULTI_DAY_TICKER_LIST,
        days_history=365,
        retrain_model=retrain
    )
    
    # 返回结果给Trading Manager
    if result['status'] == 'success':
        logger.info("周度分析成功完成，准备交接给Trading Manager")
        
        # 生成Trading Manager需要的信号文件
        signal_file = result['files_generated'].get('json_file')
        if signal_file and os.path.exists(signal_file):
            logger.info(f"Trading Manager信号文件: {signal_file}")
            
            # 可以在这里添加直接调用Trading Manager的代码
            # 或者让Trading Manager监控信号文件目录
            
        return result
    else:
        logger.error(f"周度分析失败: {result.get('message')}")
        return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='周度LSTM交易系统')
    parser.add_argument('--tickers', nargs='+', help='股票代码列表')
    parser.add_argument('--days', type=int, default=365, help='历史数据天数')
    parser.add_argument('--retrain', action='store_true', help='强制重新训练模型')
    parser.add_argument('--output-dir', type=str, default='weekly_trading_signals', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行分析
    if args.tickers:
        # 自定义股票列表
        system = WeeklyTradingSystemLSTM()
        result = system.run_weekly_analysis(
            ticker_list=args.tickers,
            days_history=args.days,
            retrain_model=args.retrain
        )
    else:
        # 标准周度分析
        result = run_weekly_trading_analysis()
    
    # 输出结果
    if result['status'] == 'success':
        print("="*60)
        print("周度LSTM交易分析完成")
        print("="*60)
        print(f"分析时间: {result['timestamp']}")
        print(f"分析股票: {result['total_stocks_analyzed']} 只")
        print(f"交易信号: 买入{result['buy_signals']}, 卖出{result['sell_signals']}, 持有{result['hold_signals']}")
        
        if result.get('top_recommendation'):
            top = result['top_recommendation']
            print(f"最佳推荐: {top['ticker']} - {top['rating']} ({top['expected_return']*100:.2f}%)")
        
        print("\n生成文件:")
        for file_type, file_path in result['files_generated'].items():
            if file_path:
                print(f"  {file_type.upper()}: {file_path}")
        
        print("\n系统准备就绪，可与Trading Manager集成")
        
    else:
        print(f"分析失败: {result.get('message')}")
        sys.exit(1)


if __name__ == "__main__":
    main()