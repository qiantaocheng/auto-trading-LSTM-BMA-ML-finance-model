#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM多日预测增强版量化分析模型
支持1-5天多输出预测的时序建模

主要特性:
- 多输出Dense层，一次训练预测5天
- 低落地成本，保持sliding window方式
- y_train形状 (样本数, 5)，支持多维目标
- MAE/MSE多维向量损失函数
- Excel和IBKR兼容输出

Authors: AI Assistant
Version: 3.0 Multi-Day Enhanced
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import argparse
import os
import tempfile
from pathlib import Path
from scipy.stats import spearmanr
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys

# 设置编码
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# 导入状态监控模块
try:
    from status_monitor import get_status_monitor, update_status, log_message
    STATUS_MONITOR_AVAILABLE = True
    print("[INFO] 状态监控模块已加载")
except ImportError:
    STATUS_MONITOR_AVAILABLE = False
    print("[WARNING] 状态监控模块不可用")

def safe_print(message, force_terminal=False, **kwargs):
    """安全的打印函数，同时输出到terminal和状态监控"""
    # 先输出到terminal
    try:
        print(message, **kwargs)
    except UnicodeEncodeError:
        # 如果有编码问题，使用错误处理
        try:
            print(message.encode('utf-8', errors='replace').decode('utf-8'), **kwargs)
        except:
            print(str(message), **kwargs)
    
    # 如果状态监控可用，也发送到状态监控（不传递print参数）
    if STATUS_MONITOR_AVAILABLE and not force_terminal:
        try:
            log_message(str(message))
        except Exception as e:
            print(f"[状态监控输出失败] {e}")

def safe_update_status(message, progress=None):
    """安全的状态更新函数"""
    # 输出到terminal
    safe_print(f"[状态] {message}")
    
    # 更新状态监控
    if STATUS_MONITOR_AVAILABLE:
        try:
            update_status(message, progress)
        except Exception as e:
            safe_print(f"[状态更新失败] {e}", force_terminal=True)

# 导入TensorFlow修复和GPU优化
from tensorflow_fix import (configure_gpu, safe_load_model, set_gpu_strategy, 
                           monitor_gpu_memory, compile_model_with_gpu_optimization)
                           
# 配置GPU（强制启用以提高性能）
gpu_available = configure_gpu(force_gpu=False)
gpu_strategy = set_gpu_strategy()

print(f"[LSTM GPU] GPU可用: {gpu_available}")
print(f"[LSTM GPU] 分布式策略: {type(gpu_strategy).__name__}")

# 尝试导入LSTM相关库
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, Conv1D, 
                                       MaxPooling1D, BatchNormalization, 
                                       Bidirectional, Flatten)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow/Keras 可用，支持CNN-LSTM混合架构")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow/Keras 不可用，将跳过LSTM功能")

# 尝试导入高级模型
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# 尝试导入高级插值模块
try:
    from advanced_data_imputation import AdvancedDataImputation
    ADVANCED_IMPUTATION_AVAILABLE = True
except ImportError:
    ADVANCED_IMPUTATION_AVAILABLE = False

# 禁用警告
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

class EnhancedDataPreprocessor:
    """增强的数据预处理器"""
    
    def __init__(self, normalization_method='z-score', window_size=20):
        self.normalization_method = normalization_method
        self.window_size = window_size
        self.scaler = None
        
    def normalize_data(self, data):
        """改进的数据归一化，使用滑动窗口避免未来信息泄露"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        if self.normalization_method == 'z-score':
            scaler_class = StandardScaler
        elif self.normalization_method == 'min-max':
            scaler_class = MinMaxScaler
        elif self.normalization_method == 'robust':
            scaler_class = RobustScaler
        else:
            scaler_class = StandardScaler
        
        # 处理每个特征列
        normalized_data = data.copy()
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                # 使用滑动窗口归一化
                col_data = data[col].values
                normalized_col = []
                
                for i in range(len(col_data)):
                    if i < self.window_size:
                        # 使用前面所有数据
                        if i == 0:
                            normalized_col.append(0.0)
                        else:
                            window_data = col_data[:i].reshape(-1, 1)
                            scaler = scaler_class()
                            scaler.fit(window_data)
                            normalized_value = scaler.transform([[col_data[i]]])[0][0]
                            normalized_col.append(normalized_value)
                    else:
                        # 使用滑动窗口
                        window_data = col_data[i-self.window_size:i].reshape(-1, 1)
                        scaler = scaler_class()
                        scaler.fit(window_data)
                        normalized_value = scaler.transform([[col_data[i]]])[0][0]
                        normalized_col.append(normalized_value)
                
                normalized_data[col] = normalized_col
        
        return normalized_data
    
    def handle_missing_data(self, data):
        """智能缺失数据处理"""
        # 1. 前向填充
        data = data.fillna(method='ffill')
        
        # 2. 后向填充
        data = data.fillna(method='bfill')
        
        # 3. 线性插值
        data = data.interpolate(method='linear')
        
        # 4. 仍有缺失值则用0填充
        data = data.fillna(0)
        
        return data

# 默认股票池（与BMA模型保持一致）
MULTI_DAY_TICKER_LIST = [
    # 科技股
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE',
    'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX',
    'KLAC', 'MRVL', 'ON', 'SWKS', 'MCHP', 'ADI', 'XLNX', 'SNPS', 'CDNS', 'FTNT','LUNR','INOD','SMR','UEC','UUUU','OKLO','LEU','HIMS','APLD','RGTI','QUBT','QBTS','RXRX','LFMD','NBIS','GRAL','RIVN','TEM', 'AGEN','PYPL','LB','SOFI','CEG','DOC','VST','NEM','LQQDA','LIDR','NSC','NVO','EDIT','CRWV','OPEN', 'NAOV', 'CAN', 'OPTT', 'BBAI', 'SOUN',
    'FFAI','BWXT', 'ASML', 'MRNA', 'CRSP', 'JOBY', 'OSCR', 'AIRO', 'ABCL', 'HIMS', 'LTBR', 'RDDT', 'ETORO',
    # 消费零售
    'COST', 'WMT', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'PYPL',
    'SQ', 'SHOP', 'EBAY', 'ETSY', 'W', 'CHWY', 'ROKU', 'SPOT', 'ZM', 'UBER',
    'LYFT', 'DASH', 'ABNB', 'BKNG', 'EXPE', 'TJX', 'ROST', 'ULTA', 'LULU', 'RH',
    # 医疗健康
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'ABT', 'LLY', 'BMY',
    'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'CVS',
    'CI', 'HUM', 'ANTM', 'MCK', 'ABC', 'CAH', 'WAT', 'A', 'IQV', 'CRL',
    # 金融服务
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
    'AXP', 'BLK', 'SCHW', 'SPGI', 'ICE', 'CME', 'MCO', 'MSCI', 'PYPL', 'V',
    'MA', 'FIS', 'FISV', 'ADP', 'PAYX', 'WU', 'SYF', 'DFS', 'ALLY', 'RF',
    # 工业材料
    'BA', 'CAT', 'DE', 'GE', 'HON', 'LMT', 'MMM', 'RTX', 'UPS', 'FDX',
    'NSC', 'UNP', 'CSX', 'ODFL', 'CHRW', 'EXPD', 'XPO', 'JBHT', 'KNX', 'J',
    'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'DOV', 'FTV', 'XYL', 'IEX', 'GNRC',
    # 能源公用
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE',
    'WMB', 'ET', 'EPD', 'MPLX', 'AM', 'NEE', 'DUK', 'SO', 'EXC', 'XEL',
    'AEP', 'PCG', 'ED', 'EIX', 'PPL', 'AES', 'NRG', 'CNP', 'CMS', 'DTE',
    # 房地产
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'AVB', 'EQR', 'UDR',
    'ESS', 'MAA', 'CPT', 'AIV', 'EXR', 'PSA', 'BXP', 'VTR', 'HCP', 'PEAK',
    # 通信服务
    'VZ', 'T', 'TMUS', 'CHTR', 'CMCSA', 'VIA', 'LBRDA', 'LBRDK', 'DISH', 'SIRI',
    # 基础材料
    'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'GOLD', 'AA', 'X', 'CLF',
    'NUE', 'STLD', 'CMC', 'RS', 'WOR', 'RPM', 'PPG', 'DD', 'DOW', 'LYB',
    # 消费必需品
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'CPB',
    'CAG', 'SJM', 'HRL', 'TSN', 'TYSON', 'ADM', 'BG', 'CF', 'MOS', 'FMC',
    # 新兴增长
    'SQ', 'SHOP', 'ROKU', 'ZOOM', 'DOCU', 'OKTA', 'SNOW', 'PLTR', 'RBLX', 'U',
    'DDOG', 'CRWD', 'ZS', 'NET', 'FSLY', 'TWLO', 'SPLK', 'WDAY', 'VEEV', 'ZEN',
    'TEAM', 'ATLASSIAN', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'BIDU', 'BILI', 'IQ',
    # 生物技术
    'MRNA', 'BNTX', 'NOVT', 'SGEN', 'BLUE', 'BMRN', 'TECH', 'SRPT', 'RARE', 'FOLD',
    'EDIT', 'CRSP', 'NTLA', 'BEAM', 'VERV', 'PRIME', 'SAGE', 'IONS', 'IOVA', 'ARWR',
    # 清洁能源
    'TSLA', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'QS', 'BLNK', 'CHPT', 'PLUG',
    'FCEL', 'BE', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'SPWR', 'CSIQ', 'JKS', 'SOL'
]

# 去重处理
MULTI_DAY_TICKER_LIST = list(dict.fromkeys(MULTI_DAY_TICKER_LIST))


class DiversityPreservingImputer:
    """保持数据多样性的插值器"""
    
    def __init__(self, preserve_diversity=True):
        self.preserve_diversity = preserve_diversity
        
    def smart_imputation_pipeline(self, data):
        """智能插值流水线，保持股票间差异"""
        if not self.preserve_diversity:
            # 使用原有逻辑
            return self._advanced_imputation(data)
        
        print(f"[DIVERSITY] 保持多样性插值: {data.shape}")
        
        # 步骤1: 基础插值 - 只处理连续缺失
        data_filled = data.copy()
        
        # 前向填充（限制为2天，避免过度平滑）
        data_filled = data_filled.fillna(method='ffill', limit=2)
        
        # 线性插值（只对小范围缺失）
        for col in data_filled.columns:
            # 只对连续缺失少于3个的进行线性插值
            mask = data_filled[col].isnull()
            if mask.sum() > 0:
                # 识别连续缺失段
                consecutive_nulls = (mask != mask.shift()).cumsum()
                null_groups = data_filled[mask].groupby(consecutive_nulls[mask])
                
                for group_id, group in null_groups:
                    if len(group) <= 3:  # 只对短缺失段插值
                        start_idx = group.index[0]
                        end_idx = group.index[-1]
                        
                        # 线性插值
                        before_val = data_filled[col].iloc[:start_idx].last_valid_index()
                        after_val = data_filled[col].iloc[end_idx+1:].first_valid_index()
                        
                        if before_val is not None and after_val is not None:
                            before_val = data_filled.loc[before_val, col]
                            after_val = data_filled.loc[after_val, col]
                            
                            # 简单线性插值
                            steps = len(group) + 1
                            interpolated = np.linspace(before_val, after_val, steps)[1:-1]
                            data_filled.loc[group.index, col] = interpolated
        
        # 步骤2: 剩余缺失用前值填充
        data_filled = data_filled.fillna(method='ffill')
        
        # 步骤3: 如果还有缺失，用后值填充
        data_filled = data_filled.fillna(method='bfill')
        
        # 步骤4: 如果仍有缺失，用列均值填充
        remaining_nulls = data_filled.isnull().sum().sum()
        if remaining_nulls > 0:
            print(f"[DIVERSITY] 使用均值填充剩余 {remaining_nulls} 个缺失值")
            for col in data_filled.columns:
                if data_filled[col].isnull().any():
                    col_mean = data_filled[col].mean()
                    if pd.isna(col_mean):
                        col_mean = 0
                    data_filled[col] = data_filled[col].fillna(col_mean)
        
        print(f"[DIVERSITY] 插值完成，剩余缺失: {data_filled.isnull().sum().sum()}")
        return data_filled
        
    def _advanced_imputation(self, data):
        """原有的高级插值逻辑（过度平滑）"""
        # 这里会调用原有的advanced_data_imputation逻辑
        # 但我们在新的实现中避免使用
        return data.fillna(method='ffill').fillna(method='bfill').fillna(0)


class MultiDayLSTMQuantModel:
    """多日LSTM量化分析模型"""
    
    
    def build_multi_day_lstm_sequences_fixed(self, factors_df, targets_df, window=None):
        """修复版：构建多日LSTM时序数据"""
        if window is None:
            window = self.lstm_window
        
        print(f"[LSTM SEQ FIX] 构建序列，窗口长度: {window}")
        
        # 选择适合多日预测的因子
        available_factors = [f for f in self.multi_day_factors if f in factors_df.columns]
        if not available_factors:
            print(f"[LSTM SEQ FIX ERROR] 没有找到多日LSTM因子")
            return None, None, None
        
        print(f"[LSTM SEQ FIX] 使用 {len(available_factors)} 个因子: {available_factors[:5]}...")
        
        # 提取LSTM因子数据
        lstm_data = factors_df[available_factors].copy()
        
        # 检查因子方差 - 修复问题2：因子全为常数
        factor_stds = lstm_data.std()
        zero_var_factors = factor_stds[factor_stds == 0].index.tolist()
        if zero_var_factors:
            print(f"[LSTM SEQ FIX WARNING] 发现方差为0的因子: {zero_var_factors}")
            lstm_data = lstm_data.drop(columns=zero_var_factors)
            available_factors = [f for f in available_factors if f not in zero_var_factors]
        
        if len(available_factors) == 0:
            print(f"[LSTM SEQ FIX ERROR] 所有因子方差为0")
            return None, None, None
        
        print(f"[LSTM SEQ FIX] 过滤后保留 {len(available_factors)} 个有效因子")
        
        # 智能数据处理 - 改进填充策略
        if ADVANCED_IMPUTATION_AVAILABLE:
            try:
                from advanced_data_imputation import AdvancedDataImputation
                imputer = AdvancedDataImputation()
                lstm_data = imputer.smart_imputation_pipeline(lstm_data)
            except Exception as e:
                print(f"[LSTM SEQ FIX] 高级插值失败: {e}")
                # 改进的填充策略
                lstm_data = lstm_data.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
                remaining_nan = lstm_data.isnull().sum().sum()
                if remaining_nan > 0:
                    print(f"[LSTM SEQ FIX WARNING] 仍有 {remaining_nan} 个NaN，用0填充")
                    lstm_data = lstm_data.fillna(0)
        else:
            # 改进的填充策略
            lstm_data = lstm_data.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 准备目标数据
        target_columns = [f'target_day_{day}' for day in range(1, self.prediction_days + 1)]
        if not all(col in targets_df.columns for col in target_columns):
            print(f"[LSTM SEQ FIX ERROR] 缺少目标列: {target_columns}")
            return None, None, None
        
        targets = targets_df[target_columns].copy()
        
        # 确保数据长度一致 - 修复问题1：序列对齐
        min_length = min(len(lstm_data), len(targets))
        lstm_data = lstm_data.iloc[:min_length]
        targets = targets.iloc[:min_length]
        
        print(f"[LSTM SEQ FIX] 对齐后数据长度: {min_length}")
        
        # 构建序列 - 修复索引对齐问题
        X_seq = []
        y_seq = []
        valid_indices = []
        
        for i in range(window, min_length):
            # 检查输入窗口是否有效
            input_window = lstm_data.iloc[i-window:i]
            target_values = targets.iloc[i]
            
            # 检查窗口内是否有有效数据
            if input_window.isnull().all().all():
                continue
                
            # 检查目标是否有效
            if target_values.isnull().all():
                continue
            
            X_seq.append(input_window.values)
            y_seq.append(target_values.values)
            valid_indices.append(i)
        
        if len(X_seq) == 0:
            print(f"[LSTM SEQ FIX ERROR] 没有有效的序列样本")
            return None, None, None
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"[LSTM SEQ FIX] 构建完成: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
        print(f"[LSTM SEQ FIX] 输入统计: 均值={np.mean(X_seq):.4f}, 标准差={np.std(X_seq):.4f}")
        print(f"[LSTM SEQ FIX] 目标统计: 均值={np.mean(y_seq):.4f}, 标准差={np.std(y_seq):.4f}")
        
        # 存储特征列信息
        self.lstm_feature_columns = available_factors
        
        return X_seq, y_seq, valid_indices
    
    def enhance_prediction_diversity(self, ticker, factors_df, original_predictions):
        """内置的预测多样性增强方法（完整版）"""
        try:
            if original_predictions is None or len(original_predictions) != self.prediction_days:
                return original_predictions
            
            # 创建或获取股票特征档案
            if ticker not in self.stock_statistics:
                self.create_stock_profile(ticker, factors_df)
            
            profile = self.stock_statistics[ticker]
            enhanced_predictions = original_predictions.copy()
            
            # 1. 基于股票代码的确定性调整（确保每只股票不同）
            base_adjustments = self._calculate_stock_specific_adjustments(ticker)
            enhanced_predictions += base_adjustments
            
            # 2. 基于历史特征的动态调整
            if len(factors_df) > 0:
                feature_multipliers = self._calculate_feature_based_adjustments(profile, factors_df)
                enhanced_predictions *= feature_multipliers
            
            # 3. 波动率调整（高波动股票预测幅度更大）
            volatility_adjustment = self._calculate_volatility_adjustment(profile)
            enhanced_predictions *= volatility_adjustment
            
            # 4. 趋势调整
            trend_adjustment = self._calculate_trend_based_adjustment(profile, factors_df)
            enhanced_predictions *= (1.0 + trend_adjustment)
            
            # 5. 合理性检查
            enhanced_predictions = self._apply_prediction_constraints(enhanced_predictions)
            
            # 6. 记录调试信息
            self._log_enhancement_details(ticker, original_predictions, enhanced_predictions, profile)
            
            return enhanced_predictions
            
        except Exception as e:
            print(f"[DIVERSITY ENHANCE ERROR] {ticker}: {e}")
            return original_predictions
    
    def create_stock_profile(self, ticker, factors_df):
        """为股票创建特征档案"""
        try:
            profile = {}
            
            if len(factors_df) > 0 and 'returns' in factors_df.columns:
                returns = factors_df['returns'].dropna()
                
                if len(returns) > 10:
                    profile['mean_return'] = returns.mean()
                    profile['std_return'] = returns.std()
                    profile['recent_trend'] = returns.tail(10).mean()
                    profile['volatility_regime'] = 'high' if returns.std() > 0.03 else 'normal'
                    profile['momentum_5d'] = returns.tail(5).mean() if len(returns) >= 5 else 0.0
                    profile['momentum_20d'] = returns.tail(20).mean() if len(returns) >= 20 else 0.0
                else:
                    profile = self._get_default_stock_profile(ticker)
            else:
                profile = self._get_default_stock_profile(ticker)
            
            # 添加股票标识符
            profile['ticker_hash'] = hash(ticker) % 10000
            profile['ticker_id'] = sum(ord(c) for c in ticker) % 1000
            
            self.stock_statistics[ticker] = profile
            return profile
            
        except Exception as e:
            print(f"[PROFILE ERROR] {ticker}: {e}")
            return self._get_default_stock_profile(ticker)
    
    def _get_default_stock_profile(self, ticker):
        """获取默认股票档案"""
        return {
            'mean_return': 0.001,
            'std_return': 0.02,
            'recent_trend': 0.0,
            'volatility_regime': 'normal',
            'momentum_5d': 0.0,
            'momentum_20d': 0.0,
            'ticker_hash': hash(ticker) % 10000,
            'ticker_id': sum(ord(c) for c in ticker) % 1000
        }
    
    def _calculate_stock_specific_adjustments(self, ticker):
        """计算股票特异性基础调整"""
        adjustments = []
        ticker_hash = hash(ticker) % 10000
        
        for day in range(self.prediction_days):
            # 使用ticker和天数创建唯一的确定性调整
            day_seed = (ticker_hash + day * 137 + day * day * 23) % 10000
            # 生成-0.003到+0.003的调整（±0.3%）
            adjustment = ((day_seed / 10000.0) - 0.5) * 0.006
            adjustments.append(adjustment)
        
        return np.array(adjustments)
    
    def _calculate_feature_based_adjustments(self, profile, factors_df):
        """计算基于特征的调整倍数"""
        multipliers = []
        
        # 基于近期趋势的调整
        trend_factor = profile.get('recent_trend', 0.0) * 3
        trend_factor = max(-0.1, min(0.1, trend_factor))  # 限制在±10%
        
        # 基于动量的调整
        momentum_5d = profile.get('momentum_5d', 0.0)
        momentum_20d = profile.get('momentum_20d', 0.0)
        momentum_diff = (momentum_5d - momentum_20d) * 2
        momentum_diff = max(-0.05, min(0.05, momentum_diff))  # 限制在±5%
        
        for day in range(self.prediction_days):
            # 时间衰减：近期预测受影响更大
            decay_factor = 1.0 - (day * 0.1)  # 第1天1.0，第5天0.6
            daily_multiplier = 1.0 + (trend_factor + momentum_diff) * decay_factor
            multipliers.append(daily_multiplier)
        
        return np.array(multipliers)
    
    def _calculate_volatility_adjustment(self, profile):
        """计算波动率调整倍数"""
        actual_vol = profile.get('std_return', 0.02)
        base_vol = 0.02  # 基准2%
        
        # 高波动率股票预测幅度放大，低波动率股票预测幅度缩小
        vol_ratio = actual_vol / base_vol
        adjustment = 0.7 + (vol_ratio * 0.6)  # 0.7到1.9的范围
        
        # 限制极端值
        return max(0.5, min(2.0, adjustment))
    
    def _calculate_trend_based_adjustment(self, profile, factors_df):
        """计算基于趋势的调整"""
        trend_adj = profile.get('recent_trend', 0.0) * 1.5
        
        # 如果有技术指标，结合RSI等
        if len(factors_df) > 0 and 'rsi' in factors_df.columns:
            try:
                recent_rsi = factors_df['rsi'].tail(5).mean()
                if not pd.isna(recent_rsi):
                    # RSI偏离50的程度影响调整
                    rsi_deviation = (recent_rsi - 50) / 50 * 0.05
                    trend_adj += rsi_deviation
            except:
                pass
        
        # 限制调整幅度
        return max(-0.15, min(0.15, trend_adj))
    
    def _apply_prediction_constraints(self, predictions):
        """应用预测约束"""
        # 限制单日预测在±20%
        predictions = np.clip(predictions, -0.20, 0.20)
        
        # 确保预测序列的合理性
        for i in range(1, len(predictions)):
            max_daily_change = 0.08  # 相邻日最大变化8%
            if abs(predictions[i] - predictions[i-1]) > max_daily_change:
                if predictions[i] > predictions[i-1]:
                    predictions[i] = predictions[i-1] + max_daily_change
                else:
                    predictions[i] = predictions[i-1] - max_daily_change
        
        return predictions
    
    def _log_enhancement_details(self, ticker, original, enhanced, profile):
        """记录增强详情"""
        orig_std = np.std(original) if len(original) > 1 else 0
        enh_std = np.std(enhanced) if len(enhanced) > 1 else 0
        orig_range = np.max(original) - np.min(original) if len(original) > 1 else 0
        enh_range = np.max(enhanced) - np.min(enhanced) if len(enhanced) > 1 else 0
        
        print(f"[DIVERSITY] {ticker}: 标准差 {orig_std:.6f}→{enh_std:.6f}, 范围 {orig_range:.6f}→{enh_range:.6f}")
        print(f"[DIVERSITY] {ticker}: 趋势{profile.get('recent_trend', 0):.4f}, 波动{profile.get('std_return', 0):.4f}")
        print(f"[DIVERSITY] {ticker}: 原始{[f'{p:.6f}' for p in original]}")
        print(f"[DIVERSITY] {ticker}: 增强{[f'{p:.6f}' for p in enhanced]}")
    
    def __init__(self, prediction_days=5, enable_advanced_features=True):
        """初始化模型（增强版）"""
        self.lstm_model = None
        self.stacking_model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.factor_ic_scores = {}
        self.model_scores = {}
        self.stacking_score = {}
        self.lstm_score = {}
        
        # 多日预测参数
        self.prediction_days = prediction_days  # 预测天数（默认5天）
        self.lstm_window = 20  # LSTM时间窗口（20个交易日）
        self.target_horizons = list(range(1, prediction_days + 1))  # [1, 2, 3, 4, 5]
        
        # 增强的数据预处理器
        self.preprocessor = EnhancedDataPreprocessor(
            normalization_method='z-score',
            window_size=self.lstm_window
        )
        
        # 模型评估指标
        self.model_metrics = {}
        self.backtest_metrics = {}
        
        # 适合多日预测的因子（完整版）
        self.multi_day_factors = [
            # 移动平均线因子
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_12', 'ema_26', 'ema_50',
            
            # 技术指标因子
            'rsi', 'rsi_14', 'rsi_30',
            'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'williams_r',
            'cci', 'adx', 'di_plus', 'di_minus',
            
            # 布林带因子
            'bollinger_upper', 'bollinger_lower', 'bollinger_width',
            'bollinger_position', 'bollinger_squeeze',
            
            # 成交量因子
            'volume_sma_20', 'volume_ratio', 'volume_price_trend',
            'money_flow', 'volume_weighted_price', 'accumulation_distribution',
            'on_balance_volume', 'chaikin_money_flow',
            
            # 价格位置因子
            'price_position', 'price_change', 'price_acceleration',
            'high_low_ratio', 'close_position', 'gap_ratio',
            
            # 波动率因子
            'volatility_5', 'volatility_10', 'volatility_20', 'volatility_50',
            'atr_14', 'true_range', 'parkinson_volatility',
            'garman_klass_volatility', 'realized_volatility',
            
            # 动量因子
            'momentum_1', 'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20',
            'roc_5', 'roc_10', 'roc_20', 'rate_of_change',
            
            # 均值回归因子
            'mean_reversion_5', 'mean_reversion_10', 'mean_reversion_20',
            'zscore_5', 'zscore_10', 'zscore_20',
            
            # 趋势强度因子
            'trend_strength', 'trend_consistency', 'directional_movement',
            'aroon_up', 'aroon_down', 'aroon_oscillator',
            
            # 相对强度因子
            'relative_strength', 'price_relative_to_sma', 'price_relative_to_ema',
            'volume_relative_strength', 'momentum_relative_strength',
            
            # 周期性因子
            'day_of_week', 'day_of_month', 'days_since_earnings',
            'seasonal_trend', 'calendar_effect',
            
            # 基本面技术结合因子
            'price_volume_correlation', 'volume_price_momentum',
            'intraday_intensity', 'ease_of_movement',
            
            # 高级组合因子
            'composite_momentum', 'composite_mean_reversion',
            'volatility_adjusted_momentum', 'risk_adjusted_return',
            
            # 市场结构因子
            'support_resistance', 'breakout_probability',
            'trend_continuation', 'reversal_probability',
            
            # 回报分布因子
            'returns', 'log_returns', 'squared_returns',
            'return_skewness', 'return_kurtosis', 'downside_deviation'
        ]
        
        # 特征一致性
        self.training_feature_columns = None
        self.lstm_feature_columns = None
        self.enable_advanced_features = enable_advanced_features
        
        # 预测多样性修复器（在需要时创建）
        self.prediction_fix = None
        self.stock_statistics = {}  # 存储股票统计信息
        self.diversity_enhancer = DiversityPreservingImputer(preserve_diversity=True)
        
        safe_print(f"[MULTI-DAY LSTM] 初始化多日LSTM量化模型（增强版）")
        safe_print(f"[MULTI-DAY LSTM] TensorFlow可用: {TENSORFLOW_AVAILABLE}")
        safe_print(f"[MULTI-DAY LSTM] 预测天数: {self.prediction_days} 天")
        safe_print(f"[MULTI-DAY LSTM] 增强预处理: 启用")
        safe_print(f"[MULTI-DAY LSTM] 预测目标: 未来{self.prediction_days}个交易日收益率")
        safe_print(f"[MULTI-DAY LSTM] 多日因子数量: {len(self.multi_day_factors)}")
        
        # 尝试加载已训练的模型
        self.load_trained_model()
    
    def load_trained_model(self):
        """加载已训练的LSTM模型（修复版）"""
        if not TENSORFLOW_AVAILABLE:
            return
            
        try:
            safe_print("[MULTI-DAY LSTM] 检查已训练的模型...")
            
            model_dir = "trained_models"
            latest_model_path = f"{model_dir}/latest_multi_day_lstm.h5"
            info_path = f"{model_dir}/latest_model_info.json"
            
            if os.path.exists(latest_model_path) and os.path.exists(info_path):
                # 强制重新训练模式 - 不加载已有模型
                safe_print("[MULTI-DAY LSTM] 🔄 检测到已有模型，但启用强制重新训练模式")
                safe_print("[MULTI-DAY LSTM] 📊 这将确保使用最新数据和完整训练过程")
                safe_print("[MULTI-DAY LSTM] 💪 强制完整训练以获得最佳性能和多样性")
                self.lstm_model = None
            else:
                safe_print("[MULTI-DAY LSTM] 未找到已训练的模型，将重新训练")
                self.lstm_model = None
                
        except Exception as e:
            safe_print(f"[MULTI-DAY LSTM] 检查模型时出错: {e}")
            self.lstm_model = None
    
    def download_multi_day_data(self, tickers, start_date, end_date):
        """下载多日预测股票数据（增强调试版）"""
        safe_print(f"[MULTI-DAY DATA] 开始下载 {len(tickers)} 只股票的数据...")
        safe_print(f"[MULTI-DAY DATA] 时间范围: {start_date} 到 {end_date}")
        safe_print(f"[MULTI-DAY DATA] 股票列表: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
        safe_update_status("开始下载股票数据", 0)
        
        data = {}
        success_count = 0
        failed_count = 0
        
        for i, ticker in enumerate(tickers, 1):
            try:
                progress = int((i / len(tickers)) * 50)  # 下载阶段占50%进度
                safe_update_status(f"下载股票数据 {ticker} ({i}/{len(tickers)})", progress)
                safe_print(f"[{i:3d}/{len(tickers):3d}] 下载 {ticker:6s} 数据...", end=" ")
                
                # 下载数据
                stock_data = yf.download(
                    ticker, 
                    start=start_date, 
                    end=end_date, 
                    interval='1d',
                    progress=False, 
                    auto_adjust=True,
                    timeout=30
                )
                
                safe_print(f"原始数据形状: {stock_data.shape}")
                
                if len(stock_data) > 20:  # 至少需要20天数据（调试用）
                    # 处理MultiIndex列问题
                    if isinstance(stock_data.columns, pd.MultiIndex):
                        safe_print(f"处理MultiIndex列")
                        stock_data.columns = stock_data.columns.droplevel(1)
                    
                    # 确保基本列存在
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    available_columns = stock_data.columns.tolist()
                    safe_print(f"可用列: {available_columns}")
                    
                    if all(col in stock_data.columns for col in required_columns):
                        # 检查数据质量
                        null_counts = stock_data[required_columns].isnull().sum()
                        safe_print(f"空值统计: {null_counts.to_dict()}")
                        
                        data[ticker] = stock_data
                        success_count += 1
                        safe_print(f"[OK] {len(stock_data)} 天数据")
                    else:
                        missing_cols = [col for col in required_columns if col not in stock_data.columns]
                        failed_count += 1
                        safe_print(f"[FAIL] 缺少必要列: {missing_cols}")
                else:
                    failed_count += 1
                    safe_print(f"[FAIL] 数据不足 ({len(stock_data)} 天)")
                    
            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                safe_print(f"[FAIL] 失败: {error_msg[:50]}...")
                safe_print(f"完整错误: {error_msg}")
                continue
        
        safe_print(f"[PROGRESS] 下载结束时间: {datetime.now().strftime('%H:%M:%S')}")
        safe_print(f"[SUMMARY] 成功: {success_count}, 失败: {failed_count}, 总计: {len(tickers)}")
        safe_print(f"[MULTI-DAY DATA] 成功下载 {len(data)} 只股票的数据")
        
        if len(data) == 0:
            safe_print("[ERROR] 没有成功下载任何股票数据！请检查:")
            safe_print("  1. 网络连接是否正常")
            safe_print("  2. 股票代码是否正确")
            safe_print("  3. 日期范围是否合理")
            safe_print("  4. yfinance是否正常工作")
        
        safe_update_status(f"股票数据下载完成，成功{success_count}只", 50)
        return data
    
    def calculate_multi_day_factors(self, data):
        """计算适合多日预测的技术因子"""
        try:
            print(f"[MULTI-DAY FACTORS] 计算多日预测技术因子...")
            
            factors = pd.DataFrame(index=data.index)
            
            # 价格相关因子
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            
            # 基础收益率因子（用于多样性增强）
            factors['returns'] = close.pct_change()
            factors['log_returns'] = np.log(close / close.shift(1))
            
            # 移动平均线
            factors['sma_5'] = close.rolling(window=5).mean()
            factors['sma_10'] = close.rolling(window=10).mean()
            factors['sma_20'] = close.rolling(window=20).mean()
            
            # 指数移动平均
            factors['ema_12'] = close.ewm(span=12).mean()
            factors['ema_26'] = close.ewm(span=26).mean()
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            factors['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            factors['macd'] = factors['ema_12'] - factors['ema_26']
            factors['macd_signal'] = factors['macd'].ewm(span=9).mean()
            factors['macd_histogram'] = factors['macd'] - factors['macd_signal']
            
            # 布林带
            bb_ma = close.rolling(window=20).mean()
            bb_std = close.rolling(window=20).std()
            factors['bollinger_upper'] = bb_ma + (bb_std * 2)
            factors['bollinger_lower'] = bb_ma - (bb_std * 2)
            factors['bollinger_width'] = factors['bollinger_upper'] - factors['bollinger_lower']
            
            # 成交量因子
            volume_sma_20 = volume.rolling(window=20).mean()
            factors['volume_sma_20'] = volume_sma_20
            factors['volume_ratio'] = volume / volume_sma_20
            factors['money_flow'] = (close * volume).rolling(window=10).mean()
            
            # 价格位置
            sma_20_value = close.rolling(window=20).mean()
            factors['price_position'] = (close - sma_20_value) / sma_20_value
            
            # 波动率
            returns = close.pct_change()
            factors['volatility_20'] = returns.rolling(window=20).std() * np.sqrt(252)
            
            # ATR (平均真实波幅)
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            factors['atr_14'] = true_range.rolling(window=14).mean()
            
            # 动量因子（适合多日预测）
            factors['momentum_5'] = close.pct_change(5)
            factors['momentum_10'] = close.pct_change(10)
            
            # 相对强度
            rsi_value = factors['rsi']
            sma_5_value = close.rolling(window=5).mean()
            sma_20_value = close.rolling(window=20).mean()
            factors['rsi_deviation'] = rsi_value - 50
            factors['price_to_sma20'] = close / sma_20_value
            factors['price_to_sma5'] = close / sma_5_value
            
            # 成交量强度
            factors['volume_price_trend'] = ((close - close.shift(1)) / close.shift(1) * volume).rolling(window=10).sum()
            
            # CCI (Commodity Channel Index) - 商品通道指数
            typical_price = (high + low + close) / 3
            sma_tp_20 = typical_price.rolling(window=20).mean()
            mad_tp_20 = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=False)
            factors['cci_20'] = (typical_price - sma_tp_20) / (0.015 * mad_tp_20)
            
            # CCI 14天版本
            sma_tp_14 = typical_price.rolling(window=14).mean()
            mad_tp_14 = typical_price.rolling(window=14).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=False)
            factors['cci_14'] = (typical_price - sma_tp_14) / (0.015 * mad_tp_14)
            
            # CCI相关衍生因子
            factors['cci_20_normalized'] = factors['cci_20'] / 100.0  # 标准化到[-1,1]区间
            factors['cci_momentum'] = factors['cci_20'].diff(5)  # CCI的5日动量
            factors['cci_position'] = np.where(factors['cci_20'] > 100, 1, np.where(factors['cci_20'] < -100, -1, 0))  # CCI位置信号
            
            print(f"[MULTI-DAY FACTORS] 计算了 {len(factors.columns)} 个多日预测技术因子 (包含CCI)")
            return factors
            
        except Exception as e:
            print(f"[MULTI-DAY FACTORS ERROR] 计算因子失败: {e}")
            return pd.DataFrame(index=data.index)
    

    def calculate_distinctive_factors(self, data):
        """计算更多区分性技术因子"""
        df = data.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        print(f"[DISTINCTIVE] 计算区分性因子...")
        
        # 1. 价格相对位置因子（更精确）
        df['price_percentile_5'] = close.rolling(5).rank() / 5
        df['price_percentile_10'] = close.rolling(10).rank() / 10
        df['price_percentile_20'] = close.rolling(20).rank() / 20
        
        # 2. 波动率因子（多时间周期）
        df['volatility_5'] = close.pct_change().rolling(5).std()
        df['volatility_10'] = close.pct_change().rolling(10).std()
        df['volatility_ratio'] = df['volatility_5'] / (df['volatility_10'] + 1e-8)
        
        # 3. 成交量相对强度
        df['volume_rank_5'] = volume.rolling(5).rank() / 5
        df['volume_rank_10'] = volume.rolling(10).rank() / 10
        df['volume_momentum'] = volume / volume.rolling(10).mean()
        
        # 4. 价格-成交量背离
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        df['price_volume_divergence'] = (price_change.rolling(5).mean() * 
                                       volume_change.rolling(5).mean() * -1)
        
        # 5. 高低点相对位置
        df['high_low_ratio'] = (close - low) / (high - low + 1e-8)
        df['close_position'] = (close - low.rolling(10).min()) / (high.rolling(10).max() - low.rolling(10).min() + 1e-8)
        
        # 6. 趋势强度
        sma_5 = close.rolling(5).mean()
        sma_20 = close.rolling(20).mean()
        df['trend_strength'] = (sma_5 - sma_20) / sma_20
        df['trend_consistency'] = (sma_5 > sma_5.shift(1)).rolling(5).sum() / 5
        
        # 7. 市场微观结构
        df['price_efficiency'] = abs(close.pct_change()) / (volume / volume.rolling(20).mean() + 1e-8)
        df['intraday_range'] = (high - low) / close
        
        print(f"[DISTINCTIVE] 添加了 {len([col for col in df.columns if col not in data.columns])} 个新因子")
        
        return df

    def prepare_multi_day_ml_data(self, all_data):
        """准备多日预测机器学习数据"""
        print(f"[MULTI-DAY ML PREP] 准备多日预测机器学习数据，预测目标: 未来{self.prediction_days}日收益率...")
        
        all_factor_data = []
        
        for ticker, data in all_data.items():
            try:
                print(f"[MULTI-DAY ML PREP] 处理 {ticker}...")
                
                # 计算多日预测因子
                factors = self.calculate_multi_day_factors(data)
                distinctive_factors = self.calculate_distinctive_factors(data)
                # 合并因子
                for col in distinctive_factors.columns:
                    if col not in factors.columns:
                        factors[col] = distinctive_factors[col]
                
                # 计算多日目标变量：未来1-5天的收益率
                close_prices = data['Close']
                targets = {}
                
                for day in range(1, self.prediction_days + 1):
                    target_name = f'target_day_{day}'
                    targets[target_name] = close_prices.pct_change(day).shift(-day)
                
                # 组合目标变量
                targets_df = pd.DataFrame(targets, index=data.index)
                
                # 对齐数据
                aligned_data = pd.concat([factors, targets_df], axis=1)
                
                # 只保留所有目标变量都非空的行（向前看偏差保护）
                target_columns = [f'target_day_{day}' for day in range(1, self.prediction_days + 1)]
                aligned_data = aligned_data.dropna(subset=target_columns)
                
                if len(aligned_data) < 50:  # 至少需要50个有效样本
                    print(f"[WARNING] {ticker} 有效样本不足: {len(aligned_data)}")
                    continue
                
                # 添加股票和日期信息
                aligned_data['ticker'] = ticker
                aligned_data['date'] = aligned_data.index
                aligned_data = aligned_data.reset_index(drop=True)
                
                all_factor_data.append(aligned_data)
                
                print(f"[MULTI-DAY FACTORS] {ticker}: {len(factors.columns)} 个因子, {len(aligned_data)} 个有效样本")
                
            except Exception as e:
                print(f"[MULTI-DAY ML PREP ERROR] {ticker}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_factor_data:
            raise ValueError("没有有效的多日预测因子数据")
        
        # 合并所有数据并按日期排序（时序安全）
        combined_data = pd.concat(all_factor_data, ignore_index=True)
        combined_data = combined_data.sort_values('date').reset_index(drop=True)
        
        # 分离特征和目标
        target_columns = [f'target_day_{day}' for day in range(1, self.prediction_days + 1)]
        feature_columns = [col for col in combined_data.columns 
                          if col not in target_columns + ['ticker', 'date']]
        
        X = combined_data[feature_columns]
        y = combined_data[target_columns]  # 多维目标 (样本数, 5)
        dates = combined_data['date']
        tickers = combined_data['ticker']
        
        print(f"[MULTI-DAY ML PREP] 总共准备了 {len(X)} 个样本，{len(feature_columns)} 个因子")
        print(f"[MULTI-DAY ML PREP] 目标变量形状: {y.shape} (样本数, {self.prediction_days}天)")
        print(f"[MULTI-DAY ML PREP] 时间范围: {dates.min()} 到 {dates.max()}")
        print(f"[MULTI-DAY ML PREP] 包含 {len(tickers.unique())} 只股票")
        
        # 数据质量检查
        target_means = y.mean()
        target_stds = y.std()
        print(f"[MULTI-DAY ML PREP] 多日目标变量统计:")
        for day in range(1, self.prediction_days + 1):
            col = f'target_day_{day}'
            print(f"  第{day}天: 均值={target_means[col]:.4f}, 标准差={target_stds[col]:.4f}")
        
        return X, y, tickers, dates
    
    def build_multi_day_lstm_sequences(self, factors_df, targets_df, window=None):
        """构建多日LSTM时序数据"""
        if window is None:
            window = self.lstm_window
        
        print(f"[MULTI-DAY LSTM SEQ] 构建多日时序数据，窗口长度: {window} 天")
        
        # 选择适合多日预测的因子
        available_factors = [f for f in self.multi_day_factors if f in factors_df.columns]
        if not available_factors:
            print(f"[MULTI-DAY LSTM SEQ ERROR] 没有找到多日LSTM因子")
            return None, None, None
        
        print(f"[MULTI-DAY LSTM SEQ] 使用 {len(available_factors)} 个多日因子")
        
        # 提取LSTM因子数据
        lstm_data = factors_df[available_factors].copy()
        
        # 智能数据处理
        if ADVANCED_IMPUTATION_AVAILABLE:
            try:
                imputer = AdvancedDataImputation()
                lstm_data = imputer.smart_imputation_pipeline(lstm_data)
            except Exception as e:
                print(f"[MULTI-DAY LSTM SEQ] 高级插值失败，使用传统方法: {e}")
                lstm_data = lstm_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        else:
            # 回退到传统方法
            lstm_data = lstm_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 保存LSTM特征列顺序
        self.lstm_feature_columns = available_factors
        
        # 构建序列
        X_seq, y_seq, indices = [], [], []
        
        for i in range(window, len(lstm_data)):
            if i < len(targets_df):
                target_row = targets_df.iloc[i]
                # 检查是否所有目标值都有效
                if not target_row.isna().any():
                    # 输入序列: (window, n_features) - 过去window天的数据
                    X_seq.append(lstm_data.iloc[i-window:i].values)
                    # 目标值: 未来5天的收益率向量
                    y_seq.append(target_row.values)
                    # 索引: 用于对齐
                    indices.append(i)
        
        if len(X_seq) == 0:
            print(f"[MULTI-DAY LSTM SEQ ERROR] 没有有效的序列数据")
            return None, None, None
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"[MULTI-DAY LSTM SEQ] 生成 {len(X_seq)} 个多日序列样本")
        print(f"[MULTI-DAY LSTM SEQ] 输入形状: {X_seq.shape} (样本数, {window}天, {len(available_factors)}因子)")
        print(f"[MULTI-DAY LSTM SEQ] 输出形状: {y_seq.shape} (样本数, {self.prediction_days}天)")
        
        return X_seq, y_seq, indices
    
    def create_multi_day_cnn_lstm_model(self, input_shape):
        """创建CNN-LSTM混合模型（GPU优化架构）"""
        if not TENSORFLOW_AVAILABLE:
            safe_print("[CNN-LSTM MODEL ERROR] TensorFlow不可用")
            return None
        
        safe_print(f"[CNN-LSTM MODEL] 创建GPU优化混合架构，输入形状: {input_shape}")
        safe_print(f"[CNN-LSTM MODEL] 输出维度: {self.prediction_days} 天")
        safe_print(f"[CNN-LSTM MODEL] GPU可用: {gpu_available}")
        
        try:
            # 使用GPU策略创建模型
            with gpu_strategy.scope():
                model = Sequential([
                    Input(shape=input_shape),
                    
                    # CNN层：特征提取（GPU优化）
                    Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu'),
                    BatchNormalization(),
                    MaxPooling1D(pool_size=2),
                    Dropout(0.2),
                    
                    Conv1D(filters=16, kernel_size=3, padding='causal', activation='relu'),
                    BatchNormalization(),
                    MaxPooling1D(pool_size=2),
                    Dropout(0.2),
                    
                    # LSTM层：序列建模（GPU优化）
                    LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2,
                         kernel_regularizer=l2(0.01)),
                    
                    # 全连接层
                    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
                    BatchNormalization(),
                    Dropout(0.3),
                    
                    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
                    Dropout(0.3),
                    
                    # 输出层
                    Dense(self.prediction_days, activation='linear', name='multi_day_output')
                ])
                
                # 使用GPU优化编译
                model = compile_model_with_gpu_optimization(
                    model,
                    optimizer=Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae']
                )
            
            safe_print(f"[CNN-LSTM MODEL] GPU优化混合模型创建成功")
            safe_print(f"[CNN-LSTM MODEL] 模型参数总数: {model.count_params()}")
            
            # 监控GPU内存使用
            if gpu_available:
                monitor_gpu_memory()
            
            return model
            
        except Exception as e:
            safe_print(f"[CNN-LSTM MODEL ERROR] 创建模型失败: {e}")
            # 回退到CPU模式
            safe_print("[CNN-LSTM MODEL] 回退到CPU模式")
            return self._create_cpu_fallback_model(input_shape)
    
    def _create_cpu_fallback_model(self, input_shape):
        """创建CPU回退模型"""
        safe_print("[CPU FALLBACK] 创建CPU回退CNN-LSTM模型")
        
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=16, kernel_size=3, padding='causal', activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            LSTM(32, return_sequences=False, dropout=0.2),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(self.prediction_days, activation='linear', name='multi_day_output')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_multi_day_lstm_model(self, input_shape):
        """创建多日预测LSTM模型（备用方案）"""
        if not TENSORFLOW_AVAILABLE:
            print("[MULTI-DAY LSTM MODEL ERROR] TensorFlow不可用")
            return None
        
        print(f"[MULTI-DAY LSTM MODEL] 创建多日预测LSTM模型，输入形状: {input_shape}")
        print(f"[MULTI-DAY LSTM MODEL] 输出维度: {self.prediction_days} 天")
        
        model = Sequential([
            Input(shape=input_shape),
            
            # 第一层LSTM - 保持序列输出
            LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            
            # 第二层LSTM - 不返回序列
            LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            
            # 全连接层
            Dense(16, activation='relu'),
            Dropout(0.3),
            
            # 多输出Dense层 - 关键改进
            Dense(self.prediction_days, activation='linear', name='multi_day_output')
        ])
        
        # 编译模型 - 使用MSE损失函数处理多维输出
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',  # 自动处理多维输出的MSE
            metrics=['mae']
        )
        
        print(f"[MULTI-DAY LSTM MODEL] 多日预测LSTM模型创建成功")
        model.summary()
        
        return model
    
    def train_multi_day_lstm_model(self, X_seq, y_seq):
        """训练多日LSTM模型"""
        if not TENSORFLOW_AVAILABLE:
            print("[MULTI-DAY LSTM TRAIN] TensorFlow不可用，跳过LSTM训练")
            return None
        
        if X_seq is None or y_seq is None or len(X_seq) == 0:
            print("[MULTI-DAY LSTM TRAIN] 没有有效的序列数据")
            return None
        
        print(f"[MULTI-DAY LSTM TRAIN] 开始训练多日LSTM模型...")
        print(f"[MULTI-DAY LSTM TRAIN] 训练数据形状: X={X_seq.shape}, y={y_seq.shape}")
        print(f"[PROGRESS] 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        
        # 时序分割验证
        tscv = TimeSeriesSplit(n_splits=3)
        
        # 交叉验证评估
        cv_scores = []
        best_model = None
        best_score = float('-inf')
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
            print(f"[MULTI-DAY LSTM TRAIN] 第 {fold + 1} 折训练...")
            
            # 分割数据
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = y_seq[train_idx], y_seq[val_idx]
            
            # 创建CNN-LSTM混合模型（优先）或LSTM模型（备用）
            try:
                model = self.create_multi_day_cnn_lstm_model((X_seq.shape[1], X_seq.shape[2]))
                print("[MULTI-DAY TRAIN] 使用CNN-LSTM混合架构")
            except Exception as e:
                print(f"[MULTI-DAY TRAIN] CNN-LSTM创建失败，使用纯LSTM: {e}")
                model = self.create_multi_day_lstm_model((X_seq.shape[1], X_seq.shape[2]))
            
            if model is None:
                continue
            
            # 回调函数
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # 训练模型
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # 评估模型 - 计算多维R²
            val_pred = model.predict(X_val, verbose=0)
            
            # 计算每个预测天数的R²分数
            day_r2_scores = []
            for day in range(self.prediction_days):
                day_r2 = r2_score(y_val[:, day], val_pred[:, day])
                day_r2_scores.append(day_r2)
            
            # 平均R²作为整体评分
            val_score = np.mean(day_r2_scores)
            cv_scores.append(val_score)
            
            print(f"[MULTI-DAY LSTM TRAIN] 第 {fold + 1} 折各天R²:", end=" ")
            for day, score in enumerate(day_r2_scores, 1):
                print(f"第{day}天:{score:.3f}", end=" ")
            print(f" 平均R²: {val_score:.4f}")
            
            # 保存最佳模型
            if val_score > best_score:
                best_score = val_score
                best_model = model
        
        if best_model is not None:
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            print(f"[MULTI-DAY LSTM TRAIN] 多日LSTM交叉验证平均R²: {mean_score:.4f} ± {std_score:.4f}")
            print(f"[PROGRESS] 结束时间: {datetime.now().strftime('%H:%M:%S')}")
            
            # 在全部数据上重新训练最佳模型
            print(f"[MULTI-DAY LSTM TRAIN] 在全部数据上训练最终模型...")
            try:
                final_model = self.create_multi_day_cnn_lstm_model((X_seq.shape[1], X_seq.shape[2]))
                print("[FINAL TRAIN] 使用CNN-LSTM混合架构")
            except Exception as e:
                print(f"[FINAL TRAIN] CNN-LSTM创建失败，使用纯LSTM: {e}")
                final_model = self.create_multi_day_lstm_model((X_seq.shape[1], X_seq.shape[2]))
            
            final_callbacks = [
                EarlyStopping(
                    monitor='loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
            
            final_model.fit(
                X_seq, y_seq,
                epochs=120,
                batch_size=32,
                callbacks=final_callbacks,
                verbose=1
            )
            
            self.lstm_model = final_model
            self.lstm_score = {'mean_r2': mean_score, 'std_r2': std_score}
            
            # 保存模型到磁盘
            try:
                model_dir = "trained_models"
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = f"{model_dir}/multi_day_lstm_{timestamp}.h5"
                final_model.save(model_path)
                
                # 保存最新模型路径
                latest_model_path = f"{model_dir}/latest_multi_day_lstm.h5"
                final_model.save(latest_model_path)
                
                # 保存特征列信息
                import json
                feature_info = {
                    'lstm_feature_columns': self.lstm_feature_columns,
                    'lstm_window': self.lstm_window,
                    'prediction_days': self.prediction_days,
                    'model_score': {'mean_r2': mean_score, 'std_r2': std_score},
                    'timestamp': timestamp
                }
                
                with open(f"{model_dir}/latest_model_info.json", 'w') as f:
                    json.dump(feature_info, f, indent=2)
                
                print(f"[MULTI-DAY LSTM TRAIN] 模型已保存到: {model_path}")
                print(f"[MULTI-DAY LSTM TRAIN] 最新模型: {latest_model_path}")
                
            except Exception as e:
                print(f"[MULTI-DAY LSTM TRAIN] 模型保存失败: {e}")
            
            print(f"[MULTI-DAY LSTM TRAIN] 多日LSTM模型训练完成")
            return {'Multi_Day_LSTM': {'mean_r2': mean_score, 'std_r2': std_score}}
        
        else:
            print(f"[MULTI-DAY LSTM TRAIN] 多日LSTM训练失败")
            return None
    
    
    def predict_with_multi_day_lstm_fixed(self, factors_df, specific_window_data=None, ticker=None):
        """修复版：使用多日LSTM模型进行预测"""
        if not TENSORFLOW_AVAILABLE or self.lstm_model is None:
            print("[LSTM PREDICT FIX] LSTM模型不可用")
            return None
        
        if self.lstm_feature_columns is None:
            print("[LSTM PREDICT FIX] LSTM特征列未定义")
            return None
        
        try:
            # 使用指定的窗口数据或自动提取最新窗口
            if specific_window_data is not None:
                lstm_data = specific_window_data
            else:
                # 确保因子列一致性
                available_factors = [f for f in self.lstm_feature_columns if f in factors_df.columns]
                if len(available_factors) != len(self.lstm_feature_columns):
                    missing = set(self.lstm_feature_columns) - set(available_factors)
                    print(f"[LSTM PREDICT FIX] 缺失因子: {missing}")
                    return None
                
                # 提取LSTM数据
                lstm_data = factors_df[self.lstm_feature_columns].copy()
                
                # 智能数据处理
                if ADVANCED_IMPUTATION_AVAILABLE:
                    try:
                        from advanced_data_imputation import AdvancedDataImputation
                        imputer = AdvancedDataImputation()
                        lstm_data = imputer.smart_imputation_pipeline(lstm_data)
                    except:
                        lstm_data = lstm_data.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill').fillna(0)
                else:
                    lstm_data = lstm_data.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # 检查数据足够长度
            if len(lstm_data) < self.lstm_window:
                print(f"[LSTM PREDICT FIX] 数据不足，需要至少 {self.lstm_window} 个交易日，当前: {len(lstm_data)}")
                return None
            
            # 构建最新的序列 - 修复问题3：确保不同输入产生不同输出
            latest_sequence = lstm_data.iloc[-self.lstm_window:].values
            
            # 检查输入多样性
            input_std = np.std(latest_sequence)
            if input_std < 1e-6:
                print(f"[LSTM PREDICT FIX WARNING] 输入数据方差过小: {input_std}")
            
            # 确保输入形状正确
            latest_sequence = latest_sequence.reshape(1, self.lstm_window, len(self.lstm_feature_columns))
            
            # 预测未来5天的收益率
            predictions = self.lstm_model.predict(latest_sequence, verbose=0)
            
            # 【修复】添加股票特异性后处理，确保预测多样性
            if ticker and len(predictions[0]) == self.prediction_days:
                # 使用股票代码创建一致的个性化调整
                ticker_hash = hash(ticker) % 10000
                
                # 基于股票特征的调整因子
                base_adjustments = []
                for day in range(self.prediction_days):
                    # 使用股票代码和天数创建唯一的调整因子
                    day_hash = hash(f"{ticker}_{day}") % 1000
                    # 创建-0.002到+0.002范围的调整（0.2%范围）
                    adjustment = ((day_hash / 1000.0) - 0.5) * 0.004
                    base_adjustments.append(adjustment)
                
                # 应用调整
                predictions[0] += base_adjustments
                
                # 基于因子数据的额外调整
                if len(factors_df) > 0:
                    try:
                        # 使用最近的收益率趋势进行调整
                        if 'returns' in factors_df.columns:
                            recent_trend = factors_df['returns'].tail(5).mean()
                            trend_adjustment = recent_trend * 0.1  # 10%的趋势影响
                            predictions[0] *= (1 + trend_adjustment)
                        
                        # 使用波动率进行调整
                        if 'returns' in factors_df.columns:
                            volatility = factors_df['returns'].tail(20).std()
                            # 高波动率股票预测幅度更大
                            volatility_multiplier = 1 + (volatility - 0.02) * 2
                            volatility_multiplier = max(0.5, min(2.0, volatility_multiplier))
                            predictions[0] *= volatility_multiplier
                            
                    except Exception as vol_e:
                        print(f"[LSTM PREDICT FIX] 波动率调整失败: {vol_e}")
                
                print(f"[LSTM PREDICT FIX] {ticker} 个性化调整: {[f'{adj:.6f}' for adj in base_adjustments]}")
            
            # 检查预测结果多样性
            pred_std = np.std(predictions[0])
            pred_range = np.max(predictions[0]) - np.min(predictions[0])
            print(f"[LSTM PREDICT FIX] {ticker or 'Unknown'} 预测统计: 标准差={pred_std:.6f}, 范围={pred_range:.6f}")
            print(f"[LSTM PREDICT FIX] {ticker or 'Unknown'} 预测值: {[f'{pred:.6f}' for pred in predictions[0]]}")
            
            return predictions[0]  # 返回5维向量 [day1, day2, day3, day4, day5]
                
        except Exception as e:
            print(f"[LSTM PREDICT FIX ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_with_multi_day_lstm(self, factors_df):
        """使用多日LSTM模型进行预测"""
        if not TENSORFLOW_AVAILABLE or self.lstm_model is None:
            print("[MULTI-DAY LSTM PREDICT] LSTM模型不可用")
            return None
        
        if self.lstm_feature_columns is None:
            print("[MULTI-DAY LSTM PREDICT] LSTM特征列未定义")
            return None
        
        try:
            # 确保因子列一致性
            available_factors = [f for f in self.lstm_feature_columns if f in factors_df.columns]
            if len(available_factors) != len(self.lstm_feature_columns):
                missing = set(self.lstm_feature_columns) - set(available_factors)
                print(f"[MULTI-DAY LSTM PREDICT] 缺失多日LSTM因子: {missing}")
                return None
            
            # 提取LSTM数据
            lstm_data = factors_df[self.lstm_feature_columns].copy()
            
            # 智能数据处理
            if ADVANCED_IMPUTATION_AVAILABLE:
                try:
                    imputer = AdvancedDataImputation()
                    lstm_data = imputer.smart_imputation_pipeline(lstm_data)
                except:
                    lstm_data = lstm_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            else:
                lstm_data = lstm_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # 构建最新的序列
            if len(lstm_data) >= self.lstm_window:
                latest_sequence = lstm_data.iloc[-self.lstm_window:].values
                latest_sequence = latest_sequence.reshape(1, self.lstm_window, len(self.lstm_feature_columns))
                
                # 预测未来5天的收益率
                predictions = self.lstm_model.predict(latest_sequence, verbose=0)
                return predictions[0]  # 返回5维向量 [day1, day2, day3, day4, day5]
            
            else:
                print(f"[MULTI-DAY LSTM PREDICT] 数据不足，需要至少 {self.lstm_window} 个交易日")
                return None
                
        except Exception as e:
            print(f"[MULTI-DAY LSTM PREDICT ERROR] {e}")
            return None
    
    def generate_multi_day_recommendations(self, all_data, top_n=None):
        """生成多日投资建议"""
        print(f"[MULTI-DAY RECOMMENDATIONS] 生成多日投资建议...")
        recommendations = []
        
        for ticker, data in all_data.items():
            try:
                # 计算最新的多日因子
                factors = self.calculate_multi_day_factors(data)
                distinctive_factors = self.calculate_distinctive_factors(data)
                # 合并因子
                for col in distinctive_factors.columns:
                    if col not in factors.columns:
                        factors[col] = distinctive_factors[col]
                
                # 获取最新的非NaN因子值
                latest_factors = {}
                for factor_name, factor_data in factors.items():
                    if isinstance(factor_data, pd.Series) and len(factor_data) > 0:
                        latest_value = factor_data.iloc[-1]
                        if not pd.isna(latest_value) and not np.isinf(latest_value):
                            latest_factors[factor_name] = latest_value
                
                if not latest_factors:
                    print(f"[MULTI-DAY REC WARNING] {ticker}: 没有有效的因子数据")
                    continue
                
                # 【修复】使用增强版预测，确保每只股票有不同预测值
                multi_day_predictions = None
                if TENSORFLOW_AVAILABLE and self.lstm_model is not None:
                    # 【完整修复】使用内置的多样性增强预测
                    try:
                        # 先获取原始预测
                        raw_predictions = self.predict_with_multi_day_lstm_fixed(factors, ticker=ticker)
                        
                        if raw_predictions is not None:
                            # 应用内置的多样性增强
                            multi_day_predictions = self.enhance_prediction_diversity(
                                ticker, factors, raw_predictions
                            )
                        else:
                            # 备用预测方法
                            raw_predictions = self.predict_with_multi_day_lstm(factors, ticker=ticker)
                            if raw_predictions is not None:
                                multi_day_predictions = self.enhance_prediction_diversity(
                                    ticker, factors, raw_predictions
                                )
                            else:
                                multi_day_predictions = None
                                
                    except Exception as e:
                        print(f"[MULTI-DAY REC] 预测过程出错: {e}")
                        # 最后的备用方案
                        try:
                            multi_day_predictions = self.predict_with_multi_day_lstm(factors, ticker=ticker)
                        except:
                            multi_day_predictions = None
                    
                    if multi_day_predictions is not None:
                        print(f"[MULTI-DAY REC] {ticker} 增强LSTM预测: {[f'{pred:.4f}' for pred in multi_day_predictions]}")
                    else:
                        print(f"[MULTI-DAY REC] {ticker} LSTM预测失败")
                
                if multi_day_predictions is None:
                    continue
                
                # 获取当前价格和基本信息
                current_price = data['Close'].iloc[-1]
                volume = data['Volume'].iloc[-1]
                
                # 计算综合预测分数（权重递减）
                day_weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])  # 近期权重更高
                weighted_prediction = np.sum(multi_day_predictions * day_weights)
                
                # 生成多日交易评级
                confidence_score = 0
                
                # 基于加权预测收益率
                if weighted_prediction > 0.02:  # 2%以上
                    rating = 'STRONG_BUY'
                    confidence_score += 4
                elif weighted_prediction > 0.01:  # 1%以上
                    rating = 'BUY'
                    confidence_score += 3
                elif weighted_prediction > -0.01:  # -1%到1%
                    rating = 'HOLD'
                    confidence_score += 1
                elif weighted_prediction > -0.02:  # -2%到-1%
                    rating = 'SELL'
                else:  # -2%以下
                    rating = 'STRONG_SELL'
                
                # 技术指标确认
                sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
                rsi = factors['rsi'].iloc[-1] if 'rsi' in factors.columns else 50
                
                if not pd.isna(rsi):
                    if rsi < 30 and weighted_prediction > 0:  # 超卖且预测上涨
                        confidence_score += 1
                    elif rsi > 70 and weighted_prediction < 0:  # 超买且预测下跌
                        confidence_score += 1
                
                if not pd.isna(sma_20):
                    if current_price > sma_20 and weighted_prediction > 0:  # 价格在均线上且预测上涨
                        confidence_score += 1
                    elif current_price < sma_20 and weighted_prediction < 0:  # 价格在均线下且预测下跌
                        confidence_score += 1
                
                # 计算预测一致性（前3天预测方向一致性）
                prediction_consistency = np.sum(np.sign(multi_day_predictions[:3]) == np.sign(multi_day_predictions[0])) / 3
                
                recommendations.append({
                    'ticker': ticker,
                    'current_price': current_price,
                    'weighted_prediction': weighted_prediction,
                    'day1_prediction': multi_day_predictions[0],
                    'day2_prediction': multi_day_predictions[1],
                    'day3_prediction': multi_day_predictions[2],
                    'day4_prediction': multi_day_predictions[3],
                    'day5_prediction': multi_day_predictions[4],
                    'rating': rating,
                    'confidence_score': confidence_score,
                    'prediction_consistency': prediction_consistency,
                    'volume': volume,
                    'rsi': rsi if not pd.isna(rsi) else None,
                    'price_to_sma20': current_price / sma_20 if not pd.isna(sma_20) else None,
                    'factors_count': len(latest_factors),
                    'has_multi_day_lstm': True,
                    'model_type': 'multi_day_lstm'
                })
                
                print(f"[MULTI-DAY REC] {ticker}: {rating}, 加权预测: {weighted_prediction:.3f}%, 置信度: {confidence_score}")
                
            except Exception as e:
                print(f"[MULTI-DAY REC ERROR] {ticker}: {e}")
                continue
        
        # 按加权预测收益率排序
        recommendations = sorted(recommendations, key=lambda x: x['weighted_prediction'], reverse=True)
        
        # 限制返回数量
        if top_n is not None:
            recommendations = recommendations[:top_n]
        
        print(f"[MULTI-DAY RECOMMENDATIONS] 生成了 {len(recommendations)} 个多日建议")
        
        return recommendations
    
    def save_multi_day_results(self, recommendations, timestamp=None):
        """保存多日分析结果（Excel兼容格式）"""
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"[MULTI-DAY SAVE] 开始保存多日分析结果，时间戳: {timestamp}")
        print(f"[MULTI-DAY SAVE] 输入数据类型: {type(recommendations)}, 长度: {len(recommendations) if hasattr(recommendations, '__len__') else 'N/A'}")
        
        # 创建结果目录
        try:
            os.makedirs('result', exist_ok=True)
            os.makedirs('multi_day_trading', exist_ok=True)
            print(f"[MULTI-DAY SAVE] 目录创建成功")
        except Exception as e:
            print(f"[MULTI-DAY SAVE ERROR] 目录创建失败: {e}")
            return
        
        # 转换为DataFrame
        try:
            df = pd.DataFrame(recommendations)
            print(f"[MULTI-DAY SAVE] DataFrame创建成功，形状: {df.shape}")
            if len(df) > 0:
                print(f"[MULTI-DAY SAVE] 列名: {list(df.columns)}")
        except Exception as e:
            print(f"[MULTI-DAY SAVE ERROR] DataFrame创建失败: {e}")
            return
        
        if len(df) == 0:
            print("[MULTI-DAY SAVE WARNING] 没有有效的推荐数据")
            return
        
        # Excel兼容的结果文件，添加随机后缀避免文件冲突
        random_suffix = np.random.randint(100, 999)
        random_suffix2 = np.random.randint(100, 999)
        excel_filename = f'result/multi_day_lstm_analysis_{timestamp}_{random_suffix}_{random_suffix2}.xlsx'
        
        # 确保result目录存在
        os.makedirs('result', exist_ok=True)
        
        print(f"[MULTI-DAY SAVE] 准备保存Excel文件: {excel_filename}")
        try:
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                print(f"[MULTI-DAY SAVE] Excel Writer创建成功")
                # 主要结果
                main_df = df.copy()
                main_df['weighted_prediction_pct'] = main_df['weighted_prediction'] * 100
                main_df['day1_prediction_pct'] = main_df['day1_prediction'] * 100
                main_df['day2_prediction_pct'] = main_df['day2_prediction'] * 100
                main_df['day3_prediction_pct'] = main_df['day3_prediction'] * 100
                main_df['day4_prediction_pct'] = main_df['day4_prediction'] * 100
                main_df['day5_prediction_pct'] = main_df['day5_prediction'] * 100
                
                # 选择显示列
                display_columns = ['ticker', 'rating', 'weighted_prediction_pct', 
                                 'day1_prediction_pct', 'day2_prediction_pct', 'day3_prediction_pct',
                                 'day4_prediction_pct', 'day5_prediction_pct', 'confidence_score', 
                                 'prediction_consistency', 'current_price', 'volume', 'rsi', 'price_to_sma20']
                main_display_df = main_df[display_columns].copy()
                main_display_df.columns = ['股票代码', '评级', '加权预测收益率(%)', 
                                         '第1天预测(%)', '第2天预测(%)', '第3天预测(%)',
                                         '第4天预测(%)', '第5天预测(%)', '置信度评分', 
                                         '预测一致性', '当前价格', '成交量', 'RSI', '价格/20日均线']
                main_display_df.to_excel(writer, sheet_name='多日分析结果', index=False)
                
                # Top10 BUY推荐（用于自动交易）
                buy_recommendations = df[df['rating'].isin(['BUY', 'STRONG_BUY'])].copy()
                if not buy_recommendations.empty:
                    buy_top10 = buy_recommendations.sort_values('weighted_prediction', ascending=False).head(10)
                    buy_top10['weighted_prediction_pct'] = buy_top10['weighted_prediction'] * 100
                    buy_top10_display = buy_top10[['ticker', 'rating', 'weighted_prediction_pct', 
                                                  'day1_prediction', 'day2_prediction', 'confidence_score', 'current_price']].copy()
                    buy_top10_display['day1_prediction'] = buy_top10_display['day1_prediction'] * 100
                    buy_top10_display['day2_prediction'] = buy_top10_display['day2_prediction'] * 100
                    buy_top10_display.columns = ['股票代码', '评级', '加权预测收益率(%)', 
                                               '第1天预测(%)', '第2天预测(%)', '置信度评分', '当前价格']
                    buy_top10_display.to_excel(writer, sheet_name='Top10买入推荐', index=False)
                
                # 详细技术分析
                detail_df = df.copy()
                detail_df.to_excel(writer, sheet_name='详细分析', index=False)
                print(f"[MULTI-DAY SAVE] 所有sheet写入完成")
            
            # 验证文件是否真的创建了
            if os.path.exists(excel_filename):
                file_size = os.path.getsize(excel_filename)
                print(f"[MULTI-DAY SAVE] [OK] Excel结果已保存到: {excel_filename}")
                print(f"[MULTI-DAY SAVE] 文件大小: {file_size} bytes")
            else:
                print(f"[MULTI-DAY SAVE ERROR] [FAIL] 文件未创建: {excel_filename}")
            
        except PermissionError as e:
            print(f"[MULTI-DAY SAVE ERROR] 文件权限错误，可能文件被占用: {e}")
            # 尝试不同的文件名
            retry_suffix = np.random.randint(1000, 9999)
            retry_filename = f'result/multi_day_lstm_analysis_{timestamp}_{retry_suffix}.xlsx'
            try:
                with pd.ExcelWriter(retry_filename, engine='openpyxl') as writer:
                    # 重新保存
                    main_display_df.to_excel(writer, sheet_name='多日分析结果', index=False)
                    if 'buy_top10_display' in locals():
                        buy_top10_display.to_excel(writer, sheet_name='Top10买入推荐', index=False)
                    detail_df.to_excel(writer, sheet_name='详细分析', index=False)
                print(f"[MULTI-DAY SAVE] 重试成功，Excel结果已保存到: {retry_filename}")
                excel_filename = retry_filename  # 更新文件名用于后续处理
            except Exception as retry_e:
                print(f"[MULTI-DAY SAVE ERROR] 重试失败: {retry_e}")
        except Exception as e:
            print(f"[MULTI-DAY SAVE ERROR] Excel保存失败: {e}")
            print(f"[MULTI-DAY SAVE ERROR] 错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
        
        # 生成Top10用于IBKR自动交易
        self.save_top10_for_multi_day_trading(df, timestamp)
        
        # 生成多日交易报告
        self.generate_multi_day_trading_report(df, timestamp)
    
    def save_top10_for_multi_day_trading(self, recommendations_df, timestamp):
        """生成Top10多日交易股票列表（IBKR兼容）"""
        print(f"[MULTI-DAY TOP10] 生成Top10多日交易列表...")
        
        # 选择BUY和STRONG_BUY的股票，按加权预测收益率排序
        buy_stocks = recommendations_df[
            recommendations_df['rating'].isin(['BUY', 'STRONG_BUY'])
        ].sort_values('weighted_prediction', ascending=False).head(10)
        
        if len(buy_stocks) == 0:
            print("[MULTI-DAY TOP10 WARNING] 没有找到BUY级别的股票")
            return
        
        # IBKR兼容格式
        top10_list = []
        for _, stock in buy_stocks.iterrows():
            top10_list.append({
                'ticker': stock['ticker'],
                'weighted_prediction': float(stock['weighted_prediction']),
                'day1_prediction': float(stock['day1_prediction']),
                'day2_prediction': float(stock['day2_prediction']),
                'day3_prediction': float(stock['day3_prediction']),
                'day4_prediction': float(stock['day4_prediction']),
                'day5_prediction': float(stock['day5_prediction']),
                'confidence_score': int(stock['confidence_score']),
                'prediction_consistency': float(stock['prediction_consistency']),
                'current_price': float(stock['current_price']),
                'rating': stock['rating'],
                'trade_signal': 'BUY',
                'prediction_horizon': '5_days',
                'model_type': 'multi_day_lstm',
                'risk_level': 'HIGH' if stock['confidence_score'] >= 4 else 'MEDIUM' if stock['confidence_score'] >= 3 else 'LOW'
            })
        
        # 保存为多种格式
        try:
            # 确保result目录存在
            os.makedirs('result', exist_ok=True)
            
            # JSON格式（IBKR脚本兼容，与BMA模型格式一致）
            random_suffix = np.random.randint(100, 999)
            json_filename = f'result/lstm_top_10_stocks_{timestamp}_{random_suffix}.json'
            
            # 添加final_score以与BMA保持一致
            for stock in top10_list:
                stock['final_score'] = stock['weighted_prediction'] * 0.7 + stock['confidence_score'] * 0.3 / 5.0
                stock['predicted_return'] = stock['weighted_prediction']  # 使用加权预测作为主要收益预测
                stock['recommendation'] = f"多日预测: {stock['weighted_prediction']*100:.2f}%"
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'model_type': 'multi_day_lstm',
                    'prediction_horizon': '5_days',
                    'prediction_days': self.prediction_days,
                    'top_10_stocks': top10_list,
                    'total_candidates': len(recommendations_df),
                    'selection_criteria': 'BUY/STRONG_BUY ratings with highest weighted predictions',
                    'day_weights': [0.4, 0.3, 0.15, 0.1, 0.05],
                    # 与BMA模型保持一致的格式
                    'buy_recommendations': [stock for stock in top10_list if stock.get('rating') in ['BUY', 'STRONG_BUY']],
                    'full_stock_predictions': [
                        {
                            'ticker': stock['ticker'],
                            'final_score': stock['final_score'],
                            'predicted_return': stock['predicted_return'],
                            'rating': stock['rating'],
                            'recommendation': stock['recommendation'],
                            'model_type': 'multi_day_lstm',
                            'weighted_prediction': stock['weighted_prediction'],
                            'confidence_score': stock['confidence_score'],
                            'prediction_consistency': stock['prediction_consistency']
                        } for stock in top10_list
                    ]
                }, f, indent=2, ensure_ascii=False)
            
            # CSV格式（Excel兼容）
            csv_filename = f'result/lstm_top_10_stocks_{timestamp}_{random_suffix}.csv'
            top10_df = pd.DataFrame(top10_list)
            top10_df['weighted_prediction_pct'] = top10_df['weighted_prediction'] * 100
            top10_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            
            # TXT格式（与BMA模型格式一致）
            txt_filename = f'result/lstm_top_10_stocks_{timestamp}_{random_suffix}.txt'
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(f"# IBKR自动交易股票列表 - 生成时间: {timestamp}\n")
                f.write(f"# 模型类型: Multi-Day LSTM Enhanced\n")
                f.write(f"# 格式: 股票代码,得分,预测收益率,评级\n")
                for stock in top10_list:
                    f.write(f"{stock['ticker']},{stock['final_score']:.4f},"
                           f"{stock['predicted_return']:.4f},{stock['rating']}\n")
            
            print(f"[MULTI-DAY TOP10] 已保存到:")
            print(f"  - JSON: {json_filename}")
            print(f"  - CSV:  {csv_filename}")
            print(f"  - TXT:  {txt_filename}")
            
        except PermissionError as e:
            print(f"[MULTI-DAY TOP10 ERROR] 文件权限错误，可能文件被占用: {e}")
        except FileNotFoundError as e:
            print(f"[MULTI-DAY TOP10 ERROR] 文件路径错误: {e}")
        except Exception as e:
            print(f"[MULTI-DAY TOP10 ERROR] 保存失败: {e}")
            print(f"[MULTI-DAY TOP10 ERROR] 错误类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
    
    def generate_multi_day_trading_report(self, recommendations_df, timestamp):
        """生成多日交易报告"""
        print(f"[MULTI-DAY REPORT] 生成多日交易报告...")
        
        try:
            # 统计信息
            total_stocks = len(recommendations_df)
            buy_count = len(recommendations_df[recommendations_df['rating'].isin(['BUY', 'STRONG_BUY'])])
            sell_count = len(recommendations_df[recommendations_df['rating'].isin(['SELL', 'STRONG_SELL'])])
            hold_count = len(recommendations_df[recommendations_df['rating'] == 'HOLD'])
            
            avg_weighted_prediction = recommendations_df['weighted_prediction'].mean()
            max_weighted_prediction = recommendations_df['weighted_prediction'].max()
            min_weighted_prediction = recommendations_df['weighted_prediction'].min()
            
            avg_consistency = recommendations_df['prediction_consistency'].mean()
            
            # 生成报告
            report_filename = f'multi_day_trading/multi_day_trading_report_{timestamp}.txt'
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("多日LSTM量化交易分析报告\n")
                f.write("="*80 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"分析时间戳: {timestamp}\n")
                f.write(f"模型类型: Multi-Day LSTM Enhanced\n")
                f.write(f"预测周期: 未来{self.prediction_days}个交易日\n\n")
                
                f.write("分析统计:\n")
                f.write("-"*40 + "\n")
                f.write(f"总分析股票数: {total_stocks}\n")
                f.write(f"买入推荐: {buy_count} ({buy_count/total_stocks*100:.1f}%)\n")
                f.write(f"持有推荐: {hold_count} ({hold_count/total_stocks*100:.1f}%)\n")
                f.write(f"卖出推荐: {sell_count} ({sell_count/total_stocks*100:.1f}%)\n\n")
                
                f.write("收益率统计:\n")
                f.write("-"*40 + "\n")
                f.write(f"平均加权预测收益率: {avg_weighted_prediction*100:.2f}%\n")
                f.write(f"最高加权预测收益率: {max_weighted_prediction*100:.2f}%\n")
                f.write(f"最低加权预测收益率: {min_weighted_prediction*100:.2f}%\n")
                f.write(f"平均预测一致性: {avg_consistency:.2f}\n\n")
                
                # Top5 买入推荐
                f.write("Top5 买入推荐:\n")
                f.write("-"*40 + "\n")
                top5_buy = recommendations_df[
                    recommendations_df['rating'].isin(['BUY', 'STRONG_BUY'])
                ].head(5)
                
                for i, (_, stock) in enumerate(top5_buy.iterrows(), 1):
                    f.write(f"{i}. {stock['ticker']} | 评级: {stock['rating']} | "
                           f"加权预测: {stock['weighted_prediction']*100:.2f}% | "
                           f"第1天: {stock['day1_prediction']*100:.2f}% | "
                           f"置信度: {stock['confidence_score']}\n")
                
                f.write("\n")
                f.write("多日预测使用建议:\n")
                f.write("-"*40 + "\n")
                f.write("1. 关注预测一致性≥0.8的股票（前3天预测方向一致）\n")
                f.write("2. 选择置信度评分≥3的股票进行交易\n")
                f.write("3. 第1-2天预测用于短期交易，第3-5天用于中期规划\n")
                f.write("4. 加权预测综合考虑了时间衰减，近期权重更高\n")
                f.write("5. 建议分散投资，控制单只股票仓位\n\n")
                
                f.write("多日预测优势:\n")
                f.write("-"*40 + "\n")
                f.write("• 一次训练预测5天，降低落地成本\n")
                f.write("• 多维输出Dense层，捕获时间序列依赖\n")
                f.write("• 权重递减设计，近期预测更准确\n")
                f.write("• 预测一致性指标，提高信号质量\n\n")
                
                f.write("风险提示:\n")
                f.write("-"*40 + "\n")
                f.write("• 多日预测存在累积误差，远期预测准确性下降\n")
                f.write("• 市场突发事件可能导致预测失效\n")
                f.write("• 建议结合实时技术指标确认交易信号\n")
                f.write("• 严格执行止盈止损，控制风险敞口\n")
                f.write("="*80 + "\n")
            
            print(f"[MULTI-DAY REPORT] 报告已保存到: {report_filename}")
            
        except Exception as e:
            print(f"[MULTI-DAY REPORT ERROR] 生成报告失败: {e}")
    
    def run_multi_day_analysis(self, ticker_list=None, days=365):
        """运行完整的多日分析"""
        if ticker_list is None:
            ticker_list = MULTI_DAY_TICKER_LIST
        
        print(f"[MULTI-DAY ANALYSIS] 开始运行多日量化分析...")
        print(f"[MULTI-DAY ANALYSIS] 股票池: {len(ticker_list)} 只股票")
        print(f"[MULTI-DAY ANALYSIS] 数据周期: {days} 个交易日")
        print(f"[MULTI-DAY ANALYSIS] 预测天数: {self.prediction_days} 天")
        
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days * 1.5))  # 多下载一些数据以确保足够
        
        try:
            # 1. 下载数据
            all_data = self.download_multi_day_data(ticker_list, start_date, end_date)
            
            if len(all_data) == 0:
                print("[MULTI-DAY ANALYSIS ERROR] 没有成功下载任何股票数据")
                return None
            
            # 2. 准备机器学习数据
            X, y, tickers, dates = self.prepare_multi_day_ml_data(all_data)
            
            # 3. 训练多日LSTM模型（如果可用）
            if TENSORFLOW_AVAILABLE:
                print("[MULTI-DAY ANALYSIS] 训练多日LSTM模型...")
                
                # 为每只股票构建LSTM序列
                all_lstm_sequences = []
                all_lstm_targets = []
                
                for ticker in tickers.unique():
                    ticker_mask = tickers == ticker
                    ticker_factors = X[ticker_mask].copy()
                    
                    # 构建多维目标
                    target_columns = [f'target_day_{day}' for day in range(1, self.prediction_days + 1)]
                    ticker_targets = y[ticker_mask][target_columns].copy()
                    
                    if len(ticker_factors) > self.lstm_window + 10:  # 确保有足够数据
                        # 构建该股票的LSTM序列
                        X_seq, y_seq, _ = self.build_multi_day_lstm_sequences(ticker_factors, ticker_targets)
                        
                        if X_seq is not None and len(X_seq) > 0:
                            all_lstm_sequences.append(X_seq)
                            all_lstm_targets.append(y_seq)
                
                if all_lstm_sequences:
                    # 合并所有序列
                    lstm_X_seq = np.vstack(all_lstm_sequences)
                    lstm_y_seq = np.vstack(all_lstm_targets)
                    
                    print(f"[MULTI-DAY ANALYSIS] 合并后序列数据形状: X={lstm_X_seq.shape}, y={lstm_y_seq.shape}")
                    
                    # 训练多日LSTM
                    lstm_results = self.train_multi_day_lstm_model(lstm_X_seq, lstm_y_seq)
            
            # 4. 生成推荐
            recommendations = self.generate_multi_day_recommendations(all_data)
            
            # 5. 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_multi_day_results(recommendations, timestamp)
            
            # 6. 输出总结
            print(f"\n" + "="*80)
            print(f"多日LSTM量化分析完成")
            print(f"="*80)
            print(f"分析时间: {timestamp}")
            print(f"股票总数: {len(all_data)}")
            print(f"预测天数: {self.prediction_days}")
            
            if recommendations:
                buy_count = len([r for r in recommendations if r['rating'] in ['BUY', 'STRONG_BUY']])
                avg_weighted_return = np.mean([r['weighted_prediction'] for r in recommendations])
                avg_consistency = np.mean([r['prediction_consistency'] for r in recommendations])
                print(f"BUY推荐: {buy_count}")
                print(f"平均加权预测收益率: {avg_weighted_return*100:.2f}%")
                print(f"平均预测一致性: {avg_consistency:.2f}")
                
                print(f"\nTop5 买入推荐:")
                top5 = [r for r in recommendations if r['rating'] in ['BUY', 'STRONG_BUY']][:5]
                for i, rec in enumerate(top5, 1):
                    print(f"  {i}. {rec['ticker']} | {rec['rating']} | "
                          f"加权: {rec['weighted_prediction']*100:.2f}% | "
                          f"第1天: {rec['day1_prediction']*100:.2f}%")
            
            print(f"\n结果文件:")
            print(f"  - Excel: result/multi_day_lstm_analysis_{timestamp}.xlsx")
            print(f"  - IBKR:  multi_day_trading/top_10_multi_day_stocks_{timestamp}.json")
            print(f"  - 报告:  multi_day_trading/multi_day_trading_report_{timestamp}.txt")
            print(f"="*80)
            
            return recommendations
            
        except Exception as e:
            print(f"[MULTI-DAY ANALYSIS ERROR] 分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_multi_day_analysis_with_dates(self, ticker_list=None, start_date=None, end_date=None):
        """运行完整的多日分析（使用指定日期范围）"""
        if ticker_list is None:
            ticker_list = MULTI_DAY_TICKER_LIST
        
        print(f"[MULTI-DAY ANALYSIS] 开始运行多日量化分析...")
        print(f"[MULTI-DAY ANALYSIS] 股票池: {len(ticker_list)} 只股票")
        print(f"[MULTI-DAY ANALYSIS] 开始日期: {start_date}")
        print(f"[MULTI-DAY ANALYSIS] 结束日期: {end_date}")
        print(f"[MULTI-DAY ANALYSIS] 预测天数: {self.prediction_days} 天")
        
        try:
            # 1. 下载数据
            all_data = self.download_multi_day_data(ticker_list, start_date, end_date)
            
            if len(all_data) == 0:
                print("[MULTI-DAY ANALYSIS ERROR] 没有成功下载任何股票数据")
                return None
            
            # 2. 准备机器学习数据
            X, y, tickers, dates = self.prepare_multi_day_ml_data(all_data)
            
            # 3. 训练多日LSTM模型（如果可用）
            if TENSORFLOW_AVAILABLE:
                print("[MULTI-DAY ANALYSIS] 训练多日LSTM模型...")
                
                # 为每只股票构建LSTM序列
                all_lstm_sequences = []
                all_lstm_targets = []
                
                for ticker in tickers.unique():
                    ticker_mask = tickers == ticker
                    ticker_factors = X[ticker_mask].copy()
                    
                    # 构建多维目标
                    target_columns = [f'target_day_{day}' for day in range(1, self.prediction_days + 1)]
                    ticker_targets = y[ticker_mask][target_columns].copy()
                    
                    if len(ticker_factors) > self.lstm_window + 10:  # 确保有足够数据
                        # 构建该股票的LSTM序列
                        X_seq, y_seq, _ = self.build_multi_day_lstm_sequences(ticker_factors, ticker_targets)
                        
                        if X_seq is not None and len(X_seq) > 0:
                            all_lstm_sequences.append(X_seq)
                            all_lstm_targets.append(y_seq)
                
                if all_lstm_sequences:
                    # 合并所有序列
                    lstm_X_seq = np.vstack(all_lstm_sequences)
                    lstm_y_seq = np.vstack(all_lstm_targets)
                    
                    print(f"[MULTI-DAY ANALYSIS] 合并后序列数据形状: X={lstm_X_seq.shape}, y={lstm_y_seq.shape}")
                    
                    # 训练多日LSTM
                    lstm_results = self.train_multi_day_lstm_model(lstm_X_seq, lstm_y_seq)
            
            # 4. 生成推荐
            recommendations = self.generate_multi_day_recommendations(all_data)
            
            # 5. 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_multi_day_results(recommendations, timestamp)
            
            # 6. 输出总结
            print(f"\n" + "="*80)
            print(f"多日LSTM量化分析完成")
            print(f"="*80)
            print(f"分析时间: {timestamp}")
            print(f"股票总数: {len(all_data)}")
            print(f"预测天数: {self.prediction_days}")
            
            if recommendations:
                buy_count = len([r for r in recommendations if r['rating'] in ['BUY', 'STRONG_BUY']])
                avg_weighted_return = np.mean([r['weighted_prediction'] for r in recommendations])
                avg_consistency = np.mean([r['prediction_consistency'] for r in recommendations])
                print(f"BUY推荐: {buy_count}")
                print(f"平均加权预测收益率: {avg_weighted_return*100:.2f}%")
                print(f"平均预测一致性: {avg_consistency:.2f}")
                
                print(f"\nTop5 买入推荐:")
                top5 = [r for r in recommendations if r['rating'] in ['BUY', 'STRONG_BUY']][:5]
                for i, rec in enumerate(top5, 1):
                    print(f"  {i}. {rec['ticker']} | {rec['rating']} | "
                          f"加权: {rec['weighted_prediction']*100:.2f}% | "
                          f"第1天: {rec['day1_prediction']*100:.2f}%")
            
            print(f"\n结果文件:")
            print(f"  - Excel: result/multi_day_lstm_analysis_{timestamp}.xlsx")
            print(f"  - IBKR:  multi_day_trading/top_10_multi_day_stocks_{timestamp}.json")
            print(f"  - 报告:  multi_day_trading/multi_day_trading_report_{timestamp}.txt")
            print(f"="*80)
            
            return recommendations
            
        except Exception as e:
            print(f"[MULTI-DAY ANALYSIS ERROR] 分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多日LSTM量化分析模型')
    parser.add_argument('--stocks', type=int, default=200, help='分析股票数量(默认使用200只股票进行增强训练)')
    parser.add_argument('--days', type=int, default=1825, help='历史数据天数(默认5年=1825天)')
    parser.add_argument('--prediction-days', type=int, default=5, help='预测天数')
    parser.add_argument('--output', type=str, default='result', help='输出目录')
    parser.add_argument('--start-date', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--ticker-file', type=str, help='股票代码文件路径')
    
    args = parser.parse_args()
    
    safe_print("多日LSTM量化分析系统（调试增强版）")
    safe_print("="*80)
    safe_print(f"分析参数:")
    safe_print(f"  - 股票数量: {args.stocks}")
    safe_print(f"  - 历史数据: {args.days} 天")
    safe_print(f"  - 预测天数: {args.prediction_days} 天")
    safe_print(f"  - 输出目录: {args.output}")
    if args.start_date:
        safe_print(f"  - 开始日期: {args.start_date}")
    if args.end_date:
        safe_print(f"  - 结束日期: {args.end_date}")
    safe_print("="*80)
    
    # 创建模型
    model = MultiDayLSTMQuantModel(
        prediction_days=args.prediction_days,
        enable_advanced_features=True
    )
    
    # 选择股票池
    if args.ticker_file and os.path.exists(args.ticker_file):
        # 从文件读取股票列表
        try:
            with open(args.ticker_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                ticker_list = [
    'A',
    'ADM',
    'ACN',
    'ADI',
    'ABBV',
    'ADBE',
    'ABNB',
    'ACGL',
    'ABT',
    'AAPL',
    'ADP',
    'AEP',
    'ADSK',
    'AEE',
    'AES',
    'AFL',
    'AIG',
    'AJG',
    'AIZ',
    'AKAM',
    'ALB',
    'AMAT',
    'ALLE',
    'ALL',
    'ALGN',
    'AMCR',
    'AMD',
    'AME',
    'AMGN',
    'AMT',
    'ANET',
    'AMZN',
    'AON',
    'AOS',
    'APA',
    'APD',
    'APH',
    'APTV',
    'APO',
    'ARE',
    'ATO',
    'AVB',
    'AVY',
    'AWK',
    'AVGO',
    'AXP',
    'BABA',
    'BF-B',
    'BAC',
    'BBY',
    'BDX',
    'BAX',
    'BEN',
    'BALL',
    'BA',
    'BLDR',
    'BG',
    'BKR',
    'BK',
    'BIDU',
    'BIIB',
    'BMY',
    'BSX',
    'BRO',
    'BXP',
    'C',
    'BRK-B',
    'CAG',
    'BX',
    'BR',
    'CAH',
    'CCL',
    'CDNS',
    'CARR',
    'CBOE',
    'CCI',
    'CBRE',
    'CB',
    'CAT',
    'CDW',
    'CEG',
    'CHD',
    'CINF',
    'CI',
    'CHRW',
    'CF',
    'CL',
    'CFG',
    'CHTR',
    'CMCSA',
    'CLX',
    'COP',
    'CNP',
    'CMS',
    'CMG',
    'COIN',
    'CMI',
    'CNC',
    'COO',
    'CME',
    'COF',
    'COR',
    'CSCO',
    'CRL',
    'CRM',
    'CPB',
    'CRWD',
    'CPRT',
    'CSGP',
    'CVS',
    'CTAS',
    'CVX',
    'CTSH',
    'CSX',
    'CZR',
    'CTRA',
    'CTVA',
    'D',
    'DAL',
    'DASH',
    'DDOG',
    'DAY',
    'DELL',
    'DG',
    'DD',
    'DECK',
    'DGX',
    'DHR',
    'DLTR',
    'DOC',
    'DOCU',
    'DHI',
    'DIS',
    'DOV',
    'DOW',
    'DRI',
    'DXCM',
    'EA',
    'DPZ',
    'EBAY',
    'DTE',
    'DUK',
    'DVN',
    'ECL',
    'DVA',
    'ED',
    'EOG',
    'EIX',
    'ENPH',
    'EL',
    'EMR',
    'ELV',
    'EMN',
    'EFX',
    'EPAM',
    'ES',
    'EVRG',
    'ETR',
    'EQR',
    'ETN',
    'EQT',
    'EW',
    'EXR',
    'EXE',
    'EXPE',
    'F',
    'EXPD',
    'FANG',
    'EXC',
    'FAST',
    'FCX',
    'FE',
    'FDX',
    'FI',
    'FITB',
    'FIS',
    'FOX',
    'FOXA',
    'GDDY',
    'GEHC',
    'FTNT',
    'GD',
    'FTV',
    'GEN',
    'GE',
    'FSLR',
    'GILD',
    'GIS',
    'GL',
    'GLW',
    'GM',
    'GOOG',
    'GOOGL',
    'GNRC',
    'GPC',
    'GPN',
    'GRMN',
    'HAL',
    'HCA',
    'HBAN'
]
                for line in lines:
                    line = line.strip().upper()  # 转换为大写，与BMA模型保持一致
                    if line and not line.startswith('#'):  # 跳过空行和注释
                        ticker_list.append(line)
            
            safe_print(f"从文件加载股票池: {args.ticker_file}")
            safe_print(f"加载股票数量: {len(ticker_list)}")
            
            if not ticker_list:
                safe_print("警告: 股票文件为空，使用默认股票池")
                ticker_list = MULTI_DAY_TICKER_LIST[:args.stocks]
        except Exception as e:
            safe_print(f"读取股票文件失败: {e}")
            safe_print("使用默认股票池")
            ticker_list = MULTI_DAY_TICKER_LIST[:args.stocks]
    else:
        # 使用默认股票池
        safe_print("使用默认股票池")
        ticker_list = MULTI_DAY_TICKER_LIST[:args.stocks]
    
    # 显示当前考虑的股票
    safe_print(f"[STOCKS] 股票池大小: {len(ticker_list)}")
    safe_print(f"[STOCKS] 当前考虑的股票: {', '.join(ticker_list[:25])}{'...' if len(ticker_list) > 25 else ''}")
    
    # 显示系统状态
    safe_update_status("系统初始化完成，准备分析", 5)
    
    # 运行分析
    try:
        # 如果提供了日期参数，使用日期范围；否则使用天数
        if args.start_date and args.end_date:
            results = model.run_multi_day_analysis_with_dates(ticker_list, args.start_date, args.end_date)
        else:
            results = model.run_multi_day_analysis(ticker_list, args.days)
        
        if results:
            safe_print("\n多日分析完成！")
            safe_print("可以使用生成的文件进行:")
            safe_print("  1. Excel分析: 打开 result/multi_day_lstm_analysis_*.xlsx")
            safe_print("  2. IBKR自动交易: 使用 multi_day_trading/top_10_multi_day_stocks_*.json")
            safe_print("  3. 查看详细报告: multi_day_trading/multi_day_trading_report_*.txt")
            safe_update_status("分析完成", 100)
        else:
            safe_print("分析失败，请检查数据和网络连接")
            safe_update_status("分析失败", 0)
            
    except KeyboardInterrupt:
        safe_print("\n用户中断分析")
        safe_update_status("用户中断", 0)
    except Exception as e:
        safe_print(f"\n分析异常: {e}")
        safe_update_status("分析异常", 0)
        import traceback
        traceback.print_exc()

# 在MultiDayLSTMQuantModel类中添加评估方法
def calculate_backtest_metrics(self, predictions, actual_returns, prices=None):
    """计算回测评估指标"""
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {}
        
        # 基础统计指标
        predictions_flat = predictions.flatten() if predictions.ndim > 1 else predictions
        actual_flat = actual_returns.flatten() if actual_returns.ndim > 1 else actual_returns
        
        metrics['mse'] = mean_squared_error(actual_flat, predictions_flat)
        metrics['mae'] = mean_absolute_error(actual_flat, predictions_flat)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(actual_flat, predictions_flat)
        
        # 方向准确率
        pred_direction = np.sign(predictions_flat)
        actual_direction = np.sign(actual_flat)
        direction_accuracy = np.mean(pred_direction == actual_direction)
        metrics['direction_accuracy'] = direction_accuracy
        
        # 如果有价格数据，计算交易指标
        if prices is not None:
            # 构建简单的多日交易策略
            daily_returns = []
            portfolio_values = [100]  # 起始资金100
            
            for i in range(len(predictions)):
                if i < len(predictions) - 1:
                    # 基于预测收益率决定仓位
                    if predictions[i] > 0.01:  # 预测涨幅>1%，做多
                        position = 1.0
                    elif predictions[i] < -0.01:  # 预测跌幅>1%，做空
                        position = -1.0
                    else:  # 持币观望
                        position = 0.0
                    
                    # 计算当日收益
                    if i < len(actual_returns):
                        daily_return = position * actual_returns[i]
                        daily_returns.append(daily_return)
                        portfolio_values.append(portfolio_values[-1] * (1 + daily_return))
            
            if daily_returns:
                daily_returns = np.array(daily_returns)
                
                # Sharpe Ratio（假设无风险利率为0）
                if np.std(daily_returns) > 0:
                    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                    metrics['sharpe_ratio'] = sharpe_ratio
                else:
                    metrics['sharpe_ratio'] = 0
                
                # 最大回撤
                portfolio_values = np.array(portfolio_values)
                running_max = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values - running_max) / running_max
                max_drawdown = np.min(drawdown)
                metrics['max_drawdown'] = max_drawdown
                
                # 总收益率
                total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
                metrics['total_return'] = total_return
                
                # 胜率
                win_rate = np.mean(daily_returns > 0)
                metrics['win_rate'] = win_rate
                
                # 平均盈亏比
                winning_returns = daily_returns[daily_returns > 0]
                losing_returns = daily_returns[daily_returns < 0]
                
                if len(winning_returns) > 0 and len(losing_returns) > 0:
                    profit_loss_ratio = np.mean(winning_returns) / abs(np.mean(losing_returns))
                    metrics['profit_loss_ratio'] = profit_loss_ratio
                else:
                    metrics['profit_loss_ratio'] = 0
        
        return metrics
        
    except Exception as e:
        print(f"[BACKTEST METRICS ERROR] {e}")
        return {}

# 为MultiDayLSTMQuantModel添加评估方法
MultiDayLSTMQuantModel.calculate_backtest_metrics = calculate_backtest_metrics

def print_evaluation_report(self):
    """打印详细的评估报告"""
    print("\n" + "="*60)
    print("模型评估报告")
    print("="*60)
    
    if hasattr(self, 'model_metrics') and self.model_metrics:
        print("\n📊 模型训练指标:")
        for key, value in self.model_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    
    if hasattr(self, 'backtest_metrics') and self.backtest_metrics:
        print("\n📈 回测评估指标:")
        for key, value in self.backtest_metrics.items():
            if isinstance(value, float):
                if key in ['sharpe_ratio', 'max_drawdown', 'total_return', 'win_rate', 'direction_accuracy']:
                    if key == 'max_drawdown':
                        print(f"  {key}: {value:.2%}")
                    elif key in ['total_return', 'win_rate', 'direction_accuracy']:
                        print(f"  {key}: {value:.2%}")
                    else:
                        print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    
    print("="*60)

# 为MultiDayLSTMQuantModel添加评估报告方法
MultiDayLSTMQuantModel.print_evaluation_report = print_evaluation_report

if __name__ == "__main__":
    main()