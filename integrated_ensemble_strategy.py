#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构双模型融合策略 - 直接集成算法核心
不再使用子进程调用，而是直接集成BMA和LSTM算法的核心功能
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
from scipy.stats import spearmanr, entropy
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

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

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class BayesianModelAveraging:
    """贝叶斯模型平均器"""
    
    def __init__(self, alpha_prior=1.5, shrinkage_factor=0.15, min_weight_threshold=0.02):
        self.alpha_prior = alpha_prior
        self.shrinkage_factor = shrinkage_factor
        self.min_weight_threshold = min_weight_threshold
        self.models = {}
        self.posterior_weights = {}
        self.model_likelihoods = {}
        
    def fit(self, X, y, models_dict):
        """训练BMA ensemble"""
        self.models = models_dict
        n_models = len(models_dict)
        
        # 时序交叉验证计算似然
        tscv = TimeSeriesSplit(n_splits=min(5, len(X)//10))
        model_scores = {}
        
        for name, model in models_dict.items():
            fold_likelihoods = []
            fold_r2_scores = []
            
            try:
                for train_idx, val_idx in tscv.split(X):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    model.fit(X_train_fold, y_train_fold)
                    y_pred_fold = model.predict(X_val_fold)
                    
                    mse = mean_squared_error(y_val_fold, y_pred_fold)
                    r2 = r2_score(y_val_fold, y_pred_fold)
                    
                    # 对数似然（假设高斯噪声）
                    if mse > 1e-10:
                        likelihood = -0.5 * len(y_val_fold) * np.log(2 * np.pi * mse) - 0.5 * len(y_val_fold)
                    else:
                        likelihood = 1000
                    
                    fold_likelihoods.append(likelihood)
                    fold_r2_scores.append(r2)
                
                avg_likelihood = np.mean(fold_likelihoods)
                avg_r2 = np.mean(fold_r2_scores)
                
                model_scores[name] = {
                    'likelihood': avg_likelihood,
                    'r2': avg_r2,
                    'std_r2': np.std(fold_r2_scores)
                }
                
            except Exception as e:
                logger.warning(f"[BMA] {name} CV失败: {e}")
                model_scores[name] = {
                    'likelihood': -np.inf,
                    'r2': -np.inf,
                    'std_r2': np.inf
                }
        
        # 保存似然分数并计算后验权重
        self.model_likelihoods = model_scores
        self._calculate_posterior_weights()
        
        # 在全部数据上重新训练所有模型
        for name, model in self.models.items():
            try:
                model.fit(X, y)
            except Exception as e:
                logger.warning(f"[BMA] {name} 最终训练失败: {e}")
        
        return self
    
    def _calculate_posterior_weights(self):
        """计算后验权重"""
        model_names = list(self.models.keys())
        n_models = len(model_names)
        
        # 提取似然值和R²
        likelihoods = np.array([self.model_likelihoods[name]['likelihood'] for name in model_names])
        r2_scores = np.array([self.model_likelihoods[name]['r2'] for name in model_names])
        
        # 处理无效值
        finite_mask = np.isfinite(likelihoods) & np.isfinite(r2_scores)
        if not finite_mask.any():
            weights = np.ones(n_models) / n_models
        else:
            # 过滤无效模型
            valid_likelihoods = likelihoods[finite_mask]
            valid_r2 = r2_scores[finite_mask]
            
            # 综合评分：似然 + R²表现
            combined_scores = 0.7 * valid_likelihoods + 0.3 * valid_r2 * 1000
            
            # 数值稳定的softmax
            exp_scores = np.exp(combined_scores - combined_scores.max())
            raw_weights_valid = exp_scores / exp_scores.sum()
            
            # 恢复到完整权重数组
            raw_weights = np.zeros(n_models)
            raw_weights[finite_mask] = raw_weights_valid
            
            # 收缩处理
            simplicity_bias = np.ones(n_models) / n_models
            weights = ((1 - self.shrinkage_factor) * raw_weights + 
                      self.shrinkage_factor * simplicity_bias)
        
        # 最小权重阈值
        weights = np.maximum(weights, self.min_weight_threshold)
        weights = weights / weights.sum()
        
        self.posterior_weights = dict(zip(model_names, weights))
        return self.posterior_weights
    
    def predict(self, X):
        """BMA预测"""
        if not self.posterior_weights:
            raise ValueError("模型未训练")
        
        predictions = {}
        valid_weights = {}
        
        # 获取每个模型的预测
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
                valid_weights[name] = self.posterior_weights[name]
            except Exception as e:
                logger.warning(f"[BMA PREDICT] {name}预测失败: {e}")
        
        if not predictions:
            return np.zeros(len(X))
        
        # 重新归一化权重
        total_weight = sum(valid_weights.values())
        if total_weight > 0:
            for name in valid_weights:
                valid_weights[name] = valid_weights[name] / total_weight
        
        # 加权平均
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            weight = valid_weights[name]
            ensemble_pred += weight * pred
        
        return ensemble_pred


class AdvancedLSTMStrategy:
    """简化的LSTM策略 - 使用传统ML代替深度学习"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def calculate_lstm_features(self, data):
        """计算LSTM风格的技术特征"""
        features = {}
        
        # 价格序列特征
        features['price_sma_5'] = data['Close'].rolling(5).mean()
        features['price_sma_10'] = data['Close'].rolling(10).mean()
        features['price_sma_20'] = data['Close'].rolling(20).mean()
        
        # 价格动量特征  
        features['returns_1d'] = data['Close'].pct_change()
        features['returns_5d'] = data['Close'].pct_change(5)
        features['returns_10d'] = data['Close'].pct_change(10)
        
        # 波动率特征
        features['volatility_5d'] = features['returns_1d'].rolling(5).std()
        features['volatility_10d'] = features['returns_1d'].rolling(10).std()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)  # 避免除零
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        
        # 成交量特征
        features['volume_sma'] = data['Volume'].rolling(10).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_sma']
        
        # 确保所有特征都是Series
        clean_features = {}
        for key, value in features.items():
            if isinstance(value, pd.Series):
                clean_features[key] = value
            else:
                # 如果是DataFrame或其他类型，尝试转换为Series
                try:
                    clean_features[key] = pd.Series(value, index=data.index)
                except:
                    logger.warning(f"[LSTM] 跳过特征 {key}：无法转换为Series")
        
        return pd.DataFrame(clean_features, index=data.index)
    
    def train_lstm_model(self, all_data, target_period=5):
        """训练LSTM风格模型"""
        logger.info("[LSTM] 开始训练LSTM风格模型...")
        
        # 准备训练数据
        all_features = []
        all_targets = []
        
        for ticker, data in all_data.items():
            try:
                # 计算特征
                features = self.calculate_lstm_features(data)
                
                # 计算目标变量（未来收益率）
                target = data['Close'].pct_change(target_period).shift(-target_period)
                
                # 对齐数据
                aligned_data = pd.concat([features, target.rename('target')], axis=1)
                aligned_data = aligned_data.dropna()
                
                if len(aligned_data) > 10:  # 确保有足够数据
                    features_df = aligned_data.drop('target', axis=1)
                    target_series = aligned_data['target']
                    
                    all_features.append(features_df)
                    all_targets.append(target_series)
                    
            except Exception as e:
                logger.warning(f"[LSTM] {ticker}特征计算失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_features:
            logger.warning("[LSTM] 没有有效特征数据")
            return False
        
        # 合并数据
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        # 确保y是一维数组
        if hasattr(y, 'values'):
            y = y.values
        y = np.array(y).flatten()
        
        # 保存特征列名
        self.feature_columns = X.columns.tolist()
        
        # 数据预处理
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        
        # 创建基学习器（代替LSTM）
        base_models = {}
        base_models['RandomForest'] = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        base_models['Ridge'] = Ridge(alpha=1.0)
        
        if XGBOOST_AVAILABLE:
            base_models['XGBoost'] = xgb.XGBRegressor(n_estimators=50, random_state=42)
        
        # 使用BMA训练
        self.model = BayesianModelAveraging()
        self.model.fit(X_scaled, y.values, base_models)
        
        logger.info(f"[LSTM] 模型训练完成，特征数: {len(self.feature_columns)}")
        return True
    
    def predict_lstm_signals(self, data):
        """预测LSTM信号"""
        if self.model is None:
            return 0.0
        
        try:
            # 计算特征
            features = self.calculate_lstm_features(data)
            
            # 获取最新特征
            if len(features) == 0:
                return 0.0
            
            latest_features = features.iloc[-1:].fillna(0)
            
            # 确保特征列一致
            for col in self.feature_columns:
                if col not in latest_features.columns:
                    latest_features[col] = 0
            
            latest_features = latest_features.reindex(columns=self.feature_columns)
            
            # 标准化并预测
            X_scaled = self.scaler.transform(latest_features)
            prediction = self.model.predict(X_scaled)[0]
            
            return prediction
            
        except Exception as e:
            logger.warning(f"[LSTM] 预测失败: {e}")
            return 0.0


class AdvancedBMAStrategy:
    """高级BMA策略"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def calculate_bma_features(self, data):
        """计算BMA风格的量化特征"""
        features = {}
        
        # 基础技术指标
        features['sma_5'] = data['Close'].rolling(5).mean()
        features['sma_10'] = data['Close'].rolling(10).mean()
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['sma_50'] = data['Close'].rolling(50).mean()
        
        # 价格动量
        features['momentum_5'] = data['Close'].pct_change(5)
        features['momentum_15'] = data['Close'].pct_change(15)
        features['price_acceleration'] = features['momentum_5'] - features['momentum_15']
        
        # 成交量因子
        features['volume_sma'] = data['Volume'].rolling(15).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_sma']
        features['volume_momentum'] = data['Volume'].pct_change(10)
        
        # 波动率因子
        features['volatility'] = data['Close'].pct_change().rolling(15).std()
        
        # 添加sma_15
        features['sma_15'] = data['Close'].rolling(15).mean()
        
        # 相对强弱
        features['rs_vs_sma'] = data['Close'] / features['sma_15'] - 1
        features['high_low_ratio'] = data['High'] / data['Low'] - 1
        features['close_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        # 市场情绪
        features['gap_ratio'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        features['intraday_return'] = (data['Close'] - data['Open']) / data['Open']
        
        # 高级因子
        features['rolling_sharpe'] = (data['Close'].pct_change().rolling(15).mean() / 
                                    data['Close'].pct_change().rolling(15).std())
        
        # CCI
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(15).mean()
        mad_tp = typical_price.rolling(15).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        features['cci'] = (typical_price - sma_tp) / (0.015 * mad_tp + 1e-8)
        features['cci_normalized'] = features['cci'] / 100.0
        
        # 确保所有特征都是Series
        clean_features = {}
        for key, value in features.items():
            if isinstance(value, pd.Series):
                clean_features[key] = value
            else:
                # 如果是DataFrame或其他类型，尝试转换为Series
                try:
                    clean_features[key] = pd.Series(value, index=data.index)
                except:
                    logger.warning(f"[BMA] 跳过特征 {key}：无法转换为Series")
        
        return pd.DataFrame(clean_features, index=data.index)
    
    def train_bma_model(self, all_data, target_period=14):
        """训练BMA模型"""
        logger.info("[BMA] 开始训练BMA量化模型...")
        
        # 准备训练数据
        all_features = []
        all_targets = []
        
        for ticker, data in all_data.items():
            try:
                # 计算特征
                features = self.calculate_bma_features(data)
                
                # 计算目标变量
                target = data['Close'].pct_change(target_period).shift(-target_period)
                
                # 对齐数据
                aligned_data = pd.concat([features, target.rename('target')], axis=1)
                aligned_data = aligned_data.replace([np.inf, -np.inf], np.nan)
                aligned_data = aligned_data.dropna()
                
                if len(aligned_data) > 20:
                    features_df = aligned_data.drop('target', axis=1)
                    target_series = aligned_data['target']
                    
                    all_features.append(features_df)
                    all_targets.append(target_series)
                    
            except Exception as e:
                logger.warning(f"[BMA] {ticker}特征计算失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_features:
            logger.warning("[BMA] 没有有效特征数据")
            return False
        
        # 合并数据
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        # 确保y是一维数组
        if hasattr(y, 'values'):
            y = y.values
        y = np.array(y).flatten()
        
        # 保存特征列名
        self.feature_columns = X.columns.tolist()
        
        # 数据预处理
        X_filled = X.fillna(X.median())
        X_scaled = self.scaler.fit_transform(X_filled)
        
        # 创建基学习器
        base_models = {}
        base_models['RandomForest'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        base_models['Ridge'] = Ridge(alpha=1.0)
        base_models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        if XGBOOST_AVAILABLE:
            base_models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        if LIGHTGBM_AVAILABLE:
            base_models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        if CATBOOST_AVAILABLE:
            base_models['CatBoost'] = CatBoostRegressor(random_state=42, verbose=False)
        
        # 使用BMA训练
        self.model = BayesianModelAveraging()
        self.model.fit(X_scaled, y.values, base_models)
        
        logger.info(f"[BMA] 模型训练完成，特征数: {len(self.feature_columns)}")
        return True
    
    def predict_bma_signals(self, data):
        """预测BMA信号"""
        if self.model is None:
            return 0.0
        
        try:
            # 计算特征
            features = self.calculate_bma_features(data)
            
            if len(features) == 0:
                return 0.0
            
            # 获取最新特征
            latest_features = features.iloc[-1:].replace([np.inf, -np.inf], np.nan)
            
            # 确保特征列一致
            for col in self.feature_columns:
                if col not in latest_features.columns:
                    latest_features[col] = 0
            
            latest_features = latest_features.reindex(columns=self.feature_columns)
            latest_features = latest_features.fillna(0)
            
            # 标准化并预测
            X_scaled = self.scaler.transform(latest_features)
            prediction = self.model.predict(X_scaled)[0]
            
            return prediction
            
        except Exception as e:
            logger.warning(f"[BMA] 预测失败: {e}")
            return 0.0


class IntegratedEnsembleStrategy:
    """集成双模型融合策略"""
    
    def __init__(self, weights_file="integrated_weights.json", lookback_weeks=12):
        self.weights_file = weights_file
        self.lookback_weeks = lookback_weeks
        self.logger = logger
        
        # 初始化策略
        self.bma_strategy = AdvancedBMAStrategy()
        self.lstm_strategy = AdvancedLSTMStrategy()
        
        # 当前权重
        self.current_weights = {"w_bma": 0.5, "w_lstm": 0.5, "date": None}
        self._load_weights()
    
    def _load_weights(self):
        """加载权重"""
        try:
            if Path(self.weights_file).exists():
                with open(self.weights_file, 'r', encoding='utf-8') as f:
                    self.current_weights = json.load(f)
                self.logger.info(f"[集成策略] 加载权重: BMA={self.current_weights['w_bma']:.3f}, LSTM={self.current_weights['w_lstm']:.3f}")
            else:
                self.logger.info("[集成策略] 权重文件不存在，使用默认权重")
        except Exception as e:
            self.logger.warning(f"[集成策略] 加载权重失败: {e}")
    
    def download_stock_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict:
        """下载股票数据"""
        self.logger.info(f"[集成策略] 下载股票数据: {len(tickers)}只股票")
        
        all_data = {}
        success_count = 0
        
        for ticker in tickers:
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if len(stock_data) > 50:  # 确保有足够数据
                    all_data[ticker] = stock_data
                    success_count += 1
            except Exception as e:
                self.logger.warning(f"[集成策略] {ticker}数据下载失败: {e}")
                continue
        
        self.logger.info(f"[集成策略] 成功下载 {success_count}/{len(tickers)} 只股票数据")
        return all_data
    
    def train_integrated_models(self, all_data: Dict):
        """训练集成模型"""
        self.logger.info("[集成策略] 开始训练集成模型...")
        
        # 训练BMA模型
        bma_success = self.bma_strategy.train_bma_model(all_data)
        
        # 训练LSTM模型  
        lstm_success = self.lstm_strategy.train_lstm_model(all_data)
        
        if not bma_success and not lstm_success:
            self.logger.error("[集成策略] 所有模型训练失败")
            return False
        elif not bma_success:
            self.logger.warning("[集成策略] BMA模型训练失败，仅使用LSTM")
            self.current_weights = {"w_bma": 0.0, "w_lstm": 1.0, "date": datetime.now().strftime('%Y-%m-%d')}
        elif not lstm_success:
            self.logger.warning("[集成策略] LSTM模型训练失败，仅使用BMA")
            self.current_weights = {"w_bma": 1.0, "w_lstm": 0.0, "date": datetime.now().strftime('%Y-%m-%d')}
        else:
            self.logger.info("[集成策略] 两个模型训练成功")
        
        return True
    
    def update_weights_by_performance(self, tickers: List[str], force_update: bool = False) -> Tuple[float, float]:
        """基于回测性能更新权重"""
        try:
            today = pd.Timestamp.now().normalize()
            
            # 检查是否需要更新
            if not force_update:
                last_update = self.current_weights.get("date")
                if last_update:
                    last_date = pd.Timestamp(last_update)
                    days_since_update = (today - last_date).days
                    if today.weekday() != 0 and days_since_update < 5:
                        return self.current_weights["w_bma"], self.current_weights["w_lstm"]
            
            self.logger.info(f"[集成策略] 基于回测性能更新权重，回望{self.lookback_weeks}周")
            
            # 计算回望期
            start_date = today - pd.Timedelta(weeks=self.lookback_weeks)
            end_date = today
            
            # 下载回测数据
            backtest_data = self.download_stock_data(tickers, 
                                                   start_date.strftime('%Y-%m-%d'), 
                                                   end_date.strftime('%Y-%m-%d'))
            
            if not backtest_data:
                self.logger.warning("[集成策略] 无回测数据，保持现有权重")
                return self.current_weights["w_bma"], self.current_weights["w_lstm"]
            
            # 训练模型用于回测
            temp_bma = AdvancedBMAStrategy()
            temp_lstm = AdvancedLSTMStrategy()
            
            bma_trained = temp_bma.train_bma_model(backtest_data)
            lstm_trained = temp_lstm.train_lstm_model(backtest_data)
            
            if not bma_trained and not lstm_trained:
                return self.current_weights["w_bma"], self.current_weights["w_lstm"]
            
            # 计算策略收益
            bma_returns = []
            lstm_returns = []
            
            for ticker, data in backtest_data.items():
                try:
                    if bma_trained:
                        bma_signal = temp_bma.predict_bma_signals(data)
                        # 简化：用信号符号作为买卖决策
                        bma_return = bma_signal * data['Close'].pct_change().iloc[-1]
                        bma_returns.append(bma_return)
                    
                    if lstm_trained:
                        lstm_signal = temp_lstm.predict_lstm_signals(data)
                        lstm_return = lstm_signal * data['Close'].pct_change().iloc[-1]
                        lstm_returns.append(lstm_return)
                        
                except Exception:
                    continue
            
            # 计算Sharpe比率
            sharpe_bma = 0.0
            sharpe_lstm = 0.0
            
            if bma_returns and np.std(bma_returns) > 0:
                sharpe_bma = np.mean(bma_returns) / np.std(bma_returns) * np.sqrt(252)
            
            if lstm_returns and np.std(lstm_returns) > 0:
                sharpe_lstm = np.mean(lstm_returns) / np.std(lstm_returns) * np.sqrt(252)
            
            self.logger.info(f"[集成策略] 回测Sharpe - BMA: {sharpe_bma:.4f}, LSTM: {sharpe_lstm:.4f}")
            
            # 权重分配
            if sharpe_bma <= 0 and sharpe_lstm <= 0:
                w_bma, w_lstm = 0.5, 0.5
            elif sharpe_bma <= 0:
                w_bma, w_lstm = 0.0, 1.0
            elif sharpe_lstm <= 0:
                w_bma, w_lstm = 1.0, 0.0
            else:
                total_sharpe = sharpe_bma + sharpe_lstm
                w_bma = sharpe_bma / total_sharpe
                w_lstm = sharpe_lstm / total_sharpe
            
            # 保存权重
            self.current_weights = {
                "date": today.strftime('%Y-%m-%d'),
                "w_bma": float(w_bma),
                "w_lstm": float(w_lstm),
                "sharpe_bma": float(sharpe_bma),
                "sharpe_lstm": float(sharpe_lstm),
                "lookback_weeks": self.lookback_weeks
            }
            
            with open(self.weights_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_weights, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"[集成策略] 权重更新完成: BMA={w_bma:.3f}, LSTM={w_lstm:.3f}")
            return w_bma, w_lstm
            
        except Exception as e:
            self.logger.error(f"[集成策略] 权重更新失败: {e}")
            return 0.5, 0.5
    
    def generate_ensemble_signals(self, tickers: List[str]) -> Dict[str, float]:
        """生成融合信号"""
        try:
            self.logger.info(f"[集成策略] 生成融合信号，股票数量: {len(tickers)}")
            
            # 下载最新数据
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            current_data = self.download_stock_data(tickers, start_date, end_date)
            
            if not current_data:
                self.logger.warning("[集成策略] 无当前数据")
                return {}
            
            # 获取权重
            w_bma = self.current_weights["w_bma"]
            w_lstm = self.current_weights["w_lstm"]
            
            # 生成信号
            ensemble_signals = {}
            bma_signals = {}
            lstm_signals = {}
            
            for ticker, data in current_data.items():
                try:
                    # BMA信号
                    if self.bma_strategy.model and w_bma > 0:
                        bma_signal = self.bma_strategy.predict_bma_signals(data)
                        bma_signals[ticker] = bma_signal
                    else:
                        bma_signals[ticker] = 0.0
                    
                    # LSTM信号
                    if self.lstm_strategy.model and w_lstm > 0:
                        lstm_signal = self.lstm_strategy.predict_lstm_signals(data)
                        lstm_signals[ticker] = lstm_signal
                    else:
                        lstm_signals[ticker] = 0.0
                    
                except Exception as e:
                    self.logger.warning(f"[集成策略] {ticker}信号生成失败: {e}")
                    bma_signals[ticker] = 0.0
                    lstm_signals[ticker] = 0.0
            
            # 标准化信号
            def normalize_signals(signals):
                values = list(signals.values())
                if not values or all(v == 0 for v in values):
                    return {k: 0.5 for k in signals.keys()}
                
                min_val, max_val = min(values), max(values)
                if max_val == min_val:
                    return {k: 0.5 for k in signals.keys()}
                
                return {k: (v - min_val) / (max_val - min_val) for k, v in signals.items()}
            
            norm_bma = normalize_signals(bma_signals)
            norm_lstm = normalize_signals(lstm_signals)
            
            # 融合信号
            for ticker in current_data.keys():
                bma_norm = norm_bma.get(ticker, 0.5)
                lstm_norm = norm_lstm.get(ticker, 0.5)
                ensemble_signal = w_bma * bma_norm + w_lstm * lstm_norm
                ensemble_signals[ticker] = ensemble_signal
            
            self.logger.info(f"[集成策略] 生成 {len(ensemble_signals)} 个融合信号")
            return ensemble_signals
            
        except Exception as e:
            self.logger.error(f"[集成策略] 信号生成失败: {e}")
            return {}
    
    def get_default_tickers(self) -> List[str]:
        """获取默认股票池"""
        try:
            pool_file = "default_stock_pool.json"
            if Path(pool_file).exists():
                with open(pool_file, 'r', encoding='utf-8') as f:
                    pool_data = json.load(f)
                    return pool_data.get('default_stock_pool', 
                        ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'])
            else:
                return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
        except:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']


def test_integrated_strategy():
    """测试集成策略"""
    logger.info("=== 测试集成双模型融合策略 ===")
    
    strategy = IntegratedEnsembleStrategy()
    
    # 测试参数
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2024-01-01'
    end_date = '2025-01-01'
    
    try:
        # 1. 下载数据
        logger.info("1. 下载数据...")
        all_data = strategy.download_stock_data(tickers, start_date, end_date)
        
        if not all_data:
            logger.error("无法下载数据")
            return
        
        # 2. 训练模型
        logger.info("2. 训练集成模型...")
        success = strategy.train_integrated_models(all_data)
        
        if not success:
            logger.error("模型训练失败")
            return
        
        # 3. 更新权重
        logger.info("3. 更新权重...")
        w_bma, w_lstm = strategy.update_weights_by_performance(tickers, force_update=True)
        print(f"权重更新结果: BMA={w_bma:.3f}, LSTM={w_lstm:.3f}")
        
        # 4. 生成信号
        logger.info("4. 生成融合信号...")
        signals = strategy.generate_ensemble_signals(tickers)
        print(f"融合信号: {signals}")
        
        logger.info("=== 测试完成 ===")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_integrated_strategy()