#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA增强版滚动前向回测系统 V2
解决训练-预测周期错配、改进评分机制、增强仓位管理和可视化
"""

import pandas as pd
import numpy as np
from polygon_client import polygon_client, download, Ticker
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy.stats import spearmanr, norm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression

# 高级模型导入
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

# CatBoost removed due to compatibility issues
CATBOOST_AVAILABLE = False

# 设置日志和警告
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 扩展的股票池 - 来自原始BMA训练集
ENHANCED_STOCK_POOL = [
    # 核心大盘股
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
    'ADBE', 'CRM', 'AVGO', 'ORCL', 'AMD', 'INTC', 'QCOM', 'AMAT',
    
    # 金融股
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'COF', 'BRK-B', 'V', 'MA', 'PYPL',
    
    # 消费股
    'JNJ', 'PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST',
    
    # 工业股  
    'BA', 'CAT', 'GE', 'MMM', 'UNP', 'RTX', 'HON', 'LMT', 'UPS', 'FDX', 'CSX',
    
    # 医疗保健
    'UNH', 'ABBV', 'TMO', 'ABT', 'AMGN', 'GILD', 'CVS', 'CI', 'ISRG', 'DHR',
    
    # 能源/公用事业
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'KMI', 'NEE', 'DUK', 'SO', 'D',
    
    # 房地产/REITs
    'AMT', 'CCI', 'EQIX', 'PLD', 'EXR', 'AVB', 'SPG', 'O', 'VTR', 'WELL'
]

class EnhancedBMAWalkForward:
    """增强版BMA滚动前向回测器"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 max_positions: int = 15,
                 rebalance_freq: str = 'W',
                 training_window_months: int = 3,
                 min_training_samples: int = 60,
                 prediction_horizon: int = 7,  # 与再平衡频率对齐
                 volatility_lookback: int = 20,
                 risk_target: float = 0.15,  # 目标年化波动率
                 min_signal_threshold: float = 0.3,
                 max_signal_threshold: float = 0.7):
        """
        初始化增强版回测器
        
        Args:
            prediction_horizon: 预测周期(天)，与再平衡频率对齐
            volatility_lookback: ATR计算回望期
            risk_target: 目标年化风险水平
            min_signal_threshold: 最小信号阈值（动态调整）
            max_signal_threshold: 最大信号阈值（动态调整）
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_positions = max_positions
        self.rebalance_freq = rebalance_freq
        self.training_window_months = training_window_months
        self.min_training_samples = min_training_samples
        self.prediction_horizon = prediction_horizon
        self.volatility_lookback = volatility_lookback
        self.risk_target = risk_target
        self.min_signal_threshold = min_signal_threshold
        self.max_signal_threshold = max_signal_threshold
        
        # 结果存储
        self.portfolio_returns = []
        self.portfolio_values = []
        self.positions_history = []
        self.trading_signals = []
        self.performance_metrics = {}
        self.model_performance_history = []
        self.risk_metrics_history = []
        
        self.logger = logger
        
    def calculate_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算增强版技术特征，包含更多因子
        """
        features = pd.DataFrame(index=data.index)
        
        try:
            # 基础价格特征
            features['sma_5'] = data['Close'].rolling(window=5).mean()
            features['sma_10'] = data['Close'].rolling(window=10).mean()
            features['sma_20'] = data['Close'].rolling(window=20).mean()
            features['sma_50'] = data['Close'].rolling(window=50).mean()
            
            # 动量特征 - 与预测周期对齐
            features[f'momentum_{self.prediction_horizon}'] = data['Close'].pct_change(self.prediction_horizon)
            features['momentum_15'] = data['Close'].pct_change(15)
            features['momentum_30'] = data['Close'].pct_change(30)
            
            # 波动率和ATR
            returns = data['Close'].pct_change()
            features['volatility_20'] = returns.rolling(window=20).std()
            
            # Average True Range (ATR) - 风险调整的关键指标
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = pd.DataFrame([high_low, high_close, low_close]).max()
            features['atr'] = true_range.rolling(window=self.volatility_lookback).mean()
            features['atr_normalized'] = features['atr'] / data['Close']  # ATR标准化
            
            # 相对强弱指标
            features['rs_vs_sma20'] = (data['Close'] / features['sma_20'] - 1).fillna(0)
            features['high_low_ratio'] = (data['High'] / data['Low'] - 1).fillna(0)
            
            # 成交量特征
            features['volume_sma_10'] = data['Volume'].rolling(window=10).mean()
            features['volume_ratio'] = (data['Volume'] / features['volume_sma_10']).fillna(1)
            features['volume_price_trend'] = (data['Volume'] * data['Close'].pct_change()).rolling(window=10).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = data['Close'].ewm(span=12).mean()
            ema26 = data['Close'].ewm(span=26).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            
            # 布林带位置
            bb_middle = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            features['bollinger_position'] = ((data['Close'] - (bb_middle - 2*bb_std)) / (4*bb_std)).fillna(0.5)
            
            # 市场情绪
            features['gap_ratio'] = ((data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)).fillna(0)
            features['intraday_return'] = ((data['Close'] - data['Open']) / data['Open']).fillna(0)
            
            # 滚动Sharpe比率
            features['rolling_sharpe'] = (
                returns.rolling(window=20).mean() / (returns.rolling(window=20).std() + 1e-8)
            ) * np.sqrt(252)
            
            return features.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            self.logger.error(f"增强特征计算失败: {e}")
            return pd.DataFrame(index=data.index)
    
    def create_enhanced_bma_model(self) -> Dict:
        """创建增强版BMA集成模型"""
        base_models = {}
        
        # 基础模型
        base_models['RandomForest'] = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        
        base_models['Ridge'] = Ridge(alpha=1.0)
        base_models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)
        
        # 高级模型
        if XGBOOST_AVAILABLE:
            base_models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                random_state=42, verbosity=0
            )
        
        if LIGHTGBM_AVAILABLE:
            base_models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                random_state=42, verbose=-1
            )
            
        # CatBoost removed due to compatibility issues
        
        return base_models
    
    def train_enhanced_bma_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict, Dict, Dict]:
        """
        训练增强版BMA集成模型，包含复合评分机制
        
        Returns:
            (trained_models, model_weights, performance_metrics)
        """
        base_models = self.create_enhanced_bma_model()
        model_weights = {}
        trained_models = {}
        performance_metrics = {}
        
        # 时序交叉验证
        n_splits = min(5, max(3, len(X) // 60))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # 特征选择
        if len(X.columns) > 20:
            selector = SelectKBest(f_regression, k=min(20, len(X.columns)))
            X_selected = pd.DataFrame(
                selector.fit_transform(X.fillna(X.median()).replace([np.inf, -np.inf], 0), y),
                index=X.index,
                columns=X.columns[selector.get_support()]
            )
        else:
            X_selected = X.fillna(X.median()).replace([np.inf, -np.inf], 0)
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # 模型训练和评估
        for name, model in base_models.items():
            try:
                scores = {'r2': [], 'mse': [], 'mae': [], 'ic': []}  # IC = Information Coefficient
                
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # 训练模型
                    model_clone = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                    model_clone.fit(X_train, y_train)
                    
                    # 预测和评估
                    y_pred = model_clone.predict(X_val)
                    
                    scores['r2'].append(r2_score(y_val, y_pred))
                    scores['mse'].append(mean_squared_error(y_val, y_pred))
                    scores['mae'].append(mean_absolute_error(y_val, y_pred))
                    
                    # Information Coefficient (IC) - 金融领域重要指标
                    ic, _ = spearmanr(y_val, y_pred)
                    scores['ic'].append(ic if not np.isnan(ic) else 0)
                
                # 计算复合评分
                mean_r2 = np.mean(scores['r2'])
                mean_ic = np.mean(scores['ic'])
                mean_mse = np.mean(scores['mse'])
                mean_mae = np.mean(scores['mae'])
                
                # 复合评分：结合R²、IC和稳定性
                stability_penalty = np.std(scores['r2']) + np.std(scores['ic'])
                composite_score = (0.4 * max(0, mean_r2) + 
                                 0.4 * max(0, mean_ic) + 
                                 0.2 * max(0, 1 - stability_penalty))
                
                # 最终训练
                final_model = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                final_model.fit(X_scaled, y)
                
                trained_models[name] = final_model
                model_weights[name] = max(0.01, composite_score)  # 避免负权重
                
                performance_metrics[name] = {
                    'r2': mean_r2,
                    'ic': mean_ic, 
                    'mse': mean_mse,
                    'mae': mean_mae,
                    'composite_score': composite_score,
                    'stability': 1 - stability_penalty
                }
                
                self.logger.info(f"{name}: R²={mean_r2:.3f}, IC={mean_ic:.3f}, 复合评分={composite_score:.3f}")
                
            except Exception as e:
                self.logger.warning(f"{name} 训练失败: {e}")
                continue
        
        # 权重归一化
        total_weight = sum(model_weights.values())
        if total_weight > 0:
            model_weights = {k: v/total_weight for k, v in model_weights.items()}
        else:
            # 如果所有模型都失败，使用均等权重
            model_weights = {k: 1.0/len(trained_models) for k in trained_models.keys()}
        
        return trained_models, model_weights, performance_metrics, scaler, selector if 'selector' in locals() else None
    
    def calculate_position_sizes_with_risk_adjustment(self, 
                                                     signals: Dict[str, float], 
                                                     current_prices: Dict[str, float],
                                                     atr_data: Dict[str, float],
                                                     available_cash: float) -> Dict[str, int]:
        """
        基于ATR的风险调整仓位计算
        """
        if not signals or available_cash <= 0:
            return {}
        
        position_sizes = {}
        
        # 过滤有效信号
        valid_signals = {k: v for k, v in signals.items() 
                        if k in current_prices and k in atr_data and abs(v) > self.min_signal_threshold}
        
        if not valid_signals:
            return {}
        
        # 计算风险调整权重
        risk_budgets = {}
        total_risk_budget = 0
        
        for ticker, signal in valid_signals.items():
            price = current_prices[ticker]
            atr = atr_data[ticker]
            
            # 计算个股风险预算（基于ATR的风险调整）
            if atr > 0 and price > 0:
                # 目标波动率 / 当前ATR = 风险调整因子
                risk_adjustment = self.risk_target / (atr / price * np.sqrt(252))
                risk_budget = abs(signal) * risk_adjustment
                risk_budgets[ticker] = risk_budget
                total_risk_budget += risk_budget
        
        # 分配资金
        if total_risk_budget > 0:
            for ticker in risk_budgets:
                weight = risk_budgets[ticker] / total_risk_budget
                target_value = available_cash * weight * 0.95  # 保留5%现金
                
                price = current_prices[ticker]
                shares = int(target_value // price) if price > 0 else 0
                
                if shares > 0:
                    position_sizes[ticker] = shares
        
        return position_sizes
    
    def calculate_dynamic_thresholds(self, all_signals: List[float]) -> Tuple[float, float]:
        """
        基于历史信号分布动态计算阈值
        """
        if len(all_signals) < 10:
            return self.min_signal_threshold, self.max_signal_threshold
        
        # 使用分位数动态调整阈值
        signals_array = np.array([abs(s) for s in all_signals])
        
        # 动态阈值：基于25%和75%分位数
        lower_threshold = max(0.1, np.percentile(signals_array, 25))
        upper_threshold = min(0.9, np.percentile(signals_array, 75))
        
        return lower_threshold, upper_threshold
    
    def _download_enhanced_price_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """增强版数据下载，支持更多股票"""
        self.logger.info(f"下载 {len(tickers)} 只股票的价格数据")
        
        price_data = {}
        success_count = 0
        
        for i, ticker in enumerate(tickers):
            try:
                self.logger.info(f"[{i+1:3d}/{len(tickers):3d}] 下载 {ticker:6s}...")
                
                for attempt in range(2):
                    try:
                        data = download(ticker, start=start_date, end=end_date)
                        
                        if data is not None and not data.empty:
                            # 处理多级列名
                            if isinstance(data.columns, pd.MultiIndex):
                                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                            
                            # 验证数据完整性
                            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                            available_columns = [col for col in required_columns if col in data.columns]
                            
                            if len(available_columns) >= 4:
                                clean_data = data[available_columns].dropna()
                                if len(clean_data) >= 30:
                                    price_data[ticker] = clean_data
                                    success_count += 1
                                    break
                        
                        if attempt < 1:
                            time.sleep(0.5)
                            
                    except Exception as download_error:
                        if attempt < 1:
                            time.sleep(1)
                        continue
                        
            except Exception as e:
                self.logger.warning(f"{ticker} 下载失败: {e}")
                continue
        
        self.logger.info(f"成功下载 {success_count}/{len(tickers)} 只股票数据")
        return price_data
    
    def run_enhanced_walkforward_backtest(self, 
                                         tickers: List[str] = None,
                                         start_date: str = "2022-01-01",
                                         end_date: str = None) -> Dict:
        """
        运行增强版滚动前向回测
        """
        if tickers is None:
            tickers = ENHANCED_STOCK_POOL[:50]  # 使用前50只股票
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        self.logger.info(f"开始增强版滚动前向回测: {start_date} 到 {end_date}")
        self.logger.info(f"股票池: {len(tickers)} 只股票")
        self.logger.info(f"预测周期: {self.prediction_horizon} 天")
        
        # 下载数据
        price_data = self._download_enhanced_price_data(tickers, start_date, end_date)
        
        if not price_data:
            raise ValueError("无法获取价格数据")
        
        # 获取共同日期
        common_dates = None
        for ticker, data in price_data.items():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)
        
        common_dates = common_dates.sort_values()
        
        if len(common_dates) < 60:
            raise ValueError(f"可用交易日期太少: {len(common_dates)} < 60")
        
        # 初始化变量
        current_capital = self.initial_capital
        current_cash = self.initial_capital
        current_portfolio = {}
        
        portfolio_values = []
        portfolio_returns = []
        positions_history = []
        trades_history = []
        signal_history = []
        
        # 训练窗口设置
        training_days = self.training_window_months * 21
        
        # 再平衡日期
        rebalance_dates = self._get_rebalance_dates(common_dates, self.rebalance_freq)
        
        self.logger.info(f"训练窗口: {training_days} 天")
        self.logger.info(f"再平衡次数: {len(rebalance_dates)} 次")
        
        # 主回测循环
        for i, rebalance_date in enumerate(rebalance_dates):
            if rebalance_date not in common_dates:
                continue
                
            current_idx = list(common_dates).index(rebalance_date)
            
            if current_idx < training_days:
                continue
                
            try:
                self.logger.info(f"再平衡 {i+1}/{len(rebalance_dates)}: {rebalance_date.strftime('%Y-%m-%d')}")
                
                # 训练数据准备
                train_start_idx = max(0, current_idx - training_days)
                train_end_idx = current_idx
                train_dates = common_dates[train_start_idx:train_end_idx]
                
                # 准备训练数据
                train_features, train_targets = self._prepare_enhanced_training_data(
                    price_data, list(price_data.keys()), train_dates
                )
                
                if len(train_features) < 20:
                    self.logger.warning(f"训练数据不足: {len(train_features)} 条")
                    continue
                
                # 训练模型
                models, weights, metrics, scaler, selector = self.train_enhanced_bma_ensemble(
                    train_features, train_targets
                )
                
                if not models:
                    continue
                
                # 生成预测信号
                current_features = self._prepare_current_enhanced_features(
                    price_data, list(price_data.keys()), rebalance_date, scaler, selector
                )
                
                if current_features.empty:
                    continue
                
                # BMA预测
                predictions = self._generate_bma_predictions(models, weights, current_features)
                
                # 获取当前价格和ATR数据
                current_prices = {}
                atr_data = {}
                
                for ticker in predictions.keys():
                    if ticker in price_data:
                        ticker_data = price_data[ticker]
                        if rebalance_date in ticker_data.index:
                            current_prices[ticker] = ticker_data.loc[rebalance_date, 'Close']
                            
                            # 计算ATR
                            if len(ticker_data.loc[:rebalance_date]) >= self.volatility_lookback:
                                recent_data = ticker_data.loc[:rebalance_date].tail(self.volatility_lookback)
                                high_low = recent_data['High'] - recent_data['Low']
                                high_close = np.abs(recent_data['High'] - recent_data['Close'].shift())
                                low_close = np.abs(recent_data['Low'] - recent_data['Close'].shift())
                                true_range = pd.DataFrame([high_low, high_close, low_close]).max()
                                atr_data[ticker] = true_range.mean()
                
                # 动态阈值调整
                all_predictions = list(predictions.values())
                min_threshold, max_threshold = self.calculate_dynamic_thresholds(
                    signal_history[-252:] if len(signal_history) >= 252 else signal_history
                )
                
                # 信号过滤和映射
                filtered_signals = {}
                for ticker, pred in predictions.items():
                    if abs(pred) > min_threshold:
                        # 信号标准化到[-1, 1]
                        normalized_signal = np.tanh(pred * 2)  # tanh映射
                        if abs(normalized_signal) > 0.2:  # 最终过滤
                            filtered_signals[ticker] = normalized_signal
                
                # 风险调整的仓位计算
                target_positions = self.calculate_position_sizes_with_risk_adjustment(
                    filtered_signals, current_prices, atr_data, current_cash
                )
                
                # 执行交易
                trades = self._execute_enhanced_trades(
                    current_portfolio, target_positions, current_prices, current_cash
                )
                
                # 更新组合
                for trade in trades:
                    if trade['action'] == 'BUY':
                        current_portfolio[trade['ticker']] = current_portfolio.get(trade['ticker'], 0) + trade['shares']
                        current_cash -= trade['value'] + trade['cost']
                    elif trade['action'] == 'SELL':
                        current_portfolio[trade['ticker']] = current_portfolio.get(trade['ticker'], 0) - trade['shares']
                        current_cash += trade['value'] - trade['cost']
                
                # 计算组合价值
                portfolio_value = current_cash
                for ticker, shares in current_portfolio.items():
                    if shares > 0 and ticker in current_prices:
                        portfolio_value += shares * current_prices[ticker]
                
                # 记录结果
                portfolio_values.append({
                    'date': rebalance_date,
                    'total_value': portfolio_value,
                    'cash': current_cash,
                    'positions_value': portfolio_value - current_cash,
                    'return': (portfolio_value - self.initial_capital) / self.initial_capital
                })
                
                if len(portfolio_values) > 1:
                    prev_value = portfolio_values[-2]['total_value']
                    period_return = (portfolio_value - prev_value) / prev_value
                    portfolio_returns.append(period_return)
                else:
                    portfolio_returns.append(0.0)
                
                # 记录信号和交易
                signal_history.extend(list(filtered_signals.values()))
                trades_history.extend(trades)
                positions_history.append({
                    'date': rebalance_date,
                    'positions': current_portfolio.copy(),
                    'signals': filtered_signals.copy(),
                    'thresholds': (min_threshold, max_threshold)
                })
                
            except Exception as e:
                self.logger.error(f"回测日期 {rebalance_date} 处理失败: {e}")
                continue
        
        # 整理结果
        results = {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'positions_history': positions_history,
            'trades_history': trades_history,
            'signal_history': signal_history,
            'performance_metrics': self._calculate_enhanced_performance_metrics(
                portfolio_returns, [pv['total_value'] for pv in portfolio_values]
            )
        }
        
        self.logger.info("增强版滚动前向回测完成")
        return results
    
    def _prepare_enhanced_training_data(self, 
                                       price_data: Dict[str, pd.DataFrame], 
                                       tickers: List[str], 
                                       train_dates: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.Series]:
        """准备增强版训练数据"""
        all_features = []
        all_targets = []
        
        for ticker in tickers:
            if ticker not in price_data:
                continue
                
            try:
                stock_data = price_data[ticker].loc[train_dates]
                if len(stock_data) < 50:
                    continue
                
                # 计算增强特征
                features = self.calculate_enhanced_features(stock_data)
                
                # 计算目标变量 - 与预测周期对齐
                target = stock_data['Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
                target.name = 'target'
                
                # 对齐数据
                aligned_data = pd.concat([features, target], axis=1)
                aligned_data = aligned_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(aligned_data) > 10:
                    all_features.append(aligned_data.drop('target', axis=1))
                    all_targets.append(aligned_data['target'])
                    
            except Exception as e:
                self.logger.warning(f"准备 {ticker} 训练数据失败: {e}")
                continue
        
        if all_features:
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_targets = pd.concat(all_targets, ignore_index=True)
            return combined_features, combined_targets
        else:
            return pd.DataFrame(), pd.Series()
    
    def _prepare_current_enhanced_features(self, 
                                          price_data: Dict[str, pd.DataFrame], 
                                          tickers: List[str], 
                                          current_date: pd.Timestamp,
                                          scaler: StandardScaler,
                                          selector) -> pd.DataFrame:
        """准备当前日期的增强特征数据"""
        current_features = []
        feature_tickers = []
        
        for ticker in tickers:
            if ticker not in price_data:
                continue
                
            try:
                stock_data = price_data[ticker].loc[:current_date]
                if len(stock_data) < 50:
                    continue
                
                features = self.calculate_enhanced_features(stock_data)
                
                if not features.empty:
                    latest_features = features.iloc[-1:].fillna(0)
                    latest_features.index = [ticker]
                    current_features.append(latest_features)
                    feature_tickers.append(ticker)
                    
            except Exception as e:
                self.logger.warning(f"准备 {ticker} 当前特征失败: {e}")
                continue
        
        if current_features:
            result = pd.concat(current_features)
            
            # 应用特征选择和标准化
            if selector is not None:
                result = pd.DataFrame(
                    selector.transform(result.fillna(0).replace([np.inf, -np.inf], 0)),
                    index=result.index,
                    columns=result.columns[selector.get_support()]
                )
            
            result_scaled = pd.DataFrame(
                scaler.transform(result.fillna(0).replace([np.inf, -np.inf], 0)),
                index=result.index,
                columns=result.columns
            )
            return result_scaled
        else:
            return pd.DataFrame()
    
    def _generate_bma_predictions(self, models: Dict, weights: Dict, features: pd.DataFrame) -> Dict[str, float]:
        """生成BMA预测"""
        predictions = {}
        
        for ticker in features.index:
            ticker_features = features.loc[[ticker]]
            weighted_prediction = 0.0
            
            for model_name, model in models.items():
                try:
                    pred = model.predict(ticker_features)[0]
                    weight = weights.get(model_name, 0)
                    weighted_prediction += pred * weight
                except:
                    continue
            
            predictions[ticker] = weighted_prediction
        
        return predictions
    
    def _execute_enhanced_trades(self, current_portfolio: Dict[str, int], 
                               target_positions: Dict[str, int],
                               current_prices: Dict[str, float],
                               current_cash: float) -> List[Dict]:
        """执行增强版交易，包含更复杂的交易成本模型"""
        trades = []
        
        # 先卖出
        for ticker in list(current_portfolio.keys()):
            current_shares = current_portfolio.get(ticker, 0)
            target_shares = target_positions.get(ticker, 0)
            
            if current_shares > target_shares:
                shares_to_sell = current_shares - target_shares
                if shares_to_sell > 0 and ticker in current_prices:
                    price = current_prices[ticker]
                    gross_value = shares_to_sell * price
                    
                    # 动态交易成本（基于交易规模）
                    base_cost = gross_value * self.transaction_cost
                    size_penalty = min(0.001, gross_value / 1000000 * 0.0001)  # 大额交易额外成本
                    total_cost = base_cost + gross_value * size_penalty
                    
                    net_value = gross_value - total_cost
                    
                    trades.append({
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': price,
                        'value': gross_value,
                        'cost': total_cost,
                        'net_value': net_value
                    })
        
        # 再买入
        for ticker, target_shares in target_positions.items():
            current_shares = current_portfolio.get(ticker, 0)
            
            if target_shares > current_shares:
                shares_to_buy = target_shares - current_shares
                if shares_to_buy > 0 and ticker in current_prices:
                    price = current_prices[ticker]
                    gross_value = shares_to_buy * price
                    
                    # 动态交易成本
                    base_cost = gross_value * self.transaction_cost
                    size_penalty = min(0.001, gross_value / 1000000 * 0.0001)
                    total_cost = base_cost + gross_value * size_penalty
                    
                    total_needed = gross_value + total_cost
                    
                    if total_needed <= current_cash:
                        trades.append({
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': price,
                            'value': gross_value,
                            'cost': total_cost,
                            'total_needed': total_needed
                        })
        
        return trades
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex, freq: str) -> List[pd.Timestamp]:
        """获取再平衡日期"""
        if freq == 'W':
            # 每周一
            rebalance_dates = []
            for date in dates:
                if date.weekday() == 0:  # Monday
                    rebalance_dates.append(date)
        elif freq == 'M':
            # 每月第一个交易日
            rebalance_dates = []
            current_month = None
            for date in dates:
                if current_month != date.month:
                    rebalance_dates.append(date)
                    current_month = date.month
        else:
            # 默认每周
            rebalance_dates = [dates[i] for i in range(0, len(dates), 5)]
        
        return rebalance_dates
    
    def _calculate_enhanced_performance_metrics(self, returns: List[float], values: List[float]) -> Dict:
        """计算增强版绩效指标"""
        if len(returns) == 0 or len(values) == 0:
            return {}
        
        try:
            returns_series = pd.Series(returns)
            
            # 基本指标
            total_return = (values[-1] - values[0]) / values[0] if len(values) > 0 else 0
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
            volatility = returns_series.std() * np.sqrt(252) if len(returns) > 1 else 0
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # 最大回撤
            cumulative = np.cumprod(1 + returns_series)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
            
            # 获胜率
            win_rate = (returns_series > 0).mean() if len(returns) > 0 else 0
            
            # Calmar比率
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # Sortino比率
            downside_returns = returns_series[returns_series < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # VaR和CVaR
            var_95 = np.percentile(returns_series, 5) if len(returns_series) > 20 else 0
            cvar_95 = returns_series[returns_series <= var_95].mean() if len(returns_series[returns_series <= var_95]) > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'total_trades': len(returns),
                'avg_return': returns_series.mean(),
                'return_skewness': returns_series.skew(),
                'return_kurtosis': returns_series.kurtosis()
            }
            
        except Exception as e:
            logger.error(f"绩效计算失败: {e}")
            return {}
    
    def create_enhanced_visualizations(self, results: Dict, save_path: str = "result"):
        """创建增强版可视化图表"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
            
            os.makedirs(save_path, exist_ok=True)
            
            if not results or 'portfolio_values' not in results:
                logger.warning("无可视化数据")
                return
            
            portfolio_values = results['portfolio_values']
            if not portfolio_values:
                logger.warning("组合价值数据为空")
                return
            
            # 1. 组合价值曲线
            dates = [pv['date'] for pv in portfolio_values]
            values = [pv['total_value'] for pv in portfolio_values]
            returns = [pv['return'] for pv in portfolio_values]
            
            fig = make_subplots(rows=3, cols=2,
                               subplot_titles=('组合价值曲线', '累积收益率', 
                                             '回撤曲线', '滚动波动率',
                                             '月度收益分布', '风险调整收益'),
                               specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                     [{"secondary_y": False}, {"secondary_y": False}],
                                     [{"secondary_y": False}, {"secondary_y": False}]])
            
            # 组合价值
            fig.add_trace(go.Scatter(x=dates, y=values, name='组合价值', 
                                   line=dict(color='blue', width=2)), row=1, col=1)
            
            # 累积收益率
            fig.add_trace(go.Scatter(x=dates, y=returns, name='累积收益率',
                                   line=dict(color='green', width=2)), row=1, col=2)
            
            # 回撤计算和绘制
            if len(values) > 1:
                peak = np.maximum.accumulate(values)
                drawdown = [(v - p) / p for v, p in zip(values, peak)]
                fig.add_trace(go.Scatter(x=dates, y=drawdown, name='回撤',
                                       fill='tozeroy', line=dict(color='red')), row=2, col=1)
            
            # 滚动波动率
            if 'portfolio_returns' in results and len(results['portfolio_returns']) > 20:
                returns_series = pd.Series(results['portfolio_returns'])
                rolling_vol = returns_series.rolling(window=20).std() * np.sqrt(252)
                fig.add_trace(go.Scatter(x=dates[1:len(rolling_vol)+1], y=rolling_vol,
                                       name='滚动波动率', line=dict(color='orange')), row=2, col=2)
            
            # 月度收益分布
            if len(results['portfolio_returns']) > 0:
                monthly_returns = results['portfolio_returns']
                fig.add_trace(go.Histogram(x=monthly_returns, name='收益分布',
                                         nbinsx=20, opacity=0.7), row=3, col=1)
            
            # 风险调整收益 (Sharpe比率滚动)
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                risk_metrics = [metrics.get('sharpe_ratio', 0)] * len(dates)
                fig.add_trace(go.Scatter(x=dates, y=risk_metrics, name='Sharpe比率',
                                       line=dict(color='purple')), row=3, col=2)
            
            fig.update_layout(height=900, title_text="BMA增强版回测结果", showlegend=True)
            fig.write_html(f"{save_path}/enhanced_backtest_results.html")
            
            # 2. 单独的交易信号分析图
            if 'positions_history' in results:
                positions_data = results['positions_history']
                
                # 提取信号数据
                signal_dates = []
                signal_counts = []
                avg_signals = []
                
                for pos in positions_data:
                    if 'signals' in pos:
                        signal_dates.append(pos['date'])
                        signals = list(pos['signals'].values())
                        signal_counts.append(len(signals))
                        avg_signals.append(np.mean([abs(s) for s in signals]) if signals else 0)
                
                # 信号分析图
                fig2 = make_subplots(rows=2, cols=1, 
                                    subplot_titles=('持仓数量变化', '平均信号强度'))
                
                fig2.add_trace(go.Bar(x=signal_dates, y=signal_counts, name='持仓数量'), row=1, col=1)
                fig2.add_trace(go.Scatter(x=signal_dates, y=avg_signals, name='平均信号强度',
                                        line=dict(color='red')), row=2, col=1)
                
                fig2.update_layout(height=600, title_text="交易信号分析")
                fig2.write_html(f"{save_path}/signal_analysis.html")
            
            # 3. 绩效指标雷达图
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                
                categories = ['年化收益', 'Sharpe比率', 'Sortino比率', 'Calmar比率', '获胜率']
                values = [
                    max(0, min(1, metrics.get('annualized_return', 0) / 0.3)),  # 归一化到0-1
                    max(0, min(1, metrics.get('sharpe_ratio', 0) / 2)),
                    max(0, min(1, metrics.get('sortino_ratio', 0) / 2)), 
                    max(0, min(1, metrics.get('calmar_ratio', 0) / 2)),
                    metrics.get('win_rate', 0)
                ]
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='BMA策略'
                ))
                
                fig3.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=True,
                    title="策略绩效雷达图"
                )
                fig3.write_html(f"{save_path}/performance_radar.html")
            
            logger.info(f"增强版可视化图表已保存到 {save_path}/")
            
        except Exception as e:
            logger.error(f"创建可视化失败: {e}")


def run_enhanced_backtest_demo():
    """运行增强版回测演示"""
    logger.info("=== BMA增强版滚动前向回测演示 ===")
    
    try:
        # 创建增强版回测器
        backtest = EnhancedBMAWalkForward(
            initial_capital=200000,
            max_positions=20,
            prediction_horizon=7,  # 7天预测周期，与周度再平衡对齐
            training_window_months=6,
            min_training_samples=120
        )
        
        # 使用更大的股票池
        test_tickers = ENHANCED_STOCK_POOL[:30]  # 使用前30只股票
        
        logger.info(f"测试股票池: {len(test_tickers)} 只股票")
        logger.info(f"股票: {', '.join(test_tickers[:10])}...")
        
        # 运行回测
        results = backtest.run_enhanced_walkforward_backtest(
            tickers=test_tickers,
            start_date="2021-01-01",
            end_date="2024-08-01"
        )
        
        # 显示结果
        if results and results['portfolio_values']:
            portfolio_values = results['portfolio_values']
            metrics = results.get('performance_metrics', {})
            
            print("\n=== 回测结果摘要 ===")
            print(f"初始资金: ${200000:,}")
            print(f"最终价值: ${portfolio_values[-1]['total_value']:,.2f}")
            print(f"总收益率: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"年化收益率: {metrics.get('annualized_return', 0)*100:.2f}%")
            print(f"年化波动率: {metrics.get('volatility', 0)*100:.2f}%")
            print(f"Sharpe比率: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"获胜率: {metrics.get('win_rate', 0)*100:.2f}%")
            print(f"交易次数: {len(results.get('trades_history', []))}")
            
            # 创建可视化
            backtest.create_enhanced_visualizations(results)
            print(f"\n可视化图表已生成在 result/ 文件夹")
            
            # 保存详细结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"result/enhanced_bma_backtest_{timestamp}.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"详细结果已保存到: {result_file}")
            
        else:
            print("回测完成但未生成有效结果")
            
    except Exception as e:
        logger.error(f"回测失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_enhanced_backtest_demo()