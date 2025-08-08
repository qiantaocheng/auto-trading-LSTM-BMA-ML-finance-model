#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA增强策略滚动前向回测系统
严格的数据分割，防止未来信息泄露
集成自动交易策略，生成收益率曲线和可视化结果
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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

# 设置日志和警告
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class BMAWalkForwardBacktest:
    """BMA策略滚动前向回测器"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 max_positions: int = 10,
                 rebalance_freq: str = 'W',  # W=weekly, M=monthly
                 training_window_months: int = 12,
                 min_training_samples: int = 200):
        """
        初始化回测器
        
        Args:
            initial_capital: 初始资金
            transaction_cost: 交易成本（包含滑点和手续费）
            max_positions: 最大持仓数量
            rebalance_freq: 再平衡频率
            training_window_months: 训练窗口月数
            min_training_samples: 最小训练样本数
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_positions = max_positions
        self.rebalance_freq = rebalance_freq
        self.training_window_months = training_window_months
        self.min_training_samples = min_training_samples
        
        # 回测结果存储
        self.portfolio_returns = []
        self.portfolio_values = []
        self.positions_history = []
        self.trading_signals = []
        self.performance_metrics = {}
        self.model_performance_history = []
        
        self.logger = logger
        
    def calculate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算BMA技术特征
        严格按时序计算，防止未来信息泄露
        """
        features = pd.DataFrame(index=data.index)
        
        try:
            # 价格特征 - 移动平均
            features['sma_5'] = data['Close'].rolling(window=5, min_periods=5).mean()
            features['sma_10'] = data['Close'].rolling(window=10, min_periods=10).mean()
            features['sma_20'] = data['Close'].rolling(window=20, min_periods=20).mean()
            features['sma_50'] = data['Close'].rolling(window=50, min_periods=50).mean()
            
            # 价格动量特征
            features['momentum_5'] = data['Close'].pct_change(5)
            features['momentum_15'] = data['Close'].pct_change(15)
            features['price_acceleration'] = features['momentum_5'] - features['momentum_15']
            
            # 相对强弱
            features['rs_vs_sma20'] = (data['Close'] / features['sma_20'] - 1).fillna(0)
            features['high_low_ratio'] = data['High'] / data['Low'] - 1
            features['close_position'] = ((data['Close'] - data['Low']) / (data['High'] - data['Low'])).fillna(0)
            
            # 成交量特征
            features['volume_sma_10'] = data['Volume'].rolling(window=10, min_periods=10).mean()
            features['volume_ratio'] = (data['Volume'] / features['volume_sma_10']).fillna(0)
            features['volume_momentum'] = data['Volume'].pct_change(10)
            
            # 波动率特征
            features['volatility_20'] = data['Close'].pct_change().rolling(window=20, min_periods=20).std()
            
            # RSI指标
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
            rs = gain / (loss + 1e-8)
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD指标
            ema12 = data['Close'].ewm(span=12).mean()
            ema26 = data['Close'].ewm(span=26).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # 布林带
            bb_middle = data['Close'].rolling(window=20, min_periods=20).mean()
            bb_std = data['Close'].rolling(window=20, min_periods=20).std()
            features['bollinger_upper'] = bb_middle + (bb_std * 2)
            features['bollinger_lower'] = bb_middle - (bb_std * 2)
            features['bollinger_position'] = ((data['Close'] - features['bollinger_lower']) / (features['bollinger_upper'] - features['bollinger_lower'])).fillna(0)
            
            # CCI指标
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            sma_tp = typical_price.rolling(window=20, min_periods=20).mean()
            mad_tp = typical_price.rolling(window=20, min_periods=20).apply(
                lambda x: np.mean(np.abs(x - x.mean())) if len(x) >= 20 else np.nan, raw=True
            )
            features['cci'] = (typical_price - sma_tp) / (0.015 * mad_tp + 1e-8)
            features['cci_normalized'] = features['cci'] / 100.0
            
            # 市场情绪特征
            features['gap_ratio'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
            features['intraday_return'] = (data['Close'] - data['Open']) / data['Open']
            
            # 滚动Sharpe比率
            returns = data['Close'].pct_change()
            features['rolling_sharpe'] = (
                returns.rolling(window=20, min_periods=20).mean() / 
                (returns.rolling(window=20, min_periods=20).std() + 1e-8)
            ) * np.sqrt(252)
            
            # 价格相对位置
            features['price_percentile'] = data['Close'].rolling(window=50, min_periods=50).rank(pct=True)
            
            return features
            
        except Exception as e:
            self.logger.error(f"特征计算失败: {e}")
            return pd.DataFrame(index=data.index)
    
    def create_bma_model(self) -> Dict:
        """创建BMA集成模型"""
        base_models = {}
        
        # 基础模型
        base_models['RandomForest'] = RandomForestRegressor(
            n_estimators=100, 
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42, 
            n_jobs=-1
        )
        
        base_models['Ridge'] = Ridge(alpha=1.0)
        base_models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)
        
        # 高级模型（如果可用）
        if XGBOOST_AVAILABLE:
            base_models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        
        if LIGHTGBM_AVAILABLE:
            base_models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            
        if CATBOOST_AVAILABLE:
            base_models['CatBoost'] = CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
        
        return base_models
    
    def train_bma_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict, Dict]:
        """
        训练BMA集成模型
        使用时序交叉验证计算权重
        """
        base_models = self.create_bma_model()
        model_weights = {}
        trained_models = {}
        
        # 时序交叉验证
        n_splits = min(5, len(X) // 50)  # 动态调整splits数量
        if n_splits < 3:
            n_splits = 3
            
        tscv = TimeSeriesSplit(n_splits=n_splits)
        model_scores = {name: [] for name in base_models.keys()}
        
        # 数据预处理
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.fillna(X.median()).replace([np.inf, -np.inf], 0))
        
        # 交叉验证评估
        for name, model in base_models.items():
            try:
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train_fold = X_scaled[train_idx]
                    X_val_fold = X_scaled[val_idx]
                    y_train_fold = y.iloc[train_idx].values
                    y_val_fold = y.iloc[val_idx].values
                    
                    # 训练模型
                    model_copy = self._clone_model(model)
                    model_copy.fit(X_train_fold, y_train_fold)
                    
                    # 预测和评估
                    y_pred = model_copy.predict(X_val_fold)
                    r2 = r2_score(y_val_fold, y_pred)
                    cv_scores.append(max(0, r2))  # 确保分数非负
                
                model_scores[name] = cv_scores
                
            except Exception as e:
                self.logger.warning(f"模型 {name} 交叉验证失败: {e}")
                model_scores[name] = [0.0] * n_splits
        
        # 计算模型权重（基于平均R²分数）
        mean_scores = {name: np.mean(scores) for name, scores in model_scores.items()}
        total_score = sum(mean_scores.values()) + 1e-8
        
        for name in base_models.keys():
            model_weights[name] = mean_scores[name] / total_score
        
        # 在全数据上训练模型
        for name, model in base_models.items():
            try:
                trained_model = self._clone_model(model)
                trained_model.fit(X_scaled, y.values)
                trained_models[name] = {
                    'model': trained_model,
                    'scaler': scaler,
                    'feature_names': X.columns.tolist()
                }
            except Exception as e:
                self.logger.warning(f"模型 {name} 最终训练失败: {e}")
        
        self.logger.info(f"BMA权重分配: {model_weights}")
        return trained_models, model_weights
    
    def _clone_model(self, model):
        """克隆模型"""
        from copy import deepcopy
        return deepcopy(model)
    
    def predict_ensemble(self, trained_models: Dict, model_weights: Dict, X: pd.DataFrame) -> np.ndarray:
        """使用BMA集成预测"""
        predictions = []
        total_weight = 0
        
        for name, model_info in trained_models.items():
            try:
                # 数据预处理
                X_processed = X.reindex(columns=model_info['feature_names'], fill_value=0)
                X_scaled = model_info['scaler'].transform(
                    X_processed.fillna(X_processed.median()).replace([np.inf, -np.inf], 0)
                )
                
                # 预测
                pred = model_info['model'].predict(X_scaled)
                weight = model_weights.get(name, 0)
                
                predictions.append(pred * weight)
                total_weight += weight
                
            except Exception as e:
                self.logger.warning(f"模型 {name} 预测失败: {e}")
                continue
        
        if predictions and total_weight > 0:
            ensemble_pred = np.sum(predictions, axis=0) / total_weight
        else:
            ensemble_pred = np.zeros(len(X))
            
        return ensemble_pred
    
    def generate_trading_signals(self, predictions: np.ndarray, tickers: List[str]) -> Dict[str, float]:
        """
        根据预测结果生成交易信号
        使用分位数方法确定买卖信号
        """
        if len(predictions) == 0 or len(tickers) == 0:
            return {}
            
        # 创建信号字典
        signals = dict(zip(tickers, predictions))
        
        # 过滤无效预测
        valid_signals = {k: v for k, v in signals.items() if not np.isnan(v) and np.isfinite(v)}
        
        if not valid_signals:
            return {}
            
        # 使用分位数确定强信号
        values = list(valid_signals.values())
        if len(values) < 3:
            return valid_signals
            
        # 计算分位数阈值
        q75 = np.percentile(values, 75)
        q25 = np.percentile(values, 25)
        
        # 标准化信号强度到[-1, 1]区间
        normalized_signals = {}
        for ticker, signal in valid_signals.items():
            if signal >= q75:
                # 强买入信号
                normalized_signals[ticker] = min(1.0, (signal - q75) / (max(values) - q75 + 1e-8) + 0.5)
            elif signal <= q25:
                # 强卖出信号  
                normalized_signals[ticker] = max(-1.0, (signal - q25) / (q25 - min(values) + 1e-8) - 0.5)
            else:
                # 中性信号
                normalized_signals[ticker] = (signal - np.median(values)) / (q75 - q25 + 1e-8) * 0.5
                
        return normalized_signals
    
    def execute_portfolio_rebalancing(self, 
                                     signals: Dict[str, float], 
                                     current_prices: Dict[str, float],
                                     current_portfolio: Dict[str, float],
                                     available_cash: float) -> Tuple[Dict[str, float], float, List[Dict]]:
        """
        执行投资组合再平衡
        返回新的持仓、现金和交易记录
        """
        trades = []
        new_portfolio = current_portfolio.copy()
        new_cash = available_cash
        
        if not signals or not current_prices:
            return new_portfolio, new_cash, trades
        
        # 选择最强的买入信号
        buy_signals = {k: v for k, v in signals.items() if v > 0.3 and k in current_prices}
        sell_signals = {k: v for k, v in signals.items() if v < -0.3 and k in current_prices}
        
        # 按信号强度排序
        sorted_buy = sorted(buy_signals.items(), key=lambda x: x[1], reverse=True)[:self.max_positions]
        sorted_sell = {k: v for k, v in sell_signals.items() if k in current_portfolio}
        
        # 先执行卖出操作
        for ticker in list(current_portfolio.keys()):
            if ticker in sorted_sell or ticker not in [t[0] for t in sorted_buy]:
                if current_portfolio[ticker] > 0:
                    # 卖出持仓
                    shares = current_portfolio[ticker]
                    sell_price = current_prices.get(ticker, 0)
                    if sell_price > 0:
                        sell_value = shares * sell_price
                        transaction_cost = sell_value * self.transaction_cost
                        net_proceeds = sell_value - transaction_cost
                        
                        new_cash += net_proceeds
                        new_portfolio[ticker] = 0
                        
                        trades.append({
                            'ticker': ticker,
                            'action': 'SELL',
                            'shares': shares,
                            'price': sell_price,
                            'value': sell_value,
                            'cost': transaction_cost,
                            'signal': signals.get(ticker, 0)
                        })
        
        # 清理零持仓
        new_portfolio = {k: v for k, v in new_portfolio.items() if v > 0}
        
        # 执行买入操作
        if sorted_buy and new_cash > 1000:  # 保留最少1000现金
            # 计算每个持仓的目标权重
            total_signal = sum([signal for _, signal in sorted_buy])
            target_cash = new_cash * 0.95  # 保留5%现金
            
            for ticker, signal in sorted_buy:
                buy_price = current_prices.get(ticker, 0)
                if buy_price > 0:
                    # 基于信号强度分配资金
                    weight = signal / total_signal
                    target_value = target_cash * weight
                    
                    # 计算交易成本后的可用资金
                    available_for_purchase = target_value / (1 + self.transaction_cost)
                    shares_to_buy = int(available_for_purchase // buy_price)
                    
                    if shares_to_buy > 0:
                        purchase_value = shares_to_buy * buy_price
                        transaction_cost = purchase_value * self.transaction_cost
                        total_cost = purchase_value + transaction_cost
                        
                        if total_cost <= new_cash:
                            new_cash -= total_cost
                            new_portfolio[ticker] = new_portfolio.get(ticker, 0) + shares_to_buy
                            
                            trades.append({
                                'ticker': ticker,
                                'action': 'BUY',
                                'shares': shares_to_buy,
                                'price': buy_price,
                                'value': purchase_value,
                                'cost': transaction_cost,
                                'signal': signal
                            })
        
        return new_portfolio, new_cash, trades
    
    def run_walkforward_backtest(self, 
                                tickers: List[str], 
                                start_date: str, 
                                end_date: str,
                                price_data: Dict[str, pd.DataFrame] = None) -> Dict:
        """
        运行滚动前向回测
        严格的时序分割，防止数据泄露
        """
        self.logger.info(f"开始滚动前向回测: {start_date} 到 {end_date}")
        self.logger.info(f"股票池: {len(tickers)} 只股票")
        
        # 下载或使用提供的价格数据
        if price_data is None:
            price_data = self._download_price_data(tickers, start_date, end_date)
        
        if not price_data:
            raise ValueError("无法获取价格数据")
        
        # 对齐所有数据的日期索引
        common_dates = self._get_common_dates(price_data)
        if len(common_dates) < 30:  # 降低最小日期要求
            raise ValueError(f"可用交易日期太少: {len(common_dates)} < 30")
        
        # 初始化回测变量
        current_capital = self.initial_capital
        current_cash = self.initial_capital
        current_portfolio = {}  # {ticker: shares}
        
        portfolio_values = []
        portfolio_returns = []
        positions_history = []
        trades_history = []
        model_performance = []
        
        # 设置回测窗口
        training_days = self.training_window_months * 21  # 每月约21个交易日
        min_training_idx = max(training_days, self.min_training_samples)
        
        # 确定再平衡日期
        rebalance_dates = self._get_rebalance_dates(common_dates, self.rebalance_freq)
        
        self.logger.info(f"训练窗口: {training_days} 天")
        self.logger.info(f"再平衡次数: {len(rebalance_dates)} 次")
        
        # 滚动前向回测主循环
        for i, rebalance_date in enumerate(rebalance_dates):
            if rebalance_date not in common_dates:
                continue
                
            current_idx = common_dates.get_loc(rebalance_date)
            
            # 确保有足够的训练数据
            if current_idx < min_training_idx:
                continue
                
            try:
                self.logger.info(f"再平衡 {i+1}/{len(rebalance_dates)}: {rebalance_date.strftime('%Y-%m-%d')}")
                
                # 1. 准备训练数据（严格使用历史数据）
                train_start_idx = max(0, current_idx - training_days)
                train_end_idx = current_idx  # 不包含当前日期
                
                train_features, train_targets = self._prepare_training_data(
                    price_data, tickers, common_dates[train_start_idx:train_end_idx]
                )
                
                if len(train_features) < 50:
                    self.logger.warning(f"训练数据不足: {len(train_features)} 样本")
                    continue
                
                # 2. 训练BMA模型
                trained_models, model_weights = self.train_bma_ensemble(train_features, train_targets)
                
                # 记录模型性能
                train_pred = self.predict_ensemble(trained_models, model_weights, train_features)
                train_r2 = r2_score(train_targets, train_pred) if len(train_pred) > 0 else 0
                
                model_performance.append({
                    'date': rebalance_date,
                    'train_r2': train_r2,
                    'train_samples': len(train_features),
                    'weights': model_weights.copy()
                })
                
                # 3. 生成当日预测（使用当日可用数据）
                current_features = self._prepare_current_features(
                    price_data, tickers, rebalance_date
                )
                
                if current_features.empty:
                    continue
                    
                predictions = self.predict_ensemble(trained_models, model_weights, current_features)
                
                # 4. 生成交易信号
                current_tickers = current_features.index.tolist()
                signals = self.generate_trading_signals(predictions, current_tickers)
                
                # 5. 获取当日价格
                current_prices = {}
                for ticker in tickers:
                    if ticker in price_data and rebalance_date in price_data[ticker].index:
                        current_prices[ticker] = price_data[ticker].loc[rebalance_date, 'Close']
                
                # 6. 执行投资组合再平衡
                new_portfolio, new_cash, trades = self.execute_portfolio_rebalancing(
                    signals, current_prices, current_portfolio, current_cash
                )
                
                # 7. 更新投资组合状态
                current_portfolio = new_portfolio
                current_cash = new_cash
                
                # 8. 计算投资组合价值
                portfolio_value = current_cash
                for ticker, shares in current_portfolio.items():
                    if shares > 0 and ticker in current_prices:
                        portfolio_value += shares * current_prices[ticker]
                
                # 9. 记录结果
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
                
                positions_history.append({
                    'date': rebalance_date,
                    'positions': current_portfolio.copy(),
                    'signals': signals.copy()
                })
                
                trades_history.extend(trades)
                
                # 记录交易信息
                if trades:
                    self.logger.info(f"执行 {len(trades)} 笔交易，投资组合价值: ${portfolio_value:,.2f}")
                
            except Exception as e:
                self.logger.error(f"回测日期 {rebalance_date} 处理失败: {e}")
                continue
        
        # 整理回测结果
        results = {
            'portfolio_values': portfolio_values,  # 保持为list格式避免索引问题
            'portfolio_returns': portfolio_returns,
            'positions_history': positions_history,
            'trades_history': trades_history,
            'model_performance': model_performance,
            'performance_metrics': self._calculate_performance_metrics(
                pd.Series(portfolio_returns[1:] if len(portfolio_returns) > 1 else []) if portfolio_returns else pd.Series([]),
                [pv['total_value'] for pv in portfolio_values] if portfolio_values else []
            )
        }
        
        self.logger.info("滚动前向回测完成")
        return results
    
    def _download_price_data(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """下载价格数据（增强版本）"""
        self.logger.info(f"下载 {len(tickers)} 只股票的价格数据")
        
        price_data = {}
        success_count = 0
        
        for ticker in tickers:
            try:
                # 尝试下载数据，使用多次重试
                for attempt in range(3):  # 最多3次重试
                    try:
                        self.logger.info(f"正在下载 {ticker} 数据... (第{attempt+1}次尝试)")
                        data = yf.download(ticker, start=start_date, end=end_date, progress=False, threads=False, auto_adjust=True, prepost=True)
                        
                        if data is not None and not data.empty and len(data) >= 50:  # 降低最小数据要求
                            # yfinance可能返回多级列名，需要处理
                            if isinstance(data.columns, pd.MultiIndex):
                                # 将多级列名展平为单级
                                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
                            
                            # 检查数据完整性
                            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                            available_columns = [col for col in required_columns if col in data.columns]
                            
                            if len(available_columns) >= 4:  # 至少需要OHLC四个列
                                # 清理数据
                                data = data[available_columns].dropna()
                                if len(data) >= 30:  # 进一步降低要求
                                    price_data[ticker] = data
                                    success_count += 1
                                    self.logger.info(f"\u2713 {ticker}: 获得 {len(data)} 个交易日数据，列: {list(data.columns)}")
                                    break  # 成功后退出重试循环
                                else:
                                    self.logger.warning(f"{ticker}: 清理后数据不足 ({len(data)} < 30)")
                            else:
                                self.logger.warning(f"{ticker}: 数据列不完整，可用: {available_columns}")
                        else:
                            self.logger.warning(f"{ticker}: 获得空数据或数据不足")
                            
                        if attempt < 2:  # 不是最后一次尝试时等待
                            time.sleep(1)
                            
                    except Exception as download_error:
                        self.logger.warning(f"{ticker} 第{attempt+1}次下载尝试失败: {download_error}")
                        if attempt < 2:
                            time.sleep(2)  # 逐渐增加等待时间
                        continue
                        
            except Exception as e:
                self.logger.error(f"{ticker} 数据下载失败: {e}")
                continue
        
        self.logger.info(f"成功下载 {success_count}/{len(tickers)} 只股票数据")
        
        # 如果没有任何数据，尝试使用默认股票池
        if not price_data:
            self.logger.warning("未能下载任何数据，尝试使用单个AAL股票进行测试")
            try:
                test_data = yf.download('AAPL', start=start_date, end=end_date, progress=False, auto_adjust=True)
                if test_data is not None and not test_data.empty:
                    # 处理多级列名
                    if isinstance(test_data.columns, pd.MultiIndex):
                        test_data.columns = [col[0] if isinstance(col, tuple) else col for col in test_data.columns]
                    
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    available_columns = [col for col in required_columns if col in test_data.columns]
                    
                    if available_columns:
                        clean_data = test_data[available_columns].dropna()
                        if len(clean_data) > 20:
                            price_data['AAPL'] = clean_data
                            self.logger.info(f"\u2713 备用数据: AAPL 获得 {len(clean_data)} 个交易日，列: {list(clean_data.columns)}")
            except Exception as e:
                self.logger.error(f"备用数据下载也失败: {e}")
        
        return price_data
    
    def _get_common_dates(self, price_data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """获取所有股票的共同交易日期"""
        if not price_data:
            return pd.DatetimeIndex([])
        
        common_dates = None
        for ticker, data in price_data.items():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)
        
        return common_dates.sort_values()
    
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
            rebalance_dates = [dates[i] for i in range(0, len(dates), 7)]
        
        return rebalance_dates[:100]  # 限制最大再平衡次数
    
    def _prepare_training_data(self, 
                              price_data: Dict[str, pd.DataFrame], 
                              tickers: List[str], 
                              train_dates: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
        all_features = []
        all_targets = []
        
        for ticker in tickers:
            if ticker not in price_data:
                continue
                
            try:
                # 获取该股票在训练期间的数据
                stock_data = price_data[ticker].loc[train_dates]
                if len(stock_data) < 50:
                    continue
                
                # 计算特征
                features = self.calculate_technical_features(stock_data)
                
                # 计算目标变量（3日后收益率，更短期）
                target = stock_data['Close'].pct_change(3).shift(-3)
                target.name = 'target'
                
                # 对齐数据
                aligned_data = pd.concat([features, target], axis=1)
                aligned_data = aligned_data.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(aligned_data) > 5:  # 降低要求
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
    
    def _prepare_current_features(self, 
                                 price_data: Dict[str, pd.DataFrame], 
                                 tickers: List[str], 
                                 current_date: pd.Timestamp) -> pd.DataFrame:
        """准备当前日期的特征数据（用于预测）"""
        current_features = []
        feature_tickers = []
        
        for ticker in tickers:
            if ticker not in price_data:
                continue
                
            try:
                # 获取到当前日期为止的数据（不包含当前日期的未来信息）
                stock_data = price_data[ticker].loc[:current_date]
                if len(stock_data) < 100:
                    continue
                
                # 计算特征（使用到current_date为止的数据）
                features = self.calculate_technical_features(stock_data)
                
                # 获取最新的特征值
                if not features.empty:
                    latest_features = features.iloc[-1:].fillna(0)
                    latest_features.index = [ticker]
                    current_features.append(latest_features)
                    feature_tickers.append(ticker)
                    
            except Exception as e:
                self.logger.warning(f"准备 {ticker} 当前特征失败: {e}")
                continue
        
        if current_features:
            return pd.concat(current_features)
        else:
            return pd.DataFrame()
    
    def _calculate_performance_metrics(self, returns: pd.Series, values: List[float]) -> Dict:
        """计算绩效指标"""
        if len(returns) == 0 or len(values) == 0:
            return {}
        
        try:
            # 基本指标
            total_return = (values[-1] - values[0]) / values[0]
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # 最大回撤
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
            
            # 获胜率
            win_rate = (returns > 0).mean() if len(returns) > 0 else 0
            
            # Calmar比率
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'total_trades': len(returns),
                'start_value': values[0],
                'end_value': values[-1]
            }
            
        except Exception as e:
            self.logger.error(f"计算绩效指标失败: {e}")
            return {}
    
    def create_performance_visualization(self, results: Dict, save_path: str = None) -> None:
        """创建绩效可视化图表"""
        
        if not results or 'portfolio_values' not in results:
            self.logger.error("无回测结果可视化")
            return
        
        portfolio_values = results['portfolio_values']
        portfolio_returns = results['portfolio_returns']
        performance_metrics = results.get('performance_metrics', {})
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '投资组合净值曲线', '累积收益率 vs 基准',
                '滚动收益率', '月度收益率热力图',  
                '回撤分析', '持仓分布'
            ),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 投资组合净值曲线
        fig.add_trace(
            go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values['total_value'],
                mode='lines',
                name='投资组合净值',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 添加现金比例
        fig.add_trace(
            go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values['cash'] / portfolio_values['total_value'] * 100,
                mode='lines',
                name='现金比例 (%)',
                line=dict(color='green', width=1),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 2. 累积收益率对比
        if not portfolio_returns.empty:
            cumulative_returns = (1 + portfolio_returns).cumprod() - 1
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values * 100,
                    mode='lines',
                    name='策略累积收益率 (%)',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )
        
        # 3. 滚动收益率
        if not portfolio_returns.empty:
            rolling_returns = portfolio_returns.rolling(window=20).mean() * 100
            fig.add_trace(
                go.Scatter(
                    x=rolling_returns.index,
                    y=rolling_returns.values,
                    mode='lines',
                    name='20期滚动收益率 (%)',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
        
        # 4. 月度收益率热力图
        if not portfolio_returns.empty:
            monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            monthly_data = monthly_returns.to_frame('return')
            monthly_data['year'] = monthly_data.index.year
            monthly_data['month'] = monthly_data.index.month
            
            if len(monthly_data) > 12:
                heatmap_data = monthly_data.pivot_table(values='return', index='year', columns='month', fill_value=0)
                
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data.values,
                        x=[f'{i}月' for i in heatmap_data.columns],
                        y=heatmap_data.index,
                        colorscale='RdYlGn',
                        colorbar=dict(title="收益率 (%)"),
                        name='月度收益率'
                    ),
                    row=2, col=2
                )
        
        # 5. 回撤分析
        if not portfolio_returns.empty:
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max * 100
            
            fig.add_trace(
                go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns.values,
                    mode='lines',
                    name='回撤 (%)',
                    fill='tonexty',
                    line=dict(color='red', width=1)
                ),
                row=3, col=1
            )
        
        # 6. 持仓分布（最新）
        if 'positions_history' in results and results['positions_history']:
            latest_positions = results['positions_history'][-1]['positions']
            if latest_positions:
                tickers = list(latest_positions.keys())
                values = list(latest_positions.values())
                
                fig.add_trace(
                    go.Pie(
                        labels=tickers,
                        values=values,
                        name='持仓分布'
                    ),
                    row=3, col=2
                )
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=f"BMA策略滚动前向回测结果<br><sub>总收益: {performance_metrics.get('total_return', 0)*100:.2f}% | "
                     f"夏普比率: {performance_metrics.get('sharpe_ratio', 0):.2f} | "
                     f"最大回撤: {performance_metrics.get('max_drawdown', 0)*100:.2f}%</sub>",
                x=0.5
            ),
            height=1200,
            width=1600,
            showlegend=True,
            template="plotly_white"
        )
        
        # 设置y轴标签
        fig.update_yaxes(title_text="投资组合净值 ($)", row=1, col=1)
        fig.update_yaxes(title_text="现金比例 (%)", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="累积收益率 (%)", row=1, col=2)
        fig.update_yaxes(title_text="滚动收益率 (%)", row=2, col=1)
        fig.update_yaxes(title_text="回撤 (%)", row=3, col=1)
        
        # 显示图表
        fig.show()
        
        # 保存图表
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"可视化图表已保存到: {save_path}")
    
    def save_results(self, results: Dict, filename: str = None) -> str:
        """保存回测结果"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'bma_walkforward_backtest_{timestamp}'
        
        # 创建结果目录
        result_dir = Path('backtest_results')
        result_dir.mkdir(exist_ok=True)
        
        try:
            # 保存Excel文件
            excel_path = result_dir / f"{filename}.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                
                # 投资组合价值
                results['portfolio_values'].to_excel(writer, sheet_name='投资组合价值')
                
                # 收益率序列
                if not results['portfolio_returns'].empty:
                    results['portfolio_returns'].to_excel(writer, sheet_name='收益率序列')
                
                # 交易记录
                if not results['trades_history'].empty:
                    results['trades_history'].to_excel(writer, sheet_name='交易记录', index=False)
                
                # 模型性能
                if not results['model_performance'].empty:
                    results['model_performance'].to_excel(writer, sheet_name='模型性能')
                
                # 绩效指标
                metrics_df = pd.DataFrame([results['performance_metrics']]).T
                metrics_df.columns = ['值']
                metrics_df.to_excel(writer, sheet_name='绩效指标')
            
            # 保存JSON文件（用于程序读取）
            json_path = result_dir / f"{filename}.json"
            json_results = {
                'performance_metrics': results['performance_metrics'],
                'backtest_params': {
                    'initial_capital': self.initial_capital,
                    'transaction_cost': self.transaction_cost,
                    'max_positions': self.max_positions,
                    'rebalance_freq': self.rebalance_freq,
                    'training_window_months': self.training_window_months
                }
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"回测结果已保存:")
            self.logger.info(f"  Excel: {excel_path}")
            self.logger.info(f"  JSON: {json_path}")
            
            return str(excel_path)
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {e}")
            return ""


def run_bma_walkforward_backtest():
    """运行BMA滚动前向回测示例"""
    
    # 配置参数
    config = {
        'initial_capital': 100000,
        'transaction_cost': 0.001,
        'max_positions': 8,
        'rebalance_freq': 'W',
        'training_window_months': 6,
        'min_training_samples': 150
    }
    
    # 股票池
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        'JPM', 'BAC', 'WMT', 'JNJ', 'PG', 'KO', 'DIS', 'HD'
    ]
    
    # 回测时间范围
    start_date = '2020-01-01'
    end_date = '2024-12-01'
    
    print("=== BMA策略滚动前向回测系统 ===")
    print(f"初始资金: ${config['initial_capital']:,}")
    print(f"交易成本: {config['transaction_cost']*100:.2f}%")
    print(f"最大持仓: {config['max_positions']} 只")
    print(f"再平衡频率: {config['rebalance_freq']}")
    print(f"股票池: {len(tickers)} 只股票")
    print(f"回测期间: {start_date} 到 {end_date}")
    
    try:
        # 创建回测器
        backtest = BMAWalkForwardBacktest(**config)
        
        # 运行回测
        print("\n开始运行滚动前向回测...")
        results = backtest.run_walkforward_backtest(tickers, start_date, end_date)
        
        # 显示结果
        if results and 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"\n=== 回测结果 ===")
            print(f"总收益率: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"年化收益率: {metrics.get('annualized_return', 0)*100:.2f}%")
            print(f"年化波动率: {metrics.get('volatility', 0)*100:.2f}%")
            print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"卡尔玛比率: {metrics.get('calmar_ratio', 0):.2f}")
            print(f"胜率: {metrics.get('win_rate', 0)*100:.2f}%")
            print(f"交易次数: {metrics.get('total_trades', 0)}")
            
            # 保存结果
            save_path = backtest.save_results(results)
            print(f"\n结果已保存到: {save_path}")
            
            # 创建可视化
            print("\n生成可视化图表...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_path = f"backtest_results/bma_backtest_visualization_{timestamp}.html"
            backtest.create_performance_visualization(results, html_path)
            
            print("回测完成！")
            
        else:
            print("回测失败，未生成有效结果")
            
    except Exception as e:
        print(f"回测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_bma_walkforward_backtest()