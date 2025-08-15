#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoTrader 回测引擎
集成现has BMA 模型、风险管理、数据库等组件
支持周频策略回测，生成专业级报告
"""

import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import sqlite3
import json

# 添加 items目根目录to路径，以便导入量化模型
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from 量化模型_bma_enhanced import QuantitativeModel, make_target
except ImportError as e:
    logging.warning(f"no法导入量化模型: {e}")
    QuantitativeModel = None

from .database import StockDatabase
# 风险管理功能集成toEnginein
# from .risk_manager import AdvancedRiskManager, RiskMetrics, PositionRisk
from .unified_polygon_factors import sma, rsi, bollinger, zscore, atr
# Bar类需要保留原有定义或从其他模块导入
try:
    from .factors import Bar
except ImportError:
    # 如果无法导入，创建简化的Bar类
    from dataclasses import dataclass
    @dataclass
    class Bar:
        ts: int
        open: float
        high: float
        low: float
        close: float
        volume: float


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str  # 回测starting日期 YYYY-MM-DD
    end_date: str    # 回测结束日期 YYYY-MM-DD
    initial_capital: float = 100000.0  # 初始资金
    rebalance_freq: str = "weekly"     # 调仓频率: daily, weekly, monthly
    max_positions: int = 20            # 最大positions数量
    commission_rate: float = 0.001     # 手续费率
    slippage_rate: float = 0.002       # 滑点率
    benchmark: str = "SPY"             # 基准指数
    
    # BMA 模型参数
    use_bma_model: bool = True
    model_retrain_freq: int = 4        # 模型重训频率（周）
    prediction_horizon: int = 5        # 预测周期（天）
    
    # 风险控制参数
    max_position_weight: float = 0.15  # 单个positions最大权重
    stop_loss_pct: float = 0.08        # 止损比例
    take_profit_pct: float = 0.20      # 止盈比例


@dataclass 
class Position:
    """positions信息"""
    symbol: str
    shares: int
    entry_price: float
    entry_date: datetime
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    unrealized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def weight(self) -> float:
        """in组合in权重（需要总市值计算）"""
        return 0.0  # willin组合层面计算


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    action: str  # BUY, SELL
    shares: int
    price: float
    timestamp: datetime
    commission: float = 0.0
    
    @property
    def gross_amount(self) -> float:
        return self.shares * self.price
    
    @property
    def net_amount(self) -> float:
        multiplier = 1 if self.action == "SELL" else -1
        return multiplier * (self.gross_amount - self.commission)


@dataclass
class Portfolio:
    """投资组合状态"""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    
    @property
    def total_value(self) -> float:
        """组合总价值"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def positions_value(self) -> float:
        return sum(pos.market_value for pos in self.positions.values())
    
    def get_position_weights(self) -> Dict[str, float]:
        """retrievalpositions权重"""
        total = self.total_value
        if total <= 0:
            return {}
        return {symbol: pos.market_value / total for symbol, pos in self.positions.items()}


class BacktestDataManager:
    """回测数据管理器"""
    
    def __init__(self, db_path: str = None):
        self.db = StockDatabase(db_path) if db_path else StockDatabase()
        self.logger = logging.getLogger("BacktestData")
        self.price_cache: Dict[str, pd.DataFrame] = {}
        
    def load_historical_prices(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """from数据库加载历史price数据"""
        try:
            with self.db._get_connection() as conn:
                historical_data = {}
                
                for symbol in symbols:
                    query = """
                    SELECT date, open, high, low, close, volume
                    FROM stock_prices 
                    WHERE symbol = ? AND date BETWEEN ? AND ?
                    ORDER BY date
                    """
                    df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
                    
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        historical_data[symbol] = df
                    else:
                        self.logger.warning(f"no历史数据: {symbol}")
                
                return historical_data
                
        except Exception as e:
            self.logger.error(f"加载历史数据failed: {e}")
            return {}
    
    def get_stock_universe(self, date: str = None) -> List[str]:
        """retrieval股票池"""
        try:
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                
                # retrieval活跃股票列表
                cursor.execute("""
                    SELECT DISTINCT symbol FROM stock_lists 
                    WHERE is_active = 1
                    ORDER BY symbol
                """)
                
                symbols = [row[0] for row in cursor.fetchall()]
                return symbols
                
        except Exception as e:
            self.logger.error(f"retrieval股票池failed: {e}")
            return []
    
    def calculate_technical_factors(self, symbol: str, price_data: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        try:
            df = price_data.copy()
            
            # 转换as Bar for象进行计算
            bars = []
            for idx, row in df.iterrows():
                bar = Bar(
                    timestamp=idx,
                    open=row['open'],
                    high=row['high'], 
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                bars.append(bar)
            
            # 计算技术指标
            df['sma_20'] = sma(bars, 20)
            df['sma_50'] = sma(bars, 50)
            df['rsi_14'] = rsi(bars, 14)
            df['bb_upper'], df['bb_lower'] = bollinger(bars, 20, 2.0)
            df['zscore_20'] = zscore(bars, 20)
            df['atr_14'] = atr(bars, 14)
            
            # price动量因子
            df['ret_1d'] = df['close'].pct_change()
            df['ret_5d'] = df['close'].pct_change(5)
            df['ret_20d'] = df['close'].pct_change(20)
            
            # execution量因子
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # 波动率因子
            df['volatility_20d'] = df['ret_1d'].rolling(20).std()
            
            # 添加股票标识
            df['ticker'] = symbol
            df['date'] = df.index
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算技术因子failed {symbol}: {e}")
            return pd.DataFrame()


class BMASignalGenerator:
    """BMA模型信号生成器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger("BMASignal")
        self.model: Optional[QuantitativeModel] = None
        self.last_train_date: Optional[datetime] = None
        
    def should_retrain_model(self, current_date: datetime) -> bool:
        """判断is否需要重新训练模型"""
        if self.model is None or self.last_train_date is None:
            return True
        
        weeks_since_train = (current_date - self.last_train_date).days // 7
        return weeks_since_train >= self.config.model_retrain_freq
    
    def prepare_training_data(self, historical_data: Dict[str, pd.DataFrame], 
                            current_date: datetime) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
        try:
            all_features = []
            
            # 计算训练窗口（过去1年数据）
            train_start = current_date - timedelta(days=365)
            
            for symbol, df in historical_data.items():
                # 过滤训练期间数据
                train_df = df[(df.index >= train_start) & (df.index < current_date)].copy()
                
                if len(train_df) < 60:  # 数据not足
                    continue
                
                # 计算因子（复use现has因子计算逻辑）
                factor_df = self._calculate_ml_factors(train_df, symbol)
                
                if not factor_df.empty:
                    all_features.append(factor_df)
            
            if not all_features:
                return pd.DataFrame(), pd.Series()
            
            # 合并所has特征
            combined_data = pd.concat(all_features, ignore_index=True)
            
            # 计算目标变量（未来5日收益）
            if QuantitativeModel is not None:
                combined_data = make_target(combined_data, 
                                          horizon=self.config.prediction_horizon,
                                          price_col='close', by='ticker')
            else:
                # 简化目标变量计算
                combined_data['target'] = combined_data.groupby('ticker')['close'].pct_change(5).shift(-5)
            
            # 移除缺失目标值行
            combined_data = combined_data.dropna(subset=['target'])
            
            if combined_data.empty:
                return pd.DataFrame(), pd.Series()
            
            # 分离特征and目标
            feature_cols = [col for col in combined_data.columns 
                          if col not in ['ticker', 'date', 'target', 'close']]
            
            X = combined_data[feature_cols + ['date', 'ticker']]
            y = combined_data['target']
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"准备训练数据failed: {e}")
            return pd.DataFrame(), pd.Series()
    
    def _calculate_ml_factors(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """计算机器学习因子（简化版）"""
        try:
            factor_df = df.copy()
            
            # price因子
            factor_df['momentum_5'] = factor_df['close'].pct_change(5)
            factor_df['momentum_20'] = factor_df['close'].pct_change(20)
            factor_df['mean_reversion_5'] = -factor_df['momentum_5']  # 反转因子
            
            # 技术指标因子
            factor_df['sma_ratio_20'] = factor_df['close'] / factor_df['close'].rolling(20).mean() - 1
            factor_df['sma_ratio_50'] = factor_df['close'] / factor_df['close'].rolling(50).mean() - 1
            
            # RSI标准化
            rsi_values = []
            for i in range(len(factor_df)):
                if i >= 14:
                    price_slice = factor_df['close'].iloc[i-14:i+1]
                    delta = price_slice.diff()
                    gain = delta.where(delta > 0, 0).mean()
                    loss = -delta.where(delta < 0, 0).mean()
                    if loss != 0:
                        rs = gain / loss
                        rsi_val = 100 - (100 / (1 + rs))
                    else:
                        rsi_val = 100
                    rsi_values.append((rsi_val - 50) / 50)  # 标准化to[-1,1]
                else:
                    rsi_values.append(0)
            
            factor_df['rsi_normalized'] = rsi_values
            
            # execution量因子
            factor_df['volume_ratio'] = (factor_df['volume'] / 
                                       factor_df['volume'].rolling(20).mean()).fillna(1)
            
            # 波动率因子
            factor_df['volatility'] = factor_df['close'].pct_change().rolling(20).std()
            
            # 添加元数据
            factor_df['ticker'] = symbol
            factor_df['date'] = factor_df.index
            
            # 移除NaN行
            factor_df = factor_df.dropna()
            
            return factor_df
            
        except Exception as e:
            self.logger.error(f"计算ML因子failed: {e}")
            return pd.DataFrame()
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, current_date: datetime) -> bool:
        """训练BMA模型"""
        try:
            if QuantitativeModel is None:
                self.logger.warning("BMA模型notcanuse，使use简化信号")
                return False
            
            self.model = QuantitativeModel()
            
            # 提取日期andticker信息
            dates = pd.to_datetime(X['date']) if 'date' in X.columns else None
            tickers = X['ticker'] if 'ticker' in X.columns else None
            
            # 训练模型
            result = self.model.train_models_with_bma(
                X=X, y=y,
                enable_hyperopt=False,
                apply_preprocessing=True,
                dates=dates,
                tickers=tickers
            )
            
            if result and 'BMA' in result:
                self.last_train_date = current_date
                self.logger.info(f"BMA模型训练completed: {result['BMA']}")
                return True
            else:
                self.logger.warning("BMA模型训练failed")
                return False
                
        except Exception as e:
            self.logger.error(f"训练模型failed: {e}")
            return False
    
    def generate_signals(self, current_data: Dict[str, pd.DataFrame], 
                        current_date: datetime) -> Dict[str, float]:
        """生execution易信号"""
        try:
            if self.model is None:
                return self._generate_simple_signals(current_data, current_date)
            
            # 准备当before数据useat预测
            prediction_features = []
            symbols = []
            
            for symbol, df in current_data.items():
                if len(df) < 20:  # 数据not足
                    continue
                
                # retrieval最新数据
                latest_data = df.tail(1).copy()
                factor_df = self._calculate_ml_factors(df.tail(60), symbol)
                
                if not factor_df.empty:
                    latest_factors = factor_df.tail(1)
                    prediction_features.append(latest_factors)
                    symbols.append(symbol)
            
            if not prediction_features:
                return {}
            
            # 合并预测特征
            pred_data = pd.concat(prediction_features, ignore_index=True)
            
            # 生成预测
            predictions = self.model.predict_with_bma(pred_data)
            
            # 转换as信号字典
            signals = {}
            for i, symbol in enumerate(symbols):
                if i < len(predictions):
                    signals[symbol] = float(predictions[i])
            
            return signals
            
        except Exception as e:
            self.logger.error(f"生成信号failed: {e}")
            return {}
    
    def _generate_simple_signals(self, current_data: Dict[str, pd.DataFrame], 
                                current_date: datetime) -> Dict[str, float]:
        """生成简化信号（当BMAnotcanusewhen）"""
        signals = {}
        
        for symbol, df in current_data.items():
            if len(df) < 20:
                continue
            
            try:
                # 简单动量+反转组合信号
                latest = df.tail(20)
                
                # 短期动量（5日）
                momentum_5 = latest['close'].iloc[-1] / latest['close'].iloc[-6] - 1
                
                # in期均线位置
                sma_20 = latest['close'].rolling(20).mean().iloc[-1]
                price_vs_sma = latest['close'].iloc[-1] / sma_20 - 1
                
                # RSI反转信号
                price_changes = latest['close'].pct_change().dropna()
                if len(price_changes) >= 14:
                    gains = price_changes.where(price_changes > 0, 0)
                    losses = -price_changes.where(price_changes < 0, 0)
                    avg_gain = gains.rolling(14).mean().iloc[-1]
                    avg_loss = losses.rolling(14).mean().iloc[-1]
                    
                    if avg_loss != 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        rsi_signal = -(rsi - 50) / 50  # RSI反转信号
                    else:
                        rsi_signal = 0
                else:
                    rsi_signal = 0
                
                # 组合信号
                signal = (0.4 * momentum_5 + 0.3 * price_vs_sma + 0.3 * rsi_signal)
                signals[symbol] = signal
                
            except Exception as e:
                self.logger.warning(f"计算简化信号failed {symbol}: {e}")
                continue
        
        return signals


class AutoTraderBacktestEngine:
    """AutoTrader回测引擎主类"""
    
    def __init__(self, config: BacktestConfig, db_path: str = None):
        self.config = config
        self.logger = logging.getLogger("BacktestEngine")
        
        # 初始化组件
        self.data_manager = BacktestDataManager(db_path)
        self.signal_generator = BMASignalGenerator(config)
        self.portfolio = Portfolio(cash=config.initial_capital)
        
        # 回测状态
        self.current_date: Optional[datetime] = None
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.performance_history: List[Dict] = []
        self.daily_returns: List[float] = []
        
        # 风险管理（简化版，not依赖IBKRconnection）
        self.risk_limits = {
            'max_position_weight': config.max_position_weight,
            'stop_loss_pct': config.stop_loss_pct,
            'take_profit_pct': config.take_profit_pct
        }
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行完整回测"""
        self.logger.info(f"starting回测: {self.config.start_date} -> {self.config.end_date}")
        
        # 1. 加载数据
        symbols = self.data_manager.get_stock_universe()
        if not symbols:
            raise ValueError("no法retrieval股票池")
        
        self.historical_data = self.data_manager.load_historical_prices(
            symbols, self.config.start_date, self.config.end_date
        )
        
        if not self.historical_data:
            raise ValueError("no法加载历史数据")
        
        # 2. 生execution易日历
        trading_dates = self._generate_trading_calendar()
        
        # 3. 主回测循环
        for date in trading_dates:
            self.current_date = date
            self._run_daily_step(date)
        
        # 4. 生成回测报告
        results = self._generate_backtest_report()
        
        self.logger.info("回测completed")
        return results
    
    def _generate_trading_calendar(self) -> List[datetime]:
        """生execution易日历"""
        start = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.config.end_date, '%Y-%m-%d')
        
        dates = []
        current = start
        
        if self.config.rebalance_freq == "daily":
            while current <= end:
                if current.weekday() < 5:  # 周一to周五
                    dates.append(current)
                current += timedelta(days=1)
        
        elif self.config.rebalance_freq == "weekly":
            # 每周五调仓
            while current <= end:
                if current.weekday() == 4:  # 周五
                    dates.append(current)
                current += timedelta(days=1)
        
        return dates
    
    def _run_daily_step(self, date: datetime):
        """执行单日回测步骤"""
        try:
            # 1. updatespositionsmarket
            self._update_portfolio_prices(date)
            
            # 2. check止损止盈
            self._check_risk_exits(date)
            
            # 3. 生成新信号（根据调仓频率）
            if self._should_rebalance(date):
                signals = self._generate_rebalance_signals(date)
                if signals:
                    self._execute_rebalance(signals, date)
            
            # 4. 记录每日表现
            self._record_daily_performance(date)
            
        except Exception as e:
            self.logger.error(f"日步骤执行failed {date}: {e}")
    
    def _update_portfolio_prices(self, date: datetime):
        """updatespositionsprice"""
        for symbol, position in self.portfolio.positions.items():
            if symbol in self.historical_data:
                price_data = self.historical_data[symbol]
                try:
                    # retrieval当日收盘价
                    if date.strftime('%Y-%m-%d') in price_data.index.strftime('%Y-%m-%d'):
                        current_price = price_data.loc[price_data.index.date == date.date(), 'close'].iloc[0]
                        position.current_price = current_price
                        position.unrealized_pnl = (current_price - position.entry_price) * position.shares
                except Exception as e:
                    self.logger.warning(f"updatespricefailed {symbol}: {e}")
    
    def _check_risk_exits(self, date: datetime):
        """check风险退出 records件"""
        positions_to_close = []
        
        for symbol, position in self.portfolio.positions.items():
            if position.current_price <= 0:
                continue
            
            # 计算收益率
            return_pct = (position.current_price - position.entry_price) / position.entry_price
            
            # check止损
            if return_pct <= -self.risk_limits['stop_loss_pct']:
                positions_to_close.append((symbol, "STOP_LOSS"))
                
            # check止盈
            elif return_pct >= self.risk_limits['take_profit_pct']:
                positions_to_close.append((symbol, "TAKE_PROFIT"))
        
        # 执行平仓
        for symbol, reason in positions_to_close:
            self._close_position(symbol, date, reason)
    
    def _should_rebalance(self, date: datetime) -> bool:
        """判断is否应该调仓"""
        if self.config.rebalance_freq == "daily":
            return True
        elif self.config.rebalance_freq == "weekly":
            return date.weekday() == 4  # 周五
        return False
    
    def _generate_rebalance_signals(self, date: datetime) -> Dict[str, float]:
        """生成调仓信号"""
        # 1. checkis否需要重训模型
        if self.signal_generator.should_retrain_model(date):
            self.logger.info(f"重新训练BMA模型: {date}")
            
            # 准备训练数据
            X, y = self.signal_generator.prepare_training_data(self.historical_data, date)
            
            if not X.empty and not y.empty:
                self.signal_generator.train_model(X, y, date)
        
        # 2. retrieval当before数据窗口
        current_data = {}
        for symbol, df in self.historical_data.items():
            # retrievalto当before日期as止数据（避免before瞻偏差）
            available_data = df[df.index.date <= date.date()]
            if len(available_data) >= 60:  # 确保has足够历史数据
                current_data[symbol] = available_data
        
        # 3. 生成信号
        signals = self.signal_generator.generate_signals(current_data, date)
        
        return signals
    
    def _execute_rebalance(self, signals: Dict[str, float], date: datetime):
        """执行调仓"""
        try:
            # 1. 选择Top N信号
            sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
            top_signals = sorted_signals[:self.config.max_positions]
            
            # 2. 确定目标positions
            target_symbols = {symbol for symbol, _ in top_signals if signal > 0}
            current_symbols = set(self.portfolio.positions.keys())
            
            # 3. 卖出notin目标列表positions
            to_sell = current_symbols - target_symbols
            for symbol in to_sell:
                self._close_position(symbol, date, "REBALANCE")
            
            # 4. 计算新positions权重（等权重）
            if target_symbols:
                target_weight = 1.0 / len(target_symbols)
                target_value_per_position = self.portfolio.total_value * target_weight
                
                # 5. 买入or调整positions
                for symbol in target_symbols:
                    self._adjust_position(symbol, target_value_per_position, date)
            
            self.logger.info(f"调仓completed: {len(target_symbols)} 个positions")
            
        except Exception as e:
            self.logger.error(f"执行调仓failed: {e}")
    
    def _close_position(self, symbol: str, date: datetime, reason: str = ""):
        """平仓"""
        if symbol not in self.portfolio.positions:
            return
        
        position = self.portfolio.positions[symbol]
        
        if position.current_price <= 0:
            self.logger.warning(f"no效price，no法平仓: {symbol}")
            return
        
        # 计算手续费and滑点
        gross_proceeds = position.shares * position.current_price
        commission = gross_proceeds * self.config.commission_rate
        slippage = gross_proceeds * self.config.slippage_rate
        net_proceeds = gross_proceeds - commission - slippage
        
        # 创建交易记录
        trade = Trade(
            symbol=symbol,
            action="SELL",
            shares=position.shares,
            price=position.current_price,
            timestamp=date,
            commission=commission
        )
        
        self.portfolio.trades.append(trade)
        self.portfolio.cash += net_proceeds
        
        # 移除positions
        del self.portfolio.positions[symbol]
        
        self.logger.debug(f"平仓 {symbol}: {position.shares}股 @ ${position.current_price:.2f} ({reason})")
    
    def _adjust_position(self, symbol: str, target_value: float, date: datetime):
        """调整positionsto目标价值"""
        if symbol not in self.historical_data:
            return
        
        # retrieval当beforeprice
        price_data = self.historical_data[symbol]
        try:
            current_price = price_data.loc[price_data.index.date == date.date(), 'close'].iloc[0]
        except:
            self.logger.warning(f"no法retrievalprice: {symbol}")
            return
        
        if current_price <= 0:
            return
        
        # 计算目标股数
        target_shares = int(target_value / current_price)
        
        if target_shares <= 0:
            return
        
        # checkis否haspositions
        current_shares = 0
        if symbol in self.portfolio.positions:
            current_shares = self.portfolio.positions[symbol].shares
        
        shares_to_trade = target_shares - current_shares
        
        if shares_to_trade > 0:
            # 买入
            cost = shares_to_trade * current_price
            commission = cost * self.config.commission_rate
            slippage = cost * self.config.slippage_rate
            total_cost = cost + commission + slippage
            
            if total_cost <= self.portfolio.cash:
                # 执行买入
                trade = Trade(
                    symbol=symbol,
                    action="BUY", 
                    shares=shares_to_trade,
                    price=current_price,
                    timestamp=date,
                    commission=commission
                )
                
                self.portfolio.trades.append(trade)
                self.portfolio.cash -= total_cost
                
                # updatespositions
                if symbol in self.portfolio.positions:
                    # 加仓
                    old_pos = self.portfolio.positions[symbol]
                    total_shares = old_pos.shares + shares_to_trade
                    avg_price = ((old_pos.shares * old_pos.entry_price) + 
                               (shares_to_trade * current_price)) / total_shares
                    
                    self.portfolio.positions[symbol] = Position(
                        symbol=symbol,
                        shares=total_shares,
                        entry_price=avg_price,
                        entry_date=old_pos.entry_date,
                        current_price=current_price
                    )
                else:
                    # 新建positions
                    self.portfolio.positions[symbol] = Position(
                        symbol=symbol,
                        shares=shares_to_trade,
                        entry_price=current_price,
                        entry_date=date,
                        current_price=current_price
                    )
                
                self.logger.debug(f"买入 {symbol}: {shares_to_trade}股 @ ${current_price:.2f}")
        
        elif shares_to_trade < 0:
            # 减仓
            shares_to_sell = -shares_to_trade
            if symbol in self.portfolio.positions:
                position = self.portfolio.positions[symbol]
                if shares_to_sell >= position.shares:
                    # 全部卖出
                    self._close_position(symbol, date, "REDUCE")
                else:
                    # 部分卖出
                    proceeds = shares_to_sell * current_price
                    commission = proceeds * self.config.commission_rate
                    slippage = proceeds * self.config.slippage_rate
                    net_proceeds = proceeds - commission - slippage
                    
                    trade = Trade(
                        symbol=symbol,
                        action="SELL",
                        shares=shares_to_sell, 
                        price=current_price,
                        timestamp=date,
                        commission=commission
                    )
                    
                    self.portfolio.trades.append(trade)
                    self.portfolio.cash += net_proceeds
                    
                    # updatespositions
                    position.shares -= shares_to_sell
                    position.current_price = current_price
                    
                    self.logger.debug(f"减仓 {symbol}: {shares_to_sell}股 @ ${current_price:.2f}")
    
    def _record_daily_performance(self, date: datetime):
        """记录每日表现"""
        total_value = self.portfolio.total_value
        
        # 计算日收益率
        if self.performance_history:
            prev_value = self.performance_history[-1]['total_value']
            daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0
        else:
            daily_return = (total_value - self.config.initial_capital) / self.config.initial_capital
        
        self.daily_returns.append(daily_return)
        
        # 记录详细信息
        performance = {
            'date': date,
            'total_value': total_value,
            'cash': self.portfolio.cash,
            'positions_value': self.portfolio.positions_value,
            'daily_return': daily_return,
            'num_positions': len(self.portfolio.positions),
            'positions': dict(self.portfolio.get_position_weights())
        }
        
        self.performance_history.append(performance)
    
    def _generate_backtest_report(self) -> Dict[str, Any]:
        """生成回测报告"""
        if not self.performance_history:
            return {}
        
        # 转换asDataFrame便at计算
        df = pd.DataFrame(self.performance_history)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 基础指标
        initial_value = self.config.initial_capital
        final_value = df['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # 年化收益率
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        annual_return = (final_value / initial_value) ** (1/years) - 1 if years > 0 else 0
        
        # 波动率and夏普比率
        returns = pd.Series(self.daily_returns)
        annual_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # 胜率
        win_rate = (returns > 0).mean()
        
        # 交易统计
        total_trades = len(self.portfolio.trades)
        trade_df = pd.DataFrame([{
            'symbol': t.symbol,
            'action': t.action, 
            'shares': t.shares,
            'price': t.price,
            'timestamp': t.timestamp,
            'gross_amount': t.gross_amount,
            'commission': t.commission
        } for t in self.portfolio.trades])
        
        # 构建报告
        report = {
            'period': {
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'trading_days': len(df)
            },
            'returns': {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            },
            'portfolio': {
                'initial_capital': initial_value,
                'final_value': final_value,
                'final_cash': self.portfolio.cash,
                'final_positions_value': self.portfolio.positions_value
            },
            'trading': {
                'total_trades': total_trades,
                'avg_trades_per_day': total_trades / len(df) if len(df) > 0 else 0
            },
            'detailed_performance': df.to_dict('records'),
            'trades': trade_df.to_dict('records') if not trade_df.empty else []
        }
        
        return report


def create_sample_config() -> BacktestConfig:
    """创建示例配置"""
    return BacktestConfig(
        start_date="2022-01-01",
        end_date="2023-12-31", 
        initial_capital=100000.0,
        rebalance_freq="weekly",
        max_positions=20,
        commission_rate=0.001,
        slippage_rate=0.002,
        use_bma_model=True,
        model_retrain_freq=4,
        prediction_horizon=5
    )


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # 创建and运行回测
    config = create_sample_config()
    engine = AutoTraderBacktestEngine(config)
    
    try:
        results = engine.run_backtest()
        
        # 打印结果summary
        print("\n" + "="*50)
        print("回测结果summary")
        print("="*50)
        print(f"回测期间: {results['period']['start_date']} -> {results['period']['end_date']}")
        print(f"总收益率: {results['returns']['total_return']:.2%}")
        print(f"年化收益率: {results['returns']['annual_return']:.2%}")
        print(f"年化波动率: {results['returns']['annual_volatility']:.2%}")
        print(f"夏普比率: {results['returns']['sharpe_ratio']:.3f}")
        print(f"最大回撤: {results['returns']['max_drawdown']:.2%}")
        print(f"胜率: {results['returns']['win_rate']:.2%}")
        print(f"总交易次数: {results['trading']['total_trades']}")
        print(f"最终资产: ${results['portfolio']['final_value']:,.2f}")
        
    except Exception as e:
        logging.error(f"回测failed: {e}")
        import traceback
        traceback.print_exc()


# =================== 回测start功能 (合并自run_backtest.py) ===================

def setup_logging(level: str = "INFO") -> None:
    """配置日志系统"""
    import os
    import sys
    from datetime import datetime
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # 创建logs目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # 配置日志格式
    log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    
    # 文件日志
    log_file = os.path.join(logs_dir, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # settingsmatplotlib日志级别（避免过多输出）
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def create_backtest_config_from_args(args) -> BacktestConfig:
    """根据命令行参数创建回测配置"""
    return BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        rebalance_freq=args.rebalance_freq,
        max_positions=args.max_positions,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        benchmark=args.benchmark,
        use_bma_model=args.use_bma_model,
        model_retrain_freq=args.model_retrain_freq,
        prediction_horizon=args.prediction_horizon,
        max_position_weight=args.max_position_weight,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct
    )


def run_backtest_with_config(config: BacktestConfig, db_path: str = None) -> Dict[str, Any]:
    """使use指定配置运行回测"""
    from datetime import datetime
    
    logger = logging.getLogger("BacktestRunner")
    
    logger.info("="*60)
    logger.info("AutoTrader 回测系统start")
    logger.info("="*60)
    logger.info(f"回测期间: {config.start_date} -> {config.end_date}")
    logger.info(f"初始资金: ${config.initial_capital:,.2f}")
    logger.info(f"调仓频率: {config.rebalance_freq}")
    logger.info(f"最大positions: {config.max_positions}")
    logger.info(f"使useBMA模型: {config.use_bma_model}")
    logger.info(f"手续费率: {config.commission_rate:.3%}")
    logger.info(f"滑点率: {config.slippage_rate:.3%}")
    logger.info("="*60)
    
    try:
        # 创建回测引擎
        engine = AutoTraderBacktestEngine(config, db_path)
        
        # 运行回测
        start_time = datetime.now()
        logger.info("starting执行回测...")
        
        results = engine.run_backtest()
        
        end_time = datetime.now()
        elapsed = end_time - start_time
        
        logger.info(f"回测completed! 耗when: {elapsed}")
        logger.info("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"回测执行failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_preset_backtests():
    """运行预设回测配置"""
    logger = logging.getLogger("PresetBacktests")
    
    presets = [
        {
            "name": "短期回测 (2023年)",
            "config": BacktestConfig(
                start_date="2023-01-01",
                end_date="2023-12-31",
                initial_capital=100000,
                max_positions=15,
                rebalance_freq="weekly"
            )
        },
        {
            "name": "in期回测 (2022-2023)",
            "config": BacktestConfig(
                start_date="2022-01-01", 
                end_date="2023-12-31",
                initial_capital=100000,
                max_positions=20,
                rebalance_freq="weekly"
            )
        }
    ]
    
    for preset in presets:
        logger.info(f"\n starting运行: {preset['name']}")
        results = run_backtest_with_config(preset['config'])
        
        if results:
            logger.info(f" {preset['name']} completed:")
            logger.info(f"   最终净值: ${results.get('final_portfolio_value', 0):,.2f}")
            logger.info(f"   总收益率: {results.get('total_return', 0):.2%}")
            logger.info(f"   夏普比率: {results.get('sharpe_ratio', 'N/A')}")
            logger.info(f"   最大回撤: {results.get('max_drawdown', 0):.2%}")


def main_backtest():
    """主函数 - 合并原run_backtest.py功能"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AutoTrader 专业级回测系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 回测期间
    parser.add_argument("--start-date", type=str, default="2022-01-01",
                       help="回测starting日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2023-12-31",
                       help="回测结束日期 (YYYY-MM-DD)")
    
    # 资金andpositions
    parser.add_argument("--initial-capital", type=float, default=100000.0,
                       help="初始资金")
    parser.add_argument("--max-positions", type=int, default=20,
                       help="最大positions数量")
    parser.add_argument("--rebalance-freq", choices=["daily", "weekly"], default="weekly",
                       help="调仓频率")
    
    # 交易成本
    parser.add_argument("--commission-rate", type=float, default=0.001,
                       help="手续费率")
    parser.add_argument("--slippage-rate", type=float, default=0.002,
                       help="滑点率")
    
    # 基准
    parser.add_argument("--benchmark", type=str, default="SPY",
                       help="基准指数")
    
    # BMA模型参数
    parser.add_argument("--use-bma-model", action="store_true", default=True,
                       help="is否使useBMA模型")
    parser.add_argument("--model-retrain-freq", type=int, default=4,
                       help="模型重训频率（周）")
    parser.add_argument("--prediction-horizon", type=int, default=5,
                       help="预测周期（天）")
    parser.add_argument("--max-position-weight", type=float, default=0.15,
                       help="单个positions最大权重")
    
    # 风险控制
    parser.add_argument("--stop-loss-pct", type=float, default=0.08,
                       help="止损百分比")
    parser.add_argument("--take-profit-pct", type=float, default=0.20,
                       help="止盈百分比")
    
    # 其他选 items
    parser.add_argument("--db-path", type=str, default=None,
                       help="数据库路径")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="日志级别")
    parser.add_argument("--preset", action="store_true",
                       help="运行预设回测配置")
    parser.add_argument("--analyze", action="store_true", default=True,
                       help="运行回测分析")
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.log_level)
    
    if args.preset:
        # 运行预设回测
        run_preset_backtests()
    else:
        # 单次回测
        config = create_backtest_config_from_args(args)
        results = run_backtest_with_config(config, args.db_path)
        
        # 分析结果
        if results and args.analyze:
            try:
                from .backtest_analyzer import analyze_backtest_results
                analyze_backtest_results(results)
            except ImportError:
                logging.warning("no法导入backtest_analyzer，跳过详细分析")


if __name__ == "__main__":
    # can以直接运行回测
    try:
        main_backtest()
    except KeyboardInterrupt:
        print("\n回测beuse户in断")
    except Exception as e:
        print(f"回测执行出错: {e}")
        import traceback
        traceback.print_exc()
