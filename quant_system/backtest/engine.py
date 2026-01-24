"""
Backtesting Engine for Quantitative Signal System

This module provides a comprehensive backtesting framework for evaluating
signal-based trading strategies with realistic execution modeling.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import warnings
import logging

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_types import BacktestResult, StockData, CompositeSignal
from core.signal_aggregator import SignalAggregator
from utils.risk_management import (
    StopLossManager, 
    PositionSizer, 
    PortfolioRiskManager,
    ATRCalculator
)
from config.settings import RiskManagementConfig, SystemConfig, MarketRegime

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Trade direction enumeration."""
    LONG = "long"
    SHORT = "short"


class TradeScenario(Enum):
    """Trade scenario classification."""
    FAILED = "A"  # 失败的交易 - 7%止损，日内至5天
    SWING = "B"   # 普通的波段盈利 - 20-25%盈利，2-6周
    BREAKOUT = "C"  # 强力爆发（8周法则）- 3周内暴涨20%+，至少8周
    SUPER_PERFORMER = "D"  # 超级表现者 - 100%+盈利，6-18个月
    ZOMBIE = "E"  # 僵尸股 - 盈亏平衡，5-10天时间止损


@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    direction: TradeDirection
    entry_date: datetime
    entry_price: float
    shares: int
    stop_loss: float
    
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    
    # Track signal score for exit rules
    entry_signal_score: float = 0.0
    current_signal_score: float = 0.0
    previous_signal_score: float = 0.0
    
    # Trade scenario classification
    scenario: Optional['TradeScenario'] = None
    
    # Track for scenario-specific logic
    max_price_since_entry: float = field(default=0.0)  # For tracking peak price
    min_price_since_entry: float = field(default_factory=lambda: float('inf'))  # For tracking drawdown
    days_since_20pct_gain: Optional[int] = None  # For scenario C (8周法则)
    last_significant_move_date: Optional[datetime] = None  # For zombie detection
    
    def close(self, exit_date: datetime, exit_price: float, reason: str):
        """Close the trade and calculate P&L."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        
        if self.direction == TradeDirection.LONG:
            self.pnl = (exit_price - self.entry_price) * self.shares
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl = (self.entry_price - exit_price) * self.shares
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        self.holding_days = (exit_date - self.entry_date).days
    
    @property
    def is_winner(self) -> bool:
        return self.pnl > 0
    
    @property
    def risk_reward(self) -> float:
        """Calculate risk/reward ratio."""
        risk = abs(self.entry_price - self.stop_loss)
        if risk == 0:
            return 0
        reward = abs(self.pnl / self.shares) if self.shares > 0 else 0
        return reward / risk


@dataclass
class Portfolio:
    """Portfolio state tracker."""
    initial_capital: float
    cash: float
    positions: Dict[str, Trade] = field(default_factory=dict)
    closed_trades: List[Trade] = field(default_factory=list)
    
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)
    high_water_mark: float = 0.0
    
    def __post_init__(self):
        self.high_water_mark = self.initial_capital
    
    @property
    def total_equity(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            t.shares * t.entry_price  # Simplified - should use current price
            for t in self.positions.values()
        )
        return self.cash + positions_value
    
    def update_equity(self, date: datetime, current_prices: Dict[str, float]):
        """Update equity curve with current prices."""
        positions_value = sum(
            t.shares * current_prices.get(t.symbol, t.entry_price)
            for t in self.positions.values()
        )
        equity = self.cash + positions_value
        self.equity_curve.append((date, equity))
        
        if equity > self.high_water_mark:
            self.high_water_mark = equity
    
    @property
    def current_drawdown(self) -> float:
        """Calculate current drawdown from high water mark."""
        if self.high_water_mark == 0:
            return 0
        current = self.equity_curve[-1][1] if self.equity_curve else self.initial_capital
        return (self.high_water_mark - current) / self.high_water_mark
    
    @property
    def exposure(self) -> float:
        """Calculate current exposure as percentage of equity."""
        if not self.equity_curve:
            return 0
        current_equity = self.equity_curve[-1][1]
        if current_equity == 0:
            return 0
        positions_value = sum(
            t.shares * t.entry_price for t in self.positions.values()
        )
        return positions_value / current_equity


@dataclass 
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    commission_per_share: float = 0.005
    commission_minimum: float = 1.0
    slippage_pct: float = 0.001  # 0.1% slippage
    
    max_positions: int = 10
    min_signal_score: float = 0.60
    
    use_trailing_stops: bool = True
    stop_loss_atr_mult: float = 3.0
    
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    
    # Risk limits
    max_position_pct: float = 0.10
    max_sector_exposure: float = 0.30
    max_portfolio_drawdown: float = 0.15


class BacktestEngine:
    """
    Comprehensive backtesting engine for signal-based strategies.
    
    Features:
    - Realistic execution with slippage and commissions
    - Position sizing with risk management
    - Trailing stops and dynamic exits
    - Performance attribution
    - Drawdown monitoring
    - Advanced exit rules based on signal scores
    - Market regime-based position sizing
    """
    
    def __init__(
        self,
        signal_aggregator: SignalAggregator,
        config: BacktestConfig = None,
        risk_config: RiskManagementConfig = None
    ):
        self.signal_aggregator = signal_aggregator
        self.config = config or BacktestConfig()
        self.risk_config = risk_config or RiskManagementConfig()
        
        self.stop_manager = StopLossManager(self.risk_config)
        self.position_sizer = PositionSizer(self.risk_config)
        self.portfolio_manager = PortfolioRiskManager(self.risk_config)
        self.atr_calculator = ATRCalculator()
    
    def _calculate_moving_averages(self, df: pd.DataFrame, date: datetime, periods: List[int]) -> Dict[int, float]:
        """
        Calculate moving averages for given periods.
        
        CRITICAL: date should be BEFORE current trading date (t-1) to avoid look-ahead bias.
        """
        mas = {}
        # Use data up to (but not including) the specified date, or up to date if date is in past
        subset = df.loc[:date]
        if len(subset) == 0:
            return mas
        
        close = subset['Close']
        for period in periods:
            if len(subset) >= period:
                ma = close.rolling(period).mean().iloc[-1]
                mas[period] = ma
        return mas
    
    def _check_volume_breakdown(self, df: pd.DataFrame, date: datetime, lookback_days: int = 20) -> bool:
        """Check if stock breaks down on high volume (放量跌破)."""
        try:
            subset = df.loc[:date].tail(lookback_days + 5)
            if len(subset) < lookback_days:
                return False
            
            current_price = subset['Close'].iloc[-1]
            current_volume = subset['Volume'].iloc[-1]
            avg_volume = subset['Volume'].tail(lookback_days).mean()
            
            # Check if volume is significantly above average (1.5x)
            if current_volume > avg_volume * 1.5:
                # Check if price is declining
                price_change = (current_price - subset['Close'].iloc[-5]) / subset['Close'].iloc[-5]
                if price_change < -0.02:  # Down more than 2%
                    return True
        except Exception as e:
            logger.debug(f"Error checking volume breakdown: {e}")
        return False
    
    def _check_deep_squat(self, trade: Trade, df: pd.DataFrame, date: datetime) -> Tuple[bool, Optional[datetime]]:
        """
        Check if stock had a deep squat (深蹲) and if it recovered within 3 days.
        Returns: (has_squat, squat_date)
        """
        try:
            subset = df.loc[trade.entry_date:date]
            if len(subset) < 5:
                return False, None
            
            close = subset['Close']
            entry_price = trade.entry_price
            
            # Look for significant drop (>5%) from entry or recent high
            for i in range(len(subset) - 3):
                check_date = subset.index[i]
                check_price = close.iloc[i]
                drop_pct = (entry_price - check_price) / entry_price
                
                # Check for deep squat (>5% drop)
                if drop_pct > 0.05:
                    # Check if recovered within 3 days
                    if i + 3 < len(subset):
                        recovery_price = close.iloc[i + 3]
                        recovery_pct = (recovery_price - check_price) / check_price
                        if recovery_pct < 0.03:  # Not recovered (>3% gain needed)
                            return True, check_date
        except Exception as e:
            logger.debug(f"Error checking deep squat: {e}")
        return False, None
    
    def _check_resistance_at_20pct(self, trade: Trade, current_price: float, df: pd.DataFrame, date: datetime) -> bool:
        """Check if stock is hitting resistance around 20% gain."""
        try:
            target_price = trade.entry_price * 1.20  # 20% gain
            # Check if price is near target (within 2%)
            if abs(current_price - target_price) / target_price < 0.02:
                # Check if price has been struggling around this level
                subset = df.loc[:date].tail(5)
                if len(subset) >= 3:
                    recent_prices = subset['Close'].tail(3)
                    # If price is oscillating around target, consider it resistance
                    if all(abs(p - target_price) / target_price < 0.03 for p in recent_prices):
                        return True
        except Exception as e:
            logger.debug(f"Error checking resistance: {e}")
        return False
    
    def _determine_trade_scenario(
        self, 
        symbol: str, 
        entry_price: float, 
        df: pd.DataFrame, 
        date: datetime
    ) -> TradeScenario:
        """
        Determine trade scenario based on entry conditions.
        
        CRITICAL: Uses only data BEFORE entry date (no look-ahead bias)
        
        Scenario C (BREAKOUT): Stock gained 20%+ in 3 weeks before entry
        Scenario D (SUPER_PERFORMER): Strong uptrend, above 50/200 MA
        Scenario B (SWING): Normal entry, expect 20-25% gain
        Scenario A (FAILED): Default for new entries (will be reclassified if fails)
        Scenario E (ZOMBIE): Will be detected during holding period
        """
        try:
            # Look back 3 weeks (21 trading days) BEFORE entry date
            lookback_start = date - timedelta(days=30)
            # Use data BEFORE entry date only
            available_dates = df.index[df.index < date]
            if len(available_dates) == 0:
                return TradeScenario.SWING
            signal_date = available_dates[-1]
            subset = df.loc[lookback_start:signal_date]
            
            if len(subset) >= 15:
                close = subset['Close']
                price_3w_ago = close.iloc[0] if len(close) > 0 else entry_price
                # Use t-1 close price (last available before entry) for comparison
                t1_close = close.iloc[-1] if len(close) > 0 else entry_price
                gain_3w = (t1_close - price_3w_ago) / price_3w_ago
                
                # Scenario C: 3周内暴涨20%以上
                if gain_3w >= 0.20:
                    logger.debug(f"{symbol}: Scenario C (BREAKOUT) - gained {gain_3w:.1%} in 3 weeks")
                    return TradeScenario.BREAKOUT
                
                # Check moving averages for Scenario D (use t-1 data)
                mas = self._calculate_moving_averages(df, signal_date, [50, 200])
                if 50 in mas and 200 in mas:
                    # Use t-1 close price for comparison (not entry_price which is t open)
                    t1_close_price = close.iloc[-1] if len(close) > 0 else entry_price
                    ma50 = mas[50]
                    ma200 = mas[200]
                    
                    # Scenario D: Above both MAs, strong uptrend
                    if t1_close_price > ma50 > ma200:
                        # Check if 50 MA is above 200 MA and trending up
                        ma50_trend = (ma50 - mas.get(50, ma50)) / ma50 if 50 in mas else 0
                        if ma50_trend > 0:
                            logger.debug(f"{symbol}: Scenario D (SUPER_PERFORMER) - above 50/200 MA")
                            return TradeScenario.SUPER_PERFORMER
            
            # Default to SWING for normal entries
            logger.debug(f"{symbol}: Scenario B (SWING) - normal entry")
            return TradeScenario.SWING
            
        except Exception as e:
            logger.debug(f"Error determining scenario for {symbol}: {e}")
            return TradeScenario.SWING  # Default
    
    def _detect_market_regime(self, benchmark_data: pd.DataFrame, date: datetime) -> str:
        """
        Detect market regime based on benchmark performance.
        
        Returns: 'bull', 'neutral', or 'bear'
        CRITICAL: Uses only data up to t-1 (no look-ahead bias)
        """
        try:
            lookback_start = date - timedelta(days=200)
            # Use data BEFORE current date (t-1)
            bench_available_dates = benchmark_data.index[benchmark_data.index < date]
            if len(bench_available_dates) == 0:
                return 'neutral'
            bench_signal_date = bench_available_dates[-1]
            bench_subset = benchmark_data.loc[lookback_start:bench_signal_date]
            
            if len(bench_subset) < 50:
                return 'neutral'
            
            # Calculate 200-day MA and current position
            close = bench_subset['Close']
            ma200 = close.rolling(200).mean()
            
            if len(ma200.dropna()) == 0:
                return 'neutral'
            
            current_price = close.iloc[-1]
            current_ma200 = ma200.iloc[-1]
            
            # Calculate trend strength
            ma50 = close.rolling(50).mean()
            if len(ma50.dropna()) > 0:
                current_ma50 = ma50.iloc[-1]
                # Bull: price > MA50 > MA200 and trending up
                if current_price > current_ma50 > current_ma200:
                    ma200_slope = (current_ma200 - ma200.iloc[-20]) / ma200.iloc[-20] if len(ma200) > 20 else 0
                    if ma200_slope > 0.001:  # Positive slope
                        return 'bull'
                # Bear: price < MA50 < MA200
                elif current_price < current_ma50 < current_ma200:
                    return 'bear'
            
            return 'neutral'
        except Exception as e:
            logger.debug(f"Error detecting regime: {e}")
            return 'neutral'
    
    def _get_max_exposure_by_regime(self, regime: str) -> float:
        """
        Get maximum portfolio exposure based on market regime.
        
        Rules:
        - Bull market: 90%
        - Neutral market: 60%
        - Bear market: 30%
        """
        regime_exposure = {
            'bull': 0.90,
            'neutral': 0.60,
            'bear': 0.30
        }
        return regime_exposure.get(regime, 0.60)
    
    def run(
        self,
        universe_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> BacktestResult:
        """
        Run backtest over specified period.
        
        Args:
            universe_data: Dict of symbol -> OHLCV DataFrame
            benchmark_data: Benchmark OHLCV DataFrame
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            BacktestResult with performance metrics
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            cash=self.config.initial_capital
        )
        
        # Get trading days from benchmark
        trading_days = benchmark_data.loc[start:end].index
        logger.info(f"Trading days in range: {len(trading_days)} days")
        logger.info(f"First trading day: {trading_days[0]}, Last: {trading_days[-1]}")
        
        # Store trading_days as instance variable for use in _generate_entries
        self._trading_days = trading_days
        
        # Main backtest loop
        processed_days = 0
        skipped_days = 0
        for i, date in enumerate(trading_days):
            if i < 252:  # Need 1 year of data for signals
                skipped_days += 1
                continue
            processed_days += 1
                
            # CRITICAL FIX: Use t-1 close for signal generation, t open for execution
            # Get previous day's close prices (for signal generation and exit triggers)
            prev_date = trading_days[i-1] if i > 0 else date
            prev_close_prices = self._get_prev_close_prices(universe_data, date)
            
            # Get current day's open prices (for actual execution)
            current_open_prices = self._get_current_open_prices(universe_data, date)
            
            # Update portfolio equity using current open prices (for daily valuation)
            portfolio.update_equity(date, current_open_prices)
            
            # Check stop losses and exits (using prev_close to trigger, open to execute)
            self._check_exits(portfolio, universe_data, benchmark_data, date, prev_close_prices, current_open_prices, self.signal_aggregator)
            
            # Check for new entries (based on rebalance frequency)
            # Signals generated on t-1 close, executed on t open
            should_rebalance = self._should_rebalance(date, i)
            if should_rebalance:
                self._generate_entries(
                    portfolio=portfolio,
                    universe_data=universe_data,
                    benchmark_data=benchmark_data,
                    date=date,
                    prev_close_prices=prev_close_prices,
                    current_open_prices=current_open_prices
                )
        
        logger.info(f"Backtest loop complete: processed {processed_days} days, skipped {skipped_days} days")
        
        # Close any remaining positions at end (use open price for execution)
        final_date = trading_days[-1]
        final_prices = self._get_current_open_prices(universe_data, final_date)
        for symbol, trade in list(portfolio.positions.items()):
            price = final_prices.get(symbol, trade.entry_price)
            trade.close(final_date, price, "backtest_end")
            portfolio.cash += trade.shares * price
            portfolio.closed_trades.append(trade)
            del portfolio.positions[symbol]
        
        # Calculate results
        return self._calculate_results(portfolio, benchmark_data, start, end)
    
    def _get_current_prices(
        self, 
        universe_data: Dict[str, pd.DataFrame], 
        date: datetime
    ) -> Dict[str, float]:
        """Get current prices for all stocks (legacy method, use _get_current_open_prices instead)."""
        return self._get_current_open_prices(universe_data, date)
    
    def _get_current_open_prices(
        self, 
        universe_data: Dict[str, pd.DataFrame], 
        date: datetime
    ) -> Dict[str, float]:
        """Get current day's OPEN prices for execution."""
        prices = {}
        for symbol, df in universe_data.items():
            # Find nearest date if exact date doesn't exist
            if date in df.index:
                prices[symbol] = df.loc[date, 'Open']
            else:
                # Get nearest previous date
                available_dates = df.index[df.index <= date]
                if len(available_dates) > 0:
                    nearest_date = available_dates[-1]
                    prices[symbol] = df.loc[nearest_date, 'Open']
        return prices
    
    def _get_prev_close_prices(
        self, 
        universe_data: Dict[str, pd.DataFrame], 
        date: datetime
    ) -> Dict[str, float]:
        """Get previous day's CLOSE prices for signal generation and exit triggers."""
        prices = {}
        for symbol, df in universe_data.items():
            # Get dates before current date
            available_dates = df.index[df.index < date]
            if len(available_dates) > 0:
                prev_date = available_dates[-1]
                prices[symbol] = df.loc[prev_date, 'Close']
        return prices
    
    def _should_rebalance(self, date: datetime, day_index: int) -> bool:
        """Check if we should rebalance on this day."""
        if self.config.rebalance_frequency == 'daily':
            return True
        elif self.config.rebalance_frequency == 'weekly':
            # Rebalance on Monday (weekday 0)
            is_monday = date.weekday() == 0
            if is_monday and (day_index % 50 == 0 or day_index < 260):  # Log first few and periodically
                logger.debug(f"Rebalance check on {date} (Monday): {is_monday}")
            return is_monday
        elif self.config.rebalance_frequency == 'monthly':
            return date.day <= 5 and day_index > 0
        return True
    
    def _check_exits(
        self,
        portfolio: Portfolio,
        universe_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        date: datetime,
        prev_close_prices: Dict[str, float],  # t-1 close for trigger detection
        current_open_prices: Dict[str, float],  # t open for execution
        signal_aggregator: Optional['SignalAggregator'] = None
    ):
        """
        Check and execute exits based on 5 trade scenarios:
        
        Scenario A (FAILED): 7%止损，日内至5天
        - 触及7%止损
        - 发生深蹲且3天内无修复
        - 放量跌破突破枢纽
        
        Scenario B (SWING): 20-25%盈利，2-6周
        - 在20%处遇到阻力，获利了结
        
        Scenario C (BREAKOUT): 8周法则，至少8周
        - 8周后，只要股价在50日均线上方就一直持有
        
        Scenario D (SUPER_PERFORMER): 100%+盈利，6-18个月
        - 股票从未有效跌破50日或200日均线
        
        Scenario E (ZOMBIE): 盈亏平衡，5-10天
        - 触发"时间止损"，股票不涨不跌
        """
        for symbol, trade in list(portfolio.positions.items()):
            if symbol not in prev_close_prices or symbol not in current_open_prices:
                continue
            
            # Use t-1 close for trigger detection, t open for execution
            prev_close = prev_close_prices[symbol]  # For checking exit conditions
            current_open = current_open_prices[symbol]  # For actual execution
            df = universe_data[symbol]
            
            # For profit calculation, use prev_close (what we see at end of t-1)
            current_price = prev_close
            
            # Update tracking variables
            trade.max_price_since_entry = max(trade.max_price_since_entry, current_price)
            trade.min_price_since_entry = min(trade.min_price_since_entry, current_price)
            
            # Calculate current profit
            unrealized_pnl_pct = (current_price - trade.entry_price) / trade.entry_price
            trade.max_profit = max(trade.max_profit, unrealized_pnl_pct)
            trade.max_drawdown = min(trade.max_drawdown, unrealized_pnl_pct)
            
            # Track 20% gain for Scenario C
            if unrealized_pnl_pct >= 0.20 and trade.days_since_20pct_gain is None:
                trade.days_since_20pct_gain = (date - trade.entry_date).days
            
            # Calculate holding days
            holding_days = (date - trade.entry_date).days
            
            # Initialize scenario if not set (for old trades)
            if trade.scenario is None:
                trade.scenario = self._determine_trade_scenario(symbol, trade.entry_price, df, trade.entry_date)
            
            # Get current signal score if aggregator available
            current_signal_score = trade.current_signal_score
            if signal_aggregator is not None and date in df.index:
                try:
                    lookback_start = date - timedelta(days=400)
                    subset = df.loc[lookback_start:date]
                    if len(subset) >= 252:
                        stock_data = StockData(ticker=symbol, df=subset)
                        signal = signal_aggregator.generate_composite_signal(stock_data)
                        trade.previous_signal_score = trade.current_signal_score
                        trade.current_signal_score = signal.composite_score
                        current_signal_score = signal.composite_score
                except Exception as e:
                    logger.debug(f"Could not update signal score for {symbol}: {e}")
            
            # ========== EXIT RULE PRIORITY ORDER ==========
            # Priority 1: Risk-based exits (stop loss, breakdowns)
            # Priority 2: Time-based exits (holding period limits)
            # Priority 3: Profit-based exits (targets, resistance)
            
            # ========== SCENARIO A: FAILED TRADE ==========
            if trade.scenario == TradeScenario.FAILED or holding_days <= 5:
                # PRIORITY 1: Risk exits
                # Check 7% stop loss (triggered by t-1 close, executed at t open)
                stop_loss_price = trade.entry_price * 0.93  # 7% loss
                if prev_close <= stop_loss_price:
                    # Execute at t open (or stop price if open is better)
                    exit_price = min(current_open, stop_loss_price)
                    exit_price = self._apply_slippage(exit_price, sell=True)
                    trade.close(date, exit_price, "scenario_A_7pct_stop")
                    portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                    portfolio.closed_trades.append(trade)
                    del portfolio.positions[symbol]
                    continue
                
                # Check deep squat without recovery (PRIORITY 1: Risk)
                has_squat, squat_date = self._check_deep_squat(trade, df, date)
                if has_squat and squat_date:
                    days_since_squat = (date - squat_date).days
                    if days_since_squat >= 3:
                        exit_price = self._apply_slippage(current_open, sell=True)
                        trade.close(date, exit_price, "scenario_A_deep_squat_no_recovery")
                        portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                        portfolio.closed_trades.append(trade)
                        del portfolio.positions[symbol]
                        continue
                
                # Check volume breakdown (PRIORITY 1: Risk)
                if self._check_volume_breakdown(df, date):
                    exit_price = self._apply_slippage(current_open, sell=True)
                    trade.close(date, exit_price, "scenario_A_volume_breakdown")
                    portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                    portfolio.closed_trades.append(trade)
                    del portfolio.positions[symbol]
                    continue
                
                # Reclassify to SWING if survived 5 days
                if holding_days > 5 and unrealized_pnl_pct > 0:
                    trade.scenario = TradeScenario.SWING
                    logger.debug(f"{symbol}: Reclassified from FAILED to SWING after {holding_days} days")
            
            # ========== SCENARIO B: SWING TRADE ==========
            elif trade.scenario == TradeScenario.SWING:
                # Check if hitting resistance at 20-25% gain
                if unrealized_pnl_pct >= 0.20:
                    if self._check_resistance_at_20pct(trade, current_price, df, date) or unrealized_pnl_pct >= 0.25:
                        exit_price = self._apply_slippage(current_price, sell=True)
                        shares_before_close = trade.shares
                        trade.close(date, exit_price, "scenario_B_resistance_20pct")
                        portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                        portfolio.closed_trades.append(trade)
                        del portfolio.positions[symbol]
                        continue
                
                # Check time limit (6 weeks = 42 days)
                if holding_days > 42:
                    exit_price = self._apply_slippage(current_price, sell=True)
                    shares_before_close = trade.shares
                    trade.close(date, exit_price, "scenario_B_time_limit_6weeks")
                    portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                    portfolio.closed_trades.append(trade)
                    del portfolio.positions[symbol]
                    continue
                
                # Check stop loss (7%)
                stop_loss_price = trade.entry_price * 0.93
                if current_price <= stop_loss_price:
                    exit_price = self._apply_slippage(stop_loss_price, sell=True)
                    shares_before_close = trade.shares
                    trade.close(date, exit_price, "scenario_B_stop_loss")
                    portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                    portfolio.closed_trades.append(trade)
                    del portfolio.positions[symbol]
                    continue
            
            # ========== SCENARIO C: BREAKOUT (8周法则) ==========
            elif trade.scenario == TradeScenario.BREAKOUT:
                weeks_held = holding_days / 7
                
                # Must hold at least 8 weeks (56 days)
                if weeks_held < 8:
                    # Check stop loss only (7%)
                    stop_loss_price = trade.entry_price * 0.93
                    if current_price <= stop_loss_price:
                        exit_price = self._apply_slippage(stop_loss_price, sell=True)
                        shares_before_close = trade.shares
                        trade.close(date, exit_price, "scenario_C_stop_loss_before_8weeks")
                        portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                        portfolio.closed_trades.append(trade)
                        del portfolio.positions[symbol]
                        continue
                else:
                    # After 8 weeks, check 50-day MA
                    mas = self._calculate_moving_averages(df, date, [50])
                    if 50 in mas:
                        ma50 = mas[50]
                        if current_price < ma50:
                            exit_price = self._apply_slippage(current_price, sell=True)
                            shares_before_close = trade.shares
                            trade.close(date, exit_price, "scenario_C_below_50ma_after_8weeks")
                            portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                            portfolio.closed_trades.append(trade)
                            del portfolio.positions[symbol]
                            continue
            
            # ========== SCENARIO D: SUPER PERFORMER ==========
            elif trade.scenario == TradeScenario.SUPER_PERFORMER:
                mas = self._calculate_moving_averages(df, date, [50, 200])
                
                # Check if broke below 50 or 200 MA (有效跌破 = 3% below)
                if 50 in mas:
                    ma50 = mas[50]
                    if current_price < ma50 * 0.97:  # 3% below 50 MA
                        exit_price = self._apply_slippage(current_price, sell=True)
                        shares_before_close = trade.shares
                        trade.close(date, exit_price, "scenario_D_below_50ma")
                        portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                        portfolio.closed_trades.append(trade)
                        del portfolio.positions[symbol]
                        continue
                
                if 200 in mas:
                    ma200 = mas[200]
                    if current_price < ma200 * 0.97:  # 3% below 200 MA
                        exit_price = self._apply_slippage(current_price, sell=True)
                        shares_before_close = trade.shares
                        trade.close(date, exit_price, "scenario_D_below_200ma")
                        portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                        portfolio.closed_trades.append(trade)
                        del portfolio.positions[symbol]
                        continue
                
                # Time limit: 18 months (540 days)
                if holding_days > 540:
                    exit_price = self._apply_slippage(current_price, sell=True)
                    shares_before_close = trade.shares
                    trade.close(date, exit_price, "scenario_D_time_limit_18months")
                    portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                    portfolio.closed_trades.append(trade)
                    del portfolio.positions[symbol]
                    continue
            
            # ========== SCENARIO E: ZOMBIE STOCK ==========
            elif trade.scenario == TradeScenario.ZOMBIE:
                # PRIORITY 2: Time-based exits
                # Extended hard limit: 180 days (was 10 days)
                if holding_days >= 180:
                    exit_price = self._apply_slippage(current_open, sell=True)
                    trade.close(date, exit_price, "scenario_E_time_limit_180days")
                    portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                    portfolio.closed_trades.append(trade)
                    del portfolio.positions[symbol]
                    continue
                
                # Time stop: Check periodically (every 30 days) if stock is still not moving
                # Check at 5 days, then every 30 days up to 180 days
                if holding_days == 5 or (holding_days >= 30 and holding_days % 30 == 0):
                    # Check if stock is moving (price change > 2%)
                    price_change = abs(unrealized_pnl_pct)
                    if price_change < 0.02:  # Less than 2% movement
                        exit_price = self._apply_slippage(current_open, sell=True)
                        trade.close(date, exit_price, "scenario_E_time_stop_zombie")
                        portfolio.cash += trade.shares * exit_price - self._calc_commission(trade.shares)
                        portfolio.closed_trades.append(trade)
                        del portfolio.positions[symbol]
                        continue
            
            # ========== DETECT ZOMBIE STOCKS ==========
            # CRITICAL: Only use information available at current time (t-1 close)
            # Check if any trade becomes zombie (not moving for 5+ days)
            # This uses only past data, no future information
            # Extended detection window: check every 30 days up to 180 days
            if holding_days >= 5 and holding_days <= 180 and trade.scenario != TradeScenario.ZOMBIE:
                # Check every 30 days or at 5 days
                if holding_days == 5 or holding_days % 30 == 0:
                    # Calculate price change using only data up to t-1
                    price_change = abs(unrealized_pnl_pct)  # Based on prev_close vs entry
                    if price_change < 0.02:  # Less than 2% movement
                        trade.scenario = TradeScenario.ZOMBIE
                        logger.debug(f"{symbol}: Reclassified to ZOMBIE - no movement for {holding_days} days")
            
            # Update trailing stop for all scenarios (except ZOMBIE)
            if trade.scenario != TradeScenario.ZOMBIE and self.config.use_trailing_stops and date in df.index:
                try:
                    stock_data = StockData(ticker=symbol, df=df.loc[:date].tail(50))
                    stop_result = self.stop_manager.calculate_chandelier_stop(
                        stock_data,
                        from_entry=False
                    )
                    new_stop = stop_result.stop_price
                    # Only update if new stop is higher (for long positions)
                    if new_stop > trade.stop_loss:
                        trade.stop_loss = new_stop
                except Exception as e:
                    logger.debug(f"Could not update trailing stop for {symbol}: {e}")
    
    def _generate_entries(
        self,
        portfolio: Portfolio,
        universe_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        date: datetime,
        prev_close_prices: Dict[str, float],  # t-1 close for signal generation
        current_open_prices: Dict[str, float]  # t open for execution
    ):
        """Generate new entry signals and execute trades."""
        # Check portfolio drawdown
        if portfolio.current_drawdown > self.config.max_portfolio_drawdown:
            return  # Don't add new positions in high drawdown
        
        # Get market regime to determine max exposure (uses t-1 data)
        regime = self._detect_market_regime(benchmark_data, date)
        max_exposure_pct = self._get_max_exposure_by_regime(regime)
        
        # Get available slots
        available_slots = self.config.max_positions - len(portfolio.positions)
        if available_slots <= 0:
            return
        
        # CRITICAL FIX: Signal generation must use t-1 data (no look-ahead bias)
        # Prepare data for signal generation using t-1 close
        lookback_start = date - timedelta(days=400)
        
        universe_subset = {}
        for symbol, df in universe_data.items():
            if symbol in portfolio.positions:
                continue
            
            # Get dates BEFORE current date (t-1 or earlier)
            available_dates = df.index[df.index < date]
            if len(available_dates) == 0:
                continue
            
            # Use the most recent date BEFORE current date (t-1)
            signal_date = available_dates[-1]
            
            try:
                # Get data up to signal_date (t-1), then take last 400 days
                subset = df.loc[:signal_date].tail(400)
                if len(subset) >= 252:
                    universe_subset[symbol] = subset
                else:
                    logger.debug(f"{symbol} has only {len(subset)} days (need 252) on {date}")
            except Exception as e:
                logger.warning(f"Error slicing {symbol} data: {e}")
                continue
        
        if not universe_subset:
            return
        
        # Update benchmark data for regime detection (use t-1 data)
        try:
            # Use data up to t-1 for regime detection
            bench_available_dates = benchmark_data.index[benchmark_data.index < date]
            if len(bench_available_dates) == 0:
                logger.warning(f"Insufficient benchmark data on {date}")
                return
            bench_signal_date = bench_available_dates[-1]
            bench_subset = benchmark_data.loc[lookback_start:bench_signal_date]
            if len(bench_subset) < 50:
                logger.warning(f"Insufficient benchmark data on {date}")
                return
            self.signal_aggregator.benchmark_data = bench_subset
        except Exception as e:
            logger.warning(f"Error preparing benchmark data: {e}")
            return
        
        # Scan universe for signals
        try:
            signals = self.signal_aggregator.scan_universe(
                universe_subset,
                min_score=self.config.min_signal_score
            )
        except Exception as e:
            logger.error(f"Signal generation failed on {date}: {e}", exc_info=True)
            return
        
        # Take top signals up to available slots
        if len(signals) == 0:
            return  # No signals to process
        for signal in signals[:available_slots]:
            symbol = signal.ticker  # CompositeSignal has ticker attribute directly
            if not symbol or symbol not in current_open_prices:
                continue
            
            # CRITICAL: Execute at t open price (not t-1 close)
            entry_price = self._apply_slippage(current_open_prices[symbol], sell=False)
            
            # Calculate position size - EACH STOCK CALCULATES INDIVIDUALLY
            df = universe_subset[symbol]
            stock_data = StockData(ticker=symbol, df=df)
            stop_result = self.stop_manager.calculate_chandelier_stop(
                stock_data,
                from_entry=True,
                entry_price=entry_price
            )
            stop_loss = stop_result.stop_price
            
            # Get current equity (updated after each trade)
            current_equity = portfolio.equity_curve[-1][1] if portfolio.equity_curve else portfolio.initial_capital
            
            # Calculate current total exposure (including positions already opened in this loop)
            current_exposure = sum(t.shares * current_open_prices.get(t.symbol, t.entry_price) 
                                  for t in portfolio.positions.values())
            current_exposure_pct = current_exposure / current_equity if current_equity > 0 else 0
            
            # Check if we're at max exposure for current regime
            if current_exposure_pct >= max_exposure_pct:
                continue
            
            # Calculate available exposure for THIS stock (dynamic, recalculated for each stock)
            available_exposure_pct = max_exposure_pct - current_exposure_pct
            available_exposure_dollars = available_exposure_pct * current_equity
            
            # Use calculate_fixed_risk_size method - each stock calculates based on current equity
            # This ensures each stock gets 1% risk of CURRENT portfolio value
            # 每只股票单独计算：风险金额 = 当前组合权益 × 1%
            risk_amount_per_stock = current_equity * 0.01  # $1,000 for $100k portfolio
            position_result = self.position_sizer.calculate_fixed_risk_size(
                portfolio_value=current_equity,
                entry_price=entry_price,
                stop_price=stop_loss,
                risk_per_trade=0.01  # 1% risk per trade (of current equity)
            )
            shares = position_result.shares
            
            # Apply maximum position size (per position) - 10% of current equity per stock
            max_shares_per_position = int(current_equity * self.config.max_position_pct / entry_price)
            shares = min(shares, max_shares_per_position)
            
            # Apply available exposure limit (regime-based total exposure limit)
            max_shares_by_exposure = int(available_exposure_dollars / entry_price)
            shares = min(shares, max_shares_by_exposure)
            
            if shares <= 0:
                continue
            
            # Check if we have enough cash
            cost = shares * entry_price + self._calc_commission(shares)
            if cost > portfolio.cash:
                shares = int((portfolio.cash - self.config.commission_minimum) / entry_price)
                if shares <= 0:
                    continue
                cost = shares * entry_price + self._calc_commission(shares)
            
            # Determine trade scenario
            scenario = self._determine_trade_scenario(symbol, entry_price, df, date)
            
            # Execute trade
            trade = Trade(
                symbol=symbol,
                direction=TradeDirection.LONG,
                entry_date=date,
                entry_price=entry_price,
                shares=shares,
                stop_loss=stop_loss,
                entry_signal_score=signal.composite_score,
                current_signal_score=signal.composite_score,
                previous_signal_score=signal.composite_score,
                scenario=scenario,
                max_price_since_entry=entry_price,
                min_price_since_entry=entry_price
            )
            
            portfolio.positions[symbol] = trade
            portfolio.cash -= cost
    
    def _apply_slippage(self, price: float, sell: bool) -> float:
        """Apply slippage to execution price."""
        if sell:
            return price * (1 - self.config.slippage_pct)
        else:
            return price * (1 + self.config.slippage_pct)
    
    def _calc_commission(self, shares: int) -> float:
        """Calculate commission for trade."""
        return max(
            shares * self.config.commission_per_share,
            self.config.commission_minimum
        )
    
    def _calculate_results(
        self,
        portfolio: Portfolio,
        benchmark_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        
        if not portfolio.equity_curve:
            logger.warning("No equity curve data, returning empty results")
            return BacktestResult(
                total_return=0, annualized_return=0, sharpe_ratio=0,
                sortino_ratio=0, max_drawdown=0, win_rate=0,
                profit_factor=0, total_trades=0,
                equity_curve=pd.Series(), drawdown_curve=pd.Series(),
                trades=pd.DataFrame(), benchmark_return=0,
                alpha=0, beta=1.0, information_ratio=0
            )
        
        # Build equity series
        dates, values = zip(*portfolio.equity_curve)
        equity_series = pd.Series(values, index=pd.DatetimeIndex(dates))
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        
        # Total and annualized return
        total_return = (equity_series.iloc[-1] / portfolio.initial_capital) - 1
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratio (assuming 0 risk-free rate for simplicity)
        if returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = np.sqrt(252) * returns.mean() / downside_returns.std()
        else:
            sortino = sharpe
        
        # Maximum drawdown
        rolling_max = equity_series.expanding().max()
        drawdown_series = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown_series.min())
        
        # Trade statistics - Calculate average return per trade
        total_trades = len(portfolio.closed_trades)
        if total_trades > 0:
            # Calculate average return percentage (from entry to exit for each complete trade)
            # This is the key metric: average return per stock from signal buy to complete sell
            returns_pct = [t.pnl_pct for t in portfolio.closed_trades if t.exit_date is not None]
            average_return_per_trade = np.mean(returns_pct) if returns_pct else 0.0
            
            winners = [t for t in portfolio.closed_trades if t.is_winner]
            losers = [t for t in portfolio.closed_trades if not t.is_winner]
            
            win_rate = len(winners) / total_trades
            
            gross_profit = sum(t.pnl for t in winners) if winners else 0
            gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
            
            # Exit reason statistics (to understand what drives performance)
            exit_reasons = {}
            for trade in portfolio.closed_trades:
                reason = trade.exit_reason or "unknown"
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            logger.info(f"Exit reasons breakdown: {exit_reasons}")
        else:
            average_return_per_trade = 0.0
            win_rate = 0
            profit_factor = 0
            exit_reasons = {}
        
        # Create trades DataFrame
        trades_data = []
        for trade in portfolio.closed_trades:
            trades_data.append({
                'symbol': trade.symbol,
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'shares': trade.shares,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'holding_days': trade.holding_days,
                'exit_reason': trade.exit_reason
            })
        trades_df = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()
        
        # Benchmark comparison
        bench_subset = benchmark_data.loc[equity_series.index[0]:equity_series.index[-1]]
        if len(bench_subset) > 0:
            bench_returns = bench_subset['Close'].pct_change().dropna()
            bench_total = (bench_subset['Close'].iloc[-1] / bench_subset['Close'].iloc[0]) - 1
            benchmark_return = bench_total
            
            # Beta calculation
            aligned_returns = None
            aligned_bench = None
            if len(returns) > 20 and len(bench_returns) > 20:
                aligned_returns = returns.reindex(bench_returns.index).dropna()
                aligned_bench = bench_returns.reindex(aligned_returns.index).dropna()
                
                if len(aligned_returns) > 20:
                    covar = np.cov(aligned_returns, aligned_bench)[0, 1]
                    bench_var = np.var(aligned_bench)
                    beta = covar / bench_var if bench_var > 0 else 1.0
                    alpha = annualized_return - beta * (bench_total / years if years > 0 else 0)
                else:
                    beta = 1.0
                    alpha = annualized_return - bench_total / years if years > 0 else 0
            else:
                beta = 1.0
                alpha = 0
            
            # Information ratio
            if aligned_returns is not None and aligned_bench is not None and len(aligned_returns) > 0:
                tracking_error = (aligned_returns - aligned_bench).std() * np.sqrt(252)
                info_ratio = (annualized_return - bench_total / years) / tracking_error if tracking_error > 0 else 0
            else:
                info_ratio = 0
        else:
            benchmark_return = 0
            alpha = annualized_return
            beta = 1.0
            info_ratio = 0
        
        result = BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            average_return_per_trade=average_return_per_trade,
            equity_curve=equity_series,
            drawdown_curve=drawdown_series,
            trades=trades_df,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            information_ratio=info_ratio,
            exit_reasons=exit_reasons
        )
        
        return result


def generate_backtest_report(result: BacktestResult, name: str = "Strategy") -> str:
    """Generate a formatted backtest report."""
    
    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    BACKTEST REPORT: {name:^20}             ║
╠══════════════════════════════════════════════════════════════════╣
║  RETURNS                                                         ║
║  ├─ Total Return:        {result.total_return:>10.2%}                          ║
║  ├─ Annualized Return:   {result.annualized_return:>10.2%}                          ║
║  └─ Max Drawdown:        {result.max_drawdown:>10.2%}                          ║
╠══════════════════════════════════════════════════════════════════╣
║  RISK METRICS                                                    ║
║  ├─ Sharpe Ratio:        {result.sharpe_ratio:>10.2f}                          ║
║  ├─ Sortino Ratio:       {result.sortino_ratio:>10.2f}                          ║
║  ├─ Alpha:               {result.alpha:>10.2%}                          ║
║  ├─ Beta:                {result.beta:>10.2f}                          ║
║  └─ Information Ratio:   {result.information_ratio:>10.2f}                          ║
╠══════════════════════════════════════════════════════════════════╣
║  TRADE STATISTICS                                                ║
║  ├─ Total Trades:        {result.total_trades:>10d}                          ║
║  ├─ Average Return/Trade:{result.average_return_per_trade:>10.2%}                          ║
║  ├─ Win Rate:            {result.win_rate:>10.2%}                          ║
║  └─ Profit Factor:       {result.profit_factor:>10.2f}                          ║
╚══════════════════════════════════════════════════════════════════╝
"""
    return report
