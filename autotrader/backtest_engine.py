"""
Backtesting engine for autotrader
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage: float = 0.001
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []


@dataclass
class BacktestResult:
    """Results from backtesting"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    daily_returns: pd.Series
    equity_curve: pd.Series
    trades: List[Dict]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_return': self.total_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades
        }


class AutoTraderBacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio_value = config.initial_capital
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict] = []
        self.daily_values: List[float] = []
        
    def run_backtest(self, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> BacktestResult:
        """Run the backtest with given signals and prices"""
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Simulate trading based on signals
        equity_curve = []
        daily_returns = []
        
        for date in pd.date_range(self.config.start_date, self.config.end_date):
            date_str = date.strftime('%Y-%m-%d')
            
            # Get signals for this date
            if date_str in signals_df.index:
                signals = signals_df.loc[date_str]
                self._execute_signals(signals, prices_df, date)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(prices_df, date)
            equity_curve.append(portfolio_value)
            
            if len(equity_curve) > 1:
                daily_return = (portfolio_value - equity_curve[-2]) / equity_curve[-2]
                daily_returns.append(daily_return)
        
        # Calculate performance metrics
        equity_series = pd.Series(equity_curve)
        returns_series = pd.Series(daily_returns)
        
        total_return = (equity_curve[-1] - self.config.initial_capital) / self.config.initial_capital
        sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(equity_series)
        win_rate = self._calculate_win_rate()
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(self.trades),
            daily_returns=returns_series,
            equity_curve=equity_series,
            trades=self.trades
        )
    
    def _execute_signals(self, signals: pd.Series, prices_df: pd.DataFrame, date: datetime):
        """Execute trading signals"""
        for symbol in signals.index:
            signal = signals[symbol]
            if abs(signal) > 0.01:  # Only trade if signal is significant
                self._place_order(symbol, signal, prices_df, date)
    
    def _place_order(self, symbol: str, signal: float, prices_df: pd.DataFrame, date: datetime):
        """Place an order based on signal"""
        date_str = date.strftime('%Y-%m-%d')
        if date_str not in prices_df.index or symbol not in prices_df.columns:
            return
            
        price = prices_df.loc[date_str, symbol]
        if pd.isna(price):
            return
            
        # Calculate position size based on signal strength
        position_size = signal * self.portfolio_value * 0.1  # Risk 10% per position
        
        # Apply slippage and commission
        execution_price = price * (1 + self.config.slippage * np.sign(signal))
        commission = abs(position_size) * self.config.commission_rate
        
        # Update positions
        current_position = self.positions.get(symbol, 0)
        self.positions[symbol] = current_position + position_size / execution_price
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'signal': signal,
            'price': execution_price,
            'quantity': position_size / execution_price,
            'commission': commission
        })
        
        logger.debug(f"Executed trade: {symbol} @ {execution_price:.2f}, qty: {position_size/execution_price:.2f}")
    
    def _calculate_portfolio_value(self, prices_df: pd.DataFrame, date: datetime) -> float:
        """Calculate current portfolio value"""
        date_str = date.strftime('%Y-%m-%d')
        if date_str not in prices_df.index:
            return self.portfolio_value
            
        total_value = 0
        for symbol, quantity in self.positions.items():
            if symbol in prices_df.columns:
                price = prices_df.loc[date_str, symbol]
                if not pd.isna(price):
                    total_value += quantity * price
        
        return total_value
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trades"""
        if not self.trades:
            return 0.0
            
        winning_trades = sum(1 for trade in self.trades if trade['signal'] > 0)
        return winning_trades / len(self.trades)


def run_backtest_with_config(config: BacktestConfig, signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> BacktestResult:
    """Run backtest with given configuration"""
    engine = AutoTraderBacktestEngine(config)
    return engine.run_backtest(signals_df, prices_df)


def run_preset_backtests() -> List[BacktestResult]:
    """Run preset backtests for common scenarios"""
    logger.info("Running preset backtests...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Generate random price data
    np.random.seed(42)
    prices_data = {}
    for symbol in symbols:
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
        prices_data[symbol] = prices
    
    prices_df = pd.DataFrame(prices_data, index=dates)
    
    # Generate random signals
    signals_data = {}
    for symbol in symbols:
        signals = np.random.randn(len(dates)) * 0.1
        signals_data[symbol] = signals
    
    signals_df = pd.DataFrame(signals_data, index=dates)
    
    results = []
    
    # Run different preset configurations
    configs = [
        BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 30),
            initial_capital=50000,
            commission_rate=0.001
        ),
        BacktestConfig(
            start_date=datetime(2023, 7, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            commission_rate=0.0005
        )
    ]
    
    for config in configs:
        result = run_backtest_with_config(config, signals_df, prices_df)
        results.append(result)
    
    return results