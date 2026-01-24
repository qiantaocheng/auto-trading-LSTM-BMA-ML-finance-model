"""Backtesting module for strategy evaluation."""

from .engine import (
    BacktestEngine,
    BacktestConfig,
    Trade,
    TradeDirection,
    Portfolio,
    generate_backtest_report,
)

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'Trade',
    'TradeDirection',
    'Portfolio',
    'generate_backtest_report',
]
