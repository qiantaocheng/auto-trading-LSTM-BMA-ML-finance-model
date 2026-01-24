"""
Backtest Runner for Quantitative Signal System
===============================================

Runs comprehensive backtest using Polygon.io data with MultiIndex format.
Tests the effectiveness of the signal system over 3 years of historical data.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import logging

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import SystemConfig, BacktestConfig
from core.signal_aggregator import SignalAggregator
from backtest.engine import BacktestEngine, generate_backtest_report
from data.polygon_data_fetcher import PolygonDataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_backtest(
    symbols: List[str],
    benchmark_symbol: str = 'SPY',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years: int = 3,
    initial_capital: float = 100000.0,
    max_positions: int = 10,
    min_signal_score: float = 0.60,
    cache_dir: Optional[str] = None
):
    """
    Run comprehensive backtest on universe of stocks.
    
    Args:
        symbols: List of stock symbols to backtest
        benchmark_symbol: Benchmark symbol (default: SPY)
        start_date: Start date (YYYY-MM-DD). If None, calculates from years
        end_date: End date (YYYY-MM-DD). If None, uses today
        years: Number of years of history
        initial_capital: Starting capital
        max_positions: Maximum concurrent positions
        min_signal_score: Minimum signal score for entry
        cache_dir: Cache directory for data
        
    Returns:
        BacktestResult object
    """
    logger.info("="*80)
    logger.info("QUANTITATIVE SIGNAL SYSTEM - BACKTEST")
    logger.info("="*80)
    logger.info(f"Universe: {len(symbols)} symbols")
    logger.info(f"Benchmark: {benchmark_symbol}")
    logger.info(f"Years: {years}")
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Max Positions: {max_positions}")
    logger.info(f"Min Signal Score: {min_signal_score}")
    
    # Calculate dates
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    
    logger.info(f"Date Range: {start_date} to {end_date}")
    
    # Initialize data fetcher
    fetcher = PolygonDataFetcher(cache_dir=cache_dir)
    
    # Fetch universe data
    logger.info("\n" + "-"*80)
    logger.info("FETCHING UNIVERSE DATA")
    logger.info("-"*80)
    
    universe_multiindex = fetcher.fetch_universe(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        years=years,
        use_cache=True
    )
    
    if universe_multiindex.empty:
        logger.error("Failed to fetch universe data")
        return None
    
    # Convert to dict format for backtest engine
    universe_dict = fetcher.get_universe_dict(universe_multiindex)
    logger.info(f"Successfully fetched data for {len(universe_dict)} symbols")
    
    # Fetch benchmark data
    logger.info(f"\nFetching benchmark data: {benchmark_symbol}")
    benchmark_data = fetcher.fetch_symbol(
        symbol=benchmark_symbol,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    
    if benchmark_data is None or benchmark_data.empty:
        logger.error(f"Failed to fetch benchmark data for {benchmark_symbol}")
        return None
    
    logger.info(f"Benchmark data: {len(benchmark_data)} days")
    
    # Initialize signal aggregator
    logger.info("\n" + "-"*80)
    logger.info("INITIALIZING SIGNAL SYSTEM")
    logger.info("-"*80)
    
    config = SystemConfig()
    aggregator = SignalAggregator(config=config, benchmark_data=benchmark_data)
    
    # Configure backtest
    backtest_config = BacktestConfig(
        initial_capital=initial_capital,
        max_positions=max_positions,
        min_signal_score=min_signal_score,
        rebalance_frequency='weekly',
        commission_per_share=0.005,
        slippage_pct=0.001
    )
    
    # Initialize backtest engine
    logger.info("\n" + "-"*80)
    logger.info("RUNNING BACKTEST")
    logger.info("-"*80)
    
    engine = BacktestEngine(
        signal_aggregator=aggregator,
        config=backtest_config,
        risk_config=config.risk
    )
    
    # Run backtest
    try:
        result = engine.run(
            universe_data=universe_dict,
            benchmark_data=benchmark_data,
            start_date=start_date,
            end_date=end_date
        )
        
        # Generate report
        logger.info("\n" + "="*80)
        logger.info("BACKTEST RESULTS")
        logger.info("="*80)
        
        report = generate_backtest_report(result, "Quantitative Signal System")
        logger.info(report)
        
        # Additional statistics
        logger.info("\n" + "-"*80)
        logger.info("ADDITIONAL STATISTICS")
        logger.info("-"*80)
        
        if hasattr(result, 'equity_curve') and not result.equity_curve.empty:
            equity = result.equity_curve
            logger.info(f"Final Equity: ${equity.iloc[-1]:,.2f}")
            logger.info(f"Peak Equity: ${equity.max():,.2f}")
            logger.info(f"Total Return: {result.total_return:.2%}")
            logger.info(f"Annualized Return: {result.annualized_return:.2%}")
            logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
            logger.info(f"Win Rate: {result.win_rate:.2%}")
            logger.info(f"Total Trades: {result.total_trades}")
            
            if hasattr(result, 'alpha'):
                logger.info(f"Alpha: {result.alpha:.2%}")
                logger.info(f"Beta: {result.beta:.2f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return None


def main():
    """Main entry point for backtest runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run backtest on quantitative signal system'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'AMD', 'INTC'],
        help='List of symbols to backtest'
    )
    parser.add_argument(
        '--benchmark',
        default='SPY',
        help='Benchmark symbol (default: SPY)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD). If not provided, calculates from --years'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--years',
        type=int,
        default=3,
        help='Number of years of history (default: 3)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial capital (default: 100000)'
    )
    parser.add_argument(
        '--max-positions',
        type=int,
        default=10,
        help='Maximum concurrent positions (default: 10)'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.60,
        help='Minimum signal score for entry (default: 0.60)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        help='Cache directory for data'
    )
    
    args = parser.parse_args()
    
    # Run backtest
    result = run_backtest(
        symbols=args.symbols,
        benchmark_symbol=args.benchmark,
        start_date=args.start_date,
        end_date=args.end_date,
        years=args.years,
        initial_capital=args.capital,
        max_positions=args.max_positions,
        min_signal_score=args.min_score,
        cache_dir=args.cache_dir
    )
    
    if result is None:
        logger.error("Backtest failed")
        sys.exit(1)
    
    logger.info("\n" + "="*80)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
