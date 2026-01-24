"""
Backtest Runner with Factor Data
=================================

Runs backtest using factor data from parquet file.
Fetches OHLCV data from Polygon and merges with existing factors.
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

from config.settings import SystemConfig
from core.signal_aggregator import SignalAggregator
from backtest.engine import BacktestEngine, BacktestConfig, generate_backtest_report
from data.factor_data_loader import FactorDataLoader, load_factor_data_for_backtest
from data.polygon_data_fetcher import PolygonDataFetcher

logging.basicConfig(
    level=logging.INFO,  # Changed to INFO to suppress DEBUG messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_backtest_with_factors(
    factor_file: str,
    benchmark_symbol: str = 'SPY',
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 100000.0,
    max_positions: int = 10,  # Not used in independent mode, kept for compatibility
    min_signal_score: float = 0.60,
    use_cache: bool = True,
    save_merged: Optional[str] = None
):
    """
    Run backtest using factor data.
    
    IMPORTANT: Each ticker runs independently with its own portfolio.
    No position limits - each stock has its own separate capital pool.
    
    Args:
        factor_file: Path to factor parquet file
        benchmark_symbol: Benchmark symbol (default: SPY)
        tickers: Optional list of tickers (if None, uses all from factor file)
        start_date: Start date (if None, uses from factor file)
        end_date: End date (if None, uses from factor file)
        initial_capital: Starting capital PER STOCK (each stock gets this amount)
        max_positions: NOT USED - kept for compatibility only
        min_signal_score: Minimum signal score for entry
        use_cache: Whether to use cached OHLCV data
        save_merged: Optional path to save merged data
        
    Returns:
        Dictionary with aggregated results across all stocks
    """
    logger.info("="*80)
    logger.info("QUANTITATIVE SIGNAL SYSTEM - INDEPENDENT STOCK BACKTEST")
    logger.info("="*80)
    logger.info(f"Factor file: {factor_file}")
    logger.info(f"Benchmark: {benchmark_symbol}")
    logger.info(f"Mode: INDEPENDENT - Each stock has its own portfolio")
    
    # Initialize factor loader
    loader = FactorDataLoader(factor_file)
    
    # Load factors
    logger.info("\n" + "-"*80)
    logger.info("LOADING FACTOR DATA")
    logger.info("-"*80)
    
    factor_data = loader.load_factors()
    
    # Get tickers
    if tickers is None:
        tickers = loader.get_tickers()
        logger.info(f"Using all {len(tickers)} tickers from factor file")
    else:
        logger.info(f"Using {len(tickers)} specified tickers")
    
    # Get date range
    if start_date is None or end_date is None:
        start_date, end_date = loader.get_date_range()
        logger.info(f"Using date range from factor file: {start_date} to {end_date}")
    else:
        logger.info(f"Using specified date range: {start_date} to {end_date}")
    
    # Merge factors with OHLCV for ALL tickers at once (more efficient)
    logger.info("\n" + "-"*80)
    logger.info("FETCHING OHLCV DATA AND MERGING")
    logger.info("-"*80)
    
    merged_data = loader.merge_factors_with_ohlcv(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache
    )
    
    # Save merged data if requested
    if save_merged:
        loader.save_merged_data(merged_data, save_merged)
        logger.info(f"Saved merged data to {save_merged}")
    
    # Convert to dictionary format
    logger.info("\nConverting to dictionary format...")
    all_universe_dict = loader.get_universe_dict(merged_data)
    logger.info(f"Prepared {len(all_universe_dict)} tickers for backtesting")
    
    # Fetch benchmark
    logger.info(f"\nFetching benchmark data: {benchmark_symbol}")
    fetcher = PolygonDataFetcher()
    benchmark_data = fetcher.fetch_symbol(
        symbol=benchmark_symbol,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache
    )
    
    if benchmark_data is None or benchmark_data.empty:
        logger.error(f"Failed to fetch benchmark data for {benchmark_symbol}")
        return None
    
    logger.info(f"Benchmark data: {len(benchmark_data)} days")
    
    # Initialize signal aggregator (shared across all stocks)
    logger.info("\n" + "-"*80)
    logger.info("INITIALIZING SIGNAL SYSTEM")
    logger.info("-"*80)
    
    config = SystemConfig()
    aggregator = SignalAggregator(config=config, benchmark_data=benchmark_data)
    
    # Configure backtest - max_positions=1 for each stock (only holds itself)
    backtest_config = BacktestConfig(
        initial_capital=initial_capital,
        max_positions=1,  # Each stock only holds itself
        min_signal_score=min_signal_score,
        rebalance_frequency='weekly',
        commission_per_share=0.005,
        slippage_pct=0.001
    )
    
    logger.info(f"Initial Capital PER STOCK: ${initial_capital:,.2f}")
    logger.info(f"Min Signal Score: {min_signal_score}")
    logger.info(f"Running {len(tickers)} independent backtests...")
    
    # Initialize backtest engine (shared, but each run is independent)
    engine = BacktestEngine(
        signal_aggregator=aggregator,
        config=backtest_config,
        risk_config=config.risk
    )
    
    # Run independent backtest for each ticker
    logger.info("\n" + "-"*80)
    logger.info("RUNNING INDEPENDENT BACKTESTS")
    logger.info("-"*80)
    
    all_results = []
    successful_runs = 0
    failed_runs = 0
    
    for i, ticker in enumerate(tickers, 1):
        if ticker not in all_universe_dict:
            logger.warning(f"Skipping {ticker}: not in universe data")
            failed_runs += 1
            continue
        
        logger.info(f"\n[{i}/{len(tickers)}] Running backtest for {ticker}...")
        
        # Create universe dict with only this ticker
        single_ticker_universe = {ticker: all_universe_dict[ticker]}
        
        try:
            result = engine.run(
                universe_data=single_ticker_universe,
                benchmark_data=benchmark_data,
                start_date=start_date,
                end_date=end_date
            )
            
            if result and result.total_trades > 0:
                all_results.append({
                    'ticker': ticker,
                    'result': result,
                    'avg_return': result.average_return_per_trade,
                    'total_trades': result.total_trades
                })
                successful_runs += 1
                logger.info(f"  {ticker}: {result.total_trades} trades, avg return: {result.average_return_per_trade:.2%}")
            else:
                logger.info(f"  {ticker}: No trades generated")
                failed_runs += 1
                
        except Exception as e:
            logger.error(f"  {ticker}: Backtest failed: {e}")
            failed_runs += 1
    
    # Aggregate results
    logger.info("\n" + "="*80)
    logger.info("AGGREGATED RESULTS (ALL STOCKS)")
    logger.info("="*80)
    
    if not all_results:
        logger.error("No successful backtests!")
        return None
    
    # Calculate average return across all stocks
    all_returns = []
    total_trades_all = 0
    all_exit_reasons = {}
    
    for stock_result in all_results:
        result = stock_result['result']
        if result.total_trades > 0:
            # Collect returns from all trades (from trades DataFrame)
            if not result.trades.empty and 'pnl_pct' in result.trades.columns:
                # Get unique rows to avoid duplicates
                returns = result.trades[['pnl_pct']].drop_duplicates()['pnl_pct'].dropna()
                # Convert to list and filter out invalid values
                returns_list = [float(r) for r in returns if pd.notna(r) and isinstance(r, (int, float))]
                all_returns.extend(returns_list)
            
            total_trades_all += result.total_trades
            
            # Aggregate exit reasons
            if hasattr(result, 'exit_reasons') and result.exit_reasons:
                for reason, count in result.exit_reasons.items():
                    all_exit_reasons[reason] = all_exit_reasons.get(reason, 0) + count
    
    if not all_returns:
        logger.error("No trades with returns found!")
        return None
    
    # Ensure all returns are valid numbers
    all_returns = [r for r in all_returns if isinstance(r, (int, float)) and not np.isnan(r) and not np.isinf(r)]
    
    if not all_returns:
        logger.error("No valid returns found!")
        return None
    
    # Calculate statistics
    avg_return_all_stocks = np.mean(all_returns)
    winners = [r for r in all_returns if r > 0]
    losers = [r for r in all_returns if r <= 0]
    num_winners = len(winners)
    num_losers = len(losers)
    total_returns = len(all_returns)
    
    # Calculate win rate safely
    if total_returns > 0:
        win_rate = (num_winners / total_returns) * 100
    else:
        win_rate = 0.0
    
    logger.info(f"\nTotal Stocks Tested: {len(tickers)}")
    logger.info(f"Successful Runs: {successful_runs}")
    logger.info(f"Failed Runs: {failed_runs}")
    logger.info(f"Total Trades Across All Stocks: {total_trades_all}")
    logger.info(f"Total Returns Collected: {total_returns}")
    logger.info(f"Winners: {num_winners}, Losers: {num_losers}")
    logger.info(f"\nAverage Return Per Trade (All Stocks): {avg_return_all_stocks:.2%}")
    logger.info(f"Win Rate (All Stocks): {win_rate:.2f}%")
    
    if all_exit_reasons:
        logger.info("\nExit Reasons Breakdown (All Stocks):")
        total_exits = sum(all_exit_reasons.values())
        for reason, count in sorted(all_exit_reasons.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_exits * 100
            logger.info(f"  {reason}: {count} ({pct:.1f}%)")
    
    # Return aggregated results
    return {
        'avg_return_per_trade': avg_return_all_stocks,
        'win_rate': win_rate,
        'total_trades': total_trades_all,
        'successful_stocks': successful_runs,
        'failed_stocks': failed_runs,
        'exit_reasons': all_exit_reasons,
        'individual_results': all_results
    }


def main():
    """Main entry point for factor-based backtest runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run backtest using factor data from parquet file'
    )
    parser.add_argument(
        '--factor-file',
        type=str,
        default=r'D:\trade\quant_system\data\polygon_factors_all_filtered_clean_final_v2_recalculated.parquet',
        help='Path to factor parquet file'
    )
    parser.add_argument(
        '--benchmark',
        default='SPY',
        help='Benchmark symbol (default: SPY)'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='List of tickers to use (if not provided, uses all from factor file)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD). If not provided, uses from factor file'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD). If not provided, uses from factor file'
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
        '--save-merged',
        type=str,
        help='Path to save merged data (optional)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching for OHLCV data'
    )
    
    args = parser.parse_args()
    
    # Run backtest
    result = run_backtest_with_factors(
        factor_file=args.factor_file,
        benchmark_symbol=args.benchmark,
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        max_positions=args.max_positions,
        min_signal_score=args.min_score,
        use_cache=not args.no_cache,
        save_merged=args.save_merged
    )
    
    if result is None:
        logger.error("Backtest failed")
        sys.exit(1)
    
    logger.info("\n" + "="*80)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
