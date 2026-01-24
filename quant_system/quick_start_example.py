"""
Quick Start Example - Polygon Data Integration
==============================================

This example demonstrates how to:
1. Fetch data from Polygon.io
2. Run signal analysis
3. Execute backtest
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.polygon_data_fetcher import PolygonDataFetcher
from core.signal_aggregator import SignalAggregator
from core.data_types import StockData
from config.settings import SystemConfig
from backtest.engine import BacktestEngine, BacktestConfig, generate_backtest_report


def example_fetch_and_analyze():
    """Example: Fetch data and analyze single stock."""
    print("="*80)
    print("EXAMPLE 1: Fetch Data and Analyze Single Stock")
    print("="*80)
    
    # Initialize fetcher
    fetcher = PolygonDataFetcher()
    
    # Fetch data for single stock
    symbol = 'AAPL'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    
    print(f"\nFetching {symbol} data from {start_date} to {end_date}...")
    df = fetcher.fetch_symbol(symbol, start_date, end_date)
    
    if df is None or df.empty:
        print(f"Failed to fetch data for {symbol}")
        return
    
    print(f"Successfully fetched {len(df)} days of data")
    print(f"\nData preview:")
    print(df.head())
    
    # Fetch benchmark
    print(f"\nFetching SPY benchmark...")
    benchmark = fetcher.fetch_symbol('SPY', start_date, end_date)
    
    # Initialize signal system
    config = SystemConfig()
    aggregator = SignalAggregator(config=config, benchmark_data=benchmark)
    
    # Generate signal
    stock_data = StockData(ticker=symbol, df=df)
    signal = aggregator.generate_composite_signal(stock_data)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"SIGNAL ANALYSIS FOR {symbol}")
    print(f"{'='*80}")
    print(f"Composite Score: {signal.composite_score:.3f}")
    print(f"Technical Score: {signal.technical_score:.3f}")
    print(f"Pattern Score: {signal.pattern_score:.3f}")
    print(f"Fundamental Score: {signal.fundamental_score:.3f}")
    
    report = aggregator.generate_signal_report(signal)
    print(report)


def example_universe_backtest():
    """Example: Fetch universe and run backtest."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Universe Backtest")
    print("="*80)
    
    # Define universe
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    print(f"\nUniverse: {', '.join(symbols)}")
    
    # Initialize fetcher
    fetcher = PolygonDataFetcher()
    
    # Fetch universe data (MultiIndex format)
    print("\nFetching universe data (this may take a few minutes)...")
    universe_multiindex = fetcher.fetch_universe(
        symbols=symbols,
        years=3,
        use_cache=True
    )
    
    if universe_multiindex.empty:
        print("Failed to fetch universe data")
        return
    
    print(f"Successfully fetched data for {len(universe_multiindex.index.get_level_values(0).unique())} symbols")
    
    # Convert to dict format
    universe_dict = fetcher.get_universe_dict(universe_multiindex)
    
    # Fetch benchmark
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    
    print(f"\nFetching SPY benchmark...")
    benchmark = fetcher.fetch_symbol('SPY', start_date, end_date)
    
    if benchmark is None or benchmark.empty:
        print("Failed to fetch benchmark")
        return
    
    # Initialize signal system
    config = SystemConfig()
    aggregator = SignalAggregator(config=config, benchmark_data=benchmark)
    
    # Configure backtest
    backtest_config = BacktestConfig(
        initial_capital=100000.0,
        max_positions=5,
        min_signal_score=0.60,
        rebalance_frequency='weekly'
    )
    
    # Initialize backtest engine
    print("\nInitializing backtest engine...")
    engine = BacktestEngine(
        signal_aggregator=aggregator,
        config=backtest_config,
        risk_config=config.risk
    )
    
    # Run backtest
    print(f"\nRunning backtest from {start_date} to {end_date}...")
    print("(This may take several minutes)")
    
    try:
        result = engine.run(
            universe_data=universe_dict,
            benchmark_data=benchmark,
            start_date=start_date,
            end_date=end_date
        )
        
        # Print results
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        
        report = generate_backtest_report(result, "Example Strategy")
        print(report)
        
        print(f"\nAdditional Statistics:")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        
        if hasattr(result, 'equity_curve') and not result.equity_curve.empty:
            print(f"  Final Equity: ${result.equity_curve.iloc[-1]:,.2f}")
            print(f"  Peak Equity: ${result.equity_curve.max():,.2f}")
        
    except Exception as e:
        print(f"\nBacktest failed: {e}")
        import traceback
        traceback.print_exc()


def example_multiindex_operations():
    """Example: Working with MultiIndex DataFrame."""
    print("\n" + "="*80)
    print("EXAMPLE 3: MultiIndex DataFrame Operations")
    print("="*80)
    
    fetcher = PolygonDataFetcher()
    
    # Fetch small universe
    symbols = ['AAPL', 'MSFT']
    universe_df = fetcher.fetch_universe(symbols=symbols, years=1, use_cache=True)
    
    if universe_df.empty:
        print("Failed to fetch data")
        return
    
    print(f"\nMultiIndex DataFrame shape: {universe_df.shape}")
    print(f"\nIndex levels: {universe_df.index.names}")
    print(f"\nUnique symbols: {universe_df.index.get_level_values(0).unique().tolist()}")
    
    print(f"\nFirst few rows:")
    print(universe_df.head(10))
    
    # Access data for specific symbol
    print(f"\n{'='*80}")
    print("AAPL Data (first 5 rows):")
    print("="*80)
    if 'AAPL' in universe_df.index.get_level_values(0):
        aapl_data = universe_df.loc['AAPL']
        print(aapl_data.head())
    
    # Convert to dict format
    print(f"\n{'='*80}")
    print("Converting to dictionary format...")
    print("="*80)
    universe_dict = fetcher.get_universe_dict(universe_df)
    print(f"Dictionary keys: {list(universe_dict.keys())}")
    print(f"AAPL DataFrame shape: {universe_dict['AAPL'].shape}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick start examples')
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3],
        default=1,
        help='Example to run: 1=Single stock analysis, 2=Universe backtest, 3=MultiIndex operations'
    )
    
    args = parser.parse_args()
    
    if args.example == 1:
        example_fetch_and_analyze()
    elif args.example == 2:
        example_universe_backtest()
    elif args.example == 3:
        example_multiindex_operations()
    else:
        print("Running all examples...")
        example_fetch_and_analyze()
        example_multiindex_operations()
        # Skip backtest by default as it takes longer
        # Uncomment to run:
        # example_universe_backtest()
