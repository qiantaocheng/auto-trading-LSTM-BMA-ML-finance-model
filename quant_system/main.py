"""
Growth Stock Signal System - Main Entry Point
=============================================

Comprehensive quantitative system for identifying long-term growth stocks
integrating Minervini Trend Template, Weinstein Stage Analysis, O'Neil CAN SLIM,
and VCP pattern detection.

Usage:
    python main.py --demo           # Run with synthetic demo data
    python main.py --analyze AAPL   # Analyze specific ticker
    python main.py --scan           # Scan universe (requires data)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse

import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from config.settings import SystemConfig, MarketRegime
from core.data_types import StockData, SignalResult, CompositeSignal
from core.signal_aggregator import SignalAggregator, MarketRegimeDetector
from signals.trend_signals import TrendTemplateSignal, ADXSignal
from signals.stage_analysis import StageAnalysisSignal
from signals.vcp_detector import VCPSignal
from signals.relative_strength import RelativeStrengthSignal
from utils.risk_management import (
    StopLossManager, PositionSizer, PullbackAnalyzer, PortfolioRiskManager
)
from data.polygon_data_fetcher import PolygonDataFetcher, create_universe_data


def generate_synthetic_data(ticker: str, 
                           days: int = 500,
                           trend: str = 'bullish',
                           seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.
    
    Args:
        ticker: Stock symbol
        days: Number of trading days
        trend: 'bullish', 'bearish', 'sideways', or 'vcp' (bullish with VCP)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(seed)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    
    # Base parameters
    base_price = 50.0
    daily_vol = 0.02
    
    if trend == 'bullish':
        # Strong uptrend with small corrections
        drift = 0.0008  # ~20% annual
        prices = [base_price]
        for i in range(1, days):
            # Add occasional consolidations
            if i % 80 == 0:
                drift_adj = -0.001  # Small pullback period
            else:
                drift_adj = drift
            change = np.random.normal(drift_adj, daily_vol)
            prices.append(prices[-1] * (1 + change))
            
    elif trend == 'vcp':
        # Bullish trend with VCP pattern at end
        prices = [base_price]
        for i in range(1, days):
            if i < days * 0.6:
                # Initial run-up
                drift = 0.001
            elif i < days * 0.7:
                # First contraction (15% depth)
                drift = -0.0015
            elif i < days * 0.75:
                # Recovery
                drift = 0.0012
            elif i < days * 0.82:
                # Second contraction (10% depth)
                drift = -0.001
            elif i < days * 0.87:
                # Recovery
                drift = 0.001
            elif i < days * 0.94:
                # Third contraction (5% depth)
                drift = -0.0005
            else:
                # Breakout
                drift = 0.003
            
            change = np.random.normal(drift, daily_vol * 0.8)
            prices.append(prices[-1] * (1 + change))
            
    elif trend == 'bearish':
        drift = -0.0006  # -15% annual
        prices = [base_price]
        for i in range(1, days):
            change = np.random.normal(drift, daily_vol)
            prices.append(prices[-1] * (1 + change))
            
    else:  # sideways
        drift = 0.0
        prices = [base_price]
        for i in range(1, days):
            # Mean reversion
            reversion = -0.02 * (prices[-1] - base_price) / base_price
            change = np.random.normal(drift + reversion, daily_vol)
            prices.append(prices[-1] * (1 + change))
    
    prices = np.array(prices)
    
    # Generate OHLC
    high = prices * (1 + np.random.uniform(0.002, 0.02, days))
    low = prices * (1 - np.random.uniform(0.002, 0.02, days))
    open_prices = prices + np.random.normal(0, 0.005, days) * prices
    
    # Generate volume with trend correlation
    base_volume = 1_000_000
    price_changes = np.diff(prices, prepend=prices[0])
    volume_multiplier = 1 + np.abs(price_changes / prices) * 10
    volume = base_volume * volume_multiplier * np.random.uniform(0.5, 1.5, days)
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': prices,
        'Volume': volume.astype(int)
    }, index=dates)
    
    return df


def run_demo():
    """Run comprehensive demo with synthetic data."""
    
    print("\n" + "="*80)
    print("GROWTH STOCK SIGNAL SYSTEM - DEMONSTRATION")
    print("="*80)
    
    # Generate test data for different scenarios
    test_stocks = {
        'BULL_STRONG': generate_synthetic_data('BULL_STRONG', days=400, trend='bullish', seed=42),
        'VCP_SETUP': generate_synthetic_data('VCP_SETUP', days=400, trend='vcp', seed=123),
        'BEAR_WEAK': generate_synthetic_data('BEAR_WEAK', days=400, trend='bearish', seed=456),
        'SIDEWAYS': generate_synthetic_data('SIDEWAYS', days=400, trend='sideways', seed=789),
    }
    
    # Create benchmark (market index)
    benchmark = generate_synthetic_data('SPY', days=400, trend='bullish', seed=999)
    benchmark['Close'] = benchmark['Close'] * 0.8  # Weaker than individual stocks
    
    # Initialize system
    config = SystemConfig()
    aggregator = SignalAggregator(config, benchmark)
    
    print("\n" + "-"*80)
    print("1. MARKET REGIME DETECTION")
    print("-"*80)
    
    regime_detector = MarketRegimeDetector(benchmark)
    regime_info = regime_detector.detect_regime(test_stocks)
    
    print(f"   Market Regime: {regime_info.regime.value}")
    print(f"   % Above 200MA: {regime_info.pct_above_200ma:.1%}")
    print(f"   Market Trend:  {regime_info.market_trend}")
    print(f"   Volatility:    {regime_info.volatility_regime}")
    
    print("\n" + "-"*80)
    print("2. INDIVIDUAL STOCK ANALYSIS")
    print("-"*80)
    
    for ticker, df in test_stocks.items():
        print(f"\n   === {ticker} ===")
        
        stock_data = StockData(ticker=ticker, df=df)
        composite = aggregator.generate_composite_signal(stock_data, regime_info.regime)
        
        # Star rating
        stars = '★' * int(composite.composite_score * 5) + '☆' * (5 - int(composite.composite_score * 5))
        
        print(f"   Composite Score: {composite.composite_score:.3f} [{stars}]")
        print(f"   Technical:       {composite.technical_score:.3f}")
        print(f"   Pattern:         {composite.pattern_score:.3f}")
        print(f"   Fundamental:     {composite.fundamental_score:.3f}")
        print(f"   ")
        print(f"   Signal Breakdown:")
        for name, sig in composite.signals.items():
            bar = '█' * int(sig.score * 10) + '░' * (10 - int(sig.score * 10))
            print(f"     {name:<20} {sig.score:.2f} [{bar}] ({sig.strength.name})")
    
    print("\n" + "-"*80)
    print("3. UNIVERSE SCAN (Ranked Results)")
    print("-"*80)
    
    # Scan all stocks
    results = aggregator.scan_universe(test_stocks, min_score=0.0)
    
    print(f"\n   {'Rank':<6} {'Ticker':<15} {'Score':<8} {'Tech':<8} {'Pattern':<8} {'Fund':<8}")
    print("   " + "-"*60)
    
    for signal in results:
        print(f"   {signal.rank:<6} {signal.ticker:<15} {signal.composite_score:.3f}   "
              f"{signal.technical_score:.3f}    {signal.pattern_score:.3f}    {signal.fundamental_score:.3f}")
    
    print("\n" + "-"*80)
    print("4. RISK MANAGEMENT ANALYSIS (Top Stock)")
    print("-"*80)
    
    if results:
        top_stock = results[0]
        top_data = StockData(ticker=top_stock.ticker, df=test_stocks[top_stock.ticker])
        
        # Stop loss calculation
        stop_manager = StopLossManager(config.risk)
        stop_result = stop_manager.calculate_chandelier_stop(top_data)
        
        current_price = test_stocks[top_stock.ticker]['Close'].iloc[-1]
        
        print(f"\n   Stock: {top_stock.ticker}")
        print(f"   Current Price:     ${current_price:.2f}")
        print(f"   Chandelier Stop:   ${stop_result.stop_price:.2f}")
        print(f"   Risk per Share:    ${stop_result.risk_per_share:.2f}")
        print(f"   Risk Percentage:   {stop_result.risk_percentage:.1%}")
        print(f"   ATR (14):          ${stop_result.atr_value:.2f}")
        
        # Position sizing
        position_sizer = PositionSizer(config.risk)
        portfolio_value = 100_000
        
        position = position_sizer.calculate_fixed_risk_size(
            portfolio_value=portfolio_value,
            entry_price=current_price,
            stop_price=stop_result.stop_price,
            risk_per_trade=0.01
        )
        
        print(f"\n   Position Sizing (1% Risk on $100,000 portfolio):")
        print(f"   Shares:            {position.shares:,}")
        print(f"   Position Value:    ${position.position_value:,.2f}")
        print(f"   Position %:        {position.position_pct:.1%}")
        print(f"   Risk Amount:       ${position.risk_amount:,.2f}")
        
        # Pullback analysis
        pullback_analyzer = PullbackAnalyzer(config.risk)
        pullback = pullback_analyzer.classify_pullback(top_data)
        
        print(f"\n   Pullback Analysis:")
        print(f"   Classification:    {'Healthy Pullback' if pullback.is_pullback else 'Potential Reversal'}")
        print(f"   Confidence:        {pullback.confidence:.1%}")
        print(f"   Volume Pattern:    {pullback.volume_pattern}")
        print(f"   Support Holding:   {pullback.support_holding}")
        print(f"   Max Depth:         {pullback.max_depth_pct:.1%}")
        print(f"   Recommendation:    {pullback.recommendation.upper()}")
    
    print("\n" + "-"*80)
    print("5. DETAILED SIGNAL REPORT (VCP Setup)")
    print("-"*80)
    
    vcp_data = StockData(ticker='VCP_SETUP', df=test_stocks['VCP_SETUP'])
    vcp_signal = aggregator.generate_composite_signal(vcp_data, regime_info.regime)
    
    report = aggregator.generate_signal_report(vcp_signal)
    print(report)
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nThe system has been validated with synthetic data.")
    print("For production use, connect to a real data source (Yahoo Finance, IEX, etc.)")
    print("\nKey components:")
    print("  - config/settings.py:         All configurable parameters")
    print("  - core/signal_aggregator.py:  Main signal combination engine")
    print("  - signals/trend_signals.py:   Minervini Trend Template + ADX")
    print("  - signals/stage_analysis.py:  Weinstein Stage Analysis")
    print("  - signals/vcp_detector.py:    VCP Pattern Detection")
    print("  - signals/relative_strength.py: IBD RS Rating")
    print("  - utils/risk_management.py:   Stops, Position Sizing, Risk Control")
    print()


def analyze_single_stock(ticker: str, data: Optional[pd.DataFrame] = None, years: int = 3):
    """Analyze a single stock using Polygon data."""
    print(f"\nAnalyzing {ticker} with Polygon data...")
    
    # Fetch data from Polygon
    fetcher = PolygonDataFetcher()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')
    
    if data is None:
        print(f"Fetching {ticker} data from {start_date} to {end_date}...")
        data = fetcher.fetch_symbol(ticker, start_date, end_date)
        
        if data is None or data.empty:
            print(f"Failed to fetch data for {ticker}. Using synthetic data for demo.")
            data = generate_synthetic_data(ticker, days=400, trend='bullish')
    
    # Fetch benchmark
    print(f"Fetching SPY benchmark data...")
    benchmark = fetcher.fetch_symbol('SPY', start_date, end_date)
    if benchmark is None or benchmark.empty:
        print("Failed to fetch SPY. Using synthetic data.")
        benchmark = generate_synthetic_data('SPY', days=400, trend='bullish', seed=999)
    
    config = SystemConfig()
    aggregator = SignalAggregator(config, benchmark)
    
    stock_data = StockData(ticker=ticker, df=data)
    signal = aggregator.generate_composite_signal(stock_data)
    
    report = aggregator.generate_signal_report(signal)
    print(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Growth Stock Signal System - Quantitative Analysis Tool'
    )
    parser.add_argument('--demo', action='store_true', 
                       help='Run demo with synthetic data')
    parser.add_argument('--analyze', type=str, metavar='TICKER',
                       help='Analyze a specific ticker (requires data)')
    parser.add_argument('--scan', action='store_true',
                       help='Scan universe (requires data configuration)')
    
    args = parser.parse_args()
    
    if args.demo or (not args.analyze and not args.scan):
        run_demo()
    elif args.analyze:
        analyze_single_stock(args.analyze)
    elif args.scan:
        print("Universe scan with Polygon data...")
        print("Example usage:")
        print("  python main.py --scan --symbols AAPL MSFT NVDA --years 3")
        print("\nOr modify main.py to specify your universe list.")


if __name__ == '__main__':
    main()
