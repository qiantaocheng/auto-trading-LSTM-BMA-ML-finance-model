"""Test signal generation with factor data"""
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.factor_data_loader import FactorDataLoader
from core.signal_aggregator import SignalAggregator
from core.data_types import StockData
from config.settings import SystemConfig
from data.polygon_data_fetcher import PolygonDataFetcher

# Load data
loader = FactorDataLoader(r'D:\trade\quant_system\data\polygon_factors_all_2021_2026_T5_final.parquet')
loader.load_factors()

# Get small subset
merged = loader.merge_factors_with_ohlcv(tickers=['AAPL'], use_cache=True)
universe_dict = loader.get_universe_dict(merged)

# Get benchmark
fetcher = PolygonDataFetcher()
benchmark = fetcher.fetch_symbol('SPY', '2021-01-19', '2024-12-31', use_cache=True)

# Test signal generation
config = SystemConfig()
aggregator = SignalAggregator(config=config, benchmark_data=benchmark)

print(f"Testing signal generation for AAPL...")
print(f"AAPL data shape: {universe_dict['AAPL'].shape}")
print(f"AAPL columns: {universe_dict['AAPL'].columns.tolist()[:10]}...")

# Test single stock
stock_data = StockData(ticker='AAPL', df=universe_dict['AAPL'])
signal = aggregator.generate_composite_signal(stock_data)

print(f"\nSignal generated:")
print(f"  Composite Score: {signal.composite_score:.3f}")
print(f"  Technical Score: {signal.technical_score:.3f}")
print(f"  Pattern Score: {signal.pattern_score:.3f}")
print(f"  Fundamental Score: {signal.fundamental_score:.3f}")

# Test universe scan
print(f"\nTesting universe scan...")
signals = aggregator.scan_universe(universe_dict, min_score=0.30)
print(f"Found {len(signals)} signals with score >= 0.30")
for sig in signals[:5]:
    print(f"  {sig.ticker}: {sig.composite_score:.3f}")
