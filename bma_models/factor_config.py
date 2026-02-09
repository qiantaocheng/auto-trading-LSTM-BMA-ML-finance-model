"""
Unified factor configuration.
Defines the standard factor universe used across model training.
"""

from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS

COMPLETE_FACTOR_SET = list(dict.fromkeys(T10_ALPHA_FACTORS))
FACTOR_COUNT = len(COMPLETE_FACTOR_SET)


LEGACY_FACTOR_MAPPING = {
    'rsi_7': 'rsi_14',
    'rsi': 'rsi_14',
    'rsi_21': 'rsi_14',
    'obv_momentum': 'liquid_momentum_10d',
    'obv_momentum_20d': 'liquid_momentum_10d',
    'reversal_5d': 'reversal_5d',
    'reversal_1d': 'reversal_3d',
    'price_efficiency_5d': 'trend_r2_20',
    'price_efficiency_10d': 'trend_r2_20',
    'nr7_breakout_bias': 'atr_pct_14',
    'overnight_intraday_gap': 'atr_pct_14',
    'max_lottery_factor': 'atr_pct_14',
    'stability_score': 'atr_pct_14',
    'liquidity_factor': 'dollar_vol_20',
    'volume_price_corr_5d': 'volume_price_corr_3d',
    'volume_price_corr_10d': 'volume_price_corr_3d',
}



FACTOR_CATEGORIES = {
    'momentum': ['momentum_10d', 'liquid_momentum_10d', 'sharpe_momentum_5d'],
    'mean_reversion': ['reversal_3d', 'reversal_5d', 'ret_skew_20d'],
    'technical': ['volume_price_corr_3d', 'rsi_14', 'price_ma20_deviation', 'trend_r2_20', 'near_52w_high'],
    'volume_liquidity': ['avg_trade_size', 'dollar_vol_20', 'amihud_20'],
    'volatility': ['atr_pct_14'],
    'distribution': [],
    'fundamental': [],
    'beta_risk': [],
}



FACTOR_DESCRIPTIONS = {
    'volume_price_corr_3d': '3-day rolling correlation between returns and volume',
    'rsi_14': '14-period RSI tuned for smoother mean reversion',
    'reversal_3d': 'Negative 3-day return capturing short-term reversal',
    'momentum_10d': '10-day price momentum',
    'liquid_momentum_10d': 'Liquidity-weighted 10-day momentum',
    'sharpe_momentum_5d': '5-day momentum divided by volatility',
    'price_ma20_deviation': 'Percentage deviation from the 20-day moving average',
    'avg_trade_size': '20-day rolling average of volume per transaction',
    'trend_r2_20': '20-day trend line goodness-of-fit',
    'dollar_vol_20': '20-day average dollar volume (Close * Volume)',
    'ret_skew_20d': '20-day return skewness',
    'reversal_5d': 'Negative 5-day return',
    'near_52w_high': 'Distance to 52-week high',
    'atr_pct_14': 'ATR14 expressed as a percent of price',
    'amihud_20': '20-day Amihud illiquidity measure',
}


TARGET_CONFIG = {
    'name': 'ret_fwd_5d',
    'description': 'T+5 forward return',
    'formula': '(Close_{t+10} - Close_t) / Close_t',
}


def validate_factor_data(factor_data, required_factors=None):
    """Validate that the factor data contains every required factor."""
    if required_factors is None:
        required_factors = COMPLETE_FACTOR_SET

    available_factors = [f for f in required_factors if f in factor_data.columns]
    missing_factors = [f for f in required_factors if f not in factor_data.columns]

    return {
        'is_valid': len(missing_factors) == 0,
        'required_count': len(required_factors),
        'available_count': len(available_factors),
        'missing_count': len(missing_factors),
        'available_factors': available_factors,
        'missing_factors': missing_factors,
        'coverage': (len(available_factors) / len(required_factors) * 100) if required_factors else 0,
    }


def get_factor_subset(categories=None):
    """Return a subset of factors filtered by category names."""
    if categories is None:
        return COMPLETE_FACTOR_SET.copy()

    factors = []
    for category in categories:
        factors.extend(FACTOR_CATEGORIES.get(category, []))

    seen = set()
    ordered_factors = []
    for factor in COMPLETE_FACTOR_SET:
        if factor in factors and factor not in seen:
            seen.add(factor)
            ordered_factors.append(factor)

    return ordered_factors


def print_factor_summary():
    """Print a summary of the factor configuration."""
    print("=" * 80)
    print("Standard Factor Configuration")
    print("=" * 80)
    print(f"Total factors: {FACTOR_COUNT}")
    print("\nFactor list:")
    for i, factor in enumerate(COMPLETE_FACTOR_SET, 1):
        category = next((cat for cat, vals in FACTOR_CATEGORIES.items() if factor in vals), 'unclassified')
        desc = FACTOR_DESCRIPTIONS.get(factor, 'n/a')
        print(f"  {i:2d}. {factor:25s} [{category:14s}] - {desc}")

    print("\nCategory counts:")
    for category, vals in FACTOR_CATEGORIES.items():
        print(f"  {category:14s}: {len(vals):2d} factors")

    print("\nPrediction target:")
    print(f"  name: {TARGET_CONFIG['name']}")
    print(f"  description: {TARGET_CONFIG['description']}")
    print(f"  formula: {TARGET_CONFIG['formula']}")
    print("=" * 80)


if __name__ == '__main__':
    print_factor_summary()

