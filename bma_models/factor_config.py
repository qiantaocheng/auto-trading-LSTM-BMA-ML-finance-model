"""
Unified factor configuration.
Defines the standard factor universe used across model training.
"""

from bma_models.simple_25_factor_engine import T10_ALPHA_FACTORS

COMPLETE_FACTOR_SET = list(dict.fromkeys(T10_ALPHA_FACTORS))
FACTOR_COUNT = len(COMPLETE_FACTOR_SET)


LEGACY_FACTOR_MAPPING = {
    'momentum_10d': 'liquid_momentum',
    'momentum_10d_ex1': 'liquid_momentum',
    'momentum_20d': 'liquid_momentum',
    'rsi_7': 'rsi_21',
    'rsi': 'rsi_21',
    'mom_accel_5_2': 'liquid_momentum',
    'mom_accel_10_5': 'liquid_momentum',
    'obv_momentum': 'obv_divergence',
    'obv_momentum_20d': 'obv_divergence',
    'reversal_5d': '5_days_reversal',
    'reversal_1d': '5_days_reversal',
    'price_efficiency_5d': 'trend_r2_60',
    'price_efficiency_10d': 'trend_r2_60',
    'nr7_breakout_bias': 'atr_ratio',
    'overnight_intraday_gap': 'ret_skew_20d',
    'max_lottery_factor': 'ret_skew_20d',
    'stability_score': 'hist_vol_40d',
    'liquidity_factor': 'vol_ratio_20d',
    'adx_14': 'rsrs_beta_18'
}



FACTOR_CATEGORIES = {
    'momentum': ['liquid_momentum'],
    'mean_reversion': ['price_ma60_deviation', '5_days_reversal'],
    'technical': ['rsi_21'],
    'volume_liquidity': ['obv_divergence', 'vol_ratio_20d'],
    'trend': ['trend_r2_60', 'near_52w_high', 'rsrs_beta_18'],
    'volatility': ['atr_ratio', 'ivol_20'],
    'distribution': [],
    'fundamental': [],
    'beta_risk': [],
}



FACTOR_DESCRIPTIONS = {
    'liquid_momentum': 'Liquidity-adjusted momentum that blends volume and price trend',
    'obv_divergence': 'OBV divergence score versus price trend direction',
    'ivol_20': '20-day implied/realized volatility ratio proxy',
    'rsrs_beta_18': 'RSRS trend beta (18 bars) capturing directional conviction',
    'rsi_21': '21-period RSI tuned for smoother mean reversion',
    'trend_r2_60': '60-day trend goodness-of-fit (R-squared)',
    'near_52w_high': 'Distance to 52-week high (252-day window)',
    # 'ret_skew_20d': '20-day return skewness',  # REMOVED
    'atr_ratio': 'Average true range ratio',
    'vol_ratio_20d': '20-day volume spike ratio (volume vs lagged 20-day mean)',
    'price_ma60_deviation': 'Distance to 60-day moving average',
    '5_days_reversal': 'Negative five-day return capturing short-term mean reversion',
}


TARGET_CONFIG = {
    'name': 'ret_fwd_10d',
    'description': 'T+10 forward return',
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
