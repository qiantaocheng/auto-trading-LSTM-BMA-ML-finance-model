#!/usr/bin/env python3
"""Sanity checks for the T+10 factor universe."""

import numpy as np
import pandas as pd

from bma_models.simple_25_factor_engine import (
    Simple17FactorEngine,
    T10_ALPHA_FACTORS,
    REQUIRED_14_FACTORS,
)

EXPECTED_FACTORS = list(T10_ALPHA_FACTORS)


def verify_factor_definition() -> bool:
    """Ensure the exported factor list matches the canonical T+10 set."""
    provided = list(REQUIRED_14_FACTORS)
    if provided != EXPECTED_FACTORS:
        missing = [f for f in EXPECTED_FACTORS if f not in provided]
        extra = [f for f in provided if f not in EXPECTED_FACTORS]
        raise AssertionError(f"Factor list mismatch. missing={missing}, extra={extra}")
    return True


def verify_factor_generation() -> pd.DataFrame:
    """Generate sample data and confirm the engine returns every factor."""
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    tickers = ['TEST1', 'TEST2']
    rows = []
    rng = np.random.default_rng(seed=7)
    for ticker in tickers:
        price = 100.0
        for date in dates:
            change = rng.normal(0, 1)
            high = price + abs(change) * 1.5
            low = price - abs(change) * 1.5
            close = price + change
            volume = rng.integers(1_000_000, 2_000_000)
            rows.append({
                'date': date,
                'ticker': ticker,
                'Open': price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume,
            })
            price = close
    market_data = pd.DataFrame(rows)
    engine = Simple17FactorEngine(lookback_days=60, enable_sentiment=False)
    factors = engine.compute_all_17_factors(market_data)

    missing = [f for f in EXPECTED_FACTORS if f not in factors.columns]
    if missing:
        raise AssertionError(f"Missing factors in output: {missing}")
    return factors


def verify_training_columns(factors: pd.DataFrame) -> pd.Index:
    """Ensure that the training feature matrix only uses the canonical factors."""
    feature_cols = [c for c in factors.columns if c not in {'target', 'Close', 'date', 'ticker'}]
    extras = [c for c in feature_cols if c not in EXPECTED_FACTORS]
    if extras:
        raise AssertionError(f"Unexpected factors in training matrix: {extras}")
    return pd.Index(feature_cols)


def main() -> None:
    print('\n' + '=' * 80)
    print('T+10 FACTOR INTEGRATION CHECKS')
    print('=' * 80)

    verify_factor_definition()
    print('? REQUIRED_14_FACTORS matches the canonical T+10 set')

    factors = verify_factor_generation()
    print(f'? Simple17FactorEngine produced all factors (columns={len(factors.columns)})')

    feature_cols = verify_training_columns(factors)
    print(f'? Training feature columns aligned ({len(feature_cols)} factors)')

    print('\nCanonical factor list (ordered):')
    for idx, factor in enumerate(EXPECTED_FACTORS, start=1):
        print(f'  {idx:2d}. {factor}')

    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
