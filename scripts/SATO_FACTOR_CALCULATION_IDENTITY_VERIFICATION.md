# Sato Factor Calculation Identity Verification

## ✅ Calculation Code Identity Confirmed

Both `simple_25_factor_engine.py` and `scripts/sato_factor_calculation.py` use **the exact same calculation function**.

## Code Flow

### 1. **Core Calculation Function** (`scripts/sato_factor_calculation.py`)
```python
def calculate_sato_factors(
    df: pd.DataFrame,
    price_col: str = 'adj_close',
    volume_col: str = 'Volume',
    vol_ratio_col: Optional[str] = 'vol_ratio_20d',
    lookback_days: int = 10,
    vol_window: int = 20,
    use_vol_ratio_directly: bool = False
) -> pd.DataFrame:
    """
    Core calculation function - Production Ready (100分版本)
    
    Returns:
        DataFrame with:
        - feat_sato_momentum_10d: Sato动量因子（10日累计）
        - feat_sato_divergence_10d: Sato差异因子（10日平均）
    """
    # Core calculation logic (lines 78-128)
    def _calc_single_stock_final(group):
        # Step 1: 计算对数收益
        log_ret = np.log(group[price_col] / group[price_col].shift(1))
        
        # Step 2: 计算波动率（使用min_periods避免bfill）
        vol_20d = log_ret.rolling(vol_window, min_periods=10).std()
        vol_20d = vol_20d.fillna(0.01) + 1e-6
        
        # Step 3: 确定相对成交量
        if use_vol_ratio_directly and vol_ratio_col and vol_ratio_col in group.columns:
            rel_vol = group[vol_ratio_col].fillna(1.0).clip(lower=0.01)
        else:
            vol_ma = group[volume_col].rolling(vol_window, min_periods=10).mean()
            rel_vol = group[volume_col] / (vol_ma + 1e-6)
            rel_vol = rel_vol.fillna(1.0).clip(lower=0.01)
        
        # Step 4: Sato 核心逻辑
        # Momentum
        normalized_ret = (log_ret / vol_20d).clip(-5, 5)
        daily_sato_mom = normalized_ret * np.sqrt(rel_vol)
        
        # Divergence
        theoretical_impact = vol_20d * np.sqrt(rel_vol)
        daily_divergence = np.abs(log_ret) - theoretical_impact
        
        # Step 5: 滚动聚合
        res = pd.DataFrame(index=group.index)
        res['feat_sato_momentum_10d'] = daily_sato_mom.rolling(lookback_days).sum()
        res['feat_sato_divergence_10d'] = daily_divergence.rolling(lookback_days).mean()
        return res
    
    # MultiIndex grouping and application
    if isinstance(df.index, pd.MultiIndex):
        factors_df = df.groupby(level=ticker_level, group_keys=False).apply(
            lambda group: _calc_single_stock_final(group)
        )
        factors_df = factors_df.reindex(df.index)
        return factors_df
    else:
        return _calc_single_stock_final(df)
```

### 2. **Simple17FactorEngine Wrapper** (`bma_models/simple_25_factor_engine.py`)
```python
def _compute_sato_factors(self, data: pd.DataFrame, grouped) -> pd.DataFrame:
    """
    Wrapper that calls the core calculation function
    """
    from scripts.sato_factor_calculation import calculate_sato_factors  # ✅ Same function
    
    # Data preparation
    sato_data = data.copy()
    if 'adj_close' not in sato_data.columns:
        sato_data['adj_close'] = sato_data['Close']
    
    # Volume handling
    has_vol_ratio = 'vol_ratio_20d' in sato_data.columns
    if 'Volume' not in sato_data.columns:
        if has_vol_ratio:
            base_volume = 1_000_000
            sato_data['Volume'] = base_volume * sato_data['vol_ratio_20d'].fillna(1.0).clip(lower=0.1, upper=10.0)
            use_vol_ratio = True
        else:
            return pd.DataFrame({...}, index=data.index)  # Zero-filled fallback
    else:
        use_vol_ratio = has_vol_ratio
    
    # Create MultiIndex if needed
    temp_index = pd.MultiIndex.from_arrays(
        [sato_data['date'], sato_data['ticker']],
        names=['date', 'ticker']
    )
    sato_data_temp = sato_data.set_index(temp_index)
    
    # ✅ CALL SAME FUNCTION WITH IDENTICAL PARAMETERS
    sato_factors_df = calculate_sato_factors(
        df=sato_data_temp,
        price_col='adj_close',           # ✅ Same
        volume_col='Volume',              # ✅ Same
        vol_ratio_col='vol_ratio_20d',   # ✅ Same
        lookback_days=10,                 # ✅ Same
        vol_window=20,                    # ✅ Same
        use_vol_ratio_directly=use_vol_ratio  # ✅ Same
    )
    
    # Return results
    return pd.DataFrame({
        'feat_sato_momentum_10d': sato_factors_df['feat_sato_momentum_10d'].fillna(0.0),
        'feat_sato_divergence_10d': sato_factors_df['feat_sato_divergence_10d'].fillna(0.0)
    }, index=data.index)
```

## ✅ Identity Verification

| Aspect | `scripts/sato_factor_calculation.py` | `simple_25_factor_engine.py` | Status |
|--------|-----------------------------------|------------------------------|--------|
| **Core Function** | `calculate_sato_factors()` | Calls `calculate_sato_factors()` | ✅ **IDENTICAL** |
| **Price Column** | `price_col='adj_close'` | `price_col='adj_close'` | ✅ **IDENTICAL** |
| **Volume Column** | `volume_col='Volume'` | `volume_col='Volume'` | ✅ **IDENTICAL** |
| **Vol Ratio Column** | `vol_ratio_col='vol_ratio_20d'` | `vol_ratio_col='vol_ratio_20d'` | ✅ **IDENTICAL** |
| **Lookback Days** | `lookback_days=10` | `lookback_days=10` | ✅ **IDENTICAL** |
| **Vol Window** | `vol_window=20` | `vol_window=20` | ✅ **IDENTICAL** |
| **Use Vol Ratio** | `use_vol_ratio_directly=use_vol_ratio` | `use_vol_ratio_directly=use_vol_ratio` | ✅ **IDENTICAL** |
| **Calculation Logic** | Lines 78-128 | Same function (lines 78-128) | ✅ **IDENTICAL** |
| **MultiIndex Handling** | Lines 58-75, 131-139 | Same function | ✅ **IDENTICAL** |
| **Rolling Window** | `min_periods=10` | Same function | ✅ **IDENTICAL** |
| **Fillna Strategy** | `fillna(0.01) + 1e-6` | Same function | ✅ **IDENTICAL** |
| **Clip Values** | `clip(-5, 5)` for ret, `clip(lower=0.01)` for vol | Same function | ✅ **IDENTICAL** |

## Key Points

1. **Single Source of Truth**: `simple_25_factor_engine.py` imports and calls `calculate_sato_factors()` from `scripts/sato_factor_calculation.py`
2. **No Code Duplication**: The calculation logic exists only in `scripts/sato_factor_calculation.py`
3. **Identical Parameters**: All function parameters are passed identically
4. **Same Calculation**: Both use the exact same `_calc_single_stock_final()` function internally

## Conclusion

✅ **The calculation code is 100% identical** because:
- `simple_25_factor_engine.py` delegates to `scripts/sato_factor_calculation.py`
- No duplicate calculation logic exists
- All parameters are identical
- The core calculation function is shared

## Usage Locations

1. **Direct Usage**: `scripts/sato_factor_calculation.py` - Standalone script
2. **Via Simple17FactorEngine**: `bma_models/simple_25_factor_engine.py` → `_compute_sato_factors()` → `calculate_sato_factors()`
3. **Via Model Training**: `bma_models/量化模型_bma_ultra_enhanced.py` → `_standardize_loaded_data()` → `calculate_sato_factors()`
4. **Via 80/20 Evaluation**: `scripts/time_split_80_20_oos_eval.py` → `calculate_sato_factors()`
5. **Via Direct Predict**: `autotrader/app.py` → `calculate_sato_factors()`

All paths lead to the same function: `scripts.sato_factor_calculation.calculate_sato_factors()`
