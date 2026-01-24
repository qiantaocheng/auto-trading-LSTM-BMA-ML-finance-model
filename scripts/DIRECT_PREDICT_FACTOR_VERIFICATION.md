# Direct Prediction Factor Verification

## Verification Date
2026-01-24

## Objective
Ensure that direct prediction uses the same factors as the 80/20 training script.

## Results

### ✅ VERIFICATION PASSED

**Training Script Factors (14):**
1. liquid_momentum
2. momentum_10d
3. momentum_60d
4. obv_divergence
5. obv_momentum_60d
6. ivol_20
7. hist_vol_40d
8. atr_ratio
9. rsi_21
10. trend_r2_60
11. near_52w_high
12. vol_ratio_20d
13. price_ma60_deviation
14. 5_days_reversal

**Direct Prediction Factors (T10_ALPHA_FACTORS, 14):**
1. liquid_momentum
2. momentum_10d
3. momentum_60d
4. obv_divergence
5. obv_momentum_60d
6. ivol_20
7. hist_vol_40d
8. atr_ratio
9. rsi_21
10. trend_r2_60
11. near_52w_high
12. vol_ratio_20d
13. price_ma60_deviation
14. 5_days_reversal

### Comparison Result
- **Perfect Match**: All 14 factors are identical
- **No missing factors**: Training script has no factors not in T10_ALPHA_FACTORS
- **No extra factors**: T10_ALPHA_FACTORS has no factors not in training script

### Removed Factors Check
All removed factors are correctly absent:
- ✅ ret_skew_20d - REMOVED from both
- ✅ making_new_low_5d - REMOVED from both
- ✅ bollinger_squeeze - REMOVED from both
- ✅ blowoff_ratio - REMOVED from both
- ✅ downside_beta_252 - REMOVED from both
- ✅ downside_beta_ewm_21 - REMOVED from both
- ✅ roa - REMOVED from both
- ✅ ebit - REMOVED from both

## How It Works

### Training Script (`scripts/time_split_80_20_oos_eval.py`)
- Uses `allowed_feature_cols` list (line ~1743)
- Filters test data to only use these 14 factors
- All factors are from `T10_ALPHA_FACTORS`

### Direct Prediction (`scripts/direct_predict_ewma_excel.py`)
- Uses `UltraEnhancedQuantitativeModel.predict_with_snapshot()`
- Which calls `Simple17FactorEngine.compute_all_17_factors()`
- `Simple17FactorEngine` uses `T10_ALPHA_FACTORS` when `horizon >= 10`
- All factors are computed dynamically from market data

### Factor Engine (`bma_models/simple_25_factor_engine.py`)
- `T10_ALPHA_FACTORS` contains 14 factors (ret_skew_20d removed)
- `Simple17FactorEngine.__init__()` selects factors based on horizon:
  ```python
  self.alpha_factors = T10_ALPHA_FACTORS if horizon_value >= 10 else T5_ALPHA_FACTORS
  ```
- `compute_all_17_factors()` uses `self.alpha_factors` to compute only the required factors

## Conclusion

✅ **Direct prediction and training use IDENTICAL factors!**

- Both use the same 14 factors from `T10_ALPHA_FACTORS`
- All removed factors are correctly absent
- Factor computation is consistent between training and prediction
- No modifications needed

## Verification Script
Run `scripts/verify_direct_predict_factors.py` to re-verify at any time.
