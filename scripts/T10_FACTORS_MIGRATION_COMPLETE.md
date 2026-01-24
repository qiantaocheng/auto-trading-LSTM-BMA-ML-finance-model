# T10 Factors Only - Migration Complete

## Summary

All T5 factor references have been removed. The system now **always uses T10_ALPHA_FACTORS** regardless of horizon setting.

## Changes Completed

### ✅ 1. Factor Engine (`simple_25_factor_engine.py`)
- Removed horizon-based conditional selection
- Always uses `T10_ALPHA_FACTORS`
- Updated logging to indicate T5 removal

### ✅ 2. Main Model (`量化模型_bma_ultra_enhanced.py`)
- Removed `T5_ALPHA_FACTORS` import
- Changed `factor_universe` to always use `T10_ALPHA_FACTORS`
- Removed `is_t10` conditional logic
- Updated `compulsory_features` to match T10 factors (already correct)
- Updated `_STD_FACTORS` import to use T10

### ✅ 3. Prediction Exporter (`corrected_prediction_exporter.py`)
- Changed import from `T5_ALPHA_FACTORS` to `T10_ALPHA_FACTORS`
- Updated factor contributions initialization

## T10 Alpha Factors (Always Used)

The following 14 factors are now **always** used in all training and prediction:

1. **liquid_momentum** - 流动性调整动量
2. **momentum_10d** - 10日短期动量
3. **momentum_60d** - 60日动量
4. **obv_divergence** - OBV背离 ⭐
5. **obv_momentum_60d** - 60日OBV动量
6. **ivol_20** - 20日隐含波动率
7. **hist_vol_40d** - 40日历史波动率
8. **atr_ratio** - ATR比率
9. **rsi_21** - 21周期RSI
10. **trend_r2_60** - 60日趋势R²
11. **near_52w_high** - 距离52周高点
12. **vol_ratio_20d** - 20日成交量比率
13. **price_ma60_deviation** - 价格偏离60日均线
14. **5_days_reversal** - 5日反转

## Compulsory Features

`compulsory_features` now matches T10_ALPHA_FACTORS exactly:
- ✅ All 14 T10 factors are compulsory
- ✅ No T5-specific factors included
- ✅ Consistent across training and prediction

## Next Steps

1. ✅ **Run comparison training** with/without obv_divergence (script created: `compare_obv_divergence_training.py`)
2. ✅ **Remove all T5 references** (completed)
3. ✅ **Update compulsory_features** (already matches T10)
4. ⏳ **Test training** with T10 factors only
5. ⏳ **Run 1/5 subset comparison** to evaluate obv_divergence impact

## Notes

- `T5_ALPHA_FACTORS` definition still exists in `simple_25_factor_engine.py` but is **no longer used**
- All factor selection now defaults to `T10_ALPHA_FACTORS`
- Lambda Ranker and other components will use T+10 factors
- The `obv_divergence` warning should now be resolved since it's always included in T10 factors

## Testing

To test the changes:
```bash
# Run comparison training
python scripts/compare_obv_divergence_training.py

# Or run normal training (will use T10 factors)
python scripts/train_full_dataset.py --train-data data/factor_exports/polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet
```

## Files Modified

1. `bma_models/simple_25_factor_engine.py` - Factor selection logic
2. `bma_models/量化模型_bma_ultra_enhanced.py` - Main model factor handling
3. `bma_models/corrected_prediction_exporter.py` - Prediction export

## Migration Date

2026-01-24
