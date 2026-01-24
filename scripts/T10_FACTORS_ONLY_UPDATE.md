# T10 Factors Only - Update Summary

## Changes Made

### 1. **simple_25_factor_engine.py**
- ✅ Changed factor selection logic to **always use T10_ALPHA_FACTORS**
- ✅ Removed horizon-based conditional (T5 vs T10)
- ✅ Updated logging to indicate T5 factors removed

### 2. **量化模型_bma_ultra_enhanced.py**
- ✅ Removed T5_ALPHA_FACTORS import
- ✅ Changed `factor_universe` to always use T10_ALPHA_FACTORS
- ✅ Removed `is_t10` conditional logic
- ✅ Updated `_STD_FACTORS` import to use T10_ALPHA_FACTORS

### 3. **corrected_prediction_exporter.py**
- ✅ Changed import from T5_ALPHA_FACTORS to T10_ALPHA_FACTORS
- ✅ Updated factor_contributions initialization

## T10 Alpha Factors (Always Used)

The following 14 factors are now always used:

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

## Next Steps

1. ✅ Run comparison training with/without obv_divergence
2. ✅ Remove all T5 references
3. ⏳ Update compulsory_features to match T10 only
4. ⏳ Test training with T10 factors only

## Notes

- T5_ALPHA_FACTORS definition still exists in code but is no longer used
- All factor selection now defaults to T10_ALPHA_FACTORS
- Lambda Ranker and other components will use T+10 factors
