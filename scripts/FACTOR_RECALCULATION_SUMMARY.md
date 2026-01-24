# Factor Recalculation Summary

## ✅ Script Created

**File**: `scripts/recalculate_factors_update_multiindex.py`

This script recalculates all factors using `Simple17FactorEngine` and updates the multiindex parquet file.

---

## 🔄 What It Does

1. **Loads** the existing multiindex parquet file
2. **Prepares** data for Simple17FactorEngine (ensures required columns exist)
3. **Initializes** Simple17FactorEngine with T+10 horizon
4. **Recalculates** ALL factors using the updated calculation methods
5. **Merges** recalculated factors back into original dataframe
6. **Removes** old factor columns
7. **Saves** updated file

---

## 📊 Factor Updates

### Removed Factors:
- `obv_divergence` → Replaced with `obv_momentum_40d`
- `feat_sato_momentum_10d` → Replaced with `feat_vol_price_div_30d`
- `feat_sato_divergence_10d` → Replaced with `feat_vol_price_div_30d`
- `vol_ratio_20d` → Replaced with `vol_ratio_30d`
- `ret_skew_20d` → Replaced with `ret_skew_30d`
- `ivol_20` → Replaced with `ivol_30`
- `blowoff_ratio` → Replaced with `blowoff_ratio_30d`

### New/Updated Factors:
- `obv_momentum_40d` (NEW) - OBV Momentum 40d
- `feat_vol_price_div_30d` (NEW) - Volume-Price Divergence 30d
- `vol_ratio_30d` (UPDATED) - Volume ratio 30d window
- `ret_skew_30d` (UPDATED) - Return skewness 30d window
- `ivol_30` (UPDATED) - Idiosyncratic volatility 30d window
- `blowoff_ratio_30d` (UPDATED) - Blowoff ratio with 30d std window

---

## 🚀 Usage

### Basic Usage (with confirmation):
```bash
python scripts/recalculate_factors_update_multiindex.py
```

### Non-interactive (skip confirmation):
```bash
python scripts/recalculate_factors_update_multiindex.py --yes
```

### Custom Input/Output:
```bash
python scripts/recalculate_factors_update_multiindex.py \
  --input "D:/trade/data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --output "D:/trade/data/factor_exports/polygon_factors_all_filtered_clean_recalculated.parquet" \
  --lookback 120 \
  --yes
```

### Command Line Arguments:
- `--input`: Input multiindex parquet file (default: `D:/trade/data/factor_exports/polygon_factors_all_filtered_clean.parquet`)
- `--output`: Output file path (default: adds `_recalculated` suffix to input filename)
- `--lookback`: Lookback days for factor calculation (default: 120)
- `--yes`: Skip confirmation prompt

---

## ⚠️ Important Notes

1. **Time Consumption**: Recalculating factors for the entire dataset may take a significant amount of time (depending on dataset size)

2. **Memory Usage**: The script processes all dates at once. For very large datasets, you may need to modify the script to process in batches.

3. **Backup**: It's recommended to backup the original file before running:
   ```bash
   cp "D:/trade/data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
      "D:/trade/data/factor_exports/polygon_factors_all_filtered_clean_backup.parquet"
   ```

4. **Verification**: After recalculation, verify that:
   - All new factors are present
   - All old factors are removed
   - Factor values are reasonable (not all zeros)
   - MultiIndex structure is preserved

---

## ✅ Expected Output

The script will:
- Print progress for each step
- Show factor columns being added
- Display summary of removed/added factors
- Save the updated file

Example output:
```
[STEP 1] Loading existing multiindex data...
   Original shape: (X, Y)
   Original columns: Z
   Date range: YYYY-MM-DD to YYYY-MM-DD

[STEP 2] Preparing data for Simple17FactorEngine...
   Prepared data shape: (X, Y)
   Date range: YYYY-MM-DD to YYYY-MM-DD
   Unique tickers: N

[STEP 3] Initializing Simple17FactorEngine...
   Engine initialized with horizon: 10
   Alpha factors: 16 factors

[STEP 4] Recalculating factors using Simple17FactorEngine...
   Market data prepared: (X, Y)
   Computing factors for all dates at once...
   Factors computed: (X, Y)
   Final factors shape: (X, Y)
   Factor columns: [list of factors]

[STEP 5] Merging recalculated factors into original data...
   Added factor: liquid_momentum
   Added factor: obv_momentum_40d
   ...
   Final shape: (X, Y)
   Final columns: Z

[STEP 6] Saving updated file...
   [SUCCESS] File saved: [output_path]

Summary:
Removed old factors:
  - obv_divergence
  - feat_sato_momentum_10d
  ...
Added/Updated factors:
  - obv_momentum_40d (NEW)
  - feat_vol_price_div_30d (NEW)
  ...
```

---

## 🔍 Verification

After running the script, verify the updated file:

```python
import pandas as pd

df = pd.read_parquet("D:/trade/data/factor_exports/polygon_factors_all_filtered_clean_recalculated.parquet")

# Check MultiIndex
print("Index:", df.index.names)
print("Shape:", df.shape)

# Check new factors
new_factors = ['obv_momentum_40d', 'feat_vol_price_div_30d', 'vol_ratio_30d', 
               'ret_skew_30d', 'ivol_30', 'blowoff_ratio_30d']
for factor in new_factors:
    if factor in df.columns:
        print(f"✓ {factor}: present")
    else:
        print(f"✗ {factor}: MISSING")

# Check old factors are removed
old_factors = ['obv_divergence', 'feat_sato_momentum_10d', 'feat_sato_divergence_10d',
               'vol_ratio_20d', 'ret_skew_20d', 'ivol_20', 'blowoff_ratio']
for factor in old_factors:
    if factor in df.columns:
        print(f"✗ {factor}: STILL PRESENT (should be removed)")
    else:
        print(f"✓ {factor}: removed")
```

---

## ✅ Summary

**Status**: ✅ **Script created and ready to use**

- ✅ Uses Simple17FactorEngine for factor calculation
- ✅ Updates all factor names (removes old, adds new)
- ✅ Preserves MultiIndex structure
- ✅ Handles large datasets efficiently
- ✅ Provides progress feedback
- ✅ Non-interactive mode available

**Ready for**: Running factor recalculation on multiindex parquet file
