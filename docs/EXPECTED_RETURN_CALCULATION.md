# Expected Return Calculation - Complete Explanation

## Overview
The **expected return** (reported as `avg_top_return`) is the **average T+10 forward return** of the **Top 30 predicted stocks** across all rebalance dates in the backtest.

---

## Step-by-Step Calculation Process

### 1. **Target Calculation (T+10 Forward Return)**

**Location**: `bma_models/simple_25_factor_engine.py` (lines 590-596)

```python
# For each stock, calculate forward return over T+10 trading days
target_series = (
    factors_df.groupby(level='ticker')['Close']
    .pct_change(self.horizon)      # Calculate % change over 10 days
    .shift(-self.horizon)          # Shift backward to align with current date
)
factors_df['target'] = target_series
```

**Formula**:
```
target = (Close[T+10] / Close[T]) - 1
```

**Example**:
- Date T: Close = $100
- Date T+10: Close = $110
- **target = (110/100) - 1 = 0.10 = 10%**

**Note**: `target` is stored as a **decimal** (0.10 = 10%), not percentage points.

---

### 2. **Rolling Prediction (Per Rebalance Date)**

**Location**: `scripts/comprehensive_model_backtest.py` (lines 500-680)

For each **rebalance date** (every 10 trading days when `rebalance_mode='horizon'`):

1. **Extract features** for that date:
   ```python
   date_data = data.xs(pred_date, level='date', drop_level=True)
   X = self._prepare_feature_matrix(date_data, all_feature_cols)
   ```

2. **Get actual target** (T+10 forward return):
   ```python
   actual_target = date_data.loc[X.index, 'target']
   ```

3. **Run model prediction**:
   ```python
   pred = model.predict(X)
   ```

4. **Store prediction + actual**:
   ```python
   pred_df = pd.DataFrame({
       'date': pred_date,
       'ticker': tickers,
       'prediction': pred.values,      # Model's predicted return
       'actual': actual_target.values  # Actual T+10 forward return
   })
   ```

---

### 3. **Top 30 Selection (Per Date)**

**Location**: `scripts/comprehensive_model_backtest.py` (lines 747-828)

For each rebalance date:

1. **Sort stocks by prediction** (descending):
   ```python
   sorted_group = valid_group.sort_values('prediction', ascending=False)
   ```

2. **Select Top 30**:
   ```python
   top_group = sorted_group.head(30)  # Top 30 predicted stocks
   ```

3. **Calculate average actual return** of Top 30:
   ```python
   top_return = top_group['actual'].mean()
   ```

**Example** (for one date):
- Top 30 stocks by prediction
- Their actual T+10 returns: [0.12, 0.08, 0.05, ..., -0.02]
- **top_return = mean([0.12, 0.08, 0.05, ..., -0.02]) = 0.0594 = 5.94%**

---

### 4. **Average Across All Dates**

**Location**: `scripts/comprehensive_model_backtest.py` (lines 819-820)

After processing all rebalance dates:

```python
results_df = pd.DataFrame(results)  # One row per date

summary = {
    'avg_top_return': results_df['top_return'].mean(),  # Average across all dates
    ...
}
```

**Formula**:
```
avg_top_return = mean(top_return_date1, top_return_date2, ..., top_return_dateN)
```

**Example**:
- Date 1: Top 30 return = 5.94%
- Date 2: Top 30 return = 4.20%
- Date 3: Top 30 return = 6.10%
- ...
- Date N: Top 30 return = 3.80%
- **avg_top_return = mean(5.94%, 4.20%, 6.10%, ..., 3.80%) = 5.20%**

---

## Complete Formula

```
avg_top_return = (1/N) × Σ[mean(actual_T+10_return of Top 30 stocks on date i)]
                 for i = 1 to N rebalance dates
```

Where:
- **N** = number of rebalance dates (e.g., 260 weeks / 10 trading days ≈ 26 dates per year)
- **actual_T+10_return** = `(Close[T+10] / Close[T]) - 1` for each stock
- **Top 30 stocks** = stocks with highest `prediction` on each date

---

## Key Properties

### 1. **Non-Overlapping Returns**
- **Rebalance frequency**: Every 10 trading days (when `rebalance_mode='horizon'`)
- **Target horizon**: T+10 trading days
- **Result**: Returns do **not overlap** (each stock's return is measured over a unique 10-day window)

### 2. **Decimal Format**
- `avg_top_return = 0.0594` means **5.94% per T+10 period**
- **Not** 0.594% or 59.4%

### 3. **Annualization** (if needed)
Since returns are non-overlapping:
```
Annualized Return = (1 + avg_top_return)^(252/10) - 1
```

**Example**:
- `avg_top_return = 0.0594` (5.94% per T+10)
- **Annualized** = `(1.0594)^(25.2) - 1 ≈ 4.19 = 419% / year`

**Note**: This assumes the same return every period, which is unrealistic. Use with caution.

---

## Example: Full Calculation Flow

### Input Data (one date):
```
Date: 2024-01-15
Stock | Prediction | Actual (T+10 return)
------|------------|--------------------
AAPL  | 0.08       | 0.12  ← Top 1
MSFT  | 0.07       | 0.09  ← Top 2
GOOGL | 0.06       | 0.05  ← Top 3
...   | ...        | ...
TSLA  | -0.05      | -0.08 ← Bottom 1
```

### Step 1: Sort by Prediction
```
Sorted (descending):
1. AAPL  (pred=0.08, actual=0.12)
2. MSFT  (pred=0.07, actual=0.09)
3. GOOGL (pred=0.06, actual=0.05)
...
30. XYZ   (pred=0.02, actual=0.03)  ← Top 30 cutoff
```

### Step 2: Calculate Top 30 Return
```
top_return_2024-01-15 = mean([0.12, 0.09, 0.05, ..., 0.03]) = 0.0594
```

### Step 3: Repeat for All Dates
```
Date 1 (2024-01-15): top_return = 0.0594
Date 2 (2024-01-29): top_return = 0.0420
Date 3 (2024-02-12): top_return = 0.0610
...
Date N: top_return = 0.0380
```

### Step 4: Average Across Dates
```
avg_top_return = mean([0.0594, 0.0420, 0.0610, ..., 0.0380]) = 0.0520
```

**Result**: `avg_top_return = 0.0520` = **5.20% per T+10 period**

---

## Code Locations Summary

| Step | File | Function/Method | Lines |
|------|------|-----------------|-------|
| **Target Calculation** | `bma_models/simple_25_factor_engine.py` | `compute_all_factors()` | 590-596 |
| **Rolling Prediction** | `scripts/comprehensive_model_backtest.py` | `rolling_prediction()` | 500-680 |
| **Top 30 Selection** | `scripts/comprehensive_model_backtest.py` | `calculate_group_returns()` | 747-828 |
| **Average Calculation** | `scripts/comprehensive_model_backtest.py` | `calculate_group_returns()` | 819-820 |
| **Report Generation** | `scripts/comprehensive_model_backtest.py` | `generate_report()` | 1255-1400 |

---

## Important Notes

1. **`actual` is NOT a prediction**: It's the **realized T+10 forward return** from historical data.

2. **`prediction` is the model output**: The model predicts a return, but we measure success by comparing it to `actual`.

3. **Top 30 is fixed count**: Not a percentage. If only 50 stocks are available on a date, Top 30 = top 30 of those 50.

4. **Missing data handling**: Stocks with `NaN` in `prediction` or `actual` are excluded from that date's calculation.

5. **Non-overlapping is critical**: The `rebalance_mode='horizon'` ensures each return window is independent, making the average meaningful.

---

## Verification

To verify the calculation, check:
1. **`performance_report_*.csv`**: Contains `avg_top_return` column
2. **`*_weekly_returns.csv`**: Contains per-date `top_return` values
3. **`*_predictions.parquet`**: Contains raw `prediction` and `actual` values for each stock on each date

**Example check**:
```python
import pandas as pd

# Load weekly returns
weekly = pd.read_csv('performance_report_*.csv')
print(f"avg_top_return from report: {weekly['avg_top_return'].iloc[0]}")

# Load detailed weekly returns
details = pd.read_csv('*_weekly_returns.csv')
print(f"Manual average: {details['top_return'].mean()}")
# These should match!
```


