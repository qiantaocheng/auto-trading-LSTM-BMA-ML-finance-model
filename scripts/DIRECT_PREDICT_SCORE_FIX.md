# Direct Predict Score Fix - All Scores Same Issue

## 🔧 Problem
Direct Predict is showing the same score (0.920046) for all top 20 stocks, which is incorrect.

## 🔍 Root Cause Analysis

The issue could be at several points in the prediction pipeline:
1. **Input features** (`ridge_input`) might have the same values for all stocks
2. **Model predictions** (`ridge_predictions_df`) might be returning the same value
3. **Final predictions** (`final_df`) might have the same value
4. **Score extraction** might be incorrectly extracting values

## ✅ Fixes Applied

### 1. Added Debugging in `bma_models/量化模型_bma_ultra_enhanced.py`

**Location**: `predict_with_snapshot` method

- **ridge_input statistics**: Logs each column's unique values, min, max, mean
- **ridge_predictions_df statistics**: Logs prediction statistics after MetaRankerStacker prediction
- **final_df statistics**: Logs final prediction statistics before creating pred_series
- **pred_series statistics**: Logs the Series that will be returned, including unique values and range

### 2. Added Debugging in `autotrader/app.py`

**Location**: `_direct_predict_snapshot` method

- **predictions_raw statistics**: Logs the raw predictions received from `predict_with_snapshot`
- **latest_predictions statistics**: Logs the predictions before creating recommendations
- **Duplicate score detection**: Warns if consecutive scores are identical

## 🔍 How to Diagnose

When you run Direct Predict again, check the logs for:

1. **Check ridge_input**:
   ```
   [SNAPSHOT] 🔍 ridge_input['pred_catboost']: unique=...
   ```
   - If any column has `unique=1`, that's the problem - all stocks have the same input value

2. **Check ridge_predictions_df**:
   ```
   [SNAPSHOT] 🔍 ridge_predictions_df score unique values: ...
   ```
   - If `unique values: 1`, the model is returning the same prediction for all stocks

3. **Check final_df**:
   ```
   [SNAPSHOT] 🔍 final_df blended_score unique values: ...
   ```
   - If `unique values: 1`, the final predictions are identical

4. **Check pred_series**:
   ```
   [SNAPSHOT] 🔍 pred_series unique values: ...
   ```
   - If `unique values: 1`, the Series has identical values

5. **Check predictions_raw**:
   ```
   [DirectPredict] 📊 predictions_raw unique values: ...
   ```
   - If `unique values: 1`, the raw predictions are identical

## 🚨 Expected Log Messages

If the issue is detected, you'll see:
```
[SNAPSHOT] ❌ CRITICAL: All ridge predictions have the same value: 0.920046
[SNAPSHOT] ❌ This indicates a problem with MetaRankerStacker predictions!
```

Or:
```
[SNAPSHOT] ❌ CRITICAL: All final predictions have the same value: 0.920046
[SNAPSHOT] ❌ This will cause all Direct Predict scores to be identical!
```

## 🔧 Next Steps

1. **Run Direct Predict again** and check the logs
2. **Identify where the duplicate values start** (ridge_input, ridge_predictions_df, final_df, or pred_series)
3. **Fix the root cause** based on the logs:
   - If `ridge_input` has duplicate values → Check first-layer predictions (CatBoost, XGBoost, LambdaRank, ElasticNet)
   - If `ridge_predictions_df` has duplicate values → Check MetaRankerStacker model
   - If `final_df` has duplicate values → Check rank-aware blending logic
   - If `pred_series` has duplicate values → Check pred_series creation logic

## 📝 Notes

The score 0.920046 is very specific, suggesting it might be:
- A default/constant value being returned
- A model bias/intercept value
- A single prediction being broadcast to all stocks

The debugging will help identify which of these is the case.
