# Feature Flow Reference Guide

This document traces how features are selected, stored, and used across all prediction paths to ensure consistency.

## Overview

Features flow through the system via:
1. **Training**: Features selected via `_get_first_layer_feature_cols_for_model()` → stored in `feature_names_by_model` dict → saved to manifest
2. **80/20 OOS Evaluation**: Features retrieved from model attributes (`feature_names_in_`, etc.) → aligned via `align_test_features_with_model()`
3. **OOF Predictions**: Uses same `feature_names_by_model` from training
4. **Live Prediction**: Features loaded from manifest `feature_names_by_model` → aligned per model

---

## 1. Training Path

### Feature Selection Process

**Location**: `bma_models/量化模型_bma_ultra_enhanced.py`

**Method**: `_unified_model_training()` → `_get_first_layer_feature_cols_for_model()`

**Flow**:
```python
# Line 11032-11036: CV Loop
use_cols = self._get_first_layer_feature_cols_for_model(name, list(X_train.columns), available_cols=X_train.columns)
if name not in feature_names_by_model:
    feature_names_by_model[name] = list(use_cols)
X_train_use = X_train[use_cols].copy()
X_val_use = X_val[use_cols].copy()
```

**Feature Selection Logic** (`_get_first_layer_feature_cols_for_model`, line 6785):
1. **Priority 1**: Global whitelist (if `_feature_whitelist_active` is True)
2. **Priority 2**: Per-model best features from `first_layer_feature_overrides` (respects `best_features_per_model.json`)
3. **Priority 3**: All available features
4. **Always Included**: Compulsory features (from `self.compulsory_features`)

**Full-Fit Training** (lines 11411-11489):
- **XGBoost** (line 11423): Uses `_get_first_layer_feature_cols_for_model()` ✅
- **CatBoost** (line 11445): Uses `_get_first_layer_feature_cols_for_model()` ✅
- **LightGBM Ranker** (line 11473): Uses `_get_first_layer_feature_cols_for_model()` ✅
- **LambdaRank** (line 10811): Uses `_get_first_layer_feature_cols_for_model()` ✅
- **ElasticNet** (line 11487): Uses `_get_first_layer_feature_cols_for_model()` ✅

**Storage** (line 11643):
```python
return {
    'feature_names': list(X.columns),  # All available features
    'feature_names_by_model': feature_names_by_model,  # Per-model feature lists
    ...
}
```

**Saved to Manifest** (`model_registry.py`, line 618-619):
```python
manifest = {
    'feature_names': list(feature_names) if feature_names else None,
    'feature_names_by_model': feature_names_by_model if feature_names_by_model else None,
    ...
}
```

---

## 2. 80/20 Time Split OOS Evaluation

### Feature Alignment Process

**Location**: `scripts/time_split_80_20_oos_eval.py`

**Method**: `align_test_features_with_model()` (line 48)

**Flow**:
```python
# Line 1046-1077: Per-model prediction
X_aligned = align_test_features_with_model(X, models_dict['elastic_net'], 'ElasticNet', logger)
pred = models_dict['elastic_net'].predict(X_aligned)
```

**Feature Retrieval** (lines 65-76):
1. **ElasticNet**: `model.feature_names_in_` (sklearn attribute)
2. **XGBoost**: `model.feature_name_` or `model._Booster.feature_names`
3. **CatBoost**: `model.feature_names_` (if available)
4. **LightGBM**: `model.feature_name_` or `model.booster_.feature_name()`
5. **LambdaRank**: Uses full `X` (handled separately, line 1082)

**Alignment Logic** (lines 78-102):
```python
# Get training features from model
train_features = list(model.feature_names_in_)  # or other attributes

# Check missing/extra features
missing_features = train_features_set - test_features  # Fill with 0.0
extra_features = test_features - train_features_set   # Silently ignored

# Align: select only training features in correct order
X_aligned = X_test[train_features].copy()
```

**Key Points**:
- ✅ Original test data (`X`) is **NOT modified** - alignment creates a copy
- ✅ Missing features are padded with `0.0`
- ✅ Extra features are silently ignored (not logged)
- ✅ Feature order matches training order

---

## 3. OOF (Out-of-Fold) Predictions

### Feature Usage During CV

**Location**: `bma_models/量化模型_bma_ultra_enhanced.py`

**Flow**: Same as training - uses `feature_names_by_model` from CV loop

**CV Loop** (line 11032-11036):
```python
# Same feature selection as training
use_cols = self._get_first_layer_feature_cols_for_model(name, list(X_train.columns), available_cols=X_train.columns)
feature_names_by_model[name] = list(use_cols)
X_train_use = X_train[use_cols].copy()
X_val_use = X_val[use_cols].copy()
```

**OOF Storage** (line 11641):
```python
return {
    'oof_predictions': oof_predictions,  # Contains predictions with same feature selection
    'feature_names_by_model': feature_names_by_model,  # Same dict used for OOF
    ...
}
```

**Key Points**:
- ✅ Uses same `feature_names_by_model` as training
- ✅ Each CV fold uses consistent feature selection
- ✅ OOF predictions align with final model training features

---

## 4. Live Prediction (predict_with_snapshot)

### Feature Loading and Alignment

**Location**: `bma_models/量化模型_bma_ultra_enhanced.py`

**Method**: `predict_with_snapshot()` (line 9544)

**Feature Loading** (lines 9663-9664):
```python
manifest = load_manifest(effective_snapshot_id)
feature_names = manifest.get('feature_names') or []
feature_names_by_model = manifest.get('feature_names_by_model') or {}
```

**Feature Preparation** (lines 9698-9704):
```python
X_df = X.copy()
# REMOVED: Upfront feature filtering to feature_names
# Keep all original features - each model will select its own features
# via feature_names_by_model, ensuring no feature deletion occurs
```

**Per-Model Alignment** (lines 9718-9756):
```python
# ElasticNet (line 9718)
cols = feature_names_by_model.get('elastic_net') or feature_names or list(X_df.columns)
X_m = X_df.copy()
missing = [c for c in cols if c not in X_m.columns]
for c in missing:
    X_m[c] = 0.0
X_m = X_m[cols].copy()
pred = enet.predict(X_m.values)

# XGBoost (line 9734)
cols = feature_names_by_model.get('xgboost') or feature_names or list(X_df.columns)
# ... same alignment logic

# CatBoost (line 9750)
cols = feature_names_by_model.get('catboost') or feature_names or list(X_df.columns)
# ... same alignment logic
```

**LambdaRank** (line 1082-1087 in time_split_80_20_oos_eval.py):
```python
# LambdaRank uses full X (no upfront filtering)
X_lambda = X.copy()
X_lambda.index = pd.MultiIndex.from_arrays([...], names=['date', 'ticker'])
pred_result = lambda_rank_stacker.predict(X_lambda)
```

**Key Points**:
- ✅ All original features preserved in `X_df`
- ✅ Each model selects its own features from `feature_names_by_model`
- ✅ Missing features padded with `0.0`
- ✅ Fallback: `feature_names` → all columns if `feature_names_by_model` missing

---

## 5. Direct Prediction (predict_with_live_data)

### Feature Flow

**Location**: `bma_models/量化模型_bma_ultra_enhanced.py`

**Method**: `predict_with_live_data()` (line 5295)

**Feature Loading** (lines 5298-5306):
```python
feature_names = training_results.get('feature_names') or ...
feature_names_by_model = training_results.get('feature_names_by_model') or {}
```

**Feature Mapping** (lines 5312-5346):
- Old factor names mapped to new T+5 standard names
- Example: `momentum_10d` → `liquid_momentum`

**Per-Model Usage** (line 5505):
```python
cols = feature_names_by_model.get(model_name) or getattr(model, 'feature_names_in_', None)
if cols is None or len(cols) == 0:
    cols = self._get_first_layer_feature_cols_for_model(model_name, list(X.columns), available_cols=X.columns)
```

---

## Feature Reference List

### Compulsory Features
Defined in: `self.compulsory_features` (model initialization)

**Common Compulsory Features**:
- Always included regardless of model-specific feature selection
- Typically includes: market cap, volume, price-based factors

### Model-Specific Features
Stored in: `feature_names_by_model` dict

**Structure**:
```python
feature_names_by_model = {
    'elastic_net': ['feature1', 'feature2', ...],
    'xgboost': ['feature1', 'feature3', ...],
    'catboost': ['feature1', 'feature4', ...],
    'lightgbm_ranker': ['feature1', 'feature5', ...],
    'lambdarank': ['feature1', 'feature6', ...],
}
```

### Feature Selection Sources

1. **best_features_per_model.json**
   - Per-model optimal feature lists
   - Loaded via `first_layer_feature_overrides`
   - Used by `_get_first_layer_feature_cols_for_model()`

2. **Global Whitelist** (if active)
   - Environment variable: `BMA_FEATURE_WHITELIST`
   - Overrides per-model selection

3. **Compulsory Features**
   - Always included
   - Defined in model config

4. **All Available Features** (fallback)
   - If no overrides specified
   - Uses all numeric columns from input data

---

## Verification Checklist

### ✅ Training
- [x] Features selected via `_get_first_layer_feature_cols_for_model()`
- [x] Stored in `feature_names_by_model` dict
- [x] Saved to manifest JSON
- [x] No hardcoded feature drops (removed XGBoost/CatBoost drops)

### ✅ 80/20 OOS Evaluation
- [x] Features retrieved from model attributes
- [x] Aligned via `align_test_features_with_model()`
- [x] Original data not modified
- [x] Missing features padded with 0.0

### ✅ OOF Predictions
- [x] Uses same `feature_names_by_model` as training
- [x] Consistent across CV folds

### ✅ Live Prediction
- [x] Features loaded from manifest
- [x] Per-model alignment from `feature_names_by_model`
- [x] No upfront filtering (all features preserved)
- [x] Missing features padded with 0.0

---

## Code Locations Summary

| Path | File | Key Methods/Lines |
|------|------|-------------------|
| **Training** | `bma_models/量化模型_bma_ultra_enhanced.py` | `_unified_model_training()` (line 10600)<br>`_get_first_layer_feature_cols_for_model()` (line 6785)<br>CV loop (line 11032)<br>Full-fit (lines 11411-11489) |
| **80/20 OOS** | `scripts/time_split_80_20_oos_eval.py` | `align_test_features_with_model()` (line 48)<br>Prediction loop (lines 1046-1077) |
| **OOF** | `bma_models/量化模型_bma_ultra_enhanced.py` | CV loop (line 11032)<br>Return dict (line 11641) |
| **Live Prediction** | `bma_models/量化模型_bma_ultra_enhanced.py` | `predict_with_snapshot()` (line 9544)<br>Feature loading (line 9663)<br>Per-model alignment (lines 9718-9756) |
| **Manifest Save** | `bma_models/model_registry.py` | `save_snapshot()` (line 618-619) |

---

## Important Notes

1. **No Feature Deletion**: All feature filtering has been removed. Original data is preserved, and models receive aligned feature subsets.

2. **Consistent Selection**: All paths use `_get_first_layer_feature_cols_for_model()` or retrieve features from saved `feature_names_by_model`.

3. **Missing Features**: Handled consistently across all paths by padding with `0.0`.

4. **Feature Order**: Preserved to match training order for all models.

5. **LambdaRank**: Uses full feature set (no per-model filtering) but respects `base_cols` from training config.

---

## Troubleshooting

### Issue: Feature Mismatch Error
**Check**:
1. Verify `feature_names_by_model` exists in manifest
2. Check model attributes (`feature_names_in_`, etc.)
3. Ensure `align_test_features_with_model()` is called

### Issue: Missing Features
**Check**:
1. Verify features exist in input data
2. Check `best_features_per_model.json` for model-specific lists
3. Verify compulsory features are included

### Issue: Extra Features Warning
**Note**: Extra features are silently ignored (by design). Original data is not modified.

---

**Last Updated**: 2026-01-18
**Status**: ✅ All feature deletion removed, consistent flow verified
