# OBV_DIVERGENCE 警告详细分析

## 警告信息
```
2026-01-24 05:12:56,979 [WARNING] [FEATURE] Compulsory features missing from dataset for elastic_net: ['obv_divergence']
```

## 问题根源分析

### 🔍 **问题定位**

#### 1. **Compulsory Features 定义** (Line 3251-3252)
```python
self.compulsory_features = [
    'liquid_momentum', 'momentum_10d', 'momentum_60d', 'obv_divergence', 'obv_momentum_60d',
    ...
]
```
- ✅ `obv_divergence` 在 `compulsory_features` 列表中

#### 2. **因子名称映射** (Line 5308)
```python
FACTOR_NAME_MAPPING = {
    ...
    'obv_divergence': 'obv_momentum_40d',  # Legacy alias: OBV Divergence → OBV Momentum (40d)
}
```
- ✅ 存在映射：`obv_divergence` → `obv_momentum_40d`

#### 3. **检查逻辑** (Line 6873-6875)
```python
missing = [c for c in self.compulsory_features if c not in available_set]
if missing:
    logger.warning(f"[FEATURE] Compulsory features missing from dataset for {model_name}: {missing}")
```
- ❌ **问题**: 检查 `obv_divergence` 是否在 `available_set` 中
- ❌ **实际情况**: 数据集中只有 `obv_momentum_40d`，没有 `obv_divergence`

### 🔴 **根本原因**

**时序问题**:
1. **因子计算阶段**: 
   - `simple_25_factor_engine.py` 计算因子时，如果 `obv_divergence` 计算失败，会创建 `pd.Series(0.0, index=data.index, name='obv_divergence')`
   - 但如果计算成功，可能返回的是 `obv_momentum_40d` 列（取决于因子引擎的实现）
   - **实际数据集列名**: `obv_momentum_40d`（或其他名称），**不是** `obv_divergence`

2. **特征检查阶段** (Line 6873):
   - `available_set` 包含数据集中**实际存在的列名**
   - 如果数据集中有 `obv_momentum_40d` 但没有 `obv_divergence`
   - 检查 `'obv_divergence' in available_set` → **False**
   - 触发警告

3. **映射应用阶段** (Line 5320-5333):
   - 因子名称映射在**模型加载/配置时**应用
   - 但**不在特征检查时应用**
   - 所以 `compulsory_features` 中的 `obv_divergence` 不会被映射到 `obv_momentum_40d`

### 📊 **数据流分析**

```
因子计算 (simple_25_factor_engine.py)
  ↓
返回因子 DataFrame
  - 列名: 'obv_momentum_40d' (如果计算成功)
  - 或: 'obv_divergence' (如果计算失败，使用 Series fix)
  ↓
特征检查 (_get_feature_cols_for_model)
  ↓
available_set = set(available_cols)  # 包含实际列名
  ↓
检查: 'obv_divergence' in available_set?
  - 如果数据集有 'obv_momentum_40d' → False ❌
  - 如果数据集有 'obv_divergence' → True ✅
  ↓
触发警告 (如果 False)
```

### ⚠️ **为什么会出现这种情况？**

#### 场景 1: 因子计算成功
- `simple_25_factor_engine.py` 计算 `obv_momentum_40d`（不是 `obv_divergence`）
- 数据集列名: `obv_momentum_40d`
- `compulsory_features` 期望: `obv_divergence`
- **结果**: 警告触发

#### 场景 2: 因子计算失败
- `simple_25_factor_engine.py` 使用 Series fix: `pd.Series(0.0, index=data.index, name='obv_divergence')`
- 数据集列名: `obv_divergence`
- `compulsory_features` 期望: `obv_divergence`
- **结果**: 无警告 ✅

#### 场景 3: 因子名称不一致
- 训练时使用: `obv_divergence`
- 预测时计算: `obv_momentum_40d`
- **结果**: 警告触发（名称不匹配）

### 🔧 **解决方案**

#### 方案 1: 更新 `compulsory_features` (推荐)
```python
# Line 3251-3252
self.compulsory_features = [
    'liquid_momentum', 'momentum_10d', 'momentum_60d', 
    'obv_momentum_40d',  # 改为实际使用的名称
    'obv_momentum_60d',
    ...
]
```
- ✅ 直接使用实际因子名称
- ✅ 避免映射复杂性

#### 方案 2: 在检查时应用映射
```python
# Line 6873
# 应用因子名称映射
mapped_compulsory = []
for feat in self.compulsory_features:
    mapped_feat = FACTOR_NAME_MAPPING.get(feat, feat)
    mapped_compulsory.append(mapped_feat)

missing = [c for c in mapped_compulsory if c not in available_set]
```
- ✅ 自动处理映射
- ⚠️ 需要确保映射逻辑正确

#### 方案 3: 确保因子计算返回正确名称
- 确保 `obv_divergence` 计算失败时返回的 Series 名称是 `'obv_divergence'`
- ✅ 已修复（使用 `name='obv_divergence'`）

### 📝 **当前状态**

1. **MultiIndex 修复**: ✅ 完成
   - 所有因子失败时使用 `pd.Series(0.0, index=data.index, name='factor_name')`
   - 确保 index 对齐

2. **因子名称问题**: ⚠️ **未完全解决**
   - `compulsory_features` 仍包含 `obv_divergence`
   - 但实际数据集可能使用 `obv_momentum_40d`
   - 需要更新 `compulsory_features` 或应用映射

### ✅ **建议**

**立即行动**:
1. 检查实际数据集中是否存在 `obv_divergence` 列
2. 如果不存在，更新 `compulsory_features` 使用 `obv_momentum_40d`
3. 或者在检查时应用 `FACTOR_NAME_MAPPING`

**长期方案**:
- 统一因子命名规范
- 确保训练和预测使用相同的因子名称
- 在 `compulsory_features` 中使用实际因子名称，而不是别名

## 详细原因分析

### 🔍 **因子计算逻辑**

#### T10_ALPHA_FACTORS (Line 52-70)
```python
T10_ALPHA_FACTORS = [
    ...
    'obv_divergence',  # Line 56 - 在因子列表中
    ...
]
```

#### 因子计算 (Lines 1334-1357)
```python
if 'obv_divergence' in getattr(self, 'alpha_factors', []):
    try:
        # 计算 obv_divergence
        out['obv_divergence'] = obv_divergence
    except Exception as e:
        # 失败时使用 Series fix
        out['obv_divergence'] = pd.Series(0.0, index=data.index, name='obv_divergence')
```

#### 同时计算 obv_momentum_40d (Lines 1360-1409)
```python
if 'obv_momentum_40d' in getattr(self, 'alpha_factors', []):
    try:
        # 计算 obv_momentum_40d
        out['obv_momentum_40d'] = obv_momentum_40d
    except Exception as e:
        # 失败时使用 Series fix
        out['obv_momentum_40d'] = pd.Series(0.0, index=data.index, name='obv_momentum_40d')
```

### ⚠️ **问题场景**

#### 场景 A: obv_divergence 计算失败
1. `obv_divergence` 在 `T10_ALPHA_FACTORS` 中
2. 计算失败 → 使用 `pd.Series(0.0, index=data.index, name='obv_divergence')`
3. **结果**: 数据集中**有** `obv_divergence` 列 ✅
4. **警告**: 不应该出现（除非其他问题）

#### 场景 B: obv_divergence 不在 alpha_factors 中
1. 如果 `self.alpha_factors` 不包含 `'obv_divergence'`
2. 计算逻辑跳过 `obv_divergence`（Line 1334 条件不满足）
3. **结果**: 数据集中**没有** `obv_divergence` 列 ❌
4. **警告**: 触发 ✅（这就是当前情况）

#### 场景 C: 使用 T5_ALPHA_FACTORS
1. 如果 horizon < 10，使用 `T5_ALPHA_FACTORS`
2. `T5_ALPHA_FACTORS` 可能不包含 `obv_divergence`
3. **结果**: 数据集中**没有** `obv_divergence` 列 ❌
4. **警告**: 触发 ✅

### 🔴 **根本原因确认**

**最可能的情况**:
1. **`obv_divergence` 不在当前使用的 `alpha_factors` 列表中**
   - 可能使用 `T5_ALPHA_FACTORS`（horizon < 10）
   - 或者 `alpha_factors` 被覆盖/修改
   - 或者 `obv_divergence` 从 `T10_ALPHA_FACTORS` 中被移除

2. **因子计算跳过 `obv_divergence`**
   - Line 1334: `if 'obv_divergence' in getattr(self, 'alpha_factors', []):`
   - 如果条件为 False，不会计算 `obv_divergence`
   - 数据集中没有 `obv_divergence` 列

3. **但 `compulsory_features` 仍包含 `obv_divergence`**
   - Line 3252: `'obv_divergence'` 在 `compulsory_features` 中
   - 检查时发现缺失 → 警告

### 📊 **验证步骤**

检查以下内容：
1. **当前使用的因子列表**:
   ```python
   # 检查 self.alpha_factors 是否包含 'obv_divergence'
   # 检查 horizon 值（决定使用 T5 还是 T10）
   ```

2. **实际数据集列名**:
   ```python
   # 检查 feature_data.columns 是否包含 'obv_divergence'
   # 检查是否包含 'obv_momentum_40d'
   ```

3. **compulsory_features 配置**:
   ```python
   # 检查 self.compulsory_features 是否包含 'obv_divergence'
   ```

## 总结

**警告原因**:
- `compulsory_features` 包含 `obv_divergence`（Line 3252）
- 但 `obv_divergence` **不在当前使用的 `alpha_factors` 列表中**
- 因子计算跳过 `obv_divergence`（Line 1334 条件不满足）
- 数据集中**没有** `obv_divergence` 列
- 检查时发现缺失 → 警告触发

**这不是 MultiIndex 问题**，而是**因子配置不一致**问题：
- `compulsory_features` 期望 `obv_divergence`
- 但实际因子计算列表不包含 `obv_divergence`
- 导致数据集中没有该列

**解决方案**:
1. 确保 `obv_divergence` 在 `alpha_factors` 中（如果使用 T10）
2. 或者从 `compulsory_features` 中移除 `obv_divergence`（如果不再使用）
3. 或者使用 `obv_momentum_40d` 替代（更新 `compulsory_features`）
