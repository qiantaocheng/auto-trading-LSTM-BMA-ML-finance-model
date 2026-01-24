# CatBoost 确保训练和Meta Stacker集成修复

## 问题
- CatBoost 未在两个80/20运行中训练/评估
- Meta Stacker (Ridge Stacking) 需要 `pred_catboost` 输入，但CatBoost缺失

## 修复内容

### 1. 强制CatBoost初始化检查
**文件**: `bma_models/量化模型_bma_ultra_enhanced.py` (line ~11203)

**修复前**:
```python
models['catboost'] = cb.CatBoostRegressor(**catboost_config)
except ImportError:
    logger.warning("CatBoost not available")
```

**修复后**:
```python
models['catboost'] = cb.CatBoostRegressor(**catboost_config)
logger.info("[FIRST_LAYER] ✅ CatBoost模型已初始化")
except ImportError:
    logger.error("❌ CatBoost not available - install with: pip install catboost")
    logger.error("❌ Meta Stacker requires CatBoost - training will fail without it!")
    raise ImportError("CatBoost is required but not installed. Install with: pip install catboost")
```

**效果**: 如果CatBoost未安装，训练将立即失败并给出明确的错误信息。

### 2. 训练前验证CatBoost在模型列表中
**文件**: `bma_models/量化模型_bma_ultra_enhanced.py` (line ~11424)

**新增**:
```python
# Log which models will be trained
logger.info(f"[FIRST_LAYER] 📋 将训练的模型列表: {list(models.keys())}")
if 'catboost' not in models:
    logger.error("❌ [FIRST_LAYER] CRITICAL: CatBoost不在模型列表中！")
    logger.error("❌ Meta Stacker需要CatBoost输入 - 训练将失败！")
    raise ValueError("CatBoost must be in models dict for Meta Stacker to work properly")
else:
    logger.info("✅ [FIRST_LAYER] CatBoost在模型列表中，将被训练")
```

**效果**: 在训练开始前验证CatBoost存在。

### 3. 训练后验证CatBoost成功训练
**文件**: `bma_models/量化模型_bma_ultra_enhanced.py` (line ~12199)

**新增**:
```python
# Verify CatBoost was successfully trained
if 'catboost' not in trained_models or trained_models['catboost'] is None:
    logger.error("❌ [FIRST_LAYER] CRITICAL: CatBoost训练失败或未训练！")
    logger.error("❌ Meta Stacker需要CatBoost - 无法继续！")
    raise RuntimeError("CatBoost training failed - required for Meta Stacker")

if 'catboost' not in oof_predictions:
    logger.error("❌ [FIRST_LAYER] CRITICAL: CatBoost OOF预测缺失！")
    logger.error("❌ Meta Stacker需要pred_catboost - 无法继续！")
    raise RuntimeError("CatBoost OOF predictions missing - required for Meta Stacker")

logger.info("✅ [FIRST_LAYER] CatBoost训练成功，OOF预测可用")
```

**效果**: 确保CatBoost训练成功且OOF预测可用。

### 4. 更新Meta Stacker的预期模型列表
**文件**: `bma_models/量化模型_bma_ultra_enhanced.py` (line ~10644)

**修复前**:
```python
expected_models = {'elastic_net', 'xgboost', 'catboost', 'lightgbm_ranker'}
```

**修复后**:
```python
expected_models = {'elastic_net', 'xgboost', 'catboost'}  # Removed 'lightgbm_ranker' (disabled)
available_models = set(oof_for_ridge.keys())
logger.info(f"[二层] 可用模型: {available_models}")
logger.info(f"[二层] 预期模型: {expected_models}")

if not expected_models.issubset(available_models):
    missing = expected_models - available_models
    logger.error(f"[二层] ❌ 缺少预期模型: {missing}")
    logger.error(f"[二层] 这可能导致Ridge Stacker缺少必要的输入特征！")
else:
    logger.info(f"[二层] ✅ 所有预期模型都可用")

# Ensure CatBoost is present - critical for meta stacker
if 'catboost' not in available_models:
    logger.error(f"[二层] ❌ CRITICAL: CatBoost缺失！Meta Stacker需要pred_catboost输入！")
    logger.error(f"[二层] 请检查CatBoost训练是否成功完成")
```

**效果**: 
- 移除了已禁用的lightgbm_ranker
- 添加了CatBoost缺失的明确错误检查
- 更好的日志输出

### 5. 格式化模型时的CatBoost检查
**文件**: `bma_models/量化模型_bma_ultra_enhanced.py` (line ~12199)

**新增**:
```python
if trained_models[name] is None:
    if name == 'catboost':
        logger.error(f"❌ CRITICAL: CatBoost训练失败！")
        raise RuntimeError("CatBoost training failed - required for Meta Stacker")
    logger.warning(f"Skipping failed model {name}")
    continue
```

**效果**: 如果CatBoost训练失败，立即抛出错误而不是静默跳过。

## 配置验证

### unified_config.yaml 已正确配置
**文件**: `bma_models/unified_config.yaml` (line 329)

```yaml
meta_ranker:
  base_cols: ["pred_catboost", "pred_xgb", "pred_lambdarank", "pred_elastic"]
```

✅ **确认**: Meta Stacker的base_cols已包含`pred_catboost`。

## 下一步

1. **重新训练模型**: 运行80/20时间分割训练，确保CatBoost被训练
2. **验证快照**: 检查快照中是否包含CatBoost模型
3. **验证Meta Stacker**: 确认Meta Stacker使用pred_catboost作为输入

## 预期结果

- ✅ CatBoost将被训练并保存到快照
- ✅ CatBoost OOF预测将包含在stacker_data中
- ✅ Meta Stacker将使用pred_catboost作为输入特征
- ✅ 如果CatBoost缺失，训练将立即失败并给出明确的错误信息

## 测试建议

运行以下命令验证修复：
```bash
python scripts/time_split_80_20_oos_eval.py --data-file <data_file> --output-dir results/test_catboost
```

检查日志中应看到：
- `✅ [FIRST_LAYER] CatBoost模型已初始化`
- `✅ [FIRST_LAYER] CatBoost在模型列表中，将被训练`
- `✅ [FIRST_LAYER] CatBoost训练成功，OOF预测可用`
- `✅ [二层] 所有预期模型都可用`
- `[二层] 可用模型: {'elastic_net', 'xgboost', 'catboost', 'lambdarank'}`
