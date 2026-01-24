# 修复子集训练挂起问题

## 根本原因

**问题**: 子集数据（1244个交易日）在6折CV中，前几个fold的训练窗可能 < 252天，导致所有fold被跳过，`oof_pred` 全0，后续Ridge Stacker训练挂起。

## 修复方案

### 方案1: 动态调整最小训练窗（推荐）

在 `_unified_model_training` 中，根据数据规模动态调整 `min_train_window_days`：

```python
# Line 11426-11432 修改为：
# 🔧 最小训练窗限制：根据数据规模动态调整
try:
    from bma_models.unified_config_loader import get_time_config
    time_config = get_time_config()
    base_min_train_window = getattr(time_config, 'min_train_window_days', 252)
    
    # 动态调整：子集数据降低要求
    unique_dates_count = len(pd.Series(groups_norm).unique()) if groups_norm is not None else sample_size // 500
    if unique_dates_count < 1500:  # 子集数据（约3年）
        min_train_window_days = max(126, base_min_train_window // 2)  # 降低到半年
        logger.info(f"[FIRST_LAYER] 子集数据检测：唯一日期={unique_dates_count}，降低最小训练窗到{min_train_window_days}天")
    else:  # 全量数据
        min_train_window_days = base_min_train_window
        logger.info(f"[FIRST_LAYER] 全量数据：唯一日期={unique_dates_count}，使用标准最小训练窗{min_train_window_days}天")
except:
    min_train_window_days = 252  # 默认1年交易日
```

### 方案2: 添加安全检查

在训练完成后检查是否有有效的fold：

```python
# Line 11989-11992 后添加：
scores_clean = [s for s in scores if not np.isnan(s) and np.isfinite(s)]
if len(scores_clean) == 0:
    error_msg = (
        f"[FIRST_LAYER][{name}] 所有CV fold都被跳过！"
        f"训练窗不足{min_train_window_days}天。"
        f"数据唯一日期数: {len(pd.Series(groups_norm).unique()) if groups_norm is not None else 'unknown'}"
    )
    logger.error(error_msg)
    raise ValueError(error_msg)

cv_scores[name] = np.mean(scores_clean) if scores_clean else 0.0
```

### 方案3: 减少CV折数（子集数据）

在创建CV分割器时，根据数据规模减少折数：

```python
# Line 11108-11126 修改为：
adapted_splits = self._CV_SPLITS
adapted_test_size = self._TEST_SIZE

enforce_full_cv = getattr(self, 'enforce_full_cv', False)

# 子集数据优化：减少CV折数
unique_dates_count = len(pd.Series(groups_norm).unique()) if groups_norm is not None else sample_size // 500
if unique_dates_count < 1500 and not enforce_full_cv:  # 子集数据
    adapted_splits = min(3, self._CV_SPLITS)  # 减少到3折
    adapted_test_size = min(42, self._TEST_SIZE)
    logger.info(f"子集数据优化: CV折数={adapted_splits}, test_size={adapted_test_size}")
elif sample_size > 1000000 and not enforce_full_cv:  # 超大数据集
    adapted_splits = min(3, self._CV_SPLITS)
    adapted_test_size = min(42, self._TEST_SIZE)
    logger.info(f"超大数据集优化: CV折数={adapted_splits}, test_size={adapted_test_size}")
elif enforce_full_cv:
    logger.info(f"Full CV enforced: 使用 splits={adapted_splits}, test_size={adapted_test_size}")
```

## 立即修复步骤

1. **修改 `_unified_model_training` 方法**:
   - 添加动态 `min_train_window_days` 调整
   - 添加安全检查，如果所有fold被跳过则报错

2. **测试修复**:
   - 使用子集数据重新运行训练
   - 确认不会挂起，要么成功要么明确报错

3. **验证**:
   - 检查日志中是否有 "所有CV fold都被跳过" 的错误
   - 如果有，说明数据不足，需要进一步降低要求或增加数据
