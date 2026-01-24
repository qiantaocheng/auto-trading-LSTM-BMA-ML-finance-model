# 修复 meta_ranker_stacker 参数错误

## 错误信息

```
TypeError: save_model_snapshot() got an unexpected keyword argument 'meta_ranker_stacker'
```

## 问题原因

`save_model_snapshot()` 函数的签名中没有 `meta_ranker_stacker` 参数：

```python
def save_model_snapshot(
    training_results: Dict[str, Any],
    ridge_stacker: Any = None,  # ✅ 接受这个参数
    lambda_rank_stacker: Any = None,
    rank_aware_blender: Any = None,
    dual_head_fusion_manager: Any = None,
    lambda_percentile_transformer: Any = None,
    tag: str = "default",
    # ❌ 没有 meta_ranker_stacker 参数
)
```

但在 `train_full_dataset.py` 中错误地传递了 `meta_ranker_stacker` 参数。

## 解决方案

`meta_ranker_stacker` 应该作为 `ridge_stacker` 参数传递，因为：

1. **函数内部会自动识别**: `save_model_snapshot()` 会检查 `ridge_stacker` 是否是 MetaRankerStacker（通过检查是否有 `lightgbm_model` 属性）

2. **向后兼容**: MetaRankerStacker 替代了旧的 RidgeStacker，但为了向后兼容，仍然使用 `ridge_stacker` 参数名

3. **代码中的正确用法**: 在 `量化模型_bma_ultra_enhanced.py` line 9453 中，正确的做法是：
   ```python
   snapshot_id = save_model_snapshot(
       training_results=snapshot_payload,
       ridge_stacker=stacker_to_save,  # meta_ranker_stacker 作为 ridge_stacker 传递
       ...
   )
   ```

## 修复内容

修改了 `scripts/train_full_dataset.py` line 125-133：

**修复前**:
```python
snapshot_id = save_model_snapshot(
    training_results=snapshot_payload,
    ridge_stacker=getattr(model, "ridge_stacker", None),
    lambda_rank_stacker=getattr(model, "lambda_rank_stacker", None),
    meta_ranker_stacker=getattr(model, "meta_ranker_stacker", None),  # ❌ 错误
    ...
)
```

**修复后**:
```python
# Priority: meta_ranker_stacker > ridge_stacker (MetaRankerStacker replaces RidgeStacker)
meta_ranker = getattr(model, "meta_ranker_stacker", None)
ridge_stacker_legacy = getattr(model, "ridge_stacker", None)
stacker_to_save = meta_ranker if meta_ranker is not None else ridge_stacker_legacy

snapshot_id = save_model_snapshot(
    training_results=snapshot_payload,
    ridge_stacker=stacker_to_save,  # ✅ 正确：将 MetaRankerStacker 作为 ridge_stacker 传递
    lambda_rank_stacker=getattr(model, "lambda_rank_stacker", None),
    ...
)
```

## 验证

修复后，训练应该能够成功保存snapshot，因为：
- `meta_ranker_stacker` 会作为 `ridge_stacker` 传递
- `save_model_snapshot()` 会识别它是 MetaRankerStacker 并正确保存
