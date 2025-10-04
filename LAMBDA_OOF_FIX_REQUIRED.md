# 🚨 Lambda OOF修复需求

## 问题诊断

### ❌ 当前实现的严重问题

**Lambda没有生成真正的OOF预测！**

**位置**: `bma_models/lambda_rank_stacker.py:500-504`

```python
# 当前代码（有问题）
if cv_models:
    return cv_models[0]  # ← 只返回第一个fold的模型！
```

**后果**:
1. Lambda使用了PurgedCV（✅ 好的）
2. 但只保留第一个fold的模型
3. 当用这个模型预测训练数据时 → **数据泄漏！**
   - fold-1模型见过了fold-2/3/4的数据
   - OOF预测包含"见过的数据"
4. Lambda percentile加入Ridge → Ridge学习到泄漏信号 → 过拟合

---

## ✅ 修复方案

### 修改1: `lambda_rank_stacker.py`

#### A. 添加OOF存储（类初始化）

在`__init__`方法中添加：
```python
# Line 80左右，在self.fitted_后面
self._oof_predictions = None  # 存储OOF预测
```

#### B. 修改CV训练逻辑（生成OOF）

在`_train_with_purged_cv`方法中（Line 433左右）：

```python
# 在cv_models = []之后添加
oof_predictions = np.zeros(len(X_scaled))  # 初始化OOF数组

# 在fold循环中，Line 482左右，val_pred = model.predict(X_val_fold)之后：
oof_predictions[val_idx] = val_pred  # 保存OOF预测

# 在return之前（Line 500-504）修改为：
if cv_models:
    # 保存OOF预测
    self._oof_predictions = oof_predictions
    logger.info(f"   ✓ OOF预测已生成: {len(oof_predictions)} 个样本")

    # 返回最后一个模型（或所有模型的平均）
    return cv_models[-1]  # 用最后一个模型（见过最多数据）
```

#### C. 添加get_oof方法

```python
def get_oof_predictions(self, df: pd.DataFrame) -> pd.Series:
    """
    获取OOF预测

    Args:
        df: 原始训练数据（用于提取索引）

    Returns:
        OOF预测Series（带MultiIndex）
    """
    if self._oof_predictions is None:
        raise RuntimeError("OOF预测未生成，可能模型未使用CV训练")

    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("df必须有MultiIndex(date, ticker)")

    # 创建Series（使用df的索引）
    oof_series = pd.Series(self._oof_predictions, index=df.index, name='lambda_oof')

    logger.info(f"✓ 返回Lambda OOF预测: {len(oof_series)} 个样本")
    return oof_series
```

---

### 修改2: `unified_parallel_training_engine.py`

修改Lambda OOF获取逻辑（Line 244左右）：

```python
# 旧代码（会泄漏）
lambda_oof = lambda_model.predict(lambda_data)

# 新代码（使用真正的OOF）
lambda_oof = lambda_model.get_oof_predictions(lambda_data)
```

---

## 📊 验证OOF正确性

训练后检查：

```python
# 1. 检查OOF是否存在
assert lambda_model._oof_predictions is not None, "OOF未生成！"

# 2. 检查OOF覆盖率
oof_count = (lambda_model._oof_predictions != 0).sum()
print(f"OOF覆盖率: {oof_count / len(lambda_model._oof_predictions) * 100:.1f}%")

# 3. 检查数据泄漏
# OOF的均值应该接近0（因为模型未见过这些数据）
# 如果均值偏离很大 → 可能泄漏
print(f"OOF均值: {lambda_model._oof_predictions.mean():.6f}")
```

---

## 🎯 总结

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| **CV训练** | ✅ 使用PurgedCV | ✅ 使用PurgedCV |
| **OOF生成** | ❌ 无 | ✅ 有 |
| **返回模型** | fold-0 | fold-last（见过最多数据）|
| **数据泄漏** | ❌ 有（严重） | ✅ 无 |
| **Ridge输入** | 泄漏的percentile | 真正的OOF percentile |

---

## ⚠️ 重要性

**不修复此问题，整个percentile融合方案都是无效的（甚至有害）！**

必须先修复Lambda OOF，再测试新融合策略的效果。
