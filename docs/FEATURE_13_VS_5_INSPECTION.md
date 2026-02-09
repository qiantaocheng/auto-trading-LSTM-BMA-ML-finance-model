# 输入 13 个因子、模型只用 5 个时，另外 8 个会被强制参与训练吗？

**结论：不会。** 代码里无论是训练还是预测，都只让模型看到“规定的那几列”；多出来的列不会被塞进模型。

---

## 1. 预测阶段（80/20 用 snapshot）

**脚本：** `scripts/time_split_80_20_oos_eval.py`

- 先按 `allowed_feature_cols` 从数据里取列，得到 `all_feature_cols`（例如 13 列），再得到 `X = date_data[all_feature_cols]`。
- 对**每个模型**预测前会调 `align_test_features_with_model(X, model, ...)`：
  - 从模型上取**训练时用的特征名**：`train_features = model.feature_names_in_`（或 `feature_name_` / `feature_names` 等）。
  - 只从 `X_test` 里**按这串名字选列**：`X_aligned = X_test[train_features].copy()`（约 99 行）。
  - 若测试数据里少某列，会补 0，不会把“多出来的 8 列”喂给该模型。

因此：**模型训练时是 5 个因子，预测时就只收到这 5 列；另外 8 列不会被强制参与预测。**

---

## 2. 训练阶段（BMA/Ultra 管线）

**逻辑位置：** `bma_models/量化模型_bma_ultra_enhanced.py`

### 2.1 训练用哪些列

- 先从数据里得到“所有特征列”：去掉 `target`、`Close` 等，得到 `feature_cols`（若数据有 13 个因子，这里就是 13 列）。
- 再经 `_apply_feature_subset(feature_cols, ...)`：只做 whitelist/blacklist/compulsory 过滤，不在这里把 13 强行扩成更多列。
- 每个**一层模型**用哪几列，由 `_get_first_layer_feature_cols_for_model(model_name, list(X.columns), available_cols=X.columns)` 决定（例如 11171 行对 lambdarank 的调用）。

### 2.2 `_get_first_layer_feature_cols_for_model`（约 6820–6862 行）

- 入参：当前 `X.columns`（例如 13 列）和 `available_cols`。
- 若设置了 **`BMA_FEATURE_OVERRIDES`**（或类内 `first_layer_feature_overrides`）且该模型有配置：
  - 例如 `first_layer_feature_overrides['lambdarank'] = [5 个因子名]`
  - 则只从 `feature_cols` 里保留这 5 个（再并上 compulsory），返回的就是 5 列。
- 若该模型**没有** overrides（`opt is None`）：
  - 返回的是 `_apply_feature_subset(cols_in_order, ...)`，即当前数据里所有特征列（例如 13 列），**不会**自动变成 5 列。

因此：  
- **若你希望某模型只用到 5 个因子**：通过 `BMA_FEATURE_OVERRIDES`（或等价 overrides）把该模型设为这 5 个即可；训练时只会用这 5 列，**另外 8 列不会参与该模型的训练**。  
- **若你没有设 overrides**：该模型会用数据里所有特征列（例如 13 列），不存在“数据有 13 列却只训练 5 列”的默认行为；反之，“只用 5 列”一定是因为某处显式限制了 5 列（overrides / snapshot 训练时的配置）。

---

## 3. 小结

| 场景 | 输入列数 | 模型实际用的列数 | 多出来的 8 列会强制参与吗？ |
|------|----------|------------------|----------------------------|
| **预测（snapshot）** | 13 | 5（由 snapshot 的 feature_names 决定） | **不会**；`align_test_features_with_model` 只传 `train_features` 那 5 列。 |
| **训练（有 overrides）** | 13 | 5（由 BMA_FEATURE_OVERRIDES 等决定） | **不会**；`_get_first_layer_feature_cols_for_model` 只返回规定的 5 列。 |
| **训练（无 overrides）** | 13 | 13 | 所有列都会用，不存在“只用 5 列、却强制用 8 列”的情况。 |

**直接回答：**  
假设你给的输入有 13 个因子，而模型只用到 5 个（例如用 snapshot 或 overrides 限制为 5）：  
**另外 8 个因子不会被强制参与训练或预测；** 训练时该模型只收到你规定的那 5 列，预测时也只传入那 5 列。
