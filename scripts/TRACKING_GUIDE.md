# 训练阻塞跟踪指南

## 已添加的跟踪日志

### 1. 模型级别跟踪

```
[FIRST_LAYER] 🔍 开始训练模型: {name}
[FIRST_LAYER] 🔍  总CV折数: {len(cv_splits_list)}
[FIRST_LAYER] 🔍  训练数据: {X.shape[0]}样本 × {X.shape[1]}特征
```

### 2. Fold级别跟踪

```
[FIRST_LAYER][{name}] 🔍 Fold {fold_idx + 1}/{len(cv_splits_list)} 开始处理
[FIRST_LAYER][{name}] 🔍  训练样本数: {len(train_idx)}, 验证样本数: {len(val_idx)}
[FIRST_LAYER][{name}] 🔍  从groups_norm计算训练窗: {train_window_days}天 (唯一日期数)
[FIRST_LAYER][{name}] 🔍  训练窗天数: {train_window_days}, 最小要求: {min_train_window_days}
[FIRST_LAYER][{name}] 🔍 Fold {fold_idx + 1}: 开始创建训练/验证数据分割
[FIRST_LAYER][{name}] 🔍 Fold {fold_idx + 1}: 数据分割完成 - X_train: {X_train.shape}, X_val: {X_val.shape}
```

### 3. 模型训练跟踪

#### ElasticNet/XGBoost/CatBoost/LambdaRank:
```
[FIRST_LAYER][{name}] 🔍 Fold {fold_idx + 1}: 开始{name}训练 (样本数: {len(X_train_use)})
[FIRST_LAYER][{name}] 🔍 Fold {fold_idx + 1}: {name}训练完成
```

#### CatBoost特殊跟踪:
```
[FIRST_LAYER][{name}] 🔍 Fold {fold_idx + 1}: CatBoost分类特征数: {len(categorical_features)}
```

## 如何定位阻塞位置

### 步骤1: 查看日志

查找最后一个成功完成的日志：

```bash
# 如果训练在运行，查看实时输出
# 或者查看日志文件（如果有）
```

### 步骤2: 识别阻塞点

根据最后一个日志判断阻塞位置：

1. **如果最后日志是 "开始训练模型: X"**
   - 阻塞在模型初始化或数据准备阶段

2. **如果最后日志是 "Fold X/Y 开始处理"**
   - 阻塞在fold处理开始阶段

3. **如果最后日志是 "训练窗天数: X"**
   - 阻塞在训练窗检查阶段

4. **如果最后日志是 "数据分割完成"**
   - 阻塞在特征选择或模型训练阶段

5. **如果最后日志是 "开始{name}训练"**
   - 阻塞在模型训练过程中（最可能的位置）

### 步骤3: 检查特定模型

如果阻塞在模型训练中：

- **ElasticNet**: 通常很快，不太可能阻塞
- **XGBoost**: 可能阻塞，检查GPU/CPU使用
- **CatBoost**: 最可能阻塞，检查od_wait参数和early stopping
- **LambdaRank**: 可能阻塞，检查LightGBM训练

### 步骤4: 检查特定fold

如果阻塞在某个fold：

- **Fold 1**: 可能是数据准备问题
- **Fold 2-5**: 可能是训练数据不足
- **Fold 6**: 可能是最后一个fold的特殊处理

## 常见阻塞原因

### 1. CatBoost训练阻塞

**症状**: 最后日志是 "开始CatBoost训练"

**可能原因**:
- `od_wait=120` 太大，等待时间过长
- Early stopping没有触发
- 内存不足

**解决方案**:
- 检查CatBoost配置中的`od_wait`参数
- 减少`iterations`参数
- 检查系统内存使用

### 2. LambdaRank训练阻塞

**症状**: 最后日志是 "开始LambdaRank训练"

**可能原因**:
- LightGBM训练时间过长
- `num_boost_round`太大
- 内存不足

**解决方案**:
- 检查LambdaRank配置中的`num_boost_round`
- 减少`early_stopping_rounds`
- 检查系统内存使用

### 3. 数据准备阻塞

**症状**: 最后日志是 "数据分割完成"或之前

**可能原因**:
- 数据量太大，复制操作慢
- 特征选择耗时
- 内存不足

**解决方案**:
- 检查数据大小
- 检查特征数量
- 检查系统内存使用

## 下一步行动

1. **运行训练并观察日志**
2. **记录最后一个成功的日志**
3. **根据日志判断阻塞位置**
4. **针对性地解决问题**
