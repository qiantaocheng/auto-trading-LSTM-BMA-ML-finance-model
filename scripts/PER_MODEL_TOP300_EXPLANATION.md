# 每个模型独立的Top300 EMA说明

## ✅ 确认：每个模型有自己独立的Top300

### 当前实现逻辑

**每个模型独立计算自己的Top300：**

1. **catboost模型**：
   - 使用catboost的prediction分数排序
   - 计算catboost的Top300
   - 只对连续3天都在catboost Top300的股票应用EMA

2. **lambdarank模型**：
   - 使用lambdarank的prediction分数排序
   - 计算lambdarank的Top300
   - 只对连续3天都在lambdarank Top300的股票应用EMA

3. **ridge_stacking模型**：
   - 使用ridge_stacking的prediction分数排序
   - 计算ridge_stacking的Top300
   - 只对连续3天都在ridge_stacking Top300的股票应用EMA

### 为什么这样设计？

**原因：**
1. **不同模型的分数分布不同**：
   - catboost的分数范围可能与lambdarank不同
   - 每个模型的预测质量不同
   - 每个模型应该有自己的高质量股票标准

2. **更合理的EMA应用**：
   - catboost的Top300股票 ≠ lambdarank的Top300股票
   - 每个模型只对自己的高质量股票应用EMA
   - 避免跨模型比较导致的偏差

3. **独立的EMA历史**：
   - 每个模型维护自己的ema_history
   - 每个模型维护自己的rank_history
   - 互不干扰

## 📊 示例说明

### 场景：某一天有3只股票

| 股票 | catboost分数 | lambdarank分数 | ridge_stacking分数 |
|------|-------------|---------------|-------------------|
| AAPL | 0.95 | 0.88 | 0.92 |
| MSFT | 0.85 | 0.95 | 0.90 |
| GOOGL | 0.80 | 0.82 | 0.85 |

### catboost模型的Top300（假设只有3只股票）

**排名：**
1. AAPL (0.95)
2. MSFT (0.85)
3. GOOGL (0.80)

**EMA应用：**
- 如果连续3天都在catboost Top300 → 应用EMA
- 每个股票的排名基于catboost分数

### lambdarank模型的Top300

**排名：**
1. MSFT (0.95)
2. AAPL (0.88)
3. GOOGL (0.82)

**EMA应用：**
- 如果连续3天都在lambdarank Top300 → 应用EMA
- 每个股票的排名基于lambdarank分数

### ridge_stacking模型的Top300

**排名：**
1. AAPL (0.92)
2. MSFT (0.90)
3. GOOGL (0.85)

**EMA应用：**
- 如果连续3天都在ridge_stacking Top300 → 应用EMA
- 每个股票的排名基于ridge_stacking分数

## 🔍 代码验证

### 主循环（time_split_80_20_oos_eval.py）

```python
for model_name, pred_list in all_results.items():
    # 每个模型分别处理
    all_results[model_name] = pd.concat(pred_list, axis=0, ignore_index=True)
    
    # 每个模型独立调用EMA函数
    all_results[model_name] = apply_ema_smoothing_top300_filter(
        all_results[model_name],  # 只包含该模型的prediction列
        model_name=model_name,    # 模型名称
        ema_history=ema_history,   # 共享字典，但按model_name隔离
        ...
    )
```

### EMA函数（apply_ema_smoothing_top300.py）

```python
for date, group in predictions_df.groupby('date'):
    # 对每个日期的股票，按该模型的prediction排序
    prediction_values = group['prediction'].values  # 该模型的分数
    
    # 计算该模型的Top300
    top300_indices = np.argpartition(prediction_values, -top_n)[-top_n:]
    
    # 每个模型独立计算排名
    rank_map = {...}  # 基于该模型的prediction
```

## ✅ 关键点

1. **每个模型独立排序**：
   - catboost按catboost分数排序
   - lambdarank按lambdarank分数排序
   - ridge_stacking按ridge_stacking分数排序

2. **每个模型独立的历史记录**：
   - `ema_history[model_name][ticker]`：按模型隔离
   - `rank_history[ticker]`：在函数内部，按模型调用隔离

3. **每个模型独立的Top300判断**：
   - 每个模型只关心自己的Top300
   - 不跨模型比较

## 📈 实际效果

### 优势

1. **更精确的EMA应用**：
   - 每个模型只对自己的高质量股票应用EMA
   - 避免低质量模型的噪声影响

2. **模型特异性**：
   - 不同模型有不同的股票偏好
   - 每个模型保持自己的特性

3. **更好的回测表现**：
   - 每个模型优化自己的Top300
   - 整体表现可能更好

### 示例统计

假设某一天：
- **catboost Top300**：300只股票
- **lambdarank Top300**：300只股票（可能不同）
- **ridge_stacking Top300**：300只股票（可能不同）

**重叠情况：**
- 三个模型都包含的股票：~150只（估计）
- 只在catboost Top300：~100只
- 只在lambdarank Top300：~100只
- 只在ridge_stacking Top300：~100只

## 🎯 总结

**是的，每个模型有自己独立的Top300！**

- ✅ catboost有自己的Top300（基于catboost分数）
- ✅ lambdarank有自己的Top300（基于lambdarank分数）
- ✅ ridge_stacking有自己的Top300（基于ridge_stacking分数）

这是**正确的设计**，因为：
1. 不同模型的分数分布不同
2. 每个模型应该有自己的高质量标准
3. 避免跨模型比较的偏差
