# EWMA平滑计算详解

## 计算公式

EWMA平滑使用**3天指数加权移动平均**，公式为：

```
S_smooth_t = 0.6 × S_t + 0.3 × S_{t-1} + 0.1 × S_{t-2}
```

其中：
- `S_t`: 今天的预测分数
- `S_{t-1}`: 昨天的预测分数
- `S_{t-2}`: 前天的预测分数
- 权重: (0.6, 0.3, 0.1) - 总和为1.0

## 计算逻辑

### 1. 按股票和日期处理

对每个股票（ticker）分别计算，按日期顺序处理：

```python
# 对每个日期，每个股票
for date, group in predictions_df.groupby('date'):
    for ticker in group:
        # 获取该股票的历史分数
        history = ema_history[model_name][ticker]
        # history格式: [S_t-1, S_t-2] (最新的在前)
```

### 2. 分情况计算

#### 情况1: 第一天（没有历史数据）
```
smooth_score = S_today
```
- 直接使用今天的原始分数
- 历史记录: `history = [S_today]`

#### 情况2: 第二天（只有1天历史）
```
smooth_score = 0.6 × S_today + 0.3 × S_yesterday
```
- 使用2天数据
- 历史记录: `history = [S_today, S_yesterday]`

#### 情况3: 第三天及以后（有2天以上历史）
```
smooth_score = 0.6 × S_today + 0.3 × S_yesterday + 0.1 × S_day_before_yesterday
```
- 使用完整的3天数据
- 历史记录: `history = [S_today, S_yesterday, S_day_before_yesterday]`
- 只保留最近3天，超过的会被删除

### 3. 历史记录更新

每次计算后更新历史记录：

```python
# 将今天的分数插入到历史记录的最前面
history.insert(0, score_today)

# 只保留最近2天的历史（加上今天的，共3天）
if len(history) > 2:
    history.pop()  # 删除最旧的历史
```

## 代码实现

```python
def apply_ema_smoothing(predictions_df, model_name, ema_history, 
                        weights=(0.6, 0.3, 0.1)):
    """
    应用3天EMA平滑
    """
    # 按日期排序
    predictions_df = predictions_df.sort_values('date')
    
    for date, group in predictions_df.groupby('date'):
        for idx, row in group.iterrows():
            ticker = row['ticker']
            score_today = row['prediction']
            
            # 获取该股票的历史分数
            if ticker not in ema_history[model_name]:
                ema_history[model_name][ticker] = []
            
            history = ema_history[model_name][ticker]
            
            # 计算平滑分数
            if len(history) == 0:
                # 第一天：使用原始分数
                smooth_score = score_today
            elif len(history) == 1:
                # 第二天：0.6*S_t + 0.3*S_{t-1}
                smooth_score = weights[0] * score_today + weights[1] * history[0]
            else:
                # 第三天及以后：0.6*S_t + 0.3*S_{t-1} + 0.1*S_{t-2}
                smooth_score = (weights[0] * score_today + 
                               weights[1] * history[0] + 
                               weights[2] * history[1])
            
            # 更新历史（保留最近3天）
            history.insert(0, score_today)
            if len(history) > 2:
                history.pop()
```

## 计算示例

假设某股票在连续3天的预测分数为：

| 日期 | 原始分数 (S_t) | 历史记录 | 平滑分数计算 | 平滑分数 |
|------|---------------|---------|-------------|---------|
| Day 1 | 0.85 | [] | S_1 | **0.85** |
| Day 2 | 0.92 | [0.85] | 0.6×0.92 + 0.3×0.85 | **0.897** |
| Day 3 | 0.78 | [0.92, 0.85] | 0.6×0.78 + 0.3×0.92 + 0.1×0.85 | **0.829** |
| Day 4 | 0.88 | [0.78, 0.92] | 0.6×0.88 + 0.3×0.78 + 0.1×0.92 | **0.854** |
| Day 5 | 0.91 | [0.88, 0.78] | 0.6×0.91 + 0.3×0.88 + 0.1×0.78 | **0.900** |

### 详细计算过程

**Day 1:**
- 历史: `[]` (空)
- 计算: `smooth_score = 0.85`
- 更新历史: `[0.85]`

**Day 2:**
- 历史: `[0.85]` (昨天的分数)
- 计算: `smooth_score = 0.6 × 0.92 + 0.3 × 0.85 = 0.552 + 0.255 = 0.807`
- 更新历史: `[0.92, 0.85]`

**Day 3:**
- 历史: `[0.92, 0.85]` (昨天和前天的分数)
- 计算: `smooth_score = 0.6 × 0.78 + 0.3 × 0.92 + 0.1 × 0.85 = 0.468 + 0.276 + 0.085 = 0.829`
- 更新历史: `[0.78, 0.92]` (删除最旧的0.85)

**Day 4:**
- 历史: `[0.78, 0.92]` (昨天和前天的分数)
- 计算: `smooth_score = 0.6 × 0.88 + 0.3 × 0.78 + 0.1 × 0.92 = 0.528 + 0.234 + 0.092 = 0.854`
- 更新历史: `[0.88, 0.78]` (删除最旧的0.92)

## 权重设计原理

权重 `(0.6, 0.3, 0.1)` 的设计考虑：

1. **今天权重最大 (0.6)**: 最新信息最重要
2. **昨天权重中等 (0.3)**: 近期信息有参考价值
3. **前天权重最小 (0.1)**: 历史信息影响较小
4. **权重总和 = 1.0**: 保证平滑后的分数在合理范围内

## 特殊处理

### NaN值处理

- 如果今天的分数是NaN，平滑分数也是NaN
- 如果历史分数是NaN，在计算时视为0.0

### 数据不足的情况

- **只有1天数据**: 使用原始分数
- **只有2天数据**: 使用2天加权平均
- **3天及以上**: 使用完整的3天加权平均

## 应用场景

EWMA平滑应用于：

1. **所有模型的预测分数**: catboost, lambdarank, ridge_stacking
2. **按股票分别计算**: 每个股票有独立的历史记录
3. **按日期顺序处理**: 确保时间顺序正确
4. **用于排名**: 使用平滑后的分数进行股票排名

## 效果

EWMA平滑的作用：

1. **减少噪声**: 平滑短期波动
2. **保持趋势**: 保留主要趋势信息
3. **提高稳定性**: 使预测分数更稳定
4. **改善排名**: 减少因单日异常导致的排名变化

## 注意事项

1. **前2天可能没有完整平滑效果**: 
   - Day 1: 无平滑
   - Day 2: 2天平滑
   - Day 3+: 完整3天平滑

2. **每个股票独立计算**: 不同股票的历史记录是分开的

3. **历史记录只保留2天**: 内存效率高，只存储必要的历史数据

4. **按日期排序**: 确保时间顺序正确，避免未来信息泄露
