# Top300 EMA策略设计

## 当前状态

**目前完全没有filter**：
- ✅ EWMA对所有股票都应用
- ✅ 没有top300过滤
- ✅ 所有股票都使用EMA平滑后的分数

## 两种Top300 EMA策略

### 策略1: 连续3天都在top300才应用EMA（推荐）

**逻辑**：
- 检查股票在过去3天是否**都在**top300
- 如果**连续3天都在top300**：应用EMA平滑
- 如果**不满足条件**：使用原始分数（不应用EMA）

**优点**：
- 只对稳定高质量股票应用EMA
- 避免对低质量股票的噪声进行平滑
- 提高EMA的质量和稳定性

**缺点**：
- 实现稍复杂
- 需要跟踪历史排名

**实现**：`apply_ema_smoothing_top300_filter()`

### 策略2: 今天在top300就应用EMA（简单版）

**逻辑**：
- 检查股票**今天**是否在top300
- 如果**今天在top300**：应用EMA平滑
- 如果**今天不在top300**：使用原始分数

**优点**：
- 实现简单
- 只对当前高质量股票应用EMA

**缺点**：
- 可能对刚进入top300的股票应用EMA（历史可能不稳定）

**实现**：`apply_ema_smoothing_top300_alternative()`

## 推荐方案：策略1（连续3天）

### 实现细节

```python
def apply_ema_smoothing_top300_filter(
    predictions_df, 
    model_name, 
    ema_history, 
    weights=(0.6, 0.3, 0.1),
    top_n=300,
    min_days_in_top=3
):
    """
    只对连续3天都在top300的股票应用EMA
    """
    # 1. 计算每天的排名
    # 2. 跟踪每只股票的历史排名
    # 3. 检查是否连续3天都在top300
    # 4. 满足条件：应用EMA
    # 5. 不满足条件：使用原始分数
```

### 计算流程

```
对于每一天：
  1. 计算所有股票的排名（按prediction降序）
  2. 对每只股票：
     a. 检查今天和过去2天的排名
     b. 如果连续3天都在top300：
        - 应用EMA: 0.6*S_t + 0.3*S_{t-1} + 0.1*S_{t-2}
     c. 如果不满足：
        - 使用原始分数: S_t
  3. 更新历史排名和分数
```

### 示例

假设某股票3天的排名和分数：

| 日期 | 排名 | 原始分数 | 是否在top300 | EMA应用 | 平滑分数 |
|------|------|---------|-------------|---------|---------|
| Day 1 | 150 | 0.85 | ✅ | ❌ (第一天) | 0.85 |
| Day 2 | 200 | 0.92 | ✅ | ❌ (只有2天) | 0.92 |
| Day 3 | 250 | 0.78 | ✅ | ✅ (连续3天) | 0.829 |
| Day 4 | 350 | 0.88 | ❌ | ❌ (不在top300) | 0.88 (原始) |
| Day 5 | 180 | 0.91 | ✅ | ❌ (不连续) | 0.91 (原始) |

**说明**：
- Day 1-2: 数据不足，使用原始分数
- Day 3: 连续3天在top300，应用EMA
- Day 4: 不在top300，使用原始分数
- Day 5: 虽然今天在top300，但Day 4不在，不连续，使用原始分数

## 集成到现有代码

### 修改 `time_split_80_20_oos_eval.py`

```python
# 原来的调用
all_results[model_name] = apply_ema_smoothing(
    all_results[model_name], 
    model_name=model_name,
    ema_history=ema_history,
    weights=(0.6, 0.3, 0.1)
)

# 改为使用top300 filter版本
from scripts.apply_ema_smoothing_top300 import apply_ema_smoothing_top300_filter

all_results[model_name] = apply_ema_smoothing_top300_filter(
    all_results[model_name], 
    model_name=model_name,
    ema_history=ema_history,
    weights=(0.6, 0.3, 0.1),
    top_n=300,  # 只对top300应用
    min_days_in_top=3  # 需要连续3天
)
```

### 添加命令行参数

```python
p.add_argument("--ema-top-n", type=int, default=None, 
               help="Only apply EMA to stocks in top N (default: None, apply to all)")
p.add_argument("--ema-min-days", type=int, default=3,
               help="Minimum consecutive days in top N to apply EMA (default: 3)")
```

## 效果预期

### 优势

1. **提高EMA质量**：
   - 只对稳定高质量股票应用EMA
   - 减少低质量股票的噪声影响

2. **更稳定的排名**：
   - Top300股票排名更稳定
   - 减少因单日异常导致的排名变化

3. **更好的回测表现**：
   - 关注高质量股票
   - 减少低质量股票的干扰

### 统计信息

可以添加统计信息：
- 每天有多少股票应用了EMA
- 每天有多少股票使用了原始分数
- EMA覆盖率（应用EMA的股票比例）

## 建议

1. **先测试策略2（简单版）**：
   - 实现简单
   - 快速验证效果

2. **如果效果好，升级到策略1**：
   - 更严格的条件
   - 更高的质量保证

3. **参数调优**：
   - `top_n`: 可以尝试200, 300, 500
   - `min_days_in_top`: 可以尝试2, 3, 4

4. **对比测试**：
   - 无filter vs top300 filter
   - 对比回测结果
   - 选择最优方案
