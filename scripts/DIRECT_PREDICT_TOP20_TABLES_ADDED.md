# Direct Predict Top20 Tables - 添加完成

## ✅ 修改内容

### 1. UI日志显示（`autotrader/app.py`）

在`_direct_predict_snapshot`方法中添加了CatBoost和LambdaRanker的Top20表格显示：

**位置**: Line ~2041-2070

**添加内容**:
- **MetaRankerStacker Top20**: 原有功能，保持不变
- **CatBoost Top20**: 新增，显示CatBoost分数最高的20只股票
- **LambdaRanker Top20**: 新增，显示LambdaRanker分数最高的20只股票

**显示格式**:
```
[DirectPredict] 🏆 CatBoost Top 20:
   1. TICKER1 : 0.123456
   2. TICKER2 : 0.123455
   ...
```

### 2. Excel报告（`scripts/direct_predict_ewma_excel.py`）

在`generate_excel_ranking_report`函数中添加了两个新的工作表：

**位置**: Line ~298-400

**新增工作表**:
1. **"CatBoost Top20"** 工作表
   - 显示CatBoost分数最高的20只股票
   - 列：Rank, Ticker, CatBoost Score
   - 按分数降序排列

2. **"LambdaRanker Top20"** 工作表
   - 显示LambdaRanker分数最高的20只股票
   - 列：Rank, Ticker, LambdaRanker Score
   - 按分数降序排列

**原有工作表保持不变**:
- "Ranking Report": MetaRankerStacker Top20（包含所有模型分数）
- "Summary": 统计摘要

### 3. 导入路径修复（`autotrader/app.py`）

修复了Excel报告函数的导入路径：

**位置**: Line ~1614-1617

**修改**:
- 添加了scripts目录到sys.path
- 确保可以正确导入`direct_predict_ewma_excel`模块

---

## 📊 Excel报告结构

生成的Excel文件现在包含以下工作表：

1. **Ranking Report** (主表)
   - MetaRankerStacker Top20
   - 包含所有模型的分数对比
   - 包含昨日分数和变化

2. **CatBoost Top20** (新增)
   - CatBoost分数最高的20只股票
   - 独立排序和显示

3. **LambdaRanker Top20** (新增)
   - LambdaRanker分数最高的20只股票
   - 独立排序和显示

4. **Summary** (统计摘要)
   - 各模型的平均分数统计

---

## 🎯 使用效果

### UI日志输出示例

```
[DirectPredict] 🏆 MetaRankerStacker Top 20 推荐:
   1. AAPL    : 0.756736
   2. MSFT    : 0.755432
   ...

[DirectPredict] 🏆 CatBoost Top 20:
   1. NVDA    : 0.823456
   2. TSLA    : 0.812345
   ...

[DirectPredict] 🏆 LambdaRanker Top 20:
   1. GOOGL   : 0.789012
   2. AMZN    : 0.778901
   ...
```

### Excel文件结构

- **Sheet 1: Ranking Report** - MetaRankerStacker综合排名
- **Sheet 2: CatBoost Top20** - CatBoost独立排名
- **Sheet 3: LambdaRanker Top20** - LambdaRanker独立排名
- **Sheet 4: Summary** - 统计摘要

---

## ⚠️ 注意事项

1. **数据可用性**:
   - CatBoost和LambdaRanker的Top20表格只有在`score_catboost`和`score_lambdarank`列存在时才会显示
   - 如果数据不可用，会在日志中显示警告信息

2. **排序逻辑**:
   - CatBoost Top20按`score_catboost`降序排列
   - LambdaRanker Top20按`score_lambdarank`降序排列
   - MetaRankerStacker Top20按`score`（MetaRankerStacker最终分数）降序排列

3. **Top N限制**:
   - 默认显示Top20（可通过`top_n`参数调整）
   - 如果可用股票少于20只，显示所有可用股票

---

## 📝 相关文件

- **UI显示**: `autotrader/app.py` line ~2041-2070
- **Excel报告**: `scripts/direct_predict_ewma_excel.py` line ~298-400
- **导入修复**: `autotrader/app.py` line ~1614-1617

---

**状态**: ✅ **已完成**

**下一步**: 重启Direct Predict，运行预测，查看UI日志和Excel报告中的Top20表格
