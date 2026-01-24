# Direct Predict 功能说明

## 📋 功能概述

Direct Predict（直接预测）是一个快速预测功能，使用已保存的模型快照进行预测，**无需重新训练模型**。该功能会自动从 Polygon API 获取市场数据，计算特征，并使用 BMA Ultra 模型生成预测结果。

---

## 🎯 核心特点

1. **无需训练**：直接使用已保存的模型快照
2. **自动数据获取**：自动从 Polygon API 获取市场数据
3. **自动特征计算**：自动计算所有17个因子特征
4. **原始预测分数**：使用模型原始输出，**不应用EMA平滑**
5. **Excel报告**：自动生成Excel排名报告
6. **数据库记录**：预测结果保存到数据库供审计

---

## 🚀 使用方法

### 启动方式

1. **打开应用**：运行 `app.py` 或 `launch_gui.py`
2. **找到按钮**：在策略选择区域找到 **"Direct Predict (Snapshot)"** 按钮
3. **点击按钮**：点击按钮启动预测流程

### 操作步骤

#### 步骤1：输入股票代码
- 如果已选择股票池：自动使用股票池中的股票
- 如果未选择：弹出对话框，输入股票代码（逗号分隔）
  - 示例：`AAPL,MSFT,GOOGL,TSLA,NVDA`

#### 步骤2：自动执行（无需输入）
- **自动使用上一个交易日的数据**
- **自动预测未来 T+10 天**（horizon=10）
- 无需任何输入，直接开始预测

#### 步骤3：查看结果
系统自动执行以下操作：
1. 获取市场数据
2. 计算特征
3. 生成预测
4. 保存结果

---

## 🔄 完整工作流程

### 流程图

```
用户点击 "Direct Predict (Snapshot)"
    ↓
获取股票列表（从股票池或数据文件默认值）
    ↓
自动使用上一个交易日作为基准日期
    ↓
自动设置预测horizon为T+10天
    ↓
初始化 BMA Ultra 模型
    ↓
计算所需数据范围（280天历史 + 预测天数）
    ↓
从 Polygon API 获取市场数据
    ↓
计算所有17个因子特征
    ↓
对每个预测日期：
    ├─ 提取该日期的特征数据
    ├─ 加载模型快照（从 latest_snapshot_id.txt）
    ├─ 使用模型生成预测
    └─ 保存预测结果
    ↓
合并所有日期的预测结果
    ↓
使用原始预测分数（无EMA平滑）
    ↓
生成Excel报告
    ↓
保存到数据库
    ↓
显示Top推荐股票
```

---

## 📊 详细逻辑说明

### 1. 股票列表获取

**优先级顺序**：
1. **已选择的股票池**：如果用户之前选择了股票池，自动使用
2. **用户输入**：如果没有股票池，弹出对话框让用户输入

**代码逻辑**：
```python
# 优先使用股票池
if hasattr(self, 'selected_pool_info') and self.selected_pool_info:
    tickers = self.selected_pool_info['tickers']
else:
    # 否则提示用户输入
    tickers = 用户输入的股票代码列表
```

---

### 2. 数据范围计算

**为什么需要280天历史数据？**

因为因子计算需要滚动窗口：
- **252天滚动窗口**：某些因子（如 `near_52w_high`）需要252个交易日的历史数据
- **缓冲天数**：考虑周末和节假日，需要额外28天缓冲
- **总计**：280天历史数据 + 预测天数

**计算公式**：
```python
MIN_REQUIRED_LOOKBACK_DAYS = 280  # 252交易日 + 28天缓冲
total_lookback_days = 280 + prediction_days
start_date = today - total_lookback_days
end_date = today
```

**示例**：
- 如果今天是 2026-01-21，预测1天
- 数据范围：2025-04-17 至 2026-01-21（约281天）

---

### 3. 市场数据获取

**数据源**：Polygon.io API

**获取方式**：
- 使用 `Simple17FactorEngine.fetch_market_data()`
- 一次性获取所有需要的数据（高效）
- 支持优化的下载器（批量下载）

**获取的数据包括**：
- 开盘价（Open）
- 收盘价（Close）
- 最高价（High）
- 最低价（Low）
- 成交量（Volume）
- 交易日期（Date）

---

### 4. 特征计算

**计算引擎**：`Simple17FactorEngine`

**计算的因子**（17个因子）：
1. `momentum_60d` - 60日动量
2. `rsi_21` - 21日RSI
3. `bollinger_squeeze` - 布林带挤压
4. `obv_momentum_60d` - 60日OBV动量
5. `atr_ratio` - ATR比率
6. `blowoff_ratio` - 抛售比率
7. `hist_vol_40d` - 40日历史波动率
8. `vol_ratio_20d` - 20日成交量比率
9. `near_52w_high` - 接近52周高点
10. `price_ma60_deviation` - 价格相对60日均线偏离
11. `ret_skew_20d` - 20日收益偏度
12. `trend_r2_60` - 60日趋势R²
13. `5_days_reversal` - 5日反转
14. `liquid_momentum` - 流动性动量
15. `obv_divergence` - OBV背离
16. `ivol_20` - 20日隐含波动率
17. `roa`, `ebit`, `downside_beta_252`, `making_new_low_5d` 等

**计算模式**：`mode='predict'`（预测模式，不计算目标变量）

---

### 5. 模型快照加载

**快照ID来源**（优先级顺序）：
1. **`latest_snapshot_id.txt`**：项目根目录的文件（优先）
2. **数据库最新快照**：如果文件不存在，使用数据库中最新的快照

**代码逻辑**：
```python
snapshot_id_to_use = None
try:
    # 尝试从文件读取
    latest_snapshot_file = Path("latest_snapshot_id.txt")
    if latest_snapshot_file.exists():
        snapshot_id_to_use = latest_snapshot_file.read_text().strip()
except Exception:
    # 如果失败，使用 None（自动使用数据库最新快照）
    pass
```

**使用的模型**：
- **BMA Ultra 模型**：`UltraEnhancedQuantitativeModel`
- **包含的模型**：
  - ElasticNet（线性模型）
  - XGBoost（梯度提升）
  - CatBoost（分类提升）
  - LambdaRank（排序模型）
  - MetaRankerStacker（元学习堆叠模型）

---

### 6. 预测生成

**对每个预测日期**：

1. **提取特征数据**：
   ```python
   # 提取到该日期为止的所有数据
   date_mask = all_feature_data.index.get_level_values('date') <= pred_date
   date_feature_data = all_feature_data[date_mask].copy()
   ```

2. **调用模型预测**：
   ```python
   results = model.predict_with_snapshot(
       feature_data=date_feature_data,
       snapshot_id=snapshot_id_to_use,
       universe_tickers=tickers,
       as_of_date=pred_date,
       prediction_days=1
   )
   ```

3. **获取原始预测**：
   ```python
   # 使用原始预测分数（无EMA平滑）
   predictions_raw = results.get('predictions_raw')
   if predictions_raw is None:
       predictions_raw = results.get('predictions')  # 备用
   ```

**预测输出格式**：
- **MultiIndex DataFrame**：`(date, ticker)` 作为索引
- **score列**：预测分数（数值越大越好）

---

### 7. 结果处理

#### 合并预测结果
```python
# 合并所有日期的预测
if len(all_predictions) == 1:
    combined_predictions = all_predictions[0]
else:
    combined_predictions = pd.concat(all_predictions, axis=0)
```

#### 使用原始分数
```python
# 不使用EMA平滑，直接使用原始预测分数
final_predictions = combined_predictions.copy()
final_predictions['score_raw'] = final_predictions['score']
```

**重要**：Direct Predict **不使用EMA平滑**，直接使用模型的原始输出分数。

---

### 8. Excel报告生成

**报告内容**：
- 所有股票的预测分数
- 按分数排序
- 包含日期和股票代码
- 原始分数（无平滑）

**保存位置**：
```
results/direct_predict_YYYYMMDD_HHMMSS.xlsx
```

**报告格式**：
- **Sheet1**：所有预测结果
- **列**：date, ticker, score, score_raw
- **排序**：按score降序排列

---

### 9. 数据库保存

**数据库**：`data/monitoring.db`

**表结构**：
```sql
CREATE TABLE direct_predictions (
    ts INTEGER,           -- 时间戳
    snapshot_id TEXT,     -- 使用的快照ID
    ticker TEXT,          -- 股票代码
    score REAL            -- 预测分数
)
```

**保存内容**：
- 最新日期的Top推荐股票
- 用于审计和追踪

---

### 10. Top推荐显示

**显示内容**：
- Top 20 推荐股票（按分数排序）
- 格式：`排名. 股票代码: 预测分数`

**示例输出**：
```
[DirectPredict] 🏆 Top 20 推荐:
   1. NVDA    : 0.852341
   2. TSLA    : 0.789123
   3. AAPL    : 0.765432
   ...
```

---

## ⚙️ 技术细节

### 数据获取优化

**一次性获取策略**：
- 不是每天单独获取数据
- 一次性获取所有需要的数据（280+天）
- 然后按日期切片使用

**优势**：
- 减少API调用次数
- 提高效率
- 降低网络延迟影响

### 特征计算优化

**预计算策略**：
- 一次性计算所有日期的特征
- 然后按日期提取使用

**优势**：
- 避免重复计算
- 提高计算效率
- 确保数据一致性

### 快照管理

**快照优先级**：
1. `latest_snapshot_id.txt`（显式指定）
2. 数据库最新快照（自动回退）

**优势**：
- 可以指定特定快照
- 自动回退机制
- 灵活的快照管理

---

## 📝 输出文件

### Excel报告
- **位置**：`results/direct_predict_YYYYMMDD_HHMMSS.xlsx`
- **内容**：所有预测结果，按分数排序

### 数据库记录
- **位置**：`data/monitoring.db`
- **表名**：`direct_predictions`
- **内容**：Top推荐股票的预测记录

### 日志输出
- **位置**：应用日志窗口
- **内容**：详细的执行过程日志

---

## ⚠️ 注意事项

### 1. 数据依赖
- **需要网络连接**：从Polygon API获取数据
- **需要API密钥**：确保Polygon API配置正确
- **数据延迟**：使用最新可用数据（可能有1-2天延迟）

### 2. 快照要求
- **快照必须存在**：确保有可用的模型快照
- **快照必须完整**：包含所有必要的模型文件
- **快照必须兼容**：使用兼容的模型版本

### 3. 股票代码格式
- **大写字母**：自动转换为大写
- **逗号分隔**：多个股票用逗号分隔
- **有效代码**：确保股票代码在Polygon API中存在

### 4. 预测Horizon
- **默认T+10天**：使用配置中的 `prediction_horizon_days=10`
- **自动设置**：无需手动输入，系统自动使用T+10
- **数据要求**：需要足够的历史数据（280天+）

---

## 🔍 故障排查

### 问题1：无法获取数据
**可能原因**：
- Polygon API密钥未配置
- 网络连接问题
- 股票代码无效

**解决方法**：
- 检查API配置
- 检查网络连接
- 验证股票代码

### 问题2：无法加载快照
**可能原因**：
- 快照文件不存在
- 快照文件损坏
- 快照ID错误

**解决方法**：
- 检查 `latest_snapshot_id.txt`
- 检查快照文件完整性
- 重新训练模型生成新快照

### 问题3：特征计算失败
**可能原因**：
- 数据不足（少于280天）
- 数据格式错误
- 因子计算错误

**解决方法**：
- 确保有足够的历史数据
- 检查数据格式
- 查看详细错误日志

---

## 📊 性能指标

### 典型执行时间
- **10只股票，1天预测**：约30-60秒
- **100只股票，1天预测**：约2-5分钟
- **1000只股票，1天预测**：约10-20分钟

### 影响因素
- **股票数量**：股票越多，时间越长
- **预测天数**：天数越多，时间越长
- **网络速度**：影响数据获取速度
- **API限制**：Polygon API速率限制

---

## ✅ 总结

Direct Predict 是一个**快速、便捷的预测工具**，特点：

1. ✅ **无需训练**：直接使用已保存的快照
2. ✅ **自动化**：自动获取数据和计算特征
3. ✅ **原始分数**：使用模型原始输出，无EMA平滑
4. ✅ **完整报告**：生成Excel报告和数据库记录
5. ✅ **易于使用**：简单的GUI操作

**适用场景**：
- 快速查看股票预测
- 不需要重新训练模型
- 需要最新预测结果
- 批量股票预测

**不适用场景**：
- 需要重新训练模型
- 需要EMA平滑的预测
- 需要历史回测

---

## 📚 相关文档

- `scripts/FULL_DATASET_TRAINING_SETUP.md` - 完整数据集训练说明
- `scripts/DIRECT_PREDICT_NO_EMA_UPDATE.md` - EMA移除更新说明
- `docs/累计收益与年化收益的区别.md` - 收益指标说明
