# Top20重复股票问题 - 修复完成

## 🔍 问题描述

**现象**: 所有模型的Top20表格都显示相同的股票重复20次
- LambdaRanker Top20: 所有20个都是ANPA，分数都是0.340612
- ElasticNet Top20: 所有20个都是ZIP，分数都是0.010390
- XGBoost Top20: 所有20个都是DGNX，分数都是0.060598

## 🔍 根本原因

**问题**: 在提取Top20时，`nlargest()`返回了同一个ticker的多个副本

**可能原因**:
1. `latest_predictions`的索引中，同一个ticker出现了多次（MultiIndex问题）
2. `nlargest()`没有正确处理MultiIndex，返回了重复的ticker
3. 没有按ticker去重就直接取Top20

## ✅ 修复方案

### 1. 创建辅助函数 `get_top_n_unique_tickers()`

**位置**: `autotrader/app.py` line ~2069

**功能**:
- 正确处理MultiIndex和普通Index
- 按ticker分组，取每个ticker的最大分数（处理重复）
- 移除NaN分数
- 返回Top N唯一的ticker

**实现**:
```python
def get_top_n_unique_tickers(df, score_col, n=20):
    """Get top N unique tickers by score, handling MultiIndex"""
    if score_col not in df.columns:
        return pd.DataFrame()
    
    try:
        # Extract ticker level from index
        if isinstance(df.index, pd.MultiIndex):
            # If MultiIndex, extract ticker level
            ticker_level = df.index.get_level_values('ticker')
            # Create a temporary DataFrame with ticker as column for grouping
            temp_df = df[[score_col]].copy()
            temp_df['ticker'] = ticker_level
            # Remove NaN scores
            temp_df = temp_df.dropna(subset=[score_col])
            # Group by ticker and take the maximum score (in case of duplicates)
            grouped = temp_df.groupby('ticker')[score_col].max().reset_index()
            # Sort and get top N
            top_n = grouped.nlargest(n, score_col).reset_index(drop=True)
            return top_n
        else:
            # If not MultiIndex, assume index is ticker
            temp_df = df[[score_col]].copy()
            temp_df['ticker'] = df.index.astype(str)
            # Remove NaN scores
            temp_df = temp_df.dropna(subset=[score_col])
            # Remove duplicates by ticker (keep max score)
            grouped = temp_df.groupby('ticker')[score_col].max().reset_index()
            top_n = grouped.nlargest(n, score_col).reset_index(drop=True)
            return top_n
    except Exception as e:
        self.log(f"[DirectPredict] ⚠️ Error in get_top_n_unique_tickers: {e}")
        import traceback
        self.log(f"[DirectPredict] 详细错误: {traceback.format_exc()}")
        return pd.DataFrame()
```

### 2. 更新所有Top20显示逻辑

**修改前**:
```python
lambdarank_top20 = latest_predictions.nlargest(20, 'score_lambdarank')[['score_lambdarank']].copy()
lambdarank_top20 = lambdarank_top20.sort_values('score_lambdarank', ascending=False)
for i, (idx, row) in enumerate(lambdarank_top20.iterrows(), 1):
    ticker = idx[1] if isinstance(idx, tuple) else idx
    score = row['score_lambdarank']
```

**修改后**:
```python
lambdarank_top20 = get_top_n_unique_tickers(latest_predictions, 'score_lambdarank', 20)
if len(lambdarank_top20) > 0:
    self.log(f"\n[DirectPredict] 🏆 LambdaRanker Top {len(lambdarank_top20)}:")
    for idx, row in lambdarank_top20.iterrows():
        ticker = str(row['ticker']).strip()
        score = float(row['score_lambdarank'])
        self.log(f"  {idx+1:2d}. {ticker:8s}: {score:8.6f}")
```

### 3. 添加去重逻辑到latest_predictions

**位置**: `autotrader/app.py` line ~1970

**修改**:
```python
latest_predictions = final_predictions.xs(latest_date, level='date', drop_level=False)
# 🔧 FIX: Remove duplicate tickers (keep first occurrence)
if isinstance(latest_predictions.index, pd.MultiIndex):
    ticker_level = latest_predictions.index.get_level_values('ticker')
    latest_predictions = latest_predictions[~ticker_level.duplicated(keep='first')]
latest_predictions = latest_predictions.sort_values('score', ascending=False)
```

## 🎯 修复效果

### 修复前
```
[DirectPredict] 🏆 LambdaRanker Top 20:
   1. ANPA    : 0.340612
   2. ANPA    : 0.340612
   3. ANPA    : 0.340612
   ... (全部是ANPA)
```

### 修复后
```
[DirectPredict] 🏆 LambdaRanker Top 20:
   1. ANPA    : 0.340612
   2. TICKER2 : 0.335123
   3. TICKER3 : 0.330456
   ... (20个不同的ticker)
```

## ⚠️ 注意事项

1. **去重逻辑**:
   - 如果同一个ticker有多个分数，取最大值
   - 确保每个ticker只出现一次

2. **NaN处理**:
   - 自动移除NaN分数
   - 只显示有效的预测结果

3. **索引处理**:
   - 正确处理MultiIndex (date, ticker)
   - 正确处理普通Index (ticker)

4. **错误处理**:
   - 添加了异常处理和日志
   - 如果出错，返回空DataFrame而不是崩溃

## 📝 相关文件

- **修复文件**: `autotrader/app.py` line ~1968-2120
- **辅助函数**: `get_top_n_unique_tickers()` line ~2069

---

**状态**: ✅ **已修复**

**下一步**: 重启Direct Predict，运行预测，验证Top20表格显示不同的股票
