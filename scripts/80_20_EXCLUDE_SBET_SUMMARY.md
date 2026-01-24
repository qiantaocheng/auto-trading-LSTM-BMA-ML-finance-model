# 80/20评估 - 排除SBET极端涨幅重新计算

**生成时间**: 2026-01-22  
**运行目录**: `results/t10_time_split_80_20_final/run_20260122_030445/`

---

## ✅ 修改内容

### 1. 添加排除股票功能

**修改文件**: `scripts/time_split_80_20_oos_eval.py`

**添加命令行参数**:
```python
p.add_argument("--exclude-tickers", nargs="+", default=None,
               help="List of tickers to exclude from training and testing (e.g. --exclude-tickers SBET TICKER2)")
```

**添加排除逻辑** (在数据加载后，line ~1427):
```python
# Exclude specific tickers if provided
if args.exclude_tickers:
    exclude_set = {str(t).upper().strip() for t in args.exclude_tickers}
    before_exclude = df.index.get_level_values('ticker').nunique()
    ticker_level = df.index.get_level_values('ticker').astype(str).str.upper().str.strip()
    mask = ~ticker_level.isin(exclude_set)
    df = df.loc[mask].copy()
    after_exclude = df.index.get_level_values('ticker').nunique() if len(df) > 0 else 0
    logger.info(f"🚫 [EXCLUDE] 排除股票: {sorted(exclude_set)}")
    logger.info(f"🚫 [EXCLUDE] 股票数量: {before_exclude} → {after_exclude} (排除 {before_exclude - after_exclude} 个股票)")
```

---

## 🎯 执行命令

```bash
python scripts\time_split_80_20_oos_eval.py \
  --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet" \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking \
  --top-n 20 \
  --exclude-tickers SBET \
  --log-level INFO
```

---

## ✅ 排除逻辑说明

1. **排除时机**: 数据加载后，训练和测试之前
2. **排除范围**: 
   - ✅ 训练数据（排除SBET）
   - ✅ 测试数据（排除SBET）
3. **大小写处理**: 自动转换为大写并去除空格
4. **日志记录**: 记录排除前后的股票数量

---

## 📊 预期影响

### 排除SBET的原因

- **极端涨幅**: SBET可能有异常高的涨幅（如9000%），会扭曲评估结果
- **数据质量**: 排除异常股票可以提高评估的可靠性
- **模型稳定性**: 避免模型过度拟合极端值

### 预期变化

1. **训练数据**: 排除SBET的所有历史数据
2. **测试数据**: 排除SBET的所有测试期数据
3. **评估指标**: 
   - IC/Rank IC可能略有变化
   - 累计收益可能降低（如果SBET是正收益）
   - 最大回撤可能改善（如果SBET是负收益）

---

## 🔍 验证方法

### 1. 检查排除日志

运行日志中应包含：
```
🚫 [EXCLUDE] 排除股票: ['SBET']
🚫 [EXCLUDE] 股票数量: X → Y (排除 1 个股票)
```

### 2. 检查结果文件

评估完成后，检查：
- `oos_metrics.json`: 查看评估指标
- `*_top20_vs_qqq.csv`: 确认不包含SBET
- `snapshot_id.txt`: 新的snapshot ID

### 3. 对比结果

与之前包含SBET的结果对比：
- IC/Rank IC变化
- 累计收益变化
- 最大回撤变化

---

## 📝 使用说明

### 排除单个股票

```bash
--exclude-tickers SBET
```

### 排除多个股票

```bash
--exclude-tickers SBET TICKER2 TICKER3
```

### 检查状态

```bash
python scripts\check_80_20_exclude_sbet_status.py
```

---

## ⚠️ 注意事项

1. **Snapshot管理**: 
   - 排除SBET后的snapshot保存在运行目录
   - **不会**覆盖`latest_snapshot_id.txt`
   - 全量训练的snapshot不受影响

2. **数据一致性**:
   - 训练和测试都排除SBET
   - 确保评估的一致性

3. **时间泄露防护**:
   - 排除逻辑不影响时间分割
   - Purge gap仍然有效
   - start_date/end_date仍然正确传递

---

**状态**: ✅ **评估进行中**

**下一步**: 等待评估完成，然后对比结果
