# 80/20评估安全运行指南

## ✅ 安全保证：不会覆盖latest_snapshot_id.txt

**重要**: 80/20评估脚本**不会**覆盖`latest_snapshot_id.txt`，可以安全运行。

---

## 🔍 验证结果

### 1. Snapshot保存位置

**80/20评估脚本**:
- ✅ Snapshot保存到: `results/t10_time_split_80_20_final/run_*/snapshot_id.txt`
- ✅ **不会**保存到: `latest_snapshot_id.txt`

**代码验证**: `time_split_80_20_oos_eval.py` line 1640
```python
(run_dir / "snapshot_id.txt").write_text(str(snapshot_id), encoding="utf-8")
```

**结论**: ✅ 只保存到运行目录，不会覆盖`latest_snapshot_id.txt`

---

### 2. train_from_document自动保存

**train_from_document行为**:
- ✅ 自动保存snapshot到数据库（正常行为）
- ✅ **不会**自动更新`latest_snapshot_id.txt`（只有`train_full_dataset.py`会更新）

**代码验证**: `量化模型_bma_ultra_enhanced.py` line 9362-9401
- 只保存snapshot到数据库
- 设置`self.active_snapshot_id`
- **不会**写入`latest_snapshot_id.txt`

**结论**: ✅ `train_from_document`不会覆盖`latest_snapshot_id.txt`

---

## ✅ 时间泄露防护验证

### 1. 时间分割逻辑 ✅

| 检查项 | 状态 | 代码位置 |
|--------|------|----------|
| split_idx计算 | ✅ | Line 1494: `split_idx = int(n_dates * split)` |
| train_end_idx计算（purge gap） | ✅ | Line 1496: `train_end_idx = max(0, split_idx - 1 - horizon)` |
| train_start设置 | ✅ | Line 1500: `train_start = dates[0]` |
| train_end设置 | ✅ | Line 1501: `train_end = dates[train_end_idx]` |

### 2. train_from_document参数传递 ✅

**代码位置**: Line 1547-1552
```python
train_res = model.train_from_document(
    training_data_path=str(Path(training_data_path)),
    top_n=50,
    start_date=str(train_start.date()),  # ✅ 使用train_start
    end_date=str(train_end.date()),      # ✅ 使用train_end
)
```

### 3. train_from_document数据过滤 ✅

**代码位置**: `量化模型_bma_ultra_enhanced.py` line 8372-8384
```python
if (start_date or end_date) and isinstance(feature_data.index, pd.MultiIndex):
    d = pd.to_datetime(feature_data.index.get_level_values('date')).tz_localize(None)
    mask = pd.Series(True, index=feature_data.index)
    if start_date:
        sd = pd.to_datetime(start_date).tz_localize(None)
        mask &= (d >= sd)
    if end_date:
        ed = pd.to_datetime(end_date).tz_localize(None)
        mask &= (d <= ed)
    feature_data = feature_data.loc[mask.values].copy()
```

**验证**: ✅ 数据过滤正确，无时间泄露

---

## 🎯 安全运行方式

### 方法1: 直接运行（推荐）

```bash
python scripts\time_split_80_20_oos_eval.py \
  --data-file "D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet" \
  --split 0.8 \
  --models catboost lambdarank ridge_stacking \
  --top-n 20
```

**安全保证**: ✅ 不会覆盖`latest_snapshot_id.txt`

### 方法2: 使用安全脚本（额外保护）

```bash
scripts\run_80_20_eval_safe_no_overwrite.bat
```

**额外保护**:
- 自动备份`latest_snapshot_id.txt`
- 运行后验证是否被修改
- 如果被修改，自动恢复备份

---

## 📋 训练数据范围

**80/20分割示例**（假设1244个交易日，split=0.8，horizon=10）:

- **总日期数**: 1244
- **split_idx**: 995（80%分割点）
- **train_end_idx**: 984（995 - 1 - 10，包含10天purge gap）
- **训练集**: dates[0] 至 dates[984]（985个交易日，约79%）
- **Purge Gap**: dates[985] 至 dates[994]（10天）
- **测试集**: dates[995] 至 dates[1243]（249个交易日，约20%）

**验证**: ✅ 训练数据范围正确，无时间泄露

---

## ✅ 使用的因子

**15个Alpha因子**（来自`t10_selected`）:
1. momentum_10d
2. ivol_30
3. near_52w_high
4. rsi_21
5. vol_ratio_30d
6. trend_r2_60
7. liquid_momentum
8. obv_momentum_40d
9. atr_ratio
10. ret_skew_30d
11. price_ma60_deviation
12. blowoff_ratio_30d
13. feat_vol_price_div_30d
14. 5_days_reversal
15. downside_beta_ewm_21

**验证**: ✅ 使用现有因子，无时间泄露

---

## 🎯 最终结论

**✅ 可以安全运行80/20评估**

1. ✅ **不会覆盖latest_snapshot_id.txt**
   - Snapshot只保存到运行目录
   - `train_from_document`不会更新`latest_snapshot_id.txt`

2. ✅ **无时间泄露**
   - 正确传递`start_date`和`end_date`
   - 包含purge gap（10天）
   - `train_from_document`正确过滤数据

3. ✅ **使用现有因子**
   - 15个Alpha因子（`t10_selected`）
   - 因子计算正确，无未来信息

**可以安全运行，不会影响正在进行的全量训练！**

---

**生成时间**: 2026-01-22  
**状态**: ✅ **安全，可以运行**
