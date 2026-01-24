# 全量训练已启动

## 📊 训练状态

**启动时间**: 2026-01-22 02:06:35

**训练配置**:
- 数据文件: `D:\trade\data\factor_exports\polygon_factors_all_filtered_clean_final_v2.parquet`
- Top N: 50
- Snapshot Tag: `FINAL_V2_FULL_DATASET_YYYYMMDD_HHMMSS`

**训练运行目录**: `results/full_dataset_training/run_20260122_020635/`

---

## ✅ 自动Snapshot更新

训练完成后，snapshot会自动：

1. **保存到训练运行目录**
   - `results/full_dataset_training/run_YYYYMMDD_HHMMSS/snapshot_id.txt`

2. **更新到项目根目录**
   - `latest_snapshot_id.txt` ← **Direct Predict默认使用这个**

3. **Snapshot包含的模型**:
   - ElasticNet (第一层)
   - XGBoost (第一层)
   - CatBoost (第一层)
   - LambdaRank (第一层)
   - MetaRankerStacker (第二层)

---

## 🔍 检查训练状态

### 方法1: 使用检查脚本
```bash
python scripts\check_training_status.py
```

### 方法2: 手动检查
```bash
# 检查latest_snapshot_id.txt
type latest_snapshot_id.txt

# 检查训练日志（如果有）
type results\full_dataset_training\training_log.txt
```

### 方法3: 检查训练运行目录
```bash
# 查看最新的训练运行目录
dir results\full_dataset_training\run_* /O-D

# 检查snapshot_id.txt
type results\full_dataset_training\run_20260122_020635\snapshot_id.txt
```

---

## ⏱️ 预计训练时间

- **预计时间**: 30-60分钟
- **实际时间**: 取决于数据量和系统性能

---

## ✅ 训练完成后的操作

训练完成后，**无需手动操作**：

1. ✅ Snapshot自动保存到`latest_snapshot_id.txt`
2. ✅ Direct Predict自动使用最新的snapshot
3. ✅ 所有模型（5个）都已训练并保存

**可以直接使用Direct Predict进行预测！**

---

## 📝 验证训练完成

训练完成后，运行以下命令验证：

```bash
python scripts\check_training_status.py
```

预期输出：
- ✅ Snapshot已更新
- ✅ Direct Predict配置正确
- ✅ 所有模型训练完成

---

## 🎯 Direct Predict使用

训练完成后，在GUI中点击"Direct Predict (Snapshot)"：

1. ✅ 自动从`final_v2.parquet`加载3,921只股票
2. ✅ 自动使用`latest_snapshot_id.txt`中的snapshot ID
3. ✅ 使用`FINAL_V2_FULL_DATASET` snapshot
4. ✅ 计算特征并进行预测

**无需任何手动配置！**

---

**生成时间**: 2026-01-22  
**状态**: 🚀 **训练已启动，等待完成**
