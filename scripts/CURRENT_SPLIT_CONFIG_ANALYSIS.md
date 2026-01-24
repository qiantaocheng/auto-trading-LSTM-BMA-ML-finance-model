# 当前Split配置分析报告

## 📊 配置检查结果

### ✅ 当前配置已更新为80/20

**检查时间**: 2026-01-22

---

## 🔍 配置详情

### 1. Split比例

**参数**: `--split`  
**默认值**: `0.8` ✅  
**含义**: 80%训练集，20%测试集  
**代码位置**: `time_split_80_20_oos_eval.py` line 346

**修改前**: `default=0.9` (90/10)  
**修改后**: `default=0.8` (80/20) ✅

---

### 2. 输出目录

**参数**: `--output-dir`  
**默认值**: `results/t10_time_split_80_20_final` ✅  
**代码位置**: `time_split_80_20_oos_eval.py` line 359

**修改前**: `default="results/t10_time_split_90_10"`  
**修改后**: `default="results/t10_time_split_80_20_final"` ✅

---

### 3. 脚本名称

**文件名**: `time_split_80_20_oos_eval.py` ✅  
**状态**: 脚本名称已包含`80_20`，与配置一致

---

## 📋 完整默认配置

```python
# 时间分割参数
--split: 0.8 (80/20) ✅
--horizon-days: 10

# 输出参数
--output-dir: "results/t10_time_split_80_20_final" ✅

# 模型参数
--models: ["catboost", "lambdarank", "ridge_stacking"]
--top-n: 20

# HAC参数
--hac-method: "newey-west"
--hac-lag: None (自动计算为 max(10, 2*horizon_days))

# 其他参数
--cost-bps: 0.0
--benchmark: "QQQ"
--ema-top-n: -1 (禁用EMA)
--log-level: "INFO"
```

---

## ✅ 验证结果

### 配置一致性检查

| 配置项 | 值 | 状态 |
|--------|-----|------|
| Split比例 | 0.8 (80/20) | ✅ 正确 |
| 输出目录 | results/t10_time_split_80_20_final | ✅ 正确 |
| 脚本名称 | time_split_80_20_oos_eval.py | ✅ 一致 |

**结论**: ✅ **所有配置都已正确设置为80/20**

---

## 🎯 使用方式

### 基本用法（使用默认80/20配置）

```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --models catboost lambdarank ridge_stacking \
  --top-n 20
```

**注意**: 不需要指定`--split`和`--output-dir`，将自动使用80/20配置

### 覆盖默认配置（如果需要90/10）

```bash
python scripts/time_split_80_20_oos_eval.py \
  --data-file "data/factor_exports/polygon_factors_all_filtered_clean.parquet" \
  --split 0.9 \
  --output-dir "results/t10_time_split_90_10" \
  --models catboost lambdarank ridge_stacking
```

---

## 📝 修改记录

### 修改内容

1. **`--split`默认值**: `0.9` → `0.8`
   - **文件**: `scripts/time_split_80_20_oos_eval.py`
   - **行号**: 346
   - **修改**: `default=0.9` → `default=0.8`
   - **帮助文本**: 更新为 `"Train split fraction by time (default 0.8 for 80/20)."`

2. **`--output-dir`默认值**: `"results/t10_time_split_90_10"` → `"results/t10_time_split_80_20_final"`
   - **文件**: `scripts/time_split_80_20_oos_eval.py`
   - **行号**: 359
   - **修改**: 更新输出目录名称

---

## 🔍 验证方法

### 使用检查脚本

```bash
python scripts/check_current_split_config.py
```

### 预期输出

```
[OK] 已设置为80/20 (0.8)
[OK] 输出目录包含80_20
[OK] 脚本名称包含80_20
[OK] 当前配置是80/20
```

---

## ⚠️ 注意事项

### 1. Purge Gap

即使使用80/20分割，Purge Gap仍然有效：
- 训练集结束日期 = `split_idx - 1 - horizon`
- 测试集开始日期 = `split_idx`
- 实际间隔 = `horizon_days`（默认10天）

### 2. 向后兼容

如果需要使用90/10配置，可以通过命令行参数覆盖：
```bash
--split 0.9 --output-dir "results/t10_time_split_90_10"
```

### 3. 现有结果目录

- 旧配置（90/10）的结果在: `results/t10_time_split_90_10/`
- 新配置（80/20）的结果在: `results/t10_time_split_80_20_final/`

---

## ✅ 总结

**当前配置状态**: ✅ **已设置为默认80/20**

- ✅ Split比例: 0.8 (80/20)
- ✅ 输出目录: results/t10_time_split_80_20_final
- ✅ 脚本名称: time_split_80_20_oos_eval.py（一致）

**所有配置都已正确对齐为80/20**，可以直接使用默认参数运行。

---

**生成时间**: 2026-01-22  
**状态**: ✅ **配置已更新为80/20**
