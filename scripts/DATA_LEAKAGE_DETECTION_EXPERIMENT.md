# 80/20 OOF 数据泄露检测实验设计

## 实验目标

设计一系列实验来检测80/20 OOF（Out-of-Fold）预测是否存在数据泄露（data leakage）。

## 数据泄露的可能来源

1. **特征工程泄露**：使用未来信息计算特征
2. **特征标准化泄露**：使用测试集统计量标准化训练集特征
3. **CV分割泄露**：CV分割时没有正确处理时间顺序，未来数据出现在训练集中
4. **目标变量泄露**：目标变量计算时使用了未来信息
5. **OOF预测泄露**：OOF预测时使用了测试集信息

## 实验设计

### 实验1：随机打乱目标变量检测

**原理**：如果OOF预测存在数据泄露，即使目标变量被随机打乱，OOF预测仍可能显示高相关性（不应该）。

**方法**：
1. 使用真实目标变量训练模型，计算OOF预测
2. 随机打乱目标变量（保持索引），重新训练模型，计算OOF预测
3. 比较两种情况下OOF预测与目标变量的相关性

**预期结果**：
- ✅ **正常**：随机目标变量的OOF预测相关性应该接近0
- ⚠️ **泄露**：随机目标变量的OOF预测相关性 > 0.1

**运行方法**：
```bash
python scripts/detect_oof_data_leakage_experiment.py \
    --data-file data/factor_exports/factors/factors_all.parquet \
    --split 0.8 \
    --horizon-days 10
```

### 实验2：特征标准化泄露检测

**原理**：特征标准化应该只使用训练集统计量，不应该使用测试集统计量。

**方法**：
1. 方法A（正确）：只使用训练集统计量标准化
2. 方法B（泄露）：使用训练+测试集统计量标准化
3. 比较两种方法在训练集上的特征统计量差异

**预期结果**：
- ✅ **正常**：两种方法的特征统计量差异应该很小（< 0.01）
- ⚠️ **泄露**：两种方法的特征统计量差异 > 0.01

**检测指标**：
- 训练集标准化均值的差异
- 训练集标准化标准差的差异

### 实验3：CV时间顺序检测

**原理**：CV分割应该按时间顺序，训练集不应该包含未来数据。

**方法**：
1. 检查每个CV fold的训练集和验证集的时间顺序
2. 验证：验证集的最小日期应该 >= 训练集的最大日期 + gap（horizon_days）
3. 检查：训练集中不应该有验证集日期之后的数据

**预期结果**：
- ✅ **正常**：所有fold的gap >= horizon_days，时间顺序正确
- ⚠️ **泄露**：存在fold的gap < horizon_days，或时间顺序错误

**检测指标**：
- 每个fold的gap天数
- 是否存在时间顺序违规

### 实验4：目标变量未来信息检测

**原理**：目标变量应该只使用T+horizon_days的信息，不应该使用更未来的信息。

**方法**：
1. 检查目标变量的分布是否合理
2. 检查是否有异常值（可能使用了未来信息）
3. 验证目标变量计算是否正确

**预期结果**：
- ✅ **正常**：目标变量分布合理，异常值 < 10%
- ⚠️ **泄露**：目标变量异常值 > 10%，或分布不合理

**检测指标**：
- 目标变量的99分位数和1分位数
- 异常值比例

### 实验5：OOF vs 测试集性能比较

**原理**：如果OOF预测存在泄露，OOF预测性能应该异常高，且远高于测试集预测性能。

**方法**：
1. 运行80/20评估，获取OOF预测和测试集预测
2. 计算OOF预测的IC和Rank IC
3. 计算测试集预测的IC和Rank IC
4. 比较两者的差异

**预期结果**：
- ✅ **正常**：OOF IC ≈ 测试集 IC（差异 < 0.1）
- ⚠️ **泄露**：OOF IC >> 测试集 IC（差异 > 0.1）

**检测指标**：
- OOF IC vs 测试集 IC
- OOF Rank IC vs 测试集 Rank IC
- OOF预测与测试集预测的相关性（不应该太高）

**运行方法**：
```bash
# 首先运行80/20评估，保存结果
python scripts/time_split_80_20_oos_eval.py \
    --data-file data/factor_exports/factors/factors_all.parquet \
    --split 0.8 \
    --horizon-days 10

# 然后比较OOF和测试集性能
python scripts/compare_oof_vs_test_performance.py \
    --snapshot-id <snapshot_id> \
    --test-predictions-file results/t10_time_split_80_20_final/run_*/report_df.csv \
    --test-actuals-file results/t10_time_split_80_20_final/run_*/test_actuals.csv
```

## 实验执行步骤

### 步骤1：运行基础检测实验

```bash
python scripts/detect_oof_data_leakage_experiment.py \
    --data-file data/factor_exports/factors/factors_all.parquet \
    --split 0.8 \
    --horizon-days 10
```

这将运行实验1-4，并生成检测报告。

### 步骤2：运行80/20评估（如果需要实验5）

```bash
python scripts/time_split_80_20_oos_eval.py \
    --data-file data/factor_exports/factors/factors_all.parquet \
    --split 0.8 \
    --horizon-days 10 \
    --output-dir results/data_leakage_test
```

### 步骤3：比较OOF和测试集性能（实验5）

```bash
python scripts/compare_oof_vs_test_performance.py \
    --snapshot-id <从步骤2获取的snapshot_id> \
    --test-predictions-file results/data_leakage_test/run_*/report_df.csv
```

## 结果解读

### 正常情况（无泄露）

- 实验1：随机目标变量相关性 ≈ 0
- 实验2：特征标准化差异 < 0.01
- 实验3：所有fold的gap >= horizon_days，时间顺序正确
- 实验4：目标变量异常值 < 10%
- 实验5：OOF IC ≈ 测试集 IC

### 泄露情况（有泄露）

- 实验1：随机目标变量相关性 > 0.1 ⚠️
- 实验2：特征标准化差异 > 0.01 ⚠️
- 实验3：存在fold的gap < horizon_days ⚠️
- 实验4：目标变量异常值 > 10% ⚠️
- 实验5：OOF IC >> 测试集 IC ⚠️

## 修复建议

如果检测到数据泄露，建议检查以下方面：

1. **特征工程**：
   - 确保所有特征只使用历史信息
   - 检查特征计算是否有未来信息泄露

2. **特征标准化**：
   - 确保只使用训练集统计量
   - 测试集特征使用训练集的均值和标准差标准化

3. **CV分割**：
   - 确保使用PurgedCV或TimeSeriesSplit
   - 确保gap >= horizon_days
   - 确保时间顺序正确

4. **目标变量**：
   - 确保目标变量只使用T+horizon_days的信息
   - 检查目标变量计算逻辑

5. **OOF预测**：
   - 确保OOF预测时只使用训练集信息
   - 检查是否有测试集信息泄露到OOF预测中

## 注意事项

1. **实验1**需要实际的模型训练，可能需要较长时间
2. **实验5**需要实际的OOF预测数据，需要先运行80/20评估
3. 某些实验可能需要根据实际的数据结构进行调整
4. 建议在子集数据上先运行实验，验证方法正确性后再在全量数据上运行

## 输出文件

- `data_leakage_detection_YYYYMMDD_HHMMSS.log`：实验日志
- `data_leakage_report_YYYYMMDD_HHMMSS.json`：检测结果JSON报告

## 参考

- [Purged Cross-Validation](https://www.researchgate.net/publication/301275844_Advances_in_Financial_Machine_Learning)
- [Data Leakage in Machine Learning](https://machinelearningmastery.com/data-leakage-machine-learning/)
