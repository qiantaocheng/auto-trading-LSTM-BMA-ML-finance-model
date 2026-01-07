# Grid Search & Complete Backtest Guide（先基模型，后 Ridge；按现有代码）

两阶段策略：
- 第1阶段：独立网格搜索四个基模型（ElasticNet / XGBoost / CatBoost / LambdaRank），每个组合 = 训练 + 回测，评分仅取目标模型行的 Top20% 平均收益。
- 第2阶段：基模型最优参数与快照确定后，再单独运行 Ridge Stacking（二层融合）。命令行若同时包含 Ridge 与其它模型，会自动跳过 Ridge 并给出提示；仅当 `--models ridge` 单独运行时才执行 Ridge 网格搜索。

所有评分均来自目标模型行的 `avg_top_return`，不再跨模型求平均，避免分数全部一样。

## 流程示意
```
full_grid_search.py  (主控)
  ├─ train_single_model.py  每个超参组合训练，生成 snapshot_id（使用临时配置覆盖目标模型参数）
  └─ comprehensive_model_backtest.py  指定 snapshot 回测，只取目标模型行的 avg_top_return
```

## 超参网格（与代码保持一致）
- ElasticNet (49): alpha [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]; l1_ratio [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
- XGBoost (625): n_estimators [100,200,300,400,500]; max_depth [3,4,5,6,7]; learning_rate [0.01,0.03,0.05,0.07,0.1]; min_child_weight [1,3,5,7,10]
- CatBoost (625): iterations [1000,2000,3000,4000,5000]; depth [4,5,6,7,8]; learning_rate [0.01,0.02,0.03,0.05,0.07]; subsample [0.6,0.7,0.8,0.9,1.0]
- LambdaRank (3125): num_boost_round [50,100,200,300,500]; learning_rate [0.01,0.03,0.05,0.07,0.1]; num_leaves [127,191,255,319,383]; max_depth [6,7,8,9,10]; lambda_l2 [1.0,5.0,10.0,20.0,50.0]
- Ridge Stacking (7，后置): alpha [0.1,0.5,1.0,5.0,10.0,50.0,100.0]

## 执行步骤

### 1) 数据检查
```bash
ls data/factor_exports/factors/factors_all.parquet
python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/factor_exports/factors/factors_all.parquet")
print(df.index)  # 需为 (date, ticker) MultiIndex 或含 date/ticker 列
PY
```

### 2) 基模型网格搜索（默认仅跑四个基模型）
```bash
python scripts/full_grid_search.py \
    --data-file data/factor_exports/factors/factors_all.parquet \
  --data-dir  data/factor_exports/factors \
  --output-dir results/grid_search_20251205 \
  --models elastic_net xgboost catboost lambdarank
```
产物：`{model}_grid_search_intermediate.csv` / `{model}_grid_search_final.csv`，分数随超参变化（按目标模型行计算）。

### 3) Ridge Stacking 单独跑（在基模型最优确定后）
仅当命令只含 `ridge` 时才会执行；混跑会跳过。
```bash
python scripts/full_grid_search.py \
    --data-file data/factor_exports/factors/factors_all.parquet \
  --data-dir  data/factor_exports/factors \
  --output-dir results/grid_search_20251205_ridge \
  --models ridge
```

### 4) 生成网格搜索报告（可选）
```bash
python scripts/grid_search_report.py \
  --input-dir results/grid_search_20251205 \
    --output-file results/grid_search_report.xlsx \
    --top-n 10
```

### 5) 完整回测 / 复核（complete backtest）
使用指定或最新 `snapshot_id` 跑全量回测，验证最优组合：
```bash
python scripts/comprehensive_model_backtest.py \
    --data-dir data/factor_exports/factors \
  --snapshot-id <best_snapshot_id_optional>
```
不传 `--snapshot-id` 则加载最新快照。报告中每个模型行含 `avg_top_return` 等指标。

## 关键实现对齐点
- `full_grid_search.py`：每个组合 = 训练 + 指定 snapshot 回测；评分函数 `_select_model_top_return` 只读取目标模型的 `avg_top_return`。当模型列表同时含 ridge 与其它模型时会跳过 ridge。
- `train_single_model.py`：为单模型生成临时配置（`BMA_TEMP_CONFIG_PATH`），然后训练并保存 snapshot。
- `comprehensive_model_backtest.py`：支持 `snapshot_id`，报告按模型拆分，供网格搜索取分；回测即“complete backtest”。

## 常见问题
- 分数都一样？现已按目标模型行取值，不会跨模型平均。
- 想直接跑 Ridge？先完成基模型最优，再独立命令只含 `ridge`。
- 数据索引错误？需保证 `(date, ticker)` MultiIndex 或含 `date/ticker` 列，并去重排序。 

