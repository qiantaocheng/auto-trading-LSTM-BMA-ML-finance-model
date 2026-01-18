# Snapshot存储位置分析

## 概述
Snapshot是训练好的模型的持久化存储，用于后续的预测和评估，无需重新训练。

## 存储位置结构

### 1. 模型文件存储（主要存储位置）

**默认路径结构：**
```
cache/
└── model_snapshots/
    └── YYYYMMDD/          # 按日期组织的目录（例如：20260118）
        └── {snapshot_id}/  # 每个snapshot的唯一UUID目录
            ├── elastic_net.pkl
            ├── xgboost.json
            ├── catboost.cbm
            ├── lambdarank.txt
            ├── lambdarank_scaler.pkl
            ├── lambdarank_meta.json
            ├── meta_ranker.txt          # MetaRankerStacker模型
            ├── meta_ranker_scaler.pkl
            ├── meta_ranker_meta.json
            ├── weights_elastic_net.json
            ├── weights_xgboost_gain.json
            ├── weights_xgboost_weight.json
            ├── weights_catboost.json
            └── manifest.json             # Snapshot元数据清单
```

**代码位置：** `bma_models/model_registry.py`
- `_default_snapshot_dir()`: 返回 `cache/model_snapshots/YYYYMMDD`
- `save_model_snapshot()`: 在 `{snapshot_dir}/{snapshot_id}/` 下保存所有模型文件

### 2. Snapshot ID记录

**在80/20时间分割评估脚本中：**
```
results/
└── extreme_filter_evaluation/
    └── run_YYYYMMDD_HHMMSS/
        └── snapshot_id.txt    # 保存snapshot_id字符串（UUID）
```

**代码位置：** `scripts/time_split_80_20_oos_eval.py:1137`
```python
(run_dir / "snapshot_id.txt").write_text(str(snapshot_id), encoding="utf-8")
```

### 3. SQLite数据库（元数据注册表）

**位置：** `data/model_registry.db`

**用途：**
- 存储snapshot的元数据（tag, 创建时间, 文件路径等）
- 提供snapshot查询和列表功能
- 自动创建（如果不存在）

**代码位置：** `bma_models/model_registry.py`
- 默认路径：`os.path.join("data", "model_registry.db")`
- 在`save_model_snapshot()`中写入元数据
- `load_manifest()`从数据库读取snapshot信息

### 4. Manifest.json（详细清单）

**位置：** `{snapshot_root_dir}/manifest.json`

**内容：**
- 所有模型文件的路径
- 特征名称列表
- 训练参数和元数据
- 模型版本信息

## Snapshot ID生成

**生成方式：** UUID v4（随机UUID）
```python
snapshot_id = str(uuid.uuid4())
```

**示例：** `"a8ab1816-59b6-4343-8865-45f06f5db85c"`

## 存储流程

### 训练时保存Snapshot

1. **训练阶段** (`scripts/time_split_80_20_oos_eval.py`)
   - 在训练窗口上训练模型
   - 调用 `save_model_snapshot()` 保存所有模型
   - Snapshot ID写入 `run_dir/snapshot_id.txt`

2. **保存过程** (`bma_models/model_registry.py`)
   - 生成UUID作为snapshot_id
   - 创建目录：`cache/model_snapshots/YYYYMMDD/{snapshot_id}/`
   - 保存所有模型文件（.pkl, .json, .cbm, .txt等）
   - 生成 `manifest.json`
   - 在SQLite数据库中注册元数据

### 加载Snapshot

1. **从snapshot_id.txt读取**
   ```python
   snapshot_id = (run_dir / "snapshot_id.txt").read_text().strip()
   ```

2. **从SQLite数据库查询**
   ```python
   manifest = load_manifest(snapshot_id)
   ```

3. **加载模型文件**
   ```python
   models = load_models_from_snapshot(snapshot_id)
   ```

## 关键代码位置

### 保存Snapshot
- **函数定义：** `bma_models/model_registry.py::save_model_snapshot()`
- **调用位置：**
  - `scripts/time_split_80_20_oos_eval.py:1125` - 保存MetaRankerStacker更新后的snapshot
  - `bma_models/量化模型_bma_ultra_enhanced.py:9350` - 训练后保存snapshot

### 加载Snapshot
- **函数定义：** `bma_models/model_registry.py::load_models_from_snapshot()`
- **调用位置：**
  - `scripts/time_split_80_20_oos_eval.py:1162` - 加载snapshot用于预测
  - `scripts/direct_predict.py` - 实时预测时加载snapshot

### Snapshot ID记录
- **保存位置：** `scripts/time_split_80_20_oos_eval.py:1137`
- **文件：** `{run_dir}/snapshot_id.txt`

## 文件大小估算

单个snapshot通常包含：
- ElasticNet: ~1-5 MB (.pkl)
- XGBoost: ~10-50 MB (.json)
- CatBoost: ~50-200 MB (.cbm)
- LambdaRank: ~5-20 MB (.txt + .pkl)
- MetaRankerStacker: ~5-20 MB (.txt + .pkl)
- 权重文件: ~1-5 MB (JSON)
- Manifest: ~100 KB (JSON)

**总计：** 约 100-300 MB 每个snapshot

## 清理建议

由于snapshot文件较大，建议：
1. 定期清理旧的snapshot（按日期目录）
2. 只保留重要的snapshot用于生产环境
3. 使用SQLite数据库查询snapshot信息，避免直接遍历文件系统

## 相关环境变量

目前没有环境变量控制snapshot存储位置，所有路径都是硬编码的：
- 默认目录：`cache/model_snapshots/YYYYMMDD`
- SQLite数据库：`data/model_registry.db`

## 总结

Snapshot存储采用三层结构：
1. **文件系统**：`cache/model_snapshots/YYYYMMDD/{snapshot_id}/` - 存储实际模型文件
2. **SQLite数据库**：`data/model_registry.db` - 存储元数据和索引
3. **结果目录**：`results/.../snapshot_id.txt` - 记录评估使用的snapshot ID

这种设计允许：
- 快速查询snapshot信息（通过SQLite）
- 按日期组织snapshot（便于管理）
- 通过UUID唯一标识每个snapshot
- 在评估结果中追踪使用的snapshot
