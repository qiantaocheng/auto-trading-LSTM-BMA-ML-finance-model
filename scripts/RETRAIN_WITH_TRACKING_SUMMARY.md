# 重新训练和80/20评估 - 详细跟踪模式

## ✅ 已完成的操作

### 1. 终止旧进程
- ✅ 已终止所有旧的Python进程

### 2. 添加详细跟踪日志

已在以下关键位置添加跟踪日志：

#### 模型级别
- 模型训练开始
- 总CV折数
- 训练数据规模

#### Fold级别
- Fold开始处理
- 训练/验证样本数
- 训练窗天数计算
- 训练窗检查结果
- 数据分割完成

#### 模型训练级别
- ElasticNet训练开始/完成
- XGBoost训练开始/完成
- CatBoost训练开始/完成（包括分类特征数）
- LambdaRank训练开始/完成

### 3. 启动训练

**进程状态**:
- 进程1: ID 13120, 内存: 13.44 MB
- 进程2: ID 32668, 内存: 1608.88 MB (训练进程)

**启动时间**: 2026-01-22 22:42:36

## 🔍 如何跟踪阻塞位置

### 方法1: 查看实时日志

训练脚本会输出详细的跟踪日志，查找以下标记：

- `🔍` - 跟踪日志标记
- `Fold X/Y 开始处理` - Fold处理开始
- `训练窗天数` - 训练窗检查
- `数据分割完成` - 数据准备完成
- `开始{name}训练` - 模型训练开始
- `{name}训练完成` - 模型训练完成

### 方法2: 检查进程状态

```powershell
Get-Process python | Select-Object Id, StartTime, @{Name="Runtime";Expression={(Get-Date) - $_.StartTime}}, @{Name="Memory(MB)";Expression={[math]::Round($_.WorkingSet64/1MB, 2)}}
```

**判断标准**:
- 内存增长: 说明在训练中
- 内存稳定: 可能阻塞
- 进程消失: 训练完成或出错

### 方法3: 检查训练目录

```powershell
ls results/full_dataset_training/run_*/
```

**判断标准**:
- 空目录: 训练进行中
- 有`snapshot_id.txt`: 训练完成

## 📋 预期日志输出示例

```
[FIRST_LAYER] 🔍 开始训练模型: elastic_net
[FIRST_LAYER] 🔍  总CV折数: 3
[FIRST_LAYER] 🔍  训练数据: 827900样本 × 28特征
[FIRST_LAYER][elastic_net] 🔍 Fold 1/3 开始处理
[FIRST_LAYER][elastic_net] 🔍  训练样本数: 551933, 验证样本数: 137983
[FIRST_LAYER][elastic_net] 🔍  从groups_norm计算训练窗: 829天 (唯一日期数)
[FIRST_LAYER][elastic_net] 🔍  训练窗天数: 829, 最小要求: 126
[FIRST_LAYER][elastic_net] 🔍 Fold 1: 开始创建训练/验证数据分割
[FIRST_LAYER][elastic_net] 🔍 Fold 1: 数据分割完成 - X_train: (551933, 28), X_val: (137983, 28)
[FIRST_LAYER][elastic_net] 🔍 Fold 1: 开始elastic_net训练 (样本数: 551933)
[FIRST_LAYER][elastic_net] 🔍 Fold 1: elastic_net训练完成
...
```

## 🎯 定位阻塞位置的步骤

1. **等待5-10分钟**: 让训练开始并产生日志

2. **检查最后一个日志**:
   - 如果最后是 "开始{name}训练" → 阻塞在模型训练中
   - 如果最后是 "数据分割完成" → 阻塞在特征选择或模型初始化
   - 如果最后是 "训练窗天数" → 阻塞在训练窗检查

3. **检查特定模型**:
   - CatBoost最可能阻塞（od_wait=120）
   - LambdaRank可能阻塞（训练时间长）
   - XGBoost较少阻塞
   - ElasticNet很少阻塞

4. **检查特定fold**:
   - Fold 1: 可能是数据准备问题
   - Fold 2-3: 可能是训练数据问题
   - 所有fold: 可能是模型配置问题

## 📊 当前状态

- **训练状态**: 🔄 进行中
- **进程内存**: 1608.88 MB (正常，说明在训练中)
- **预计完成时间**: 1.5-2小时

## 🔧 如果发现阻塞

根据最后一个日志的位置：

1. **阻塞在CatBoost训练**:
   - 检查`od_wait`参数
   - 减少`iterations`
   - 检查early stopping

2. **阻塞在LambdaRank训练**:
   - 检查`num_boost_round`
   - 减少`early_stopping_rounds`
   - 检查LightGBM配置

3. **阻塞在数据准备**:
   - 检查数据大小
   - 检查内存使用
   - 检查特征数量

## 📝 下一步

1. **等待并观察日志输出**
2. **记录最后一个成功的日志**
3. **根据日志判断阻塞位置**
4. **针对性地解决问题**
