# LSTM多日预测高级增强版

## 概述

这是一个全面增强的LSTM多日预测量化分析模型，集成了您要求的所有高级功能：

### 🚀 核心增强功能

1. **超参数优化**
   - ✅ Optuna TPE采样器自动调参
   - ✅ Bayesian优化备用方案
   - ✅ 目标指标：Sharpe比率、最大回撤、信息比率
   - ✅ 自动正则化参数优化

2. **多模型融合与元学习**
   - ✅ 短期模型（1-2天）：专注短期波动
   - ✅ 中期模型（3-5天）：平衡短期和趋势
   - ✅ 长期模型（5天+）：捕获长期趋势
   - ✅ 动态权重基于最近表现调整

3. **特征工程与选取**
   - ✅ 信息系数（IC）检验和因子筛选
   - ✅ 因子中性化（市值/行业）
   - ✅ PCA/因子分析降维
   - ✅ VIF多重共线性检测

4. **在线学习**
   - ✅ 增量模型更新
   - ✅ 性能监控和自适应重训练
   - ✅ 模型漂移检测

## 🗂️ 文件结构

```
D:\trade\
├── lstm_multi_day_advanced.py      # 主模型文件
├── run_advanced_lstm.py            # 使用示例
├── install_advanced_dependencies.py # 依赖安装脚本
├── LSTM_ADVANCED_README.md         # 本文档
├── models/                         # 模型保存目录
├── logs/                          # 日志目录
└── result/                        # 结果输出目录
```

## 🔧 安装依赖

### 自动安装（推荐）

```bash
# 标准安装
python install_advanced_dependencies.py

# 中国用户（使用镜像源）
python install_advanced_dependencies.py --mirror

# 仅检查安装情况
python install_advanced_dependencies.py --check-only
```

### 手动安装

```bash
pip install tensorflow>=2.8.0
pip install optuna>=3.0.0
pip install scikit-optimize>=0.9.0
pip install statsmodels>=0.13.0
pip install factor_analyzer>=0.4.0
pip install yfinance>=0.2.0
```

## 🚦 快速开始

### 基础使用

```python
from lstm_multi_day_advanced import AdvancedLSTMMultiDayModel

# 创建模型（启用所有高级功能）
model = AdvancedLSTMMultiDayModel(
    prediction_days=5,
    enable_optimization=True,
    enable_ensemble=True,
    enable_online_learning=True
)

# 高级特征工程
processed_factors = model.prepare_advanced_features(
    factors_df, returns_df, market_cap_df, industry_df
)

# 创建序列数据
X, y = model.create_multi_day_sequences(processed_factors, returns_df)

# 训练模型
model.train_advanced_model(X_train, y_train, X_val, y_val)

# 预测
predictions = model.predict_advanced(X_test)
```

### 完整示例

```bash
# 运行完整示例（包含AAPL、MSFT、GOOGL）
python run_advanced_lstm.py
```

## 📊 核心功能详解

### 1. 超参数优化引擎

```python
# 自定义优化参数
optimizer = OptimizationEngine(
    optimization_metric='sharpe_ratio',  # 或 'max_drawdown', 'information_ratio'
    n_trials=100,
    study_name='my_lstm_optimization'
)

# 执行优化
best_params = optimizer.optimize_hyperparameters(
    X_train, y_train, X_val, y_val, returns_train, returns_val
)
```

**优化参数空间：**
- LSTM单元数：32-128
- Dropout率：0.1-0.5
- 学习率：1e-4 到 1e-2
- 正则化参数：L1/L2
- 批大小：16/32/64
- 优化器：Adam/AdamW/RMSprop

### 2. 多模型融合系统

```python
ensemble = MultiModelEnsemble(prediction_horizons=[1, 3, 5])

# 训练专门化模型
ensemble.train_ensemble(X_train, y_train, X_val, y_val)

# 动态权重更新
recent_performance = {'short_term': 0.8, 'medium_term': 0.6, 'long_term': 0.7}
ensemble.update_weights_by_performance(recent_performance)

# 集成预测
predictions = ensemble.predict_ensemble(X_test)
```

### 3. 高级特征工程

```python
feature_engineer = AdvancedFeatureEngineer(
    ic_threshold=0.02,           # IC阈值
    neutralize_market_cap=True,  # 市值中性化
    neutralize_industry=True,    # 行业中性化
    max_vif=10.0,               # VIF阈值
    pca_variance_threshold=0.95  # PCA方差保留率
)

# 完整特征工程流水线
processed_factors = feature_engineer.prepare_advanced_features(
    factors_df, returns_df, market_cap_df, industry_df
)
```

**处理步骤：**
1. **IC检验**：计算每个因子与收益率的信息系数
2. **因子筛选**：保留高IC因子
3. **中性化**：回归去除市值/行业影响
4. **共线性检测**：VIF检测去除多重共线性
5. **降维**：PCA或因子分析

### 4. 在线学习引擎

```python
online_learner = OnlineLearningEngine(
    update_frequency=5,        # 每5天更新一次
    memory_window=252,         # 保留1年数据
    performance_threshold=0.1   # 性能阈值
)

# 增量更新
online_learner.incremental_update(model, X_new, y_new)

# 自适应重训练
new_model = online_learner.adaptive_retraining(
    model_factory, X_history, y_history, X_new, y_new
)
```

## 📈 性能评估

模型提供多维度性能评估：

### 统计指标
- MSE/MAE：预测精度
- 方向准确率：预测方向正确率
- R²：决定系数

### 金融指标
- Sharpe比率：风险调整收益
- 最大回撤：最大损失
- 信息比率：超额收益风险比

### 每日预测精度
- 分别评估1-5天预测准确性
- 识别模型在不同时间跨度的表现

## 🎛️ 配置选项

### 模型配置

```python
model = AdvancedLSTMMultiDayModel(
    prediction_days=5,              # 预测天数
    lstm_window=20,                 # LSTM窗口长度
    enable_optimization=True,       # 启用超参数优化
    enable_ensemble=True,           # 启用模型集成
    enable_online_learning=True     # 启用在线学习
)
```

### 特征工程配置

```python
feature_engineer = AdvancedFeatureEngineer(
    ic_threshold=0.02,              # IC筛选阈值
    neutralize_market_cap=True,     # 是否市值中性化
    neutralize_industry=True,       # 是否行业中性化  
    max_vif=10.0,                  # VIF阈值
    pca_variance_threshold=0.95     # PCA方差保留率
)
```

### 优化配置

```python
optimizer = OptimizationEngine(
    optimization_metric='sharpe_ratio',  # 优化目标
    n_trials=100,                       # 试验次数
    study_name='lstm_optimization'      # 研究名称
)
```

## 🔍 高级使用案例

### 案例1：多股票轮动策略

```python
# 分析多只股票
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
models = {}

for ticker in tickers:
    # 为每只股票训练专门的模型
    model = AdvancedLSTMMultiDayModel(prediction_days=5)
    # ... 训练过程
    models[ticker] = model

# 基于预测结果进行资产配置
allocations = calculate_optimal_allocation(models, current_factors)
```

### 案例2：实时策略更新

```python
# 每日更新流程
def daily_update_process():
    # 1. 获取新数据
    new_data = fetch_latest_market_data()
    
    # 2. 特征提取
    new_factors = extract_factors(new_data)
    
    # 3. 模型预测
    predictions = model.predict_advanced(new_factors)
    
    # 4. 在线学习更新
    if should_update_model(recent_performance):
        model.online_update(new_factors, actual_returns)
    
    # 5. 生成交易信号
    signals = generate_trading_signals(predictions)
    
    return signals
```

### 案例3：风险管理集成

```python
# 风险调整的预测
def risk_adjusted_prediction(model, factors, risk_factors):
    # 基础预测
    base_predictions = model.predict_advanced(factors)
    
    # 风险调整
    risk_multipliers = calculate_risk_multipliers(risk_factors)
    adjusted_predictions = base_predictions * risk_multipliers
    
    # 约束检查
    adjusted_predictions = apply_risk_constraints(adjusted_predictions)
    
    return adjusted_predictions
```

## 🐛 故障排除

### 常见问题

1. **TensorFlow安装问题**
   ```bash
   # 如果GPU版本有问题，安装CPU版本
   pip uninstall tensorflow
   pip install tensorflow-cpu
   ```

2. **内存不足**
   ```python
   # 减少批大小和LSTM单元数
   model = AdvancedLSTMMultiDayModel(lstm_window=10)  # 减小窗口
   # 或在训练时使用更小的批大小
   ```

3. **Optuna优化失败**
   ```python
   # 禁用优化，使用默认参数
   model = AdvancedLSTMMultiDayModel(enable_optimization=False)
   ```

4. **数据质量问题**
   ```python
   # 检查数据质量
   print("缺失值比例:", factors_df.isnull().sum() / len(factors_df))
   print("数据范围:", factors_df.describe())
   ```

### 性能调优建议

1. **提高训练速度**
   - 减少超参数优化试验次数
   - 使用更小的LSTM窗口
   - 禁用不需要的功能

2. **提高预测精度**
   - 增加更多相关因子
   - 调整IC筛选阈值
   - 使用更长的训练历史

3. **减少内存使用**
   - 减少PCA组件数量
   - 使用滚动训练窗口
   - 定期清理模型缓存

## 📚 参考资料

- [Optuna官方文档](https://optuna.readthedocs.io/)
- [TensorFlow LSTM指南](https://www.tensorflow.org/guide/keras/rnn)
- [因子投资理论](https://en.wikipedia.org/wiki/Factor_investing)
- [信息系数(IC)计算方法](https://www.investopedia.com/terms/i/information-coefficient.asp)

## 🤝 贡献

如果您有改进建议或发现了bug，请：

1. 检查现有的issues
2. 创建详细的bug报告或功能请求
3. 提供可重现的代码示例

## 📄 许可证

本项目仅供学习和研究使用。请遵守相关金融法规和交易所规定。

---

**⚠️ 风险提示**：本模型仅供学术研究和教育目的。实际投资应谨慎评估风险，过往表现不代表未来结果。