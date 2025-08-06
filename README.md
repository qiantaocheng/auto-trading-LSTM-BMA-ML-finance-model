# 🚀 量化交易管理软件

一个功能强大的量化交易分析平台，集成了先进的机器学习算法、因子分析和回测系统，为投资决策提供科学的数据支持。

## 📋 目录

- [功能特性](#功能特性)
- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [安装指南](#安装指南)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [技术架构](#技术架构)
- [故障排除](#故障排除)
- [更新日志](#更新日志)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## ✨ 功能特性

### 🎯 核心功能
- **智能量化分析**: 基于机器学习的股票筛选和评分系统
- **多因子模型**: 集成技术指标、基本面数据和市场情绪因子
- **回测系统**: 历史数据验证和策略评估
- **实时监控**: 市场数据实时更新和预警系统
- **结果管理**: 自动保存分析结果和生成报告

### 🤖 机器学习算法
- **集成学习**: XGBoost、LightGBM、CatBoost多模型融合
- **因子选择**: 信息系数(IC)筛选和多重共线性检测
- **异常值处理**: Winsorizing技术处理极端值
- **因子中性化**: 消除市值和行业偏差
- **超参数优化**: 自动调优模型参数

### 🖥️ 用户界面
- **现代化GUI**: 基于Tkinter的友好界面
- **动画效果**: 按钮悬停和加载动画
- **背景图片**: 美观的界面背景
- **状态显示**: 实时进度条和状态信息
- **系统通知**: 任务完成提醒

### 📊 数据分析
- **多股票分析**: 支持自定义股票池
- **时间范围**: 灵活的时间设置
- **结果导出**: Excel格式输出
- **图表生成**: 自动生成分析图表
- **历史记录**: 数据库存储分析历史

## 💻 系统要求

### 最低配置
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8 或更高版本
- **内存**: 至少 4GB RAM
- **存储**: 至少 2GB 可用空间
- **网络**: 稳定的互联网连接

### 推荐配置
- **操作系统**: Windows 11 或 Ubuntu 20.04+
- **Python**: 3.10 或更高版本
- **内存**: 8GB RAM 或更多
- **存储**: 5GB 可用空间
- **网络**: 高速互联网连接

## 🚀 快速开始

### 方法一：一键启动（推荐）
```bash
# Windows用户
双击 "启动量化交易软件_修复版.bat"

# Linux/Mac用户
python3 quantitative_trading_manager.py
```

### 方法二：命令行启动
```bash
# 激活虚拟环境
source trading_env/bin/activate  # Linux/Mac
trading_env\Scripts\activate.bat  # Windows

# 启动软件
python quantitative_trading_manager.py
```

## 📦 安装指南

### 1. 环境准备

#### 必需软件
```bash
# 检查Python版本
python --version

# 创建虚拟环境（可选）
python -m venv trading_env
```

#### 推荐安装的软件
- **SQLite Browser**: 数据库查看工具（用于查看trading_results.db）
  ```bash
  # Windows: 下载 DB Browser for SQLite
  # https://sqlitebrowser.org/
  
  # Linux
  sudo apt-get install sqlitebrowser
  
  # macOS
  brew install db-browser-for-sqlite
  ```

### 2. 数据库和数据集

#### 金融数据库
- **yfinance**: 雅虎财经数据（已包含，用于获取美股数据）

#### 本地数据库
- **SQLite**: 轻量级数据库（已包含，存储分析历史记录）

### 3. 依赖安装
```bash
# 方法一：使用一键安装脚本
python "首先一键安装所有依赖.py"

# 方法二：手动安装
pip install -r requirements_portable.txt

# 方法三：逐个安装核心依赖
pip install pandas numpy yfinance scikit-learn matplotlib seaborn
pip install openpyxl Pillow plyer APScheduler tkcalendar pywin32

# 方法四：安装额外依赖（可选）
pip install sqlalchemy
```

### 4. 验证安装
```bash
# 检查关键依赖
python -c "import pandas, numpy, yfinance, sklearn, matplotlib, seaborn, openpyxl, PIL, plyer, apscheduler, tkcalendar; print('所有依赖安装成功！')"

# 检查yfinance连接
python -c "import yfinance as yf; print('yfinance安装成功！')"
```

## 📖 使用方法

### 1. 启动软件
- 双击 `启动量化交易软件_修复版.bat` (Windows)
- 或运行 `python quantitative_trading_manager.py`

### 2. 主界面功能
- **量化分析模型**: 运行增强版量化分析
- **回测分析**: 执行历史回测
- **ML回测**: 机器学习模型回测
- **设置**: 配置软件参数
- **历史**: 查看分析历史
- **帮助**: 获取使用帮助

### 3. 分析流程
1. **选择时间范围**: 设置分析的时间段
2. **配置股票池**: 选择要分析的股票
3. **运行分析**: 点击相应的分析按钮
4. **查看结果**: 在result目录查看输出文件

### 4. 结果解读
- **Excel文件**: 包含详细的股票评分和推荐
- **CSV文件**: 缺失数据报告
- **图表**: 自动生成的分析图表
- **数据库**: 历史记录存储

## 📁 项目结构

```
量化交易软件/
├── 核心程序/
│   ├── quantitative_trading_manager.py    # 主程序 (97KB)
│   ├── 量化模型_enhanced.py              # 增强版模型 (41KB)
│   ├── ml_rolling_backtest_clean.py      # ML回测 (47KB)
│   └── comprehensive_category_backtest.py # 分类回测 (43KB)
│
├── 启动脚本/
│   ├── 启动量化交易软件_修复版.bat      # 修复版启动
│   ├── launch.py                         # 启动器 (12KB)
│   ├── launch_portable.py                # 便携启动器
│   └── start_trading_software.py         # 启动包 (13KB)
│
├── 安装和配置/
│   ├── 首先一键安装所有依赖.py          # 一键安装脚本
│   ├── install_enhanced_requirements.py   # 增强依赖安装
│   ├── install_requirements.py            # 基础依赖安装
│   ├── requirements_portable.txt          # 依赖列表
│   └── hyperopt_config.py                # 超参数配置
│
├── 工具脚本/
│   ├── check_database.py                 # 数据库检查
│   ├── view_database.py                  # 数据库查看器
│   ├── setup_scheduler.py                # 定时任务设置
│   ├── test_system.py                    # 系统测试
│   └── cleanup_files.py                  # 文件清理工具
│
├── 数据和分析/
│   ├── result/                           # 分析结果目录
│   ├── category_analysis_results/         # 分类分析结果
│   ├── logs/                             # 日志文件
│   └── trading_results.db                # SQLite数据库
│
├── 资源文件/
│   ├── quagsire.png                      # 界面图标
│   ├── ChatGPT Image 2025年8月1日 03_26_16.png  # 背景图片
│   └── README_PORTABLE.md                # 便携版说明
│
└── 虚拟环境/
    └── trading_env/                      # Python虚拟环境
```

## 🏗️ 技术架构

### 核心技术栈
- **Python 3.12**: 主要编程语言
- **Tkinter**: GUI界面框架
- **SQLite**: 本地数据库
- **Pandas/NumPy**: 数据处理
- **Scikit-learn**: 机器学习
- **Matplotlib/Seaborn**: 数据可视化
- **yfinance**: 金融数据获取

### 数据库支持
- **SQLite**: 轻量级本地数据库（默认，存储分析历史）

### 数据源集成
- **yfinance**: 雅虎财经数据（美股数据获取）

### 机器学习模型
- **XGBoost**: 梯度提升树
- **LightGBM**: 轻量级梯度提升
- **CatBoost**: 分类提升
- **Random Forest**: 随机森林
- **Ridge Regression**: 岭回归

### 因子分析技术
- **信息系数(IC)**: 因子有效性评估
- **Winsorizing**: 异常值处理
- **因子中性化**: 消除偏差
- **多重共线性检测**: 因子去重
- **PCA降维**: 维度压缩

### 系统架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GUI界面层     │    │   业务逻辑层     │    │   数据访问层     │
│                 │    │                 │    │                 │
│ • Tkinter界面   │◄──►│ • 量化分析引擎  │◄──►│ • SQLite数据库  │
│ • 用户交互      │    │ • 机器学习模型  │    │ • 文件系统      │
│ • 状态显示      │    │ • 因子计算      │    │ • 网络API       │
│ • 结果展示      │    │ • 回测系统      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                    ┌─────────────────┐
                    │   数据源层      │
                    │                 │
                    │ • yfinance      │
                    └─────────────────┘
```

## 🔧 故障排除

### 常见问题

#### 1. 启动失败
```bash
# 检查Python版本
python --version

# 检查虚拟环境
python -c "import sys; print(sys.prefix)"

# 检查依赖
python -c "import pandas, numpy, yfinance; print('依赖正常')"
```

#### 2. 依赖安装失败
```bash
# 更新pip
python -m pip install --upgrade pip

# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name

# 重新安装依赖
python "首先一键安装所有依赖.py"
```

#### 3. 数据库问题
```bash
# 检查数据库
python check_database.py

# 查看数据库内容
python view_database.py

# 清理数据库
python -c "import os; os.remove('trading_results.db') if os.path.exists('trading_results.db') else None"
```

#### 4. 数据源连接问题
```bash
# 检查yfinance连接
python -c "import yfinance as yf; print(yf.Ticker('AAPL').info['shortName'])"
```

#### 4. 内存不足
- 减少分析的股票数量
- 缩短时间范围
- 关闭其他应用程序
- 增加系统虚拟内存

#### 5. 网络连接问题
- 检查网络连接
- 配置代理设置
- 使用VPN
- 稍后重试

### 系统特定问题

#### Windows
- 确保安装了Visual C++ Redistributable
- 检查Windows Defender防火墙设置
- 以管理员身份运行

#### Linux
```bash
# 安装系统依赖
sudo apt-get update
sudo apt-get install python3-dev python3-tk

# 设置权限
chmod +x *.sh
```

#### macOS
```bash
# 安装Xcode命令行工具
xcode-select --install

到此结束了mlgb#   a u t o - t r a d i n g - L S T M - B M A - M L - f i n a n c e - m o d e l  
 