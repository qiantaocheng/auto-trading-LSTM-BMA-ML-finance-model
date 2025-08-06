#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版量化分析模型
实施了高级因子选择、模型优化和集成学习技术
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import argparse
import os
import tempfile
from pathlib import Path
from scipy.stats import spearmanr
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 尝试导入高级模型
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost未安装，将使用传统模型")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("警告: LightGBM未安装，将使用传统模型")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("警告: CatBoost未安装，将使用传统模型")

# 尝试导入超参数调优工具
try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.model_selection import ParameterSampler
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False
    print("警告: 超参数调优工具未安装")

warnings.filterwarnings('ignore')

# 解析命令行参数
def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='增强版量化模型分析工具')
    parser.add_argument('--start-date', 
                        type=str, 
                        default=None,
                        help='开始日期 (YYYY-MM-DD格式，默认为3年前)')
    parser.add_argument('--end-date', 
                        type=str, 
                        default=None,
                        help='结束日期 (YYYY-MM-DD格式，默认为今天)')
    parser.add_argument('--ticker-file', 
                        type=str, 
                        default=None,
                        help='自定义股票列表文件路径 (每行一个股票代码)')
    parser.add_argument('--enable-hyperopt', 
                        action='store_true',
                        default=True,
                        help='启用超参数调优 (默认启用)')
    parser.add_argument('--disable-hyperopt', 
                        action='store_true',
                        default=False,
                        help='禁用超参数调优')
    parser.add_argument('--hyperopt-method', 
                        type=str, 
                        choices=['grid', 'randomized'],
                        default='randomized',
                        help='超参数调优方法 (grid=网格搜索, randomized=随机搜索)')
    parser.add_argument('--hyperopt-iter', 
                        type=int, 
                        default=20,
                        help='随机搜索的迭代次数 (默认20)')
    
    args = parser.parse_args()
    
    # 处理超参数调优设置
    if args.disable_hyperopt:
        args.enable_hyperopt = False
    
    return args

# 获取命令行参数
args = parse_arguments()

# 日期范围设置
if args.start_date:
    start_date = args.start_date
else:
    start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")

if args.end_date:
    end_date = args.end_date
else:
    end_date = datetime.now().strftime("%Y-%m-%d")

print(f"[DATE] 分析时间范围: {start_date} 到 {end_date}")

# 自定义股票列表加载函数
def load_custom_ticker_list(file_path):
    """从文件加载自定义股票列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except Exception as e:
        print(f"[ERROR] 加载自定义股票列表失败: {e}")
        return None

# 默认股票池
ticker_list_raw = [ # [WINNER] 核心科技股 (高成长+高质量)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE',
    'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX',
    'KLAC', 'MRVL', 'ON', 'SWKS', 'MCHP', 'ADI', 'XLNX', 'SNPS', 'CDNS', 'FTNT',
    
    # 消费与零售
    'COST', 'WMT', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'PYPL',
    'SQ', 'SHOP', 'EBAY', 'ETSY', 'W', 'CHWY', 'ROKU', 'SPOT', 'ZM', 'UBER',
    'LYFT', 'DASH', 'ABNB', 'BKNG', 'EXPE', 'TJX', 'ROST', 'ULTA', 'LULU', 'RH',
    
    # 医疗健康与生物技术
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'ABT', 'LLY', 'BMY',
    'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'CVS',
    'CI', 'HUM', 'ANTM', 'MCK', 'ABC', 'CAH', 'WAT', 'A', 'IQV', 'CRL',
    
    # 金融服务
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
    'AXP', 'BLK', 'SCHW', 'SPGI', 'ICE', 'CME', 'MCO', 'MSCI', 'PYPL', 'V',
    'MA', 'FIS', 'FISV', 'ADP', 'PAYX', 'WU', 'SYF', 'DFS', 'ALLY', 'RF',
    
    # 工业与材料
    'BA', 'CAT', 'DE', 'GE', 'HON', 'LMT', 'MMM', 'RTX', 'UPS', 'FDX',
    'NSC', 'UNP', 'CSX', 'ODFL', 'CHRW', 'EXPD', 'XPO', 'JBHT', 'KNX', 'J',
    'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'DOV', 'FTV', 'XYL', 'IEX', 'GNRC',
    
    # 能源与公用事业
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE',
    'WMB', 'ET', 'EPD', 'MPLX', 'AM', 'NEE', 'DUK', 'SO', 'EXC', 'XEL',
    'AEP', 'PCG', 'ED', 'EIX', 'PPL', 'AES', 'NRG', 'CNP', 'CMS', 'DTE',
    
    # 房地产
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'AVB', 'EQR', 'UDR',
    'ESS', 'MAA', 'CPT', 'AIV', 'EXR', 'PSA', 'BXP', 'VTR', 'HCP', 'PEAK',
    
    # 通信服务
    'VZ', 'T', 'TMUS', 'CHTR', 'CMCSA', 'VIA', 'LBRDA', 'LBRDK', 'DISH', 'SIRI',
    
    # 基础材料
    'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'GOLD', 'AA', 'X', 'CLF',
    'NUE', 'STLD', 'CMC', 'RS', 'WOR', 'RPM', 'PPG', 'DD', 'DOW', 'LYB',
    
    # 消费必需品
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'CPB',
    'CAG', 'SJM', 'HRL', 'TSN', 'TYSON', 'ADM', 'BG', 'CF', 'MOS', 'FMC',
    
    # 新兴增长股
    'SQ', 'SHOP', 'ROKU', 'ZOOM', 'DOCU', 'OKTA', 'SNOW', 'PLTR', 'RBLX', 'U',
    'DDOG', 'CRWD', 'ZS', 'NET', 'FSLY', 'TWLO', 'SPLK', 'WDAY', 'VEEV', 'ZEN',
    'TEAM', 'ATLASSIAN', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'BIDU', 'BILI', 'IQ',
    
    # 生物技术与创新医疗
    'MRNA', 'BNTX', 'NOVT', 'SGEN', 'BLUE', 'BMRN', 'TECH', 'SRPT', 'RARE', 'FOLD',
    'EDIT', 'CRSP', 'NTLA', 'BEAM', 'VERV', 'PRIME', 'SAGE', 'IONS', 'IOVA', 'ARWR',
    
    # 清洁能源与电动车
    'TSLA', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'QS', 'BLNK', 'CHPT', 'PLUG',
    'FCEL', 'BE', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'SPWR', 'CSIQ', 'JKS', 'SOL' # [WINNER] 核心科技股 (高成长+高质量)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE',
    'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX',
    'KLAC', 'MRVL', 'ON', 'SWKS', 'MCHP', 'ADI', 'XLNX', 'SNPS', 'CDNS', 'FTNT',
    
    # 消费与零售
    'COST', 'WMT', 'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'PYPL',
    'SQ', 'SHOP', 'EBAY', 'ETSY', 'W', 'CHWY', 'ROKU', 'SPOT', 'ZM', 'UBER',
    'LYFT', 'DASH', 'ABNB', 'BKNG', 'EXPE', 'TJX', 'ROST', 'ULTA', 'LULU', 'RH',
    
    # 医疗健康与生物技术
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'ABT', 'LLY', 'BMY',
    'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX', 'ZTS', 'CVS',
    'CI', 'HUM', 'ANTM', 'MCK', 'ABC', 'CAH', 'WAT', 'A', 'IQV', 'CRL',
    
    # 金融服务
    'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF',
    'AXP', 'BLK', 'SCHW', 'SPGI', 'ICE', 'CME', 'MCO', 'MSCI', 'PYPL', 'V',
    'MA', 'FIS', 'FISV', 'ADP', 'PAYX', 'WU', 'SYF', 'DFS', 'ALLY', 'RF',
    
    # 工业与材料
    'BA', 'CAT', 'DE', 'GE', 'HON', 'LMT', 'MMM', 'RTX', 'UPS', 'FDX',
    'NSC', 'UNP', 'CSX', 'ODFL', 'CHRW', 'EXPD', 'XPO', 'JBHT', 'KNX', 'J',
    'EMR', 'ETN', 'ITW', 'PH', 'ROK', 'DOV', 'FTV', 'XYL', 'IEX', 'GNRC',
    
    # 能源与公用事业
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE',
    'WMB', 'ET', 'EPD', 'MPLX', 'AM', 'NEE', 'DUK', 'SO', 'EXC', 'XEL',
    'AEP', 'PCG', 'ED', 'EIX', 'PPL', 'AES', 'NRG', 'CNP', 'CMS', 'DTE',
    
    # 房地产
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'AVB', 'EQR', 'UDR',
    'ESS', 'MAA', 'CPT', 'AIV', 'EXR', 'PSA', 'BXP', 'VTR', 'HCP', 'PEAK',
    
    # 通信服务
    'VZ', 'T', 'TMUS', 'CHTR', 'CMCSA', 'VIA', 'LBRDA', 'LBRDK', 'DISH', 'SIRI',
    
    # 基础材料
    'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'GOLD', 'AA', 'X', 'CLF',
    'NUE', 'STLD', 'CMC', 'RS', 'WOR', 'RPM', 'PPG', 'DD', 'DOW', 'LYB',
    
    # 消费必需品
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'CPB',
    'CAG', 'SJM', 'HRL', 'TSN', 'TYSON', 'ADM', 'BG', 'CF', 'MOS', 'FMC',
    
    # 新兴增长股
    'SQ', 'SHOP', 'ROKU', 'ZOOM', 'DOCU', 'OKTA', 'SNOW', 'PLTR', 'RBLX', 'U',
    'DDOG', 'CRWD', 'ZS', 'NET', 'FSLY', 'TWLO', 'SPLK', 'WDAY', 'VEEV', 'ZEN',
    'TEAM', 'ATLASSIAN', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'BIDU', 'BILI', 'IQ',
    
    # 生物技术与创新医疗
    'MRNA', 'BNTX', 'NOVT', 'SGEN', 'BLUE', 'BMRN', 'TECH', 'SRPT', 'RARE', 'FOLD',
    'EDIT', 'CRSP', 'NTLA', 'BEAM', 'VERV', 'PRIME', 'SAGE', 'IONS', 'IOVA', 'ARWR',
    
    # 清洁能源与电动车
    'TSLA', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'QS', 'BLNK', 'CHPT', 'PLUG',
    'FCEL', 'BE', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'SPWR', 'CSIQ', 'JKS', 'SOL']

# 确定使用的股票列表
if args.ticker_file:
    custom_list = load_custom_ticker_list(args.ticker_file)
    if custom_list:
        ticker_list = custom_list
        print(f"[CUSTOM] 从文件 {args.ticker_file} 加载 {len(ticker_list)} 只股票")
    else:
        print(f"[FALLBACK] 自定义列表加载失败，使用默认列表")
        ticker_list = list(dict.fromkeys(ticker_list_raw))
else:
    ticker_list = list(dict.fromkeys(ticker_list_raw))

print(f"[FINAL] 最终股票池: {len(ticker_list)} 只股票")

class EnhancedQuantitativeModel:
    """增强版量化分析模型"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 保持95%的方差
        self.models = {}
        self.feature_importance = {}
        self.factor_ic_scores = {}
        self.best_params = {}  # 存储最佳超参数
        
    def get_hyperparameter_grids(self):
        """定义超参数网格"""
        param_grids = {}
        
        # RandomForest 超参数
        param_grids['RandomForest'] = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Ridge 超参数
        param_grids['Ridge'] = {
            'alpha': [0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr']
        }
        
        # XGBoost 超参数
        if XGBOOST_AVAILABLE:
            param_grids['XGBoost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2]
            }
        
        # LightGBM 超参数
        if LIGHTGBM_AVAILABLE:
            param_grids['LightGBM'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2],
                'num_leaves': [31, 50, 100]
            }
        
        # CatBoost 超参数
        if CATBOOST_AVAILABLE:
            param_grids['CatBoost'] = {
                'iterations': [50, 100, 200],
                'depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5],
                'border_count': [32, 64, 128]
            }
        
        return param_grids
    
    def tune_hyperparameters(self, X, y, model_name, base_model, param_grid, method='randomized', n_iter=20):
        """超参数调优"""
        print(f"[TUNING] 开始调优 {model_name}...")
        
        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=3)  # 减少分割数以提高速度
        
        try:
            if method == 'grid' and len(param_grid) > 0:
                # 网格搜索
                search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=tscv,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=0
                )
            elif method == 'randomized' and len(param_grid) > 0:
                # 随机搜索
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=n_iter,
                    cv=tscv,
                    scoring='r2',
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
            else:
                # 如果没有参数网格，返回原模型
                print(f"[TUNING] {model_name}: 没有参数网格，使用默认参数")
                return base_model, {}
            
            # 执行搜索
            search.fit(X, y)
            
            best_params = search.best_params_
            best_score = search.best_score_
            
            print(f"[TUNING] {model_name}: 最佳R² = {best_score:.4f}")
            print(f"[TUNING] {model_name}: 最佳参数 = {best_params}")
            
            return search.best_estimator_, best_params
            
        except Exception as e:
            print(f"[TUNING ERROR] {model_name}: {e}")
            print(f"[TUNING] {model_name}: 使用默认参数")
            return base_model, {}
    
    def download_data(self, tickers, start_date, end_date):
        """下载股票数据"""
        print(f"[DATA] 下载 {len(tickers)} 只股票的数据...")
        all_data = {}
        
        for i, ticker in enumerate(tickers):
            try:
                print(f"[{i+1}/{len(tickers)}] 下载 {ticker}...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty and len(data) > 60:  # 至少需要60天数据
                    all_data[ticker] = data
                else:
                    print(f"[SKIP] {ticker}: 数据不足")
                    
            except Exception as e:
                print(f"[ERROR] {ticker}: {e}")
                continue
                
        print(f"[SUCCESS] 成功下载 {len(all_data)} 只股票数据")
        return all_data
    
    def winsorize(self, data, lower_percentile=0.05, upper_percentile=0.95):
        """Winsorizing处理异常值"""
        lower_bound = data.quantile(lower_percentile)
        upper_bound = data.quantile(upper_percentile)
        return data.clip(lower=lower_bound, upper=upper_bound)
    
    def calculate_ic(self, factor_data, returns):
        """计算信息系数(IC)"""
        try:
            # 确保数据对齐
            aligned_factor, aligned_returns = factor_data.align(returns, join='inner')
            
            # 移除NaN值
            mask = ~(aligned_factor.isna() | aligned_returns.isna())
            if mask.sum() < 10:  # 至少需要10个有效数据点
                return 0.0
                
            clean_factor = aligned_factor[mask]
            clean_returns = aligned_returns[mask]
            
            # 计算Spearman相关系数
            ic, p_value = spearmanr(clean_factor, clean_returns)
            return ic if not np.isnan(ic) else 0.0
            
        except Exception as e:
            print(f"[IC CALC ERROR] {e}")
            return 0.0
    
    def neutralize_factor(self, factor_data, industry_data=None, market_cap_data=None):
        """因子中性化处理"""
        try:
            # 简化版：只使用市值中性化
            if market_cap_data is not None:
                # 确保数据对齐
                aligned_factor, aligned_cap = factor_data.align(market_cap_data, join='inner')
                
                # 移除NaN值
                mask = ~(aligned_factor.isna() | aligned_cap.isna())
                if mask.sum() < 10:
                    return factor_data
                
                clean_factor = aligned_factor[mask]
                clean_cap = aligned_cap[mask]
                
                # 线性回归
                X = sm.add_constant(clean_cap)
                model = sm.OLS(clean_factor, X).fit()
                residuals = model.resid
                
                # 创建完整的残差序列
                result = factor_data.copy()
                result.loc[mask] = residuals
                return result
            
            return factor_data
            
        except Exception as e:
            print(f"[NEUTRALIZE ERROR] {e}")
            return factor_data
    
    def calculate_factors(self, data):
        """计算技术和基本面因子"""
        factors = {}
        
        try:
            # 确保必要的列存在
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                print(f"[FACTOR ERROR] 缺少必要的数据列")
                return factors
            
            # 价格相关因子
            factors['returns_1d'] = data['Close'].pct_change()
            factors['returns_5d'] = data['Close'].pct_change(5)
            factors['returns_20d'] = data['Close'].pct_change(20)
            
            # 移动平均因子
            factors['ma5'] = data['Close'].rolling(5).mean() / data['Close'] - 1
            factors['ma20'] = data['Close'].rolling(20).mean() / data['Close'] - 1
            factors['ma60'] = data['Close'].rolling(60).mean() / data['Close'] - 1
            
            # 波动率因子
            factors['volatility_5d'] = data['Close'].pct_change().rolling(5).std()
            factors['volatility_20d'] = data['Close'].pct_change().rolling(20).std()
            
            # 技术指标
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            factors['rsi'] = 100 - (100 / (1 + gain / loss))
            
            # MACD
            exp1 = data['Close'].ewm(span=12).mean()
            exp2 = data['Close'].ewm(span=26).mean()
            factors['macd'] = exp1 - exp2
            factors['macd_signal'] = factors['macd'].ewm(span=9).mean()
            factors['macd_histogram'] = factors['macd'] - factors['macd_signal']
            
            # 布林带
            bb_middle = data['Close'].rolling(20).mean()
            bb_std = data['Close'].rolling(20).std()
            factors['bb_upper'] = (bb_middle + 2 * bb_std) / data['Close'] - 1
            factors['bb_lower'] = (bb_middle - 2 * bb_std) / data['Close'] - 1
            factors['bb_position'] = (data['Close'] - bb_middle) / (2 * bb_std)
            
            # 成交量因子
            factors['volume_ma5'] = data['Volume'].rolling(5).mean() / data['Volume'] - 1
            factors['volume_ma20'] = data['Volume'].rolling(20).mean() / data['Volume'] - 1
            
            # 价量因子
            factors['price_volume'] = data['Close'].pct_change() * data['Volume']
            
            # 市值估计 (简化)
            factors['market_cap_proxy'] = data['Close'] * data['Volume']
            
            # 清理无穷大和NaN值
            for factor_name, factor_data in factors.items():
                factors[factor_name] = factor_data.replace([np.inf, -np.inf], np.nan)
            
            print(f"[FACTORS] 计算了 {len(factors)} 个因子")
            return factors
            
        except Exception as e:
            print(f"[FACTOR CALC ERROR] {e}")
            return {}
    
    def select_factors_by_ic(self, factor_dict, returns, ic_threshold=0.05):
        """基于IC选择有效因子"""
        print(f"[FACTOR SELECTION] 开始因子筛选...")
        
        selected_factors = {}
        ic_scores = {}
        
        for factor_name, factor_data in factor_dict.items():
            ic_score = self.calculate_ic(factor_data, returns)
            ic_scores[factor_name] = ic_score
            
            if abs(ic_score) > ic_threshold:
                selected_factors[factor_name] = factor_data
                print(f"[SELECTED] {factor_name}: IC = {ic_score:.4f}")
            else:
                print(f"[REJECTED] {factor_name}: IC = {ic_score:.4f} (低于阈值 {ic_threshold})")
        
        self.factor_ic_scores = ic_scores
        print(f"[FACTOR SELECTION] 从 {len(factor_dict)} 个因子中选择了 {len(selected_factors)} 个")
        
        return selected_factors
    
    def remove_multicollinear_factors(self, factor_dict, correlation_threshold=0.9):
        """移除高度相关的因子"""
        print(f"[MULTICOLLINEARITY] 移除相关性高于 {correlation_threshold} 的因子...")
        
        # 创建因子DataFrame
        factor_df = pd.DataFrame(factor_dict)
        factor_df = factor_df.dropna()
        
        if factor_df.empty:
            return factor_dict
        
        # 计算相关性矩阵
        corr_matrix = factor_df.corr().abs()
        
        # 找到高度相关的因子对
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 找到需要删除的因子
        to_drop = [column for column in upper_triangle.columns 
                  if any(upper_triangle[column] > correlation_threshold)]
        
        # 创建筛选后的因子字典
        filtered_factors = {k: v for k, v in factor_dict.items() if k not in to_drop}
        
        print(f"[MULTICOLLINEARITY] 移除了 {len(to_drop)} 个高度相关的因子: {to_drop}")
        print(f"[MULTICOLLINEARITY] 保留了 {len(filtered_factors)} 个因子")
        
        return filtered_factors
    
    def prepare_ml_data(self, all_data):
        """准备机器学习数据"""
        print(f"[ML PREP] 准备机器学习数据...")
        
        all_features = []
        all_targets = []
        stock_names = []
        dates = []
        
        for ticker, data in all_data.items():
            try:
                print(f"[ML PREP] 处理 {ticker}...")
                
                # 计算因子
                factors = self.calculate_factors(data)
                if not factors:
                    continue
                
                # 计算未来收益率 (目标变量)
                future_returns = data['Close'].pct_change(20).shift(-20)  # 20天后的收益率
                
                # 因子筛选
                effective_factors = self.select_factors_by_ic(factors, future_returns)
                if not effective_factors:
                    continue
                
                # 移除多重共线性
                final_factors = self.remove_multicollinear_factors(effective_factors)
                if not final_factors:
                    continue
                
                # Winsorizing处理
                for factor_name, factor_data in final_factors.items():
                    final_factors[factor_name] = self.winsorize(factor_data)
                
                # 因子中性化
                market_cap_proxy = factors.get('market_cap_proxy')
                for factor_name, factor_data in final_factors.items():
                    if factor_name != 'market_cap_proxy':
                        final_factors[factor_name] = self.neutralize_factor(
                            factor_data, market_cap_data=market_cap_proxy
                        )
                
                # 创建特征DataFrame
                feature_df = pd.DataFrame(final_factors)
                feature_df['target'] = future_returns
                
                # 删除NaN行
                feature_df = feature_df.dropna()
                
                if len(feature_df) < 50:  # 需要足够的样本
                    continue
                
                # 分离特征和目标
                X = feature_df.drop('target', axis=1)
                y = feature_df['target']
                
                # 添加股票名称和日期
                stock_name_series = pd.Series([ticker] * len(X), index=X.index)
                date_series = X.index
                
                all_features.append(X)
                all_targets.append(y)
                stock_names.extend(stock_name_series.tolist())
                dates.extend(date_series.tolist())
                
            except Exception as e:
                print(f"[ML PREP ERROR] {ticker}: {e}")
                continue
        
        if not all_features:
            raise ValueError("没有有效的特征数据")
        
        # 合并所有数据
        X_combined = pd.concat(all_features, ignore_index=True)
        y_combined = pd.concat(all_targets, ignore_index=True)
        
        print(f"[ML PREP] 总共准备了 {len(X_combined)} 个样本，{X_combined.shape[1]} 个特征")
        
        return X_combined, y_combined, stock_names, dates
    
    def train_models(self, X, y, enable_hyperopt=True):
        """训练多个模型"""
        print(f"[MODEL TRAINING] 开始训练模型...")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA降维
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"[PCA] 降维后保留 {X_pca.shape[1]} 个主成分")
        
        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 获取超参数网格
        param_grids = self.get_hyperparameter_grids() if enable_hyperopt and HYPEROPT_AVAILABLE else {}
        
        # 训练传统模型
        base_models = {
            'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'Ridge': Ridge()
        }
        
        # 添加高级模型
        if XGBOOST_AVAILABLE:
            base_models['XGBoost'] = xgb.XGBRegressor(random_state=42)
        
        if LIGHTGBM_AVAILABLE:
            base_models['LightGBM'] = lgb.LGBMRegressor(random_state=42, verbose=-1)
        
        if CATBOOST_AVAILABLE:
            base_models['CatBoost'] = CatBoostRegressor(random_state=42, verbose=False)
        
        # 训练和评估模型
        model_scores = {}
        
        for name, base_model in base_models.items():
            try:
                print(f"[TRAINING] {name}...")
                
                # 超参数调优
                if enable_hyperopt and HYPEROPT_AVAILABLE and name in param_grids:
                    # 获取超参数调优配置
                    hyperopt_method = getattr(args, 'hyperopt_method', 'randomized')
                    hyperopt_iter = getattr(args, 'hyperopt_iter', 20)
                    
                    tuned_model, best_params = self.tune_hyperparameters(
                        X_pca, y, name, base_model, param_grids[name],
                        method=hyperopt_method, n_iter=hyperopt_iter
                    )
                    self.best_params[name] = best_params
                    final_model = tuned_model
                else:
                    # 使用默认参数
                    if name == 'RandomForest':
                        final_model = RandomForestRegressor(
                            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
                        )
                    elif name == 'Ridge':
                        final_model = Ridge(alpha=1.0)
                    elif name == 'XGBoost':
                        final_model = xgb.XGBRegressor(
                            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
                        )
                    elif name == 'LightGBM':
                        final_model = lgb.LGBMRegressor(
                            n_estimators=100, max_depth=6, learning_rate=0.1, 
                            random_state=42, verbose=-1
                        )
                    elif name == 'CatBoost':
                        final_model = CatBoostRegressor(
                            iterations=100, depth=6, learning_rate=0.1, 
                            random_state=42, verbose=False
                        )
                    else:
                        final_model = base_model
                
                # 交叉验证评分
                cv_scores = cross_val_score(
                    final_model, X_pca, y, 
                    cv=tscv, 
                    scoring='r2',
                    n_jobs=-1
                )
                
                mean_score = cv_scores.mean()
                std_score = cv_scores.std()
                
                print(f"[{name}] R² = {mean_score:.4f} ± {std_score:.4f}")
                
                # 训练最终模型
                final_model.fit(X_pca, y)
                self.models[name] = final_model
                model_scores[name] = mean_score
                
                # 记录特征重要性（如果支持）
                if hasattr(final_model, 'feature_importances_'):
                    self.feature_importance[name] = final_model.feature_importances_
                
            except Exception as e:
                print(f"[TRAINING ERROR] {name}: {e}")
                continue
        
        print(f"[TRAINING] 成功训练了 {len(self.models)} 个模型")
        if enable_hyperopt and HYPEROPT_AVAILABLE:
            print(f"[HYPEROPT] 完成超参数调优")
        
        return model_scores
    
    def ensemble_predict(self, X, method='weighted_average'):
        """集成预测"""
        if not self.models:
            raise ValueError("没有训练好的模型")
        
        # 数据预处理
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        predictions = {}
        
        # 获取各模型预测
        for name, model in self.models.items():
            try:
                pred = model.predict(X_pca)
                predictions[name] = pred
            except Exception as e:
                print(f"[PREDICTION ERROR] {name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("所有模型预测都失败了")
        
        # 集成预测
        if method == 'simple_average':
            # 简单平均
            pred_array = np.array(list(predictions.values()))
            final_prediction = np.mean(pred_array, axis=0)
            
        elif method == 'weighted_average':
            # 加权平均（基于模型性能）
            weights = {
                'RandomForest': 0.2,
                'Ridge': 0.1,
                'XGBoost': 0.3,
                'LightGBM': 0.25,
                'CatBoost': 0.15
            }
            
            weighted_sum = np.zeros(len(list(predictions.values())[0]))
            total_weight = 0
            
            for name, pred in predictions.items():
                weight = weights.get(name, 0.1)
                weighted_sum += weight * pred
                total_weight += weight
            
            final_prediction = weighted_sum / total_weight
            
        else:
            # 默认简单平均
            pred_array = np.array(list(predictions.values()))
            final_prediction = np.mean(pred_array, axis=0)
        
        return final_prediction, predictions
    
    def generate_recommendations(self, all_data, top_n=20):
        """生成投资建议"""
        print(f"[RECOMMENDATIONS] 生成投资建议...")
        
        recommendations = []
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        for ticker, data in all_data.items():
            try:
                # 计算当前因子
                factors = self.calculate_factors(data)
                if not factors:
                    continue
                
                # 获取最新数据
                latest_factors = {}
                for factor_name, factor_data in factors.items():
                    if not factor_data.empty:
                        latest_value = factor_data.iloc[-1]
                        if not pd.isna(latest_value):
                            latest_factors[factor_name] = latest_value
                
                if not latest_factors:
                    continue
                
                # 创建特征向量
                feature_names = list(latest_factors.keys())
                feature_values = [latest_factors[name] for name in feature_names]
                X_current = pd.DataFrame([feature_values], columns=feature_names)
                
                # 预测
                prediction, individual_preds = self.ensemble_predict(X_current)
                predicted_return = prediction[0]
                
                # 获取当前价格和基本信息
                current_price = data['Close'].iloc[-1]
                
                # 计算技术指标
                returns_1d = data['Close'].pct_change().iloc[-1] if len(data) > 1 else 0
                returns_5d = data['Close'].pct_change(5).iloc[-1] if len(data) > 5 else 0
                returns_20d = data['Close'].pct_change(20).iloc[-1] if len(data) > 20 else 0
                
                volatility = data['Close'].pct_change().rolling(20).std().iloc[-1] if len(data) > 20 else 0
                volume_avg = data['Volume'].rolling(20).mean().iloc[-1] if len(data) > 20 else 0
                
                # 创建推荐记录
                recommendation = {
                    'ticker': ticker,
                    'current_price': current_price,
                    'predicted_return': predicted_return,
                    'prediction_confidence': abs(predicted_return),  # 简化的置信度
                    'returns_1d': returns_1d,
                    'returns_5d': returns_5d,
                    'returns_20d': returns_20d,
                    'volatility_20d': volatility,
                    'volume_avg_20d': volume_avg,
                    'individual_predictions': individual_preds,
                    'analysis_date': current_date
                }
                
                recommendations.append(recommendation)
                
            except Exception as e:
                print(f"[REC ERROR] {ticker}: {e}")
                continue
        
        # 按预测收益率排序
        recommendations.sort(key=lambda x: x['predicted_return'], reverse=True)
        
        # 返回top N
        top_recommendations = recommendations[:top_n]
        
        print(f"[RECOMMENDATIONS] 生成了 {len(recommendations)} 个建议，返回前 {len(top_recommendations)} 个")
        
        return top_recommendations
    
    def save_results(self, recommendations, factor_ic_scores):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建结果目录
        result_dir = Path("result")
        result_dir.mkdir(exist_ok=True)
        
        # 保存推荐结果
        recommendations_df = pd.DataFrame(recommendations)
        
        # 添加评级
        if not recommendations_df.empty:
            # 基于预测收益率分级
            quantiles = recommendations_df['predicted_return'].quantile([0.33, 0.67])
            
            def get_rating(return_val):
                if return_val >= quantiles.iloc[1]:
                    return 'BUY'
                elif return_val >= quantiles.iloc[0]:
                    return 'HOLD'
                else:
                    return 'SELL'
            
            recommendations_df['rating'] = recommendations_df['predicted_return'].apply(get_rating)
            
            # 保存到Excel - 使用与标准模型相同的前缀
            excel_file = result_dir / f"quantitative_analysis_{timestamp}.xlsx"
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 主要推荐
                recommendations_df.to_excel(writer, sheet_name='推荐结果', index=False)
                
                # 因子IC评分
                ic_df = pd.DataFrame(list(factor_ic_scores.items()), 
                                   columns=['Factor', 'IC_Score'])
                ic_df = ic_df.sort_values('IC_Score', key=abs, ascending=False)
                ic_df.to_excel(writer, sheet_name='因子IC评分', index=False)
                
                # 模型性能对比
                if hasattr(self, 'models') and self.models:
                    model_info = []
                    for name, model in self.models.items():
                        model_info.append({
                            'Model': name,
                            'Type': type(model).__name__,
                            'Available': True,
                            'Hyperparameters_Tuned': name in self.best_params,
                            'Best_Params': str(self.best_params.get(name, 'Default'))
                        })
                    
                    model_df = pd.DataFrame(model_info)
                    model_df.to_excel(writer, sheet_name='模型信息', index=False)
                
                # 超参数调优结果
                if hasattr(self, 'best_params') and self.best_params:
                    hyperopt_data = []
                    for model_name, params in self.best_params.items():
                        for param_name, param_value in params.items():
                            hyperopt_data.append({
                                'Model': model_name,
                                'Parameter': param_name,
                                'Best_Value': param_value
                            })
                    
                    if hyperopt_data:
                        hyperopt_df = pd.DataFrame(hyperopt_data)
                        hyperopt_df.to_excel(writer, sheet_name='超参数调优', index=False)
            
            print(f"[SAVE] 结果已保存到: {excel_file}")
            
            # 打印总结
            print(f"\n=== 增强版量化分析结果总结 ===")
            print(f"分析时间: {timestamp}")
            print(f"股票总数: {len(recommendations_df)}")
            print(f"BUY推荐: {len(recommendations_df[recommendations_df['rating'] == 'BUY'])}")
            print(f"HOLD推荐: {len(recommendations_df[recommendations_df['rating'] == 'HOLD'])}")
            print(f"SELL推荐: {len(recommendations_df[recommendations_df['rating'] == 'SELL'])}")
            print(f"平均预测收益率: {recommendations_df['predicted_return'].mean():.4f}")
            print(f"有效因子数量: {len([ic for ic in factor_ic_scores.values() if abs(ic) > 0.05])}")
            print(f"使用模型数量: {len(self.models)}")
            
            return excel_file
        
        return None

def main():
    """主函数"""
    print("=== 增强版量化分析模型启动 ===")
    print(f"使用高级模型: XGBoost={XGBOOST_AVAILABLE}, LightGBM={LIGHTGBM_AVAILABLE}, CatBoost={CATBOOST_AVAILABLE}")
    print(f"超参数调优: {HYPEROPT_AVAILABLE and args.enable_hyperopt}")
    
    try:
        # 创建模型实例
        model = EnhancedQuantitativeModel()
        
        # 下载数据
        all_data = model.download_data(ticker_list, start_date, end_date)
        
        if not all_data:
            print("[ERROR] 没有获取到有效数据")
            return
        
        # 准备机器学习数据
        X, y, stock_names, dates = model.prepare_ml_data(all_data)
        
        # 训练模型
        model_scores = model.train_models(X, y, enable_hyperopt=args.enable_hyperopt)
        
        # 生成投资建议
        recommendations = model.generate_recommendations(all_data)
        
        # 保存结果
        result_file = model.save_results(recommendations, model.factor_ic_scores)
        
        print(f"\n=== 分析完成 ===")
        if result_file:
            print(f"结果文件: {result_file}")
        
        if args.enable_hyperopt and HYPEROPT_AVAILABLE:
            print(f"\n=== 超参数调优总结 ===")
            for model_name, params in model.best_params.items():
                print(f"{model_name}: {len(params)} 个参数已优化")
        
    except Exception as e:
        print(f"[MAIN ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 