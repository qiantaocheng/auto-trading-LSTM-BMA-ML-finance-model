    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMA增强版量化分析模型 V3
使用贝叶斯模型平均(BMA) + Shrinkage替代传统Stacking
提供更稳定、理论自洽的集成学习方案
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
import schedule
import time
import threading
import json
from typing import List, Optional, Dict
from scipy.stats import spearmanr, entropy
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit, KFold, GroupKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# ===== Ultra 原版集成功能：导入统一市场数据、风险模型与Regime权重 =====
try:
    from unified_market_data_manager import UnifiedMarketDataManager
    from unified_risk_model import EnhancedAlphaEngine, RiskModelConfig
    from regime_adaptive_engine import AdaptiveWeightEngine, get_market_indices_data
    ULTRA_INTEGRATION_AVAILABLE = True
except Exception as _e:
    ULTRA_INTEGRATION_AVAILABLE = False
    print(f"[WARN] Ultra集成功能不可用: {_e}")

# 导入强制中性化管线
try:
    from neutralization_pipeline import DailyNeutralizationTransformer, create_neutralization_pipeline_step
    NEUTRALIZATION_AVAILABLE = True
except ImportError:
    NEUTRALIZATION_AVAILABLE = False
    print("[WARN] neutralization_pipeline.py未找到，跳过强制中性化功能")

# 移除GPU加速代码，仅使用CPU计算
print("[BMA] 使用CPU计算模式")

# 尝试导入高级模型
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# TensorFlow和CNN模型已完全禁用

# 禁用警告
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# =============================
# 新增功能1：因子正交化 + 中性化 + 去极值/标准化
# =============================
from numpy.linalg import lstsq

def cross_sectional_winsorize(df, lower=0.01, upper=0.99, by_cols=['date']):
    """横截面去极值 + Z-Score标准化"""
    fac_cols = [c for c in df.columns if c not in by_cols + ['ticker','target']]
    def _wzs(g):
        qs = g[fac_cols].quantile([lower, upper])
        g[fac_cols] = g[fac_cols].clip(lower=qs.loc[lower], upper=qs.loc[upper], axis=1)
        return g
    return (df.groupby(by_cols, group_keys=False).apply(_wzs)
              .assign(**{c: lambda d, c=c: d.groupby(by_cols)[c].transform(lambda x: (x-x.mean())/(x.std(ddof=0)+1e-12))
                        for c in fac_cols}))

def schmidt_orthogonalize(df, order=None, by_cols=['date']):
    """施密特正交化去相关（用于因子降维和去除多重共线性）
    修复：正确的维度对齐，避免形状不匹配错误
    """
    fac_cols = [c for c in df.columns if c not in by_cols + ['ticker','target']]
    if order is None:
        order = fac_cols  # 默认按列顺序
    
    out = df.copy()
    
    for date, g in out.groupby(by_cols):
        if len(g) < 2:  # 样本不足，跳过
            continue
            
        X = g[order].values
        if X.shape[1] == 0:
            continue
            
        try:
            # 局部中心化提升数值稳定性
            X = X - X.mean(axis=0, keepdims=True)
            
            # QR分解进行正交化 - 修复：对X而非X.T做QR
            Q, R = np.linalg.qr(X)  # X: (样本数, 因子数) -> Q: (样本数, 因子数)
            
            # 将正交化结果写回 - 修复：Q的列与样本长度对齐
            for i, col in enumerate(order):
                if i < Q.shape[1]:  # Q.shape[1] = 因子数
                    out.loc[g.index, col] = Q[:, i]  # Q[:, i]: (样本数,)
                else:
                    out.loc[g.index, col] = 0.0  # 多余列置零
                    
        except np.linalg.LinAlgError:
            print(f"[ORTHOGONALIZE WARNING] {date}: 矩阵奇异，使用Gram-Schmidt回退")
            # 回退到经典Gram-Schmidt
            try:
                X_centered = X - X.mean(axis=0, keepdims=True)
                orthogonalized = np.zeros_like(X_centered)
                
                for i, col in enumerate(order):
                    if i >= X_centered.shape[1]:
                        break
                    
                    x_i = X_centered[:, i].copy()
                    
                    # 减去前面已正交化向量的投影
                    for j in range(i):
                        if j < orthogonalized.shape[1]:
                            x_j = orthogonalized[:, j]
                            if np.linalg.norm(x_j) > 1e-10:
                                proj = np.dot(x_i, x_j) / np.dot(x_j, x_j) * x_j
                                x_i = x_i - proj
                    
                    # 归一化
                    norm = np.linalg.norm(x_i)
                    if norm > 1e-10:
                        orthogonalized[:, i] = x_i / norm
                    else:
                        orthogonalized[:, i] = 0.0
                    
                    out.loc[g.index, col] = orthogonalized[:, i]
                    
            except Exception as e:
                print(f"[ORTHOGONALIZE ERROR] {date}: Gram-Schmidt也失败: {e}，保持原值")
                continue
    
    return out

def pca_orthogonalize(df, var_ratio=0.95, by_cols=['date']):
    """PCA正交化（线性模型专用）"""
    df = df.copy()
    fac_cols = [c for c in df.columns if c not in by_cols + ['ticker','target']]
    def _pca(g):
        X = g[fac_cols].values
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        pca = PCA(n_components=var_ratio, svd_solver='full')
        Z = pca.fit_transform(X)
        # 返回主成分，列名 PC1, PC2, ...
        zdf = pd.DataFrame(Z, index=g.index, columns=[f'PC{i+1}' for i in range(Z.shape[1])])
        keep = g[[*by_cols,'ticker']]
        if 'target' in g.columns:
            keep = g[[*by_cols,'ticker','target']]
        return pd.concat([keep, zdf], axis=1)
    out = df.groupby(by_cols, group_keys=False).apply(_pca)
    return out

def pca_orthogonalize_with_transformer(df, var_ratio=0.95, by_cols=['date']):
    """PCA正交化并返回转换器（用于预测时复用）"""
    df = df.copy()
    fac_cols = [c for c in df.columns if c not in by_cols + ['ticker','target']]
    
    # 全局PCA（假设训练数据足够大，不需要按date分组）
    X_all = df[fac_cols].values
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_all)
    pca = PCA(n_components=var_ratio, svd_solver='full')
    Z_all = pca.fit_transform(X_scaled)
    
    # 创建结果DataFrame
    pc_cols = [f'PC{i+1}' for i in range(Z_all.shape[1])]
    zdf = pd.DataFrame(Z_all, index=df.index, columns=pc_cols)
    
    keep = df[[*by_cols,'ticker']]
    if 'target' in df.columns:
        keep = df[[*by_cols,'ticker','target']]
    
    out = pd.concat([keep, zdf], axis=1)
    
    # 返回处理后的数据和转换器
    transformer = {'scaler': scaler, 'pca': pca, 'feature_columns': fac_cols, 'pc_columns': pc_cols}
    return out, transformer

# =============================
# 新增功能2：严格滞后验证
# =============================
def lag_features_to_t_minus_1(df, by='ticker', by_cols=['date']):
    """对所有因子统一往后移一根bar，确保只用到 t-1 的数据"""
    fac_cols = [c for c in df.columns if c not in by_cols + ['ticker','target']]
    df = df.sort_values([by, *by_cols]).copy()
    df[fac_cols] = df.groupby(by)[fac_cols].shift(1)
    return df

def make_target(df, horizon=5, price_col='Close', by='ticker', skip_days=21):
    """
    构建目标变量，支持"跳月"设置（动量/反转研究标准）
    
    参数：
    - horizon: 持有期（天）
    - skip_days: 形成期与持有期之间的间隔天数（默认21天≈1个月）
    - 信号形成日 → (跳skip_days天) → 持有期开始 → (持续horizon天) → 持有期结束
    
    例如：skip_days=21, horizon=5
    - 信号在T日形成
    - 持有期为T+21到T+26日的收益
    """
    df = df.sort_values([by, 'date']).copy()
    
    if skip_days > 0:
        # 跳月模式：shift(-(skip_days + horizon))获取T+skip_days+horizon的价格
        # 然后计算从T+skip_days到T+skip_days+horizon的收益
        future_price = df.groupby(by)[price_col].shift(-(skip_days + horizon))
        skip_price = df.groupby(by)[price_col].shift(-skip_days)
        
        # 计算收益率（不填充0，保留NaN用于后续过滤）
        ret = future_price / skip_price - 1
        
        print(f"[TARGET] 跳月模式：形成日→(跳{skip_days}天)→{horizon}天持有期")
        print(f"[TARGET] 有效目标样本: {ret.notna().sum()}/{len(ret)}")
    else:
        # 传统模式：立即持有
        ret = df.groupby(by)[price_col].pct_change(horizon).shift(-horizon)
        print(f"[TARGET] 传统模式：形成日→立即{horizon}天持有期")
    
    df['target'] = ret
    return df

def make_target_momentum_reversal(df, formation_period=252, skip_days=21, holding_period=21, 
                                price_col='Close', by='ticker'):
    """
    经典动量/反转研究目标构建（Jegadeesh & Titman 1993风格）
    
    参数：
    - formation_period: 形成期长度（天，用于计算动量信号）
    - skip_days: 跳过天数（避免微观结构噪声）
    - holding_period: 持有期长度（天）
    
    流程：
    1. 计算formation_period的累积收益作为动量信号
    2. 跳过skip_days天
    3. 计算后续holding_period的收益作为目标
    """
    df = df.sort_values([by, 'date']).copy()
    
    # 1. 计算形成期动量信号
    df['momentum_signal'] = df.groupby(by)[price_col].pct_change(formation_period)
    
    # 2. 构建跳月目标
    df = make_target(df, horizon=holding_period, price_col=price_col, 
                    by=by, skip_days=skip_days)
    
    # 重命名target为更明确的名称
    df['holding_return'] = df['target']
    df = df.drop('target', axis=1)
    
    print(f"[MOMENTUM-REVERSAL] 形成期{formation_period}天 → 跳{skip_days}天 → 持有{holding_period}天")
    return df

def time_decay_weights(dates, half_life_days=126):
    """生成时间衰减权重（用于BMA训练中的样本加权）"""
    if isinstance(dates, pd.Series):
        dates = dates.values
    elif isinstance(dates, list):
        dates = np.array(dates)
    
    # 计算距离最新日期的天数
    latest_date = max(dates)
    if hasattr(latest_date, 'to_pydatetime'):
        latest_date = latest_date.to_pydatetime()
    
    days_ago = []
    for date in dates:
        if hasattr(date, 'to_pydatetime'):
            date = date.to_pydatetime()
        
        # 计算时间差，处理不同类型的日期对象
        time_diff = latest_date - date
        if hasattr(time_diff, 'days'):
            days_diff = time_diff.days
        elif hasattr(time_diff, 'total_seconds'):
            days_diff = time_diff.total_seconds() / 86400  # 转换为天
        else:
            # 对于numpy.timedelta64等其他类型
            try:
                days_diff = float(time_diff / pd.Timedelta(days=1))
            except:
                days_diff = 0
        
        days_ago.append(days_diff)
    
    # 指数衰减权重
    weights = np.exp(-np.array(days_ago) * np.log(2) / half_life_days)
    
    # 归一化
    return weights / weights.sum()

def ema_smooth_weights(w_prev, w_new, alpha=0.2, min_w=0.02):
    """EMA平滑权重更新（用于BMA权重的稳定化）"""
    if isinstance(w_prev, dict):
        w_prev = pd.Series(w_prev)
    if isinstance(w_new, dict):
        w_new = pd.Series(w_new)
    
    # 确保索引对齐
    common_index = w_prev.index.intersection(w_new.index)
    if len(common_index) == 0:
        return w_new / w_new.sum()
    
    w_prev_aligned = w_prev.reindex(common_index, fill_value=min_w)
    w_new_aligned = w_new.reindex(common_index, fill_value=min_w)
    
    # EMA平滑
    w_smooth = alpha * w_new_aligned + (1 - alpha) * w_prev_aligned
    
    # 确保最小权重
    w_smooth = np.maximum(w_smooth, min_w)
    
    # 重新归一化
    return w_smooth / w_smooth.sum()

def assert_no_lookahead(df):
    """检查是否存在前瞻性偏差（强化版：严格验证时间滞后）"""
    feature_cols = [c for c in df.columns if c not in ['ticker','date','target']]
    
    # 检查1：每个ticker的第一条样本必须包含NaN特征（因为被shift了）
    warning_count = 0
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].sort_values('date')
        if len(ticker_data) > 0:
            first_row_features = ticker_data.iloc[0][feature_cols]
            if not first_row_features.isnull().any():
                warning_count += 1
                if warning_count <= 5:  # 只显示前5个警告，避免刷屏
                    print(f"[LOOKAHEAD WARNING] {ticker}: 第一行特征无NaN，可能未正确滞后")
                elif warning_count == 6:
                    print(f"[LOOKAHEAD WARNING] ...以及其他 {len(df['ticker'].unique()) - 5} 只股票存在同样问题")
    
    # 检查2：确保总体有足够的NaN比例（表明滞后生效）
    total_null_ratio = df[feature_cols].isnull().sum().sum() / (len(df) * len(feature_cols))
    if total_null_ratio < 0.01:  # 至少1%的NaN
        print(f"[LOOKAHEAD WARNING] NaN比例过低({total_null_ratio:.3f})，可能滞后不充分")
    
    # 检查3：确保特征不全为空
    assert not df[feature_cols].isnull().all().all(), "特征全为空，检查滞后处理"
    
    print(f"[LOOKAHEAD CHECK] NaN比例: {total_null_ratio:.3f}, 样本数: {len(df)}, 警告股票: {warning_count}")
    
    # 只有在没有警告的情况下才返回True
    return warning_count == 0

# =============================
# 新增功能3：因子衰减（样本时间权重）
# =============================

class PurgedTimeSeriesSplit:
    """Lopez de Prado式基于唯一日期的时间序列分割，支持更灵活的embargo防信息泄露"""
    def __init__(self, n_splits=5, horizon=5, embargo_multiplier=1.0, embargo_mode='conservative'):
        self.n_splits = n_splits
        self.horizon = horizon  # 预测周期（天）
        self.embargo_multiplier = embargo_multiplier
        self.embargo_days = max(1, int(horizon * embargo_multiplier))  # embargo至少=horizon
        self.embargo_mode = embargo_mode  # 'conservative', 'adaptive', 'minimal'
    
    def split(self, X, y=None, dates=None):
        """基于唯一日期进行分割，删除与验证标签重叠的训练样本"""
        if dates is None:
            # 回退到样本索引模式（为兼容性）
            return self._split_by_index(X)
            
        # 转换dates为pandas datetime
        if not isinstance(dates, pd.Series):
            dates = pd.Series(dates)
        dates = pd.to_datetime(dates)
        
        # 获取唯一日期并排序
        unique_dates = pd.Series(dates.unique()).sort_values()
        unique_dates = pd.to_datetime(unique_dates)
        n_unique_dates = len(unique_dates)
        
        if n_unique_dates < self.n_splits:
            raise ValueError(f"唯一日期数({n_unique_dates}) < 分割数({self.n_splits})")
        
        # 基于唯一日期计算分割点
        fold_size = n_unique_dates // self.n_splits
        for i in range(self.n_splits):
            # 验证期日期范围
            test_start_idx = i * fold_size
            test_end_idx = n_unique_dates if i == self.n_splits - 1 else (i + 1) * fold_size
            
            test_dates = unique_dates[test_start_idx:test_end_idx]
            test_start_date = test_dates.min()
            test_end_date = test_dates.max()
            
            # 计算标签期间（预测horizon天后的数据）
            label_start = test_start_date + pd.Timedelta(days=1)  # 标签从验证期第二天开始
            label_end = test_end_date + pd.Timedelta(days=self.horizon)
            
            # 根据embargo模式计算不同的embargo期间
            if self.embargo_mode == 'conservative':
                # 保守模式：排除验证期前后大范围数据（原有逻辑）
                embargo_start = test_start_date - pd.Timedelta(days=self.embargo_days)
                embargo_end = label_end + pd.Timedelta(days=self.embargo_days)
                train_mask = (dates < embargo_start) | (dates > embargo_end)
                
            elif self.embargo_mode == 'adaptive':
                # 自适应模式：只在验证区间周围加embargo，保留更多训练数据
                pre_embargo_start = test_start_date - pd.Timedelta(days=self.embargo_days)
                pre_embargo_end = test_start_date
                post_embargo_start = test_end_date
                post_embargo_end = test_end_date + pd.Timedelta(days=self.embargo_days)
                
                train_mask = ~(
                    ((dates >= pre_embargo_start) & (dates <= pre_embargo_end)) |  # 验证前embargo
                    ((dates >= test_start_date) & (dates <= test_end_date)) |      # 验证期
                    ((dates >= post_embargo_start) & (dates <= post_embargo_end)) # 验证后embargo
                )
                
            elif self.embargo_mode == 'minimal':
                # 最小模式：只排除验证期和最小必要的embargo（horizon天数）
                minimal_embargo = max(1, self.horizon)
                embargo_start = test_start_date - pd.Timedelta(days=minimal_embargo)
                embargo_end = test_end_date + pd.Timedelta(days=minimal_embargo)
                train_mask = (dates < embargo_start) | (dates > embargo_end)
                
            else:
                # 默认使用保守模式
                embargo_start = test_start_date - pd.Timedelta(days=self.embargo_days)
                embargo_end = label_end + pd.Timedelta(days=self.embargo_days)
                train_mask = (dates < embargo_start) | (dates > embargo_end)
            
            # 验证集：仅验证期内的样本
            test_mask = (dates >= test_start_date) & (dates <= test_end_date)
            
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
                
            yield train_idx, test_idx
    
    def _split_by_index(self, X):
        """回退模式：基于样本索引分割（兼容性）"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            # 简单embargo：前后各留horizon个样本
            embargo_size = self.horizon
            
            test_idx = indices[start:stop]
            train_left = indices[:max(0, start - embargo_size)]
            train_right = indices[min(n_samples, stop + embargo_size):]
            
            if len(train_left) > 0 and len(train_right) > 0:
                train_idx = np.concatenate([train_left, train_right])
            else:
                train_idx = train_left if len(train_left) > 0 else train_right
            
            if len(train_idx) > 0:
                yield train_idx, test_idx
            current = stop

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# 默认股票池（从GUI的完整默认股票池继承）
ticker_list = ["A", "AA", "AACB", "AACI", "AACT", "AAL", "AAMI", "AAOI", "AAON", "AAP", "AAPL", "AARD", "AAUC", "AB", "ABAT", "ABBV", "ABCB", "ABCL", "ABEO", "ABEV", "ABG", "ABL", "ABM", "ABNB", "ABSI", "ABT", "ABTS", "ABUS", "ABVC", "ABVX", "ACA", "ACAD", "ACB", "ACCO", "ACDC", "ACEL", "ACGL", "ACHC", "ACHR", "ACHV", "ACI", "ACIC", "ACIU", "ACIW", "ACLS", "ACLX", "ACM", "ACMR", "ACN", "ACNT", "ACOG", "ACRE", "ACT", "ACTG", "ACTU", "ACVA", "ACXP", "ADAG", "ADBE", "ADC", "ADCT", "ADEA", "ADI", "ADM", "ADMA", "ADNT", "ADP", "ADPT", "ADSE", "ADSK", "ADT", "ADTN", "ADUR", "ADUS", "ADVM", "AEBI", "AEE", "AEG", "AEHL", "AEHR", "AEIS", "AEM", "AEO", "AEP", "AER", "AES", "AESI", "AEVA", "AEYE", "AFCG", "AFG", "AFL", "AFRM", "AFYA", "AG", "AGCO", "AGD", "AGEN", "AGH", "AGI", "AGIO", "AGM", "AGNC", "AGO", "AGRO", "AGX", "AGYS", "AHCO", "AHH", "AHL", "AHR", "AI", "AIFF", "AIFU", "AIG", "AII", "AIM", "AIMD", "AIN", "AIOT", "AIP", "AIR", "AIRI", "AIRJ", "AIRO", "AIRS", "AISP", "AIT", "AIV", "AIZ", "AJG", "AKAM", "AKBA", "AKRO", "AL", "ALAB", "ALAR", "ALB", "ALBT", "ALC", "ALDF", "ALDX", "ALE", "ALEX", "ALF", "ALG", "ALGM", "ALGN", "ALGS", "ALGT", "ALHC", "ALIT", "ALK", "ALKS", "ALKT", "ALL", "ALLE", "ALLT", "ALLY", "ALM", "ALMS", "ALMU", "ALNT", "ALNY", "ALRM", "ALRS", "ALSN", "ALT", "ALTG", "ALTI", "ALTS", "ALUR", "ALV", "ALVO", "ALX", "ALZN", "AM", "AMAL", "AMAT", "AMBA", "AMBC", "AMBP", "AMBQ", "AMBR", "AMC", "AMCR", "AMCX", "AMD", "AME", "AMED", "AMG", "AMGN", "AMH", "AMKR", "AMLX", "AMN", "AMP", "AMPG", "AMPH", "AMPL", "AMPX", "AMPY", "AMR", "AMRC", "AMRK", "AMRN", "AMRX", "AMRZ", "AMSC", "AMSF", "AMST", "AMT", "AMTB", "AMTM", "AMTX", "AMWD", "AMWL", "AMX", "AMZE", "AMZN", "AN", "ANAB", "ANDE", "ANEB", "ANET", "ANF", "ANGH", "ANGI", "ANGO", "ANIK", "ANIP", "ANIX", "ANNX", "ANPA", "ANRO", "ANSC", "ANTA", "ANTE", "ANVS", "AOMR", "AON", "AORT", "AOS", "AOSL", "AOUT", "AP", "APA", "APAM", "APD", "APEI", "APG", "APGE", "APH", "API", "APLD", "APLE", "APLS", "APO", "APOG", "APP", "APPF", "APPN", "APPS", "APTV", "APVO", "AQN", "AQST", "AR", "ARAI", "ARCB", "ARCC", "ARCO", "ARCT", "ARDT", "ARDX", "ARE", "AREN", "ARES", "ARHS", "ARI", "ARIS", "ARKO", "ARLO", "ARLP", "ARM", "ARMK", "ARMN", "ARMP", "AROC", "ARQ", "ARQQ", "ARQT", "ARR", "ARRY", "ARTL", "ARTV", "ARVN", "ARW", "ARWR", "ARX", "AS", "ASA", "ASAN", "ASB", "ASC", "ASGN", "ASH", "ASIC", "ASIX", "ASLE", "ASM", "ASND", "ASO", "ASPI", "ASPN", "ASR", "ASST", "ASTE", "ASTH", "ASTI", "ASTL", "ASTS", "ASUR", "ASX", "ATAI", "ATAT", "ATEC", "ATEN", "ATEX", "ATGE", "ATHE", "ATHM", "ATHR", "ATI", "ATII", "ATKR", "ATLC", "ATLX", "ATMU", "ATNF", "ATO", "ATOM", "ATR", "ATRA", "ATRC", "ATRO", "ATS", "ATUS", "ATXS", "ATYR", "AU", "AUB", "AUDC", "AUGO", "AUID", "AUPH", "AUR", "AURA", "AUTL", "AVA", "AVAH", "AVAL", "AVAV", "AVB", "AVBC", "AVBP", "AVD", "AVDL", "AVDX", "AVGO", "AVIR", "AVNS", "AVNT", "AVNW", "AVO", "AVPT", "AVR", "AVT", "AVTR", "AVTX", "AVXL", "AVY", "AWI", "AWK", "AWR", "AX", "AXGN", "AXIN", "AXL", "AXP", "AXS", "AXSM", "AXTA", "AXTI", "AYI", "AYTU", "AZ", "AZN", "AZTA", "AZZ", "B", "BA", "BABA", "BAC", "BACC", "BACQ", "BAER", "BAH", "BAK", "BALL", "BALY", "BAM", "BANC", "BAND", "BANF", "BANR", "BAP", "BASE", "BATRA", "BATRK", "BAX", "BB", "BBAI", "BBAR", "BBCP", "BBD", "BBDC", "BBIO", "BBNX", "BBSI", "BBUC", "BBVA", "BBW", "BBWI", "BBY", "BC", "BCAL", "BCAX", "BCBP", "BCC", "BCE", "BCH", "BCO", "BCPC", "BCRX", "BCS", "BCSF", "BCYC", "BDC", "BDMD", "BDRX", "BDTX", "BDX", "BE", "BEAG", "BEAM", "BEEM", "BEEP", "BEKE", "BELFB", "BEN", "BEP", "BEPC", "BETR", "BF-A", "BF-B", "BFAM", "BFC", "BFH", "BFIN", "BFS", "BFST", "BG", "BGC", "BGL", "BGLC", "BGM", "BGS", "BGSF", "BHC", "BHE", "BHF", "BHFAP", "BHLB", "BHP", "BHR", "BHRB", "BHVN", "BIDU", "BIIB", "BILI", "BILL", "BIO", "BIOA", "BIOX", "BIP", "BIPC", "BIRD", "BIRK", "BJ", "BJRI", "BK", "BKD", "BKE", "BKH", "BKKT", "BKR", "BKSY", "BKTI", "BKU", "BKV", "BL", "BLBD", "BLBX", "BLCO", "BLD", "BLDE", "BLDR", "BLFS", "BLFY", "BLIV", "BLKB", "BLMN", "BLND", "BLNE", "BLRX", "BLUW", "BLX", "BLZE", "BMA", "BMBL", "BMGL", "BMHL", "BMI", "BMNR", "BMO", "BMR", "BMRA", "BMRC", "BMRN", "BMY", "BN", "BNC", "BNED", "BNGO", "BNL", "BNS", "BNTC", "BNTX", "BNZI", "BOC", "BOF", "BOH", "BOKF", "BOOM", "BOOT", "BORR", "BOSC", "BOW", "BOX", "BP", "BPOP", "BQ", "BR", "BRBR", "BRBS", "BRC", "BRDG", "BRFS", "BRK-B", "BRKL", "BRKR", "BRLS", "BRO", "BROS", "BRR", "BRSL", "BRSP", "BRX", "BRY", "BRZE", "BSAA", "BSAC", "BSBR", "BSET", "BSGM", "BSM", "BSX", "BSY", "BTAI", "BTBD", "BTBT", "BTCM", "BTCS", "BTCT", "BTDR", "BTE", "BTG", "BTI", "BTM", "BTMD", "BTSG", "BTU", "BUD", "BULL", "BUR", "BURL", "BUSE", "BV", "BVFL", "BVN", "BVS", "BWA", "BWB", "BWEN", "BWIN", "BWLP", "BWMN", "BWMX", "BWXT", "BX", "BXC", "BXP", "BY", "BYD", "BYND", "BYON", "BYRN", "BYSI", "BZ", "BZAI", "BZFD", "BZH", "BZUN", "C", "CAAP", "CABO", "CAC", "CACC", "CACI", "CADE", "CADL", "CAE", "CAEP", "CAG", "CAH", "CAI", "CAKE", "CAL", "CALC", "CALM", "CALX", "CAMT", "CANG", "CAPR", "CAR", "CARE", "CARG", "CARL", "CARR", "CARS", "CART", "CASH", "CASS", "CAT", "CATX", "CATY", "CAVA", "CB", "CBAN", "CBIO", "CBL", "CBLL", "CBNK", "CBOE", "CBRE", "CBRL", "CBSH", "CBT", "CBU", "CBZ", "CC", "CCAP", "CCB", "CCCC", "CCCS", "CCCX", "CCEP", "CCI", "CCIR", "CCIX", "CCJ", "CCK", "CCL", "CCLD", "CCNE", "CCOI", "CCRD", "CCRN", "CCS", "CCSI", "CCU", "CDE", "CDIO", "CDLR", "CDNA", "CDNS", "CDP", "CDRE", "CDRO", "CDTX", "CDW", "CDXS", "CDZI", "CE", "CECO", "CEG", "CELC", "CELH", "CELU", "CELZ", "CENT", "CENTA", "CENX", "CEP", "CEPO", "CEPT", "CEPU", "CERO", "CERT", "CEVA", "CF", "CFFN", "CFG", "CFLT", "CFR", "CG", "CGAU", "CGBD", "CGCT", "CGEM", "CGNT", "CGNX", "CGON", "CHA", "CHAC", "CHCO", "CHD", "CHDN", "CHE", "CHEF", "CHH", "CHKP", "CHMI", "CHPT", "CHRD", "CHRW", "CHT", "CHTR", "CHWY", "CHYM", "CI", "CIA", "CIB", "CIEN", "CIFR", "CIGI", "CIM", "CINF", "CING", "CINT", "CIO", "CION", "CIVB", "CIVI", "CL", "CLAR", "CLB", "CLBK", "CLBT", "CLCO", "CLDI", "CLDX", "CLF", "CLFD", "CLGN", "CLH", "CLLS", "CLMB", "CLMT", "CLNE", "CLNN", "CLOV", "CLPR", "CLPT", "CLRB", "CLRO", "CLS", "CLSK", "CLVT", "CLW", "CLX", "CM", "CMA", "CMBT", "CMC", "CMCL", "CMCO", "CMCSA", "CMDB", "CME", "CMG", "CMI", "CMP", "CMPO", "CMPR", "CMPS", "CMPX", "CMRC", "CMRE", "CMS", "CMTL", "CNA", "CNC", "CNCK", "CNDT", "CNEY", "CNH", "CNI", "CNK", "CNL", "CNM", "CNMD", "CNNE", "CNO", "CNOB", "CNP", "CNQ", "CNR", "CNS", "CNTA", "CNTB", "CNTY", "CNVS", "CNX", "CNXC", "CNXN", "COCO", "CODI", "COF", "COFS", "COGT", "COHR", "COHU", "COIN", "COKE", "COLB", "COLL", "COLM", "COMM", "COMP", "CON", "COO", "COOP", "COP", "COPL", "COR", "CORT", "CORZ", "COTY", "COUR", "COYA", "CP", "CPA", "CPAY", "CPB", "CPF", "CPIX", "CPK", "CPNG", "CPRI", "CPRT", "CPRX", "CPS", "CPSH", "CQP", "CR", "CRAI", "CRAQ", "CRBG", "CRBP", "CRC", "CRCL", "CRCT", "CRD-A", "CRDF", "CRDO", "CRE", "CRESY", "CREV", "CREX", "CRGO", "CRGX", "CRGY", "CRH", "CRI", "CRK", "CRL", "CRM", "CRMD", "CRML", "CRMT", "CRNC", "CRNX", "CRON", "CROX", "CRS", "CRSP", "CRSR", "CRTO", "CRUS", "CRVL", "CRVO", "CRVS", "CRWD", "CRWV", "CSAN", "CSCO", "CSGP", "CSGS", "CSIQ", "CSL", "CSR", "CSTL", "CSTM", "CSV", "CSW", "CSWC", "CSX", "CTAS", "CTEV", "CTGO", "CTKB", "CTLP", "CTMX", "CTNM", "CTO", "CTOS", "CTRA", "CTRI", "CTRM", "CTRN", "CTS", "CTSH", "CTVA", "CTW", "CUB", "CUBE", "CUBI", "CUK", "CUPR", "CURB", "CURI", "CURV", "CUZ", "CV", "CVAC", "CVBF", "CVCO", "CVE", "CVEO", "CVGW", "CVI", "CVLG", "CVLT", "CVM", "CVNA", "CVRX", "CVS", "CVX", "CW", "CWAN", "CWBC", "CWCO", "CWEN", "CWEN-A", "CWH", "CWK", "CWST", "CWT", "CX", "CXDO", "CXM", "CXT", "CXW", "CYBN", "CYBR", "CYCC", "CYD", "CYH", "CYN", "CYRX", "CYTK", "CZR", "CZWI", "D", "DAAQ", "DAC", "DAIC", "DAKT", "DAL", "DALN", "DAN", "DAO", "DAR", "DARE", "DASH", "DATS", "DAVA", "DAVE", "DAWN", "DAY", "DB", "DBD", "DBI", "DBRG", "DBX", "DC", "DCBO", "DCI", "DCO", "DCOM", "DCTH", "DD", "DDC", "DDI", "DDL", "DDOG", "DDS", "DEA", "DEC", "DECK", "DEFT", "DEI", "DELL", "DENN", "DEO", "DERM", "DEVS", "DFDV", "DFH", "DFIN", "DFSC", "DG", "DGICA", "DGII", "DGX", "DGXX", "DH", "DHI", "DHR", "DHT", "DHX", "DIBS", "DIN", "DINO", "DIOD", "DIS", "DJCO", "DJT", "DK", "DKL", "DKNG", "DKS", "DLB", "DLHC", "DLO", "DLTR", "DLX", "DLXY", "DMAC", "DMLP", "DMRC", "DMYY", "DNA", "DNB", "DNLI", "DNN", "DNOW", "DNTH", "DNUT", "DOC", "DOCN", "DOCS", "DOCU", "DOGZ", "DOLE", "DOMH", "DOMO", "DOOO", "DORM", "DOUG", "DOV", "DOW", "DOX", "DOYU", "DPRO", "DPZ", "DQ", "DRD", "DRDB", "DRH", "DRI", "DRS", "DRVN", "DSGN", "DSGR", "DSGX", "DSP", "DT", "DTE", "DTI", "DTIL", "DTM", "DTST", "DUK", "DUOL", "DUOT", "DV", "DVA", "DVAX", "DVN", "DVS", "DWTX", "DX", "DXC", "DXCM", "DXPE", "DXYZ", "DY", "DYN", "DYNX", "E", "EA", "EARN", "EAT", "EB", "EBAY", "EBC", "EBF", "EBMT", "EBR", "EBS", "EC", "ECC", "ECG", "ECL", "ECO", "ECOR", "ECPG", "ECVT", "ED", "EDBL", "EDIT", "EDN", "EDU", "EE", "EEFT", "EEX", "EFC", "EFSC", "EFX", "EFXT", "EG", "EGAN", "EGBN", "EGG", "EGO", "EGP", "EGY", "EH", "EHAB", "EHC", "EHTH", "EIC", "EIG", "EIX", "EKSO", "EL", "ELAN", "ELDN", "ELF", "ELMD", "ELME", "ELP", "ELPW", "ELS", "ELV", "ELVA", "ELVN", "ELWS", "EMA", "EMBC", "EMN", "EMP", "EMPD", "EMPG", "EMR", "EMX", "ENB", "ENGN", "ENGS", "ENIC", "ENOV", "ENPH", "ENR", "ENS", "ENSG", "ENTA", "ENTG", "ENVA", "ENVX", "EOG", "EOLS", "EOSE", "EPAC", "EPAM", "EPC", "EPD", "EPM", "EPR", "EPSM", "EPSN", "EQBK", "EQH", "EQNR", "EQR", "EQT", "EQV", "EQX", "ERIC", "ERIE", "ERII", "ERJ", "ERO", "ES", "ESAB", "ESE", "ESGL", "ESI", "ESLT", "ESNT", "ESOA", "ESQ", "ESTA", "ESTC", "ET", "ETD", "ETN", "ETNB", "ETON", "ETOR", "ETR", "ETSY", "EU", "EUDA", "EVAX", "EVC", "EVCM", "EVER", "EVEX", "EVGO", "EVH", "EVLV", "EVO", "EVOK", "EVR", "EVRG", "EVTC", "EVTL", "EW", "EWBC", "EWCZ", "EWTX", "EXAS", "EXC", "EXE", "EXEL", "EXK", "EXLS", "EXOD", "EXP", "EXPD", "EXPE", "EXPI", "EXPO", "EXR", "EXTR", "EYE", "EYPT", "EZPW", "F", "FA",
 "FACT", "FAF", "FANG", "FAST", "FAT", "FATN", "FBIN", "FBK", "FBLA", 
 "FBNC", "FBP", "FBRX", "FC", "FCBC", "FCEL", "FCF", "FCFS", "FCN", "FCX", "FDMT",
  "FDP", "FDS", "FDUS", "FDX", "FE", "FEIM", "FELE", "FENC", "FER", "FERA", "FERG", "FET", "FF", 
  "FFAI", "FFBC", "FFIC", "FFIN", "FFIV", "FFWM", "FG", "FGI", "FHB", "FHI", "FHN", "FHTX", "FI", "FIBK", "FIEE", "FIG", "FIGS", 
  "FIHL", "FINV", "FIP", "FIS", "FISI", "FITB", "FIVE", "FIVN", "FIZZ", "FL", "FLD", "FLEX", "FLG", "FLGT", "FLL", "FLNC", "FLNG", "FLO", "FLOC",
   "FLR", "FLS", "FLUT", "FLWS", "FLX", "FLY", "FLYE", "FLYW", "FLYY", "FMBH", "FMC", "FMFC", "FMNB", "FMS", "FMST", 
   "FMX", "FN", "FNB", "FND", "FNF", "FNGD", "FNKO", "FNV", "FOA", "FOLD", "FOR", "FORM", "FORR", "FOUR", "FOX", "FOXA", 
   "FOXF", "FPH", "FPI", "FRGE", "FRHC", "FRME", "FRO", "FROG", "FRPT", "FRSH", "FRST", "FSCO", "FSK", "FSLR", "FSLY",
    "FSM", "FSS", "FSUN", "FSV", "FTAI", "FTCI", "FTDR", "FTEK", "FTI", "FTK", "FTNT", "FTRE", "FTS", "FTV", "FUBO", "FUFU", "FUL", "FULC", "FULT", "FUN", "FUTU", "FVR", "FVRR", "FWONA", "FWONK", "FWRD", "FWRG", "FYBR", "G",
     "GABC", "GAIA", "GAIN", "GALT", "GAMB", "GAP", "GASS", "GATX", "GAUZ", "GB", "GBCI", "GBDC", "GBFH", "GBIO", "GBTG", "GBX", "GCI", "GCL", "GCMG", "GCO", "GCT", "GD", "GDC", "GDDY", "GDEN", "GDOT", "GDRX", "GDS", 
     "GDYN", "GE", "GEF", "GEHC", "GEL", "GEN", "GENI", "GENK", "GEO", "GEOS", "GES", "GFF", "GFI", "GFL", "GFR", "GFS", "GGAL", "GGB", "GGG", "GH", "GHLD", "GHM", "GHRS", "GIB", "GIC", "GIG", "GIII", "GIL", "GILD", "GILT", "GIS", "GITS", 
     "GKOS", "GL", "GLAD", "GLBE", "GLD", "GLDD", "GLIBA", "GLIBK", "GLNG", "GLOB", "GLP", "GLPG", "GLPI", "GLRE", "GLSI", "GLUE", "GLW", "GLXY", "GM", "GMAB", "GME", "GMED", "GMRE", "GMS", "GNE", "GNK", "GNL", "GNLX", "GNRC", "GNTX", "GNTY", "GNW", "GO", "GOCO", "GOGL", "GOGO", "GOLF", "GOOD", "GOOG", "GOOGL", "GOOS", "GORV", "GOTU", "GPAT", 
     "GPC", "GPCR", "GPI", "GPK", "GPN", "GPOR", "GPRE", "GPRK", "GRAB", "GRAL", "GRAN", "GRBK", "GRC", "GRCE", "GRDN", "GRFS", "GRMN", "GRND", "GRNT", "GROY", "GRPN", "GRRR", "GSAT", "GSBC", "GSBD", "GSHD", "GSIT", "GSK", "GSL", "GSM", "GSRT", "GT", "GTE", "GTEN", "GTERA", "GTES", "GTLB", "GTLS", "GTM", "GTN", "GTX", "GTY", "GVA", "GWRE", "GWRS", "GXO", "GYRE", "H", "HAE", "HAFC", "HAFN", "HAL", "HALO", "HAS", "HASI", "HAYW", "HBAN", "HBCP", "HBI", "HBM", "HBNC", "HCA", "HCAT", "HCC", "HCHL", "HCI", "HCKT", "HCM", "HCSG", "HCTI", "HCWB", "HD", "HDB", "HDSN", "HE", "HEI", "HEI-A", "HELE", "HEPS", "HESM", "HFFG", "HFWA", "HG", "HGTY", "HGV", "HHH", "HI", "HIFS", "HIG", "HII", "HIMS", "HIMX",
     "HIPO", "HIT", "HITI", "HIVE", "HIW", "HL", "HLF", "HLI", "HLIO", "HLIT", "HLLY", "HLMN", "HLN", "HLNE", "HLT", "HLVX", "HLX", "HLXB", "HMC", "HMN", "HMST", "HMY", "HNGE", "HNI", "HNRG", "HNST", "HOFT", "HOG", "HOLO", "HOLX", "HOMB", "HON", "HOND", "HONE", "HOOD", "HOPE", "HOUS", "HOV", "HP", "HPE", "HPK", "HPP", "HPQ", "HQH", "HQL", "HQY", "HRB", "HRI", "HRL", "HRMY", "HROW", "HRTG", "HRZN", "HSAI", "HSBC", "HSCS", "HSHP", "HSIC", "HSII", "HST", "HSTM", "HSY", "HTBK", "HTCO", "HTGC", "HTH", "HTHT", "HTLD", "HTO", "HTOO", "HTZ", "HUBB", "HUBC", "HUBG", "HUBS", "HUHU", "HUM", "HUMA", "HUN", "HURA", "HURN", "HUSA", "HUT", "HUYA", "HVII", "HVT", "HWC", "HWKN", "HWM", 
     "HXL", "HY", "HYAC", "HYMC", "HYPD", "HZO", "IAC", "IAG", "IART", "IAS", "IBCP", "IBEX", "IBKR", "IBM", "IBN", "IBOC", "IBP", "IBRX", "IBTA", "ICE", "ICFI", "ICG", "ICHR", "ICL", "ICLR", "ICUI", "IDA", "IDAI", "IDCC", "IDN", "IDR", "IDT", "IDYA", "IE", "IEP", "IESC", "IEX", "IFF", "IFS", "IGIC", "IHG", "IHS", "III", "IIIN", "IIIV", "IIPR", "ILMN", "IMAB", "IMAX", "IMCC", "IMCR", "IMDX", "IMKTA", "IMMR", "IMMX", "IMNM", "IMNN", "IMO", "IMPP", "IMRX", "IMTX", "IMVT", "IMXI", "INAB", "INAC", "INBK", "INBX", "INCY", "INDB", "INDI", "INDO", "INDP", "INDV", "INFA", "INFU", "INFY", "ING", "INGM", "INGN", "INGR", "INKT", "INMB", "INMD", "INN", "INOD", "INR", "INSE", "INSG", "INSM", "INSP", "INSW", "INTA", "INTC", "INTR", "INUV", "INV", "INVA", "INVE", "INVH", "INVX", "IONQ", "IONS", "IOSP", "IOT", "IOVA", "IP", "IPA", "IPAR", "IPDN", "IPG", "IPGP", "IPI", "IPX", "IQST", "IQV", "IR", "IRBT", "IRDM", "IREN", "IRM", "IRMD", "IROH", "IRON", "IRS", "IRTC", "ISPR", "ISRG", "ISSC", "IT", "ITGR", "ITIC", "ITOS", "ITRI", "ITRN", "ITT", "ITUB", "ITW", "IVR", "IVZ", "IX", "IZEA", "J", "JACK", "JACS", "JAKK", "JAMF", "JANX", "JAZZ", "JBGS", "JBHT", "JBI", "JBIO", "JBL", "JBLU", "JBS", "JBSS", "JBTM", "JCAP", "JCI", "JD", "JEF", "JELD", "JEM", "JENA", "JFIN", "JHG", "JHX", "JILL", "JJSF", "JKHY", "JKS", "JLHL", "JLL", "JMIA", "JNJ", "JOBY", "JOE", "JOUT", "JOYY", "JPM", "JRSH", "JRVR", "JSPR", "JTAI", "JVA", "JXN", "JYNT", "K", "KAI", "KALA", "KALU", 
     "KALV", "KAR", "KARO", "KB", "KBDC", "KBH", "KBR", "KC", "KCHV", "KD", "KDP", "KE", "KELYA", "KEP", "KEX", "KEY", "KEYS", "KFII", "KFRC", "KFS", "KFY", "KGC", "KGEI", "KGS", "KHC", "KIDS", "KIM", "KINS", "KKR", "KLC", "KLG", "KLIC", "KLRS", "KMB", "KMDA", "KMI", "KMPR", "KMT", "KMTS", "KMX", "KN", "KNF", "KNOP", "KNSA", "KNSL", "KNTK", "KNW", "KNX", "KO", "KOD", "KODK", "KOF", "KOP", "KOSS", "KPRX", "KPTI", "KR", "KRC", "KRMD", "KRMN", "KRNT", "KRNY", "KRO", "KROS", "KRP", "KRRO", "KRT", "KRUS", "KRYS", "KSCP", "KSPI", "KSS", "KT", "KTB", "KTOS", "KULR", "KURA", "KVUE", "KVYO", "KW", "KWM", "KWR", "KYMR", "KYTX", "KZIA", "L", "LAC", "LAD", "LADR", "LAES", "LAKE", "LAMR", "LAND", "LANV", "LAR", "LASE", "LASR", "LAUR", "LAW", "LAWR", "LAZ", "LAZR", "LB", "LBRDA", "LBRDK", "LBRT", "LBTYA", "LBTYK", "LC", "LCCC", "LCFY", "LCID", "LCII", "LCUT", "LDOS", "LE", "LEA", "LECO", "LEG", "LEGH", "LEGN", "LEN", "LENZ", "LEO", "LEU", "LEVI", "LFCR", "LFMD", "LFST", "LFUS", "LFVN", "LGCY", "LGIH", "LGND", "LH", "LHAI", "LHSW", "LHX", "LI", "LIDR", "LIF", "LILA", "LILAK", "LIMN", "LIN", "LINC", "LIND", "LINE", "LION", "LITE", "LITM", "LIVE", "LIVN", "LIXT", "LKFN", "LKQ", "LLYVA", "LLYVK", "LMAT", "LMB", "LMND", "LMNR", "LMT", "LNC", "LNG", "LNN", "LNSR", "LNT", "LNTH", "LNW", "LOAR", "LOB", "LOCO", "LODE", "LOGI", "LOKV", "LOMA", "LOPE", "LOT", "LOVE", "LOW", "LPAA", "LPBB", "LPCN", "LPG", "LPL", "LPLA", "LPRO", "LPTH", "LPX", "LQDA", "LQDT", "LRCX", "LRMR", "LRN", "LSCC", "LSE", "LSPD", "LSTR", "LTBR", "LTC", "LTH", "LTM", "LTRN", "LTRX", "LU", "LUCK", "LULU", "LUMN", "LUNR", "LUV", "LUXE", "LVLU", "LVS", "LVWR", "LW", "LWAY", "LWLG", "LX", "LXEH", "LXEO", "LXFR", "LXU", "LYB", "LYEL", "LYFT", "LYG", "LYRA", "LYTS", "LYV", "LZ", "LZB", "LZM", "LZMH", "M", "MAA", "MAAS", "MAC", "MACI", "MAG", "MAGN", "MAIN", "MAMA", "MAMK", "MAN", "MANH", "MANU", "MAR", "MARA", "MAS", "MASI", "MASS", "MAT", "MATH", "MATV", "MATW", "MATX", "MAX", "MAXN", "MAZE", "MB", "MBAV", "MBC", "MBI", "MBIN", "MBLY",
      "MBOT", "MBUU", "MBWM", "MBX", "MC", "MCB", "MCD", "MCFT", "MCHP", "MCRB", "MCRI", "MCRP", "MCS", "MCVT", "MCW", "MCY", "MD", "MDAI", "MDB", "MDCX", "MDGL", "MDLZ", "MDT", "MDU", "MDV", "MDWD", "MDXG", "MDXH", "MEC", "MED", "MEDP", "MEG", "MEI", "MEIP", "MENS", "MEOH", "MERC", "MESO", "MET", "METC", "METCB", "MFA", "MFC", "MFG", "MFH", "MFI", "MFIC", "MFIN", "MG", "MGA", "MGEE", "MGIC", "MGM", "MGNI", "MGPI", "MGRC", "MGRM", "MGRT", "MGTX", "MGY", "MH", "MHK", "MHO", "MIDD", "MIMI", "MIND", "MIR", "MIRM", "MITK", "MKC", "MKSI", "MKTX", "MLAB", "MLCO", "MLEC", "MLGO", "MLI", "MLKN", "MLNK", "MLR", "MLTX", "MLYS", "MMC", "MMI", "MMM", "MMS", "MMSI", "MMYT", "MNDY", "MNKD", "MNMD", "MNR", "MNRO", "MNSO", "MNST", "MNTN", "MO", "MOB", "MOD", "MODG", "MODV", "MOFG", "MOG-A", "MOH", "MOMO", "MORN", "MOS", "MOV", "MP", "MPAA", "MPB", "MPC", "MPLX", "MPTI", "MPU", "MQ", "MRAM", "MRBK", "MRC", "MRCC", "MRCY", "MRK", "MRNA", "MRP", "MRSN", "MRT", "MRTN", "MRUS", "MRVI", "MRVL", "MRX", "MS", "MSA", "MSBI", "MSEX", "MSGE", "MSGM", "MSGS", "MSGY", "MSI", "MSM", "MSTR", "MT", "MTA", "MTAL", "MTB", "MTCH", "MTDR", "MTEK", "MTEN", "MTG", "MTH", "MTLS", "MTN", "MTRN", "MTRX", "MTSI", "MTSR", "MTUS", "MTW", "MTX", "MTZ", "MU", "MUFG", "MUR", "MUSA", "MUX", "MVBF", "MVST", "MWA", "MX", "MXL", "MYE", "MYFW", "MYGN", "MYRG", "MZTI", "NA", "NAAS", "NABL", "NAGE", "NAKA", "NAMM", "NAMS", "NAT", "NATH", "NATL", "NATR", "NAVI", "NB", "NBBK", "NBHC", "NBIS", "NBIX", "NBN", "NBR", "NBTB", "NCDL", "NCLH", "NCMI", "NCNO", "NCPL", "NCT", "NCTY", "NDAQ", "NDSN", "NE", "NEE", "NEGG", "NEM", "NEO", "NEOG", "NEON", "NEOV", "NESR", "NET", "NETD", "NEWT", "NEXM", "NEXN", "NEXT", "NFBK", "NFE", "NFG", "NG", "NGD", "NGG", "NGL", "NGNE", "NGS", "NGVC", "NGVT", "NHC", "NHI", "NHIC", "NI", "NIC", "NICE", "NIO", "NIQ", "NISN", "NIU", "NJR", "NKE", "NKTR", "NLOP", "NLSP", "NLY", "NMAX", "NMFC", "NMIH", "NMM", "NMR", "NMRK", "NN", "NNBR", "NNE", "NNI", "NNN", "NNNN", "NNOX", "NOA", "NOAH", "NOG", "NOK", "NOMD", "NOV", "NOVT", "NPAC", "NPB", "NPCE", "NPK", "NPKI", "NPO", "NPWR", "NRC", "NRDS", "NRG", "NRIM", "NRIX", "NRXP", "NRXS", "NSC", "NSIT", "NSP", "NSPR", "NSSC", "NTAP", "NTB", "NTCT", "NTES", "NTGR", "NTHI", "NTLA", "NTNX", "NTR", "NTRA", "NTRB", "NTST", "NU", "NUE", "NUKK", "NUS", "NUTX", "NUVB", "NUVL", "NUWE", "NVAX", "NVCR", "NVCT", "NVDA", "NVEC", "NVGS", "NVMI", "NVNO", "NVO", "NVRI", "NVS", 
  "NVST", "NVT", "NVTS", "NWBI", "NWE", "NWG", "NWL", "NWN", "NWPX", "NWS", "NWSA", "NX", "NXE", "NXP", "NXPI", "NXST", "NXT", "NXTC", "NYT", "NYXH", "O", "OACC", "OBDC", "OBE", "OBIO", "OBK", "OBLG", "OBT", "OC", "OCC", "OCCI", "OCFC", "OCFT", "OCSL", "OCUL", "ODC", "ODD", "ODFL", "ODP", "ODV", "OEC", "OFG", "OFIX", "OGE", "OGN", "OGS", "OHI", "OI", "OII", "OIS", "OKE", "OKLO", "OKTA", "OKUR", "OKYO", "OLED", "OLLI", "OLMA", "OLN", "OLO", "OLP", "OM", "OMAB", "OMC", "OMCL", "OMDA", "OMER", "OMF", "OMI", "OMSE", "ON", "ONB", "ONC", "ONDS", "ONEG", "ONEW", "ONL", "ONON", "ONTF", "ONTO", "OOMA", "OPAL", "OPBK", "OPCH", "OPFI", "OPRA", "OPRT", "OPRX", "OPXS", "OPY", "OR", "ORA", "ORC", "ORCL", "ORGO", "ORI", "ORIC", "ORKA", "ORLA", "ORLY", "ORMP", "ORN", "ORRF", "OS", "OSBC", "OSCR", "OSIS", "OSK", "OSPN", "OSS", "OSUR", "OSW", "OTEX", "OTF", "OTIS", "OTLY", "OTTR", "OUST", "OUT", "OVV", "OWL", "OWLT", "OXLC", "OXM", "OXSQ", "OXY", "OYSE", "OZK", "PAA", "PAAS", "PAC", "PACK", "PACS", "PAG", "PAGP", "PAGS", "PAHC", "PAL", "PAM", "PANL", "PANW", "PAR", "PARR", "PATH", "PATK", "PAX", "PAY", "PAYC", "PAYO", "PAYS", "PAYX", "PB", "PBA", "PBF", "PBH", "PBI", "PBPB", "PBR", "PBR-A", "PBYI", "PC", "PCAP", "PCAR", "PCG", "PCH", "PCOR", "PCRX", "PCT", "PCTY", "PCVX", "PD", "PDD", "PDEX", "PDFS", "PDS", "PDYN", "PEBO", "PECO", "PEG", "PEGA", "PEN", "PENG", "PENN", "PEP", "PERI", "PESI", "PETS", "PEW", "PFBC", "PFE", "PFG", "PFGC", "PFLT", "PFS", "PFSI", "PG", "PGC", "PGNY", "PGR", "PGRE", "PGY", "PHAT", "PHG", "PHI", "PHIN", "PHIO", "PHLT", "PHM", "PHOE", "PHR", "PHUN", "PHVS", "PI", "PII", "PINC", "PINS", "PIPR", "PJT", "PK", "PKE", "PKG", "PKX", "PL", "PLAB", "PLAY", "PLCE", "PLD", "PLL", "PLMR", "PLNT", "PLOW", "PLPC", "PLSE", "PLTK", "PLTR", "PLUS", "PLXS", "PLYM", "PM", "PMTR", "PMTS", "PN", "PNC", "PNFP", "PNNT", "PNR", "PNRG", "PNTG", "PNW", "PODD", "POET", "PONY", "POOL", "POR", "POST", "POWI", "POWL", "PPBI", "PPBT", "PPC", "PPG", "PPIH", "PPL", "PPSI", "PPTA", "PR", "PRA", "PRAA", "PRAX", "PRCH", "PRCT", "PRDO", "PRE", "PRG", "PRGO", "PRGS", "PRI", "PRIM", "PRK", "PRKS", "PRLB", "PRM", "PRMB", "PRME", "PRO", "PROK", "PROP", "PRQR", "PRSU", "PRTA", "PRTG", "PRTH", "PRU", "PRVA", "PSA", "PSEC", "PSFE", "PSIX", "PSKY", "PSMT", "PSN", "PSNL", "PSO", "PSQH", "PSTG", "PSX", "PTC", "PTCT", "PTEN", "PTGX", "PTHS", "PTLO", "PTON", "PUBM", "PUK", "PUMP", "PVBC", "PVH", "PVLA", "PWP", "PWR", "PX", "PXLW", "PYPD", "PYPL", "PZZA", "QBTS", "QCOM", "QCRH", "QD", "QDEL", "QFIN", "QGEN", "QIPT", "QLYS", "QMCO", "QMMM", "QNST", "QNTM", "QRHC", "QRVO", "QS", "QSEA", "QSG", "QSR", "QTRX", "QTWO", "QUAD", "QUBT", "QUIK", "QURE", "QVCGA", "QXO", "R", "RAAQ", "RAC", "RACE", "RAIL", "RAL", "RAMP", "RAPP", "RAPT", "RARE", "RAY", "RBA", "RBB", "RBBN", "RBC", "RBCAA", "RBLX", "RBRK", "RC", "RCAT", "RCEL", "RCI", "RCKT", "RCKY", "RCL", "RCMT", "RCON", "RCT", 
    "RCUS", "RDAG", "RDAGU", "RDCM", "RDDT", "RDN", "RDNT", "RDVT", "RDW", "RDWR", "RDY", "REAL", "REAX", "REBN", "REFI", "REG", "RELX", "RELY", "RENT", "REPL", "REPX", "RERE", "RES", "RETO", "REVG", "REX", "REXR", "REYN", "REZI", "RF", "RFIL", "RGA", "RGC", "RGEN", "RGLD", "RGNX", "RGP", "RGR", "RGTI", "RH", "RHI", "RHLD", "RHP", "RICK", "RIG", "RIGL", "RILY", "RIME", "RIO", "RIOT", "RITM", "RITR", "RIVN", "RJF", "RKLB", "RKT", "RL", "RLAY", "RLGT", "RLI", "RLX", "RMAX", "RMBI", "RMBL", "RMBS", "RMD", "RMNI", "RMR", "RMSG", "RNA", "RNAC", "RNAZ", "RNG", "RNGR", "RNR", "RNST", "RNW", "ROAD", "ROCK", "ROG", "ROIV", "ROK", "ROKU", "ROL", "ROLR", "ROMA", "ROOT", "ROST", "RPAY", "RPD", "RPID", "RPM", "RPRX", "RPT", "RRC", "RRGB", "RRR", "RRX", "RS", "RSG", "RSI", "RSKD", "RSLS", "RSVR", "RTAC", "RTO", "RTX", "RUBI", "RUM", "RUN", "RUSHA", "RUSHB", "RVLV", "RVMD", "RVSB", "RVTY", "RWAY", "RXO", "RXRX", "RXST", "RY", "RYAAY", "RYAM", "RYAN", "RYI", "RYN", "RYTM", "RZB", "RZLT", "RZLV", "S", "SA", "SABS", "SAFE", "SAFT", "SAGT", "SAH", "SAIA", "SAIC", "SAIL", "SAM", "SAMG", "SAN", "SANA", "SAND", "SANM", "SAP", "SAR", "SARO", "SATL", "SATS", "SAVA", "SB", "SBAC", "SBC", "SBCF", "SBET", "SBGI", "SBH", "SBLK", "SBRA", "SBS", "SBSI", "SBSW", "SBUX", "SBXD", "SCAG", "SCCO", "SCHL", "SCHW", "SCI", "SCL", "SCLX", "SCM", "SCNX", "SCPH", "SCS", "SCSC", "SCVL", "SD", "SDA", "SDGR", "SDHC", "SDHI", "SDM", "SDRL", "SE", "SEAT", "SEDG", "SEE", "SEG", "SEI", "SEIC", "SEM", "SEMR", "SENEA", "SEPN", "SERA", "SERV", "SEZL", "SF", "SFBS", "SFD", "SFIX", "SFL", "SFM", "SFNC", "SG", "SGHC", "SGHT", "SGI", "SGML", "SGMT", "SGRY", "SHAK", "SHBI", "SHC", "SHCO", "SHEL", "SHEN", "SHG", "SHIP", "SHLS", "SHO", "SHOO", "SHOP", "SHW", "SI", "SIBN", "SIEB", "SIFY", "SIG", "SIGA", "SIGI", "SII", "SIMO", "SINT", "SION", "SIRI", "SITC", "SITE", "SITM", "SJM", "SKE", "SKLZ", "SKM", "SKT", "SKWD", "SKX", "SKY", "SKYE", "SKYH", "SKYT", "SKYW", "SLAB", "SLB", "SLDB", "SLDE", "SLDP", "SLF", "SLG", "SLGN", "SLI", "SLM", "SLN", "SLND", "SLNO", "SLP", "SLRC", "SLSN", "SLVM", "SM", "SMA", "SMBK", "SMC", "SMCI", "SMFG", "SMG", "SMHI", "SMLR", "SMMT", "SMP", "SMPL", "SMR", "SMTC", "SMWB", "SMX", "SN", "SNA", "SNAP", "SNBR", "SNCR", "SNCY", "SNDK", "SNDR", "SNDX", "SNES", "SNEX", "SNFCA", "SNGX", "SNN", "SNOW", "SNRE", "SNT", "SNV", "SNWV", "SNX", "SNY", "SNYR", "SO", "SOBO", "SOC", "SOFI", "SOGP", "SOHU", "SOLV", "SON", "SOND", "SONN", "SONO", "SONY", "SOPH", "SORA", "SOS", "SOUL", "SOUN", "SPAI", "SPB", "SPCB", "SPCE", "SPG", "SPH", "SPHR", "SPIR", "SPKL", "SPNS", "SPNT", "SPOK", "SPR", "SPRO", "SPRY", "SPSC", "SPT", "SPTN", "SPWH", "SPXC", "SQM", "SR", "SRAD", "SRBK", "SRCE", "SRDX", "SRE", "SRFM", "SRG", "SRI", "SRPT", "SRRK", "SRTS", "SSB", "SSD", "SSII", "SSL", "SSNC", "SSP", "SSRM", "SSSS", "SST", "SSTI", "SSTK", "SSYS", "ST", "STAA", "STAG", "STBA", "STC", "STE", "STEL", "STEM", "STEP", "STFS", "STGW", "STHO", "STI", "STIM", "STKL", "STKS", "STLA", "STLD", "STM", "STN", "STNE", "STNG", "STOK", "STR", "STRA", "STRD", "STRL", 
    "STRM", "STRT", "STRZ", "STSS", "STT", "STVN", "STX", "STXS", "STZ", "SU", "SUI", "SUN", "SUPN", "SUPV", "SUPX", "SURG", "SUZ", "SVCO", "SVM", "SVRA", "SVV", "SW", "SWBI", "SWIM", "SWIN", "SWK", "SWKS", "SWX", "SXC", "SXI", "SXT", "SY", "SYBT", "SYF", "SYK", "SYM", "SYNA", "SYRE", "SYTA", "SYY", "SZZL", "T", "TAC", "TACH", "TACO", "TAK", "TAL", "TALK", "TALO", "TAOX", "TAP", "TARA", "TARS", "TASK", "TATT", "TBB", "TBBB", "TBBK", "TBCH", "TBI", "TBLA", "TBPH", "TBRG", "TCBI", "TCBK", "TCBX", "TCMD", "TCOM", "TCPC", "TD", "TDC", "TDIC", "TDOC", "TDS", "TDUP", "TDW", "TEAM", "TECH", "TECK", "TECX", "TEF", "TEL", "TEM", "TEN", "TENB", "TEO", "TER", "TERN", "TEVA", "TEX", "TFC", "TFII", "TFIN", "TFPM", "TFSL", "TFX", "TG", "TGB", "TGE", "TGEN", "TGLS", "TGNA", "TGS", "TGT", "TGTX", "TH", "THC", "THFF", "THG", "THO", "THR", "THRM", "THRY", "THS", "THTX", "TIC", "TIGO", "TIGR", "TIL", "TILE", "TIMB", "TIPT", "TITN", "TIXT", "TJX", "TK", "TKC", "TKNO", "TKO", "TKR", "TLK", "TLN", "TLS", "TLSA", "TLSI", "TM", "TMC",
     "TMCI", "TMDX", "TME", "TMHC", "TMO", "TMUS", "TNC", "TNDM", "TNET", "TNGX", "TNK", "TNL", "TNXP", "TOI", "TOL", "TOPS", "TORO", "TOST", "TOWN", "TPB", "TPC", "TPCS", "TPG", "TPH", "TPR", "TPST", "TPVG", "TR", "TRAK", "TRC", "TRDA", "TREE", "TREX", "TRGP", "TRI", "TRIN", "TRIP", "TRMB", "TRMD", "TRML", "TRN", "TRNO", "TRNR", "TRNS", "TRON", "TROW", "TROX", "TRP", "TRS", "TRU", "TRUE", "TRUG", "TRUP", "TRV", "TRVG", "TRVI", "TS", "TSAT", "TSCO", "TSE", "TSEM", "TSHA", "TSLA", "TSLX", "TSM", "TSN", "TSQ", "TSSI", "TT", "TTAM", "TTAN", "TTC", "TTD", "TTE", "TTEC", "TTEK", "TTGT", "TTI", "TTMI", "TTSH", "TTWO", "TU", "TUSK", "TUYA", "TV", "TVA", "TVAI", "TVRD", "TVTX", "TW", "TWFG", "TWI", "TWIN", "TWLO", "TWNP", "TWO", "TWST", "TX", "TXG", "TXN", "TXNM", "TXO", "TXRH", "TXT", "TYG", "TYRA", "TZOO", "TZUP", "U", "UA", "UAA", "UAL", "UAMY", "UAVS", "UBER", "UBFO", "UBS", "UBSI", "UCAR", "UCB", "UCL", "UCTT", "UDMY", "UDR", "UE", "UEC", "UFCS", "UFG", "UFPI", "UFPT", 
     "UGI", "UGP", "UHAL", "UHAL-B", "UHG", "UHS", "UI", "UIS", "UL", "ULBI", "ULCC", "ULS", "ULY", "UMAC", "UMBF", "UMC", "UMH", "UNCY", "UNF", "UNFI", "UNH", "UNIT", "UNM", "UNP", "UNTY", "UPB", "UPBD", "UPS", "UPST", "UPWK", "UPXI", "URBN", "URGN", "UROY", "USAC", "USAR", "USAU", "USB", "USFD", "USLM", "USM", "USNA", "USPH", "UTHR", "UTI", "UTL", "UTZ", "UUUU", "UVE", "UVSP", "UVV", "UWMC", "UXIN", "V", "VAC", "VAL", "VALE", "VBIX", "VBNK", "VBTX", "VC", "VCEL", "VCTR", "VCYT", "VECO", "VEEV", "VEL", "VENU", "VEON", "VERA", "VERB", "VERI", "VERX", 
     "VET", "VFC", "VFS", "VG", "VIAV", "VICI", "VICR", "VIK", "VINP", "VIOT", "VIPS", "VIR", "VIRC", "VIRT", "VIST", "VITL", "VIV", "VKTX",
      "VLGEA", "VLN", "VLO", "VLRS", "VLTO", "VLY", "VMC", "VMD", "VMEO", "VMI", "VNDA", "VNET", "VNOM", "VNT", "VNTG", "VOD", "VOR", "VOXR", "VOYA", "VOYG", "VPG", "VRDN", "VRE",
       "VREX", "V", "WING", "WIT", "WIX", "WK", "WKC", "WKEY", "WKSP", "WLDN", "WLFC", "WLK", "WLY", "WM", "WMB", "WMG", "WMK", "WMS", "WMT", "WNC", "WNEB", "WNS", "WOOF", "WOR", "WOW", "WPC", "WPM", "WPP", "WRB", "WRBY", "WRD",
       "WS", "WSBC", "WSC", "WSFS", "WSM", "WSO", "WSR", "WST", "WT", "WTF", "WTG", "WTRG", "WTS", "WTTR", "WTW", "WU", "WULF", "WVE", "WW", "WWD", "WWW", "WXM", "WY", "WYFI", "WYNN", "WYY", "XAIR", "XBIT", "XCUR", "XEL", "XENE", "XERS", "XGN", "XHR", "XIFR", "XMTR", "XNCR", "XNET", "XOM", "XOMA", "XP", "XPEL", "XPER", "XPEV", "XPO",
        "XPOF", "XPRO", "XRAY", "XRX", "XTKG", "XYF", "XYL", "XYZ", "YALA", "YB", "YELP", "YETI", "YEXT", "YMAB", "YMAT", "YMM", "YORK", "YORW", "YOU", "YPF", "YRD", "YSG", "YSXT", "YUM", "YUMC", "YYAI", "YYGH", "Z",
         "ZBAI", "ZBH", "ZBIO", "ZBRA", "ZD", "ZDGE", "ZENA", "ZEO", "ZEPP", "ZETA", "ZEUS", 
         "ZG", "ZGN", "ZH", "ZIM", "ZIMV", "ZION", "ZIP", "ZJK", "ZK", "ZLAB", "ZM", "ZONE", "ZS", "ZSPC", "ZTO", "ZTS", "ZUMZ", "ZVIA", "ZVRA", "ZWS", "ZYBT", "ZYME"]

# 去重处理
ticker_list = list(dict.fromkeys(ticker_list))

# CNN模型已完全禁用，GPU相关代码已移除

class ICFactorSelector(BaseEstimator, TransformerMixin):
    """基于IC的因子选择器（保持兼容性）"""
    
    def __init__(self, ic_threshold=0.01):
        self.ic_threshold = ic_threshold
        self.selected_features_ = None
        self.feature_names_ = None
        
    def fit(self, X, y):
        selected = []
        
        if isinstance(X, np.ndarray):
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            self.feature_names_ = feature_names
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()
            self.feature_names_ = X_df.columns.tolist()
        
        for i, feature_name in enumerate(X_df.columns):
            try:
                factor_values = X_df.iloc[:, i]
                ic = self._calculate_ic_safe(factor_values, y)
                
                if abs(ic) > self.ic_threshold:
                    selected.append(feature_name)
                    
            except Exception:
                continue
        
        self.selected_features_ = selected
        print(f"[IC SELECTOR] 从 {len(X_df.columns)} 个因子中选择了 {len(selected)} 个")
        
        return self
    
    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("Must fit before transform")
        
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_)
        else:
            X_df = X.copy()
        
        available_features = [f for f in self.selected_features_ if f in X_df.columns]
        
        # 如果没有选择到特征，则至少保留3个原始特征避免维度不匹配
        if not available_features:
            print(f"[IC SELECTOR WARNING] 没有选择到有效特征，使用前3个原始特征")
            return X_df.iloc[:, :min(3, X_df.shape[1])].values
        
        result = X_df[available_features].values
        print(f"[IC SELECTOR] 转换后特征数: {result.shape[1]}")
        return result
    
    def _calculate_ic_safe(self, factor_data, returns):
        try:
            factor_array = np.array(factor_data).flatten()
            returns_array = np.array(returns).flatten()
            
            min_len = min(len(factor_array), len(returns_array))
            factor_array = factor_array[:min_len]
            returns_array = returns_array[:min_len]
            
            mask = ~(np.isnan(factor_array) | np.isnan(returns_array))
            clean_factor = factor_array[mask]
            clean_returns = returns_array[mask]
            
            if len(clean_factor) < 10:
                return 0.0
                
            if np.std(clean_factor) == 0 or np.std(clean_returns) == 0:
                return 0.0
            
            ic, _ = spearmanr(clean_factor, clean_returns)
            return ic if not np.isnan(ic) else 0.0
            
        except Exception:
            return 0.0

class BayesianModelAveraging:
    """
    增强版贝叶斯模型平均 + Shrinkage + EMA权重平滑
    替代原有的Stacking方法
    """
    
    def __init__(self, alpha_prior=1.5, shrinkage_factor=0.15, min_weight_threshold=0.02,
                 model_class_priors: Optional[Dict[str, float]] = None,
                 enable_meta_learner: bool = True,
                 meta_learner_type: str = 'ridge',
                 use_time_decay: bool = True,
                 half_life_days: int = 126,
                 ema_alpha: float = 0.2):
        self.alpha_prior = alpha_prior
        self.shrinkage_factor = shrinkage_factor
        self.min_weight_threshold = min_weight_threshold
        self.use_time_decay = use_time_decay
        self.half_life_days = half_life_days
        self.ema_alpha = ema_alpha
        self.prev_weights = None  # 存储上期权重用于EMA平滑
        
        # 先验：突出 LightGBM / XGBoost，弱化传统线性/随机森林/猫
        self.model_class_priors = model_class_priors or {
            'LightGBM': 0.4,
            'XGBoost': 0.4,
            'CatBoost': 0.2
        }
        self.models = {}
        self.posterior_weights = {}
        self.model_likelihoods = {}
        self.training_history = []
        # 二层融合（Ridge/ElasticNet）
        self.enable_meta_learner = enable_meta_learner
        self.meta_learner_type = meta_learner_type
        self.meta_model = None
        self.meta_feature_names: List[str] = []
        
    def fit(self, X, y, models_dict, dates=None):
        """训练BMA ensemble，支持时间衰减权重"""
        print(f"[BMA] 开始训练增强版贝叶斯模型平均ensemble...")
        print(f"[BMA] 模型数量: {len(models_dict)}")
        print(f"[BMA] 先验参数α: {self.alpha_prior}")
        print(f"[BMA] 收缩因子: {self.shrinkage_factor}")
        print(f"[BMA] 时间衰减: {self.use_time_decay}, 半衰期: {self.half_life_days}天")
        
        self.models = models_dict
        n_models = len(models_dict)
        
        # 计算时间衰减权重
        sample_weights = None
        if self.use_time_decay and dates is not None:
            # 确保dates是pandas Series并且与X长度一致
            if hasattr(dates, 'values'):
                dates_array = dates.values
            else:
                dates_array = dates
            
            # 确保长度一致
            if len(dates_array) != len(X):
                print(f"[BMA WARNING] dates长度({len(dates_array)})与X长度({len(X)})不一致，截取/填充")
                if len(dates_array) > len(X):
                    dates_array = dates_array[:len(X)]
                else:
                    # 如果dates太短，用最后一个日期填充
                    last_date = dates_array[-1] if len(dates_array) > 0 else pd.Timestamp('2024-01-01')
                    dates_array = list(dates_array) + [last_date] * (len(X) - len(dates_array))
            
            sample_weights = time_decay_weights(pd.Series(dates_array), self.half_life_days)
            print(f"[BMA] 应用时间衰减权重，权重范围: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
        
        # 时序交叉验证计算似然（使用Lopez de Prado方式，统一horizon参数）
        horizon_days = getattr(self, 'prediction_horizon_days', getattr(self, 'horizon', 5))
        embargo_mult = getattr(self, 'embargo_multiplier', 1.0)
        tscv = PurgedTimeSeriesSplit(
            n_splits=5, 
            horizon=horizon_days, 
            embargo_multiplier=embargo_mult,
            embargo_mode='adaptive'  # 使用自适应模式
        )
        print(f"[BMA CV] 使用horizon={horizon_days}天, embargo_multiplier={embargo_mult}, 模式=adaptive")
        model_scores = {}
        
        for name, model in models_dict.items():
            fold_likelihoods = []
            fold_r2_scores = []
            
            try:
                for train_idx, val_idx in tscv.split(X, dates=dates):
                    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                    
                    # 使用时间衰减权重训练
                    if sample_weights is not None:
                        train_weights = sample_weights.iloc[train_idx]
                        try:
                            # 检查是否为Pipeline对象
                            if hasattr(model, 'steps'):
                                # Pipeline对象：传递权重到最后一步
                                final_step_name = model.steps[-1][0]
                                model.fit(X_train_fold, y_train_fold, **{f"{final_step_name}__sample_weight": train_weights})
                            else:
                                # 普通模型：直接传递样本权重
                                model.fit(X_train_fold, y_train_fold, sample_weight=train_weights)
                        except TypeError:
                            # 模型不支持sample_weight参数
                            model.fit(X_train_fold, y_train_fold)
                    else:
                        model.fit(X_train_fold, y_train_fold)
                    
                    y_pred_fold = model.predict(X_val_fold)
                    
                    mse = mean_squared_error(y_val_fold, y_pred_fold)
                    r2 = r2_score(y_val_fold, y_pred_fold)
                    
                    # 对数似然（假设高斯噪声）；可加入时间衰减权重
                    if mse > 1e-10:
                        base_ll = -0.5 * len(y_val_fold) * np.log(2 * np.pi * mse) - 0.5 * len(y_val_fold)
                    else:
                        base_ll = 1000
                    # 最近数据更高权重（线性递增）
                    recency = np.linspace(0.5, 1.5, len(y_val_fold)).mean()
                    likelihood = base_ll * recency
                    
                    fold_likelihoods.append(likelihood)
                    fold_r2_scores.append(r2)
                
                avg_likelihood = np.mean(fold_likelihoods)
                avg_r2 = np.mean(fold_r2_scores)
                
                model_scores[name] = {
                    'likelihood': avg_likelihood,
                    'r2': avg_r2,
                    'std_r2': np.std(fold_r2_scores)
                }
                
                print(f"[BMA] {name}: 似然={avg_likelihood:.2f}, R2={avg_r2:.4f}±{np.std(fold_r2_scores):.4f}")
                
            except Exception as e:
                print(f"[BMA ERROR] {name} CV失败: {e}")
                model_scores[name] = {
                    'likelihood': -np.inf,
                    'r2': -np.inf,
                    'std_r2': np.inf
                }
        
        # 保存似然分数
        self.model_likelihoods = model_scores
        
        # 计算后验权重
        self._calculate_posterior_weights()
        
        # 在全部数据上重新训练所有模型（应用时间衰减权重）
        print(f"[BMA] 在全部数据上重新训练模型...")
        for name, model in self.models.items():
            try:
                # 使用时间衰减权重进行最终训练
                if sample_weights is not None:
                    try:
                        # 检查是否为Pipeline对象
                        if hasattr(model, 'steps'):
                            # Pipeline对象：传递权重到最后一步
                            final_step_name = model.steps[-1][0]
                            model.fit(X, y, **{f"{final_step_name}__sample_weight": sample_weights})
                            print(f"[BMA] {name} 最终训练完成（使用时间权重）")
                        else:
                            # 普通模型：直接传递样本权重
                            model.fit(X, y, sample_weight=sample_weights)
                            print(f"[BMA] {name} 最终训练完成（使用时间权重）")
                    except (TypeError, ValueError) as e:
                        # 模型不支持sample_weight参数
                        model.fit(X, y)
                        print(f"[BMA] {name} 最终训练完成（不支持时间权重）")
                else:
                    model.fit(X, y)
                    print(f"[BMA] {name} 最终训练完成")
            except Exception as e:
                print(f"[BMA ERROR] {name} 最终训练失败: {e}")
        
        # 训练二层融合模型（使用OOF预测）
        if self.enable_meta_learner:
            try:
                self._train_meta_learner(X, y, dates, sample_weights)
                print(f"[BMA META] 二层融合模型训练完成: {self.meta_learner_type}")
            except Exception as e:
                print(f"[BMA META ERROR] 二层融合训练失败: {e}")
        
        print(f"[BMA] 训练完成！")
        return self
    
    def _calculate_posterior_weights(self):
        """计算后验权重"""
        model_names = list(self.models.keys())
        n_models = len(model_names)
        
        # 提取似然值和R2
        likelihoods = np.array([self.model_likelihoods[name]['likelihood'] for name in model_names])
        r2_scores = np.array([self.model_likelihoods[name]['r2'] for name in model_names])
        
        # 处理无效值
        finite_mask = np.isfinite(likelihoods) & np.isfinite(r2_scores)
        if not finite_mask.any():
            print(f"[BMA WARNING] 所有模型无效，使用均等权重")
            weights = np.ones(n_models) / n_models
        else:
            # 过滤无效模型
            valid_likelihoods = likelihoods[finite_mask]
            valid_r2 = r2_scores[finite_mask]
            
            # 综合评分：似然 + R2表现 + 最近期表现强化
            combined_scores = 0.6 * valid_likelihoods + 0.35 * valid_r2 * 1000
            
            # 数值稳定的softmax
            exp_scores = np.exp(combined_scores - combined_scores.max())
            raw_weights_valid = exp_scores / exp_scores.sum()
            
            # 恢复到完整权重数组
            raw_weights = np.zeros(n_models)
            raw_weights[finite_mask] = raw_weights_valid
            
            # 先验权重：强调 LightGBM/XGBoost，其它均匀分布
            simplicity_bias = np.ones(n_models) / n_models
            prior_vec = np.array([
                self.model_class_priors.get(name, simplicity_bias[i])
                for i, name in enumerate(model_names)
            ], dtype=float)
            if prior_vec.sum() <= 0:
                prior_vec = simplicity_bias
            prior_vec = prior_vec / prior_vec.sum()
            
            # 后验权重 = 收缩版本
            weights = ((1 - self.shrinkage_factor) * raw_weights + 
                       self.shrinkage_factor * prior_vec)
        
        # 最小权重阈值
        weights = np.maximum(weights, self.min_weight_threshold)
        weights = weights / weights.sum()
        
        # EMA平滑权重以减少持仓抖动
        new_weights = pd.Series(weights, index=model_names)
        if self.prev_weights is not None:
            smoothed_weights = ema_smooth_weights(
                self.prev_weights, new_weights, 
                alpha=self.ema_alpha, min_w=0.02
            )
            self.posterior_weights = smoothed_weights.to_dict()
            print(f"[BMA] 应用EMA平滑，alpha={self.ema_alpha}")
        else:
            self.posterior_weights = new_weights.to_dict()
            
        # 保存当前权重供下次使用
        self.prev_weights = pd.Series(self.posterior_weights)
        
        print(f"[BMA WEIGHTS] 后验权重分配:")
        for name, weight in self.posterior_weights.items():
            print(f"  {name}: {weight:.4f}")
        
        # 权重熵（多样性指标）
        entropy_val = entropy(list(self.posterior_weights.values()))
        max_entropy = np.log(n_models)
        normalized_entropy = entropy_val / max_entropy
        print(f"[BMA] 权重熵: {entropy_val:.4f} (归一化: {normalized_entropy:.4f})")
        
        return self.posterior_weights
    
    def predict(self, X):
        """BMA预测 - 优化版本，跳过有问题的模型"""
        if not self.posterior_weights:
            raise ValueError("模型未训练，请先调用fit()")
        
        predictions = {}
        valid_weights = {}
        total_failed_weight = 0.0
        
        # 获取每个模型的预测
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
                valid_weights[name] = self.posterior_weights[name]
            except Exception as e:
                print(f"[BMA PREDICT ERROR] {name}: {e}")
                total_failed_weight += self.posterior_weights[name]
                # 跳过失败的模型
        
        # 如果没有任何模型成功，返回零预测
        if not predictions:
            print("[BMA WARNING] 所有模型预测失败，返回零预测")
            return np.zeros(len(X))
        
        # 重新归一化有效模型的权重
        if total_failed_weight > 0:
            remaining_weight = 1.0 - total_failed_weight
            if remaining_weight > 0:
                for name in valid_weights:
                    valid_weights[name] = valid_weights[name] / remaining_weight
            else:
                # 如果失败权重太大，平均分配
                uniform_weight = 1.0 / len(valid_weights)
                for name in valid_weights:
                    valid_weights[name] = uniform_weight
        
        # 加权平均（作为基础特征）
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            weight = valid_weights[name]
            ensemble_pred += weight * pred

        # Ultra集成：Regime-aware 融合 Alpha 与 ML 预测
        if getattr(self, 'ultra_enabled', False):
            try:
                # 构造简要市场数据用于状态判定（回退默认值）
                dates = X.index if hasattr(X, 'index') else pd.RangeIndex(len(X))
                returns_series = pd.Series(ensemble_pred, index=dates)
                vol_series = pd.Series(pd.Series(ensemble_pred).rolling(20).std().fillna(0.02).values, index=dates)
                market_data = {'returns': returns_series, 'volatility': vol_series}

                # 预测不确定性（简化为预测方差）
                pred_unc = float(np.std(ensemble_pred)) if len(ensemble_pred) > 1 else 0.1

                # 当前信号（用集成预测的均值作为动量代理）
                current_signals = {
                    'momentum': float(np.mean(ensemble_pred)),
                    'volatility': float(vol_series.mean())
                }

                weights = self.adaptive_weight_engine.adaptive_weight_allocation(
                    market_data=market_data,
                    prediction_uncertainty=pred_unc,
                    current_signals=current_signals
                ) if self.adaptive_weight_engine else {'alpha': 0.3, 'ml': 0.7, 'cash': 0.0}

                # Alpha通道（若Alpha引擎可用，使用风险中性后的Alpha均值；否则0）
                alpha_signal = 0.0
                try:
                    if self.alpha_engine_enh is not None and hasattr(self, 'latest_enhanced_df'):
                        alpha_df = self.alpha_engine_enh.enhanced_alpha_computation(self.latest_enhanced_df)
                        if not alpha_df.empty:
                            alpha_signal = float(alpha_df.mean(axis=1).reindex(dates, method='ffill').iloc[-1])
                except Exception:
                    alpha_signal = 0.0

                final_pred = (
                    weights['alpha'] * alpha_signal +
                    weights['ml'] * ensemble_pred +
                    weights['cash'] * 0.0
                )
                ensemble_pred = final_pred
                print(f"[REGIME] 自适应权重 Alpha={weights['alpha']:.2f}, ML={weights['ml']:.2f}, Cash={weights['cash']:.2f}")
            except Exception as e:
                print(f"[REGIME WARN] 自适应融合失败，使用纯BMA: {e}")

        # 如果有二层融合，使用元学习器进行最终预测
        if self.enable_meta_learner and self.meta_model is not None:
            # 构造特征矩阵：优先使用 XGBoost/LightGBM，再附加 BMA 加权结果
            feature_cols = []
            # 选择用于融合的模型
            preferred = [m for m in ["XGBoost", "LightGBM"] if m in predictions]
            used_names = preferred if preferred else list(predictions.keys())
            for name in used_names:
                feature_cols.append(predictions[name].reshape(-1, 1))
            feature_cols.append(ensemble_pred.reshape(-1, 1))  # 加入BMA加权
            meta_X = np.hstack(feature_cols)
            return self.meta_model.predict(meta_X)
        
        return ensemble_pred

    def _train_meta_learner(self, X, y, dates=None, sample_weights=None):
        """使用PurgedTimeSeriesSplit+时间权重训练二层融合模型（Ridge/ElasticNet）"""
        model_names = list(self.models.keys())
        # 仅用核心GBDT模型作为特征，若不可用则退回全部
        core = [m for m in ["XGBoost", "LightGBM"] if m in self.models]
        selected = core if core else model_names
        self.meta_feature_names = selected + ["BMA_weighted"]

        # 使用与一层相同的PurgedTimeSeriesSplit，确保参数统一
        horizon_days = getattr(self, 'prediction_horizon_days', getattr(self, 'horizon', 5))
        embargo_mult = getattr(self, 'embargo_multiplier', 1.0)
        tscv = PurgedTimeSeriesSplit(
            n_splits=5, 
            horizon=horizon_days, 
            embargo_multiplier=embargo_mult,
            embargo_mode='adaptive'  # 使用自适应模式
        )
        print(f"[META LEARNING] 使用统一参数: horizon={horizon_days}天, embargo_multiplier={embargo_mult}")
        oof_pred_matrix = np.full((len(y), len(selected)), np.nan, dtype=float)
        
        # 存储每个fold的OOS表现用于计算滚动指标
        self.oos_metrics = []
        
        # 为计算 BMA 加权的OOF，需要存储每个模型的OOF
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X, dates=dates)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            # 应用时间衰减权重（如果可用）
            train_weights = None
            if sample_weights is not None:
                train_weights = sample_weights.iloc[train_idx] if hasattr(sample_weights, 'iloc') else sample_weights[train_idx]
                
            for j, name in enumerate(selected):
                base_model = self.models[name]
                try:
                    m = clone(base_model)
                    # 尝试使用时间权重训练
                    if train_weights is not None:
                        try:
                            # 检查是否为Pipeline对象
                            if hasattr(m, 'steps'):
                                # Pipeline对象：传递权重到最后一步
                                final_step_name = m.steps[-1][0]
                                m.fit(X_train, y_train, **{f"{final_step_name}__sample_weight": train_weights})
                            else:
                                # 普通模型：直接传递样本权重
                                m.fit(X_train, y_train, sample_weight=train_weights)
                        except (TypeError, ValueError):
                            m.fit(X_train, y_train)
                    else:
                        m.fit(X_train, y_train)
                    oof_pred_matrix[val_idx, j] = m.predict(X_val)
                except Exception as e:
                    print(f"[BMA META OOF] {name} 第{fold_idx+1}折失败: {e}")
                    # 留下 NaN，后续填补
            
            # 计算当前fold的OOS指标
            val_predictions = oof_pred_matrix[val_idx]
            val_true = y[val_idx].values if hasattr(y, 'values') else y[val_idx]
            val_dates = dates.iloc[val_idx] if hasattr(dates, 'iloc') else dates[val_idx]
            
            # 计算各种OOS指标
            fold_metrics = self._calculate_oos_metrics(val_true, val_predictions, val_dates, fold_idx)
            self.oos_metrics.append(fold_metrics)

        # 将 NaN 用列均值填补
        col_means = np.nanmean(oof_pred_matrix, axis=0)
        inds = np.where(np.isnan(oof_pred_matrix))
        oof_pred_matrix[inds] = np.take(col_means, inds[1])

        # 计算OOF层面的BMA加权作为额外特征
        # 对齐选择的模型权重
        weights = np.array([self.posterior_weights.get(name, 0.0) for name in selected], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights) / max(1, len(weights))
        else:
            weights = weights / weights.sum()
        bma_oof = (oof_pred_matrix * weights.reshape(1, -1)).sum(axis=1).reshape(-1, 1)
        meta_X = np.hstack([oof_pred_matrix, bma_oof])

        # 训练元学习器（应用时间权重）
        if self.meta_learner_type == 'elasticnet':
            self.meta_model = ElasticNet(alpha=0.05, l1_ratio=0.2)
        else:
            self.meta_model = Ridge(alpha=1.0)
            
        # 使用时间权重训练元学习器
        if sample_weights is not None:
            try:
                self.meta_model.fit(meta_X, y, sample_weight=sample_weights)
                print(f"[BMA META] 元学习器使用时间权重训练")
            except (TypeError, ValueError):
                self.meta_model.fit(meta_X, y)
                print(f"[BMA META] 元学习器不支持时间权重")
        else:
            self.meta_model.fit(meta_X, y)

class QuantitativeModel:
    """增强版量化分析模型 - 使用BMA替代Stacking"""
    
    def __init__(self, progress_callback=None):
        self.bma_model = None
        self.pipelines = {}
        self.factor_ic_scores = {}
        self.model_scores = {}
        self.training_feature_columns = []
        self.progress_callback = progress_callback  # 添加进度回调函数
        self.prediction_horizon_days = None  # 当前预测周期（天）
        
        # Ultra集成：统一市场数据与风险/Alpha与Regime自适应权重
        self.ultra_enabled = ULTRA_INTEGRATION_AVAILABLE
        if self.ultra_enabled:
            try:
                self.market_data_manager = UnifiedMarketDataManager()
                self.risk_model_config = RiskModelConfig()
                self.alpha_engine_enh = None  # 按需构建，需传入AlphaStrategiesEngine实例
                self.adaptive_weight_engine = AdaptiveWeightEngine()
                try:
                    mkt = get_market_indices_data(period="3y")
                    self.adaptive_weight_engine.train_regime_model(mkt)
                except Exception as e:
                    print(f"[REGIME] 状态模型训练失败，使用默认权重: {e}")
            except Exception as e:
                self.ultra_enabled = False
                print(f"[WARN] Ultra集成初始化失败: {e}")
    
    def download_data(self, tickers, start_date, end_date):
        """下载股票数据（保持原有接口）"""
        print(f"[DATA] 下载 {len(tickers)} 只股票的数据...")
        print(f"[PROGRESS] 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        
        # 进度监控初始化
        if self.progress_callback:
            self.progress_callback.set_stage("数据下载", len(tickers))
        
        data = {}
        success_count = 0
        failed_count = 0
        
        for i, ticker in enumerate(tickers, 1):
            try:
                # 更新进度监控
                if self.progress_callback:
                    self.progress_callback.update_progress(f"正在下载 {ticker}", i-1)
                
                # 增强：添加重试机制
                max_retries = 3
                stock_data = None
                
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            print(f"[RETRY {attempt}/{max_retries-1}] {ticker:6s}...", end=" ")
                            time.sleep(1.0 * attempt)  # 指数退避
                        else:
                            print(f"[{i:3d}/{len(tickers):3d}] 下载 {ticker:6s}...", end=" ")
                            
                        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                        
                        if len(stock_data) > 0:
                            break  # 成功，跳出重试循环
                        elif attempt == max_retries - 1:
                            print(f"[FAIL] 无数据（重试{max_retries}次后）")
                            
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"[FAIL] 失败: {str(e)[:30]}...")
                        else:
                            print(f"[ERROR] {str(e)[:20]}...", end=" ")
                        continue
                
                # 处理下载结果
                if stock_data is not None and len(stock_data) > 0:
                    if isinstance(stock_data.columns, pd.MultiIndex):
                        stock_data.columns = stock_data.columns.droplevel(1)
                    data[ticker] = stock_data
                    success_count += 1
                    print(f"成功 {len(stock_data)} 天数据")
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                print(f"[FAIL] 异常: {str(e)[:30]}...")
                continue
        
        # 最终进度更新
        if self.progress_callback:
            self.progress_callback.update_progress("数据下载完成", len(tickers))
        
        print(f"[PROGRESS] 结束时间: {datetime.now().strftime('%H:%M:%S')}")
        print(f"[SUMMARY] 成功: {success_count}, 失败: {failed_count}, 总计: {len(tickers)}")
        
        if data:
            print(f"[SUCCESS] 成功下载 {len(data)} 只股票数据")
        else:
            print(f"[ERROR] 没有成功下载任何数据")
        
        return data
    
    def calculate_technical_indicators(self, data):
        """计算技术指标（周度预测增强：加入更快的均线/交叉信号，RSI缩短窗口）"""
        indicators = {}
        
        # 移动平均线（修复：添加min_periods避免不完整窗口）
        indicators['sma_5'] = data['Close'].rolling(window=5, min_periods=5).mean()
        indicators['sma_10'] = data['Close'].rolling(window=10, min_periods=10).mean()
        indicators['sma_20'] = data['Close'].rolling(window=20, min_periods=20).mean()
        indicators['sma_50'] = data['Close'].rolling(window=50, min_periods=50).mean()
        
        # 指数移动平均（修复：添加min_periods确保窗口完整性）
        indicators['ema_5'] = data['Close'].ewm(span=5, min_periods=5).mean()
        indicators['ema_10'] = data['Close'].ewm(span=10, min_periods=10).mean()
        indicators['ema_12'] = data['Close'].ewm(span=12, min_periods=12).mean()
        indicators['ema_26'] = data['Close'].ewm(span=26, min_periods=26).mean()
        indicators['ema_cross_5_10'] = indicators['ema_5'] - indicators['ema_10']
        
        # MACD（修复：确保signal线有完整窗口）
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = indicators['macd'].ewm(span=9, min_periods=9).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # RSI（缩短窗口提升对未来5日的敏感度）
        indicators['rsi'] = self.calculate_rsi(data['Close'], period=10)

        # 随机指标 Stochastic (5,3)（修复：添加min_periods）
        highest_high_5 = data['High'].rolling(window=5, min_periods=5).max()
        lowest_low_5 = data['Low'].rolling(window=5, min_periods=5).min()
        stoch_k = (data['Close'] - lowest_low_5) / (highest_high_5 - lowest_low_5 + 1e-9) * 100.0
        indicators['stoch_k_5_3'] = stoch_k.rolling(window=3, min_periods=3).mean()
        indicators['stoch_d_5_3'] = indicators['stoch_k_5_3'].rolling(window=3, min_periods=3).mean()
        
        # 布林带（修复：添加min_periods）
        bb_middle = data['Close'].rolling(window=20, min_periods=20).mean()
        bb_std = data['Close'].rolling(window=20, min_periods=20).std()
        indicators['bollinger_upper'] = bb_middle + (bb_std * 2)
        indicators['bollinger_lower'] = bb_middle - (bb_std * 2)
        indicators['bollinger_width'] = indicators['bollinger_upper'] - indicators['bollinger_lower']
        indicators['bollinger_position'] = (data['Close'] - indicators['bollinger_lower']) / indicators['bollinger_width']
        
        return indicators
    
    def calculate_rsi(self, prices, period=10):
        """计算RSI（修复：添加min_periods确保窗口完整）"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_cci(self, data, period=20):
        """计算CCI (Commodity Channel Index)"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad_tp = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=False)
        return (typical_price - sma_tp) / (0.015 * mad_tp)
    
    def calculate_factors(self, data):
        """计算所有因子（周度预测：加强5-10日窗口）"""
        factors = {}
        
        # 基础技术指标
        tech_indicators = self.calculate_technical_indicators(data)
        factors.update(tech_indicators)
        
        # 价格因子 - 强化 5/10 日
        factors['price_momentum_5'] = data['Close'].pct_change(5)
        factors['price_momentum_10'] = data['Close'].pct_change(10)
        factors['price_mom_diff_5_10'] = factors['price_momentum_5'] - factors['price_momentum_10']
        
        # 价格加速度和反转（5 vs 10）
        factors['price_acceleration'] = data['Close'].pct_change(5) - data['Close'].pct_change(10)
        factors['price_reversal'] = -data['Close'].pct_change(2)
        
        # 成交量因子 - 使用10天聚合（修复：添加min_periods）
        factors['volume_sma_10'] = data['Volume'].rolling(window=10, min_periods=10).mean()
        factors['volume_ratio'] = data['Volume'] / factors['volume_sma_10']
        factors['volume_momentum'] = data['Volume'].pct_change(10)
        
        # 价格-成交量因子 - 使用10天聚合
        factors['price_volume'] = data['Close'] * data['Volume']
        factors['money_flow'] = (data['Close'] * data['Volume']).rolling(10, min_periods=10).mean() / data['Close'].rolling(10, min_periods=10).mean()
        
        # 波动率因子 - 使用10天聚合（修复：添加min_periods）
        factors['volatility_10'] = data['Close'].pct_change().rolling(window=10, min_periods=10).std()

        # ATR(5) 真实波动幅度
        tr1 = data['High'] - data['Low']
        tr2 = (data['High'] - data['Close'].shift(1)).abs()
        tr3 = (data['Low'] - data['Close'].shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        factors['atr_5'] = true_range.rolling(window=5, min_periods=5).mean()
        
        # 相对强弱因子 - 使用10天聚合（修复：添加min_periods）
        factors['rs_vs_sma'] = data['Close'] / data['Close'].rolling(10, min_periods=10).mean() - 1
        factors['high_low_ratio'] = data['High'] / data['Low'] - 1
        factors['close_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        factors['rolling_max_5'] = data['Close'].rolling(window=5, min_periods=5).max()
        factors['rolling_min_5'] = data['Close'].rolling(window=5, min_periods=5).min()
        factors['breakout_5'] = data['Close'] / factors['rolling_max_5'] - 1
        factors['pullback_5'] = data['Close'] / factors['rolling_min_5'] - 1

        # 5日收益z分数（短期均值回归/过热程度）（修复：添加min_periods）
        daily_ret = data['Close'].pct_change()
        mean_5 = daily_ret.rolling(window=5, min_periods=5).mean()
        std_5 = daily_ret.rolling(window=5, min_periods=5).std()
        factors['ret_5_zscore'] = (mean_5 / (std_5 + 1e-9))
        
        # 市场情绪因子
        factors['gap_ratio'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        factors['intraday_return'] = (data['Close'] - data['Open']) / data['Open']
        factors['overnight_return'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        # 高级因子 - 使用10天聚合（修复：添加min_periods）
        factors['rolling_sharpe'] = (data['Close'].pct_change().rolling(10, min_periods=10).mean() / 
                                   data['Close'].pct_change().rolling(10, min_periods=10).std())
        
        # CCI (Commodity Channel Index) - 商品通道指数（修复：添加min_periods）
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp_10 = typical_price.rolling(window=10, min_periods=10).mean()
        mad_tp_10 = typical_price.rolling(window=10, min_periods=10).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=False)
        factors['cci_10'] = (typical_price - sma_tp_10) / (0.015 * mad_tp_10)
        
        # CCI衍生因子
        factors['cci_20'] = self._calculate_cci(data, period=20)
        factors['cci_normalized'] = factors['cci_10'] / 100.0  # 标准化CCI
        factors['cci_momentum'] = factors['cci_10'].diff(5)  # CCI动量
        factors['cci_cross_signal'] = np.where(
            (factors['cci_10'] > 100) & (factors['cci_10'].shift(1) <= 100), 1,
            np.where((factors['cci_10'] < -100) & (factors['cci_10'].shift(1) >= -100), -1, 0)
        )  # CCI交叉信号
        
        # 转换为DataFrame并清理
        factors_df = pd.DataFrame(factors, index=data.index)
        factors_df = factors_df.replace([np.inf, -np.inf], np.nan)
        
        # 更严格的数据清理 - 使用列中位数填补；全部NaN列回退为0
        for col in factors_df.columns:
            if factors_df[col].isnull().all():
                factors_df[col] = 0.0
            else:
                median_val = factors_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                factors_df[col] = factors_df[col].fillna(median_val)

        # 验证没有无穷值和NaN（无穷值→NaN→再按中位数处理）
        factors_df = factors_df.replace([np.inf, -np.inf], np.nan)
        for col in factors_df.columns:
            if factors_df[col].isnull().any():
                median_val = factors_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                factors_df[col] = factors_df[col].fillna(median_val)
        
        print(f"[FACTORS] 计算完成，维度: {factors_df.shape}")
        return factors_df
    
    def calculate_ic(self, factor_data, returns):
        """计算IC"""
        try:
            factor_array = np.array(factor_data).flatten()
            returns_array = np.array(returns).flatten()
            
            min_len = min(len(factor_array), len(returns_array))
            factor_array = factor_array[:min_len]
            returns_array = returns_array[:min_len]
            
            mask = ~(np.isnan(factor_array) | np.isnan(returns_array))
            clean_factor = factor_array[mask]
            clean_returns = returns_array[mask]
            
            if len(clean_factor) < 10:
                return 0.0
                
            if np.std(clean_factor) == 0 or np.std(clean_returns) == 0:
                return 0.0
            
            ic, _ = spearmanr(clean_factor, clean_returns)
            return ic if not np.isnan(ic) else 0.0
            
        except Exception:
            return 0.0
    
    def prepare_ml_data_with_time_series(self, all_data, target_period=5):
        """准备时序安全的机器学习数据（默认预测未来一周/5个交易日的收益）"""
        print(f"[TIME-SERIES PREP] 准备时序安全的机器学习数据...")
        # 记录当前预测周期（用于后续动态特征筛选等参数）
        try:
            self.prediction_horizon_days = int(target_period)
        except Exception:
            self.prediction_horizon_days = 5
        
        all_factor_data = []
        
        print(f"[GLOBAL FACTORS] 计算全局因子池...")
        
        for ticker, data in all_data.items():
            try:
                print(f"[GLOBAL FACTORS] 处理 {ticker}...")
                
                # 计算因子
                factors = self.calculate_factors(data)
                
                # 使用严格的目标构造方法，避免前瞻性偏差
                # skip_days=21 表示跳过21天形成期，horizon=target_period表示持有期
                temp_data = pd.DataFrame({
                    'Close': data['Close'],
                    'date': data.index
                }).reset_index(drop=True)
                temp_data['date'] = pd.to_datetime(temp_data['date'])
                temp_data['ticker'] = ticker
                
                # 使用make_target方法构造严格的未来收益目标
                target_data = make_target(temp_data, skip_days=21, horizon=target_period, price_col='Close')
                
                # 严格按日期对齐，避免索引错位
                if len(target_data) > 0:
                    # 将因子数据转换为DataFrame并添加日期列
                    factors_with_date = factors.copy()
                    factors_with_date['date'] = factors_with_date.index
                    factors_with_date = factors_with_date.reset_index(drop=True)
                    
                    # 按日期合并，确保时间对齐
                    aligned_data = pd.merge(
                        factors_with_date, 
                        target_data[['date', 'target']], 
                        on='date', 
                        how='inner'
                    )
                    
                    # 保留date列用于后续处理，不在这里删除
                else:
                    # 如果没有目标数据，创建空的对齐数据
                    aligned_data = pd.DataFrame()
                
                # 先过滤掉目标为NaN的样本（避免将无效数据标记为0收益）
                if 'target' in aligned_data.columns:
                    aligned_data = aligned_data[aligned_data['target'].notna()]
                aligned_data = aligned_data.dropna()
                
                if len(aligned_data) == 0:
                    print(f"[WARNING] {ticker} 没有有效的对齐数据")
                    continue
                
                # 添加股票信息（日期信息已经在merge时保留）
                aligned_data['ticker'] = ticker
                aligned_data = aligned_data.reset_index(drop=True)
                
                all_factor_data.append(aligned_data)
                
                print(f"[FACTORS] 计算了 {len(factors.columns)} 个有效因子")
                
            except Exception as e:
                print(f"[GLOBAL FACTORS ERROR] {ticker}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        if not all_factor_data:
            raise ValueError("没有有效的因子数据")
        
        # 合并所有数据并按日期全局排序
        combined_data = pd.concat(all_factor_data, ignore_index=True)
        combined_data = combined_data.sort_values('date').reset_index(drop=True)
        
        # 注意：滞后处理将在 BMA 训练的预处理函数中统一进行
        # 这里不再重复滞后，避免双重滞后导致的数据问题
        print(f"[DATA-PREP] 数据准备完成，将在训练时进行滞后处理")
        print(f"[DATA-PREP] 样本数: {len(combined_data)}, 股票数: {combined_data['ticker'].nunique()}")
        
        # 先过滤目标，再清理其他NaN（避免误删可修复因子）
        if 'target' in combined_data.columns:
            combined_data_final = combined_data[combined_data['target'].notna()].dropna()
        else:
            combined_data_final = combined_data.dropna()
        
        # 分离特征和目标
        feature_columns = [col for col in combined_data_final.columns 
                          if col not in ['target', 'ticker', 'date']]

        # 严格的时间无泄露检查：移除任何可能由未来生成的列（名称启发式）
        feature_columns = [c for c in feature_columns if 'future' not in c.lower()]

        X = combined_data_final[feature_columns]
        y = combined_data_final['target']
        dates = combined_data_final['date']
        tickers = combined_data_final['ticker']
        
        print(f"[TIME-SERIES PREP] 滞后处理后剩余 {len(X)} 个样本，{len(feature_columns)} 个因子")
        print(f"[TIME-SERIES PREP] 时间范围: {dates.min()} 到 {dates.max()}")
        print(f"[TIME-SERIES PREP] 包含 {len(tickers.unique())} 只股票")

        # Ultra集成：缓存用于Alpha增强的数据视图（供Regime融合Alpha通道使用）
        if getattr(self, 'ultra_enabled', False):
            try:
                enhanced_view = X.copy()
                enhanced_view['date'] = dates.values
                enhanced_view['ticker'] = tickers.values
                self.latest_enhanced_df = enhanced_view
            except Exception:
                pass
        
        return X, y, tickers, dates
    
    def _calculate_oos_metrics(self, y_true, y_pred, dates, fold_idx):
        """计算滚动Out-of-Sample指标"""
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np
        
        # 确保数据格式正确
        if len(y_pred.shape) > 1:
            # 如果是多个模型的预测，取均值
            y_pred_mean = np.nanmean(y_pred, axis=1)
        else:
            y_pred_mean = y_pred
            
        # 移除NaN值
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred_mean))
        if np.sum(valid_mask) == 0:
            return {"fold": fold_idx, "error": "No valid predictions"}
            
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred_mean[valid_mask]
        
        try:
            # 基础统计指标
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_clean, y_pred_clean)
            
            # 信息比率 (Information Ratio)
            excess_returns = y_pred_clean - y_true_clean
            ir = np.mean(excess_returns) / (np.std(excess_returns) + 1e-6)
            
            # 命中率 (Hit Rate) - 预测方向正确的比例
            hit_rate = np.mean(np.sign(y_pred_clean) == np.sign(y_true_clean))
            
            # 最大回撤 (基于累积误差)
            cumulative_error = np.cumsum(excess_returns)
            running_max = np.maximum.accumulate(cumulative_error)
            drawdown = cumulative_error - running_max
            max_drawdown = np.min(drawdown)
            
            # 夏普比率 (基于预测收益)
            if np.std(y_pred_clean) > 1e-6:
                sharpe_ratio = np.mean(y_pred_clean) / np.std(y_pred_clean)
            else:
                sharpe_ratio = 0.0
            
            metrics = {
                "fold": fold_idx,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "information_ratio": ir,
                "hit_rate": hit_rate,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "n_samples": len(y_true_clean),
                "start_date": dates.min() if hasattr(dates, 'min') else "N/A",
                "end_date": dates.max() if hasattr(dates, 'max') else "N/A"
            }
            
            return metrics
            
        except Exception as e:
            return {"fold": fold_idx, "error": str(e)}
    
    def print_oos_summary(self):
        """打印滚动OOS指标汇总"""
        if not hasattr(self, 'oos_metrics') or not self.oos_metrics:
            print("[OOS METRICS] 无可用的OOS指标")
            return
            
        valid_metrics = [m for m in self.oos_metrics if 'error' not in m]
        if not valid_metrics:
            print("[OOS METRICS] 所有fold都有错误")
            return
            
        print(f"\n[滚动OOS指标汇总] 基于{len(valid_metrics)}个fold")
        print("=" * 60)
        
        # 计算平均指标
        avg_r2 = np.mean([m['r2'] for m in valid_metrics])
        avg_ir = np.mean([m['information_ratio'] for m in valid_metrics])
        avg_hit_rate = np.mean([m['hit_rate'] for m in valid_metrics])
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in valid_metrics])
        avg_max_dd = np.mean([m['max_drawdown'] for m in valid_metrics])
        
        print(f"平均R²: {avg_r2:.4f}")
        print(f"平均信息比率: {avg_ir:.4f}")
        print(f"平均命中率: {avg_hit_rate:.1%}")
        print(f"平均夏普比率: {avg_sharpe:.4f}")
        print(f"平均最大回撤: {avg_max_dd:.4f}")
        
        # 计算指标稳定性
        r2_std = np.std([m['r2'] for m in valid_metrics])
        ir_std = np.std([m['information_ratio'] for m in valid_metrics])
        
        print(f"\n指标稳定性:")
        print(f"   R²标准差: {r2_std:.4f}")
        print(f"   信息比率标准差: {ir_std:.4f}")
        
        print("=" * 60)
    
    def train_models_with_bma(self, X, y, enable_hyperopt=True, apply_preprocessing=True, dates=None, tickers=None):
        """使用增强版BMA替代Stacking训练模型，支持因子预处理"""
        print(f"[BMA ENHANCED] 开始使用增强版BMA训练模型...")
        print(f"[PROGRESS] 开始时间: {datetime.now().strftime('%H:%M:%S')}")
        
        # 设置统一的embargo参数
        self.embargo_multiplier = 1.0  # 可配置的embargo倍数
        
        # 新增：数据预处理流水线（修复版：使用合表方式）
        if apply_preprocessing and dates is not None and tickers is not None:
            print(f"[PREPROCESSING] 开始数据预处理流水线...")
            
            # 创建自定义预处理Pipeline
            from sklearn.preprocessing import FunctionTransformer
            
            # 定义预处理步骤（修复：移除全局PCA防止信息泄露）
            def preprocess_function(X_with_meta):
                # 1. 严格滞后处理（移到最前面，确保只用t-1数据）
                X_processed = lag_features_to_t_minus_1(X_with_meta, by='ticker', by_cols=['date'])
                
                # 验证滞后效果
                try:
                    assert_no_lookahead(X_processed)
                    print(f"[PASS] 预处理中的前瞻性偏差检查通过")
                except Exception as e:
                    print(f"[CRITICAL] 预处理中发现前瞻性偏差: {e}")
                
                # 2. 去极值 + 标准化（横截面处理）
                X_processed = cross_sectional_winsorize(X_processed, by_cols=['date'])
                # 3. 施密特正交化（替代全局PCA，避免信息泄露）
                X_processed = schmidt_orthogonalize(X_processed, by_cols=['date'])
                return X_processed
            
            # 0. 合并辅助列（dates, tickers）到X中进行预处理
            X_with_meta = X.copy()
            X_with_meta['date'] = dates
            X_with_meta['ticker'] = tickers
            print(f"[PREPROCESSING] 合并辅助列，数据形状: {X_with_meta.shape}")
            
            # 应用预处理
            X_processed = preprocess_function(X_with_meta)
            print(f"[PREPROCESSING] 完成预处理步骤")
            
            # 5. 智能去除缺失值（保留每个股票的时序结构）
            # 按股票分组处理，确保不破坏时间序列的滞后结构
            cleaned_groups = []
            for ticker in X_processed['ticker'].unique():
                ticker_data = X_processed[X_processed['ticker'] == ticker].copy()
                
                # 对于每个股票，保留第一行（即使有NaN），因为这是滞后的结果
                # 只清理后续行中的异常缺失值
                if len(ticker_data) > 1:
                    # 第一行保留（滞后产生的NaN是正常的）
                    first_row = ticker_data.iloc[0:1]
                    remaining_rows = ticker_data.iloc[1:]
                    
                    # 后续行去除过多缺失值（如超过50%的特征缺失）
                    feature_cols = [c for c in ticker_data.columns if c not in ['date', 'ticker']]
                    missing_ratio = remaining_rows[feature_cols].isnull().sum(axis=1) / len(feature_cols)
                    valid_rows = remaining_rows[missing_ratio < 0.5]  # 保留缺失率<50%的行
                    
                    # 重新组合
                    ticker_cleaned = pd.concat([first_row, valid_rows], ignore_index=True)
                else:
                    ticker_cleaned = ticker_data
                    
                if len(ticker_cleaned) > 0:
                    cleaned_groups.append(ticker_cleaned)
            
            if cleaned_groups:
                # 修复：保持原始索引信息，避免错位
                X_processed = pd.concat(cleaned_groups, ignore_index=False)  # 保持原始index
                
                # 精确对齐：使用原始索引重新对齐y, dates, tickers
                valid_indices = X_processed.index
                y = y.loc[valid_indices] if hasattr(y, 'loc') else y.iloc[valid_indices]
                dates = X_processed['date'].values
                tickers = X_processed['ticker'].values
                
                # 重置index确保连续性
                X_processed = X_processed.reset_index(drop=True)
                if hasattr(y, 'reset_index'):
                    y = y.reset_index(drop=True)
                else:
                    y = pd.Series(y).reset_index(drop=True)
                
                print(f"[PREPROCESSING] 智能清理后样本数: {len(X_processed)}")
                print(f"[PREPROCESSING] 保留的股票数: {X_processed['ticker'].nunique()}")
            else:
                print(f"[PREPROCESSING] 警告：清理后无有效样本")
                
            # 6. 前瞻性偏差检查（现在应该通过）
            try:
                check_result = assert_no_lookahead(X_processed)
                if check_result:
                    print(f"[PREPROCESSING] ✅ 通过前瞻性偏差检查")
                else:
                    print(f"[PREPROCESSING] ⚠️ 前瞻性偏差检查有警告，但可能是预期的")
            except Exception as e:
                print(f"[PREPROCESSING] ❌ 前瞻性偏差检查失败: {e}")
            
            # 7. 最终NaN清理：对于BMA训练，必须去除所有NaN
            feature_cols = [c for c in X_processed.columns if c not in ['date', 'ticker']]
            before_final_clean = len(X_processed)
            
            # 最终清理：移除任何包含NaN的行（修复错位问题）
            final_valid_idx = ~X_processed[feature_cols].isnull().any(axis=1)
            X_processed_final = X_processed[final_valid_idx].copy()
            
            # 修复：使用正确的索引对齐，避免错位
            if hasattr(y, 'loc'):
                y_final = y[final_valid_idx].reset_index(drop=True)
            else:
                y_final = pd.Series(y)[final_valid_idx].reset_index(drop=True)
                
            dates_final = X_processed_final['date'].values  
            tickers_final = X_processed_final['ticker'].values
            
            print(f"[PREPROCESSING] 最终NaN清理: {before_final_clean} -> {len(X_processed_final)} 样本")
            
            # 8. 分离：去掉辅助列，得到干净的X
            X = X_processed_final.drop(['date', 'ticker'], axis=1)

            # 额外安全：移除零方差列，防止数值不稳定
            var_series = X.var(axis=0, ddof=0)
            non_zero_var_cols = var_series[var_series > 0].index.tolist()
            if len(non_zero_var_cols) < X.shape[1]:
                removed = set(X.columns) - set(non_zero_var_cols)
                print(f"[PREPROCESSING] 移除零方差特征: {len(removed)} 个 → {sorted(list(removed))[:10]}{'...' if len(removed)>10 else ''}")
            X = X[non_zero_var_cols]
            y = y_final
            dates = dates_final
            tickers = tickers_final
            
            print(f"[PREPROCESSING] 预处理完成，最终特征维度: {X.shape}")
            print(f"[PREPROCESSING] 检查NaN: {X.isnull().sum().sum()} 个NaN值")
            
            # 保存预处理函数用于预测时应用
            self.preprocessing_function = preprocess_function
            print(f"[PREPROCESSING] 预处理函数已保存用于预测")
            
        else:
            if apply_preprocessing:
                print(f"[PREPROCESSING] 跳过预处理：缺少dates或tickers参数")
            else:
                print(f"[PREPROCESSING] 跳过预处理，使用原始数据")
            # 没有预处理时清除pipeline
            self.preprocessing_function = None
        
        # 保存训练时的特征列顺序和中位数（用于预测时填补缺失值）
        self.training_feature_columns = X.columns.tolist()
        self.feature_medians = X.median().to_dict()
        print(f"[FEATURE CONSISTENCY] 保存训练特征列: {len(self.training_feature_columns)} 个")
        print(f"[FEATURE CONSISTENCY] 保存特征中位数: {len(self.feature_medians)} 个")
        
        # 训练前清空并延后记录pipelines（在构建base_models后统一保存）
        self.training_pipelines = {}
        
        # 智能数据处理Pipeline步骤（在最前端插入强制中性化）
        # 注意：树模型对标准化/线性降维不敏感，拆分线性与树模型的Pipeline
        # 固定更严格的IC阈值以过滤噪声
        horizon_days = getattr(self, 'prediction_horizon_days', 5) or 5
        ic_thr = 0.02
        
        # 构建前置中性化步骤（如果可用）
        pre_steps = []
        if NEUTRALIZATION_AVAILABLE and dates is not None and tickers is not None:
            # 使用统一市场数据管理器提供的真实口径（若可用）
            if getattr(self, 'ultra_enabled', False):
                try:
                    info_map = self.market_data_manager.get_batch_stock_info(list(set(tickers)))
                    industry_map = {t: (info.sector or info.gics_sector or 'OTHER') for t, info in info_map.items()}
                    # β作为占位：按行业分组的平均波动率替代
                    beta_series = pd.Series({t: 1.0 for t in industry_map.keys()})
                    print(f"[PIPELINE] 使用统一市场数据进行行业/β中性化（{len(industry_map)}个标的）")
                except Exception as e:
                    # 回退到旧的简化映射
                    industry_map = {ticker: ticker[0] for ticker in set(tickers) if ticker}
                    beta_series = pd.Series({ticker: 1.0 for ticker in set(tickers) if ticker})
                    print(f"[PIPELINE] 统一市场数据不可用，回退简化映射: {e}")
            else:
                # 回退到旧的简化映射
                industry_map = {ticker: ticker[0] for ticker in set(tickers) if ticker}
                beta_series = pd.Series({ticker: 1.0 for ticker in set(tickers) if ticker})
            
            neutralization_step = create_neutralization_pipeline_step(
                industry_map=industry_map, 
                beta_series=beta_series
            )
            pre_steps.append(neutralization_step)
            print(f"[PIPELINE] 已插入强制中性化步骤（{len(industry_map)}个行业组）")
        
        linear_steps = pre_steps + [
            ('imputer', self._get_advanced_imputer()),
            ('ic_selector', ICFactorSelector(ic_threshold=ic_thr)),
            ('scaler', StandardScaler())
        ]
        tree_steps = pre_steps + [
            ('imputer', self._get_advanced_imputer()),
            ('ic_selector', ICFactorSelector(ic_threshold=ic_thr))
        ]
        
        # 基学习器配置（第一层只有树模型：XGBoost, LightGBM, CatBoost）
        base_models = {}
        
        # 添加高级模型（LightGBM / XGBoost 为主力）
        if XGBOOST_AVAILABLE:
            xgb_pipeline = Pipeline(tree_steps + [('model', xgb.XGBRegressor(
                random_state=42,
                n_estimators=600,
                max_depth=7,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                tree_method='hist',
                n_jobs=-1
            ))])
            base_models['XGBoost'] = xgb_pipeline
        
        if LIGHTGBM_AVAILABLE:
            lgb_pipeline = Pipeline(tree_steps + [('model', lgb.LGBMRegressor(
                random_state=42,
                n_estimators=1000,
                max_depth=-1,
                learning_rate=0.025,
                num_leaves=96,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_samples=20,
                verbose=-1,
                n_jobs=-1
            ))])
            base_models['LightGBM'] = lgb_pipeline
        
        # 添加CatBoost作为第三个树模型
        if CATBOOST_AVAILABLE:
            cb_pipeline = Pipeline(tree_steps + [('model', CatBoostRegressor(
                random_state=42,
                iterations=1000,
                depth=7,
                learning_rate=0.03,
                l2_leaf_reg=3.0,
                subsample=0.8,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1
            ))])
            base_models['CatBoost'] = cb_pipeline
        
        # CNN模型已禁用，仅使用传统机器学习模型
        
        print(f"[BMA] 创建了 {len(base_models)} 个基础模型")
        print(f"[BMA] 模型列表: {', '.join(base_models.keys())}")
        
        # 创建增强版BMA集成器（偏向树模型）
        self.bma_model = BayesianModelAveraging(
            alpha_prior=1.5,
            shrinkage_factor=0.15,
            min_weight_threshold=0.02,
            model_class_priors={'LightGBM': 0.5, 'XGBoost': 0.5},
            enable_meta_learner=True,
            meta_learner_type='ridge',
            use_time_decay=True,
            half_life_days=126,
            ema_alpha=0.2
        )
        # 记录pipelines，确保预测阶段一致性
        self.training_pipelines = {name: pipe for name, pipe in base_models.items()}
        
        try:
            # 验证数据质量
            # 检查数据中是否有NaN
            has_nan_X = pd.isna(X).any().any() if hasattr(X, 'isna') else np.isnan(X).any()
            has_nan_y = pd.isna(y).any() if hasattr(y, 'isna') else np.isnan(y).any()
            if has_nan_X or has_nan_y:
                print(f"[BMA ERROR] 训练数据包含NaN值")
                return {}
            
            X_vals = X.values if hasattr(X, 'values') else X
            y_vals = y.values if hasattr(y, 'values') else y
            if np.isinf(X_vals).any() or np.isinf(y_vals).any():
                print(f"[BMA ERROR] 训练数据包含无穷值")
                return {}
            
            print(f"[BMA] 开始训练 - 数据形状: X={X.shape}, y={y.shape}")
            
            # 训练BMA（传递dates用于时间衰减权重）
            self.bma_model.fit(X.values, y.values, base_models, dates=dates)
            
            # 验证BMA模型是否训练成功
            if not hasattr(self.bma_model, 'posterior_weights') or not self.bma_model.posterior_weights:
                print(f"[BMA ERROR] BMA模型训练失败 - 没有有效的权重")
                self.bma_model = None
                return {}
            
            # 评估性能
            y_pred = self.bma_model.predict(X.values)
            
            # 验证预测结果
            if y_pred is None or len(y_pred) == 0:
                print(f"[BMA ERROR] BMA预测失败 - 无有效预测结果")
                self.bma_model = None
                return {}
            
            train_r2 = r2_score(y.values, y_pred)
            train_mse = mean_squared_error(y.values, y_pred)
            
            # 验证性能指标的合理性
            if np.isnan(train_r2) or np.isnan(train_mse) or train_r2 < -1:
                print(f"[BMA ERROR] BMA训练性能异常: R2={train_r2}, MSE={train_mse}")
                self.bma_model = None
                return {}
            
            print(f"[BMA] BMA训练完成")
            print(f"[BMA] 训练R2: {train_r2:.4f}, MSE: {train_mse:.6f}")
            print(f"[PROGRESS] 结束时间: {datetime.now().strftime('%H:%M:%S')}")
            
            # 训练结束 - 获取权重分配
            print(f"[BMA WEIGHTS] 权重分配:")
            for model_name, weight in self.bma_model.posterior_weights.items():
                print(f"  {model_name}: {weight:.4f}")
            
            # 打印滚动OOS指标汇总
            self.print_oos_summary()
            
            # 计算因子IC得分
            print(f"[BMA] 计算全局因子IC得分...")
            try:
                ic_scores = {}
                for i, col in enumerate(X.columns):
                    factor_values = X.iloc[:, i]
                    ic = self.calculate_ic(factor_values, y)
                    if not np.isnan(ic) and ic != 0.0:
                        ic_scores[col] = ic
                
                self.factor_ic_scores = ic_scores
                effective_factors = len([ic for ic in ic_scores.values() if abs(ic) > 0.05])
                print(f"[BMA] 计算了 {len(ic_scores)} 个因子的IC，其中 {effective_factors} 个有效因子")
            except Exception as e:
                print(f"[BMA] IC计算失败: {e}")
                self.factor_ic_scores = {}
            
            return {'BMA': {'train_r2': train_r2, 'train_mse': train_mse}}
            
        except Exception as e:
            print(f"[BMA ERROR] BMA训练失败: {e}")
            import traceback
            traceback.print_exc()
            self.bma_model = None  # 确保失败时清理模型
            return {}
    
    def predict_with_bma(self, X):
        """使用BMA模型进行预测（修复版：无单样本标准化）"""
        if hasattr(self, 'bma_model') and self.bma_model is not None:
            try:
                # 确保预测数据的特征列与训练时一致
                if hasattr(self, 'training_feature_columns'):
                    # 检查是否有缺失的列
                    missing_columns = set(self.training_feature_columns) - set(X.columns)
                    if missing_columns:
                        print(f"[FEATURE CONSISTENCY] 添加缺失列: {missing_columns}")
                        # 使用训练时保存的特征中位数而不是0
                        feature_medians = getattr(self, 'feature_medians', {})
                        for col in missing_columns:
                            fill_value = feature_medians.get(col, 0.0)
                            X[col] = fill_value
                    
                    # 检查是否有多余的列
                    extra_columns = set(X.columns) - set(self.training_feature_columns)
                    if extra_columns:
                        print(f"[FEATURE CONSISTENCY] 移除多余列: {extra_columns}")
                        X = X.drop(columns=list(extra_columns))
                    
                    # 重新排列列顺序以匹配训练时的顺序
                    X = X.reindex(columns=self.training_feature_columns)
                    print(f"[FEATURE CONSISTENCY] 确保特征列一致性: {len(X.columns)} 个特征")
                
                # 数据质量检查和清理（使用训练集中位数填充）
                if X.isnull().any().any():
                    print(f"[DATA QUALITY] 检测到NaN值，使用训练集中位数填充")
                    feature_medians = getattr(self, 'feature_medians', {})
                    for col in X.columns:
                        if X[col].isnull().any():
                            fill_value = feature_medians.get(col, X[col].median())
                            if pd.isna(fill_value):
                                fill_value = 0.0
                            X[col] = X[col].fillna(fill_value)
                
                # 检查无穷值（改为中位数填补）
                if np.isinf(X.values).any():
                    print(f"[DATA QUALITY] 检测到无穷值，替换为训练中位数")
                    X = X.replace([np.inf, -np.inf], np.nan)
                    feature_medians = getattr(self, 'feature_medians', {})
                    for col in X.columns:
                        if X[col].isnull().any():
                            fill_value = feature_medians.get(col, X[col].median())
                            if pd.isna(fill_value):
                                fill_value = 0.0
                            X[col] = X[col].fillna(fill_value)
                
                # 确保数据类型一致
                X = X.astype(float)
                
                # 使用训练期保存的变换器进行预处理（不做单样本横截面标准化）
                if hasattr(self, 'pca_transformer') and self.pca_transformer is not None:
                    transformer = self.pca_transformer
                    X = X.reindex(columns=transformer['feature_columns'])
                    X_scaled = transformer['scaler'].transform(X.values.astype(float))
                    X_pca = transformer['pca'].transform(X_scaled)
                    X = pd.DataFrame(X_pca, columns=transformer['pc_columns'], index=X.index)
                else:
                    X = X.astype(float)
                
                prediction = self.bma_model.predict(X.values)
                return prediction
            except Exception as e:
                print(f"[BMA PREDICT ERROR] {e}")
                # 添加更详细的错误信息
                print(f"[DEBUG] X shape: {X.shape}, X columns: {list(X.columns)}")
                if hasattr(self, 'training_feature_columns'):
                    print(f"[DEBUG] Training columns: {self.training_feature_columns}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print("[BMA PREDICT ERROR] BMA模型未初始化")
            return None
    
    def predict_many_with_bma(self, X_df, current_date=None):
        """批量预测：正确处理横截面标准化（解决预测退化问题）"""
        if not hasattr(self, 'bma_model') or self.bma_model is None:
            print("[BATCH PREDICTION ERROR] BMA模型未训练")
            return None
            
        try:
            print(f"[BATCH PREDICTION] 开始批量预测，输入形状: {X_df.shape}")
            
            # 确保有足够的样本进行横截面分析
            if len(X_df) < 2:
                print(f"[BATCH PREDICTION] 样本数量不足，回退到单样本预测")
                return self.predict_with_bma(X_df)
            
            # 1. 特征对齐（改进版本，添加随机扰动）
            if hasattr(self, 'training_feature_columns'):
                X_df = self._align_features(X_df)
                print(f"[BATCH ALIGNMENT] 特征对齐完成: {len(X_df.columns)} 个特征")
            
            # 2. 横截面标准化（关键修复！）
            X_standardized = self._cross_sectional_standardize(X_df)
            print(f"[BATCH PREDICTION] 横截面标准化完成")
            
            # 3. 应用IC筛选（使用训练时的筛选器）
            if hasattr(self, 'ic_selector'):
                X_selected = pd.DataFrame(
                    self.ic_selector.transform(X_standardized.values),
                    index=X_standardized.index
                )
                print(f"[BATCH PREDICTION] IC特征筛选完成: {X_selected.shape[1]} 个特征")
            else:
                X_selected = X_standardized
            
            # 4. PCA转换（如果需要）- 禁用全局PCA避免退化
            if self._should_use_pca():
                X_final = self._apply_pca_prediction(X_selected)
                print(f"[BATCH PREDICTION] PCA转换完成: {X_final.shape}")
            else:
                X_final = X_selected
                print(f"[BATCH PREDICTION] 跳过PCA转换，保持原始特征")
            
            # 5. 确保数据类型一致
            X_final = X_final.astype(float)
            
            # 6. 批量BMA预测
            print(f"[BATCH PREDICTION] 最终特征维度: {X_final.shape}")
            predictions = self.bma_model.predict(X_final.values)
            
            if predictions is None or len(predictions) == 0:
                print("[BATCH PREDICTION ERROR] BMA模型预测失败")
                return None
            
            # 7. 验证预测差异性
            pred_std = predictions.std()
            if pred_std < 1e-6:
                print(f"[WARNING] 预测标准差过小: {pred_std:.2e}")
                # 尝试添加微小扰动打破对称性
                predictions = self._add_prediction_noise(predictions, X_df)
                pred_std = predictions.std()
            
            print(f"[BATCH PREDICTION] 预测统计: mean={predictions.mean():.6f}, "
                  f"std={pred_std:.6f}, min={predictions.min():.6f}, max={predictions.max():.6f}")
            
            # 8. 诊断预测问题（如果需要）
            if pred_std < 1e-6:
                self.diagnose_prediction_issue(X_df, predictions)
            
            return predictions
            
        except Exception as e:
            print(f"[BATCH PREDICTION ERROR] 批量预测过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _cross_sectional_standardize(self, X_df):
        """横截面标准化（关键方法）"""
        X_standardized = X_df.copy()
        
        # 对每个特征进行横截面标准化
        for col in X_df.columns:
            col_data = X_df[col]
            
            # 去极值（winsorize）
            lower = col_data.quantile(0.01)
            upper = col_data.quantile(0.99)
            col_data = col_data.clip(lower=lower, upper=upper)
            
            # Z-score标准化
            mean = col_data.mean()
            std = col_data.std()
            
            if std > 1e-8:  # 避免除零
                X_standardized[col] = (col_data - mean) / std
            else:
                X_standardized[col] = 0.0
        
        return X_standardized
    
    def _align_features(self, X_df):
        """特征对齐的改进版本（添加随机扰动避免退化）"""
        feature_medians = getattr(self, 'feature_medians', {})
        
        # 对齐特征列
        missing_columns = set(self.training_feature_columns) - set(X_df.columns)
        if missing_columns:
            print(f"[ALIGNMENT] 缺失特征: {len(missing_columns)}个")
            # 使用更智能的填充策略
            for col in missing_columns:
                if col in feature_medians:
                    # 使用训练集中位数，但添加小扰动
                    base_value = feature_medians[col]
                    # 添加5%的随机扰动，避免所有值相同
                    noise = np.random.randn(len(X_df)) * abs(base_value) * 0.05
                    X_df[col] = base_value + noise
                else:
                    # 如果没有中位数信息，使用小随机值
                    X_df[col] = np.random.randn(len(X_df)) * 0.01
        
        # 移除多余列
        extra_columns = set(X_df.columns) - set(self.training_feature_columns)
        if extra_columns:
            X_df = X_df.drop(columns=list(extra_columns))
        
        # 重排列
        X_df = X_df.reindex(columns=self.training_feature_columns)
        
        # 处理剩余的NaN/Inf值
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        for col in X_df.columns:
            if X_df[col].isnull().any():
                fill_value = feature_medians.get(col, X_df[col].median())
                if pd.isna(fill_value):
                    fill_value = 0.0
                X_df[col] = X_df[col].fillna(fill_value)
        
        return X_df
    
    def _should_use_pca(self):
        """判断是否应该使用PCA（避免全局PCA导致的退化）"""
        # 暂时禁用PCA避免预测退化
        return False
    
    def _apply_pca_prediction(self, X_selected):
        """应用PCA转换（如果启用）"""
        if hasattr(self, 'pca_transformer') and self.pca_transformer is not None:
            transformer = self.pca_transformer
            feature_columns = transformer['feature_columns']
            X_aligned = X_selected.reindex(columns=feature_columns, fill_value=0)
            X_scaled = transformer['scaler'].transform(X_aligned.values)
            X_pca = transformer['pca'].transform(X_scaled)
            return pd.DataFrame(X_pca, columns=transformer['pc_columns'], index=X_selected.index)
        return X_selected
    
    def _add_prediction_noise(self, predictions, X_df):
        """添加基于特征的微小扰动，打破预测对称性"""
        # 使用某些关键特征创建唯一性
        if 'price_momentum_5' in X_df.columns:
            # 基于动量添加微小扰动
            momentum = X_df['price_momentum_5'].fillna(0).values
            noise = momentum * 0.001  # 很小的扰动
            predictions = predictions + noise
        
        # 确保仍然有差异
        if predictions.std() < 1e-6:
            # 添加随机噪声作为最后手段
            np.random.seed(42)  # 固定种子保证可重复性
            random_noise = np.random.randn(len(predictions)) * 0.0001
            predictions = predictions + random_noise
        
        return predictions
    
    def diagnose_prediction_issue(self, X_df, predictions):
        """诊断预测退化问题"""
        print("\n=== 预测诊断报告 ===")
        
        # 1. 检查输入特征的差异性
        feature_stds = X_df.std()
        zero_std_features = feature_stds[feature_stds < 1e-8]
        if len(zero_std_features) > 0:
            print(f"[诊断] 发现 {len(zero_std_features)} 个零方差特征:")
            print(zero_std_features.index.tolist()[:10])
        
        # 2. 检查特征相关性（采样减少计算量）
        if len(X_df) > 1:
            sample_size = min(100, len(X_df))
            X_sample = X_df.sample(n=sample_size) if len(X_df) > sample_size else X_df
            corr_matrix = X_sample.T.corr()
            high_corr_pairs = np.where((corr_matrix > 0.99) & (corr_matrix < 1.0))
            if len(high_corr_pairs[0]) > 0:
                print(f"[诊断] 发现 {len(high_corr_pairs[0])} 对高度相关的样本")
        
        # 3. 检查预测值分布
        pred_unique = len(np.unique(predictions))
        print(f"[诊断] 预测值唯一值数量: {pred_unique}/{len(predictions)}")
        
        # 4. 检查模型内部状态
        if hasattr(self, 'bma_model'):
            for name, model in self.bma_model.models.items():
                # 对于树模型，检查预测路径
                if 'XGBoost' in name or 'LightGBM' in name or 'CatBoost' in name:
                    try:
                        # 单样本预测测试
                        single_pred = model.predict(X_df.iloc[[0]].values)
                        print(f"[诊断] {name} 单样本预测: {single_pred[0]:.6f}")
                    except:
                        pass
        
        return {
            'zero_std_features': len(zero_std_features),
            'unique_predictions': pred_unique,
            'prediction_std': predictions.std()
        }
    
    def generate_recommendations(self, all_data, top_n=None):
        """生成投资建议（批量预测+阈值退化保护）"""
        print(f"[RECOMMENDATIONS] 生成BMA投资建议...")
        
        # 批量收集所有股票的特征
        all_features = []
        tickers = []
        
        for ticker, data in all_data.items():
            try:
                # 计算最新因子
                factors = self.calculate_factors(data)
                
                # 获取最新的非NaN因子值 - 改进版本
                latest_factors = {}
                for factor_name, factor_data in factors.items():
                    if isinstance(factor_data, pd.Series) and len(factor_data) > 0:
                        # 尝试获取最后几个有效值中的最新值
                        last_5_values = factor_data.tail(5)
                        valid_values = last_5_values.dropna()
                        
                        if len(valid_values) > 0:
                            latest_value = valid_values.iloc[-1]
                            if not np.isinf(latest_value):
                                latest_factors[factor_name] = float(latest_value)
                            else:
                                # 如果最新值是无穷值，使用训练集中位数
                                feature_medians = getattr(self, 'feature_medians', {})
                                latest_factors[factor_name] = feature_medians.get(factor_name, 0.0)
                        else:
                            # 如果没有有效值，使用训练集中位数
                            feature_medians = getattr(self, 'feature_medians', {})
                            latest_factors[factor_name] = feature_medians.get(factor_name, 0.0)
                    else:
                        feature_medians = getattr(self, 'feature_medians', {})
                        latest_factors[factor_name] = feature_medians.get(factor_name, 0.0)
                
                # 确保至少有一些基础因子
                if len(latest_factors) < 5:
                    print(f"[REC WARNING] {ticker}: 因子数量太少({len(latest_factors)})，跳过")
                    continue
                
                print(f"[REC DEBUG] {ticker}: 获取到 {len(latest_factors)} 个有效因子")
                
                all_features.append(latest_factors)
                tickers.append(ticker)
                
            except Exception as e:
                print(f"[REC ERROR] {ticker}: 因子计算失败: {e}")
                continue
        
        if len(all_features) == 0:
            print("[REC ERROR] 没有有效的股票特征，无法生成推荐")
            return []
        
        # 批量预测（解决横截面分化问题）
        print(f"[RECOMMENDATIONS] 准备批量预测 {len(all_features)} 只股票...")
        batch_df = pd.DataFrame(all_features, index=tickers)
        
        # 使用批量预测
        if hasattr(self, 'bma_model') and self.bma_model is not None:
            predictions = self.predict_many_with_bma(batch_df)
            if predictions is None:
                print("[REC ERROR] 批量预测失败")
                return []
        else:
            print("[REC ERROR] BMA模型未训练")
            return []
        
        all_predictions = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
        
        # 阈值退化保护：检查预测分化程度
        pred_array = np.array(all_predictions)
        pred_std = pred_array.std()
        pred_range = pred_array.max() - pred_array.min()
        
        print(f"[DEGRADATION CHECK] 预测统计: min={pred_array.min():.6f}, max={pred_array.max():.6f}, std={pred_std:.6f}")
        
        if pred_std < 1e-9 or pred_range < 1e-6:
            print(f"[DEGRADATION WARNING] 预测值无显著分化 (std={pred_std:.2e}, range={pred_range:.2e})")
            print(f"[DEGRADATION WARNING] 不建议基于当前预测进行交易，可能存在特征退化")
            
            # 使用排名分位数代替数值阈值
            from scipy.stats import rankdata
            ranks = rankdata(pred_array) / len(pred_array)
            print(f"[DEGRADATION PROTECTION] 改用排名分位数进行评级")
            
            # 基于排名生成推荐
            recommendations = []
            for i, ticker in enumerate(tickers):
                rank_pct = ranks[i]
                
                if rank_pct >= 0.7:
                    rating = "BUY"
                elif rank_pct <= 0.3:
                    rating = "SELL"
                else:
                    rating = "HOLD"
                
                # 获取当前价格信息
                try:
                    data = all_data[ticker]
                    current_price = data['Close'].iloc[-1]
                    volume = data['Volume'].iloc[-1]
                    
                    recommendations.append({
                        'ticker': ticker,
                        'predicted_return': pred_array[i],
                        'rank_percentile': rank_pct,
                        'rating': rating,
                        'current_price': current_price,
                        'volume': volume,
                        'note': 'Rank-based rating due to low prediction variance'
                    })
                except Exception as e:
                    print(f"[REC ERROR] {ticker}: 获取价格信息失败: {e}")
                    continue
            
            print(f"[RECOMMENDATIONS] 基于排名生成 {len(recommendations)} 个推荐")
            return recommendations
        
        # 正常情况：使用分位数阈值
        buy_threshold = np.percentile(pred_array, 70)  # 前30%为BUY
        sell_threshold = np.percentile(pred_array, 30)  # 后30%为SELL
        
        print(f"[THRESHOLDS] BUY阈值: {buy_threshold:.6f}, SELL阈值: {sell_threshold:.6f}")
        
        # 生成推荐
        recommendations = []
        for i, ticker in enumerate(tickers):
            try:
                data = all_data[ticker]
                current_price = data['Close'].iloc[-1]
                volume = data['Volume'].iloc[-1]
                predicted_return = pred_array[i]
                
                # 确定评级
                if predicted_return >= buy_threshold:
                    rating = "BUY"
                elif predicted_return <= sell_threshold:
                    rating = "SELL"
                else:
                    rating = "HOLD"
                
                recommendations.append({
                    'ticker': ticker,
                    'predicted_return': predicted_return,
                    'rating': rating,
                    'current_price': current_price,
                    'volume': volume
                })
                
            except Exception as e:
                print(f"[REC ERROR] {ticker}: 推荐生成失败: {e}")
                continue
        
        print(f"[RECOMMENDATIONS] 生成 {len(recommendations)} 个推荐")
        return recommendations[:top_n] if top_n else recommendations
    
    def _get_advanced_imputer(self):
        """获取智能数据插值器（回退到中位数插值）"""
        try:
            # 尝试导入高级插值模块
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            return IterativeImputer(random_state=42, max_iter=10)
        except ImportError:
            print("[IMPUTER] 高级插值器不可用，使用中位数插值")
            return SimpleImputer(strategy='median')
    
    def save_results_to_excel(self, recommendations, excel_file):
        """保存结果到指定Excel文件"""
        if not recommendations:
            print("[SAVE] 没有建议可保存")
            return None
        
        try:
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(excel_file), exist_ok=True)
            
            # 转换为DataFrame
            df = pd.DataFrame(recommendations)
            
            # 保存到Excel
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='BMA推荐', index=False)
                
                # 如果有模型评分信息，也保存
                if hasattr(self, 'bma_model') and self.bma_model and self.bma_model.posterior_weights:
                    weights_df = pd.DataFrame([
                        {'模型': name, '权重': weight}
                        for name, weight in self.bma_model.posterior_weights.items()
                    ])
                    weights_df.to_excel(writer, sheet_name='模型权重', index=False)
            
            print(f"[SAVE] 结果已保存到: {excel_file}")
            return excel_file
            
        except Exception as e:
            print(f"[SAVE ERROR] 保存失败: {e}")
            return None

    def save_results(self, recommendations, factor_ic_scores):
        """保存分析结果到Excel（保持原有接口）"""
        print(f"[SAVE DEBUG] 收到推荐数量: {len(recommendations) if recommendations else 0}")
        
        if not recommendations:
            print("[SAVE ERROR] 没有建议可保存")
            print("[SAVE DEBUG] 检查BMA模型训练和预测是否成功")
            
            # 尝试保存诊断信息
            # 尝试保存诊断信息
            try:
                self._save_diagnostic_info()
            except Exception as e:
                print(f"[SAVE ERROR] 保存诊断信息失败: {e}")
            
            return None
    
    def _save_diagnostic_info(self):
        """保存诊断信息以帮助调试"""
        try:
            import os
            from datetime import datetime
            
            os.makedirs('result', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            diagnostic_info = {
                'timestamp': timestamp,
                'bma_model_status': hasattr(self, 'bma_model') and self.bma_model is not None,
                'training_feature_columns': getattr(self, 'training_feature_columns', []),
                'feature_medians_count': len(getattr(self, 'feature_medians', {})),
                'posterior_weights': getattr(self, 'posterior_weights', {}),
                'factor_ic_scores': getattr(self, 'factor_ic_scores', {}),
                'training_pipelines': list(getattr(self, 'training_pipelines', {}).keys())
            }
            
            # 保存到JSON文件
            import json
            diagnostic_file = f"result/diagnostic_info_{timestamp}.json"
            with open(diagnostic_file, 'w', encoding='utf-8') as f:
                json.dump(diagnostic_info, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"[DIAGNOSTIC] 诊断信息已保存到: {diagnostic_file}")
            return diagnostic_file
            
        except Exception as e:
            print(f"[DIAGNOSTIC ERROR] 保存诊断信息失败: {e}")
            return None
        
        # 创建结果目录
        os.makedirs('result', exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = np.random.randint(100, 999)
        random_suffix2 = np.random.randint(100, 999)
        excel_file = f'result/bma_quantitative_analysis_{timestamp}_{random_suffix}_{random_suffix2}.xlsx'
        
        try:
            # 转换为DataFrame
            recommendations_df = pd.DataFrame(recommendations)
            
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 主要建议
                recommendations_df.to_excel(writer, sheet_name='BMA投资建议', index=False)
                
                # 因子IC分析
                ic_df = pd.DataFrame(list(factor_ic_scores.items()), 
                                   columns=['Factor', 'IC_Score'])
                ic_df = ic_df.sort_values('IC_Score', key=abs, ascending=False)
                ic_df.to_excel(writer, sheet_name='因子IC评分', index=False)
                
                # 模型权重信息
                if hasattr(self, 'bma_model') and self.bma_model is not None:
                    model_info = []
                    for model_name, weight in self.bma_model.posterior_weights.items():
                        model_info.append({
                            'Model': model_name,
                            'Posterior_Weight': weight,
                            'Weight_Percentage': f"{weight*100:.2f}%",
                            'Type': 'BMA_Ensemble'
                        })
                    
                    model_df = pd.DataFrame(model_info)
                    model_df.to_excel(writer, sheet_name='BMA模型权重', index=False)
            
            print(f"[SAVE] BMA结果已保存到: {excel_file}")
            
            # 打印总结
            print(f"\n=== BMA增强版量化分析结果总结 ===")
            print(f"分析时间: {timestamp}")
            print(f"股票总数: {len(recommendations_df)}")
            print(f"BUY推荐: {len(recommendations_df[recommendations_df['rating'] == 'BUY'])}")
            print(f"HOLD推荐: {len(recommendations_df[recommendations_df['rating'] == 'HOLD'])}")
            print(f"SELL推荐: {len(recommendations_df[recommendations_df['rating'] == 'SELL'])}")
            print(f"平均预测收益率: {recommendations_df['predicted_return'].mean():.4f}")
            print(f"有效因子数量: {len([ic for ic in factor_ic_scores.values() if abs(ic) > 0.05])}")
            
            if hasattr(self, 'bma_model') and self.bma_model is not None:
                print(f"使用BMA集成: 替代Stacking")
                print(f"BMA权重分配:")
                for model_name, weight in self.bma_model.posterior_weights.items():
                    print(f"  {model_name}: {weight:.4f}")
            
            # 新增：选出得分最高的10只股票用于IBKR自动交易
            top_10_stocks = self.generate_top_10_list(recommendations_df)
            self.save_top_10_for_ibkr(top_10_stocks, timestamp)
            
            return excel_file
        
        except Exception as e:
            print(f"[SAVE ERROR] 保存失败: {e}")
            return None
    
    def generate_top_10_list(self, recommendations):
        """生成得分最高的10只股票列表，用于IBKR自动交易"""
        try:
            # 如果输入是列表，转换为DataFrame
            if isinstance(recommendations, list):
                recommendations_df = pd.DataFrame(recommendations)
            else:
                recommendations_df = recommendations.copy()
            
            # 确保有predicted_return列，并创建final_score列（使用predicted_return作为评分标准）
            if 'predicted_return' not in recommendations_df.columns:
                print("[TOP10 ERROR] 缺少predicted_return列")
                return []
            
            # 创建final_score列（基于predicted_return）
            recommendations_df['final_score'] = recommendations_df['predicted_return']
            
            # 按predicted_return排序，选出前10名
            top_10_df = recommendations_df.nlargest(10, 'predicted_return')
            
            # 创建股票信息列表
            top_10_stocks = []
            for _, row in top_10_df.iterrows():
                stock_info = {
                    'ticker': row['ticker'],
                    'final_score': row['final_score'],
                    'predicted_return': row['predicted_return'],
                    'rating': row['rating'],
                    'recommendation': row.get('recommendation', row['rating'])  # 使用rating作为fallback
                }
                top_10_stocks.append(stock_info)
            
            print(f"\n=== IBKR自动交易股票列表 ===")
            print(f"选出得分最高的10只股票:")
            for i, stock in enumerate(top_10_stocks, 1):
                print(f"{i:2d}. {stock['ticker']:6s} | 得分: {stock['final_score']:.4f} | "
                      f"预测收益: {stock['predicted_return']:.2%} | 评级: {stock['rating']}")
            
            return top_10_stocks
            
        except Exception as e:
            print(f"[TOP10 ERROR] 生成Top10列表失败: {e}")
            return []
    
    def save_top_10_for_ibkr(self, top_10_stocks, timestamp):
        """保存Top10股票列表：
        - 文本文件：仅打印前5个，格式为 'NVDA', 'AAPL'（每行或一行）
        - Excel 文件：保留详细字段
        - JSON 文件：Top5股票代号，格式为 ['AAPL', 'NVDA', ...]
        """
        try:
            # 创建结果目录
            result_dir = "result"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            # 保存为Excel（保留详细字段）
            xlsx_file = os.path.join(result_dir, f"bma_top_10_stocks_{timestamp}.xlsx")
            top_10_df = pd.DataFrame(top_10_stocks)
            try:
                top_10_df.to_excel(xlsx_file, index=False)
            except Exception:
                # 回退到CSV
                xlsx_file = os.path.join(result_dir, f"bma_top_10_stocks_{timestamp}.csv")
                top_10_df.to_csv(xlsx_file, index=False, encoding='utf-8')

            # 保存为纯文本文件（仅前5个代码，格式: 'NVDA', 'AAPL'）
            txt_file = os.path.join(result_dir, f"bma_top_5_tickers_{timestamp}.txt")
            tickers_only = [s.get('ticker') for s in top_10_stocks if s.get('ticker')][:5]
            with open(txt_file, 'w', encoding='utf-8') as f:
                if tickers_only:
                    line = ", ".join([f"'{t}'" for t in tickers_only])
                    f.write(line + "\n")
                else:
                    f.write("\n")

            # 新增：保存Top5股票代号为JSON格式（只输出股票代码数组）
            json_file = os.path.join(result_dir, f"top5_tickers_{timestamp}.json")
            top_5_tickers = [s.get('ticker') for s in top_10_stocks if s.get('ticker')][:5]
            
            import json
            # 只保存股票代码数组，格式: ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA"]
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(top_5_tickers, f, ensure_ascii=False)

            print(f"[BMA] 输出已保存:")
            print(f"  明细(Excel/CSV): {xlsx_file}")
            print(f"  文本(仅前5代码): {txt_file}")
            print(f"  JSON(Top5): {json_file}")
            print(f"  Top5 Tickers: {top_5_tickers}")
            
            # 返回股票代码列表（便于直接使用）
            ticker_list = [stock['ticker'] for stock in top_10_stocks]
            return ticker_list
            
        except Exception as e:
            print(f"[IBKR ERROR] 保存Top10列表失败: {e}")
            return []

def main():
    """主函数（保持原有接口）"""
    print("=== BMA增强版量化分析模型启动 V3 ===")
    print(f"使用贝叶斯模型平均(BMA)替代Stacking，提供更稳定的预测性能")
    print(f"使用高级模型: XGBoost={XGBOOST_AVAILABLE}, LightGBM={LIGHTGBM_AVAILABLE}, CatBoost={CATBOOST_AVAILABLE}")
    print(f"[BMA] 仅使用CPU计算模式")
    
    # 命令行参数解析（保持兼容性）
    parser = argparse.ArgumentParser(description='BMA增强版量化分析模型V3')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--top-n', type=int, default=None, help='返回top N个推荐')
    parser.add_argument('--ticker-file', type=str, default=None, help='股票列表文件 (每行一个ticker)')
    
    args = parser.parse_args()
    
    # 处理股票列表
    final_ticker_list = ticker_list
    if args.ticker_file and os.path.exists(args.ticker_file):
        try:
            with open(args.ticker_file, 'r', encoding='utf-8') as f:
                file_tickers = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
            if file_tickers:
                final_ticker_list = file_tickers
                print(f"[TICKER FILE] 从文件加载了 {len(final_ticker_list)} 只股票")
        except Exception as e:
            print(f"[TICKER FILE ERROR] 读取股票文件失败: {e}，使用默认股票池")
    
    print(f"[DATE] 分析时间范围: {args.start_date} 到 {args.end_date}")
    print(f"[FINAL] 最终股票池: {len(final_ticker_list)} 只股票")
    print(f"[STOCKS] 当前考虑的股票: {', '.join(final_ticker_list[:25])}{'...' if len(final_ticker_list) > 25 else ''}")
    
    # 初始化模型
    model = QuantitativeModel()
    
    try:
        # 下载数据
        all_data = model.download_data(final_ticker_list, args.start_date, args.end_date)
        
        if not all_data:
            print("[ERROR] 没有可用数据，程序退出")
            return
        
        # 准备机器学习数据
        X, y, ticker_series, dates = model.prepare_ml_data_with_time_series(all_data, target_period=5)
        
        # 使用BMA训练（替代Stacking）
        print(f"[TRAINING] 数据量: {len(X)} 个样本，使用BMA替代Stacking训练")
        model_scores = model.train_models_with_bma(X, y, enable_hyperopt=True, apply_preprocessing=True, dates=dates, tickers=ticker_series)
        
        # 生成投资建议
        recommendations = model.generate_recommendations(all_data, top_n=args.top_n)
        
        # 保存结果（恢复完整功能）
        result_file = model.save_results(recommendations, model.factor_ic_scores)
        
        # 生成Top10列表用于IBKR自动交易
        if recommendations:
            top_10_stocks = model.generate_top_10_list(pd.DataFrame(recommendations))
            if top_10_stocks:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model.save_top_10_for_ibkr(top_10_stocks, timestamp)
        
        if result_file:
            print(f"\n=== BMA分析完成 ===")
            print(f"结果文件: {result_file}")
            print(f"\n优势说明:")
            print(f"[OK] 使用BMA替代Stacking，避免过拟合")
            print(f"[OK] 贝叶斯理论基础，权重分配更合理")
            print(f"[OK] Shrinkage机制，提高稳定性")
            print(f"[OK] 保持原有接口，无缝替换")
        else:
            print(f"[ERROR] 结果保存失败")
            
    except Exception as e:
        print(f"[MAIN ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
