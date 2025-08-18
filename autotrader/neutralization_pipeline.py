#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强制中性化管线 - 在Pipeline最前端插入
逐日处理：winsorize → 标准化 → 行业/β中性化 → 正交化
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy.linalg import lstsq

class DailyNeutralizationTransformer(BaseEstimator, TransformerMixin):
    """逐日强制中性化转换器，在Pipeline最前端使用"""
    
    def __init__(self, 
                 winsorize_quantiles=(0.01, 0.99),
                 neutralize_industry=True,
                 neutralize_beta=True,
                  orthogonalize_method='schmidt',  # 'pca' or 'schmidt' or None
                 pca_var_ratio=0.95,
                 industry_map=None,
                 beta_series=None,
                 date_col='date',
                 ticker_col='ticker'):
        self.winsorize_quantiles = winsorize_quantiles
        self.neutralize_industry = neutralize_industry
        self.neutralize_beta = neutralize_beta
        self.orthogonalize_method = orthogonalize_method
        self.pca_var_ratio = pca_var_ratio
        self.industry_map = industry_map or {}
        if beta_series is None:
            self.beta_series = pd.Series(dtype=float)
        else:
            self.beta_series = beta_series
        self.date_col = date_col
        self.ticker_col = ticker_col
        
        # 记录训练期参数
        self.feature_names_in_ = None
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        """拟合过程：记录特征名称"""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = [c for c in X.columns 
                                    if c not in [self.date_col, self.ticker_col]]
        else:
            self.feature_names_in_ = [f'feature_{i}' for i in range(X.shape[1])]
            
        self.is_fitted_ = True
        return self
        
    def transform(self, X):
        """转换过程：单次 groupby(['date']) + transform 的高效中性化

        步骤：
        1) 分位数winsorize：一次性 groupby(date) 计算分位数向量并 clip
        2) 截面标准化：groupby(date).transform 做 Z-score
        3) 行业/β中性化与正交化：仅在需要时进行（保持逐日，但已极简）
        """
        if not self.is_fitted_:
            raise ValueError("Transformer not fitted. Call fit() first.")
            
        # 转为DataFrame处理
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
            
        # 检查必要列
        if self.date_col not in X.columns or self.ticker_col not in X.columns:
            # 如果没有日期/股票列，跳过中性化，只做标准化
            return self._simple_standardize(X)
            
        df = X.copy()
        feature_cols = [c for c in df.columns if c not in [self.date_col, self.ticker_col]]

        if not feature_cols:
            return self._simple_standardize(df)

        # 1) winsorize（向量化）：每日分位数 -> clip
        if self.winsorize_quantiles:
            lower_q, upper_q = self.winsorize_quantiles
            low = df.groupby(self.date_col)[feature_cols].transform(lambda g: g.quantile(lower_q))
            high = df.groupby(self.date_col)[feature_cols].transform(lambda g: g.quantile(upper_q))
            df[feature_cols] = df[feature_cols].clip(lower=low, upper=high)

        # 2) 截面Z-score（向量化）
        means = df.groupby(self.date_col)[feature_cols].transform('mean')
        stds = df.groupby(self.date_col)[feature_cols].transform(lambda g: g.std(ddof=0).replace(0, np.nan))
        df[feature_cols] = (df[feature_cols] - means) / (stds + 1e-12)

        # 3) 行业/β中性化（如启用）
        if (self.neutralize_industry or self.neutralize_beta) and len(df) > 1:
            # 逐日 apply，内部用最小代价回归
            df = df.groupby(self.date_col, group_keys=False).apply(
                lambda g: self._neutralize_group(g, [c for c in g.columns if c in feature_cols])
            )

        # 4) 正交化（可选）
        if self.orthogonalize_method and len(feature_cols) > 1:
            df = df.groupby(self.date_col, group_keys=False).apply(
                lambda g: self._orthogonalize_group(g, feature_cols, g[self.date_col].iloc[0])
            )

        # 返回特征矩阵
        return df[feature_cols]
            
    def _simple_standardize(self, X):
        """简单标准化（无日期信息时的回退）"""
        feature_cols = [c for c in X.columns 
                       if c not in [self.date_col, self.ticker_col]]
        X_features = X[feature_cols]
        
        # 横截面标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_features)
        return pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
    def _process_daily_group(self, group, date):
        """处理单日横截面数据"""
        feature_cols = [c for c in group.columns 
                       if c not in [self.date_col, self.ticker_col]]
        
        if len(feature_cols) == 0:
            return group
            
        group = group.copy()
        
        # 1. 去极值
        if self.winsorize_quantiles:
            lower, upper = self.winsorize_quantiles
            for col in feature_cols:
                q_low, q_high = group[col].quantile([lower, upper])
                group[col] = group[col].clip(lower=q_low, upper=q_high)
                
        # 2. 标准化（横截面）
        for col in feature_cols:
            mean_val = group[col].mean()
            std_val = group[col].std()
            if std_val > 1e-12:
                group[col] = (group[col] - mean_val) / std_val
                
        # 3. 行业/β中性化
        if (self.neutralize_industry or self.neutralize_beta) and len(group) > 1:
            group = self._neutralize_group(group, feature_cols)
            
        # 4. 正交化
        if self.orthogonalize_method and len(feature_cols) > 1:
            group = self._orthogonalize_group(group, feature_cols, date)
            
        return group
        
    def _neutralize_group(self, group, feature_cols):
        """行业/β中性化"""
        # 构建中性化变量
        neutralize_vars = []
        
        if self.neutralize_industry and self.industry_map:
            group['industry'] = group[self.ticker_col].map(self.industry_map).fillna('OTHER')
            # 行业虚拟变量
            industry_dummies = pd.get_dummies(group['industry'], prefix='ind')
            neutralize_vars.append(industry_dummies)
            
        if self.neutralize_beta and not self.beta_series.empty:
            group['beta'] = group[self.ticker_col].map(self.beta_series).fillna(0.0)
            neutralize_vars.append(group[['beta']])
            
        if not neutralize_vars:
            return group
            
        # 合并中性化变量
        X_neutral = pd.concat(neutralize_vars, axis=1).values
        
        # 对每个因子进行中性化
        for col in feature_cols:
            y = group[col].values
            try:
                coef, _, _, _ = lstsq(X_neutral, y, rcond=None)
                y_pred = X_neutral @ coef
                group[col] = y - y_pred  # 残差
            except Exception as e:
                # 记录中性化失败原因，便于诊断数据质量问题
                logging.warning(f"列 {col} 中性化失败: {e}")
                # 保持原始值，不进行中性化
                
        # 清理临时列
        group = group.drop(columns=['industry', 'beta'], errors='ignore')
        return group
        
    def _orthogonalize_group(self, group, feature_cols, date):
        """正交化处理"""
        if self.orthogonalize_method == 'pca':
            return self._pca_orthogonalize(group, feature_cols, date)
        elif self.orthogonalize_method == 'schmidt':
            return self._schmidt_orthogonalize(group, feature_cols)
        else:
            return group
            
    def _pca_orthogonalize(self, group, feature_cols, date):
        """PCA正交化"""
        X = group[feature_cols].values
        
        # 标准化
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-12] = 1.0
        X_scaled = (X - X_mean) / X_std
        
        # PCA
        try:
            pca = PCA(n_components=self.pca_var_ratio, svd_solver='full')
            X_pca = pca.fit_transform(X_scaled)
            
            # 创建主成分DataFrame
            pc_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
            pc_df = pd.DataFrame(X_pca, columns=pc_cols, index=group.index)
            
            # 保留原有的日期/股票列并添加主成分
            result = group[[self.date_col, self.ticker_col]].copy()
            result = pd.concat([result, pc_df], axis=1)
            
            return result
        except Exception as e:
            # PCA失败时记录错误并保持原样
            logging.warning(f"PCA降维失败: {e}")
            return group
            
    def _schmidt_orthogonalize(self, group, feature_cols):
        """施密特正交化"""
        X = group[feature_cols].values
        
        # 逐列正交化
        X_orth = X.copy()
        for i in range(1, X.shape[1]):
            for j in range(i):
                # X[:, i] 对 X[:, j] 的投影
                proj = np.dot(X_orth[:, i], X_orth[:, j]) / np.dot(X_orth[:, j], X_orth[:, j])
                X_orth[:, i] -= proj * X_orth[:, j]
                
        # 更新group中的特征列
        group[feature_cols] = X_orth
        return group

def create_neutralization_pipeline_step(industry_map=None, beta_series=None):
    """创建中性化管线步骤，可插入到sklearn Pipeline中"""
    return ('neutralization', DailyNeutralizationTransformer(
        winsorize_quantiles=(0.01, 0.99),
        neutralize_industry=industry_map is not None,
        neutralize_beta=beta_series is not None and len(beta_series) > 0,
        orthogonalize_method='schmidt',
        pca_var_ratio=0.95,
        industry_map=industry_map,
        beta_series=beta_series
    ))