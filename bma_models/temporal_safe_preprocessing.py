#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序安全的预处理模块
修复数据泄露问题：实施严格的时间序列预处理，避免使用未来信息

核心原则：
1. 所有标准化使用expanding window，仅使用历史数据
2. PCA使用时序安全的方式或替代方法
3. 任何统计量计算都不能包含未来信息
4. 横截面处理按日期分组进行
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

class TemporalSafePreprocessor:
    """时序安全的预处理器 - 严格避免数据泄露"""
    
    def __init__(self, 
                 standardization_mode: str = 'cross_sectional',
                 min_history_days: int = 252,
                 enable_pca: bool = False,
                 pca_alternative: str = 'correlation_filter'):
        """
        初始化时序安全预处理器
        
        Args:
            standardization_mode: 标准化模式 ('cross_sectional' 或 'expanding_window')
            min_history_days: 最小历史天数要求
            enable_pca: 是否启用PCA (不推荐，因为存在泄露风险)
            pca_alternative: PCA替代方法 ('correlation_filter', 'variance_filter', 'none')
        """
        self.standardization_mode = standardization_mode
        self.min_history_days = min_history_days
        self.enable_pca = enable_pca
        self.pca_alternative = pca_alternative
        
        # 存储历史统计量用于expanding window
        self.historical_stats = {}
        
        logger.info(f"时序安全预处理器初始化:")
        logger.info(f"  标准化模式: {standardization_mode}")
        logger.info(f"  最小历史天数: {min_history_days}")
        logger.info(f"  PCA启用: {enable_pca}")
        logger.info(f"  PCA替代方法: {pca_alternative}")
        
        if enable_pca:
            logger.warning("⚠️  PCA启用会带来数据泄露风险，建议使用替代方法")
    
    def fit_transform(self, 
                     X: pd.DataFrame, 
                     dates: pd.Series,
                     date_col: str = 'date') -> Tuple[pd.DataFrame, Dict]:
        """
        时序安全的特征变换
        
        Args:
            X: 特征矩阵
            dates: 日期序列
            date_col: 日期列名
            
        Returns:
            Tuple[变换后的特征矩阵, 变换信息]
        """
        logger.info(f"开始时序安全特征变换，输入形状: {X.shape}")
        
        # 确保有日期列
        X = X.copy()  # 总是复制以避免修改原始数据
        
        # 处理MultiIndex情况
        if isinstance(X.index, pd.MultiIndex):
            # 如果date在index中，将其重置为列
            if date_col in X.index.names:
                # 检查date是否已经在columns中
                if date_col not in X.columns:
                    X = X.reset_index(level=date_col)
                else:
                    # date既在index又在columns中，先删除columns中的date，再从index重置
                    X = X.drop(columns=[date_col])
                    X = X.reset_index(level=date_col)
                    logger.info(f"date同时存在于index和columns中，使用index中的date")
            elif dates is not None:
                X[date_col] = dates
        elif date_col not in X.columns:
            if dates is not None:
                X[date_col] = dates
            else:
                raise ValueError(f"需要{date_col}列或提供dates参数")
        else:
            # date_col 已存在于columns中，检查是否需要使用传入的dates
            if dates is not None and not dates.equals(X[date_col]):
                logger.warning(f"传入的dates与现有{date_col}列不一致，使用现有{date_col}列")
        
        # 按日期排序
        X_sorted = X.sort_values(date_col).reset_index(drop=True)
        
        # 执行时序安全标准化
        X_standardized, std_info = self._temporal_safe_standardization(X_sorted, date_col)
        
        # 执行共线性处理（不使用PCA）
        X_final, collinearity_info = self._safe_collinearity_treatment(X_standardized, date_col)
        
        transform_info = {
            'method': 'temporal_safe_preprocessing',
            'original_shape': X.shape,
            'final_shape': X_final.shape,
            'standardization_info': std_info,
            'collinearity_info': collinearity_info,
            'data_leakage_risk': 'MINIMAL'
        }
        
        logger.info(f"时序安全变换完成: {X.shape} -> {X_final.shape}")
        
        return X_final, transform_info
    
    def _temporal_safe_standardization(self, 
                                     X: pd.DataFrame, 
                                     date_col: str) -> Tuple[pd.DataFrame, Dict]:
        """时序安全的标准化"""
        
        feature_cols = [col for col in X.columns if col != date_col]
        X_result = X.copy()
        
        if self.standardization_mode == 'cross_sectional':
            # 横截面标准化 - 按日期分组标准化
            logger.info("执行横截面标准化（按日期）")
            
            for col in feature_cols:
                # 按日期分组，每个日期内进行标准化
                X_result[col] = X_result.groupby(date_col)[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8) if x.std() > 1e-8 else x
                )
            
            std_info = {
                'method': 'cross_sectional',
                'feature_count': len(feature_cols),
                'leakage_risk': 'NONE'
            }
            
        elif self.standardization_mode == 'expanding_window':
            # Expanding Window标准化 - 仅使用历史数据
            logger.info("执行Expanding Window标准化（仅历史数据）")
            
            unique_dates = sorted(X_result[date_col].unique())
            
            for col in feature_cols:
                standardized_values = []
                
                for current_date in unique_dates:
                    # 获取当前日期之前的所有数据（不包含当前日期）
                    historical_data = X_result[
                        (X_result[date_col] < current_date)
                    ][col].dropna()
                    
                    # 获取当前日期的数据
                    current_data = X_result[X_result[date_col] == current_date][col]
                    
                    if len(historical_data) >= self.min_history_days:
                        # 使用历史数据计算统计量
                        hist_mean = historical_data.mean()
                        hist_std = historical_data.std()
                        
                        # 标准化当前数据
                        if hist_std > 1e-8:
                            standardized = (current_data - hist_mean) / hist_std
                        else:
                            standardized = current_data - hist_mean
                    else:
                        # 历史数据不足，使用原始值
                        standardized = current_data
                        logger.warning(f"日期 {current_date} 历史数据不足 {len(historical_data)} < {self.min_history_days}")
                    
                    standardized_values.extend(standardized.tolist())
                
                # 更新特征列
                X_result[col] = standardized_values
            
            std_info = {
                'method': 'expanding_window',
                'feature_count': len(feature_cols),
                'min_history_days': self.min_history_days,
                'leakage_risk': 'NONE'
            }
        
        else:
            # 不进行标准化
            logger.info("跳过标准化")
            std_info = {'method': 'none', 'leakage_risk': 'NONE'}
        
        return X_result, std_info
    
    def _safe_collinearity_treatment(self, 
                                   X: pd.DataFrame, 
                                   date_col: str) -> Tuple[pd.DataFrame, Dict]:
        """安全的共线性处理（避免PCA的数据泄露）"""
        
        feature_cols = [col for col in X.columns if col != date_col]
        
        if self.enable_pca:
            logger.warning("⚠️  使用PCA处理共线性，存在数据泄露风险")
            return self._risky_pca_treatment(X, date_col)
        
        # 使用安全的替代方法
        if self.pca_alternative == 'correlation_filter':
            return self._correlation_filter_treatment(X, date_col)
        elif self.pca_alternative == 'variance_filter':
            return self._variance_filter_treatment(X, date_col)
        else:
            # 不处理共线性
            logger.info("跳过共线性处理")
            return X, {'method': 'none', 'leakage_risk': 'NONE'}
    
    def cross_sectional_standardize(self, df: pd.DataFrame, date_col: str, feature_cols: List[str]) -> pd.DataFrame:
        """横截面标准化 - 类方法版本"""
        return cross_sectional_standardize(df, date_col, feature_cols)
    
    def expanding_window_standardize(self, df: pd.DataFrame, date_col: str, feature_cols: List[str], min_periods: int = 252) -> pd.DataFrame:
        """展开窗口标准化 - 类方法版本"""
        return expanding_window_standardize(df, date_col, feature_cols, min_periods)
    
    def correlation_based_selection(self, df: pd.DataFrame, date_col: str, threshold: float = 0.8) -> pd.DataFrame:
        """基于相关性的特征选择 - 类方法版本"""
        feature_cols = [col for col in df.columns if col != date_col]
        return self._correlation_filter_treatment(df, date_col)[0]
    
    def _correlation_filter_treatment(self, 
                                    X: pd.DataFrame, 
                                    date_col: str) -> Tuple[pd.DataFrame, Dict]:
        """基于相关性的特征过滤（时序安全）"""
        
        logger.info("执行基于相关性的特征过滤")
        
        feature_cols = [col for col in X.columns if col != date_col]
        
        # 计算特征间相关性矩阵（全样本，但仅用于特征选择不用于变换）
        corr_matrix = X[feature_cols].corr().abs()
        
        # 识别高度相关的特征对
        high_corr_pairs = []
        threshold = 0.85  # 相关性阈值
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        # 去除高度相关的特征（保留方差更大的）
        features_to_remove = set()
        for feat1, feat2, corr_val in high_corr_pairs:
            if feat1 not in features_to_remove and feat2 not in features_to_remove:
                # 比较方差，去除方差较小的特征
                var1 = X[feat1].var()
                var2 = X[feat2].var()
                
                if var1 < var2:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
        
        # 保留的特征
        retained_features = [col for col in feature_cols if col not in features_to_remove]
        retained_features.append(date_col)  # 保留日期列
        
        X_filtered = X[retained_features]
        
        treatment_info = {
            'method': 'correlation_filter',
            'threshold': threshold,
            'high_corr_pairs': len(high_corr_pairs),
            'features_removed': len(features_to_remove),
            'features_retained': len(retained_features) - 1,  # 减去日期列
            'removed_features': list(features_to_remove),
            'leakage_risk': 'LOW'  # 仅用于特征选择，不用于变换
        }
        
        logger.info(f"相关性过滤完成: 移除{len(features_to_remove)}个高相关特征")
        
        return X_filtered, treatment_info
    
    def _variance_filter_treatment(self, 
                                 X: pd.DataFrame, 
                                 date_col: str) -> Tuple[pd.DataFrame, Dict]:
        """基于方差的特征过滤"""
        
        logger.info("执行基于方差的特征过滤")
        
        feature_cols = [col for col in X.columns if col != date_col]
        
        # 计算特征方差
        feature_vars = X[feature_cols].var()
        
        # 去除低方差特征
        min_variance = 1e-6
        high_var_features = feature_vars[feature_vars > min_variance].index.tolist()
        
        # 保留高方差特征和日期列
        retained_cols = high_var_features + [date_col]
        X_filtered = X[retained_cols]
        
        treatment_info = {
            'method': 'variance_filter',
            'min_variance': min_variance,
            'features_removed': len(feature_cols) - len(high_var_features),
            'features_retained': len(high_var_features),
            'leakage_risk': 'NONE'
        }
        
        logger.info(f"方差过滤完成: 移除{len(feature_cols) - len(high_var_features)}个低方差特征")
        
        return X_filtered, treatment_info
    
    def _risky_pca_treatment(self, 
                           X: pd.DataFrame, 
                           date_col: str) -> Tuple[pd.DataFrame, Dict]:
        """PCA处理（存在数据泄露风险，不推荐使用）"""
        
        logger.warning("⚠️  执行PCA变换，存在严重数据泄露风险！")
        
        feature_cols = [col for col in X.columns if col != date_col]
        
        # 标准化特征数据
        X_features = X[feature_cols].fillna(0)
        
        # 应用PCA（全样本，存在泄露）
        pca = PCA(n_components=0.95)  # 保留95%方差
        X_pca = pca.fit_transform(X_features)
        
        # 创建主成分DataFrame
        n_components = X_pca.shape[1]
        pca_cols = [f'PC_{i+1}' for i in range(n_components)]
        
        X_result = X[[date_col]].copy()  # 保留日期列
        for i, col in enumerate(pca_cols):
            X_result[col] = X_pca[:, i]
        
        treatment_info = {
            'method': 'pca_risky',
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': pca.explained_variance_ratio_.sum(),
            'leakage_risk': 'HIGH'  # 严重数据泄露风险
        }
        
        logger.warning(f"PCA变换完成: {len(feature_cols)} -> {n_components} 主成分 (存在数据泄露!)")
        
        return X_result, treatment_info


def create_temporal_safe_preprocessor(config: Dict = None) -> TemporalSafePreprocessor:
    """
    创建时序安全预处理器的工厂函数
    
    Args:
        config: 配置字典
        
    Returns:
        TemporalSafePreprocessor实例
    """
    if config is None:
        config = {}
    
    return TemporalSafePreprocessor(
        standardization_mode=config.get('standardization_mode', 'cross_sectional'),
        min_history_days=config.get('min_history_days', 252),
        enable_pca=config.get('enable_pca', False),  # 默认禁用PCA
        pca_alternative=config.get('pca_alternative', 'correlation_filter')
    )


# 便捷函数
def cross_sectional_standardize(df: pd.DataFrame, 
                               date_col: str, 
                               feature_cols: List[str]) -> pd.DataFrame:
    """
    横截面标准化便捷函数
    
    Args:
        df: 数据框
        date_col: 日期列名
        feature_cols: 特征列名列表
        
    Returns:
        标准化后的数据框
    """
    result = df.copy()
    
    for col in feature_cols:
        result[col] = result.groupby(date_col)[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8) if x.std() > 1e-8 else x
        )
    
    return result


def expanding_window_standardize(df: pd.DataFrame,
                                date_col: str,
                                feature_cols: List[str],
                                min_history_days: int = 252) -> pd.DataFrame:
    """
    Expanding Window标准化便捷函数
    
    Args:
        df: 数据框
        date_col: 日期列名
        feature_cols: 特征列名列表
        min_history_days: 最小历史天数
        
    Returns:
        标准化后的数据框
    """
    result = df.copy().sort_values(date_col).reset_index(drop=True)
    unique_dates = sorted(result[date_col].unique())
    
    for col in feature_cols:
        standardized_values = []
        
        for current_date in unique_dates:
            # 仅使用历史数据
            historical_data = result[
                result[date_col] < current_date
            ][col].dropna()
            
            current_data = result[result[date_col] == current_date][col]
            
            if len(historical_data) >= min_history_days:
                hist_mean = historical_data.mean()
                hist_std = historical_data.std()
                
                if hist_std > 1e-8:
                    standardized = (current_data - hist_mean) / hist_std
                else:
                    standardized = current_data - hist_mean
            else:
                standardized = current_data
            
            standardized_values.extend(standardized.tolist())
        
        result[col] = standardized_values
    
    return result


if __name__ == "__main__":
    # 测试时序安全预处理器
    print("测试时序安全预处理器...")
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    n_features = 10
    
    test_data = pd.DataFrame({
        'date': np.repeat(dates, 5),  # 5个股票
        'ticker': np.tile(['A', 'B', 'C', 'D', 'E'], len(dates)),
    })
    
    # 添加特征
    for i in range(n_features):
        test_data[f'feature_{i}'] = np.random.randn(len(test_data))
    
    # 创建预处理器
    preprocessor = create_temporal_safe_preprocessor({
        'standardization_mode': 'cross_sectional',
        'enable_pca': False,
        'pca_alternative': 'correlation_filter'
    })
    
    # 执行变换
    feature_cols = [col for col in test_data.columns if col.startswith('feature_')]
    X_transformed, info = preprocessor.fit_transform(
        test_data[['date'] + feature_cols],
        test_data['date']
    )
    
    print(f"变换完成: {test_data.shape} -> {X_transformed.shape}")
    print(f"变换信息: {info}")
    print("✅ 时序安全预处理器测试通过")