#!/usr/bin/env python3
"""
增强的数据处理和缺失值处理系统
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_samples: int
    total_features: int
    missing_ratio: float
    outlier_ratio: float
    duplicate_ratio: float
    feature_quality: Dict[str, Dict[str, Any]]
    processing_actions: List[str]
    data_issues: List[str]

class SmartImputer:
    """智能缺失值处理器"""
    
    def __init__(self, strategy: str = 'adaptive'):
        """
        初始化缺失值处理器
        
        Args:
            strategy: 'adaptive', 'knn', 'iterative', 'simple'
        """
        self.strategy = strategy
        self.imputers = {}
        self.feature_strategies = {}
        
    def fit(self, X: pd.DataFrame, feature_types: Optional[Dict[str, str]] = None) -> 'SmartImputer':
        """
        训练缺失值处理器
        
        Args:
            X: 训练数据
            feature_types: 特征类型映射 {'feature_name': 'price'/'volume'/'technical'/'fundamental'}
        """
        if feature_types is None:
            feature_types = self._infer_feature_types(X)
        
        self.feature_types = feature_types
        
        for feature_type in set(feature_types.values()):
            features = [col for col, ftype in feature_types.items() if ftype == feature_type]
            
            if not features:
                continue
                
            X_subset = X[features]
            
            # 根据特征类型选择最佳策略
            if feature_type == 'price':
                # 价格特征：前向填充 + 线性插值
                self.feature_strategies[feature_type] = 'forward_fill'
                
            elif feature_type == 'volume':
                # 成交量特征：零填充或中位数
                self.feature_strategies[feature_type] = 'median_fill'
                
            elif feature_type == 'technical':
                # 技术指标：KNN插值
                imputer = KNNImputer(n_neighbors=5, weights='distance')
                self.imputers[feature_type] = imputer.fit(X_subset)
                self.feature_strategies[feature_type] = 'knn'
                
            elif feature_type == 'fundamental':
                # 基本面数据：迭代插值
                imputer = IterativeImputer(random_state=42, max_iter=10)
                self.imputers[feature_type] = imputer.fit(X_subset)
                self.feature_strategies[feature_type] = 'iterative'
                
            else:
                # 默认：KNN
                imputer = KNNImputer(n_neighbors=5)
                self.imputers[feature_type] = imputer.fit(X_subset)
                self.feature_strategies[feature_type] = 'knn'
        
        logger.info(f"缺失值处理器训练完成，策略: {self.feature_strategies}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用缺失值处理"""
        X_imputed = X.copy()
        
        for feature_type, strategy in self.feature_strategies.items():
            features = [col for col, ftype in self.feature_types.items() if ftype == feature_type]
            
            if not features or not any(col in X_imputed.columns for col in features):
                continue
            
            available_features = [col for col in features if col in X_imputed.columns]
            
            if strategy == 'forward_fill':
                # 前向填充
                X_imputed[available_features] = X_imputed[available_features].fillna(method='ffill')
                # 回填第一个值
                X_imputed[available_features] = X_imputed[available_features].fillna(method='bfill')
                
            elif strategy == 'median_fill':
                # 中位数填充
                for col in available_features:
                    median_val = X_imputed[col].median()
                    X_imputed[col] = X_imputed[col].fillna(median_val)
                    
            elif strategy == 'knn':
                # KNN插值
                if feature_type in self.imputers:
                    X_subset = X_imputed[available_features]
                    X_imputed[available_features] = self.imputers[feature_type].transform(X_subset)
                    
            elif strategy == 'iterative':
                # 迭代插值
                if feature_type in self.imputers:
                    X_subset = X_imputed[available_features]
                    X_imputed[available_features] = self.imputers[feature_type].transform(X_subset)
        
        # 标记原本缺失的位置
        for col in X.columns:
            if X[col].isna().any():
                X_imputed[f'{col}_was_missing'] = X[col].isna().astype(int)
        
        return X_imputed
    
    def _infer_feature_types(self, X: pd.DataFrame) -> Dict[str, str]:
        """推断特征类型"""
        feature_types = {}
        
        for col in X.columns:
            col_lower = col.lower()
            
            if any(keyword in col_lower for keyword in ['price', 'open', 'high', 'low', 'close', 'adj']):
                feature_types[col] = 'price'
            elif any(keyword in col_lower for keyword in ['volume', 'amount', 'turnover']):
                feature_types[col] = 'volume'
            elif any(keyword in col_lower for keyword in ['ma_', 'ema_', 'rsi', 'macd', 'bb_', 'momentum']):
                feature_types[col] = 'technical'
            elif any(keyword in col_lower for keyword in ['pe', 'pb', 'roe', 'roa', 'eps', 'revenue']):
                feature_types[col] = 'fundamental'
            else:
                feature_types[col] = 'other'
        
        return feature_types

class AnomalyDetector:
    """多策略异常值检测器"""
    
    def __init__(self, contamination: float = 0.01):
        """
        初始化异常值检测器
        
        Args:
            contamination: 预期异常值比例
        """
        self.contamination = contamination
        self.detectors = {
            'isolation_forest': IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            ),
            'local_outlier_factor': LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=20
            )
        }
        
    def fit(self, X: pd.DataFrame) -> 'AnomalyDetector':
        """训练异常检测器"""
        X_numeric = X.select_dtypes(include=[np.number])
        
        if len(X_numeric.columns) == 0:
            logger.warning("没有数值型特征用于异常检测")
            return self
        
        # 标准化数据
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_numeric.fillna(X_numeric.median()))
        
        # 训练检测器
        try:
            self.detectors['isolation_forest'].fit(X_scaled)
            logger.info("Isolation Forest异常检测器训练完成")
        except Exception as e:
            logger.warning(f"Isolation Forest训练失败: {e}")
        
        return self
    
    def detect_outliers(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """检测异常值"""
        X_numeric = X.select_dtypes(include=[np.number])
        
        if len(X_numeric.columns) == 0:
            return {'combined': np.zeros(len(X), dtype=bool)}
        
        # 标准化数据
        X_scaled = self.scaler.transform(X_numeric.fillna(X_numeric.median()))
        
        outlier_masks = {}
        
        # Isolation Forest
        try:
            outlier_masks['isolation_forest'] = self.detectors['isolation_forest'].predict(X_scaled) == -1
        except Exception as e:
            logger.warning(f"Isolation Forest检测失败: {e}")
            outlier_masks['isolation_forest'] = np.zeros(len(X), dtype=bool)
        
        # Local Outlier Factor
        try:
            outlier_masks['local_outlier_factor'] = self.detectors['local_outlier_factor'].fit_predict(X_scaled) == -1
        except Exception as e:
            logger.warning(f"LOF检测失败: {e}")
            outlier_masks['local_outlier_factor'] = np.zeros(len(X), dtype=bool)
        
        # 统计学方法：IQR
        outlier_masks['statistical'] = self._statistical_outliers(X_numeric)
        
        # 组合结果：至少两种方法认为是异常值
        combined_mask = np.zeros(len(X), dtype=bool)
        for i in range(len(X)):
            outlier_count = sum(mask[i] for mask in outlier_masks.values())
            combined_mask[i] = outlier_count >= 2
        
        outlier_masks['combined'] = combined_mask
        
        return outlier_masks
    
    def _statistical_outliers(self, X: pd.DataFrame) -> np.ndarray:
        """基于IQR的统计学异常检测"""
        outlier_mask = np.zeros(len(X), dtype=bool)
        
        for col in X.columns:
            if X[col].dtype in [np.number]:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = (X[col] < lower_bound) | (X[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers.fillna(False)
        
        return outlier_mask

class EnhancedDataProcessor:
    """增强的数据处理器"""
    
    def __init__(self, handle_outliers: bool = True, 
                 handle_missing: bool = True,
                 optimize_dtypes: bool = True):
        """
        初始化数据处理器
        
        Args:
            handle_outliers: 是否处理异常值
            handle_missing: 是否处理缺失值
            optimize_dtypes: 是否优化数据类型
        """
        self.handle_outliers = handle_outliers
        self.handle_missing = handle_missing
        self.optimize_dtypes = optimize_dtypes
        
        self.imputer = SmartImputer()
        self.anomaly_detector = AnomalyDetector()
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, feature_types: Optional[Dict[str, str]] = None) -> 'EnhancedDataProcessor':
        """训练数据处理器"""
        logger.info("开始训练数据处理器...")
        
        if self.handle_missing:
            self.imputer.fit(X, feature_types)
        
        if self.handle_outliers:
            self.anomaly_detector.fit(X)
        
        self.fitted = True
        logger.info("数据处理器训练完成")
        
        return self
    
    def transform(self, X: pd.DataFrame, remove_outliers: bool = False) -> Tuple[pd.DataFrame, DataQualityReport]:
        """处理数据并生成质量报告"""
        if not self.fitted:
            raise ValueError("数据处理器尚未训练，请先调用fit()方法")
        
        logger.info("开始数据处理...")
        X_processed = X.copy()
        processing_actions = []
        data_issues = []
        
        # 1. 数据类型优化
        if self.optimize_dtypes:
            X_processed, dtype_actions = self._optimize_dtypes(X_processed)
            processing_actions.extend(dtype_actions)
        
        # 2. 重复值处理
        initial_len = len(X_processed)
        X_processed = X_processed.drop_duplicates()
        duplicate_count = initial_len - len(X_processed)
        
        if duplicate_count > 0:
            processing_actions.append(f"移除{duplicate_count}个重复行")
            data_issues.append(f"发现{duplicate_count}个重复行")
        
        # 3. 异常值检测
        outlier_masks = {}
        if self.handle_outliers:
            outlier_masks = self.anomaly_detector.detect_outliers(X_processed)
            outlier_count = outlier_masks['combined'].sum()
            
            if outlier_count > 0:
                data_issues.append(f"检测到{outlier_count}个异常值")
                
                if remove_outliers:
                    X_processed = X_processed[~outlier_masks['combined']]
                    processing_actions.append(f"移除{outlier_count}个异常值")
                else:
                    # 标记异常值
                    X_processed['is_outlier'] = outlier_masks['combined']
                    processing_actions.append(f"标记{outlier_count}个异常值")
        
        # 4. 缺失值处理
        missing_before = X_processed.isna().sum().sum()
        if self.handle_missing and missing_before > 0:
            X_processed = self.imputer.transform(X_processed)
            missing_after = X_processed.isna().sum().sum()
            
            processing_actions.append(f"处理缺失值: {missing_before} -> {missing_after}")
            
            if missing_before > 0:
                data_issues.append(f"原始数据包含{missing_before}个缺失值")
        
        # 5. 生成质量报告
        quality_report = self._generate_quality_report(
            X_original=X,
            X_processed=X_processed,
            outlier_masks=outlier_masks,
            processing_actions=processing_actions,
            data_issues=data_issues,
            duplicate_count=duplicate_count
        )
        
        logger.info(f"数据处理完成，质量得分: {quality_report.missing_ratio:.2%} 缺失, {quality_report.outlier_ratio:.2%} 异常")
        
        return X_processed, quality_report
    
    def _optimize_dtypes(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """优化数据类型以节省内存"""
        actions = []
        X_optimized = X.copy()
        
        for col in X_optimized.columns:
            col_type = X_optimized[col].dtype
            
            if str(col_type)[:3] == 'int':
                c_min = X_optimized[col].min()
                c_max = X_optimized[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    X_optimized[col] = X_optimized[col].astype(np.int8)
                    actions.append(f"{col}: int64 -> int8")
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    X_optimized[col] = X_optimized[col].astype(np.int16)
                    actions.append(f"{col}: int64 -> int16")
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    X_optimized[col] = X_optimized[col].astype(np.int32)
                    actions.append(f"{col}: int64 -> int32")
                    
            elif str(col_type)[:5] == 'float':
                c_min = X_optimized[col].min()
                c_max = X_optimized[col].max()
                
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    X_optimized[col] = X_optimized[col].astype(np.float32)
                    actions.append(f"{col}: float64 -> float32")
        
        return X_optimized, actions
    
    def _generate_quality_report(self, X_original: pd.DataFrame, X_processed: pd.DataFrame,
                                outlier_masks: Dict, processing_actions: List[str],
                                data_issues: List[str], duplicate_count: int) -> DataQualityReport:
        """生成数据质量报告"""
        
        # 基本统计
        total_samples = len(X_original)
        total_features = len(X_original.columns)
        missing_ratio = X_original.isna().sum().sum() / (total_samples * total_features)
        outlier_ratio = outlier_masks.get('combined', np.array([])).sum() / total_samples if outlier_masks else 0.0
        duplicate_ratio = duplicate_count / total_samples
        
        # 特征质量分析
        feature_quality = {}
        for col in X_original.columns:
            feature_quality[col] = {
                'missing_ratio': X_original[col].isna().mean(),
                'unique_ratio': X_original[col].nunique() / len(X_original),
                'dtype': str(X_original[col].dtype),
                'zero_ratio': (X_original[col] == 0).mean() if X_original[col].dtype in [np.number] else 0,
                'infinite_values': np.isinf(X_original[col]).sum() if X_original[col].dtype in [np.number] else 0
            }
        
        return DataQualityReport(
            total_samples=total_samples,
            total_features=total_features,
            missing_ratio=missing_ratio,
            outlier_ratio=outlier_ratio,
            duplicate_ratio=duplicate_ratio,
            feature_quality=feature_quality,
            processing_actions=processing_actions,
            data_issues=data_issues
        )


def example_usage():
    """示例用法"""
    print("🔧 增强数据处理系统示例")
    
    # 生成带问题的测试数据
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'close_price': np.random.normal(100, 10, n_samples),
        'volume': np.random.exponential(1000000, n_samples),
        'ma_20': np.random.normal(100, 8, n_samples),
        'rsi': np.random.uniform(0, 100, n_samples),
        'pe_ratio': np.random.normal(15, 5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 引入数据问题
    # 1. 缺失值
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    df.loc[missing_indices, 'close_price'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices, 'pe_ratio'] = np.nan
    
    # 2. 异常值
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    df.loc[outlier_indices, 'close_price'] = np.random.normal(200, 50, len(outlier_indices))
    
    # 3. 重复值
    duplicate_indices = np.random.choice(n_samples, size=20, replace=False)
    df = pd.concat([df, df.iloc[duplicate_indices]], ignore_index=True)
    
    print(f"📊 原始数据: {len(df)}行, {len(df.columns)}列")
    print(f"   缺失值: {df.isna().sum().sum()}")
    print(f"   重复行: {df.duplicated().sum()}")
    
    # 创建处理器
    processor = EnhancedDataProcessor(
        handle_outliers=True,
        handle_missing=True,
        optimize_dtypes=True
    )
    
    # 训练和处理
    processor.fit(df)
    df_processed, quality_report = processor.transform(df, remove_outliers=False)
    
    print(f"\n✅ 处理后数据: {len(df_processed)}行, {len(df_processed.columns)}列")
    print(f"   缺失值: {df_processed.isna().sum().sum()}")
    print(f"   异常值比例: {quality_report.outlier_ratio:.2%}")
    print(f"   数据质量: {(1 - quality_report.missing_ratio) * 100:.1f}%")
    
    print(f"\n🔧 处理动作:")
    for action in quality_report.processing_actions:
        print(f"   - {action}")
    
    print(f"\n⚠️  数据问题:")
    for issue in quality_report.data_issues:
        print(f"   - {issue}")
    
    return processor, quality_report


if __name__ == "__main__":
    example_usage()
