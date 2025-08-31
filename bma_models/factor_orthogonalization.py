"""
因子正交化系统
===============
剔除因子间共线性，提高模型稳定性
基于Gram-Schmidt和主成分正交化方法
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy import linalg
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class FactorOrthogonalizer:
    """
    因子正交化处理器
    剔除因子间的共线性，保持因子独立性
    """
    
    def __init__(self, method: str = 'sequential', threshold: float = 0.85):
        """
        Args:
            method: 正交化方法 ('sequential', 'symmetric', 'pca')
            threshold: 相关性阈值，超过此值的因子将被正交化
        """
        self.method = method
        self.threshold = threshold
        self.orthogonalization_matrix = None
        self.factor_order = None
        self.correlation_before = None
        self.correlation_after = None
        
    def fit_transform(self, factors: pd.DataFrame, 
                     base_factors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        拟合并转换因子为正交化版本
        
        Args:
            factors: 原始因子数据
            base_factors: 基础因子列表（这些因子不被修改，其他因子相对于它们正交化）
        
        Returns:
            正交化后的因子
        """
        logger.info(f"开始因子正交化，方法: {self.method}")
        
        # 记录原始相关性
        self.correlation_before = factors.corr()
        
        # 检测高相关性因子对
        high_corr_pairs = self._detect_high_correlation(factors)
        logger.info(f"发现 {len(high_corr_pairs)} 对高相关因子（相关性>{self.threshold}）")
        
        # 根据方法执行正交化
        if self.method == 'sequential':
            orthogonal_factors = self._sequential_orthogonalization(factors, base_factors)
        elif self.method == 'symmetric':
            orthogonal_factors = self._symmetric_orthogonalization(factors)
        elif self.method == 'pca':
            orthogonal_factors = self._pca_orthogonalization(factors)
        else:
            raise ValueError(f"未知的正交化方法: {self.method}")
        
        # 记录正交化后的相关性
        self.correlation_after = orthogonal_factors.corr()
        
        # 验证正交化效果
        self._validate_orthogonalization(orthogonal_factors)
        
        return orthogonal_factors
    
    def _detect_high_correlation(self, factors: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """检测高相关性的因子对"""
        corr_matrix = factors.corr().abs()
        
        # 获取上三角矩阵（避免重复）
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 找到高相关性的因子对
        high_corr_pairs = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                corr_value = upper_tri.loc[idx, col]
                if pd.notna(corr_value) and corr_value > self.threshold:
                    high_corr_pairs.append((idx, col, corr_value))
        
        # 按相关性排序
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # 打印top相关性
        if high_corr_pairs:
            logger.info("Top 5 高相关因子对:")
            for factor1, factor2, corr in high_corr_pairs[:5]:
                logger.info(f"  {factor1} <-> {factor2}: {corr:.3f}")
        
        return high_corr_pairs
    
    def _sequential_orthogonalization(self, factors: pd.DataFrame, 
                                     base_factors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        顺序正交化（Modified Gram-Schmidt）
        按重要性顺序保持因子，后续因子相对于前面因子正交化
        """
        orthogonal = factors.copy()
        
        # 确定因子顺序（基础因子优先）
        if base_factors:
            # 基础因子不变，其他因子相对于它们正交化
            other_factors = [col for col in factors.columns if col not in base_factors]
            factor_order = base_factors + other_factors
        else:
            # 按因子方差（重要性）排序
            factor_importance = factors.var().sort_values(ascending=False)
            factor_order = factor_importance.index.tolist()
        
        self.factor_order = factor_order
        
        # Modified Gram-Schmidt过程
        for i, factor_i in enumerate(factor_order):
            if i == 0 or (base_factors and factor_i in base_factors):
                # 第一个因子或基础因子保持不变（只标准化）
                orthogonal[factor_i] = self._standardize(orthogonal[factor_i])
            else:
                # 从当前因子中移除前面所有因子的投影
                residual = orthogonal[factor_i].copy()
                
                for j in range(i):
                    factor_j = factor_order[j]
                    # 计算投影系数
                    projection_coef = np.dot(residual, orthogonal[factor_j]) / np.dot(orthogonal[factor_j], orthogonal[factor_j])
                    # 移除投影
                    residual = residual - projection_coef * orthogonal[factor_j]
                
                # 标准化
                orthogonal[factor_i] = self._standardize(residual)
        
        return orthogonal
    
    def _symmetric_orthogonalization(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        对称正交化
        所有因子同等对待，通过特征值分解实现
        """
        # 计算相关矩阵
        corr_matrix = factors.corr()
        
        # 特征值分解
        eigenvalues, eigenvectors = linalg.eigh(corr_matrix)
        
        # 构造正交化矩阵 (C^(-1/2))
        # 处理小特征值（避免数值不稳定）
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        
        # 计算 C^(-1/2) = V * D^(-1/2) * V^T
        D_sqrt_inv = np.diag(1.0 / np.sqrt(eigenvalues))
        orthogonalization_matrix = eigenvectors @ D_sqrt_inv @ eigenvectors.T
        
        # 应用正交化
        factors_standardized = factors.apply(self._standardize)
        orthogonal_values = factors_standardized.values @ orthogonalization_matrix
        
        # 创建DataFrame
        orthogonal = pd.DataFrame(
            orthogonal_values,
            index=factors.index,
            columns=factors.columns
        )
        
        self.orthogonalization_matrix = orthogonalization_matrix
        
        return orthogonal
    
    def _pca_orthogonalization(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        PCA正交化
        通过主成分分析得到正交因子
        """
        # 标准化
        factors_standardized = factors.apply(self._standardize)
        
        # PCA
        pca = PCA(n_components=len(factors.columns))
        principal_components = pca.fit_transform(factors_standardized)
        
        # 创建新的因子名称
        pc_names = [f'PC{i+1}' for i in range(len(factors.columns))]
        
        # 创建DataFrame
        orthogonal = pd.DataFrame(
            principal_components,
            index=factors.index,
            columns=pc_names
        )
        
        # 保存PCA对象用于解释
        self.pca_model = pca
        
        # 打印方差解释率
        logger.info("PCA方差解释率:")
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        for i, (var, cum_var) in enumerate(zip(pca.explained_variance_ratio_, cumsum_var)):
            if i < 5 or var > 0.01:  # 只显示前5个或重要的成分
                logger.info(f"  PC{i+1}: {var*100:.2f}% (累计: {cum_var*100:.2f}%)")
        
        return orthogonal
    
    def _standardize(self, series: pd.Series) -> pd.Series:
        """标准化（z-score）"""
        mean = series.mean()
        std = series.std()
        if std > 0:
            return (series - mean) / std
        else:
            return series - mean
    
    def _validate_orthogonalization(self, orthogonal_factors: pd.DataFrame):
        """验证正交化效果"""
        corr_after = orthogonal_factors.corr().abs()
        
        # 计算非对角线元素的平均相关性
        n = len(corr_after)
        off_diagonal_mask = ~np.eye(n, dtype=bool)
        
        avg_corr_before = self.correlation_before.abs().values[off_diagonal_mask].mean()
        avg_corr_after = corr_after.values[off_diagonal_mask].mean()
        max_corr_after = corr_after.values[off_diagonal_mask].max()
        
        logger.info("正交化效果:")
        logger.info(f"  平均相关性: {avg_corr_before:.3f} -> {avg_corr_after:.3f} "
                   f"(降低 {(1-avg_corr_after/avg_corr_before)*100:.1f}%)")
        logger.info(f"  最大相关性: {max_corr_after:.3f}")
        
        # 检查是否还有高相关性
        if max_corr_after > self.threshold:
            logger.warning(f"警告: 正交化后仍存在高相关性 (>{self.threshold})")
        else:
            logger.info(f"✓ 所有因子相关性已降至 {self.threshold} 以下")
    
    def get_orthogonalization_report(self) -> pd.DataFrame:
        """生成正交化报告"""
        if self.correlation_before is None or self.correlation_after is None:
            return pd.DataFrame()
        
        report_data = []
        
        for factor in self.correlation_before.columns:
            # 计算每个因子的平均相关性变化
            avg_corr_before = self.correlation_before[factor].abs().mean()
            avg_corr_after = self.correlation_after[factor].abs().mean() if factor in self.correlation_after.columns else 0
            
            report_data.append({
                'factor': factor,
                'avg_corr_before': avg_corr_before,
                'avg_corr_after': avg_corr_after,
                'corr_reduction': (avg_corr_before - avg_corr_after) / avg_corr_before * 100 if avg_corr_before > 0 else 0
            })
        
        report = pd.DataFrame(report_data)
        report = report.sort_values('corr_reduction', ascending=False)
        
        return report
    
    def transform(self, new_factors: pd.DataFrame) -> pd.DataFrame:
        """
        使用已拟合的正交化转换新数据
        """
        if self.method == 'symmetric' and self.orthogonalization_matrix is not None:
            # 使用保存的正交化矩阵
            factors_standardized = new_factors.apply(self._standardize)
            orthogonal_values = factors_standardized.values @ self.orthogonalization_matrix
            
            return pd.DataFrame(
                orthogonal_values,
                index=new_factors.index,
                columns=new_factors.columns
            )
        elif self.method == 'sequential' and self.factor_order is not None:
            # 重新执行顺序正交化
            return self._sequential_orthogonalization(new_factors, base_factors=None)
        elif self.method == 'pca' and hasattr(self, 'pca_model'):
            # 使用PCA模型转换
            factors_standardized = new_factors.apply(self._standardize)
            principal_components = self.pca_model.transform(factors_standardized)
            
            pc_names = [f'PC{i+1}' for i in range(principal_components.shape[1])]
            return pd.DataFrame(
                principal_components,
                index=new_factors.index,
                columns=pc_names
            )
        else:
            raise ValueError("模型未拟合，请先调用fit_transform")


def adaptive_orthogonalization(factors: pd.DataFrame, 
                              market_regime: str = 'normal') -> pd.DataFrame:
    """
    自适应正交化
    根据市场状态选择不同的正交化策略
    """
    if market_regime == 'high_vol':
        # 高波动市场：使用PCA降维
        logger.info("高波动市场：使用PCA正交化")
        orthogonalizer = FactorOrthogonalizer(method='pca', threshold=0.7)
    elif market_regime == 'crisis':
        # 危机时期：使用对称正交化（所有因子同等重要）
        logger.info("危机市场：使用对称正交化")
        orthogonalizer = FactorOrthogonalizer(method='symmetric', threshold=0.6)
    else:
        # 正常市场：使用顺序正交化
        logger.info("正常市场：使用顺序正交化")
        orthogonalizer = FactorOrthogonalizer(method='sequential', threshold=0.85)
    
    return orthogonalizer.fit_transform(factors)


def orthogonalize_factors_predictive_safe(factors: pd.DataFrame, 
                                        method: str = 'sequential',
                                        threshold: float = 0.85) -> pd.DataFrame:
    """
    预测安全的因子正交化函数
    确保在预测阶段不引入未来信息偏差
    
    Args:
        factors: 因子数据DataFrame
        method: 正交化方法
        threshold: 相关性阈值
    
    Returns:
        正交化后的因子DataFrame
    """
    if factors.empty or len(factors.columns) <= 1:
        return factors
    
    try:
        orthogonalizer = FactorOrthogonalizer(method=method, threshold=threshold)
        return orthogonalizer.fit_transform(factors)
    except Exception as e:
        logger.warning(f"因子正交化失败，返回原始数据: {e}")
        return factors