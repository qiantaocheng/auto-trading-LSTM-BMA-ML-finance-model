#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一特征管道 - 解决训练-预测特征维度不匹配问题
确保训练和预测使用完全相同的特征处理流程
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

@dataclass
class FeaturePipelineConfig:
    """特征管道配置"""
    enable_alpha_summary: bool = True
    enable_pca: bool = True
    pca_variance_threshold: float = 0.95
    enable_scaling: bool = True
    scaler_type: str = 'robust'  # 'standard' or 'robust'
    imputation_strategy: str = 'median'
    feature_selection_threshold: float = 0.001
    save_pipeline: bool = True
    cache_dir: str = "cache/feature_pipeline"

class UnifiedFeaturePipeline:
    """
    统一特征管道 - 确保训练和预测时特征完全一致
    
    核心功能:
    1. 统一特征生成流程
    2. 保存/加载特征转换器状态
    3. 维度一致性检查
    4. 可重现的特征处理
    """
    
    def __init__(self, config: FeaturePipelineConfig):
        self.config = config
        self.is_fitted = False
        self.feature_names = None
        self.feature_dim = None
        
        # 特征处理组件
        self.scaler = None
        self.pca = None
        self.imputer = None
        self.alpha_summary_generator = None
        
        # 缓存目录
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 特征转换历史
        self.transform_history = {}
        
    def fit_transform(self, 
                     base_features: pd.DataFrame, 
                     targets: Optional[pd.Series] = None,
                     dates: Optional[pd.Series] = None,
                     alpha_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        拟合特征管道并转换数据
        
        Args:
            base_features: 基础特征数据
            targets: 目标变量 (用于有监督特征选择)
            dates: 日期序列
            alpha_data: Alpha因子数据
            
        Returns:
            (转换后特征, 转换信息)
        """
        logger.info(f"开始拟合特征管道，基础特征维度: {base_features.shape}")
        
        # 1. 数据预处理和清洗
        X_clean = self._clean_features(base_features)
        logger.info(f"特征清洗完成: {base_features.shape} -> {X_clean.shape}")
        
        # 2. 生成Alpha摘要特征 (如果启用)
        if self.config.enable_alpha_summary and alpha_data is not None:
            alpha_summary = self._generate_alpha_summary_features(alpha_data)
            logger.info(f"Alpha摘要特征生成: {alpha_summary.shape}")
            
            # 融合特征 - 确保索引对齐
            X_fused = self._fuse_features(X_clean, alpha_summary)
            logger.info(f"特征融合完成: {X_fused.shape}")
        else:
            X_fused = X_clean
            logger.info("跳过Alpha摘要特征，使用基础特征")
        
        # 3. 拟合数据插补器
        self.imputer = SimpleImputer(strategy=self.config.imputation_strategy)
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_fused),
            index=X_fused.index,
            columns=X_fused.columns
        )
        logger.info(f"数据插补完成，NaN处理策略: {self.config.imputation_strategy}")
        
        # 4. 拟合标准化器
        if self.config.enable_scaling:
            if self.config.scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
                
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_imputed),
                index=X_imputed.index,
                columns=X_imputed.columns
            )
            logger.info(f"特征标准化完成，方法: {self.config.scaler_type}")
        else:
            X_scaled = X_imputed
        
        # 5. 拟合PCA降维 (如果启用)
        if self.config.enable_pca and X_scaled.shape[1] > 10:
            self.pca = PCA(n_components=self.config.pca_variance_threshold)
            X_pca = self.pca.fit_transform(X_scaled)
            
            # 创建PCA特征名称
            pca_columns = [f'PC_{i+1}' for i in range(X_pca.shape[1])]
            X_final = pd.DataFrame(X_pca, index=X_scaled.index, columns=pca_columns)
            
            logger.info(f"PCA降维完成: {X_scaled.shape[1]} -> {X_final.shape[1]} "
                       f"(解释方差: {self.pca.explained_variance_ratio_.sum():.3f})")
        else:
            X_final = X_scaled
            logger.info("跳过PCA降维")
        
        # 6. 保存管道状态
        self.feature_names = list(X_final.columns)
        self.feature_dim = X_final.shape[1]
        self.is_fitted = True
        
        # 7. 生成转换信息
        transform_info = {
            'input_shape': base_features.shape,
            'output_shape': X_final.shape,
            'feature_names': self.feature_names,
            'alpha_summary_enabled': self.config.enable_alpha_summary and alpha_data is not None,
            'pca_enabled': self.config.enable_pca and hasattr(self, 'pca') and self.pca is not None,
            'pca_components': getattr(self.pca, 'n_components_', None),
            'explained_variance_ratio': getattr(self.pca, 'explained_variance_ratio_', None),
            'scaling_enabled': self.config.enable_scaling,
            'scaler_type': self.config.scaler_type if self.config.enable_scaling else None
        }
        
        # 8. 保存管道 (如果启用)
        if self.config.save_pipeline:
            self.save_pipeline()
        
        logger.info(f"特征管道拟合完成，最终特征维度: {X_final.shape}")
        return X_final, transform_info
    
    def transform(self, 
                 base_features: pd.DataFrame,
                 alpha_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        使用已拟合的管道转换新数据
        
        Args:
            base_features: 基础特征数据
            alpha_data: Alpha因子数据
            
        Returns:
            转换后的特征数据
        """
        if not self.is_fitted:
            raise ValueError("特征管道尚未拟合，请先调用fit_transform()")
        
        logger.info(f"使用已拟合管道转换特征，输入维度: {base_features.shape}")
        
        # 1. 清洗特征
        X_clean = self._clean_features(base_features)
        
        # 2. 生成Alpha摘要特征 (如果在训练时启用了)
        if self.config.enable_alpha_summary and alpha_data is not None:
            alpha_summary = self._generate_alpha_summary_features(alpha_data)
            X_fused = self._fuse_features(X_clean, alpha_summary)
        else:
            X_fused = X_clean
        
        # 3. 应用数据插补
        if self.imputer is not None:
            X_imputed = pd.DataFrame(
                self.imputer.transform(X_fused),
                index=X_fused.index,
                columns=X_fused.columns
            )
        else:
            X_imputed = X_fused
        
        # 4. 应用标准化
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_imputed),
                index=X_imputed.index,
                columns=X_imputed.columns
            )
        else:
            X_scaled = X_imputed
        
        # 5. 应用PCA
        if self.pca is not None:
            X_pca = self.pca.transform(X_scaled)
            pca_columns = [f'PC_{i+1}' for i in range(X_pca.shape[1])]
            X_final = pd.DataFrame(X_pca, index=X_scaled.index, columns=pca_columns)
        else:
            X_final = X_scaled
        
        # 6. 维度一致性检查
        if X_final.shape[1] != self.feature_dim:
            logger.error(f"特征维度不匹配: 期望{self.feature_dim}, 实际{X_final.shape[1]}")
            raise ValueError(f"特征维度不匹配: 期望{self.feature_dim}, 实际{X_final.shape[1]}")
        
        # 7. 确保列名一致
        X_final.columns = self.feature_names
        
        logger.info(f"特征转换完成，输出维度: {X_final.shape}")
        return X_final
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """清洗特征数据"""
        # 只保留数值特征
        numeric_features = features.select_dtypes(include=[np.number])
        
        # 移除全NaN列
        numeric_features = numeric_features.dropna(axis=1, how='all')
        
        # 移除常数列
        numeric_features = numeric_features.loc[:, numeric_features.std() > 1e-8]
        
        return numeric_features
    
    def _generate_alpha_summary_features(self, alpha_data: pd.DataFrame) -> pd.DataFrame:
        """生成Alpha摘要特征"""
        # 这里重用现有的Alpha摘要特征生成逻辑
        # 为了简化，先使用PCA压缩Alpha特征
        alpha_clean = self._clean_features(alpha_data)
        
        if alpha_clean.shape[1] > 8:
            # 使用PCA压缩Alpha特征
            alpha_pca = PCA(n_components=8)
            alpha_compressed = alpha_pca.fit_transform(alpha_clean)
            alpha_columns = [f'alpha_pc{i+1}' for i in range(8)]
            alpha_summary = pd.DataFrame(
                alpha_compressed, 
                index=alpha_clean.index, 
                columns=alpha_columns
            )
        else:
            alpha_summary = alpha_clean
        
        # 添加Alpha统计特征
        alpha_summary['alpha_mean'] = alpha_clean.mean(axis=1)
        alpha_summary['alpha_std'] = alpha_clean.std(axis=1)
        alpha_summary['alpha_max'] = alpha_clean.max(axis=1)
        alpha_summary['alpha_min'] = alpha_clean.min(axis=1)
        
        return alpha_summary
    
    def _fuse_features(self, base_features: pd.DataFrame, alpha_features: pd.DataFrame) -> pd.DataFrame:
        """融合基础特征和Alpha特征"""
        # 使用索引交集对齐
        common_index = base_features.index.intersection(alpha_features.index)
        if len(common_index) == 0:
            logger.warning("基础特征和Alpha特征无公共索引，使用长度对齐")
            min_len = min(len(base_features), len(alpha_features))
            fused = pd.concat([
                base_features.iloc[:min_len], 
                alpha_features.iloc[:min_len]
            ], axis=1)
        else:
            fused = pd.concat([
                base_features.loc[common_index], 
                alpha_features.loc[common_index]
            ], axis=1)
        
        return fused
    
    def save_pipeline(self, filepath: Optional[str] = None) -> str:
        """保存特征管道"""
        if filepath is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.cache_dir / f"feature_pipeline_{timestamp}.pkl"
        
        pipeline_state = {
            'config': self.config,
            'scaler': self.scaler,
            'pca': self.pca,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'feature_dim': self.feature_dim,
            'is_fitted': self.is_fitted,
            'transform_history': self.transform_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_state, f)
        
        logger.info(f"特征管道已保存: {filepath}")
        return str(filepath)
    
    def load_pipeline(self, filepath: str) -> None:
        """加载特征管道"""
        with open(filepath, 'rb') as f:
            pipeline_state = pickle.load(f)
        
        self.config = pipeline_state['config']
        self.scaler = pipeline_state['scaler']
        self.pca = pipeline_state['pca']
        self.imputer = pipeline_state['imputer']
        self.feature_names = pipeline_state['feature_names']
        self.feature_dim = pipeline_state['feature_dim']
        self.is_fitted = pipeline_state['is_fitted']
        self.transform_history = pipeline_state.get('transform_history', {})
        
        logger.info(f"特征管道已加载: {filepath}")
        logger.info(f"管道特征维度: {self.feature_dim}")
        
    @classmethod
    def create_production_pipeline(cls) -> 'UnifiedFeaturePipeline':
        """创建生产级特征管道"""
        config = FeaturePipelineConfig(
            enable_alpha_summary=True,
            enable_pca=True,
            pca_variance_threshold=0.95,
            enable_scaling=True,
            scaler_type='robust',
            imputation_strategy='median',
            save_pipeline=True
        )
        return cls(config)
    
    def get_feature_info(self) -> Dict[str, Any]:
        """获取特征管道信息"""
        return {
            'is_fitted': self.is_fitted,
            'feature_dim': self.feature_dim,
            'feature_names': self.feature_names[:10] if self.feature_names else None,  # 只显示前10个
            'total_features': len(self.feature_names) if self.feature_names else 0,
            'has_pca': self.pca is not None,
            'has_scaler': self.scaler is not None,
            'has_imputer': self.imputer is not None,
            'config': {
                'enable_alpha_summary': self.config.enable_alpha_summary,
                'enable_pca': self.config.enable_pca,
                'enable_scaling': self.config.enable_scaling,
                'scaler_type': self.config.scaler_type
            }
        }