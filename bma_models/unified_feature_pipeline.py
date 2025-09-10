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
# PCA COMPLETELY REMOVED - NO DIMENSIONALITY REDUCTION
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

@dataclass
class FeaturePipelineConfig:
    """特征管道配置 - 🚫 NO PCA - DIRECT FEATURE USAGE"""
    enable_alpha_summary: bool = False
    # 🚫 PCA COMPLETELY REMOVED - NO DIMENSIONALITY REDUCTION
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
        
        # 特征处理组件 - 🚫 NO PCA
        self.scaler = None
        # self.pca = None  # 🚫 REMOVED - NO PCA
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
        
        # 5. 🚫 NO PCA - DIRECT FEATURE USAGE
        # PCA COMPLETELY REMOVED - USE FEATURES DIRECTLY
        X_final = X_scaled
        logger.info(f"🚫 NO PCA - 直接使用特征: {X_scaled.shape[1]} 个特征")
        logger.info("✅ 特征维度保持完全一致，确保训练预测无差异")
        
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
            'pca_enabled': False,  # 🚫 ALWAYS FALSE - NO PCA
            'pca_components': 0,   # 🚫 NO PCA COMPONENTS
            'explained_variance_ratio': None,  # 🚫 NO PCA ANALYSIS
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
        
        # 5. 🚫 NO PCA - DIRECT FEATURE USAGE
        # PCA COMPLETELY REMOVED - USE FEATURES DIRECTLY
        X_final = X_scaled
        logger.info(f"🚫 NO PCA 转换 - 直接使用 {X_scaled.shape[1]} 个特征")
        
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
        """生成Alpha摘要特征 - 不做PCA预处理，留给统一PCA处理"""
        # 清洗Alpha数据但不做PCA压缩
        alpha_clean = self._clean_features(alpha_data)
        
        # 直接返回清洗后的Alpha特征，让它们与传统特征一起做统一PCA
        return alpha_clean
    
    def _fuse_features(self, base_features: pd.DataFrame, alpha_features: pd.DataFrame) -> pd.DataFrame:
        """融合基础特征和Alpha特征 - 修复索引对齐问题"""
        
        # 🔥 CRITICAL FIX: 确保两者都是MultiIndex格式
        logger.info(f"特征合并开始: base_features{base_features.shape}, alpha_features{alpha_features.shape}")
        logger.info(f"base索引类型: {type(base_features.index)}, alpha索引类型: {type(alpha_features.index)}")
        
        # 1. 标准化索引格式
        base_standardized = self._ensure_multiindex_format(base_features, "base_features")
        alpha_standardized = self._ensure_multiindex_format(alpha_features, "alpha_features") 
        
        if base_standardized is None or alpha_standardized is None:
            logger.error("❌ 特征标准化失败")
            return base_features  # 回退到基础特征
        
        # 2. 使用智能索引对齐
        try:
            common_index = base_standardized.index.intersection(alpha_standardized.index)
            logger.info(f"公共索引数量: {len(common_index)} / base:{len(base_standardized)} / alpha:{len(alpha_standardized)}")
            
            if len(common_index) > 0:
                # 使用公共索引对齐
                base_aligned = base_standardized.loc[common_index]
                alpha_aligned = alpha_standardized.loc[common_index]
                
                # 检查列名重叠
                overlapping_cols = set(base_aligned.columns) & set(alpha_aligned.columns)
                if overlapping_cols:
                    logger.warning(f"发现重叠列: {overlapping_cols}")
                    # 重命名alpha特征以避免冲突
                    alpha_renamed = alpha_aligned.rename(columns={col: f"alpha_{col}" for col in overlapping_cols})
                    fused = pd.concat([base_aligned, alpha_renamed], axis=1)
                else:
                    fused = pd.concat([base_aligned, alpha_aligned], axis=1)
                    
                logger.info(f"✅ 索引对齐合并成功: {fused.shape}")
                
            else:
                # 回退策略：时间戳近似对齐
                logger.warning("⚠️ 无公共索引，尝试时间戳近似对齐")
                fused = self._approximate_time_alignment(base_standardized, alpha_standardized)
                
                if fused is None:
                    # 最后回退：长度截断对齐
                    logger.warning("⚠️ 时间对齐失败，使用长度截断")
                    min_len = min(len(base_standardized), len(alpha_standardized))
                    fused = pd.concat([
                        base_standardized.iloc[:min_len], 
                        alpha_standardized.iloc[:min_len]
                    ], axis=1)
                    logger.info(f"⚠️ 长度对齐完成: {fused.shape}")
            
        except Exception as e:
            logger.error(f"❌ 特征合并失败: {e}")
            logger.info("🔄 回退到基础特征")
            return base_features
        
        logger.info(f"🎯 特征合并完成: {base_features.shape} + {alpha_features.shape} → {fused.shape}")
        return fused
    
    def _ensure_multiindex_format(self, df: pd.DataFrame, data_name: str) -> pd.DataFrame:
        """确保DataFrame是MultiIndex格式"""
        if isinstance(df.index, pd.MultiIndex):
            logger.debug(f"✅ {data_name} 已是MultiIndex格式")
            return df
            
        if 'date' in df.columns and 'ticker' in df.columns:
            try:
                dates = pd.to_datetime(df['date'])
                tickers = df['ticker']
                multi_idx = pd.MultiIndex.from_arrays([dates, tickers], names=['date', 'ticker'])
                
                df_multiindex = df.drop(['date', 'ticker'], axis=1).copy()
                df_multiindex.index = multi_idx
                
                logger.info(f"✅ {data_name} 转换为MultiIndex: {df_multiindex.shape}")
                return df_multiindex
                
            except Exception as e:
                logger.error(f"❌ {data_name} MultiIndex转换失败: {e}")
                return None
        else:
            logger.warning(f"⚠️ {data_name} 缺少date/ticker列，无法创建MultiIndex")
            return None
    
    def _approximate_time_alignment(self, base_df: pd.DataFrame, alpha_df: pd.DataFrame) -> pd.DataFrame:
        """时间戳近似对齐策略"""
        try:
            # 获取日期级别的索引
            base_dates = base_df.index.get_level_values('date').unique()
            alpha_dates = alpha_df.index.get_level_values('date').unique()
            
            # 找到时间重叠范围
            date_overlap = pd.Index(base_dates).intersection(pd.Index(alpha_dates))
            
            if len(date_overlap) > 0:
                # 按日期过滤
                base_filtered = base_df[base_df.index.get_level_values('date').isin(date_overlap)]
                alpha_filtered = alpha_df[alpha_df.index.get_level_values('date').isin(date_overlap)]
                
                # 再次尝试索引交集
                common_index = base_filtered.index.intersection(alpha_filtered.index)
                if len(common_index) > 0:
                    fused = pd.concat([
                        base_filtered.loc[common_index],
                        alpha_filtered.loc[common_index]
                    ], axis=1)
                    logger.info(f"✅ 时间近似对齐成功: {len(date_overlap)}个重叠日期, 最终{fused.shape}")
                    return fused
                    
        except Exception as e:
            logger.warning(f"时间近似对齐失败: {e}")
            
        return None
    
    def save_pipeline(self, filepath: Optional[str] = None) -> str:
        """保存特征管道"""
        if filepath is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.cache_dir / f"feature_pipeline_{timestamp}.pkl"
        
        pipeline_state = {
            'config': self.config,
            'scaler': self.scaler,
            # 'pca': None,  # 🚫 REMOVED - NO PCA
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
        # self.pca = None  # 🚫 REMOVED - NO PCA
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
            enable_pca=False,  # 🚫 ALWAYS FALSE - NO PCA
            # pca_variance_threshold=0.95,  # 🚫 REMOVED - NO PCA
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
            'has_pca': False,  # 🚫 ALWAYS FALSE - NO PCA
            'has_scaler': self.scaler is not None,
            'has_imputer': self.imputer is not None,
            'config': {
                'enable_alpha_summary': self.config.enable_alpha_summary,
                'enable_pca': False,  # 🚫 ALWAYS FALSE - NO PCA
                'enable_scaling': self.config.enable_scaling,
                'scaler_type': self.config.scaler_type
            }
        }