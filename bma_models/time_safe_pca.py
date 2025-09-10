"""
Time-Series Safe PCA Implementation
=====================================
防止时间泄露的PCA降维实现，确保每个时点只使用历史数据
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class TimeSeriesSafePCA:
    """时间序列安全的PCA降维器"""
    
    def __init__(self, 
                 n_components: float = 0.85,
                 min_history_days: int = 60,
                 refit_frequency: int = 21,
                 max_components: int = 10):
        """
        Args:
            n_components: 解释方差比例阈值或固定组件数
            min_history_days: 最小历史数据天数
            refit_frequency: PCA重新拟合频率(天)
            max_components: 最大主成分数量
        """
        self.n_components = n_components
        self.min_history_days = min_history_days
        self.refit_frequency = refit_frequency
        self.max_components = max_components
        
        # 内部状态
        self.pca_models = {}  # 按时间存储的PCA模型
        self.last_refit_date = None
        self.current_components = None
        self.imputer = SimpleImputer(strategy='median')
        
        logger.info(f"TimeSeriesSafePCA initialized: n_components={n_components}, "
                   f"min_history={min_history_days}d, refit_freq={refit_frequency}d")
    
    def fit_transform_safe(self, 
                          alpha_data: pd.DataFrame,
                          date_column: str = 'date') -> Tuple[pd.DataFrame, Dict]:
        """
        时间安全的PCA拟合和转换
        
        Args:
            alpha_data: MultiIndex DataFrame (date, ticker) 或包含date列
            date_column: 日期列名
            
        Returns:
            (pca_features, stats): PCA特征和统计信息
        """
        if alpha_data.empty:
            logger.warning("Alpha数据为空")
            return pd.DataFrame(), {}
        
        # 处理MultiIndex
        if isinstance(alpha_data.index, pd.MultiIndex):
            if 'date' in alpha_data.index.names:
                data = alpha_data.reset_index()
                date_column = 'date'
            else:
                logger.error("MultiIndex中缺少date级别")
                return pd.DataFrame(), {}
        else:
            data = alpha_data.copy()
        
        # 确保日期列存在且为datetime - 支持MultiIndex
        if date_column not in data.columns:
            # Check if date is in the index (MultiIndex case)
            if hasattr(data.index, 'names') and date_column in data.index.names:
                # Reset index to make date a column
                data = data.reset_index()
                logger.info(f"从MultiIndex中提取日期列: {date_column}")
            else:
                logger.error(f"缺少日期列: {date_column}")
                return pd.DataFrame(), {}
        
        data[date_column] = pd.to_datetime(data[date_column])
        
        # 获取数值特征列
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in [date_column, 'ticker']]
        
        if len(feature_cols) == 0:
            logger.warning("没有数值特征用于PCA")
            return pd.DataFrame(), {}
        
        # 按日期排序
        data_sorted = data.sort_values(date_column)
        unique_dates = sorted(data_sorted[date_column].unique())
        
        # 存储结果
        pca_results = []
        stats = {
            'n_dates_processed': 0,
            'avg_components': 0,
            'variance_explained_history': [],
            'refit_dates': []
        }
        
        logger.info(f"开始时间安全PCA处理: {len(unique_dates)} 个交易日, {len(feature_cols)} 个特征")
        
        for i, current_date in enumerate(unique_dates):
            # 获取当前日期数据
            current_data = data_sorted[data_sorted[date_column] == current_date]
            
            if len(current_data) == 0:
                continue
            
            # 获取历史数据（不包括当前日期）
            historical_dates = unique_dates[:i]  # 只使用历史日期
            
            if len(historical_dates) < self.min_history_days:
                # 历史数据不足，跳过
                logger.debug(f"日期 {current_date}: 历史数据不足({len(historical_dates)}天)")
                continue
            
            # 准备历史数据用于PCA拟合
            historical_data = data_sorted[data_sorted[date_column].isin(historical_dates)]
            hist_features = historical_data[feature_cols].fillna(0)
            
            # 检查是否需要重新拟合PCA
            need_refit = (
                self.last_refit_date is None or
                (current_date - self.last_refit_date).days >= self.refit_frequency or
                self.current_components is None
            )
            
            if need_refit and len(hist_features) > 10:  # 至少需要10个样本
                self._refit_pca(hist_features, current_date, stats)
            
            # 使用当前PCA模型转换当前日期数据
            if self.current_components is not None:
                current_features = current_data[feature_cols].fillna(0)
                pca_transformed = self._transform_current_data(current_features, current_date)
                
                if pca_transformed is not None:
                    # 添加日期和ticker信息
                    result_df = pd.DataFrame(pca_transformed)
                    result_df[date_column] = current_date
                    
                    if 'ticker' in current_data.columns:
                        result_df['ticker'] = current_data['ticker'].values
                    
                    pca_results.append(result_df)
                    stats['n_dates_processed'] += 1
        
        if not pca_results:
            logger.warning("PCA处理失败，无有效结果")
            return pd.DataFrame(), stats
        
        # 合并所有结果
        final_result = pd.concat(pca_results, ignore_index=True)
        
        # 重命名PCA列
        pca_cols = [col for col in final_result.columns 
                   if col not in [date_column, 'ticker']]
        rename_dict = {col: f'alpha_pca_{i+1}' for i, col in enumerate(pca_cols)}
        final_result = final_result.rename(columns=rename_dict)
        
        # 设置MultiIndex（如果有date和ticker列）
        if date_column in final_result.columns and 'ticker' in final_result.columns:
            final_result = final_result.set_index([date_column, 'ticker'])
            final_result.index.names = ['date', 'ticker']
            logger.info(f"[DEBUG] TimeSafePCA恢复MultiIndex结构 - 形状: {final_result.shape}")
        elif isinstance(alpha_data.index, pd.MultiIndex):
            # 保持原始MultiIndex结构
            logger.info(f"[DEBUG] TimeSafePCA保持原始MultiIndex结构")
        
        stats['final_shape'] = final_result.shape
        stats['avg_components'] = np.mean([len(self.pca_models[date]['n_components']) 
                                         for date in self.pca_models.keys()] if self.pca_models else [0])
        
        logger.info(f"时间安全PCA完成: {stats['n_dates_processed']} 天处理成功, "
                   f"平均 {stats['avg_components']:.1f} 个主成分")
        
        return final_result, stats
    
    def _refit_pca(self, historical_features: pd.DataFrame, current_date, stats: Dict):
        """重新拟合PCA模型"""
        try:
            # 填充缺失值
            hist_imputed = self.imputer.fit_transform(historical_features)
            
            # 拟合PCA
            pca = PCA()
            pca.fit(hist_imputed)
            
            # 确定主成分数量
            if isinstance(self.n_components, float):
                # 基于解释方差比例
                cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum_variance >= self.n_components) + 1
            else:
                # 固定数量
                n_components = int(self.n_components)
            
            # 限制最大组件数
            n_components = min(n_components, self.max_components, historical_features.shape[1])
            n_components = max(n_components, 1)
            
            # 重新拟合最终PCA
            final_pca = PCA(n_components=n_components)
            final_pca.fit(hist_imputed)
            
            # 存储模型
            self.pca_models[current_date] = {
                'pca': final_pca,
                'imputer': self.imputer,
                'n_components': n_components,
                'variance_explained': final_pca.explained_variance_ratio_.sum()
            }
            
            self.current_components = final_pca
            self.last_refit_date = current_date
            
            stats['refit_dates'].append(current_date)
            stats['variance_explained_history'].append(final_pca.explained_variance_ratio_.sum())
            
            logger.debug(f"PCA重新拟合: {current_date}, {n_components}个组件, "
                        f"解释方差: {final_pca.explained_variance_ratio_.sum():.3f}")
            
        except Exception as e:
            logger.warning(f"PCA拟合失败 {current_date}: {e}")
    
    def _transform_current_data(self, current_features: pd.DataFrame, current_date) -> Optional[np.ndarray]:
        """使用当前PCA模型转换数据"""
        try:
            if self.current_components is None:
                return None
            
            # 使用相同的填充策略
            current_imputed = self.imputer.transform(current_features.fillna(0))
            
            # PCA转换
            pca_transformed = self.current_components.transform(current_imputed)
            
            return pca_transformed
            
        except Exception as e:
            logger.warning(f"PCA转换失败 {current_date}: {e}")
            return None
    
    def get_feature_names(self, n_components: Optional[int] = None) -> List[str]:
        """获取PCA特征名称"""
        if self.current_components is not None:
            n_comp = self.current_components.n_components_
        else:
            n_comp = n_components or 5
        
        return [f'alpha_pca_{i+1}' for i in range(n_comp)]


def create_time_safe_alpha_summarizer(n_components: float = 0.85,
                                     min_history: int = 60,
                                     refit_freq: int = 21) -> TimeSeriesSafePCA:
    """创建时间安全的Alpha摘要器"""
    return TimeSeriesSafePCA(
        n_components=n_components,
        min_history_days=min_history,
        refit_frequency=refit_freq
    )


# 向后兼容性函数
def apply_time_safe_pca(alpha_data: pd.DataFrame,
                       n_components: float = 0.85,
                       **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    应用时间安全的PCA降维
    
    向后兼容原有接口的包装函数
    """
    safe_pca = TimeSeriesSafePCA(n_components=n_components, **kwargs)
    return safe_pca.fit_transform_safe(alpha_data)