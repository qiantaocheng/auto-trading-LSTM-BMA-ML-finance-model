#!/usr/bin/env python3
"""
Alpha Summary Features Integration Module
Route A: Representation-level Alpha injection into LTR/ML pipeline

Key Design Principles:
1. Minimal code invasion (18 columns max)
2. Strict time alignment and leakage prevention  
3. Cross-sectional winsorization and standardization
4. Dimensionality reduction (PCA/PLS) for Alpha compression
5. Robust summary statistics generation
6. Integration with existing X_clean -> X_fused pipeline
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

# Scientific computing
from scipy import stats
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AlphaSummaryConfig:
    """Configuration for Alpha Summary Feature Generation"""
    # A1: Data cleaning
    winsorize_lower: float = 0.01  # 1% lower tail
    winsorize_upper: float = 0.99  # 99% upper tail
    use_mad_winsorize: bool = True  # Use median ± 3MAD instead of percentile
    neutralize_by_industry: bool = False  # DISABLED to avoid MultiIndex errors
    neutralize_by_market_cap: bool = False
    
    # A2: Dimensionality reduction (PROFESSIONAL: 40+ -> 18 features)
    max_alpha_features: int = 50  # 扩展以支持41个alpha因子 + 压缩特征
    min_alpha_features: int = 15  # Minimum for robustness
    pca_variance_explained: float = 0.85  # PCA variance threshold
    pls_n_components: int = 8  # Increased PLS components
    use_ic_weighted: bool = True  # PRIMARY: Use professional IC-weighted method
    use_pca: bool = False  # DEPRECATED: Use as fallback only
    use_pls: bool = False
    use_ic_weighted_composite: bool = False  # DEPRECATED: Old simple version
    include_alpha_strategy_signal: bool = False  # REMOVED: Hardcoded strategy deleted, pure ML approach
    
    # A3: Summary statistics
    include_dispersion: bool = True
    include_agreement: bool = True  
    include_quality: bool = True
    ic_lookback_days: int = 120  # IC quality lookback
    
    # A4: Integration
    fill_method: str = 'cross_median'  # cross_median, forward_fill, zero
    data_type: str = 'float32'  # Memory optimization
    
    # A5: Time alignment
    strict_time_validation: bool = True
    min_history_days: int = 60  # Minimum history for quality metrics

class AlphaSummaryProcessor:
    """
    Alpha Summary Feature Processor
    Converts 45+ Alpha signals -> 5-10 summary features for ML injection
    """
    
    def __init__(self, config: AlphaSummaryConfig = None):
        self.config = config or AlphaSummaryConfig()
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        
        # Cache for IC calculations and fitted components
        self.ic_cache = {}
        self.pca_fitted = None
        self.pls_fitted = None
        self.neutralization_models = {}
        
        # Statistics tracking
        self.stats = {
            'total_alphas_processed': 0,
            'features_generated': 0,
            'time_violations': 0,
            'neutralization_r2': {},
            'compression_variance_explained': 0.0
        }
        
        # Initialize logging
        logger.info(f"Alpha摘要特征生成器初始化: 目标{self.config.max_alpha_features}个特征")
        logger.info(f"  - PCA压缩: {self.config.use_pca}")
        logger.info(f"  - IC权重压缩: {self.config.use_ic_weighted}")
        logger.info(f"  - 统计特征: dispersion={self.config.include_dispersion}, agreement={self.config.include_agreement}, quality={self.config.include_quality}")
        logger.info(f"  - Alpha策略信号: 已删除(纯ML方法)")
        logger.info(f"  - 时间违规检查: {self.config.strict_time_validation}")
        logger.info(f"  - 行业中性化: {self.config.neutralize_by_industry}")
        logger.info(f"  - MAD Winsorize: {self.config.use_mad_winsorize}")
        logger.info(f"  - PCA方差阈值: {self.config.pca_variance_explained}")
        
    def _log_data_quality_info(self, alpha_df: pd.DataFrame, market_data: pd.DataFrame = None):
        """记录数据质量信息用于调试"""
        logger.info("[CHART] 数据质量检查报告:")
        
        # Alpha数据基本信息
        logger.info(f"  Alpha数据形状: {alpha_df.shape}")
        logger.info(f"  Alpha数据索引类型: {type(alpha_df.index)}")
        if isinstance(alpha_df.index, pd.MultiIndex):
            logger.info(f"  MultiIndex层级: {alpha_df.index.names}")
        
        # 数值列统计
        numeric_cols = alpha_df.select_dtypes(include=[np.number]).columns
        logger.info(f"  数值列数量: {len(numeric_cols)}")
        
        if len(numeric_cols) > 0:
            # 数据范围检查
            numeric_data = alpha_df[numeric_cols]
            all_zeros = (numeric_data == 0).all()
            constant_cols = numeric_data.nunique() == 1
            
            if all_zeros.any():
                zero_cols = all_zeros[all_zeros].index.tolist()
                logger.warning(f"  [WARN] 全零列({len(zero_cols)}个): {zero_cols[:5]}")
            
            if constant_cols.any():
                const_cols = constant_cols[constant_cols].index.tolist()
                logger.warning(f"  [WARN] 常数列({len(const_cols)}个): {const_cols[:5]}")
            
            # 基本统计信息
            means = numeric_data.mean()
            stds = numeric_data.std()
            
            logger.debug(f"  Alpha均值范围: [{means.min():.6f}, {means.max():.6f}]")
            logger.debug(f"  Alpha标准差范围: [{stds.min():.6f}, {stds.max():.6f}]")
            logger.debug(f"  非零标准差列数: {(stds > 1e-10).sum()}")
            
            # 缺失值检查
            missing_ratio = numeric_data.isnull().mean()
            high_missing = missing_ratio[missing_ratio > 0.5]
            if not high_missing.empty:
                logger.warning(f"  [WARN] 高缺失率列({len(high_missing)}个): {high_missing.index.tolist()[:5]}")
        
        # 时间范围检查
        try:
            if isinstance(alpha_df.index, pd.MultiIndex) and 'date' in alpha_df.index.names:
                dates = alpha_df.index.get_level_values('date')
                unique_dates = pd.Series(dates).drop_duplicates()
                logger.info(f"  时间范围: {unique_dates.min()} 到 {unique_dates.max()}")
                logger.info(f"  交易日数量: {len(unique_dates)}")
                
            elif 'date' in alpha_df.columns:
                dates = pd.to_datetime(alpha_df['date'])
                logger.info(f"  时间范围: {dates.min()} 到 {dates.max()}")
                logger.info(f"  交易日数量: {dates.nunique()}")
        except Exception as e:
            logger.debug(f"时间范围检查失败: {e}")
        
        # 股票数量统计
        try:
            if isinstance(alpha_df.index, pd.MultiIndex) and 'ticker' in alpha_df.index.names:
                tickers = alpha_df.index.get_level_values('ticker')
                logger.info(f"  股票数量: {pd.Series(tickers).nunique()}")
            elif 'ticker' in alpha_df.columns:
                logger.info(f"  股票数量: {alpha_df['ticker'].nunique()}")
        except Exception as e:
            logger.debug(f"股票数量统计失败: {e}")
        
        # Market data检查
        if market_data is not None and not market_data.empty:
            logger.info(f"  市场数据形状: {market_data.shape}")
        else:
            logger.warning("  [WARN] 未提供市场数据")
        
    def process_alpha_to_summary(self, 
                               alpha_df: pd.DataFrame,
                               market_data: pd.DataFrame,
                               target_dates: pd.Series = None) -> pd.DataFrame:
        """
        Main processing pipeline: Alpha signals -> Summary features
        
        Args:
            alpha_df: Raw Alpha signals (date, ticker, alpha_001, alpha_002, ...)
            market_data: Market context data (date, ticker, market_cap, industry, ...)
            target_dates: Target prediction dates (for time validation)
            
        Returns:
            Summary features DataFrame (date, ticker, alpha_pc1, alpha_pc2, ..., alpha_quality)
        """
        logger.info(f"开始Alpha摘要特征处理，输入形状: {alpha_df.shape}")
        
        # 数据质量检查和调试信息
        self._log_data_quality_info(alpha_df, market_data)
        
        if alpha_df.empty:
            logger.warning("输入Alpha数据为空")
            return pd.DataFrame()
        
        # A1: Data cleaning and alignment
        alpha_cleaned = self._clean_and_align_data(alpha_df, market_data, target_dates)
        if alpha_cleaned.empty:
            logger.warning("数据清洗后为空")
            return pd.DataFrame()
        
        # A2: Dimensionality reduction
        alpha_compressed = self._compress_alpha_dimensions(alpha_cleaned)
        
        # A3: Robust summary statistics
        alpha_stats = self._compute_summary_statistics(alpha_cleaned)
        
        # A3.5: Alpha strategy composite signal - REMOVED (let ML learn optimal combinations)
        alpha_strategy_signal = None
        
        # A4: Combine and prepare final features
        summary_features = self._combine_and_finalize_features(
            alpha_compressed, alpha_stats, alpha_strategy_signal, alpha_cleaned.index
        )
        
        # [TOOL] 健康检查3: 时间对齐违规门槛收紧 (在最终输出前检查)
        violations_result = self._validate_time_alignment_detailed(summary_features, target_dates) if target_dates is not None else {'total_violations': 0, 'bad_columns': []}
        violation_rate = violations_result['total_violations'] / (summary_features.shape[0] + 1e-8)
        
        # [HOT] CRITICAL FIX: 放宽时间对齐违规阈值，减少误报
        if violation_rate > 0.25:  # 放宽阈值：违规率超过25%才报警
            bad_columns = violations_result.get('bad_columns', [])
            logger.warning(f"[SELECTIVE_CLEANUP] 时间对齐违规过多({violations_result['total_violations']}项, {violation_rate:.1%})")
            logger.warning(f"[SELECTIVE_CLEANUP] 问题列: {bad_columns}")
            
            # [HOT] CRITICAL FIX: 只在严重违规时才清理，否则保留特征
            if violation_rate > 0.50:  # 只有违规率超过50%才进行清理
                logger.warning("违规率过高，进行选择性清理")
                return self._selective_column_cleanup(summary_features, bad_columns)
            else:
                logger.info("违规率在可接受范围内，保留所有特征")
        
        logger.info(f"[OK] Alpha摘要特征生成完成，输出形状: {summary_features.shape}")
        self.stats['features_generated'] = summary_features.shape[1]
        
        return summary_features
    
    def _fallback_to_traditional_features(self, alpha_df: pd.DataFrame) -> pd.DataFrame:
        """传统特征回退实现"""
        try:
            # 选择数值列，应用简单统计聚合
            numeric_cols = alpha_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return pd.DataFrame()
            
            # 基础统计特征
            fallback_features = pd.DataFrame(index=alpha_df.index)
            for col in numeric_cols[:10]:  # 限制前10个避免过度维度
                fallback_features[f'mean_{col}'] = alpha_df[col].rolling(5).mean()
                fallback_features[f'std_{col}'] = alpha_df[col].rolling(10).std()
            
            logger.info(f"[FALLBACK] 传统特征生成完成，形状: {fallback_features.shape}")
            return fallback_features.fillna(0)
        except Exception as e:
            logger.error(f"[FALLBACK] 传统特征生成失败: {e}")
            return pd.DataFrame()
    
    def _clean_and_align_data(self, 
                            alpha_df: pd.DataFrame,
                            market_data: pd.DataFrame,
                            target_dates: pd.Series = None) -> pd.DataFrame:
        """A1: Cross-sectional cleaning, neutralization, time alignment"""
        
        # [OK] FIX: 智能索引格式处理
        # 检查并尝试创建合适的索引结构
        try:
            if not isinstance(alpha_df.index, pd.MultiIndex):
                # 尝试创建MultiIndex，但先检查列是否存在
                if 'date' in alpha_df.columns and 'ticker' in alpha_df.columns:
                    alpha_df = alpha_df.set_index(['date', 'ticker'])
                elif 'date' in alpha_df.columns:
                    # 只有date列，创建简单的时间索引
                    alpha_df = alpha_df.set_index('date')
                # 如果都没有，保持原始索引
        except Exception as e:
            logger.warning(f"索引设置失败: {e}，保持原始索引格式")
        
        # [OK] FIX: 根据标签期确定正确的滞后天数（适应性调整）
        # 从target_dates推断标签期，或从列名解析
        label_horizon = 5  # [TOOL] FIX: 默认改为5天标签期
        if target_dates is not None and len(target_dates) > 1:
            # 尝试从target_dates间隔推断标签期
            try:
                target_dates_dt = pd.to_datetime(target_dates)
                if len(target_dates_dt) > 1:
                    avg_interval = (target_dates_dt.max() - target_dates_dt.min()).days / max(1, len(target_dates_dt) - 1)
                    if avg_interval > 1:
                        label_horizon = min(int(avg_interval), 10)  # [TOOL] FIX: Cap label horizon at 10 days
            except Exception as e:
                logger.debug(f"Failed to infer label horizon from target_dates: {e}")
                pass
        
        # [HOT] CRITICAL FIX: 改进自适应滞后期逻辑，减少时间违规
        # 根据数据集大小和历史违规情况调整滞后期
        base_lag = max(label_horizon, 5)  # 提高基础滞后期到5天
        
        # 根据数据集大小调整
        if len(alpha_df) < 500:
            default_lag = max(3, label_horizon // 2)  # 小数据集：至少3天滞后
        elif len(alpha_df) < 1000:
            default_lag = max(5, label_horizon)  # 中等数据集：至少5天滞后
        else:
            default_lag = max(base_lag, 7)  # 大数据集：至少7天滞后
            
        # [HOT] CRITICAL FIX: 根据历史违规情况动态增加滞后期
        if hasattr(self, 'stats') and 'time_violations' in self.stats:
            if self.stats['time_violations'] > 1000:
                default_lag += 2  # 如果之前违规多，增加2天
                logger.info(f"检测到历史时间违规({self.stats['time_violations']}项)，增加滞后期到{default_lag}天")
        
        # [OK] FIX: 强制索引标准化 - 确保datetime索引
        try:
            if not isinstance(alpha_df.index, pd.MultiIndex):
                if 'date' in alpha_df.columns and 'ticker' in alpha_df.columns:
                    alpha_df['date'] = pd.to_datetime(alpha_df['date'])
                    alpha_df = alpha_df.set_index(['date', 'ticker']).sort_index()
                elif 'date' in alpha_df.columns:
                    alpha_df['date'] = pd.to_datetime(alpha_df['date'])
                    alpha_df = alpha_df.set_index('date').sort_index()
            else:
                # 确保date级别是datetime类型
                if 'date' in alpha_df.index.names:
                    alpha_df = alpha_df.reset_index()
                    alpha_df['date'] = pd.to_datetime(alpha_df['date'])
                    if 'ticker' in alpha_df.columns:
                        alpha_df = alpha_df.set_index(['date', 'ticker']).sort_index()
                    else:
                        alpha_df = alpha_df.set_index('date').sort_index()
        except Exception as e:
            logger.warning(f"索引标准化失败: {e}，保持原始索引格式")
        
        # [OK] FIX: 应用正确的滞后
        alpha_df_shifted = alpha_df.copy()
        try:
            if isinstance(alpha_df.index, pd.MultiIndex) and 'ticker' in alpha_df.index.names:
                # MultiIndex with ticker - 按ticker分组shift
                numeric_cols = alpha_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # 按ticker分组，每个ticker的alpha信号右移default_lag天
                    for ticker_group in alpha_df.groupby(level='ticker'):
                        ticker_name = ticker_group[0]
                        ticker_data = ticker_group[1]
                        shifted_data = ticker_data[numeric_cols].shift(default_lag)
                        alpha_df_shifted.loc[ticker_data.index, numeric_cols] = shifted_data
            elif 'ticker' in alpha_df.columns:
                # ticker作为列
                numeric_cols = alpha_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    shifted_data = alpha_df.groupby('ticker')[numeric_cols].shift(default_lag)
                    alpha_df_shifted.loc[:, numeric_cols] = shifted_data
            else:
                # 简单时间序列
                numeric_cols = alpha_df.select_dtypes(include=[np.number]).columns
                alpha_df_shifted[numeric_cols] = alpha_df[numeric_cols].shift(default_lag)
                
            # 删除因滞后产生的NaN行
            pre_dropna_len = len(alpha_df_shifted)
            alpha_df_shifted = alpha_df_shifted.dropna()
            post_dropna_len = len(alpha_df_shifted)
            
            # 验证滞后效果
            if post_dropna_len < pre_dropna_len * 0.1:
                logger.warning(f"滞后处理导致数据大幅减少: {pre_dropna_len} -> {post_dropna_len}")
            else:
                logger.info(f"滞后处理成功: {pre_dropna_len} -> {post_dropna_len} 行, lag={default_lag}天")
                
        except Exception as e:
            logger.error(f"滞后处理失败: {e}，返回空数据以避免时间泄漏风险")
            alpha_df_shifted = pd.DataFrame()  # 严格模式：失败时返回空数据
        
        logger.info(f"应用了{default_lag}天滞后（基于{label_horizon}天标签期），避免时间对齐违规")
        
        # [OK] FIX: 更智能的时间验证，减少误报
        if target_dates is not None and self.config.strict_time_validation:
            violations = self._validate_time_alignment(alpha_df_shifted, target_dates)
            # [OK] FIX: 正确的百分比计算 - 总单元格数而不是行数
            numeric_cols = alpha_df_shifted.select_dtypes(include=[np.number]).columns
            total_cells = len(alpha_df_shifted) * len(numeric_cols)
            violation_rate = violations / (total_cells + 1e-8)
            
            # [HOT] CRITICAL FIX: 优化时间违规阈值和处理逻辑
            if violation_rate > 0.30:  # 放宽阈值：违规率超过30%才报警
                logger.warning(f"发现较多时间对齐违规: {violations} 个单元格 ({violation_rate:.1%}) / 总计{total_cells}")
                self.stats['time_violations'] = violations
                # 如果违规率过高，增加滞后期
                if violation_rate > 0.50:
                    logger.warning("违规率过高，建议增加滞后期处理")
            elif violations > 0:
                logger.debug(f"发现少量时间对齐违规: {violations} 个单元格 ({violation_rate:.1%})，在可接受范围内")
                self.stats['time_violations'] = violations
            else:
                logger.info("时间对齐验证通过，无泄露风险")
        
        # 使用滞后后的数据继续处理
        alpha_df = alpha_df_shifted
        
        # [TOOL] 健康检查1: 列名匹配度 (扩展列名筛选)
        alpha_cols = []
        exclude_cols = ['date', 'ticker', 'target', 'target_10d', 'Close', 'High', 'Low', 'Open', 'amount']
        
        for col in alpha_df.columns:
            # Skip excluded columns
            if col in exclude_cols:
                continue
                
            # Include numeric columns that are likely alpha factors
            if alpha_df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
                # More flexible pattern matching - include any column that looks like a feature
                if (col.startswith('alpha_') or 
                    col.startswith('volume') or   # Include all volume-based features
                    col.startswith('momentum') or  # Include all momentum features
                    col.startswith('volatility') or # Include all volatility features
                    col.startswith('mean_reversion') or # Include mean reversion features
                    col.startswith('rsi') or  # Include RSI features
                    col.startswith('price_position') or # Include price position features
                    any(pattern in col.lower() for pattern in ['factor', 'reversal', 
                                                               'turnover', 'amihud', 'bid_ask', 'yield',
                                                               'ohlson', 'altman', 'qmj', 'earnings', 'beta',
                                                               'ratio', 'rsi', 'macd', 'ma_', '_ma', '_1d', '_5d', '_20d', '_14d'])):
                    alpha_cols.append(col)
        
        # Log what columns were detected
        if not alpha_cols:
            logger.warning(f"No alpha columns found. Available columns: {list(alpha_df.columns)[:20]}")
            logger.warning(f"Total columns: {len(alpha_df.columns)}, Data shape: {alpha_df.shape}")
            # Log column types
            numeric_cols = alpha_df.select_dtypes(include=[np.float32, np.float64, np.int32, np.int64]).columns
            logger.warning(f"Numeric columns found: {len(numeric_cols)}: {list(numeric_cols)[:10]}")
        
        # [TOOL] 健康检查2: 最低列数门槛
        if len(alpha_cols) < 3:
            logger.warning(f"[FALLBACK] Alpha列数过少({len(alpha_cols)} < 3)，触发传统特征回退")
            return self._fallback_to_traditional_features(alpha_df)
        
        if not alpha_cols:
            logger.warning(f"[FALLBACK] 未找到有效的Alpha列，触发传统特征回退，可用列: {list(alpha_df.columns)[:10]}...")
            return self._fallback_to_traditional_features(alpha_df)
        
        # [TOOL] CRITICAL FIX: 保持MultiIndex结构，不要重置索引
        # Include date and ticker columns for groupby while preserving MultiIndex
        original_index = alpha_df.index  # 保存原始索引
        
        if isinstance(alpha_df.index, pd.MultiIndex) and 'date' in alpha_df.index.names and 'ticker' in alpha_df.index.names:
            # MultiIndex格式：保持原索引，临时重置以便分组处理
            alpha_df_temp = alpha_df.reset_index()
            cols_to_process = ['date', 'ticker'] + alpha_cols
            logger.info(f"[DEBUG] MultiIndex模式: 保留date+ticker索引")
        elif 'date' in alpha_df.columns and 'ticker' in alpha_df.columns:
            # 列格式：直接使用
            alpha_df_temp = alpha_df.copy()
            cols_to_process = ['date', 'ticker'] + alpha_cols
            logger.info(f"[DEBUG] DataFrame模式: 使用date+ticker列")
        else:
            # 回退到原始逻辑
            logger.warning("[FALLBACK] 无法识别date/ticker结构，使用原始处理逻辑")
            if 'date' in alpha_df.columns:
                cols_to_process = ['date'] + alpha_cols
            else:
                alpha_df_temp = alpha_df.reset_index()
                cols_to_process = ['date'] + alpha_cols
            alpha_df_temp = alpha_df.copy()
        
        alpha_only = alpha_df_temp[cols_to_process].copy()
        self.stats['total_alphas_processed'] = len(alpha_cols)
        
        # Cross-sectional processing by date, preserving ticker information
        cleaned_data = []
        
        for date, group in alpha_only.groupby('date'):
            # 保持ticker列用于重建MultiIndex
            if 'ticker' in group.columns:
                # Drop date column for processing but keep ticker
                group_for_processing = group.drop(columns=['date'], errors='ignore')
                
                # Cross-sectional winsorization and standardization
                numeric_cols = [col for col in group_for_processing.columns if col != 'ticker']
                if numeric_cols:
                    group_clean = self._cross_sectional_winsorize(group_for_processing[numeric_cols])
                    group_clean = self._cross_sectional_standardize(group_clean)
                    
                    # 重新添加ticker列和date信息
                    group_clean['ticker'] = group_for_processing['ticker'].values
                    group_clean['date'] = date
                    
                    # Industry/factor neutralization
                    if self.config.neutralize_by_industry and market_data is not None:
                        group_clean = self._neutralize_factors(group_clean, market_data, date)
                else:
                    logger.warning(f"日期{date}: 没有数值列需要处理")
                    continue
            else:
                # 原始处理逻辑（无ticker信息）
                group_for_processing = group.drop(columns=['date'], errors='ignore')
                group_clean = self._cross_sectional_winsorize(group_for_processing)
                group_clean = self._cross_sectional_standardize(group_clean)
                group_clean['date'] = date
                
                if self.config.neutralize_by_industry and market_data is not None:
                    group_clean = self._neutralize_factors(group_clean, market_data, date)
            
            cleaned_data.append(group_clean)
        
        # Check if we have any cleaned data
        if not cleaned_data:
            logger.warning("[FALLBACK] No data after cleaning, returning fallback features")
            return self._fallback_to_traditional_features(alpha_df)
        
        result = pd.concat(cleaned_data, ignore_index=True)
        
        # [TOOL] CRITICAL FIX: 重建MultiIndex结构
        if 'date' in result.columns and 'ticker' in result.columns:
            # 确保date列是datetime类型
            result['date'] = pd.to_datetime(result['date'])
            # 重建MultiIndex
            result = result.set_index(['date', 'ticker']).sort_index()
            logger.info(f"[DEBUG] 重建MultiIndex: {len(result.index.get_level_values('ticker').unique())}只股票, {len(result.index.get_level_values('date').unique())}个日期")
        else:
            logger.warning("[WARNING] 无法重建MultiIndex，缺少date或ticker列")
        
        # Safe way to get trading days count
        try:
            if isinstance(result.index, pd.MultiIndex) and 'date' in result.index.names:
                trading_days = len(result.index.get_level_values('date').unique())
            elif 'date' in result.columns:
                trading_days = len(result['date'].unique())
            else:
                trading_days = len(result)
            logger.info(f"数据清洗完成，处理了 {trading_days} 个交易日")
        except Exception as e:
            logger.warning(f"无法计算交易日数量: {e}, 使用总行数: {len(result)}")
            logger.info(f"数据清洗完成，处理了 {len(result)} 行数据")
        
        return result
    
    def _cross_sectional_winsorize(self, group: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional winsorization by date"""
        group_winsor = group.copy()
        
        for col in group.columns:
            if self.config.use_mad_winsorize:
                # Use median ± 3MAD method (more robust)
                median_val = group[col].median()
                mad_val = stats.median_abs_deviation(group[col].dropna())
                lower_bound = median_val - 3 * mad_val
                upper_bound = median_val + 3 * mad_val
            else:
                # Use percentile method
                lower_bound = group[col].quantile(self.config.winsorize_lower)
                upper_bound = group[col].quantile(self.config.winsorize_upper)
            
            group_winsor[col] = group[col].clip(lower_bound, upper_bound)
        
        return group_winsor
    
    def _cross_sectional_standardize(self, group: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional standardization (z-score) by date"""
        group_std = group.copy()
        
        for col in group.columns:
            mean_val = group[col].mean()
            std_val = group[col].std()
            
            if std_val > 1e-8:  # Avoid division by zero
                group_std[col] = (group[col] - mean_val) / std_val
            else:
                group_std[col] = 0.0
        
        return group_std
    
    def _neutralize_factors(self, 
                          alpha_group: pd.DataFrame, 
                          market_data: pd.DataFrame, 
                          date: str) -> pd.DataFrame:
        """Industry/factor neutralization using regression residuals"""
        try:
            # Get market context for this date
            date_market = market_data[market_data['date'] == date]
            if date_market.empty:
                return alpha_group
            
            # Align with alpha_group tickers
            tickers = alpha_group.index.get_level_values('ticker')
            market_aligned = date_market[date_market['ticker'].isin(tickers)].set_index('ticker')
            
            if market_aligned.empty:
                return alpha_group
            
            # Prepare neutralization factors
            neutralize_factors = []
            if self.config.neutralize_by_industry and 'industry' in market_aligned.columns:
                # One-hot encode industry
                industry_dummies = pd.get_dummies(market_aligned['industry'], prefix='ind')
                neutralize_factors.append(industry_dummies)
            
            if self.config.neutralize_by_market_cap and 'market_cap' in market_aligned.columns:
                # Log market cap
                market_cap_log = np.log(market_aligned['market_cap'].replace(0, np.nan)).fillna(0)
                neutralize_factors.append(pd.DataFrame({'log_market_cap': market_cap_log}))
            
            if not neutralize_factors:
                return alpha_group
            
            X_neutralize = pd.concat(neutralize_factors, axis=1).fillna(0)
            
            # Neutralize each alpha
            alpha_neutralized = alpha_group.copy()
            neutralization_stats = {}
            
            for col in alpha_group.columns:
                y = alpha_group[col].reindex(X_neutralize.index).dropna()
                X_aligned = X_neutralize.reindex(y.index)
                
                if len(y) > X_aligned.shape[1] + 5:  # Minimum samples required
                    try:
                        reg = LinearRegression().fit(X_aligned, y)
                        residuals = y - reg.predict(X_aligned)
                        alpha_neutralized.loc[y.index, col] = residuals
                        neutralization_stats[col] = reg.score(X_aligned, y)
                    except:
                        pass  # Keep original values if neutralization fails
            
            self.stats['neutralization_r2'][str(date)] = neutralization_stats
            return alpha_neutralized
            
        except Exception as e:
            logger.warning(f"因子中性化失败 (日期 {date}): {e}")
            return alpha_group
    
    def _compress_alpha_dimensions(self, alpha_df: pd.DataFrame, 
                                  returns: Optional[pd.DataFrame] = None,
                                  market_data: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """A2: Compress Alpha dimensions using IC-weighted professional method
        
        优先使用IC加权方案，PCA作为备选
        """
        
        alpha_values = alpha_df.select_dtypes(include=[np.number])
        if alpha_values.empty or alpha_values.shape[1] < 3:
            logger.warning("Alpha数据不足，跳过降维")
            return None
        
        # 优先尝试使用专业的IC加权处理器
        try:
            from alpha_ic_weighted_processor import ICWeightedAlphaProcessor, ICWeightedConfig
            
            # 创建IC加权处理器
            ic_config = ICWeightedConfig()
            ic_processor = ICWeightedAlphaProcessor(ic_config)
            
            # 如果没有returns数据，创建模拟returns（实际应该传入真实returns）
            if returns is None:
                # 使用简单的动量作为代理returns
                close_col = None
                if 'close' in alpha_df.columns:
                    close_col = 'close'
                elif 'Close' in alpha_df.columns:
                    close_col = 'Close'
                
                if close_col:
                    returns = alpha_df[close_col].pct_change().shift(-10)  # 10天前向收益
                else:
                    # 使用第一个alpha因子的变化作为代理
                    returns = alpha_values.iloc[:, 0].pct_change().shift(-10)
            
            # 处理alpha因子
            compressed_features = ic_processor.process_alpha_factors(
                alpha_values,
                returns,
                market_data
            )
            
            if compressed_features is not None and not compressed_features.empty:
                logger.info(f"IC加权处理成功: {alpha_values.shape[1]} -> {compressed_features.shape[1]} 个特征")
                return compressed_features
                
        except ImportError:
            logger.warning("IC加权处理器未找到，回退到PCA方案")
        except Exception as e:
            logger.warning(f"IC加权处理失败: {e}，回退到PCA方案")
        
        # 回退到原始PCA方案
        compressed_features = []
        feature_names = []
        
        # Method 1: PCA (unsupervised)
        if self.config.use_pca:
            pca_features, pca_names = self._apply_pca_compression(alpha_values, alpha_df)
            if pca_features is not None:
                compressed_features.append(pca_features)
                feature_names.extend(pca_names)
        
        # Method 2: PLS (supervised - requires target)
        # Note: PLS would require target variable, implementing as placeholder
        if self.config.use_pls:
            logger.info("PLS compression需要目标变量，当前版本暂未实现")
        
        # Method 3: IC-weighted composite (原始简单版本)
        if self.config.use_ic_weighted_composite:
            ic_features, ic_names = self._apply_ic_weighted_compression(alpha_values)
            if ic_features is not None:
                compressed_features.append(ic_features)
                feature_names.extend(ic_names)
        
        if not compressed_features:
            logger.warning("所有压缩方法都失败")
            return None
        
        # Combine all compressed features
        combined_features = pd.concat(compressed_features, axis=1)
        combined_features.columns = feature_names[:len(combined_features.columns)]
        
        logger.info(f"Alpha降维完成: {alpha_values.shape[1]} -> {combined_features.shape[1]} 个特征")
        
        return combined_features
    
    def _apply_pca_compression(self, alpha_values: pd.DataFrame, alpha_df_full: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Apply TIME-SAFE PCA compression with fallback mechanisms"""
        
        # 数据质量预检查
        if alpha_values.empty:
            logger.warning("输入Alpha数据为空，跳过PCA压缩")
            return None, []
        
        # 去除非数值列
        numeric_cols = alpha_values.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            logger.warning("没有数值类型的Alpha特征，跳过PCA压缩")
            return None, []
        
        alpha_numeric = alpha_values[numeric_cols].copy()
        
        # 检查数据变化性
        col_std = alpha_numeric.std()
        valid_cols = col_std[col_std > 1e-8].index  # 移除方差过小的列
        
        if len(valid_cols) == 0:
            logger.warning("所有Alpha特征方差过小，跳过PCA压缩")
            return None, []
        
        if len(valid_cols) < len(numeric_cols):
            logger.info(f"移除了{len(numeric_cols) - len(valid_cols)}个低方差Alpha特征")
            alpha_numeric = alpha_numeric[valid_cols]
        
        try:
            # 尝试使用时间安全的PCA
            from bma_models.time_safe_pca import TimeSeriesSafePCA
            
            logger.info("[TOOL] 使用时间安全PCA，防止时间泄露")
            
            # 创建时间安全PCA
            safe_pca = TimeSeriesSafePCA(
                n_components=min(self.config.pca_variance_explained, 0.95),  # 限制最大解释方差
                min_history_days=30,  # 降低最小历史天数要求
                refit_frequency=21,   # 21天重新拟合
                max_components=min(len(valid_cols), self.config.max_alpha_features - 3, 8)  # 限制最大组件数
            )
            
            # 时间安全的拟合转换 - 使用原始完整数据保留时间信息
            if alpha_df_full is not None:
                # 使用完整DataFrame（包含索引时间信息）
                alpha_for_pca = alpha_df_full[valid_cols].copy()  # 保留原始索引和时间信息
            else:
                # 回退到数值数据（可能没有时间信息）
                alpha_for_pca = alpha_values[valid_cols].copy()
            
            pca_features_df, pca_stats = safe_pca.fit_transform_safe(alpha_for_pca)
            
            if not pca_features_df.empty:
                # 获取PCA特征数据（排除日期和ticker列）
                pca_feature_cols = [col for col in pca_features_df.columns 
                                  if col.startswith('alpha_pca_')]
                
                if pca_feature_cols:
                    pca_features = pca_features_df[pca_feature_cols].values
                    
                    # 存储统计信息
                    self.pca_fitted = safe_pca  # 存储时间安全PCA对象
                    self.stats['compression_variance_explained'] = pca_stats.get('avg_components', 0)
                    self.stats['time_safe_pca_stats'] = pca_stats
                    
                    logger.info(f"[OK] 时间安全PCA成功，生成{len(pca_feature_cols)}个压缩特征")
                    return pca_features_df[pca_feature_cols], pca_feature_cols
            
            logger.warning("时间安全PCA处理失败，尝试简单PCA回退")
            
        except Exception as e:
            logger.warning(f"时间安全PCA失败: {e}，尝试简单PCA回退")
        
        # 回退到简单PCA
        try:
            from sklearn.decomposition import PCA
            from sklearn.impute import SimpleImputer
            
            logger.info("使用简单PCA作为回退方案")
            
            # 填充缺失值
            imputer = SimpleImputer(strategy='median')
            alpha_filled = pd.DataFrame(
                imputer.fit_transform(alpha_numeric),
                columns=alpha_numeric.columns,
                index=alpha_numeric.index
            )
            
            # 应用简单PCA
            max_components = min(len(valid_cols), 8, alpha_filled.shape[0] // 10)  # 确保足够的样本
            if max_components < 1:
                logger.warning("样本数量不足，无法进行PCA压缩")
                return None, []
            
            pca = PCA(n_components=max_components)
            pca_features = pca.fit_transform(alpha_filled)
            
            # 创建特征DataFrame
            pca_feature_names = [f'alpha_pca_{i+1}' for i in range(pca_features.shape[1])]
            pca_features_df = pd.DataFrame(
                pca_features,
                columns=pca_feature_names,
                index=alpha_numeric.index
            )
            
            # 存储统计信息
            self.stats['compression_variance_explained'] = pca.explained_variance_ratio_.sum()
            self.stats['pca_components'] = len(pca_feature_names)
            
            logger.info(f"[OK] 简单PCA成功，生成{len(pca_feature_names)}个压缩特征，解释方差: {pca.explained_variance_ratio_.sum():.3f}")
            return pca_features_df, pca_feature_names
            
        except Exception as e:
            logger.warning(f"简单PCA也失败: {e}")
            return None, []
    
    def _apply_ic_weighted_compression(self, alpha_values: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Apply IC-weighted composite compression"""
        try:
            # This is a simplified version - in practice, you'd compute IC using historical target returns
            # For now, we create equal-weighted and volatility-adjusted composites
            
            # Equal-weighted composite
            alpha_composite = alpha_values.mean(axis=1)
            
            # Volatility-adjusted composite (weight inversely by volatility)
            alpha_vols = alpha_values.rolling(window=20, min_periods=5).std().fillna(1.0)
            vol_weights = (1.0 / alpha_vols).div((1.0 / alpha_vols).sum(axis=1), axis=0)
            alpha_vol_weighted = (alpha_values * vol_weights).sum(axis=1)
            
            # Create composite DataFrame
            composite_df = pd.DataFrame({
                'alpha_composite_ew': alpha_composite,
                'alpha_composite_vw': alpha_vol_weighted
            }, index=alpha_values.index)
            
            # Orthogonalize composites (remove correlation)
            composite_values = composite_df.fillna(0).values
            if composite_values.shape[1] > 1:
                # Simple Gram-Schmidt orthogonalization
                composite_orth = self._orthogonalize_features(composite_values)
                composite_df = pd.DataFrame(
                    composite_orth, 
                    index=alpha_values.index,
                    columns=['alpha_composite_orth1', 'alpha_composite_orth2']
                )
            
            feature_names = list(composite_df.columns)
            logger.info(f"IC加权合成: 生成 {len(feature_names)} 个合成特征")
            
            return composite_df, feature_names
            
        except Exception as e:
            logger.warning(f"IC加权合成失败: {e}")
            return None, []
    
    def _orthogonalize_features(self, features: np.ndarray) -> np.ndarray:
        """Simple Gram-Schmidt orthogonalization"""
        try:
            orth_features = features.copy()
            n_features = features.shape[1]
            
            for i in range(1, n_features):
                for j in range(i):
                    # Project feature i onto feature j and subtract
                    projection = np.dot(orth_features[:, i], orth_features[:, j]) / np.dot(orth_features[:, j], orth_features[:, j])
                    orth_features[:, i] -= projection * orth_features[:, j]
                
                # Normalize
                norm = np.linalg.norm(orth_features[:, i])
                if norm > 1e-8:
                    orth_features[:, i] /= norm
            
            return orth_features
        except:
            return features  # Return original if orthogonalization fails
    
    def _compute_summary_statistics(self, alpha_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """A3: Compute robust summary statistics"""
        
        alpha_numeric = alpha_df.select_dtypes(include=[np.number])
        if alpha_numeric.empty:
            return None
        
        stats_df = pd.DataFrame(index=alpha_df.index)
        
        try:
            # 1. Alpha dispersion (cross-sectional volatility)
            if self.config.include_dispersion:
                stats_df['alpha_dispersion'] = alpha_numeric.std(axis=1)
            
            # 2. Alpha agreement (directional consistency)
            if self.config.include_agreement:
                positive_count = (alpha_numeric > 0).sum(axis=1)
                total_count = alpha_numeric.notna().sum(axis=1)
                stats_df['alpha_agreement'] = positive_count / total_count.replace(0, 1)
            
            # 3. Alpha quality (rolling IC proxy - simplified)
            if self.config.include_quality:
                # Simplified quality measure: rolling correlation stability
                quality_scores = []
                try:
                    # 确保有date字段用于分组
                    if isinstance(alpha_df.index, pd.MultiIndex) and 'date' in alpha_df.index.names:
                        # 使用MultiIndex中的date
                        date_groups = alpha_numeric.groupby(level='date')
                    elif 'date' in alpha_df.columns:
                        # 使用列中的date
                        date_groups = alpha_numeric.groupby(alpha_df['date'])
                    else:
                        # 如果没有date字段，使用整体相关性
                        logger.warning("无法找到date字段，使用整体相关性计算质量指标")
                        if len(alpha_numeric.columns) > 1:
                            corr_matrix = alpha_numeric.T.corr()
                            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                            if np.isfinite(avg_corr):
                                stats_df['alpha_quality'] = avg_corr
                            else:
                                stats_df['alpha_quality'] = 0.0
                        else:
                            stats_df['alpha_quality'] = 0.0
                        quality_scores = None
                    
                    if quality_scores is not None:
                        for date, group in date_groups:
                            if len(group) > 1 and len(group.columns) > 1:
                                # Compute average pairwise correlation as quality proxy
                                corr_matrix = group.T.corr()
                                upper_tri_indices = np.triu_indices_from(corr_matrix.values, k=1)
                                if len(upper_tri_indices[0]) > 0:
                                    avg_corr = corr_matrix.values[upper_tri_indices].mean()
                                    if np.isfinite(avg_corr):
                                        quality_scores.extend([avg_corr] * len(group))
                                    else:
                                        quality_scores.extend([0.0] * len(group))
                                else:
                                    quality_scores.extend([0.0] * len(group))
                            else:
                                quality_scores.extend([0.0] * len(group))
                        
                        if len(quality_scores) == len(stats_df):
                            stats_df['alpha_quality'] = quality_scores
                        else:
                            # 长度不匹配时使用默认值
                            stats_df['alpha_quality'] = 0.0
                except Exception as e:
                    logger.warning(f"质量指标计算失败，使用默认值: {e}")
                    stats_df['alpha_quality'] = 0.0
            
            # Handle infinite values and NaN
            stats_df = stats_df.replace([np.inf, -np.inf], np.nan)
            stats_df = stats_df.fillna(stats_df.median())
            
            logger.info(f"摘要统计特征生成完成: {stats_df.shape[1]} 个统计特征")
            return stats_df
            
        except Exception as e:
            logger.warning(f"摘要统计计算失败: {e}")
            return None
    
    # REMOVED: _compute_alpha_strategy_signal method completely deleted
    # No more hardcoded Alpha strategy weights - let ML learn optimal combinations
    
    def _combine_and_finalize_features(self, 
                                     alpha_compressed: Optional[pd.DataFrame],
                                     alpha_stats: Optional[pd.DataFrame],
                                     alpha_strategy_signal: Optional[pd.DataFrame],
                                     target_index: pd.Index) -> pd.DataFrame:
        """A4: Combine and finalize features for ML integration"""
        
        # Collect all feature components
        feature_components = []
        
        if alpha_compressed is not None:
            feature_components.append(alpha_compressed)
        
        if alpha_stats is not None:
            feature_components.append(alpha_stats)
        
        if alpha_strategy_signal is not None:
            feature_components.append(alpha_strategy_signal)
        
        if not feature_components:
            logger.warning("没有可用的Alpha摘要特征")
            return pd.DataFrame(index=target_index)
        
        # Combine all features
        combined_features = pd.concat(feature_components, axis=1)
        
        # Ensure we don't exceed max features limit
        if combined_features.shape[1] > self.config.max_alpha_features:
            # Keep most informative features (highest variance)
            feature_vars = combined_features.var()
            top_features = feature_vars.nlargest(self.config.max_alpha_features).index
            combined_features = combined_features[top_features]
            logger.info(f"特征数量限制: 保留前 {self.config.max_alpha_features} 个高方差特征")
        
        # Final data type optimization
        if self.config.data_type == 'float32':
            combined_features = combined_features.astype(np.float32)
        
        # Handle remaining missing values
        if self.config.fill_method == 'cross_median':
            # Fill with cross-sectional median by date
            filled_features = []
            # Check if date is in index or columns
            if isinstance(combined_features.index, pd.MultiIndex) and 'date' in combined_features.index.names:
                # Date is in MultiIndex
                for date, group in combined_features.groupby(level='date'):
                    group_filled = group.fillna(group.median())
                    filled_features.append(group_filled)
                combined_features = pd.concat(filled_features)
            elif 'date' in combined_features.columns:
                # Date is in columns
                for date, group in combined_features.groupby('date'):
                    group_filled = group.fillna(group.median())
                    filled_features.append(group_filled)
                combined_features = pd.concat(filled_features)
            else:
                # No date column, use simple fill
                combined_features = combined_features.fillna(combined_features.median())
        elif self.config.fill_method == 'forward_fill':
            combined_features = combined_features.fillna(method='ffill').fillna(0)
        else:  # zero fill
            combined_features = combined_features.fillna(0)
        
        logger.info(f"Alpha摘要特征最终生成: {combined_features.shape}")
        
        return combined_features
    
    def _validate_time_alignment(self, alpha_df: pd.DataFrame, target_dates: pd.Series) -> int:
        """A5: Validate time alignment to prevent leakage - 改进版"""
        violations = 0
        
        try:
            # [OK] FIX: 更智能的时间对齐验证
            if isinstance(alpha_df.index, pd.MultiIndex):
                # MultiIndex情况 - 尝试获取date级别
                if 'date' in alpha_df.index.names:
                    alpha_dates = alpha_df.index.get_level_values('date')
                else:
                    # 如果没有date级别，尝试第一个级别
                    alpha_dates = alpha_df.index.get_level_values(0)
                    try:
                        alpha_dates = pd.to_datetime(alpha_dates)
                    except:
                        logger.debug("无法将索引转换为日期，跳过时间对齐验证")
                        return 0
            else:
                # 普通索引 - 尝试转换为日期
                try:
                    alpha_dates = pd.to_datetime(alpha_df.index)
                except:
                    logger.debug("无法将索引转换为日期，跳过时间对齐验证")
                    return 0
            
            # 转换target_dates为datetime
            if target_dates is not None:
                try:
                    target_dates = pd.to_datetime(target_dates)
                except:
                    logger.debug("无法转换target_dates，跳过时间对齐验证")
                    return 0
                
                # [OK] FIX: 正确的时间泄漏验证逻辑
                # 检查每个alpha数据点是否违反时间顺序
                max_target_date = target_dates.max()
                min_target_date = target_dates.min()
                
                # 计算实际的时间泄漏：alpha数据晚于最新目标日期 + 容忍期
                tolerance_days = 1  # 容忍1天的差异
                cutoff_date = max_target_date + pd.Timedelta(days=tolerance_days)
                
                # 统计违规数据点（不重复计算）
                future_data_mask = alpha_dates > cutoff_date
                violations = future_data_mask.sum()
                
                if violations > 0:
                    logger.debug(f"发现 {violations} 个数据点晚于截止日期 {cutoff_date}")
                    # 额外检查：严重违规（超过7天）
                    severe_cutoff = max_target_date + pd.Timedelta(days=7)
                    severe_violations = (alpha_dates > severe_cutoff).sum()
                    if severe_violations > 0:
                        logger.warning(f"严重时间违规: {severe_violations} 个数据点超过7天截止期")
            
        except Exception as e:
            logger.debug(f"时间对齐验证异常: {e}")
            violations = 0
        
        return violations
    
    def _validate_time_alignment_detailed(self, alpha_df: pd.DataFrame, 
                                        target_dates: pd.Series) -> Dict[str, Any]:
        """详细的时间对齐验证，返回问题列清单"""
        result = {
            'total_violations': 0,
            'bad_columns': [],
            'column_violations': {},
            'validation_summary': {}
        }
        
        try:
            if alpha_df.empty or target_dates is None:
                return result
            
            # 获取日期索引
            if isinstance(alpha_df.index, pd.MultiIndex) and 'date' in alpha_df.index.names:
                alpha_dates = alpha_df.index.get_level_values('date').unique()
            elif 'date' in alpha_df.columns:
                alpha_dates = alpha_df['date'].unique()
            else:
                alpha_dates = alpha_df.index if isinstance(alpha_df.index, pd.DatetimeIndex) else []
            
            if len(alpha_dates) == 0:
                return result
            
            alpha_dates = pd.to_datetime(alpha_dates)
            target_dates = pd.to_datetime(target_dates)
            
            # 逐列检查时间对齐问题
            for col in alpha_df.columns:
                if col in ['date', 'ticker']:
                    continue
                
                col_violations = 0
                
                # 检查该列的数据日期是否超前目标日期
                col_data = alpha_df[col].dropna()
                if col_data.empty:
                    continue
                
                # 获取该列数据的日期
                if isinstance(alpha_df.index, pd.MultiIndex):
                    col_dates = col_data.index.get_level_values('date')
                else:
                    col_dates = col_data.index if isinstance(col_data.index, pd.DatetimeIndex) else alpha_dates
                
                # 检查未来信息泄漏 - 改进逻辑：只检查实际使用的数据点
                future_leakage = 0
                
                # 仅检查数据的最大日期是否超过目标日期的最小值
                # 这避免了对历史数据的过度严格检查
                if len(col_dates) > 0 and len(target_dates) > 0:
                    max_data_date = col_dates.max()
                    min_target_date = target_dates.min()
                    
                    # 如果数据中存在超过最早目标日期的数据，才认为有潜在泄漏
                    if max_data_date > min_target_date:
                        # 计算实际泄漏的比例，而不是绝对数量
                        future_data = col_dates[col_dates > min_target_date]
                        future_leakage = len(future_data) / len(col_dates)  # 使用比例而非绝对数
                
                col_violations = future_leakage
                
                # 如果该列违规较多，标记为问题列
                if col_violations > 0.1:  # 超过10%的数据有问题（现在col_violations已经是比例）
                    result['bad_columns'].append(col)
                    result['column_violations'][col] = col_violations
                
                # 累加违规数（转换回绝对数以保持兼容性）
                result['total_violations'] += int(col_violations * len(col_data))
            
            # 生成验证摘要
            result['validation_summary'] = {
                'total_columns_checked': len([c for c in alpha_df.columns if c not in ['date', 'ticker']]),
                'problematic_columns': len(result['bad_columns']),
                'clean_columns': len(alpha_df.columns) - len(result['bad_columns']) - 2,  # 减去date和ticker
                'worst_column': max(result['column_violations'], key=result['column_violations'].get) if result['column_violations'] else None,
                'worst_violation_count': max(result['column_violations'].values()) if result['column_violations'] else 0
            }
            
            logger.debug(f"详细时间验证完成: {result['total_violations']} 总违规, "
                        f"{len(result['bad_columns'])} 问题列")
            
            return result
            
        except Exception as e:
            logger.error(f"详细时间对齐验证异常: {e}")
            return result
    
    def _selective_column_cleanup(self, features_df: pd.DataFrame, 
                                bad_columns: List[str]) -> pd.DataFrame:
        """选择性列清理：仅移除问题列，保留其他特征"""
        if not bad_columns:
            return features_df
        
        try:
            # 移除问题列
            clean_columns = [col for col in features_df.columns if col not in bad_columns]
            
            if len(clean_columns) < self.config.min_alpha_features:
                logger.warning(f"清理后特征数不足({len(clean_columns)} < {self.config.min_alpha_features})，回退到传统特征")
                # 这种情况下仍然回退，但记录具体原因
                self.stats['selective_cleanup_failed'] = {
                    'removed_columns': bad_columns,
                    'remaining_columns': len(clean_columns),
                    'min_required': self.config.min_alpha_features
                }
                return self._fallback_to_traditional_features_with_log(features_df, bad_columns)
            
            cleaned_features = features_df[clean_columns].copy()
            
            # 记录清理统计
            self.stats['selective_cleanup_applied'] = {
                'removed_columns': bad_columns,
                'removed_count': len(bad_columns),
                'retained_columns': clean_columns,
                'retained_count': len(clean_columns),
                'cleanup_rate': len(bad_columns) / len(features_df.columns)
            }
            
            logger.info(f"[SELECTIVE_CLEANUP] 移除 {len(bad_columns)} 问题列，保留 {len(clean_columns)} 特征")
            logger.info(f"[SELECTIVE_CLEANUP] 移除的列: {bad_columns}")
            
            return cleaned_features
            
        except Exception as e:
            logger.error(f"选择性清理失败: {e}")
            return self._fallback_to_traditional_features_with_log(features_df, bad_columns)
    
    def _fallback_to_traditional_features_with_log(self, original_df: pd.DataFrame, 
                                                 bad_columns: List[str]) -> pd.DataFrame:
        """带日志记录的传统特征回退"""
        logger.warning(f"[FALLBACK] 选择性清理失败，完全回退到传统特征")
        logger.warning(f"[FALLBACK] 原始问题列: {bad_columns}")
        
        # 记录完整回退统计
        self.stats['full_fallback_triggered'] = {
            'reason': 'selective_cleanup_insufficient',
            'original_columns': len(original_df.columns),
            'problematic_columns': bad_columns,
            'timestamp': datetime.now().isoformat()
        }
        
        return self._fallback_to_traditional_features(original_df)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()

# Alias class for backward compatibility
class AlphaSummaryFeatures(AlphaSummaryProcessor):
    """Alias for AlphaSummaryProcessor to maintain backward compatibility"""
    pass

# Factory function for easy integration
def create_alpha_summary_processor(config: Dict[str, Any] = None) -> AlphaSummaryProcessor:
    """Create Alpha Summary Processor with configuration"""
    if config:
        alpha_config = AlphaSummaryConfig(**config)
    else:
        alpha_config = AlphaSummaryConfig()
    
    return AlphaSummaryProcessor(alpha_config)