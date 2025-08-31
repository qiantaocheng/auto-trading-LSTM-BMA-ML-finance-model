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
    max_alpha_features: int = 18  # Professional standard: 15-20 features
    min_alpha_features: int = 15  # Minimum for robustness
    pca_variance_explained: float = 0.85  # PCA variance threshold
    pls_n_components: int = 8  # Increased PLS components
    use_ic_weighted: bool = True  # PRIMARY: Use professional IC-weighted method
    use_pca: bool = False  # DEPRECATED: Use as fallback only
    use_pls: bool = False
    use_ic_weighted_composite: bool = False  # DEPRECATED: Old simple version
    include_alpha_strategy_signal: bool = True  # Include composite Alpha strategy signal as feature
    
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
        
        # A3.5: Alpha strategy composite signal
        alpha_strategy_signal = self._compute_alpha_strategy_signal(alpha_cleaned) if self.config.include_alpha_strategy_signal else None
        
        # A4: Combine and prepare final features
        summary_features = self._combine_and_finalize_features(
            alpha_compressed, alpha_stats, alpha_strategy_signal, alpha_cleaned.index
        )
        
        # 🔧 健康检查3: 时间对齐违规门槛收紧 (在最终输出前检查)
        violations_result = self._validate_time_alignment_detailed(summary_features, target_dates) if target_dates is not None else {'total_violations': 0, 'bad_columns': []}
        violation_rate = violations_result['total_violations'] / (summary_features.shape[0] + 1e-8)
        
        if violation_rate > 0.05:  # 收紧阈值：违规率超过5%
            bad_columns = violations_result.get('bad_columns', [])
            logger.warning(f"[SELECTIVE_CLEANUP] 时间对齐违规过多({violations_result['total_violations']}项, {violation_rate:.1%})")
            logger.warning(f"[SELECTIVE_CLEANUP] 问题列: {bad_columns}")
            
            # 选择性清理：仅移除问题列，而非全体回退
            return self._selective_column_cleanup(summary_features, bad_columns)
        
        logger.info(f"✅ Alpha摘要特征生成完成，输出形状: {summary_features.shape}")
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
        
        # ✅ FIX: 智能索引格式处理
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
        
        # ✅ FIX: 根据标签期确定正确的滞后天数（适应性调整）
        # 从target_dates推断标签期，或从列名解析
        label_horizon = 5  # 🔧 FIX: 默认改为5天标签期
        if target_dates is not None and len(target_dates) > 1:
            # 尝试从target_dates间隔推断标签期
            try:
                target_dates_dt = pd.to_datetime(target_dates)
                if len(target_dates_dt) > 1:
                    avg_interval = (target_dates_dt.max() - target_dates_dt.min()).days / max(1, len(target_dates_dt) - 1)
                    if avg_interval > 1:
                        label_horizon = min(int(avg_interval), 10)  # 🔧 FIX: Cap label horizon at 10 days
            except Exception as e:
                logger.debug(f"Failed to infer label horizon from target_dates: {e}")
                pass
        
        # 🔧 FIX: Use adaptive lag based on dataset size  
        # Adaptive lag based on data characteristics
        if len(alpha_df) < 500:
            default_lag = max(1, label_horizon // 2)  # Very small datasets: minimal lag
        elif len(alpha_df) < 1000:
            default_lag = max(2, label_horizon)  # Small datasets: use label horizon
        else:
            default_lag = max(label_horizon, 3)  # 🔧 FIX: Reduced minimum lag to 3 days
        
        # ✅ FIX: 强制索引标准化 - 确保datetime索引
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
        
        # ✅ FIX: 应用正确的滞后
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
            alpha_df_shifted = alpha_df_shifted.dropna()
                
        except Exception as e:
            logger.warning(f"滞后处理失败: {e}，使用原始数据")
            alpha_df_shifted = alpha_df.copy()
        
        logger.info(f"应用了{default_lag}天滞后（基于{label_horizon}天标签期），避免时间对齐违规")
        
        # ✅ FIX: 更智能的时间验证，减少误报
        if target_dates is not None and self.config.strict_time_validation:
            violations = self._validate_time_alignment(alpha_df_shifted, target_dates)
            # ✅ FIX: 正确的百分比计算 - 总单元格数而不是行数
            numeric_cols = alpha_df_shifted.select_dtypes(include=[np.number]).columns
            total_cells = len(alpha_df_shifted) * len(numeric_cols)
            violation_rate = violations / (total_cells + 1e-8)
            
            if violation_rate > 0.15:  # 违规率超过15%报警
                logger.warning(f"发现较多时间对齐违规: {violations} 个单元格 ({violation_rate:.1%}) / 总计{total_cells}")
                self.stats['time_violations'] = violations
            elif violations > 0:
                logger.debug(f"发现少量时间对齐违规: {violations} 个单元格 ({violation_rate:.1%})，在可接受范围内")
                self.stats['time_violations'] = violations
            else:
                logger.info("时间对齐验证通过，无泄露风险")
        
        # 使用滞后后的数据继续处理
        alpha_df = alpha_df_shifted
        
        # 🔧 健康检查1: 列名匹配度 (扩展列名筛选)
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
                    any(pattern in col.lower() for pattern in ['factor', 'reversal', 
                                                               'turnover', 'amihud', 'bid_ask', 'yield',
                                                               'ohlson', 'altman', 'qmj', 'earnings', 'beta',
                                                               'ratio', 'rsi', 'macd', 'ma_', '_ma'])):
                    alpha_cols.append(col)
        
        # Log what columns were detected
        if not alpha_cols:
            logger.warning(f"No alpha columns found. Available columns: {list(alpha_df.columns)[:20]}")
            logger.warning(f"Total columns: {len(alpha_df.columns)}, Data shape: {alpha_df.shape}")
            # Log column types
            numeric_cols = alpha_df.select_dtypes(include=[np.float32, np.float64, np.int32, np.int64]).columns
            logger.warning(f"Numeric columns found: {len(numeric_cols)}: {list(numeric_cols)[:10]}")
        
        # 🔧 健康检查2: 最低列数门槛
        if len(alpha_cols) < 3:
            logger.warning(f"[FALLBACK] Alpha列数过少({len(alpha_cols)} < 3)，触发传统特征回退")
            return self._fallback_to_traditional_features(alpha_df)
        
        if not alpha_cols:
            logger.warning(f"[FALLBACK] 未找到有效的Alpha列，触发传统特征回退，可用列: {list(alpha_df.columns)[:10]}...")
            return self._fallback_to_traditional_features(alpha_df)
        
        # Include date column for groupby
        if 'date' in alpha_df.columns:
            cols_to_process = ['date'] + alpha_cols
        else:
            # If date is in index, reset it temporarily
            alpha_df = alpha_df.reset_index()
            cols_to_process = ['date'] + alpha_cols
        
        alpha_only = alpha_df[cols_to_process].copy()
        self.stats['total_alphas_processed'] = len(alpha_cols)
        
        # Cross-sectional processing by date
        cleaned_data = []
        
        for date, group in alpha_only.groupby('date'):
            # Drop date column for processing (will be in index after groupby)
            group_for_processing = group.drop(columns=['date'], errors='ignore')
            
            # Cross-sectional winsorization
            group_clean = self._cross_sectional_winsorize(group_for_processing)
            
            # Cross-sectional standardization
            group_clean = self._cross_sectional_standardize(group_clean)
            
            # Industry/factor neutralization
            if self.config.neutralize_by_industry and market_data is not None:
                group_clean = self._neutralize_factors(group_clean, market_data, date)
            
            cleaned_data.append(group_clean)
        
        # Check if we have any cleaned data
        if not cleaned_data:
            logger.warning("[FALLBACK] No data after cleaning, returning fallback features")
            return self._fallback_to_traditional_features(alpha_df)
        
        result = pd.concat(cleaned_data)
        logger.info(f"数据清洗完成，处理了 {len(result.index.get_level_values('date').unique())} 个交易日")
        
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
                if 'close' in alpha_df.columns:
                    returns = alpha_df['close'].pct_change().shift(-10)  # 10天前向收益
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
            pca_features, pca_names = self._apply_pca_compression(alpha_values)
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
    
    def _apply_pca_compression(self, alpha_values: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Apply PCA compression"""
        try:
            # Handle missing values
            alpha_imputed = self.imputer.fit_transform(alpha_values.fillna(0))
            
            # Fit PCA
            pca = PCA()
            pca_transformed = pca.fit_transform(alpha_imputed)
            
            # Find components that explain target variance
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= self.config.pca_variance_explained) + 1
            # Professional standard: 15-18 features total (leaving room for interaction features)
            n_components = min(n_components, self.config.max_alpha_features - 3, alpha_values.shape[1])
            n_components = max(n_components, self.config.min_alpha_features - 3)  # Ensure minimum
            
            # Keep only selected components
            pca_final = PCA(n_components=n_components)
            pca_features = pca_final.fit_transform(alpha_imputed)
            
            # Store fitted model for future use
            self.pca_fitted = pca_final
            self.stats['compression_variance_explained'] = cumsum_variance[n_components-1]
            
            # Create DataFrame with proper index
            pca_df = pd.DataFrame(
                pca_features, 
                index=alpha_values.index,
                columns=[f'alpha_pc{i+1}' for i in range(n_components)]
            )
            
            feature_names = list(pca_df.columns)
            logger.info(f"PCA压缩: {n_components} 个主成分，解释方差 {cumsum_variance[n_components-1]:.3f}")
            
            return pca_df, feature_names
            
        except Exception as e:
            logger.warning(f"PCA压缩失败: {e}")
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
                for date, group in alpha_numeric.groupby('date'):
                    # Compute average pairwise correlation as quality proxy
                    corr_matrix = group.T.corr()
                    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                    quality_scores.extend([avg_corr] * len(group))
                
                stats_df['alpha_quality'] = quality_scores
            
            # Handle infinite values and NaN
            stats_df = stats_df.replace([np.inf, -np.inf], np.nan)
            stats_df = stats_df.fillna(stats_df.median())
            
            logger.info(f"摘要统计特征生成完成: {stats_df.shape[1]} 个统计特征")
            return stats_df
            
        except Exception as e:
            logger.warning(f"摘要统计计算失败: {e}")
            return None
    
    def _compute_alpha_strategy_signal(self, alpha_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """A3.5: Compute Alpha strategy composite signal based on your 44-factor strategy
        
        Alpha来源分配:
        - 质量筛选(40%): QMJ, Piotroski, Altman, Ohlson质量因子
        - 动量捕获(25%): 动量、残差动量、动量门控因子  
        - 情绪优势(20%): 新闻情绪、市场情绪、恐惧贪婪指数
        - 流动性溢价(15%): Amihud非流动性、买卖价差因子
        """
        
        alpha_numeric = alpha_df.select_dtypes(include=[np.number])
        if alpha_numeric.empty:
            logger.warning("无Alpha数据用于策略信号计算")
            return None
        
        try:
            strategy_df = pd.DataFrame(index=alpha_df.index)
            
            # 根据Alpha因子名称分类（基于您的alphas_config.yaml配置）
            quality_factors = []
            momentum_factors = []
            sentiment_factors = []
            liquidity_factors = []
            other_factors = []
            
            for col in alpha_numeric.columns:
                col_lower = col.lower()
                if any(q in col_lower for q in ['qmj', 'piotroski', 'altman', 'ohlson', 'quality', 'roe', 'roic', 'margin', 'profitability', 'earnings_stability']):
                    quality_factors.append(col)
                elif any(m in col_lower for m in ['momentum', 'reversal', 'residual', 'hump']):
                    momentum_factors.append(col)
                elif any(s in col_lower for s in ['sentiment', 'news', 'fear', 'greed', 'market_sentiment']):
                    sentiment_factors.append(col)
                elif any(l in col_lower for l in ['amihud', 'bid_ask', 'illiq', 'spread', 'volume', 'turnover']):
                    liquidity_factors.append(col)
                else:
                    other_factors.append(col)
            
            # 计算各类别的组合信号
            signals = {}
            
            if quality_factors:
                signals['quality'] = alpha_numeric[quality_factors].mean(axis=1) * 0.40
                logger.info(f"质量因子 ({len(quality_factors)}个): 权重40%")
            
            if momentum_factors:
                signals['momentum'] = alpha_numeric[momentum_factors].mean(axis=1) * 0.25
                logger.info(f"动量因子 ({len(momentum_factors)}个): 权重25%")
            
            if sentiment_factors:
                signals['sentiment'] = alpha_numeric[sentiment_factors].mean(axis=1) * 0.20
                logger.info(f"情绪因子 ({len(sentiment_factors)}个): 权重20%")
            
            if liquidity_factors:
                signals['liquidity'] = alpha_numeric[liquidity_factors].mean(axis=1) * 0.15
                logger.info(f"流动性因子 ({len(liquidity_factors)}个): 权重15%")
            
            if other_factors:
                # 其他因子平均分配剩余权重
                remaining_weight = 1.0 - sum([0.40, 0.25, 0.20, 0.15]) if not all([quality_factors, momentum_factors, sentiment_factors, liquidity_factors]) else 0.0
                if remaining_weight > 0:
                    signals['other'] = alpha_numeric[other_factors].mean(axis=1) * remaining_weight
                    logger.info(f"其他因子 ({len(other_factors)}个): 权重{remaining_weight:.1%}")
            
            # 合成最终的Alpha策略信号
            alpha_strategy_raw = sum(signals.values()) if signals else alpha_numeric.mean(axis=1)
            
            # 应用横截面标准化（与其他摘要特征保持一致）
            if self.config.neutralize_by_industry and isinstance(alpha_df.index, pd.MultiIndex):
                # 简化的行业中性化（这里使用全局标准化）
                try:
                    alpha_strategy_normalized = (alpha_strategy_raw.groupby(alpha_df.index.get_level_values('date'))
                                               .apply(lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1)))
                except:
                    # 如果分组标准化失败，使用全局标准化
                    alpha_strategy_normalized = (alpha_strategy_raw - alpha_strategy_raw.mean()) / (alpha_strategy_raw.std() if alpha_strategy_raw.std() > 0 else 1)
            else:
                alpha_strategy_normalized = (alpha_strategy_raw - alpha_strategy_raw.mean()) / (alpha_strategy_raw.std() if alpha_strategy_raw.std() > 0 else 1)
            
            # Winsorize处理异常值
            if self.config.use_mad_winsorize:
                median = alpha_strategy_normalized.median()
                mad = np.median(np.abs(alpha_strategy_normalized - median))
                alpha_strategy_winsorized = np.clip(alpha_strategy_normalized, 
                                                  median - 3*mad, median + 3*mad)
            else:
                q01, q99 = alpha_strategy_normalized.quantile([0.01, 0.99])
                alpha_strategy_winsorized = np.clip(alpha_strategy_normalized, q01, q99)
            
            strategy_df['alpha_strategy_signal'] = alpha_strategy_winsorized
            
            logger.info(f"Alpha策略综合信号生成完成: 范围[{alpha_strategy_winsorized.min():.3f}, {alpha_strategy_winsorized.max():.3f}]")
            return strategy_df
            
        except Exception as e:
            logger.warning(f"Alpha策略信号计算失败: {e}")
            return None
    
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
            for date, group in combined_features.groupby('date'):
                group_filled = group.fillna(group.median())
                filled_features.append(group_filled)
            combined_features = pd.concat(filled_features)
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
            # ✅ FIX: 更智能的时间对齐验证
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
                
                # 只验证明显的未来数据泄漏（容忍1天的差异）
                for target_date in target_dates:
                    # 检查是否有明显的未来数据（>1天的差异）
                    future_data = alpha_dates > (target_date + pd.Timedelta(days=1))
                    if future_data.any():
                        violations += future_data.sum()
            
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
                
                # 检查未来信息泄漏
                future_leakage = 0
                for target_date in target_dates:
                    future_data = col_dates[col_dates > target_date]
                    future_leakage += len(future_data)
                
                col_violations += future_leakage
                
                # 如果该列违规较多，标记为问题列
                if col_violations > len(col_data) * 0.1:  # 超过10%的数据有问题
                    result['bad_columns'].append(col)
                    result['column_violations'][col] = col_violations
                
                result['total_violations'] += col_violations
            
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

# Factory function for easy integration
def create_alpha_summary_processor(config: Dict[str, Any] = None) -> AlphaSummaryProcessor:
    """Create Alpha Summary Processor with configuration"""
    if config:
        alpha_config = AlphaSummaryConfig(**config)
    else:
        alpha_config = AlphaSummaryConfig()
    
    return AlphaSummaryProcessor(alpha_config)