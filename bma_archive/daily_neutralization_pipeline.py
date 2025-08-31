#!/usr/bin/env python3
"""
逐日正交化与中性化管线 - 机构级特征工程标准
====================================================
统一特征处理流水线：去极值→标准化→行业/规模/国家中性化→正交→排序
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class NeutralizationConfig:
    """中性化配置"""
    # 去极值配置
    winsorize_method: str = "mad"              # mad/quantile/zscore
    winsorize_factor: float = 3.0              # MAD倍数或Z-score倍数
    quantile_limits: Tuple[float, float] = (0.01, 0.99)  # 分位数限制
    
    # 标准化配置  
    standardization_method: str = "robust"     # robust/standard/none
    center: bool = True                        # 是否去中心化
    scale: bool = True                         # 是否标准化
    target_std: float = 1.0                    # 目标标准差
    std_tolerance: float = 0.05                # 标准差容差
    enforce_std_precision: bool = True         # 强制标准差精度
    
    # 中性化配置
    industry_neutralize: bool = True           # 行业中性化
    size_neutralize: bool = True               # 规模中性化
    country_neutralize: bool = True            # 国家中性化
    market_cap_log: bool = True                # 对数市值
    
    # 正交化配置
    orthogonalize: bool = True                 # 启用正交化
    correlation_threshold: float = 0.85        # 相关性阈值
    orthogonalize_method: str = "qr"           # qr/gram_schmidt/regression
    
    # 排序配置
    final_ranking: bool = True                 # 最终排序
    ranking_method: str = "normal"             # normal/uniform
    
    # 质量控制
    min_samples_per_date: int = 20             # 每日最少样本数
    max_missing_rate: float = 0.5              # 最大缺失率
    handle_outliers_after_neutralize: bool = True  # 中性化后再次去极值

@dataclass
class MarketData:
    """市场数据容器"""
    market_cap: pd.DataFrame = None            # 市值数据
    industry_codes: pd.DataFrame = None        # 行业代码
    country_codes: pd.DataFrame = None         # 国家代码
    trading_volume: pd.DataFrame = None        # 成交量
    list_date: pd.DataFrame = None             # 上市日期

class DailyNeutralizationPipeline:
    """逐日中性化管线"""
    
    def __init__(self, config: NeutralizationConfig = None, 
                 market_data: MarketData = None):
        """初始化中性化管线"""
        self.config = config or NeutralizationConfig()
        self.market_data = market_data or MarketData()
        
        # 缓存标准化器（逐日重新拟合）
        self.daily_scalers = {}
        
        # 统计信息
        self.stats = {
            'dates_processed': 0,
            'factors_processed': 0,
            'outliers_winsorized': 0,
            'neutralization_applied': 0,
            'orthogonalization_applied': 0,
            'quality_issues': 0
        }
        
        logger.info("逐日中性化管线初始化完成")
    
    def process_daily_factors(self, factor_data: pd.DataFrame,
                             date: pd.Timestamp) -> pd.DataFrame:
        """
        处理单日因子数据
        
        Args:
            factor_data: 单日因子数据 (index=tickers, columns=factors)
            date: 日期
            
        Returns:
            处理后的因子数据
        """
        if factor_data.empty:
            return factor_data
        
        # 检查数据质量
        if len(factor_data) < self.config.min_samples_per_date:
            logger.warning(f"日期 {date} 样本数不足: {len(factor_data)}")
            self.stats['quality_issues'] += 1
            return factor_data
        
        processed_data = factor_data.copy()
        
        try:
            # 步骤1: 去极值
            processed_data = self._winsorize_factors(processed_data, date)
            
            # 步骤2: 中性化（🔧 修复：在标准化之前进行）
            # 正确顺序：中性化会改变分布，应该在标准化之前完成
            processed_data = self._neutralize_factors(processed_data, date)
            
            # 步骤3: 标准化（在中性化之后）
            processed_data = self._standardize_factors(processed_data, date)
            
            # 步骤4: 中性化后再次去极值（可选）
            if self.config.handle_outliers_after_neutralize:
                processed_data = self._winsorize_factors(processed_data, date, suffix="_post")
            
            # 步骤5: 正交化
            processed_data = self._orthogonalize_factors(processed_data, date)
            
            # 步骤6: 最终排序
            if self.config.final_ranking:
                processed_data = self._rank_factors(processed_data, date)
            
            self.stats['dates_processed'] += 1
            self.stats['factors_processed'] += processed_data.shape[1]
            
        except Exception as e:
            logger.error(f"日期 {date} 因子处理失败: {e}")
            self.stats['quality_issues'] += 1
            return factor_data
        
        return processed_data
    
    def process_multi_date_factors(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        处理多日因子数据（主接口）
        
        Args:
            factor_data: 多日因子数据 (index=date, columns=MultiIndex[ticker, factor])
                        或 (index=[date, ticker], columns=factors)
        
        Returns:
            处理后的因子数据
        """
        if factor_data.empty:
            return factor_data
        
        # 检测数据格式
        if isinstance(factor_data.index, pd.MultiIndex):
            # 格式: MultiIndex[date, ticker] x factors
            return self._process_multiindex_format(factor_data)
        elif isinstance(factor_data.columns, pd.MultiIndex):
            # 格式: date x MultiIndex[ticker, factor]
            return self._process_wide_format(factor_data)
        else:
            # 假设单日数据
            if len(factor_data.index.unique()) == 1:
                date = factor_data.index[0] if hasattr(factor_data.index[0], 'date') else pd.Timestamp.now()
                return self.process_daily_factors(factor_data, date)
            else:
                logger.error("无法识别因子数据格式")
                return factor_data
    
    def _process_wide_format(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """处理宽格式数据 (date x MultiIndex[ticker, factor])"""
        processed_dates = []
        
        for date in factor_data.index:
            # 提取当日数据并转置为 tickers x factors
            daily_data = factor_data.loc[date].unstack().fillna(method='ffill')
            
            # 处理当日数据
            processed_daily = self.process_daily_factors(daily_data, date)
            
            # 转换回原格式并添加到结果
            processed_wide = processed_daily.stack().to_frame().T
            processed_wide.index = [date]
            processed_dates.append(processed_wide)
        
        if processed_dates:
            return pd.concat(processed_dates, axis=0)
        else:
            return pd.DataFrame()
    
    def _process_multiindex_format(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """处理MultiIndex格式数据 (MultiIndex[date, ticker] x factors)"""
        processed_data = []
        
        for date in factor_data.index.get_level_values(0).unique():
            # 提取当日数据
            daily_data = factor_data.xs(date, level=0)
            
            # 处理当日数据
            processed_daily = self.process_daily_factors(daily_data, date)
            
            # 添加日期索引
            processed_daily.index = pd.MultiIndex.from_product(
                [[date], processed_daily.index], names=['date', 'ticker']
            )
            processed_data.append(processed_daily)
        
        if processed_data:
            return pd.concat(processed_data, axis=0)
        else:
            return pd.DataFrame()
    
    def _winsorize_factors(self, factor_data: pd.DataFrame, 
                          date: pd.Timestamp, suffix: str = "") -> pd.DataFrame:
        """去极值处理"""
        winsorized_data = factor_data.copy()
        
        for factor_name in factor_data.columns:
            factor_values = factor_data[factor_name].dropna()
            
            if len(factor_values) < 10:  # 样本太少跳过
                continue
            
            if self.config.winsorize_method == "mad":
                # MAD方法
                median = factor_values.median()
                mad = np.median(np.abs(factor_values - median))
                lower_bound = median - self.config.winsorize_factor * mad
                upper_bound = median + self.config.winsorize_factor * mad
                
            elif self.config.winsorize_method == "quantile":
                # 分位数方法
                lower_bound = factor_values.quantile(self.config.quantile_limits[0])
                upper_bound = factor_values.quantile(self.config.quantile_limits[1])
                
            elif self.config.winsorize_method == "zscore":
                # Z-score方法
                mean = factor_values.mean()
                std = factor_values.std()
                lower_bound = mean - self.config.winsorize_factor * std
                upper_bound = mean + self.config.winsorize_factor * std
            
            else:
                continue
            
            # 应用Winsorization
            outlier_mask = (factor_data[factor_name] < lower_bound) | (factor_data[factor_name] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            winsorized_data.loc[factor_data[factor_name] < lower_bound, factor_name] = lower_bound
            winsorized_data.loc[factor_data[factor_name] > upper_bound, factor_name] = upper_bound
            
            if outlier_count > 0:
                self.stats['outliers_winsorized'] += outlier_count
        
        return winsorized_data
    
    def _standardize_factors(self, factor_data: pd.DataFrame, 
                           date: pd.Timestamp) -> pd.DataFrame:
        """标准化处理"""
        if self.config.standardization_method == "none":
            return factor_data
        
        standardized_data = factor_data.copy()
        
        for factor_name in factor_data.columns:
            factor_values = factor_data[factor_name].dropna()
            
            if len(factor_values) < 10:
                continue
            
            if self.config.standardization_method == "robust":
                # 使用中位数和MAD标准化
                median = factor_values.median()
                mad = np.median(np.abs(factor_values - median))
                if mad > 0:
                    standardized_values = (factor_data[factor_name] - median) / mad
                    
                    # 精度控制：调整到目标标准差
                    if self.config.enforce_std_precision:
                        current_std = standardized_values.dropna().std()
                        if current_std > 0:
                            adjustment_factor = self.config.target_std / current_std
                            standardized_values = standardized_values * adjustment_factor
                            
                            # 验证标准差精度
                            final_std = standardized_values.dropna().std()
                            if abs(final_std - self.config.target_std) > self.config.std_tolerance:
                                logger.warning(f"因子{factor_name}标准差精度警告: {final_std:.4f} (目标:{self.config.target_std})")
                    
                    standardized_data[factor_name] = standardized_values
                    
            elif self.config.standardization_method == "standard":
                # 使用均值和标准差
                mean = factor_values.mean()
                std = factor_values.std()
                if std > 0:
                    standardized_values = (factor_data[factor_name] - mean) / std
                    
                    # 精度控制：确保标准差为1
                    if self.config.enforce_std_precision:
                        current_std = standardized_values.dropna().std()
                        if abs(current_std - self.config.target_std) > self.config.std_tolerance:
                            # 直接使用sklearn的StandardScaler确保精度
                            scaler = StandardScaler(with_mean=self.config.center, with_std=self.config.scale)
                            valid_mask = ~factor_data[factor_name].isna()
                            if valid_mask.sum() > 0:
                                scaled_values = scaler.fit_transform(factor_data.loc[valid_mask, factor_name].values.reshape(-1, 1))
                                standardized_values = factor_data[factor_name].copy()
                                standardized_values.loc[valid_mask] = scaled_values.flatten()
                    
                    standardized_data[factor_name] = standardized_values
        
        # 验证标准化质量
        if self.config.enforce_std_precision:
            self._validate_standardization_quality(standardized_data, date)
        
        return standardized_data
    
    def _neutralize_factors(self, factor_data: pd.DataFrame, 
                          date: pd.Timestamp) -> pd.DataFrame:
        """中性化处理"""
        neutralized_data = factor_data.copy()
        
        # 构建中性化变量
        neutralization_vars = self._build_neutralization_variables(
            factor_data.index, date
        )
        
        if neutralization_vars.empty:
            logger.warning(f"日期 {date} 无中性化变量，跳过中性化")
            return factor_data
        
        # 对每个因子进行中性化
        for factor_name in factor_data.columns:
            try:
                factor_values = factor_data[factor_name].dropna()
                
                # 对齐中性化变量
                common_tickers = factor_values.index.intersection(neutralization_vars.index)
                
                if len(common_tickers) < 10:
                    continue
                
                y = factor_values.loc[common_tickers]
                X = neutralization_vars.loc[common_tickers]
                
                # 移除常数列
                X = X.loc[:, X.std() > 1e-8]
                
                if X.empty:
                    continue
                
                # 回归中性化
                reg = LinearRegression(fit_intercept=True)
                reg.fit(X, y)
                
                # 获取残差
                predicted = reg.predict(X)
                residuals = y - predicted
                
                # 更新中性化后的值
                neutralized_data.loc[common_tickers, factor_name] = residuals
                
                self.stats['neutralization_applied'] += 1
                
            except Exception as e:
                logger.debug(f"因子 {factor_name} 中性化失败: {e}")
                continue
        
        return neutralized_data
    
    def _build_neutralization_variables(self, tickers: pd.Index, 
                                      date: pd.Timestamp) -> pd.DataFrame:
        """构建中性化变量"""
        neutralization_data = []
        
        # 行业哑变量
        if self.config.industry_neutralize and self.market_data.industry_codes is not None:
            try:
                industry_data = self._get_market_data_for_date(
                    self.market_data.industry_codes, date, tickers
                )
                if not industry_data.empty:
                    # 创建行业哑变量
                    industry_dummies = pd.get_dummies(
                        industry_data.iloc[:, 0], prefix='industry'
                    )
                    industry_dummies.index = industry_data.index
                    neutralization_data.append(industry_dummies)
            except Exception as e:
                logger.debug(f"行业中性化变量构建失败: {e}")
        
        # 规模因子（对数市值）
        if self.config.size_neutralize and self.market_data.market_cap is not None:
            try:
                market_cap_data = self._get_market_data_for_date(
                    self.market_data.market_cap, date, tickers
                )
                if not market_cap_data.empty:
                    log_market_cap = np.log(market_cap_data.iloc[:, 0] + 1)
                    log_market_cap.name = 'log_market_cap'
                    neutralization_data.append(log_market_cap.to_frame())
            except Exception as e:
                logger.debug(f"规模中性化变量构建失败: {e}")
        
        # 国家哑变量
        if self.config.country_neutralize and self.market_data.country_codes is not None:
            try:
                country_data = self._get_market_data_for_date(
                    self.market_data.country_codes, date, tickers
                )
                if not country_data.empty:
                    country_dummies = pd.get_dummies(
                        country_data.iloc[:, 0], prefix='country'
                    )
                    country_dummies.index = country_data.index
                    neutralization_data.append(country_dummies)
            except Exception as e:
                logger.debug(f"国家中性化变量构建失败: {e}")
        
        # 合并所有中性化变量
        if neutralization_data:
            combined_neutralization = pd.concat(neutralization_data, axis=1)
            # 移除常数列和高度相关列
            combined_neutralization = combined_neutralization.loc[:, combined_neutralization.std() > 1e-8]
            return combined_neutralization.fillna(0)
        
        return pd.DataFrame()
    
    def _get_market_data_for_date(self, market_data: pd.DataFrame, 
                                date: pd.Timestamp, 
                                tickers: pd.Index) -> pd.DataFrame:
        """获取特定日期的市场数据"""
        if market_data is None or market_data.empty:
            return pd.DataFrame()
        
        # 处理日期索引
        if isinstance(market_data.index, pd.DatetimeIndex):
            # 选择最接近的日期
            available_dates = market_data.index[market_data.index <= date]
            if available_dates.empty:
                return pd.DataFrame()
            closest_date = available_dates.max()
            date_data = market_data.loc[closest_date:closest_date]
        else:
            # 假设为ticker索引
            date_data = market_data
        
        # 筛选目标tickers
        common_tickers = date_data.columns.intersection(tickers) if hasattr(date_data, 'columns') else tickers
        if not common_tickers.empty:
            return date_data[common_tickers].T
        
        return pd.DataFrame()
    
    def _orthogonalize_factors(self, factor_data: pd.DataFrame, 
                             date: pd.Timestamp) -> pd.DataFrame:
        """因子正交化"""
        if not self.config.orthogonalize or factor_data.shape[1] <= 1:
            return factor_data
        
        # 计算因子间相关性
        correlation_matrix = factor_data.corr()
        
        # 寻找高相关因子对
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > self.config.correlation_threshold:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))
        
        if not high_corr_pairs:
            return factor_data
        
        orthogonalized_data = factor_data.copy()
        
        try:
            if self.config.orthogonalize_method == "qr":
                # QR分解正交化
                Q, R = np.linalg.qr(factor_data.fillna(0).values)
                orthogonal_factors = pd.DataFrame(
                    Q, index=factor_data.index, 
                    columns=[f"orth_{i}" for i in range(Q.shape[1])]
                )
                return orthogonal_factors
                
            elif self.config.orthogonalize_method == "regression":
                # 逐步回归正交化
                for factor1, factor2, corr_val in high_corr_pairs:
                    # 用factor1回归factor2，保留残差
                    valid_mask = ~(factor_data[factor1].isna() | factor_data[factor2].isna())
                    
                    if valid_mask.sum() < 10:
                        continue
                    
                    X = factor_data.loc[valid_mask, [factor1]]
                    y = factor_data.loc[valid_mask, factor2]
                    
                    reg = LinearRegression(fit_intercept=True)
                    reg.fit(X, y)
                    
                    predicted = reg.predict(X)
                    residuals = y - predicted
                    
                    # 替换factor2为残差
                    orthogonalized_data.loc[valid_mask, factor2] = residuals
                
                self.stats['orthogonalization_applied'] += len(high_corr_pairs)
                
        except Exception as e:
            logger.debug(f"正交化失败: {e}")
            return factor_data
        
        return orthogonalized_data
    
    def _rank_factors(self, factor_data: pd.DataFrame, 
                     date: pd.Timestamp) -> pd.DataFrame:
        """因子排序"""
        ranked_data = factor_data.copy()
        
        for factor_name in factor_data.columns:
            factor_values = factor_data[factor_name].dropna()
            
            if len(factor_values) < 10:
                continue
            
            if self.config.ranking_method == "normal":
                # 正态化排序
                ranks = stats.rankdata(factor_values, method='average')
                normal_scores = stats.norm.ppf(ranks / (len(ranks) + 1))
                ranked_data.loc[factor_values.index, factor_name] = normal_scores
                
            elif self.config.ranking_method == "uniform":
                # 均匀分布排序
                ranks = factor_values.rank(method='average')
                uniform_scores = (ranks - 1) / (len(ranks) - 1)
                ranked_data.loc[factor_values.index, factor_name] = uniform_scores
        
        return ranked_data
    
    def _validate_standardization_quality(self, standardized_data: pd.DataFrame, 
                                        date: pd.Timestamp) -> None:
        """验证标准化质量"""
        quality_issues = []
        
        for factor_name in standardized_data.columns:
            factor_values = standardized_data[factor_name].dropna()
            
            if len(factor_values) < 5:
                continue
            
            # 检查标准差
            current_std = factor_values.std()
            if abs(current_std - self.config.target_std) > self.config.std_tolerance:
                quality_issues.append({
                    'factor': factor_name,
                    'issue': 'std_deviation',
                    'current_std': current_std,
                    'target_std': self.config.target_std,
                    'tolerance': self.config.std_tolerance
                })
            
            # 检查均值（应该接近0）
            current_mean = abs(factor_values.mean())
            if current_mean > 0.1:  # 均值偏离0超过0.1
                quality_issues.append({
                    'factor': factor_name,
                    'issue': 'mean_deviation', 
                    'current_mean': factor_values.mean(),
                    'abs_mean': current_mean
                })
                
            # 检查是否有异常值（绝对值超过5）
            extreme_values = (abs(factor_values) > 5).sum()
            if extreme_values > 0:
                quality_issues.append({
                    'factor': factor_name,
                    'issue': 'extreme_values',
                    'extreme_count': extreme_values,
                    'total_count': len(factor_values)
                })
        
        if quality_issues:
            logger.warning(f"日期 {date} 标准化质量问题: {len(quality_issues)}项")
            for issue in quality_issues[:3]:  # 只显示前3个问题
                if issue['issue'] == 'std_deviation':
                    logger.warning(f"  {issue['factor']}: 标准差 {issue['current_std']:.4f} (目标: {issue['target_std']})")
                elif issue['issue'] == 'mean_deviation':
                    logger.warning(f"  {issue['factor']}: 均值偏离 {issue['current_mean']:.4f}")
                elif issue['issue'] == 'extreme_values':
                    logger.warning(f"  {issue['factor']}: 极值 {issue['extreme_count']}/{issue['total_count']}")
            
            # 更新统计
            self.stats['standardization_quality_issues'] = self.stats.get('standardization_quality_issues', 0) + len(quality_issues)
        else:
            self.stats['standardization_quality_passed'] = self.stats.get('standardization_quality_passed', 0) + len(standardized_data.columns)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """获取管线统计信息"""
        return {
            'processing_stats': self.stats,
            'config': self.config.__dict__,
            'market_data_availability': {
                'market_cap': self.market_data.market_cap is not None,
                'industry_codes': self.market_data.industry_codes is not None,
                'country_codes': self.market_data.country_codes is not None,
                'trading_volume': self.market_data.trading_volume is not None
            }
        }

# 全局中性化管线实例
def create_neutralization_pipeline(config: NeutralizationConfig = None,
                                 market_data: MarketData = None) -> DailyNeutralizationPipeline:
    """创建中性化管线实例"""
    return DailyNeutralizationPipeline(config, market_data)

if __name__ == "__main__":
    # 测试中性化管线
    pipeline = create_neutralization_pipeline()
    
    # 模拟数据
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    factors = ['momentum', 'value', 'quality']
    
    # 创建模拟因子数据
    # np.random.seed removed
    factor_data_list = []
    
    for date in dates:
        daily_factor_data = pd.DataFrame(
            np.zeros(len(tickers), len(factors)),
            index=tickers,
            columns=factors
        )
        processed_daily = pipeline.process_daily_factors(daily_factor_data, date)
        print(f"日期 {date.date()} - 处理前形状: {daily_factor_data.shape}, 处理后形状: {processed_daily.shape}")
    
    print("\n=== 中性化管线统计 ===")
    stats = pipeline.get_pipeline_stats()
    for key, value in stats['processing_stats'].items():
        print(f"{key}: {value}")