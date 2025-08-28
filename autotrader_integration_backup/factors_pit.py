"""
PIT强实证因子 + 中性化系统
============================

Point-in-Time基本面因子 + 行业/规模/Beta中性化
先叠加到现有因子，再替换弱项
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

class PITFundamentalFactors:
    """Point-in-Time基本面因子计算器"""
    
    def __init__(self):
        self.last_update = {}
        self.factor_cache = {}
        
    def align_financials_to_trading_dates(self, financial_data: pd.DataFrame, 
                                        trading_dates: pd.DatetimeIndex,
                                        announcement_lag_days: int = 90) -> pd.DataFrame:
        """
        将财报数据按披露日对齐到交易日
        
        Args:
            financial_data: 财报数据 (MultiIndex: (report_date, symbol))
            trading_dates: 交易日序列
            announcement_lag_days: 披露滞后天数
            
        Returns:
            DataFrame: PIT对齐的财报数据 (MultiIndex: (trading_date, symbol))
        """
        pit_data = {}
        
        for symbol in financial_data.index.get_level_values(1).unique():
            symbol_data = financial_data.xs(symbol, level=1)
            symbol_pit = pd.DataFrame(index=trading_dates, columns=financial_data.columns)
            
            for trading_date in trading_dates:
                # 找到这个交易日之前最近的可用财报
                available_reports = symbol_data[
                    symbol_data.index <= trading_date - timedelta(days=announcement_lag_days)
                ]
                
                if not available_reports.empty:
                    latest_report = available_reports.iloc[-1]
                    symbol_pit.loc[trading_date] = latest_report
                    
            pit_data[symbol] = symbol_pit
            
        # 重建MultiIndex
        result_data = []
        for symbol, data in pit_data.items():
            for date in data.index:
                if not data.loc[date].isna().all():
                    result_data.append({
                        'date': date,
                        'symbol': symbol,
                        **data.loc[date].to_dict()
                    })
                    
        if not result_data:
            return pd.DataFrame()
            
        result_df = pd.DataFrame(result_data)
        return result_df.set_index(['date', 'symbol'])
    
    def compute_value_factors(self, pit_financials: pd.DataFrame, 
                            market_data: pd.DataFrame) -> pd.DataFrame:
        """计算价值因子"""
        factors = pd.DataFrame(index=pit_financials.index)
        
        try:
            # 需要的字段
            required_fields = ['total_equity', 'total_assets', 'revenue', 'net_income', 
                             'operating_cash_flow', 'free_cash_flow']
            
            missing_fields = [f for f in required_fields if f not in pit_financials.columns]
            if missing_fields:
                logger.warning(f"缺少财务字段: {missing_fields}")
            
            # 获取市值数据
            market_cap = market_data['market_cap'] if 'market_cap' in market_data.columns else None
            share_price = market_data['adj_close'] if 'adj_close' in market_data.columns else None
            
            if market_cap is not None:
                # Book-to-Price
                if 'total_equity' in pit_financials.columns:
                    book_value = pit_financials['total_equity']
                    factors['book_to_price'] = book_value / market_cap
                    
                # Earnings-to-Price  
                if 'net_income' in pit_financials.columns:
                    factors['earnings_to_price'] = pit_financials['net_income'] / market_cap
                    
                # Sales-to-Price
                if 'revenue' in pit_financials.columns:
                    factors['sales_to_price'] = pit_financials['revenue'] / market_cap
                    
                # Cash Flow-to-Price
                if 'operating_cash_flow' in pit_financials.columns:
                    factors['cf_to_price'] = pit_financials['operating_cash_flow'] / market_cap
                    
                # Free Cash Flow-to-Price
                if 'free_cash_flow' in pit_financials.columns:
                    factors['fcf_to_price'] = pit_financials['free_cash_flow'] / market_cap
                    
            # Enterprise Value ratios (如果有企业价值数据)
            if 'enterprise_value' in market_data.columns:
                ev = market_data['enterprise_value']
                if 'ebitda' in pit_financials.columns:
                    factors['ev_to_ebitda'] = ev / pit_financials['ebitda']
                if 'revenue' in pit_financials.columns:
                    factors['ev_to_sales'] = ev / pit_financials['revenue']
                    
        except Exception as e:
            logger.error(f"价值因子计算失败: {e}")
            
        return factors
    
    def compute_quality_factors(self, pit_financials: pd.DataFrame) -> pd.DataFrame:
        """计算质量因子"""
        factors = pd.DataFrame(index=pit_financials.index)
        
        try:
            # ROE (Return on Equity)
            if 'net_income' in pit_financials.columns and 'total_equity' in pit_financials.columns:
                factors['roe'] = pit_financials['net_income'] / pit_financials['total_equity']
                
            # ROA (Return on Assets)
            if 'net_income' in pit_financials.columns and 'total_assets' in pit_financials.columns:
                factors['roa'] = pit_financials['net_income'] / pit_financials['total_assets']
                
            # ROIC (Return on Invested Capital)
            if all(col in pit_financials.columns for col in ['net_income', 'total_debt', 'total_equity']):
                invested_capital = pit_financials['total_debt'] + pit_financials['total_equity']
                factors['roic'] = pit_financials['net_income'] / invested_capital
                
            # Gross Margin
            if 'gross_profit' in pit_financials.columns and 'revenue' in pit_financials.columns:
                factors['gross_margin'] = pit_financials['gross_profit'] / pit_financials['revenue']
            elif 'cost_of_goods_sold' in pit_financials.columns and 'revenue' in pit_financials.columns:
                factors['gross_margin'] = (pit_financials['revenue'] - pit_financials['cost_of_goods_sold']) / pit_financials['revenue']
                
            # Operating Margin  
            if 'operating_income' in pit_financials.columns and 'revenue' in pit_financials.columns:
                factors['operating_margin'] = pit_financials['operating_income'] / pit_financials['revenue']
                
            # Asset Turnover
            if 'revenue' in pit_financials.columns and 'total_assets' in pit_financials.columns:
                factors['asset_turnover'] = pit_financials['revenue'] / pit_financials['total_assets']
                
            # Debt-to-Equity
            if 'total_debt' in pit_financials.columns and 'total_equity' in pit_financials.columns:
                factors['debt_to_equity'] = pit_financials['total_debt'] / pit_financials['total_equity']
                
            # Interest Coverage
            if 'operating_income' in pit_financials.columns and 'interest_expense' in pit_financials.columns:
                factors['interest_coverage'] = pit_financials['operating_income'] / pit_financials['interest_expense'].replace(0, np.nan)
                
        except Exception as e:
            logger.error(f"质量因子计算失败: {e}")
            
        return factors
    
    def compute_growth_factors(self, pit_financials: pd.DataFrame, 
                             lookback_periods: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """计算成长因子"""
        factors = pd.DataFrame(index=pit_financials.index)
        
        try:
            # 按股票分组计算增长率
            growth_metrics = ['revenue', 'net_income', 'operating_income', 'total_assets', 'total_equity']
            
            for metric in growth_metrics:
                if metric not in pit_financials.columns:
                    continue
                    
                for period in lookback_periods:
                    col_name = f'{metric}_growth_{period}y'
                    
                    # 按symbol分组计算同比增长
                    for symbol in pit_financials.index.get_level_values(1).unique():
                        symbol_data = pit_financials.xs(symbol, level=1)[metric]
                        
                        if len(symbol_data) > period:
                            growth_rate = symbol_data.pct_change(periods=period * 4)  # 假设季度数据
                            factors.loc[(factors.index.get_level_values(1) == symbol), col_name] = growth_rate
                            
        except Exception as e:
            logger.error(f"成长因子计算失败: {e}")
            
        return factors
    
    def compute_accruals_factors(self, pit_financials: pd.DataFrame) -> pd.DataFrame:
        """计算应计项目因子"""
        factors = pd.DataFrame(index=pit_financials.index)
        
        try:
            # Total Accruals = (Income - Cash Flow) / Total Assets
            if all(col in pit_financials.columns for col in ['net_income', 'operating_cash_flow', 'total_assets']):
                accruals = (pit_financials['net_income'] - pit_financials['operating_cash_flow']) / pit_financials['total_assets']
                factors['total_accruals'] = -accruals  # 负号：低应计项目好
                
            # Working Capital Accruals
            if all(col in pit_financials.columns for col in ['current_assets', 'current_liabilities', 'total_assets']):
                wc_accruals = (pit_financials['current_assets'] - pit_financials['current_liabilities']) / pit_financials['total_assets']
                # 计算同比变化
                for symbol in factors.index.get_level_values(1).unique():
                    symbol_mask = factors.index.get_level_values(1) == symbol
                    symbol_wc = wc_accruals[symbol_mask]
                    if len(symbol_wc) > 4:  # 至少1年数据
                        wc_change = symbol_wc.diff(4)  # 年度变化
                        factors.loc[symbol_mask, 'wc_accruals'] = -wc_change  # 负号：低变化好
                        
        except Exception as e:
            logger.error(f"应计项目因子计算失败: {e}")
            
        return factors
    
    def compute_investment_factors(self, pit_financials: pd.DataFrame) -> pd.DataFrame:
        """计算投资因子"""
        factors = pd.DataFrame(index=pit_financials.index)
        
        try:
            # Asset Growth (负面因子)
            if 'total_assets' in pit_financials.columns:
                for symbol in factors.index.get_level_values(1).unique():
                    symbol_mask = factors.index.get_level_values(1) == symbol
                    symbol_assets = pit_financials.loc[symbol_mask, 'total_assets']
                    if len(symbol_assets) > 4:
                        asset_growth = symbol_assets.pct_change(4)  # 年度增长率
                        factors.loc[symbol_mask, 'asset_growth'] = -asset_growth  # 负号：低增长好
                        
            # Capex-to-Assets
            if 'capital_expenditure' in pit_financials.columns and 'total_assets' in pit_financials.columns:
                factors['capex_to_assets'] = pit_financials['capital_expenditure'] / pit_financials['total_assets']
                
            # R&D-to-Assets  
            if 'rd_expense' in pit_financials.columns and 'total_assets' in pit_financials.columns:
                factors['rd_to_assets'] = pit_financials['rd_expense'] / pit_financials['total_assets']
                
        except Exception as e:
            logger.error(f"投资因子计算失败: {e}")
            
        return factors

class FactorNeutralizer:
    """因子中性化器"""
    
    def __init__(self):
        self.industry_mapping = {}
        self.neutralization_stats = {}
        
    def get_industry_dummies(self, industry_df: pd.DataFrame) -> pd.DataFrame:
        """生成行业哑变量"""
        return pd.get_dummies(industry_df, prefix='industry')
    
    def cross_sectional_neutralize(self, factor_panel: pd.DataFrame,
                                 industry_df: pd.DataFrame,
                                 log_mcap: pd.Series,
                                 beta: Optional[pd.Series] = None,
                                 min_stocks_per_date: int = 50) -> pd.DataFrame:
        """
        横截面中性化
        
        Args:
            factor_panel: 因子面板 (MultiIndex: (date, symbol))
            industry_df: 行业分类 (MultiIndex: (date, symbol))  
            log_mcap: 对数市值 (MultiIndex: (date, symbol))
            beta: Beta值 (MultiIndex: (date, symbol))
            min_stocks_per_date: 每日最小股票数
            
        Returns:
            DataFrame: 中性化后的因子
        """
        neutralized_factors = pd.DataFrame(index=factor_panel.index, columns=factor_panel.columns)
        
        # 获取所有交易日
        dates = factor_panel.index.get_level_values(0).unique().sort_values()
        
        for date in dates:
            try:
                # 提取当日数据
                date_mask = factor_panel.index.get_level_values(0) == date
                factors_t = factor_panel[date_mask]
                
                if len(factors_t) < min_stocks_per_date:
                    continue
                    
                # 获取当日的控制变量
                log_mcap_t = log_mcap[log_mcap.index.get_level_values(0) == date]
                industry_t = industry_df[industry_df.index.get_level_values(0) == date]
                
                # 对齐数据
                common_symbols = (factors_t.index.get_level_values(1)
                                .intersection(log_mcap_t.index.get_level_values(1))
                                .intersection(industry_t.index.get_level_values(1)))
                
                if len(common_symbols) < min_stocks_per_date:
                    continue
                    
                # 重建当日数据
                factors_aligned = factors_t[factors_t.index.get_level_values(1).isin(common_symbols)]
                mcap_aligned = log_mcap_t[log_mcap_t.index.get_level_values(1).isin(common_symbols)]
                industry_aligned = industry_t[industry_t.index.get_level_values(1).isin(common_symbols)]
                
                # 生成行业哑变量
                industry_dummies = self.get_industry_dummies(industry_aligned)
                
                # 构建回归矩阵 X = [常数项, 行业哑变量, log(市值), beta]
                X_components = [np.ones(len(factors_aligned))]  # 常数项
                
                # 行业哑变量
                if not industry_dummies.empty:
                    X_components.append(industry_dummies.values)
                    
                # 对数市值
                if not mcap_aligned.empty:
                    X_components.append(mcap_aligned.values.reshape(-1, 1))
                    
                # Beta (如果提供)
                if beta is not None:
                    beta_t = beta[beta.index.get_level_values(0) == date]
                    beta_aligned = beta_t[beta_t.index.get_level_values(1).isin(common_symbols)]
                    if not beta_aligned.empty:
                        X_components.append(beta_aligned.values.reshape(-1, 1))
                
                # 合并回归矩阵
                X = np.concatenate(X_components, axis=1)
                
                # 对每个因子做回归并取残差
                for factor_name in factors_aligned.columns:
                    y = factors_aligned[factor_name].values
                    
                    # 移除NaN
                    valid_mask = ~(pd.isna(y) | pd.isna(X).any(axis=1))
                    if valid_mask.sum() < 20:  # 需要足够样本
                        continue
                        
                    X_clean = X[valid_mask]
                    y_clean = y[valid_mask]
                    
                    try:
                        # OLS回归
                        reg = LinearRegression(fit_intercept=False)  # 已包含常数项
                        reg.fit(X_clean, y_clean)
                        
                        # 计算残差
                        y_pred = reg.predict(X_clean)
                        residuals = y_clean - y_pred
                        
                        # 保存中性化结果
                        valid_indices = factors_aligned.index[valid_mask]
                        neutralized_factors.loc[valid_indices, factor_name] = residuals
                        
                    except np.linalg.LinAlgError:
                        # 处理奇异矩阵
                        logger.warning(f"日期{date}因子{factor_name}中性化失败：奇异矩阵")
                        continue
                        
            except Exception as e:
                logger.warning(f"日期{date}中性化失败: {e}")
                continue
                
        # 记录中性化统计
        self.neutralization_stats[datetime.now()] = {
            'processed_dates': len(dates),
            'successful_dates': (~neutralized_factors.isna().all(axis=1)).sum(),
            'avg_stocks_per_date': factor_panel.groupby(level=0).size().mean()
        }
        
        return neutralized_factors
    
    def winsorize_and_standardize(self, factor_panel: pd.DataFrame,
                                winsorize_limits: Tuple[float, float] = (0.01, 0.99),
                                standardize_method: str = 'zscore') -> pd.DataFrame:
        """因子预处理：截尾和标准化"""
        processed_factors = factor_panel.copy()
        
        for factor_name in factor_panel.columns:
            factor_series = factor_panel[factor_name]
            
            # 1. 截尾处理
            lower_q, upper_q = winsorize_limits
            lower_bound = factor_series.quantile(lower_q)
            upper_bound = factor_series.quantile(upper_q)
            factor_series = factor_series.clip(lower=lower_bound, upper=upper_bound)
            
            # 2. 标准化
            if standardize_method == 'zscore':
                # 滚动Z-score标准化
                dates = factor_series.index.get_level_values(0).unique().sort_values()
                
                for date in dates:
                    date_mask = factor_series.index.get_level_values(0) == date
                    date_values = factor_series[date_mask]
                    
                    if len(date_values) > 10:  # 需要足够样本
                        mean_val = date_values.mean()
                        std_val = date_values.std()
                        
                        if std_val > 0:
                            standardized = (date_values - mean_val) / std_val
                            processed_factors.loc[date_mask, factor_name] = standardized
                            
            elif standardize_method == 'rank':
                # 分位数排序
                dates = factor_series.index.get_level_values(0).unique().sort_values()
                
                for date in dates:
                    date_mask = factor_series.index.get_level_values(0) == date
                    date_values = factor_series[date_mask]
                    
                    if len(date_values) > 5:
                        ranks = date_values.rank(method='average', na_option='keep')
                        normalized_ranks = (ranks - 1) / (len(date_values) - 1)  # 0-1范围
                        processed_factors.loc[date_mask, factor_name] = normalized_ranks
                        
        return processed_factors

class EnhancedFactorPipeline:
    """增强因子产线 - 集成PIT因子 + 中性化"""
    
    def __init__(self, config: Optional[Dict] = None):
        default_config = {
            'announcement_lag_days': 90,
            'min_stocks_per_date': 30,
            'winsorize_limits': (0.01, 0.99),
            'standardize_method': 'zscore',
            'enable_neutralization': True,
            'growth_lookback_periods': [1, 2, 3]
        }
        
        self.config = {**default_config, **(config or {})}
        self.pit_factors = PITFundamentalFactors()
        self.neutralizer = FactorNeutralizer()
        
    def compute_all_pit_factors(self, financial_data: pd.DataFrame,
                               market_data: pd.DataFrame,
                               trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """计算所有PIT因子"""
        logger.info("开始计算PIT基本面因子")
        
        # PIT对齐
        pit_financials = self.pit_factors.align_financials_to_trading_dates(
            financial_data, trading_dates, self.config['announcement_lag_days']
        )
        
        if pit_financials.empty:
            logger.warning("PIT对齐后无数据")
            return pd.DataFrame()
        
        all_factors = []
        
        # 价值因子
        try:
            value_factors = self.pit_factors.compute_value_factors(pit_financials, market_data)
            if not value_factors.empty:
                all_factors.append(value_factors)
                logger.info(f"价值因子: {len(value_factors.columns)}个")
        except Exception as e:
            logger.warning(f"价值因子计算失败: {e}")
            
        # 质量因子
        try:
            quality_factors = self.pit_factors.compute_quality_factors(pit_financials)
            if not quality_factors.empty:
                all_factors.append(quality_factors)
                logger.info(f"质量因子: {len(quality_factors.columns)}个")
        except Exception as e:
            logger.warning(f"质量因子计算失败: {e}")
            
        # 成长因子
        try:
            growth_factors = self.pit_factors.compute_growth_factors(
                pit_financials, self.config['growth_lookback_periods']
            )
            if not growth_factors.empty:
                all_factors.append(growth_factors)
                logger.info(f"成长因子: {len(growth_factors.columns)}个")
        except Exception as e:
            logger.warning(f"成长因子计算失败: {e}")
            
        # 应计项目因子
        try:
            accruals_factors = self.pit_factors.compute_accruals_factors(pit_financials)
            if not accruals_factors.empty:
                all_factors.append(accruals_factors)
                logger.info(f"应计项目因子: {len(accruals_factors.columns)}个")
        except Exception as e:
            logger.warning(f"应计项目因子计算失败: {e}")
            
        # 投资因子
        try:
            investment_factors = self.pit_factors.compute_investment_factors(pit_financials)
            if not investment_factors.empty:
                all_factors.append(investment_factors)
                logger.info(f"投资因子: {len(investment_factors.columns)}个")
        except Exception as e:
            logger.warning(f"投资因子计算失败: {e}")
            
        if not all_factors:
            return pd.DataFrame()
            
        # 合并所有因子
        combined_factors = pd.concat(all_factors, axis=1)
        logger.info(f"PIT因子计算完成，共{len(combined_factors.columns)}个因子")
        
        return combined_factors
    
    def neutralize_factors(self, factors: pd.DataFrame,
                          industry_data: pd.DataFrame,
                          market_data: pd.DataFrame) -> pd.DataFrame:
        """因子中性化"""
        if not self.config['enable_neutralization']:
            return factors
            
        logger.info("开始因子中性化")
        
        # 准备控制变量
        log_mcap = None
        beta = None
        
        if 'market_cap' in market_data.columns:
            log_mcap = np.log(market_data['market_cap'].replace(0, np.nan))
        elif 'adj_close' in market_data.columns and 'shares_outstanding' in market_data.columns:
            market_cap = market_data['adj_close'] * market_data['shares_outstanding']
            log_mcap = np.log(market_cap.replace(0, np.nan))
            
        if 'beta' in market_data.columns:
            beta = market_data['beta']
            
        if log_mcap is None:
            logger.warning("无市值数据，跳过市值中性化")
            return factors
            
        # 预处理：截尾和标准化
        processed_factors = self.neutralizer.winsorize_and_standardize(
            factors, 
            self.config['winsorize_limits'],
            self.config['standardize_method']
        )
        
        # 中性化
        neutralized_factors = self.neutralizer.cross_sectional_neutralize(
            processed_factors, industry_data, log_mcap, beta,
            self.config['min_stocks_per_date']
        )
        
        logger.info("因子中性化完成")
        return neutralized_factors
    
    def integrate_with_existing_factors(self, pit_factors: pd.DataFrame,
                                      existing_factors: pd.DataFrame,
                                      integration_method: str = 'concat') -> pd.DataFrame:
        """与现有因子集成"""
        if existing_factors.empty:
            return pit_factors
            
        if integration_method == 'concat':
            # 简单合并
            return pd.concat([existing_factors, pit_factors], axis=1)
            
        elif integration_method == 'weighted':
            # 加权合并 (需要权重配置)
            # 这里使用简单平均作为示例
            common_index = existing_factors.index.intersection(pit_factors.index)
            
            if len(common_index) == 0:
                return pd.concat([existing_factors, pit_factors], axis=1)
                
            # 标准化后平均
            existing_std = existing_factors.loc[common_index].std()
            pit_std = pit_factors.loc[common_index].std()
            
            weight_existing = 0.5  # 可配置
            weight_pit = 0.5
            
            combined = pd.DataFrame(index=common_index)
            
            # 现有因子加权
            for col in existing_factors.columns:
                if col in existing_factors.loc[common_index].columns:
                    combined[f'existing_{col}'] = (existing_factors.loc[common_index, col] * weight_existing)
                    
            # PIT因子加权
            for col in pit_factors.columns:
                if col in pit_factors.loc[common_index].columns:
                    combined[f'pit_{col}'] = (pit_factors.loc[common_index, col] * weight_pit)
                    
            return combined
            
        else:
            raise ValueError(f"未知的集成方法: {integration_method}")

# 工厂函数
def create_enhanced_factor_pipeline(config: Optional[Dict] = None) -> EnhancedFactorPipeline:
    """创建增强因子产线"""
    return EnhancedFactorPipeline(config)

# 使用示例  
def example_integration_with_existing_system():
    """与现有系统集成示例"""
    
    # 配置
    config = {
        'announcement_lag_days': 90,
        'enable_neutralization': True,
        'min_stocks_per_date': 50,
        'standardize_method': 'zscore'
    }
    
    # 创建产线
    pipeline = create_enhanced_factor_pipeline(config)
    
    # 步骤1：计算PIT因子
    # pit_factors = pipeline.compute_all_pit_factors(financial_data, market_data, trading_dates)
    
    # 步骤2：中性化
    # neutralized_factors = pipeline.neutralize_factors(pit_factors, industry_data, market_data)
    
    # 步骤3：与现有因子集成
    # final_factors = pipeline.integrate_with_existing_factors(neutralized_factors, existing_factors)
    
    # 步骤4：输入到现有的信号生成系统
    # 现有的信号生成无需修改，只需要把final_factors作为输入即可
    
    return pipeline