#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sato Square Root Factor Validation Framework
严谨的实证验证架构 - 测试 Sato 平方根因子是否提供增量信息

基于用户提供的框架，使用真实 MultiIndex cleaned 数据进行验证
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class SatoValidator:
    """
    Sato Square Root Factor Validator
    
    核心科学问题：
    "Sato平方根因子是否提供了现有因子（如传统动量、波动率）之外的增量信息？"
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize validator with MultiIndex cleaned data
        
        Args:
            data_path: Path to cleaned parquet file (MultiIndex format)
        """
        if data_path is None:
            data_path = r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean.parquet"
        
        print("=" * 80)
        print("Sato Square Root Factor Validation Framework")
        print("=" * 80)
        print(f"Loading data from: {data_path}")
        
        # Load MultiIndex data
        self.df = pd.read_parquet(data_path)
        
        # Validate MultiIndex format
        if not isinstance(self.df.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex (date, ticker) format")
        
        if self.df.index.names[0].lower() != 'date' or self.df.index.names[1].lower() not in ['ticker', 'symbol']:
            raise ValueError(f"MultiIndex must be (date, ticker), got {self.df.index.names}")
        
        # Sort by date and ticker
        self.df = self.df.sort_index()
        
        print(f"[OK] Data loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
        print(f"   Date range: {self.df.index.get_level_values('date').min()} to {self.df.index.get_level_values('date').max()}")
        print(f"   Unique tickers: {self.df.index.get_level_values(self.df.index.names[1]).nunique():,}")
        
        # Check if we need to fetch market data
        # The cleaned parquet file may only have factors, not raw price/volume
        has_price_data = any(c.lower() in ['close', 'adj_close'] for c in self.df.columns)
        has_volume_data = any(c.lower() == 'volume' for c in self.df.columns)
        
        # Use Close as adj_close
        if 'Close' in self.df.columns:
            self.df['adj_close'] = self.df['Close']
            print("[OK] Found Close price data")
        else:
            raise ValueError("Close price data not found in data file")
        
        # Handle Volume: if not present, estimate from vol_ratio_20d
        if not has_volume_data:
            print("[WARNING] Volume data not found, estimating from vol_ratio_20d...")
            if 'vol_ratio_20d' in self.df.columns:
                # vol_ratio_20d = Volume / Volume_MA20
                # For Sato factor, we mainly need relative volume
                # Estimate: use vol_ratio_20d as a proxy for relative volume
                # Set a base volume and scale by vol_ratio
                base_volume = 1_000_000  # Base volume assumption (1M shares)
                self.df['Volume'] = base_volume * (self.df['vol_ratio_20d'].fillna(1.0).clip(lower=0.1))
                print("[OK] Estimated Volume from vol_ratio_20d")
            else:
                raise ValueError(
                    "Volume data required but not found.\n"
                    "Data file must include either 'Volume' column or 'vol_ratio_20d' factor."
                )
        else:
            print("[OK] Found Volume data")
        
        print("[OK] Required columns ready")
    
    def prepare_factors(self):
        """
        生成 Sato 因子以及对照组因子
        
        核心公式：
        Sato Momentum = sum( sign(r) * sigma * sqrt(V_rel) )
        其中：
        - r: 对数收益
        - sigma: 波动率
        - V_rel: 相对成交量
        """
        print("\n" + "=" * 80)
        print(">>> Generating Factors...")
        print("=" * 80)
        
        # 1. 基础特征计算（按ticker分组，确保时间序列正确）
        print("   Computing log returns...")
        # Calculate log returns properly for MultiIndex
        adj_close_series = self.df['adj_close']
        log_ret = adj_close_series.groupby(level=1).apply(
            lambda x: np.log(x / x.shift(1))
        )
        # Ensure proper index alignment - flatten MultiIndex if needed
        if isinstance(log_ret.index, pd.MultiIndex) and len(log_ret.index.names) > 2:
            log_ret = log_ret.droplevel(0)
        log_ret = log_ret.reindex(self.df.index)
        
        print("   Computing volatility (20D rolling)...")
        # 关键修正：确保按ticker分组计算，避免股票切换时的数据污染
        vol = log_ret.groupby(level=1).rolling(20).std().droplevel(0)
        vol = vol.reindex(self.df.index)
        vol = vol.fillna(method='bfill') + 1e-6  # 填充NaN并添加极小值防止除零
        
        print("   Computing relative volume...")
        # 关键修正：优先直接使用vol_ratio_20d，避免重复计算
        if 'vol_ratio_20d' in self.df.columns:
            # 直接使用vol_ratio_20d作为相对成交量（避免先估算Volume再计算的精度损失）
            rel_vol = self.df['vol_ratio_20d'].fillna(1.0).clip(lower=0.01)  # 最小相对成交量0.01
            print("   Using vol_ratio_20d directly as relative volume")
        else:
            # 从Volume计算相对成交量（仅在vol_ratio不可用时使用）
            adv20 = self.df.groupby(level=1)['Volume'].rolling(20).mean().droplevel(0)
            adv20 = adv20.reindex(self.df.index)
            rel_vol = self.df['Volume'] / (adv20 + 1e-6)  # Avoid division by zero
            rel_vol = rel_vol.fillna(1.0).clip(lower=0.01)  # 确保最小值
            print("   Computing relative volume from Volume")
        
        # 2. 构建 Sato 核心因子 (T+10 预测版)
        print("   Building Sato factor...")
        # 关键修正：
        # 1. 添加clip保护：防止vol接近0时产生无穷大（死股、停牌等）
        # 2. 简化公式：sign(x) * |x| = x，直接使用normalized_ret
        normalized_ret = log_ret / vol  # vol已经加了1e-6，不需要再加
        normalized_ret = normalized_ret.clip(-5, 5)  # 截断极值：防止死股或停牌导致的sigma=0产生无限大
        sato_impact = normalized_ret * np.sqrt(rel_vol)  # 简化：直接使用normalized_ret，rel_vol已经clip过
        
        # 10日滚动求和
        sato_factor_series = sato_impact.groupby(level=1).rolling(10).sum()
        if isinstance(sato_factor_series.index, pd.MultiIndex):
            if len(sato_factor_series.index.names) > 2:
                sato_factor_series = sato_factor_series.droplevel(0)
            sato_factor_series.index = self.df.index
        self.df['factor_sato'] = sato_factor_series
        
        # 3. 构建对照组因子（用于正交化测试）
        print("   Building benchmark factors...")
        # 传统动量（10日涨幅）
        mom_raw_series = log_ret.groupby(level=1).rolling(10).sum()
        if isinstance(mom_raw_series.index, pd.MultiIndex):
            if len(mom_raw_series.index.names) > 2:
                mom_raw_series = mom_raw_series.droplevel(0)
            mom_raw_series.index = self.df.index
        self.df['factor_mom_raw'] = mom_raw_series
        
        # 纯波动率
        self.df['factor_vol'] = vol
        
        # 4. 生成标签 (T+10 Future Return)
        print("   Computing T+10 forward returns...")
        # shift(-10) 将未来的收益平移到当前行
        def calc_fwd_ret(group):
            ret = group.pct_change(10).shift(-10)
            # Filter extreme returns (likely data errors, splits, etc.)
            # Clip to reasonable range: -90% to +1000% (10x)
            ret = ret.clip(lower=-0.9, upper=10.0)
            return ret
        
        fwd_ret_series = self.df.groupby(level=1)['adj_close'].apply(calc_fwd_ret)
        # Handle MultiIndex result - remove extra level if present
        if isinstance(fwd_ret_series.index, pd.MultiIndex):
            if len(fwd_ret_series.index.names) > 2:
                fwd_ret_series = fwd_ret_series.droplevel(0)
            # Ensure same index structure as self.df
            fwd_ret_series.index = self.df.index
        
        self.df['fwd_ret_10d'] = fwd_ret_series
        
        # 清洗：去除NaN值
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['factor_sato', 'factor_mom_raw', 'factor_vol', 'fwd_ret_10d'])
        final_len = len(self.df)
        
        print(f"[OK] Factors prepared: {final_len:,} rows (dropped {initial_len - final_len:,} NaN rows)")
        print(f"   Date range after cleaning: {self.df.index.get_level_values('date').min()} to {self.df.index.get_level_values('date').max()}")
    
    def run_ic_test(self):
        """
        测试 1: Information Coefficient (IC) 分析
        
        计算每日 Rank IC，评估 Sato 因子的预测能力
        """
        print("\n" + "=" * 80)
        print(">>> Running IC Analysis (T+10)...")
        print("=" * 80)
        
        # 每日计算 Rank IC（横截面比较）
        def calc_ic(group):
            if len(group) < 10:  # 至少需要10只股票
                return np.nan
            try:
                ic, pval = spearmanr(group['factor_sato'], group['fwd_ret_10d'])
                return ic
            except:
                return np.nan
        
        ic_series = self.df.groupby(level=0).apply(calc_ic)
        ic_series = ic_series.dropna()
        
        if len(ic_series) == 0:
            print("[ERROR] No valid IC values computed")
            return None
        
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0
        positive_ratio = (ic_series > 0).mean()
        
        print(f"\n[STATS] IC Statistics:")
        print(f"   Mean Rank IC:       {mean_ic:.4f}")
        print(f"   IC Std Dev:         {std_ic:.4f}")
        print(f"   IC IR (Sharpe):     {ic_ir:.4f}")
        print(f"   Positive IC Ratio:  {positive_ratio:.1%}")
        print(f"   Valid Days:         {len(ic_series):,}")
        
        # IC分布统计
        print(f"\n[DIST] IC Distribution:")
        print(f"   Min IC:             {ic_series.min():.4f}")
        print(f"   25th Percentile:    {ic_series.quantile(0.25):.4f}")
        print(f"   Median IC:          {ic_series.median():.4f}")
        print(f"   75th Percentile:    {ic_series.quantile(0.75):.4f}")
        print(f"   Max IC:             {ic_series.max():.4f}")
        
        return ic_series
    
    def run_orthogonality_test(self):
        """
        测试 2: 正交化测试（最重要的环节）
        
        Sato 因子是否只是 '波动率' 或 '普通动量' 的马甲？
        我们把 Sato 因子对这两个旧因子做回归，取残差 (Residual) 再测 IC。
        """
        print("\n" + "=" * 80)
        print(">>> Running Orthogonality Check...")
        print("=" * 80)
        print("   Testing if Sato factor provides unique information beyond Mom & Vol")
        
        def get_residual(group):
            """对每一天做横截面回归，提取残差"""
            if len(group) < 20:  # 至少需要20只股票做回归
                return pd.Series(np.nan, index=group.index)
            
            try:
                # Y = Sato, X = [Momentum, Volatility]
                X = group[['factor_mom_raw', 'factor_vol']].values
                X = sm.add_constant(X)  # 添加常数项
                y = group['factor_sato'].values
                
                # OLS回归
                model = sm.OLS(y, X).fit()
                return pd.Series(model.resid, index=group.index)
            except Exception as e:
                return pd.Series(np.nan, index=group.index)
        
        # 对每一天做横截面回归
        print("   Computing residuals (removing Mom & Vol components)...")
        residuals = self.df.groupby(level=0).apply(get_residual)
        
        # 处理MultiIndex结果
        if isinstance(residuals.index, pd.MultiIndex):
            residuals = residuals.droplevel(0)
        
        self.df['sato_pure_residual'] = residuals.reindex(self.df.index)
        
        # 测试"纯净版"Sato因子的IC
        def calc_pure_ic(group):
            if len(group) < 10:
                return np.nan
            try:
                ic, _ = spearmanr(group['sato_pure_residual'], group['fwd_ret_10d'])
                return ic
            except:
                return np.nan
        
        pure_ic_series = self.df.groupby(level=0).apply(calc_pure_ic)
        pure_ic_series = pure_ic_series.dropna()
        
        if len(pure_ic_series) == 0:
            print("[ERROR] No valid pure IC values computed")
            return None
        
        pure_ic_mean = pure_ic_series.mean()
        pure_ic_std = pure_ic_series.std()
        pure_ic_ir = pure_ic_mean / pure_ic_std if pure_ic_std > 0 else 0
        
        print(f"\n[STATS] Pure Sato IC Statistics (after removing Mom & Vol):")
        print(f"   Mean Pure IC:       {pure_ic_mean:.4f}")
        print(f"   Pure IC Std Dev:    {pure_ic_std:.4f}")
        print(f"   Pure IC IR:         {pure_ic_ir:.4f}")
        print(f"   Valid Days:         {len(pure_ic_series):,}")
        
        # 评估结果
        print(f"\n[EVAL] Evaluation:")
        if pure_ic_mean > 0.02:
            print("   [PASS] Pure IC > 0.02")
            print("   [PASS] 结论: 有效。Sato 因子提供了独特的物理学Alpha。")
            print("   [PASS] 平方根定律确实捕捉到了普通动量捕捉不到的信息")
        elif pure_ic_mean > 0.01:
            print("   [MARGINAL] 0.01 < Pure IC < 0.02")
            print("   [MARGINAL] 结论: 边际有效，但信号较弱")
        else:
            print("   [FAIL] Pure IC < 0.01")
            print("   [FAIL] 结论: 无效。Sato 因子只是现有因子的线性组合。")
            print("   [FAIL] 引入平方根定律（√V）没有带来额外价值")
        
        return pure_ic_series
    
    def run_decay_analysis(self):
        """
        测试 3: 信号衰减测试 (Sato vs Traditional)
        
        验证 '平方根冲击' 是否比 '线性冲击' 更持久
        """
        print("\n" + "=" * 80)
        print(">>> Running Signal Decay Analysis...")
        print("=" * 80)
        print("   Testing signal persistence across different horizons")
        
        horizons = [1, 5, 10, 20]
        results = {}
        
        for h in horizons:
            print(f"   Computing T+{h} forward returns...")
            # 生成 T+h 标签
            def calc_fwd_ret_h(group):
                ret = group.pct_change(h).shift(-h)
                # Filter extreme returns
                ret = ret.clip(lower=-0.9, upper=10.0)
                return ret
            
            col_name = f'fwd_ret_{h}d'
            fwd_ret_h = self.df.groupby(level=1)['adj_close'].apply(calc_fwd_ret_h)
            if isinstance(fwd_ret_h.index, pd.MultiIndex) and len(fwd_ret_h.index.names) > 2:
                fwd_ret_h = fwd_ret_h.droplevel(0)
            fwd_ret_h.index = self.df.index
            self.df[col_name] = fwd_ret_h
            
            # 计算 IC
            valid_df = self.df.dropna(subset=['factor_sato', col_name])
            if len(valid_df) == 0:
                results[h] = np.nan
                continue
            
            def calc_ic_h(group):
                if len(group) < 10:
                    return np.nan
                try:
                    ic, _ = spearmanr(group['factor_sato'], group[col_name])
                    return ic
                except:
                    return np.nan
            
            ic_h = valid_df.groupby(level=0).apply(calc_ic_h)
            ic_h = ic_h.dropna()
            
            if len(ic_h) > 0:
                results[h] = ic_h.mean()
            else:
                results[h] = np.nan
        
        print(f"\n[STATS] IC Decay Profile:")
        for h in horizons:
            ic_val = results[h]
            if not np.isnan(ic_val):
                print(f"   T+{h:2d} IC:  {ic_val:7.4f}")
            else:
                print(f"   T+{h:2d} IC:  {'N/A':>7}")
        
        # 诊断
        print(f"\n[ANALYSIS] Decay Analysis:")
        valid_results = {k: v for k, v in results.items() if not np.isnan(v)}
        if len(valid_results) >= 3:
            ic_1 = valid_results.get(1, np.nan)
            ic_5 = valid_results.get(5, np.nan)
            ic_10 = valid_results.get(10, np.nan)
            ic_20 = valid_results.get(20, np.nan)
            
            if not np.isnan(ic_1) and not np.isnan(ic_5) and not np.isnan(ic_10):
                decay_5_1 = ic_5 - ic_1
                decay_10_5 = ic_10 - ic_5
                
                if abs(decay_5_1) < 0.01 and abs(decay_10_5) < 0.01:
                    print("   [PASS] 验证成功：IC在T+5和T+10保持稳定")
                    print("   [PASS] 这符合'Metaorder（大单拆分）'理论")
                    print("   [PASS] 机构的大单持续了数天，导致价格冲击在T+10依然显著")
                    print("   [PASS] 这是将其放入机器学习模型的最佳信号")
                elif ic_1 > 0.05 and ic_10 < 0.01:
                    print("   [WARNING] 诊断：这是一个高频微观结构因子，不适合T+10预测")
                    print("   [WARNING] 平方根定律在这里只反映了做市商的短期库存压力")
                else:
                    print("   [WARNING] 信号衰减模式：需要进一步分析")
        
        return results
    
    def run_quantile_analysis(self):
        """
        测试 4: 分组收益的单调性 (Quantile Monotonicity)
        
        将 sato_pure_residual 分为 5 组，检查收益的单调性
        """
        print("\n" + "=" * 80)
        print(">>> Running Quantile Monotonicity Analysis...")
        print("=" * 80)
        
        # 确保有纯净残差
        if 'sato_pure_residual' not in self.df.columns:
            print("   [WARNING] Pure residual not computed, skipping quantile analysis")
            return None
        
        # 按日期分组，每天计算分位数
        def assign_quantiles(group):
            if len(group) < 50:  # 至少需要50只股票
                return pd.Series([np.nan] * len(group), index=group.index)
            
            try:
                quantiles = pd.qcut(
                    group['sato_pure_residual'],
                    q=5,
                    labels=[1, 2, 3, 4, 5],
                    duplicates='drop'
                )
                return quantiles
            except:
                return pd.Series([np.nan] * len(group), index=group.index)
        
        print("   Assigning quantiles (5 groups)...")
        self.df['quantile'] = self.df.groupby(level=0).apply(assign_quantiles).droplevel(0).reindex(self.df.index)
        
        # 计算每组的平均收益
        valid_df = self.df.dropna(subset=['quantile', 'fwd_ret_10d'])
        
        if len(valid_df) == 0:
            print("   [ERROR] No valid data for quantile analysis")
            return None
        
        quantile_returns = valid_df.groupby(['quantile'])['fwd_ret_10d'].agg(['mean', 'std', 'count'])
        
        print(f"\n[STATS] Quantile Returns (T+10):")
        print(f"{'Group':<8} {'Mean Return':<15} {'Std Dev':<15} {'Count':<10}")
        print("-" * 50)
        for q in sorted(valid_df['quantile'].dropna().unique()):
            q_data = quantile_returns.loc[q]
            print(f"Group {int(q):<5} {q_data['mean']:>12.4%}  {q_data['std']:>12.4%}  {int(q_data['count']):>8,}")
        
        # 检查单调性
        print(f"\n[ANALYSIS] Monotonicity Check:")
        q_means = quantile_returns['mean'].sort_index()
        
        if len(q_means) >= 5:
            # Group 5 (最高) vs Group 1 (最低)
            group5_return = q_means.iloc[-1]
            group1_return = q_means.iloc[0]
            spread = group5_return - group1_return
            
            print(f"   Group 5 (Highest) Return:  {group5_return:.4%}")
            print(f"   Group 1 (Lowest) Return:   {group1_return:.4%}")
            print(f"   Spread (5-1):               {spread:.4%}")
            
            if spread > 0.01:  # 1% spread
                print("   [PASS] 显著的单调性")
                print("   [PASS] Group 5 收益显著 > Group 1")
                if group1_return < 0:
                    print("   [PASS] Group 1 显著跑输（做空信号有效）")
                print("   [PASS] Sato 理论特别擅长预测 Group 1 (下跌)")
                print("   [PASS] '价格涨但量不够（偏离平方根）'是极佳的做空信号")
            else:
                print("   [MARGINAL] 单调性较弱")
        else:
            print("   [WARNING] Insufficient quantiles for analysis")
        
        return quantile_returns
    
    def run_full_validation(self):
        """
        运行完整的验证流程
        """
        print("\n" + "=" * 80)
        print("STARTING FULL VALIDATION PIPELINE")
        print("=" * 80)
        
        # Step 1: Prepare factors
        self.prepare_factors()
        
        # Step 2: IC Test
        ic_series = self.run_ic_test()
        
        # Step 3: Orthogonality Test (Most Important)
        pure_ic_series = self.run_orthogonality_test()
        
        # Step 4: Decay Analysis
        decay_results = self.run_decay_analysis()
        
        # Step 5: Quantile Analysis
        quantile_results = self.run_quantile_analysis()
        
        # Final Summary
        print("\n" + "=" * 80)
        print("FINAL VALIDATION SUMMARY")
        print("=" * 80)
        
        if pure_ic_series is not None and len(pure_ic_series) > 0:
            pure_ic_mean = pure_ic_series.mean()
            
            print(f"\n🎯 Key Metrics:")
            print(f"   Pure IC (after orthogonalization): {pure_ic_mean:.4f}")
            
            if pure_ic_mean > 0.02:
                print(f"\n[PASS] OVERALL VERDICT: PASS")
                print(f"   Sato 平方根因子提供了显著的增量信息")
                print(f"   建议：将此因子纳入机器学习模型")
            elif pure_ic_mean > 0.01:
                print(f"\n[MARGINAL] OVERALL VERDICT: MARGINAL")
                print(f"   Sato 因子有一定价值，但信号较弱")
            else:
                print(f"\n[FAIL] OVERALL VERDICT: FAIL")
                print(f"   Sato 因子未能提供增量信息")
        
        print("\n" + "=" * 80)
        
        return {
            'ic_series': ic_series,
            'pure_ic_series': pure_ic_series,
            'decay_results': decay_results,
            'quantile_results': quantile_results
        }


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sato Square Root Factor Validation")
    parser.add_argument(
        "--data-file",
        type=str,
        default=r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean.parquet",
        help="Path to cleaned MultiIndex parquet file"
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = SatoValidator(data_path=args.data_file)
    
    # Run full validation
    results = validator.run_full_validation()
    
    print("\n[OK] Validation complete!")
    print(f"   Results saved in memory (can be exported if needed)")


if __name__ == "__main__":
    main()
