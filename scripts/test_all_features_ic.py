#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All Features IC Test
对所有MultiIndex cleaned数据中的features进行IC测试
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
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


class AllFeaturesICTester:
    """
    对所有features进行IC测试
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize tester with MultiIndex cleaned data
        
        Args:
            data_path: Path to cleaned parquet file (MultiIndex format)
        """
        if data_path is None:
            data_path = r"D:\trade\data\factor_exports\polygon_factors_all_filtered_clean.parquet"
        
        print("=" * 80)
        print("All Features IC Test")
        print("=" * 80)
        print(f"Loading data from: {data_path}")
        
        # Load MultiIndex data
        self.df = pd.read_parquet(data_path)
        
        # Validate MultiIndex format
        if not isinstance(self.df.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex (date, ticker) format")
        
        # Sort by date and ticker
        self.df = self.df.sort_index()
        
        print(f"[OK] Data loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
        print(f"   Date range: {self.df.index.get_level_values('date').min()} to {self.df.index.get_level_values('date').max()}")
        print(f"   Unique tickers: {self.df.index.get_level_values(self.df.index.names[1]).nunique():,}")
        
        # Identify feature columns (exclude Close, target, and index columns)
        exclude_cols = ['Close', 'target', 'date', 'ticker']
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        print(f"\n[INFO] Found {len(self.feature_cols)} features:")
        for i, col in enumerate(self.feature_cols, 1):
            print(f"   {i:2d}. {col}")
        
        # Prepare forward returns
        self._prepare_forward_returns()
    
    def _prepare_forward_returns(self):
        """计算T+10 forward returns"""
        print("\n[INFO] Computing T+10 forward returns...")
        
        def calc_fwd_ret(group):
            ret = group.pct_change(10).shift(-10)
            # Filter extreme returns (likely data errors, splits, etc.)
            ret = ret.clip(lower=-0.9, upper=10.0)
            return ret
        
        fwd_ret_series = self.df.groupby(level=1)['Close'].apply(calc_fwd_ret)
        # Handle MultiIndex result
        if isinstance(fwd_ret_series.index, pd.MultiIndex):
            if len(fwd_ret_series.index.names) > 2:
                fwd_ret_series = fwd_ret_series.droplevel(0)
            fwd_ret_series.index = self.df.index
        
        self.df['fwd_ret_10d'] = fwd_ret_series
        
        # Drop rows with NaN forward returns
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['fwd_ret_10d'])
        final_len = len(self.df)
        
        print(f"[OK] Forward returns prepared: {final_len:,} rows (dropped {initial_len - final_len:,} NaN rows)")
    
    def test_single_feature_ic(self, feature_col: str):
        """
        测试单个feature的IC
        
        Args:
            feature_col: Feature column name
        
        Returns:
            dict with IC statistics
        """
        # 检查feature是否存在
        if feature_col not in self.df.columns:
            return None
        
        # 每日计算 Rank IC（横截面比较）
        def calc_ic(group):
            if len(group) < 10:  # 至少需要10只股票
                return np.nan
            
            # 检查是否有足够的非NaN值
            valid_mask = group[[feature_col, 'fwd_ret_10d']].notna().all(axis=1)
            if valid_mask.sum() < 10:
                return np.nan
            
            try:
                ic, pval = spearmanr(group[feature_col], group['fwd_ret_10d'])
                return ic
            except:
                return np.nan
        
        ic_series = self.df.groupby(level=0).apply(calc_ic)
        ic_series = ic_series.dropna()
        
        if len(ic_series) == 0:
            return None
        
        # 计算统计量
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0
        positive_ratio = (ic_series > 0).mean()
        
        return {
            'feature': feature_col,
            'mean_ic': mean_ic,
            'std_ic': std_ic,
            'ic_ir': ic_ir,
            'positive_ratio': positive_ratio,
            'valid_days': len(ic_series),
            'min_ic': ic_series.min(),
            'p25_ic': ic_series.quantile(0.25),
            'median_ic': ic_series.median(),
            'p75_ic': ic_series.quantile(0.75),
            'max_ic': ic_series.max(),
            'ic_series': ic_series  # 保存完整的IC序列用于进一步分析
        }
    
    def test_all_features(self):
        """
        测试所有features的IC
        
        Returns:
            DataFrame with IC statistics for all features
        """
        print("\n" + "=" * 80)
        print(">>> Testing All Features IC...")
        print("=" * 80)
        
        results = []
        
        for i, feature_col in enumerate(self.feature_cols, 1):
            print(f"\n[{i}/{len(self.feature_cols)}] Testing {feature_col}...", end='', flush=True)
            
            result = self.test_single_feature_ic(feature_col)
            
            if result is None:
                print(" [SKIP] Insufficient data")
                continue
            
            results.append(result)
            print(f" [OK] Mean IC: {result['mean_ic']:.4f}, IR: {result['ic_ir']:.4f}")
        
        if len(results) == 0:
            print("\n[ERROR] No valid results computed")
            return None
        
        # 转换为DataFrame
        results_df = pd.DataFrame([
            {
                'feature': r['feature'],
                'mean_ic': r['mean_ic'],
                'std_ic': r['std_ic'],
                'ic_ir': r['ic_ir'],
                'positive_ratio': r['positive_ratio'],
                'valid_days': r['valid_days'],
                'min_ic': r['min_ic'],
                'p25_ic': r['p25_ic'],
                'median_ic': r['median_ic'],
                'p75_ic': r['p75_ic'],
                'max_ic': r['max_ic']
            }
            for r in results
        ])
        
        # 按mean_ic排序
        results_df = results_df.sort_values('mean_ic', ascending=False)
        
        return results_df, results
    
    def print_summary(self, results_df: pd.DataFrame):
        """打印汇总报告"""
        print("\n" + "=" * 80)
        print("IC TEST SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal features tested: {len(results_df)}")
        print(f"Features with positive IC: {(results_df['mean_ic'] > 0).sum()}")
        print(f"Features with IC > 0.01: {(results_df['mean_ic'] > 0.01).sum()}")
        print(f"Features with IC > 0.02: {(results_df['mean_ic'] > 0.02).sum()}")
        print(f"Features with IC IR > 0.5: {(results_df['ic_ir'] > 0.5).sum()}")
        
        print("\n" + "-" * 80)
        print("Top 10 Features by Mean IC:")
        print("-" * 80)
        top10 = results_df.head(10)
        for idx, row in top10.iterrows():
            print(f"{row['feature']:30s} | Mean IC: {row['mean_ic']:7.4f} | IR: {row['ic_ir']:6.2f} | Pos Ratio: {row['positive_ratio']:5.1%}")
        
        print("\n" + "-" * 80)
        print("Bottom 10 Features by Mean IC:")
        print("-" * 80)
        bottom10 = results_df.tail(10)
        for idx, row in bottom10.iterrows():
            print(f"{row['feature']:30s} | Mean IC: {row['mean_ic']:7.4f} | IR: {row['ic_ir']:6.2f} | Pos Ratio: {row['positive_ratio']:5.1%}")
        
        print("\n" + "-" * 80)
        print("Top 10 Features by IC IR (Information Ratio):")
        print("-" * 80)
        top10_ir = results_df.nlargest(10, 'ic_ir')
        for idx, row in top10_ir.iterrows():
            print(f"{row['feature']:30s} | Mean IC: {row['mean_ic']:7.4f} | IR: {row['ic_ir']:6.2f} | Pos Ratio: {row['positive_ratio']:5.1%}")
        
        print("\n" + "=" * 80)
        print("Detailed Results Table:")
        print("=" * 80)
        print(results_df.to_string(index=False))
    
    def save_results(self, results_df: pd.DataFrame, output_path: str = None):
        """保存结果到CSV"""
        if output_path is None:
            output_path = r"D:\trade\scripts\all_features_ic_results.csv"
        
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[OK] Results saved to: {output_path}")


def main():
    """主函数"""
    tester = AllFeaturesICTester()
    
    # 测试所有features
    results_df, results_dict = tester.test_all_features()
    
    if results_df is None:
        print("\n[ERROR] No results to display")
        return
    
    # 打印汇总
    tester.print_summary(results_df)
    
    # 保存结果
    tester.save_results(results_df)
    
    print("\n[OK] All features IC test complete!")


if __name__ == "__main__":
    main()
