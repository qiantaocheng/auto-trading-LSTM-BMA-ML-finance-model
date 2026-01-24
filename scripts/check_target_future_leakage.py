#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查target是否包含未来信息泄露

关键问题：测试集最后几天的target可能使用了测试期外的数据
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Fix module import path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("check_target_leakage")


def check_target_future_leakage(data_file: str, test_start_date: str, test_end_date: str, horizon: int = 10):
    """检查target是否包含未来信息泄露"""
    logger.info("=" * 80)
    logger.info("检查target是否包含未来信息泄露")
    logger.info("=" * 80)
    
    # 加载数据
    logger.info(f"Loading data from {data_file}")
    df = pd.read_parquet(data_file)
    
    # 获取所有日期
    all_dates = df.index.get_level_values('date').unique().sort_values()
    logger.info(f"Total dates in data: {len(all_dates)}")
    logger.info(f"Date range: {all_dates[0].date()} to {all_dates[-1].date()}")
    
    # 过滤测试期数据
    test_start = pd.to_datetime(test_start_date)
    test_end = pd.to_datetime(test_end_date)
    
    test_data = df.loc[
        (df.index.get_level_values('date') >= test_start) & 
        (df.index.get_level_values('date') <= test_end)
    ].copy()
    
    test_dates = test_data.index.get_level_values('date').unique().sort_values()
    logger.info(f"\nTest period: {test_start.date()} to {test_end.date()}")
    logger.info(f"Test dates: {len(test_dates)} unique dates")
    
    # 检查最后几天的target
    logger.info("\n" + "=" * 80)
    logger.info("检查最后几天的target（可能包含未来信息）")
    logger.info("=" * 80)
    
    # target是T+10的forward return
    # 对于测试期的最后一天，target需要未来10天的数据
    # 如果测试期结束日期是2025-01-23，那么最后一天的target需要到2025-02-06的数据
    
    last_test_date = test_dates[-1]
    required_future_date = last_test_date + pd.Timedelta(days=horizon * 2)  # 考虑交易日
    
    logger.info(f"Last test date: {last_test_date.date()}")
    logger.info(f"Required future date for target: ~{required_future_date.date()}")
    logger.info(f"Last available date in data: {all_dates[-1].date()}")
    
    if all_dates[-1] < required_future_date:
        logger.warning(f"\n⚠️ POTENTIAL FUTURE LEAKAGE DETECTED!")
        logger.warning(f"   Last test date: {last_test_date.date()}")
        logger.warning(f"   Required future date: ~{required_future_date.date()}")
        logger.warning(f"   Last available date: {all_dates[-1].date()}")
        logger.warning(f"   Difference: {(required_future_date - all_dates[-1]).days} days")
        logger.warning(f"\n   这意味着测试期最后几天的target可能使用了测试期外的数据！")
        logger.warning(f"   或者target被错误计算（使用了历史数据而不是未来数据）")
    
    # 检查最后几天的target值
    logger.info("\n" + "=" * 80)
    logger.info("检查最后几天的target值")
    logger.info("=" * 80)
    
    last_n_days = 5
    for i, date in enumerate(test_dates[-last_n_days:]):
        date_data = test_data.xs(date, level='date', drop_level=False)
        
        # 计算这个日期需要的未来日期
        future_date_needed = date + pd.Timedelta(days=horizon * 2)
        has_future_data = all_dates[-1] >= future_date_needed
        
        logger.info(f"\nDate {i+1}: {date.date()}")
        logger.info(f"  Future date needed: ~{future_date_needed.date()}")
        logger.info(f"  Has future data: {has_future_data}")
        logger.info(f"  Target mean: {date_data['target'].mean():.6f}")
        logger.info(f"  Target std: {date_data['target'].std():.6f}")
        logger.info(f"  Target min: {date_data['target'].min():.6f}")
        logger.info(f"  Target max: {date_data['target'].max():.6f}")
        
        # 检查是否有异常值
        extreme_mask = date_data['target'].abs() > 0.5
        if extreme_mask.sum() > 0:
            logger.warning(f"  ⚠️ Found {extreme_mask.sum()} extreme values (|target| > 0.5)")
            extreme_values = date_data.loc[extreme_mask, 'target']
            logger.warning(f"    Extreme values: {extreme_values.head(5).tolist()}")
    
    # 检查target的计算是否正确
    logger.info("\n" + "=" * 80)
    logger.info("检查target的计算是否正确")
    logger.info("=" * 80)
    
    # 对于测试期的第一天，手动计算target
    first_test_date = test_dates[0]
    logger.info(f"\nFirst test date: {first_test_date.date()}")
    
    # 获取这个日期的数据
    first_date_data = test_data.xs(first_test_date, level='date', drop_level=False)
    
    # 检查是否有Close列
    if 'Close' in first_date_data.columns:
        # 手动计算target（T+10 forward return）
        # target = (Close[t+10] - Close[t]) / Close[t]
        
        # 对于每个ticker，计算forward return
        sample_tickers = first_date_data.index.get_level_values('ticker').unique()[:5]
        
        logger.info(f"\n手动验证target计算（前5个ticker）:")
        for ticker in sample_tickers:
            ticker_data = df.xs(ticker, level='ticker', drop_level=False)
            ticker_data = ticker_data.sort_index()
            
            # 找到first_test_date的位置
            if first_test_date in ticker_data.index.get_level_values('date'):
                date_idx = ticker_data.index.get_level_values('date').get_loc(first_test_date)
                if isinstance(date_idx, slice):
                    date_idx = date_idx.start
                
                # 获取当前和未来10天的Close
                current_close = ticker_data.iloc[date_idx]['Close']
                
                # 找到未来10天的Close
                future_idx = date_idx + horizon
                if future_idx < len(ticker_data):
                    future_close = ticker_data.iloc[future_idx]['Close']
                    manual_target = (future_close - current_close) / current_close
                    
                    # 获取实际的target值
                    actual_target = first_date_data.xs(ticker, level='ticker', drop_level=False)['target'].iloc[0]
                    
                    logger.info(f"  {ticker}:")
                    logger.info(f"    Manual target: {manual_target:.6f}")
                    logger.info(f"    Actual target: {actual_target:.6f}")
                    logger.info(f"    Difference: {abs(manual_target - actual_target):.6f}")
                    
                    if abs(manual_target - actual_target) > 0.01:
                        logger.warning(f"    ⚠️ Large difference detected!")


def main():
    parser = argparse.ArgumentParser(description="检查target是否包含未来信息泄露")
    parser.add_argument("--data-file", type=str, required=True,
                       help="数据文件路径")
    parser.add_argument("--test-start-date", type=str, required=True,
                       help="测试开始日期 (YYYY-MM-DD)")
    parser.add_argument("--test-end-date", type=str, required=True,
                       help="测试结束日期 (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=10,
                       help="预测horizon（天数）")
    
    args = parser.parse_args()
    
    check_target_future_leakage(
        data_file=args.data_file,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        horizon=args.horizon
    )


if __name__ == "__main__":
    main()
