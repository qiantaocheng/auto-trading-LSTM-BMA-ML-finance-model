#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
排查 CatBoost 收益率达到 9000% 的具体股票代码和日期
检查是否混入了未复权数据或价格异常的低价股
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bma_models.model_registry import list_snapshots, load_manifest
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_extreme_returns_in_predictions(pred_file: str, return_threshold: float = 50.0):
    """
    在预测文件中查找收益率异常高的股票和日期
    
    Args:
        pred_file: 预测文件路径（parquet或csv）
        return_threshold: 收益率阈值（默认50.0 = 5000%）
    
    Returns:
        DataFrame with extreme returns
    """
    logger.info(f"读取预测文件: {pred_file}")
    
    # 读取文件
    if pred_file.endswith('.parquet'):
        df = pd.read_parquet(pred_file)
    elif pred_file.endswith('.csv'):
        df = pd.read_csv(pred_file, index_col=0)
        if isinstance(df.index, pd.MultiIndex) or 'date' in df.columns:
            # 尝试解析MultiIndex
            if 'date' in df.columns and 'ticker' in df.columns:
                df = df.set_index(['date', 'ticker'])
            elif isinstance(df.index, pd.MultiIndex):
                pass
    else:
        raise ValueError(f"不支持的文件格式: {pred_file}")
    
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"列名: {df.columns.tolist()}")
    
    # 查找收益率列
    return_cols = [col for col in df.columns if 'return' in col.lower() or 'ret' in col.lower() or 'actual' in col.lower()]
    logger.info(f"找到收益率相关列: {return_cols}")
    
    if not return_cols:
        logger.warning("未找到收益率列，尝试查找target列")
        if 'target' in df.columns:
            return_cols = ['target']
        else:
            logger.error("无法找到收益率列")
            return pd.DataFrame()
    
    # 检查每个收益率列
    extreme_records = []
    
    for col in return_cols:
        logger.info(f"\n检查列: {col}")
        returns = df[col].copy()
        
        # 转换为百分比（如果是小数形式）
        if returns.abs().max() < 10:
            returns_pct = returns * 100  # 转换为百分比
        else:
            returns_pct = returns  # 已经是百分比
        
        # 查找极端值
        extreme_mask = returns_pct.abs() >= return_threshold
        extreme_count = extreme_mask.sum()
        
        logger.info(f"  收益率范围: {returns_pct.min():.2f}% ~ {returns_pct.max():.2f}%")
        logger.info(f"  极端值数量 (>= {return_threshold}%): {extreme_count}")
        
        if extreme_count > 0:
            extreme_df = df[extreme_mask].copy()
            extreme_df['return_pct'] = returns_pct[extreme_mask]
            extreme_df['return_column'] = col
            
            # 提取日期和股票代码
            if isinstance(df.index, pd.MultiIndex):
                extreme_df['date'] = extreme_df.index.get_level_values('date')
                extreme_df['ticker'] = extreme_df.index.get_level_values('ticker')
            elif 'date' in extreme_df.columns and 'ticker' in extreme_df.columns:
                pass  # 已经有date和ticker列
            else:
                logger.warning("无法提取日期和股票代码")
                extreme_df['date'] = None
                extreme_df['ticker'] = None
            
            extreme_records.append(extreme_df)
    
    if extreme_records:
        result = pd.concat(extreme_records, ignore_index=False)
        result = result.sort_values('return_pct', key=abs, ascending=False)
        return result
    else:
        return pd.DataFrame()


def check_price_data(ticker: str, date: str, data_source: str = None):
    """
    检查特定股票和日期的价格数据，判断是否是未复权数据或低价股
    
    Args:
        ticker: 股票代码
        date: 日期
        data_source: 数据源路径（可选）
    
    Returns:
        dict with price information
    """
    logger.info(f"检查股票 {ticker} 在 {date} 的价格数据")
    
    # 尝试从数据源读取价格
    if data_source:
        try:
            if data_source.endswith('.parquet'):
                df = pd.read_parquet(data_source)
                if isinstance(df.index, pd.MultiIndex):
                    ticker_data = df.xs(ticker, level='ticker', drop_level=False)
                    date_data = ticker_data.xs(date, level='date', drop_level=False)
                    
                    if 'close' in date_data.columns:
                        close_price = date_data['close'].iloc[0]
                        return {
                            'ticker': ticker,
                            'date': date,
                            'close_price': close_price,
                            'is_low_price': close_price < 1.0,  # 低于1美元可能是低价股或未复权
                            'is_very_low_price': close_price < 0.1,  # 低于0.1美元可能是未复权
                        }
        except Exception as e:
            logger.warning(f"无法从数据源读取价格: {e}")
    
    return {
        'ticker': ticker,
        'date': date,
        'close_price': None,
        'is_low_price': None,
        'is_very_low_price': None,
    }


def investigate_catboost_extreme_returns(snapshot_id: str = None, 
                                         data_file: str = None,
                                         return_threshold: float = 50.0):
    """
    排查 CatBoost 收益率异常
    
    Args:
        snapshot_id: 快照ID（None则使用最新）
        data_file: 原始数据文件路径（用于检查价格）
        return_threshold: 收益率阈值（默认50.0 = 5000%）
    """
    logger.info("=" * 80)
    logger.info("排查 CatBoost 收益率异常")
    logger.info("=" * 80)
    
    # 1. 获取最新快照
    if snapshot_id is None:
        snaps = list_snapshots()
        if not snaps:
            logger.error("未找到任何快照")
            return
        snapshot_id = snaps[0][0]
        logger.info(f"使用最新快照: {snapshot_id}")
    
    # 2. 加载快照清单
    try:
        manifest = load_manifest(snapshot_id)
        logger.info(f"快照标签: {manifest.get('tag', 'N/A')}")
    except Exception as e:
        logger.error(f"无法加载快照清单: {e}")
        return
    
    # 3. 查找 CatBoost 预测结果文件
    # 通常在 result/ 目录下
    result_dir = project_root / "result"
    catboost_files = []
    
    # 查找所有可能的 CatBoost 结果文件
    for pattern in ['**/*catboost*.parquet', '**/*catboost*.csv', '**/model_backtest/*catboost*.parquet']:
        files = list(result_dir.glob(pattern))
        catboost_files.extend(files)
    
    if not catboost_files:
        logger.warning("未找到 CatBoost 预测结果文件")
        logger.info("尝试查找所有结果文件...")
        all_files = list(result_dir.glob("**/*.parquet")) + list(result_dir.glob("**/*.csv"))
        logger.info(f"找到 {len(all_files)} 个结果文件")
        for f in all_files[:10]:
            logger.info(f"  - {f}")
    else:
        logger.info(f"找到 {len(catboost_files)} 个 CatBoost 结果文件")
        for f in catboost_files:
            logger.info(f"  - {f}")
    
    # 4. 检查每个文件中的极端收益率
    all_extreme_records = []
    
    for pred_file in catboost_files:
        logger.info(f"\n检查文件: {pred_file}")
        try:
            extreme_df = find_extreme_returns_in_predictions(str(pred_file), return_threshold)
            if not extreme_df.empty:
                logger.info(f"找到 {len(extreme_df)} 条极端收益率记录")
                all_extreme_records.append(extreme_df)
        except Exception as e:
            logger.error(f"检查文件失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. 汇总结果
    if all_extreme_records:
        combined_df = pd.concat(all_extreme_records, ignore_index=False)
        combined_df = combined_df.sort_values('return_pct', key=abs, ascending=False)
        
        logger.info("\n" + "=" * 80)
        logger.info("极端收益率汇总")
        logger.info("=" * 80)
        logger.info(f"\n总共找到 {len(combined_df)} 条极端收益率记录 (>= {return_threshold}%)")
        
        # 显示Top 20
        top_20 = combined_df.head(20)
        logger.info("\nTop 20 极端收益率:")
        for idx, row in top_20.iterrows():
            ticker = row.get('ticker', 'N/A')
            date = row.get('date', 'N/A')
            return_pct = row.get('return_pct', 0)
            logger.info(f"  {ticker} @ {date}: {return_pct:.2f}%")
        
        # 保存结果
        output_file = project_root / "result" / f"catboost_extreme_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        combined_df.to_csv(output_file)
        logger.info(f"\n结果已保存到: {output_file}")
        
        # 6. 检查价格数据（如果提供了数据文件）
        if data_file and Path(data_file).exists():
            logger.info("\n检查价格数据...")
            price_checks = []
            for idx, row in top_20.iterrows():
                ticker = row.get('ticker', None)
                date = row.get('date', None)
                if ticker and date:
                    price_info = check_price_data(ticker, date, data_file)
                    price_checks.append(price_info)
            
            if price_checks:
                price_df = pd.DataFrame(price_checks)
                logger.info("\n价格检查结果:")
                logger.info(price_df.to_string())
                
                # 检查低价股
                low_price_count = price_df['is_low_price'].sum() if 'is_low_price' in price_df.columns else 0
                very_low_price_count = price_df['is_very_low_price'].sum() if 'is_very_low_price' in price_df.columns else 0
                
                logger.info(f"\n低价股 (< $1): {low_price_count}")
                logger.info(f"极低价股 (< $0.1): {very_low_price_count}")
        
        return combined_df
    else:
        logger.info("\n未找到极端收益率记录")
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="排查 CatBoost 收益率异常")
    parser.add_argument("--snapshot-id", type=str, default=None, help="快照ID（默认使用最新）")
    parser.add_argument("--data-file", type=str, default=None, help="原始数据文件路径（用于检查价格）")
    parser.add_argument("--threshold", type=float, default=50.0, help="收益率阈值（默认50.0 = 5000%）")
    
    args = parser.parse_args()
    
    # 默认数据文件路径
    if args.data_file is None:
        default_data_file = project_root / "data" / "factor_exports" / "polygon_factors_all_filtered.parquet"
        if default_data_file.exists():
            args.data_file = str(default_data_file)
            logger.info(f"使用默认数据文件: {args.data_file}")
    
    investigate_catboost_extreme_returns(
        snapshot_id=args.snapshot_id,
        data_file=args.data_file,
        return_threshold=args.threshold
    )
