#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算LambdaRank、CatBoost和Stacker的non-overlapping max drawdown

Non-overlapping max drawdown:
1. 将时间序列分成不重叠的窗口（窗口大小=horizon_days，默认10天）
2. 在每个窗口内计算累计收益
3. 计算每个窗口的最大回撤
4. 取所有窗口的最大回撤的最大值
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_non_overlapping_max_drawdown(
    returns_series: pd.Series,
    horizon_days: int = 10,
    window_size: Optional[int] = None
) -> Dict[str, float]:
    """
    计算non-overlapping max drawdown
    
    Args:
        returns_series: 收益率序列（按日期排序）
        horizon_days: 预测horizon（用于确定窗口大小）
        window_size: 窗口大小（如果不指定，使用horizon_days）
    
    Returns:
        Dict包含:
        - max_drawdown: 所有窗口的最大回撤的最大值
        - window_drawdowns: 每个窗口的最大回撤列表
        - n_windows: 窗口数量
    """
    if window_size is None:
        window_size = horizon_days
    
    if len(returns_series) == 0:
        return {
            'max_drawdown': np.nan,
            'window_drawdowns': [],
            'n_windows': 0,
            'mean_window_dd': np.nan,
            'median_window_dd': np.nan
        }
    
    # 确保按日期排序
    returns_series = returns_series.sort_index()
    
    # 将时间序列分成不重叠的窗口
    n_samples = len(returns_series)
    n_windows = n_samples // window_size
    
    if n_windows == 0:
        # 如果数据不足一个窗口，使用整个序列
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        return {
            'max_drawdown': max_dd,
            'window_drawdowns': [max_dd],
            'n_windows': 1,
            'mean_window_dd': max_dd,
            'median_window_dd': max_dd
        }
    
    window_drawdowns = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        window_returns = returns_series.iloc[start_idx:end_idx]
        
        if len(window_returns) == 0:
            continue
        
        # 计算窗口内的累计收益
        cumulative = (1 + window_returns).cumprod()
        
        # 计算最大回撤
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        window_drawdowns.append(max_dd)
    
    # 处理剩余数据（如果有）
    if n_samples % window_size > 0:
        remaining_returns = returns_series.iloc[n_windows * window_size:]
        if len(remaining_returns) > 1:
            cumulative = (1 + remaining_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = abs(drawdown.min())
            window_drawdowns.append(max_dd)
    
    if len(window_drawdowns) == 0:
        return {
            'max_drawdown': np.nan,
            'window_drawdowns': [],
            'n_windows': 0,
            'mean_window_dd': np.nan,
            'median_window_dd': np.nan
        }
    
    return {
        'max_drawdown': max(window_drawdowns),
        'window_drawdowns': window_drawdowns,
        'n_windows': len(window_drawdowns),
        'mean_window_dd': np.mean(window_drawdowns),
        'median_window_dd': np.median(window_drawdowns)
    }


def load_bucket_returns_from_report(
    report_csv_path: str,
    model_name: str
) -> Optional[pd.Series]:
    """
    从report_df.csv中提取模型的bucket returns
    
    注意：report_df.csv可能不包含时间序列数据，需要从其他文件加载
    """
    try:
        report_df = pd.read_csv(report_csv_path)
        if model_name not in report_df['Model'].values:
            logger.warning(f"模型 {model_name} 不在report_df中")
            return None
        
        # report_df只包含汇总统计，不包含时间序列
        # 需要从其他文件加载时间序列数据
        logger.warning(f"report_df.csv只包含汇总统计，无法提取时间序列")
        return None
    except Exception as e:
        logger.error(f"加载report_df失败: {e}")
        return None


def load_time_series_from_bucket_file(
    bucket_csv_path: str,
    model_name: str,
    bucket_name: str = 'top1_10'
) -> Optional[pd.Series]:
    """
    从bucket returns CSV文件中加载时间序列
    
    Args:
        bucket_csv_path: bucket returns CSV文件路径
        model_name: 模型名称（用于匹配列名）
        bucket_name: bucket名称（如'top1_10', 'top5_15'等）
    
    Returns:
        收益率时间序列
    """
    try:
        df = pd.read_csv(bucket_csv_path)
        
        # 查找包含模型名称和bucket名称的列
        # 可能的列名格式: 'lambdarank_top1_10', 'catboost_top1_10', 'ridge_stacking_top1_10'
        col_name = f"{model_name}_{bucket_name}"
        
        if col_name in df.columns:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            return df[col_name]
        else:
            logger.warning(f"列 {col_name} 不在bucket文件中")
            logger.info(f"可用列: {df.columns.tolist()}")
            return None
    except Exception as e:
        logger.error(f"加载bucket文件失败: {e}")
        return None


def find_latest_eval_results(base_dir: str = "result") -> Optional[Path]:
    """查找最新的80/20评估结果目录"""
    base_path = Path(base_dir)
    
    # 查找所有run_*目录
    run_dirs = sorted(base_path.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not run_dirs:
        logger.warning(f"未找到run_*目录在 {base_dir}")
        return None
    
    latest_dir = run_dirs[0]
    logger.info(f"找到最新评估结果目录: {latest_dir}")
    return latest_dir


def calculate_from_group_returns(
    group_ts_csv_path: str,
    model_name: str,
    horizon_days: int = 10
) -> Optional[Dict[str, float]]:
    """
    从group returns时间序列CSV计算non-overlapping max drawdown
    
    Args:
        group_ts_csv_path: group returns时间序列CSV路径
        model_name: 模型名称
        horizon_days: 预测horizon天数
    """
    try:
        df = pd.read_csv(group_ts_csv_path)
        
        if 'date' not in df.columns:
            logger.error("CSV文件缺少'date'列")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # 查找Top-N收益列（通常为top_return_net_mean或top_return）
        return_col = None
        for col in ['top_return_net_mean', 'top_return_net', 'top_return']:
            if col in df.columns:
                return_col = col
                break
        
        if return_col is None:
            logger.warning(f"未找到收益列，可用列: {df.columns.tolist()}")
            return None
        
        returns_series = df[return_col].dropna()
        
        # 检查数据格式：如果是百分比形式（绝对值>1），转换为小数
        if returns_series.abs().max() > 1.0:
            logger.info(f"检测到百分比格式数据，转换为小数（除以100）")
            returns_series = returns_series / 100.0
        
        if len(returns_series) == 0:
            logger.warning(f"模型 {model_name} 没有有效收益数据")
            return None
        
        result = calculate_non_overlapping_max_drawdown(returns_series, horizon_days)
        result['model'] = model_name
        result['n_samples'] = len(returns_series)
        result['return_col'] = return_col
        
        return result
        
    except Exception as e:
        logger.error(f"计算失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='计算non-overlapping max drawdown')
    parser.add_argument('--result-dir', type=str, default=None, help='结果目录（默认查找最新的run_*目录）')
    parser.add_argument('--horizon-days', type=int, default=10, help='预测horizon天数（默认10）')
    parser.add_argument('--window-size', type=int, default=None, help='窗口大小（默认使用horizon_days）')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Non-Overlapping Max Drawdown 计算")
    logger.info("="*80)
    
    # 查找结果目录
    if args.result_dir:
        result_dir = Path(args.result_dir)
    else:
        result_dir = find_latest_eval_results()
        if result_dir is None:
            logger.error("未找到评估结果目录")
            return 1
    
    logger.info(f"\n使用结果目录: {result_dir}")
    
    # 查找report_df.csv以获取模型列表
    report_csv = result_dir / "report_df.csv"
    if not report_csv.exists():
        logger.error(f"未找到report_df.csv在 {result_dir}")
        return 1
    
    report_df = pd.read_csv(report_csv)
    logger.info(f"\n找到 {len(report_df)} 个模型在report_df.csv")
    
    # 定义要分析的模型
    target_models = {
        'lambdarank': 'lambdarank',
        'catboost': 'catboost',
        'ridge_stacking': 'ridge_stacking'  # Stacker
    }
    
    # 查找每个模型的时间序列文件
    results = {}
    
    for display_name, model_name in target_models.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"分析模型: {display_name} ({model_name})")
        logger.info(f"{'='*60}")
        
        # 查找模型的时间序列文件
        # 可能的文件名格式: {model_name}_top20_timeseries.csv, {model_name}_top30_timeseries.csv
        ts_files = list(result_dir.glob(f"{model_name}_top*_timeseries.csv"))
        
        # 特殊处理：ridge_stacking的文件名可能是ridge_top20_timeseries.csv
        if not ts_files and model_name == 'ridge_stacking':
            ts_files = list(result_dir.glob("ridge_top*_timeseries.csv"))
        
        if not ts_files:
            logger.warning(f"未找到 {model_name} 的时间序列文件")
            # 尝试查找通用格式
            ts_files = list(result_dir.glob(f"*{model_name}*timeseries.csv"))
        
        if not ts_files:
            logger.warning(f"模型 {model_name} 的时间序列文件不存在，跳过")
            continue
        
        # 使用第一个找到的文件
        ts_file = ts_files[0]
        logger.info(f"使用时间序列文件: {ts_file.name}")
        
        # 计算non-overlapping max drawdown
        result = calculate_from_group_returns(
            str(ts_file),
            model_name,
            horizon_days=args.horizon_days
        )
        
        if result:
            results[display_name] = result
            logger.info(f"\n结果:")
            logger.info(f"  Non-Overlapping Max Drawdown: {result['max_drawdown']:.4f} ({result['max_drawdown']*100:.2f}%)")
            logger.info(f"  窗口数量: {result['n_windows']}")
            logger.info(f"  平均窗口回撤: {result['mean_window_dd']:.4f} ({result['mean_window_dd']*100:.2f}%)")
            logger.info(f"  中位数窗口回撤: {result['median_window_dd']:.4f} ({result['median_window_dd']*100:.2f}%)")
            logger.info(f"  样本数量: {result['n_samples']}")
        else:
            logger.warning(f"模型 {display_name} 计算失败")
    
    # 输出汇总
    logger.info("\n" + "="*80)
    logger.info("汇总结果")
    logger.info("="*80)
    
    if results:
        summary_df = pd.DataFrame([
            {
                'Model': name,
                'Non-Overlapping Max Drawdown': f"{r['max_drawdown']:.4f} ({r['max_drawdown']*100:.2f}%)",
                'N_Windows': r['n_windows'],
                'Mean Window DD': f"{r['mean_window_dd']:.4f} ({r['mean_window_dd']*100:.2f}%)",
                'Median Window DD': f"{r['median_window_dd']:.4f} ({r['median_window_dd']*100:.2f}%)",
                'N_Samples': r['n_samples']
            }
            for name, r in results.items()
        ])
        
        print("\n" + summary_df.to_string(index=False))
        
        # 保存结果
        output_file = result_dir / "non_overlapping_max_drawdown.csv"
        summary_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"\n✅ 结果已保存到: {output_file}")
    else:
        logger.warning("没有成功计算任何模型的结果")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
