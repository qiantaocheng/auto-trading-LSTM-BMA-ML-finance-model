#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 CatBoost 和 LambdaRank 的累计收益（基于非重叠回测）
从最新的评估运行中提取预测数据并计算
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import glob

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.time_split_80_20_oos_eval import calculate_group_returns_hold10d_nonoverlap

def find_latest_run_with_predictions():
    """找到最新的包含预测数据的运行目录"""
    result_dirs = glob.glob(str(project_root / "results" / "t10_time_split_80_20" / "run_*"))
    if not result_dirs:
        return None
    
    # Sort by modification time
    latest_dir = max(result_dirs, key=lambda x: Path(x).stat().st_mtime)
    return Path(latest_dir)

def load_predictions_from_run(run_dir: Path):
    """从运行目录加载预测数据（如果已保存）"""
    # Check if there's a predictions file
    pred_files = list(run_dir.glob("*_predictions*.parquet")) + list(run_dir.glob("*_predictions*.csv"))
    if pred_files:
        latest_pred = max(pred_files, key=lambda x: x.stat().st_mtime)
        if latest_pred.suffix == '.parquet':
            return pd.read_parquet(latest_pred)
        else:
            return pd.read_csv(latest_pred)
    return None

def calculate_accumulated_returns():
    """计算 CatBoost 和 LambdaRank 的累计收益"""
    print("=" * 80)
    print("计算 CatBoost 和 LambdaRank 累计收益（非重叠回测）")
    print("=" * 80)
    
    # Find latest run
    latest_run = find_latest_run_with_predictions()
    if not latest_run:
        print("❌ 未找到运行目录")
        return
    
    print(f"\n📁 最新运行目录: {latest_run.name}")
    
    # Try to load predictions
    predictions = load_predictions_from_run(latest_run)
    
    if predictions is None:
        print("\n⚠️  未找到保存的预测数据文件")
        print("💡 需要重新运行评估以生成预测数据")
        print(f"\n运行命令:")
        print(f"python scripts/time_split_80_20_oos_eval.py \\")
        print(f"    --horizon-days 10 --top-n 20 --cost-bps 10 \\")
        print(f"    --output-dir results/t10_time_split_80_20 \\")
        print(f"    --models catboost lambdarank --snapshot-id <snapshot_id>")
        return
    
    print(f"✅ 加载预测数据: {len(predictions)} 条记录")
    
    # Check required columns
    required_cols = ['date', 'ticker', 'prediction', 'actual']
    missing = [col for col in required_cols if col not in predictions.columns]
    if missing:
        print(f"❌ 缺少必需的列: {missing}")
        return
    
    # Calculate for each model
    models = ['catboost', 'lambdarank']
    results = {}
    
    for model_name in models:
        # Filter predictions for this model
        # Assuming predictions have a 'model' column or we need to filter differently
        if 'model' in predictions.columns:
            model_preds = predictions[predictions['model'] == model_name].copy()
        else:
            # If no model column, assume all predictions are for the same model
            # We'll need to check the actual structure
            print(f"\n⚠️  预测数据中没有'model'列，尝试使用全部数据")
            model_preds = predictions.copy()
        
        if model_preds.empty:
            print(f"\n⚠️  {model_name}: 未找到预测数据")
            continue
        
        print(f"\n{'='*60}")
        print(f"📊 {model_name.upper()}")
        print(f"{'='*60}")
        print(f"预测数量: {len(model_preds)}")
        print(f"日期范围: {model_preds['date'].min()} 到 {model_preds['date'].max()}")
        
        # Calculate non-overlapping returns
        try:
            group_ts = calculate_group_returns_hold10d_nonoverlap(
                model_preds, 
                top_n=20, 
                horizon_days=10, 
                cost_bps=10.0, 
                start_offset=0
            )
            
            if group_ts.empty:
                print(f"❌ {model_name}: 时间序列为空")
                continue
            
            print(f"\n✅ 非重叠回测结果:")
            print(f"   时间序列行数: {len(group_ts)} (每10天一期)")
            print(f"   日期范围: {group_ts['date'].min()} 到 {group_ts['date'].max()}")
            
            # Calculate cumulative returns
            def _cum_pct(s_pct: pd.Series) -> pd.Series:
                r = pd.to_numeric(s_pct, errors="coerce").fillna(0.0) / 100.0
                return (1.0 + r).cumprod() - 1.0
            
            # Convert to percent and calculate cumulative
            top_return_pct = group_ts['top_return'] * 100.0
            top_return_net_pct = group_ts['top_return_net'] * 100.0
            
            cum_gross = _cum_pct(top_return_pct) * 100.0
            cum_net = _cum_pct(top_return_net_pct) * 100.0
            
            final_gross = cum_gross.iloc[-1]
            final_net = cum_net.iloc[-1]
            
            print(f"\n📈 累计收益:")
            print(f"   最终累计收益 (Gross): {final_gross:.2f}%")
            print(f"   最终累计收益 (Net):   {final_net:.2f}%")
            
            # Calculate statistics
            periods_per_year = 252.0 / 10
            net_series = group_ts['top_return_net'].dropna()
            if len(net_series) > 1 and net_series.std() > 0:
                sharpe = (net_series.mean() / net_series.std()) * np.sqrt(periods_per_year)
                win_rate = (net_series > 0).mean()
                print(f"\n📊 统计指标:")
                print(f"   Sharpe Ratio: {sharpe:.4f}")
                print(f"   胜率: {win_rate:.2%}")
                print(f"   平均收益 (Net): {net_series.mean()*100:.4f}%")
            
            results[model_name] = {
                'final_gross': final_gross,
                'final_net': final_net,
                'num_periods': len(group_ts),
                'timeseries': group_ts
            }
            
            # Save timeseries
            output_file = latest_run / f"{model_name}_top20_nonoverlap_timeseries.csv"
            group_ts.to_csv(output_file, index=False)
            print(f"\n💾 时间序列已保存: {output_file.name}")
            
        except Exception as e:
            print(f"❌ {model_name}: 计算失败 - {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    if results:
        print(f"\n{'='*80}")
        print("📊 累计收益对比总结")
        print(f"{'='*80}")
        print(f"{'模型':<15} {'最终累计收益 (Gross)':<25} {'最终累计收益 (Net)':<25} {'期数':<10}")
        print("-" * 80)
        for model_name, data in results.items():
            print(f"{model_name:<15} {data['final_gross']:>20.2f}% {data['final_net']:>20.2f}% {data['num_periods']:>10}")

if __name__ == "__main__":
    calculate_accumulated_returns()
