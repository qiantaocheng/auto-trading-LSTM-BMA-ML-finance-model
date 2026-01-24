#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接从CSV文件生成EMA vs 无EMA对比报告
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def load_metrics_from_csv(run_dir: Path, model_name: str) -> dict:
    """从CSV文件加载指标"""
    metrics = {}
    
    # Overlap metrics from bucket_returns.csv
    bucket_file = run_dir / f"{model_name}_bucket_returns.csv"
    if bucket_file.exists():
        df = pd.read_csv(bucket_file)
        if 'top_5_15_return' in df.columns:
            returns = df['top_5_15_return'].dropna() / 100.0
            metrics['overlap'] = {
                'mean_return': returns.mean() * 100,
                'median_return': returns.median() * 100,
                'std_return': returns.std() * 100,
                'win_rate': (returns > 0).mean() * 100,
                'sharpe': (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0,
                'n_obs': len(returns)
            }
    
    # Non-Overlap metrics from accumulated.csv
    nonoverlap_file = run_dir / f"{model_name}_top5_15_rebalance10d_accumulated.csv"
    if nonoverlap_file.exists():
        df = pd.read_csv(nonoverlap_file)
        if 'top_gross_return' in df.columns:
            period_returns = df['top_gross_return']
            metrics['nonoverlap'] = {
                'mean_period_return': period_returns.mean() * 100,
                'median_period_return': period_returns.median() * 100,
                'std_period_return': period_returns.std() * 100,
                'win_rate': (period_returns > 0).mean() * 100,
                'n_periods': len(period_returns)
            }
        
        if 'acc_return' in df.columns:
            final_acc = df['acc_return'].iloc[-1] * 100
            total_days = len(df) * 10
            annualized = ((1 + final_acc/100) ** (252 / total_days) - 1) * 100
            metrics['nonoverlap']['cumulative_return'] = final_acc
            metrics['nonoverlap']['annualized_return'] = annualized
            
            if 'drawdown' in df.columns:
                metrics['nonoverlap']['max_drawdown'] = df['drawdown'].min()
            else:
                cum_returns = (1 + period_returns).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns / running_max - 1) * 100
                metrics['nonoverlap']['max_drawdown'] = drawdown.min()
            
            if period_returns.std() > 0:
                metrics['nonoverlap']['sharpe'] = (period_returns.mean() / period_returns.std()) * (25 ** 0.5)
    
    return metrics


def generate_comparison_report(ema_dir: Path, no_ema_dir: Path, output_file: Path):
    """生成对比报告"""
    ema_metrics_all = {}
    no_ema_metrics_all = {}
    
    models = ['catboost', 'lambdarank', 'ridge_stacking']
    
    for model_name in models:
        ema_metrics_all[model_name] = load_metrics_from_csv(ema_dir, model_name)
        no_ema_metrics_all[model_name] = load_metrics_from_csv(no_ema_dir, model_name)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("EMA vs 无EMA 结果对比报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"EMA结果目录: {ema_dir}")
    report_lines.append(f"无EMA结果目录: {no_ema_dir}")
    report_lines.append("")
    
    for model_name in models:
        if model_name not in ema_metrics_all or model_name not in no_ema_metrics_all:
            continue
        
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"【{model_name.upper()}】")
        report_lines.append("=" * 80)
        
        ema_m = ema_metrics_all[model_name]
        no_ema_m = no_ema_metrics_all[model_name]
        
        # Overlap metrics对比
        if 'overlap' in ema_m and 'overlap' in no_ema_m:
            report_lines.append("\n【Overlap指标对比（每日观测）】")
            report_lines.append("-" * 80)
            
            ema_o = ema_m['overlap']
            no_ema_o = no_ema_m['overlap']
            
            metrics_to_compare = [
                ('平均收益 (%)', 'mean_return'),
                ('中位数收益 (%)', 'median_return'),
                ('标准差 (%)', 'std_return'),
                ('胜率 (%)', 'win_rate'),
                ('Sharpe Ratio', 'sharpe')
            ]
            
            report_lines.append(f"{'指标':<20} {'使用EMA':>15} {'不使用EMA':>15} {'差异':>15} {'改善':>10}")
            report_lines.append("-" * 80)
            
            for label, key in metrics_to_compare:
                ema_val = ema_o.get(key, 0)
                no_ema_val = no_ema_o.get(key, 0)
                diff = ema_val - no_ema_val
                if key in ['mean_return', 'median_return', 'win_rate', 'sharpe']:
                    improvement = "BETTER" if diff > 0 else "WORSE"
                else:  # std_return
                    improvement = "BETTER" if diff < 0 else "WORSE"
                
                report_lines.append(f"{label:<20} {ema_val:>15.4f} {no_ema_val:>15.4f} {diff:>+15.4f} {improvement:>10}")
        
        # Non-Overlap metrics对比
        if 'nonoverlap' in ema_m and 'nonoverlap' in no_ema_m:
            report_lines.append("\n【Non-Overlap指标对比（10天期间）】")
            report_lines.append("-" * 80)
            
            ema_no = ema_m['nonoverlap']
            no_ema_no = no_ema_m['nonoverlap']
            
            metrics_to_compare = [
                ('平均期间收益 (%)', 'mean_period_return'),
                ('中位数期间收益 (%)', 'median_period_return'),
                ('标准差 (%)', 'std_period_return'),
                ('胜率 (%)', 'win_rate'),
                ('累积收益 (%)', 'cumulative_return'),
                ('最大回撤 (%)', 'max_drawdown'),
                ('年化收益 (%)', 'annualized_return'),
                ('Sharpe Ratio', 'sharpe')
            ]
            
            report_lines.append(f"{'指标':<25} {'使用EMA':>15} {'不使用EMA':>15} {'差异':>15} {'改善':>10}")
            report_lines.append("-" * 80)
            
            for label, key in metrics_to_compare:
                if key in ema_no and key in no_ema_no:
                    ema_val = ema_no[key]
                    no_ema_val = no_ema_no[key]
                    diff = ema_val - no_ema_val
                    
                    # 判断改善
                    if key == 'max_drawdown':
                        improvement = "BETTER" if diff > 0 else "WORSE"  # 回撤越小（绝对值）越好
                    elif key == 'std_period_return':
                        improvement = "BETTER" if diff < 0 else "WORSE"  # 标准差越小越好
                    else:
                        improvement = "BETTER" if diff > 0 else "WORSE"  # 其他指标越大越好
                    
                    report_lines.append(f"{label:<25} {ema_val:>15.4f} {no_ema_val:>15.4f} {diff:>+15.4f} {improvement:>10}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("【总结】")
    report_lines.append("-" * 80)
    report_lines.append("BETTER 表示EMA版本更好")
    report_lines.append("WORSE 表示无EMA版本更好")
    report_lines.append("=" * 80)
    
    # 保存报告
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(report_lines), encoding="utf-8")
    
    return report_lines


if __name__ == "__main__":
    ema_dir = Path("results/t10_time_split_90_10_ewma_train/run_20260120_111847")
    no_ema_dir = Path("results/t10_time_split_90_10_no_ema/run_20260120_143949")
    output_file = Path("results/ema_comparison_report.txt")
    
    report_lines = generate_comparison_report(ema_dir, no_ema_dir, output_file)
    
    # 打印报告（避免编码问题）
    for line in report_lines:
        try:
            print(line)
        except:
            print(line.encode('utf-8', errors='ignore').decode('utf-8'))
    
    print(f"\n对比报告已保存: {output_file}")
