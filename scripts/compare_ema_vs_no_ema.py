#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比EMA vs 无EMA的结果

运行两次评估：
1. 使用EMA（默认Top300 filter）
2. 不使用EMA（原始分数）

然后对比结果并生成对比报告
"""

import subprocess
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse


def run_evaluation(use_ema: bool, ema_top_n: int, output_suffix: str, base_args: list) -> Path:
    """运行一次评估"""
    cmd = [
        sys.executable,
        "scripts/time_split_80_20_oos_eval.py"
    ] + base_args
    
    if use_ema:
        cmd.extend(["--ema-top-n", str(ema_top_n)])
        print(f"\n{'='*80}")
        print(f"运行评估：使用EMA (Top{ema_top_n} filter)")
        print(f"{'='*80}")
    else:
        cmd.extend(["--ema-top-n", "-1"])  # -1 means disable EMA
        print(f"\n{'='*80}")
        print(f"运行评估：不使用EMA（原始分数）")
        print(f"{'='*80}")
    
    # 修改output-dir
    output_dir_idx = None
    for i, arg in enumerate(cmd):
        if arg == "--output-dir":
            output_dir_idx = i + 1
            break
    
    if output_dir_idx:
        original_dir = cmd[output_dir_idx]
        cmd[output_dir_idx] = f"{original_dir}_{output_suffix}"
    
    print(f"命令: {' '.join(cmd)}")
    print()
    
    # 运行评估
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 评估失败:")
        print(result.stderr)
        return None
    
    # 找到输出目录
    if output_dir_idx:
        output_dir = Path(cmd[output_dir_idx])
        # 找到最新的run目录
        run_dirs = sorted(output_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if run_dirs:
            return run_dirs[0]
    
    return None


def load_metrics(run_dir: Path, model_name: str) -> dict:
    """加载某个模型的指标"""
    metrics = {}
    
    # 加载bucket returns (Overlap metrics)
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
    
    # 加载non-overlap accumulated returns
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
                # Calculate if not present
                cum_returns = (1 + period_returns).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns / running_max - 1) * 100
                metrics['nonoverlap']['max_drawdown'] = drawdown.min()
            
            if period_returns.std() > 0:
                metrics['nonoverlap']['sharpe'] = (period_returns.mean() / period_returns.std()) * (25 ** 0.5)
    
    return metrics


def generate_comparison_report(ema_dir: Path, no_ema_dir: Path, models: list, output_file: Path):
    """生成对比报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("EMA vs 无EMA 结果对比报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    comparison_data = {}
    
    for model_name in models:
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"【{model_name.upper()}】")
        report_lines.append("=" * 80)
        
        ema_metrics = load_metrics(ema_dir, model_name)
        no_ema_metrics = load_metrics(no_ema_dir, model_name)
        
        comparison_data[model_name] = {
            'with_ema': ema_metrics,
            'without_ema': no_ema_metrics
        }
        
        # Overlap metrics对比
        if 'overlap' in ema_metrics and 'overlap' in no_ema_metrics:
            report_lines.append("\n【Overlap指标对比（每日观测）】")
            report_lines.append("-" * 80)
            
            ema_o = ema_metrics['overlap']
            no_ema_o = no_ema_metrics['overlap']
            
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
                improvement = "✅" if (key in ['mean_return', 'median_return', 'win_rate', 'sharpe'] and diff > 0) or \
                                  (key == 'std_return' and diff < 0) else "❌"
                
                report_lines.append(f"{label:<20} {ema_val:>15.4f} {no_ema_val:>15.4f} {diff:>+15.4f} {improvement:>10}")
        
        # Non-Overlap metrics对比
        if 'nonoverlap' in ema_metrics and 'nonoverlap' in no_ema_metrics:
            report_lines.append("\n【Non-Overlap指标对比（10天期间）】")
            report_lines.append("-" * 80)
            
            ema_no = ema_metrics['nonoverlap']
            no_ema_no = no_ema_metrics['nonoverlap']
            
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
                    
                    # 判断改善：收益类指标越大越好，回撤越小越好
                    if key == 'max_drawdown':
                        improvement = "✅" if diff > 0 else "❌"  # 回撤越小（绝对值）越好
                    elif key == 'std_period_return':
                        improvement = "✅" if diff < 0 else "❌"  # 标准差越小越好
                    else:
                        improvement = "✅" if diff > 0 else "❌"  # 其他指标越大越好
                    
                    report_lines.append(f"{label:<25} {ema_val:>15.4f} {no_ema_val:>15.4f} {diff:>+15.4f} {improvement:>10}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("【总结】")
    report_lines.append("-" * 80)
    report_lines.append("✅ 表示EMA版本更好")
    report_lines.append("❌ 表示无EMA版本更好")
    report_lines.append("=" * 80)
    
    # 保存报告
    output_file.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n✅ 对比报告已保存: {output_file}")
    
    # 保存JSON数据
    json_file = output_file.parent / f"{output_file.stem}.json"
    json_file.write_text(json.dumps(comparison_data, indent=2, default=str), encoding="utf-8")
    print(f"✅ 对比数据（JSON）已保存: {json_file}")
    
    return comparison_data


def main():
    parser = argparse.ArgumentParser(description="对比EMA vs 无EMA的结果")
    parser.add_argument("--base-args", nargs=argparse.REMAINDER, 
                       help="传递给time_split_80_20_oos_eval.py的基础参数")
    parser.add_argument("--ema-top-n", type=int, default=300,
                       help="EMA Top N参数（默认300）")
    parser.add_argument("--models", nargs="+", default=["catboost", "lambdarank", "ridge_stacking"],
                       help="要对比的模型列表")
    
    args = parser.parse_args()
    
    # 解析base_args，提取output-dir
    base_args = args.base_args or []
    output_dir = None
    for i, arg in enumerate(base_args):
        if arg == "--output-dir" and i + 1 < len(base_args):
            output_dir = base_args[i + 1]
            break
    
    if not output_dir:
        output_dir = "results/ema_comparison"
    
    # 运行两次评估
    print("开始对比评估...")
    
    # 1. 使用EMA
    ema_run_dir = run_evaluation(use_ema=True, ema_top_n=args.ema_top_n, 
                                  output_suffix="with_ema", base_args=base_args)
    
    if not ema_run_dir:
        print("❌ EMA评估失败，无法继续对比")
        return 1
    
    # 2. 不使用EMA
    no_ema_run_dir = run_evaluation(use_ema=False, ema_top_n=0,
                                     output_suffix="no_ema", base_args=base_args)
    
    if not no_ema_run_dir:
        print("❌ 无EMA评估失败，无法继续对比")
        return 1
    
    # 生成对比报告
    output_path = Path(output_dir) / "ema_comparison_report.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    comparison_data = generate_comparison_report(
        ema_run_dir, no_ema_run_dir, args.models, output_path
    )
    
    # 打印报告
    print("\n" + "=" * 80)
    print("对比报告预览:")
    print("=" * 80)
    print(output_path.read_text(encoding="utf-8")[:2000])  # 打印前2000字符
    print("\n... (完整报告请查看文件)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
