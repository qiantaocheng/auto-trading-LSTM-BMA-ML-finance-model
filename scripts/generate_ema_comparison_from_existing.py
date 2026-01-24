#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从现有结果生成EMA vs 无EMA对比报告
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def load_metrics_from_report(report_file: Path) -> dict:
    """从complete_metrics_report.txt加载指标"""
    if not report_file.exists():
        return {}
    
    content = report_file.read_text(encoding="utf-8")
    metrics = {}
    current_model = None
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 检测模型名称
        if line.startswith('【') and line.endswith('】') and '=' not in line:
            current_model = line.replace('【', '').replace('】', '').lower()
            metrics[current_model] = {'overlap': {}, 'nonoverlap': {}}
        
        # Overlap指标
        if 'Overlap 指标' in line and current_model:
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('【'):
                metric_line = lines[i].strip()
                if '平均收益:' in metric_line:
                    metrics[current_model]['overlap']['mean_return'] = float(metric_line.split(':')[1].replace('%', '').strip())
                elif '中位数收益:' in metric_line:
                    metrics[current_model]['overlap']['median_return'] = float(metric_line.split(':')[1].replace('%', '').strip())
                elif '标准差:' in metric_line:
                    metrics[current_model]['overlap']['std_return'] = float(metric_line.split(':')[1].replace('%', '').strip())
                elif 'Overlap 胜率:' in metric_line:
                    metrics[current_model]['overlap']['win_rate'] = float(metric_line.split(':')[1].replace('%', '').strip())
                elif 'Sharpe Ratio (年化):' in metric_line:
                    metrics[current_model]['overlap']['sharpe'] = float(metric_line.split(':')[1].strip())
                i += 1
            continue
        
        # Non-Overlap指标
        if 'Non-Overlap 指标' in line and current_model:
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('【'):
                metric_line = lines[i].strip()
                if '平均期间收益:' in metric_line:
                    metrics[current_model]['nonoverlap']['mean_period_return'] = float(metric_line.split(':')[1].replace('%', '').strip())
                elif '中位数期间收益:' in metric_line:
                    metrics[current_model]['nonoverlap']['median_period_return'] = float(metric_line.split(':')[1].replace('%', '').strip())
                elif '标准差:' in metric_line and 'Non-Overlap' in lines[i-1]:
                    metrics[current_model]['nonoverlap']['std_period_return'] = float(metric_line.split(':')[1].replace('%', '').strip())
                elif 'Non-Overlap 胜率:' in metric_line:
                    metrics[current_model]['nonoverlap']['win_rate'] = float(metric_line.split(':')[1].replace('%', '').strip())
                elif 'Sharpe Ratio (基于期间):' in metric_line:
                    metrics[current_model]['nonoverlap']['sharpe'] = float(metric_line.split(':')[1].strip())
                elif '累积收益:' in metric_line:
                    metrics[current_model]['nonoverlap']['cumulative_return'] = float(metric_line.split(':')[1].replace('%', '').strip())
                elif '最大回撤:' in metric_line:
                    metrics[current_model]['nonoverlap']['max_drawdown'] = float(metric_line.split(':')[1].replace('%', '').strip())
                elif '年化收益:' in metric_line:
                    metrics[current_model]['nonoverlap']['annualized_return'] = float(metric_line.split(':')[1].replace('%', '').strip())
                i += 1
            continue
        
        i += 1
    
    return metrics


def generate_comparison_report(ema_dir: Path, no_ema_dir: Path, output_file: Path):
    """生成对比报告"""
    ema_report = ema_dir / "complete_metrics_report.txt"
    no_ema_report = no_ema_dir / "complete_metrics_report.txt"
    
    ema_metrics = load_metrics_from_report(ema_report)
    no_ema_metrics = load_metrics_from_report(no_ema_report)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("EMA vs 无EMA 结果对比报告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"EMA结果目录: {ema_dir}")
    report_lines.append(f"无EMA结果目录: {no_ema_dir}")
    report_lines.append("")
    
    models = ['catboost', 'lambdarank', 'ridge_stacking']
    
    for model_name in models:
        if model_name not in ema_metrics or model_name not in no_ema_metrics:
            continue
        
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"【{model_name.upper()}】")
        report_lines.append("=" * 80)
        
        ema_m = ema_metrics[model_name]
        no_ema_m = no_ema_metrics[model_name]
        
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
                improvement = "✅" if (key in ['mean_return', 'median_return', 'win_rate', 'sharpe'] and diff > 0) or \
                                  (key == 'std_return' and diff < 0) else "❌"
                
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
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(report_lines), encoding="utf-8")
    
    return report_lines


if __name__ == "__main__":
    ema_dir = Path("results/t10_time_split_90_10_ewma_train/run_20260120_111847")
    no_ema_dir = Path("results/t10_time_split_90_10_no_ema/run_20260120_143949")
    output_file = Path("results/ema_comparison_report.txt")
    
    report_lines = generate_comparison_report(ema_dir, no_ema_dir, output_file)
    
    print("\n".join(report_lines))
    print(f"\n✅ 对比报告已保存: {output_file}")
