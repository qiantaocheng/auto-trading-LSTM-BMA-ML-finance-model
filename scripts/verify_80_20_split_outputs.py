#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证80/20 split所有功能是否启用并成功产出结果
检查所有输出文件是否都能正确生成
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def verify_80_20_split_outputs(run_dir: Path):
    """验证80/20 split的所有输出文件"""
    print("=" * 80)
    print("80/20 Split 输出文件验证")
    print("=" * 80)
    print(f"\n检查目录: {run_dir}")
    
    if not run_dir.exists():
        print(f"❌ 目录不存在: {run_dir}")
        return False
    
    all_ok = True
    missing_files = []
    
    # 1. 核心报告文件
    print("\n[1] 核心报告文件:")
    core_files = {
        "snapshot_id.txt": "Snapshot ID",
        "report_df.csv": "核心报告（所有模型指标）",
        "results_summary_for_word_doc.json": "JSON格式结果摘要",
        "complete_metrics_report.txt": "完整指标报告（文本）",
        "oos_metrics.csv": "OOS指标（CSV）",
        "oos_metrics.json": "OOS指标（JSON）",
        "oos_topn_vs_benchmark_all_models.csv": "所有模型OOS Top N vs基准",
    }
    
    for filename, description in core_files.items():
        filepath = run_dir / filename
        if filepath.exists():
            print(f"  ✅ {filename} - {description}")
            # 验证文件内容
            if filename.endswith('.csv'):
                try:
                    df = pd.read_csv(filepath)
                    print(f"     行数: {len(df)}, 列数: {len(df.columns)}")
                except Exception as e:
                    print(f"     ⚠️ 读取失败: {e}")
                    all_ok = False
            elif filename.endswith('.json'):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"     键数: {len(data)}")
                except Exception as e:
                    print(f"     ⚠️ 读取失败: {e}")
                    all_ok = False
        else:
            print(f"  ❌ {filename} - {description} (缺失)")
            missing_files.append(filename)
            all_ok = False
    
    # 2. 检查report_df.csv的内容
    print("\n[2] report_df.csv 内容验证:")
    report_df_path = run_dir / "report_df.csv"
    if report_df_path.exists():
        try:
            report_df = pd.read_csv(report_df_path)
            print(f"  ✅ 成功读取report_df.csv")
            print(f"     模型数量: {len(report_df)}")
            print(f"     列: {list(report_df.columns)}")
            
            # 检查必需的列
            required_cols = ['Model', 'IC', 'Rank_IC', 'MSE', 'MAE', 'R2']
            missing_cols = [col for col in required_cols if col not in report_df.columns]
            if missing_cols:
                print(f"     ⚠️ 缺少必需的列: {missing_cols}")
                all_ok = False
            else:
                print(f"     ✅ 所有必需的列都存在")
            
            # 检查HAC修正列
            hac_cols = ['IC_tstat', 'IC_se_hac', 'Rank_IC_tstat', 'Rank_IC_se_hac']
            missing_hac = [col for col in hac_cols if col not in report_df.columns]
            if missing_hac:
                print(f"     ⚠️ 缺少HAC修正列: {missing_hac}")
            else:
                print(f"     ✅ HAC修正列都存在")
                
        except Exception as e:
            print(f"  ❌ 读取report_df.csv失败: {e}")
            all_ok = False
    else:
        print(f"  ❌ report_df.csv不存在")
        all_ok = False
    
    # 3. 检查模型特定文件
    print("\n[3] 模型特定文件:")
    
    # 从report_df获取模型列表
    models = []
    if report_df_path.exists():
        try:
            report_df = pd.read_csv(report_df_path)
            models = report_df['Model'].unique().tolist()
        except:
            pass
    
    # 如果没有从report_df获取到，尝试从目录中推断
    if not models:
        # 查找所有模型特定的CSV文件
        for csv_file in run_dir.glob("*_top20_timeseries.csv"):
            model_name = csv_file.name.replace("_top20_timeseries.csv", "")
            models.append(model_name)
    
    if not models:
        print("  ⚠️ 无法确定模型列表，跳过模型特定文件检查")
    else:
        print(f"  找到 {len(models)} 个模型: {models}")
        
        model_file_patterns = {
            "top20_timeseries.csv": "Top 20时间序列",
            "top30_nonoverlap_timeseries.csv": "Top 30非重叠时间序列",
            "top5_15_rebalance10d_accumulated.csv": "Top 5-15累计收益",
            "bucket_returns.csv": "分桶收益数据",
            "bucket_summary.csv": "分桶摘要",
        }
        
        model_image_patterns = {
            "top20_vs_qqq.png": "Top 20 vs QQQ对比图",
            "top20_vs_qqq_cumulative.png": "累计收益对比图",
            "bucket_returns_period.png": "分桶收益期间图",
            "bucket_returns_cumulative.png": "分桶累计收益图",
            "top5_15_rebalance10d_accumulated.png": "Top 5-15累计收益图",
        }
        
        for model_name in models:
            print(f"\n  [{model_name}]")
            
            # CSV文件
            for pattern, description in model_file_patterns.items():
                filename = f"{model_name}_{pattern}"
                filepath = run_dir / filename
                if filepath.exists():
                    print(f"    ✅ {filename} - {description}")
                    # 验证CSV内容
                    try:
                        df = pd.read_csv(filepath)
                        print(f"       行数: {len(df)}, 列数: {len(df.columns)}")
                    except Exception as e:
                        print(f"       ⚠️ 读取失败: {e}")
                else:
                    print(f"    ❌ {filename} - {description} (缺失)")
                    missing_files.append(filename)
                    all_ok = False
            
            # PNG文件
            for pattern, description in model_image_patterns.items():
                filename = f"{model_name}_{pattern}"
                filepath = run_dir / filename
                if filepath.exists():
                    file_size = filepath.stat().st_size
                    print(f"    ✅ {filename} - {description} ({file_size} bytes)")
                else:
                    print(f"    ❌ {filename} - {description} (缺失)")
                    missing_files.append(filename)
                    all_ok = False
    
    # 4. 验证results_summary_for_word_doc.json的结构
    print("\n[4] results_summary_for_word_doc.json 结构验证:")
    summary_path = run_dir / "results_summary_for_word_doc.json"
    if summary_path.exists():
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            print(f"  ✅ 成功读取JSON摘要")
            
            # 检查metadata
            if 'metadata' in summary:
                metadata = summary['metadata']
                print(f"     ✅ metadata存在")
                print(f"        snapshot_id: {metadata.get('snapshot_id', 'N/A')}")
                print(f"        train_start: {metadata.get('train_start', 'N/A')}")
                print(f"        test_start: {metadata.get('test_start', 'N/A')}")
                print(f"        hac_correction_applied: {metadata.get('hac_correction_applied', False)}")
                print(f"        hac_method: {metadata.get('hac_method', 'N/A')}")
            else:
                print(f"     ⚠️ metadata缺失")
                all_ok = False
            
            # 检查每个模型的指标
            model_keys = [k for k in summary.keys() if k != 'metadata']
            print(f"     模型数量: {len(model_keys)}")
            
            for model_key in model_keys:
                model_data = summary[model_key]
                if 'metrics' in model_data:
                    metrics = model_data['metrics']
                    print(f"     [{model_key}] metrics: IC={metrics.get('IC', 'N/A'):.4f}, Rank_IC={metrics.get('Rank_IC', 'N/A'):.4f}")
                    
                    # 检查HAC统计量
                    if 'IC_tstat' in metrics and 'IC_se_hac' in metrics:
                        print(f"              HAC: t-stat={metrics.get('IC_tstat', 'N/A'):.2f}, SE={metrics.get('IC_se_hac', 'N/A'):.6f}")
                    else:
                        print(f"              ⚠️ HAC统计量缺失")
                else:
                    print(f"     [{model_key}] ⚠️ metrics缺失")
                    all_ok = False
                    
        except Exception as e:
            print(f"  ❌ 读取JSON摘要失败: {e}")
            all_ok = False
    else:
        print(f"  ❌ results_summary_for_word_doc.json不存在")
        all_ok = False
    
    # 5. 验证complete_metrics_report.txt
    print("\n[5] complete_metrics_report.txt 验证:")
    report_txt_path = run_dir / "complete_metrics_report.txt"
    if report_txt_path.exists():
        try:
            with open(report_txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"  ✅ 成功读取完整指标报告")
            print(f"     文件大小: {len(content)} 字符")
            print(f"     行数: {len(content.splitlines())}")
            
            # 检查关键内容
            if "Overlap 指标" in content:
                print(f"     ✅ 包含Overlap指标")
            else:
                print(f"     ⚠️ 缺少Overlap指标")
            
            if "Non-Overlap 指标" in content:
                print(f"     ✅ 包含Non-Overlap指标")
            else:
                print(f"     ⚠️ 缺少Non-Overlap指标")
                
        except Exception as e:
            print(f"  ❌ 读取完整指标报告失败: {e}")
            all_ok = False
    else:
        print(f"  ❌ complete_metrics_report.txt不存在")
        all_ok = False
    
    # 6. 总结
    print("\n" + "=" * 80)
    print("[总结]")
    print("=" * 80)
    
    if all_ok:
        print("✅ 所有核心文件都存在且可读取")
    else:
        print("⚠️ 发现以下问题:")
        if missing_files:
            print(f"  缺失文件 ({len(missing_files)}个):")
            for f in missing_files[:20]:  # 只显示前20个
                print(f"    - {f}")
            if len(missing_files) > 20:
                print(f"    ... 还有 {len(missing_files) - 20} 个文件缺失")
    
    return all_ok

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="验证80/20 split的输出文件")
    parser.add_argument("--run-dir", type=str, required=True, help="运行目录路径")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    success = verify_80_20_split_outputs(run_dir)
    
    sys.exit(0 if success else 1)
