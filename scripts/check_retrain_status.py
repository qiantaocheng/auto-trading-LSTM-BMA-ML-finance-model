#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查80/20时间分割重新训练状态
"""

import json
from pathlib import Path
from datetime import datetime

# 查找最新的运行目录
output_base = Path(r"d:\trade\results\t10_time_split_80_20_final")

if not output_base.exists():
    print(f"[ERROR] 输出目录不存在: {output_base}")
    exit(1)

# 查找所有运行目录
run_dirs = sorted([d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("run_")], 
                  key=lambda x: x.stat().st_mtime, reverse=True)

if not run_dirs:
    print("[INFO] 没有找到运行目录")
    exit(0)

latest_run = run_dirs[0]

print("=" * 80)
print("80/20时间分割重新训练状态")
print("=" * 80)
print(f"\n最新运行目录: {latest_run.name}")
print(f"创建时间: {datetime.fromtimestamp(latest_run.stat().st_ctime).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"修改时间: {datetime.fromtimestamp(latest_run.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

# 检查关键文件
files_to_check = {
    "snapshot_id.txt": "训练完成",
    "oos_metrics.json": "回测完成",
    "oos_topn_vs_benchmark_all_models.csv": "结果文件生成",
    "report_df.csv": "报告生成",
    "所有模型完整指标说明.md": "完整报告生成"
}

print("\n文件状态:")
for filename, description in files_to_check.items():
    filepath = latest_run / filename
    if filepath.exists():
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        print(f"  [OK] {filename:45s} - {description} ({mtime.strftime('%H:%M:%S')})")
        if filename == "snapshot_id.txt":
            try:
                content = filepath.read_text(encoding='utf-8').strip()
                print(f"       快照ID: {content}")
            except:
                pass
        elif filename == "oos_metrics.json":
            try:
                data = json.loads(filepath.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    for model_name, metrics in data.items():
                        if isinstance(metrics, dict):
                            ic = metrics.get('IC', 0)
                            rank_ic = metrics.get('Rank_IC', 0)
                            print(f"       {model_name}: IC={ic:.4f}, Rank IC={rank_ic:.4f}")
            except Exception as e:
                print(f"       解析错误: {e}")
    else:
        print(f"  [PENDING] {filename:45s} - {description}")

# 统计文件
all_files = list(latest_run.glob("*"))
csv_files = list(latest_run.glob("*.csv"))
png_files = list(latest_run.glob("*.png"))
json_files = list(latest_run.glob("*.json"))
md_files = list(latest_run.glob("*.md"))

print(f"\n文件统计:")
print(f"  总文件数: {len(all_files)}")
print(f"  CSV文件: {len(csv_files)}")
print(f"  PNG文件: {len(png_files)}")
print(f"  JSON文件: {len(json_files)}")
print(f"  MD文件: {len(md_files)}")

# 检查是否完成
has_snapshot = (latest_run / "snapshot_id.txt").exists()
has_metrics = (latest_run / "oos_metrics.json").exists()
has_report = (latest_run / "report_df.csv").exists()

if has_snapshot and has_metrics and has_report:
    print("\n[OK] 训练和评估已完成！")
    print(f"\n结果文件位置: {latest_run}")
    print("\n主要结果文件:")
    for f in sorted(csv_files + png_files + md_files)[:10]:
        print(f"  - {f.name}")
elif has_snapshot:
    print("\n[INFO] 训练已完成，回测进行中...")
    print("  预计还需要: 10-30分钟")
else:
    print("\n[INFO] 训练进行中...")
    print("  预计还需要: 30-60分钟")

print("\n" + "=" * 80)
print("\n提示: 运行此脚本查看最新状态:")
print(f"  python scripts/check_retrain_status.py")
