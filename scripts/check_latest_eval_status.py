#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from datetime import datetime

run_dir = Path(r"d:\trade\results\t10_time_split_80_20_final\run_20260122_030445")

print("=" * 70)
print("排除SBET后的80/20评估状态")
print("=" * 70)
print(f"\n运行目录: {run_dir.name}")
print(f"创建时间: {datetime.fromtimestamp(run_dir.stat().st_ctime)}")
print(f"修改时间: {datetime.fromtimestamp(run_dir.stat().st_mtime)}")

# 检查关键文件
files_to_check = {
    "snapshot_id.txt": "训练完成",
    "oos_metrics.json": "回测完成",
    "oos_topn_vs_benchmark_all_models.csv": "结果文件生成",
    "report_df.csv": "报告生成"
}

print("\n文件状态:")
for filename, description in files_to_check.items():
    filepath = run_dir / filename
    if filepath.exists():
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        print(f"  [OK] {filename:40s} - {description} ({mtime.strftime('%H:%M:%S')})")
        if filename == "snapshot_id.txt":
            print(f"       内容: {filepath.read_text(encoding='utf-8').strip()}")
        elif filename == "oos_metrics.json":
            try:
                data = json.loads(filepath.read_text(encoding='utf-8'))
                print(f"       IC: {data.get('IC', 0):.4f}, Rank IC: {data.get('Rank_IC', 0):.4f}")
            except:
                pass
    else:
        print(f"  [PENDING] {filename:40s} - {description}")

# 统计文件
all_files = list(run_dir.glob("*"))
csv_files = list(run_dir.glob("*.csv"))
png_files = list(run_dir.glob("*.png"))
json_files = list(run_dir.glob("*.json"))

print(f"\n文件统计:")
print(f"  总文件数: {len(all_files)}")
print(f"  CSV文件: {len(csv_files)}")
print(f"  PNG文件: {len(png_files)}")
print(f"  JSON文件: {len(json_files)}")

if len(csv_files) > 0 or len(png_files) > 0:
    print("\n[OK] 评估已完成！")
    print(f"\n结果文件:")
    for f in sorted(csv_files + png_files)[:10]:
        print(f"  - {f.name}")
else:
    print("\n[INFO] 评估仍在进行中...")
    print("  当前状态: 训练完成，回测进行中")
    print("  预计还需要: 10-30分钟")

print("\n" + "=" * 70)
