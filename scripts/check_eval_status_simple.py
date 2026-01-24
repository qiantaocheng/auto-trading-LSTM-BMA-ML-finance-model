#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path

run_dir = Path(r"d:\trade\results\t10_time_split_80_20_final\run_20260122_030445")

print("=" * 60)
print("评估状态检查")
print("=" * 60)

# 检查文件
files = {
    "snapshot_id.txt": run_dir / "snapshot_id.txt",
    "oos_metrics.json": run_dir / "oos_metrics.json",
    "oos_topn_vs_benchmark_all_models.csv": run_dir / "oos_topn_vs_benchmark_all_models.csv"
}

for name, path in files.items():
    if path.exists():
        print(f"[OK] {name} 存在")
        if name == "snapshot_id.txt":
            print(f"    内容: {path.read_text(encoding='utf-8').strip()}")
        elif name == "oos_metrics.json":
            data = json.loads(path.read_text(encoding='utf-8'))
            print(f"    IC: {data.get('IC', 0):.4f}")
            print(f"    Rank IC: {data.get('Rank_IC', 0):.4f}")
            print(f"    平均收益: {data.get('avg_top_return', 0)*100:.2f}%")
    else:
        print(f"[PENDING] {name} 不存在")

# 检查CSV文件
csv_files = list(run_dir.glob("*.csv"))
png_files = list(run_dir.glob("*.png"))

print(f"\nCSV文件数量: {len(csv_files)}")
print(f"PNG文件数量: {len(png_files)}")

if len(csv_files) > 0 or len(png_files) > 0:
    print("\n[OK] 评估已完成！")
else:
    print("\n[INFO] 评估仍在进行中（只有snapshot_id.txt）")
