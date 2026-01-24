#!/usr/bin/env python3
"""Quick script to check training status from log file"""
import os
import glob
from pathlib import Path

log_dir = Path("logs")
log_files = sorted(log_dir.glob("80_20_retrain_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)

if not log_files:
    print("No log files found")
    exit(1)

latest_log = log_files[0]
print(f"Checking: {latest_log.name}")
print("=" * 80)

with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# Check for key indicators
keywords = [
    "CatBoost",
    "FIRST_LAYER",
    "二层",
    "snapshot",
    "训练完成",
    "训练开始",
    "ERROR",
    "CRITICAL"
]

found_lines = []
for i, line in enumerate(lines):
    for keyword in keywords:
        if keyword.lower() in line.lower():
            found_lines.append((i+1, line.strip()))
            break

# Show last 30 relevant lines
print("\nLast 30 relevant log entries:")
print("-" * 80)
for line_num, line in found_lines[-30:]:
    print(f"{line_num:6d}: {line}")

# Check file size and last modified
import datetime
mtime = datetime.datetime.fromtimestamp(latest_log.stat().st_mtime)
size_mb = latest_log.stat().st_size / (1024 * 1024)
print(f"\nLog file: {size_mb:.2f} MB, last modified: {mtime}")

# Check if process is still running
import psutil
python_procs = [p for p in psutil.process_iter(['pid', 'name', 'cmdline']) 
                if p.info['name'] == 'python.exe' and 
                any('time_split_80_20' in str(cmd) for cmd in (p.info.get('cmdline') or []))]
if python_procs:
    print(f"\nTraining process running: PID {python_procs[0].pid}")
else:
    print("\nNo training process found")
