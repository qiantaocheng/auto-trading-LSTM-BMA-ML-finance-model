#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Update DIRECT_PREDICT_SNAPSHOT_ID in app.py and direct_predict_ewma_excel.py to the given or latest snapshot."""

import argparse
import re
import sys
from pathlib import Path

def find_latest_snapshot_id(results_dir: Path) -> str | None:
    """Find snapshot_id.txt in the most recent run_* under results_dir."""
    if not results_dir.exists():
        return None
    run_dirs = sorted((results_dir / d.name for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")), reverse=True)
    for run_dir in run_dirs:
        snapshot_file = run_dir / "snapshot_id.txt"
        if snapshot_file.exists():
            return snapshot_file.read_text(encoding="utf-8").strip()
    return None

def update_app_py(project_root: Path, snapshot_id: str) -> bool:
    app_py = project_root / "autotrader" / "app.py"
    if not app_py.exists():
        return False
    text = app_py.read_text(encoding="utf-8")
    pattern = r'(DIRECT_PREDICT_SNAPSHOT_ID\s*=\s*)["\'][^"\']*["\']'
    replacement = rf'\1"{snapshot_id}"'
    new_text, n = re.subn(pattern, replacement, text, count=1)
    if n == 0:
        return False
    app_py.write_text(new_text, encoding="utf-8")
    return True

def update_direct_predict_script(project_root: Path, snapshot_id: str) -> bool:
    script = project_root / "scripts" / "direct_predict_ewma_excel.py"
    if not script.exists():
        return False
    text = script.read_text(encoding="utf-8")
    # Update the fallback in os.environ.get("DIRECT_PREDICT_SNAPSHOT_ID", "old-uuid")
    pattern = r'(os\.environ\.get\s*\(\s*["\']DIRECT_PREDICT_SNAPSHOT_ID["\']\s*,\s*)["\'][^"\']*["\']'
    replacement = rf'\1"{snapshot_id}"'
    new_text, n = re.subn(pattern, replacement, text, count=1)
    if n == 0:
        return False
    script.write_text(new_text, encoding="utf-8")
    return True

def main() -> int:
    parser = argparse.ArgumentParser(description="Update direct prediction snapshot ID in app.py and direct_predict_ewma_excel.py")
    parser.add_argument("snapshot_id", nargs="?", default=None, help="Snapshot ID (default: read from latest results/full_dataset_training/run_*)")
    parser.add_argument("--results-dir", type=str, default=None, help="Results dir to search for snapshot_id.txt (default: results/full_dataset_training)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    snapshot_id = args.snapshot_id
    if not snapshot_id:
        results_dir = Path(args.results_dir) if args.results_dir else project_root / "results" / "full_dataset_training"
        snapshot_id = find_latest_snapshot_id(results_dir)
        if not snapshot_id:
            print(f"No snapshot_id.txt found under {results_dir}. Run full training first or pass snapshot_id.")
            return 1
        print(f"Using latest snapshot from {results_dir}: {snapshot_id}")

    ok_app = update_app_py(project_root, snapshot_id)
    ok_script = update_direct_predict_script(project_root, snapshot_id)
    print(f"autotrader/app.py DIRECT_PREDICT_SNAPSHOT_ID: {'updated' if ok_app else 'not found/unchanged'}")
    print(f"scripts/direct_predict_ewma_excel.py default snapshot: {'updated' if ok_script else 'not found/unchanged'}")
    if ok_app or ok_script:
        print(f"Direct prediction snapshot set to: {snapshot_id}")
        return 0
    print("No files were updated.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
