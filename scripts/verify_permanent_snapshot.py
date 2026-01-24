#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证永久snapshot状态
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def main():
    """验证永久snapshot状态"""
    print("=" * 80)
    print("永久Snapshot验证")
    print("=" * 80)
    
    # 读取当前snapshot ID
    latest_snapshot_file = project_root / "latest_snapshot_id.txt"
    snapshot_id = latest_snapshot_file.read_text(encoding="utf-8").strip()
    
    print(f"\n[INFO] Direct Predict使用的Snapshot ID: {snapshot_id}")
    
    # 连接数据库
    db_path = project_root / "data" / "model_registry.db"
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        
        # 查询snapshot信息
        cur.execute("SELECT id, tag, created_at FROM model_snapshots WHERE id = ?", (snapshot_id,))
        result = cur.fetchone()
        
        if result:
            snapshot_id_db, tag, created_at = result
            created_time = datetime.fromtimestamp(created_at)
            
            print(f"\n[Snapshot信息]")
            print(f"  ID: {snapshot_id_db}")
            print(f"  Tag: {tag}")
            print(f"  创建时间: {created_time}")
            
            if tag.startswith("PERMANENT_"):
                print(f"\n[OK] Snapshot已标记为永久")
                print(f"[OK] Tag前缀: PERMANENT_")
            else:
                print(f"\n[WARN] Snapshot未标记为永久")
                print(f"[WARN] 当前Tag: {tag}")
        
        # 列出所有永久snapshot
        print(f"\n[所有永久Snapshot]")
        cur.execute("SELECT id, tag, created_at FROM model_snapshots WHERE tag LIKE 'PERMANENT_%' ORDER BY created_at DESC")
        permanent_snapshots = cur.fetchall()
        
        if permanent_snapshots:
            for sid, stag, sat in permanent_snapshots:
                stime = datetime.fromtimestamp(sat)
                is_current = "← 当前使用" if sid == snapshot_id else ""
                print(f"  - {sid[:8]}... | {stag} | {stime} {is_current}")
        else:
            print("  (无)")
        
    finally:
        conn.close()
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
