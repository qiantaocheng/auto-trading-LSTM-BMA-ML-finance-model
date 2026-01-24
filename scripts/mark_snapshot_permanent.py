#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将当前Direct Predict使用的snapshot标记为永久snapshot
"""

import sys
import sqlite3
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def main():
    """将当前snapshot标记为永久"""
    print("=" * 80)
    print("标记Snapshot为永久")
    print("=" * 80)
    
    # 读取当前snapshot ID
    latest_snapshot_file = project_root / "latest_snapshot_id.txt"
    if not latest_snapshot_file.exists():
        print(f"[ERROR] latest_snapshot_id.txt不存在")
        return 1
    
    snapshot_id = latest_snapshot_file.read_text(encoding="utf-8").strip()
    print(f"\n[INFO] 当前Direct Predict使用的Snapshot ID: {snapshot_id}")
    
    # 连接数据库
    db_path = project_root / "data" / "model_registry.db"
    if not db_path.exists():
        print(f"[ERROR] 数据库文件不存在: {db_path}")
        return 1
    
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        
        # 检查snapshot是否存在
        cur.execute("SELECT id, tag, created_at FROM model_snapshots WHERE id = ?", (snapshot_id,))
        result = cur.fetchone()
        
        if not result:
            print(f"[ERROR] Snapshot {snapshot_id} 在数据库中不存在")
            return 1
        
        old_tag = result[1]
        created_at = result[2]
        
        print(f"\n[INFO] 当前Tag: {old_tag}")
        print(f"[INFO] 创建时间: {created_at}")
        
        # 更新tag为永久标记
        new_tag = f"PERMANENT_{old_tag}" if not old_tag.startswith("PERMANENT_") else old_tag
        if new_tag == old_tag:
            print(f"\n[INFO] Snapshot已经是永久标记: {new_tag}")
        else:
            cur.execute(
                "UPDATE model_snapshots SET tag = ? WHERE id = ?",
                (new_tag, snapshot_id)
            )
            conn.commit()
            print(f"\n[OK] 已更新Tag: {old_tag} → {new_tag}")
        
        # 验证更新
        cur.execute("SELECT id, tag FROM model_snapshots WHERE id = ?", (snapshot_id,))
        updated_result = cur.fetchone()
        print(f"\n[验证] Snapshot ID: {updated_result[0]}")
        print(f"[验证] Tag: {updated_result[1]}")
        
        print(f"\n[OK] Snapshot已标记为永久: {snapshot_id}")
        print(f"[OK] Tag: {updated_result[1]}")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] 更新失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()

if __name__ == "__main__":
    sys.exit(main())
