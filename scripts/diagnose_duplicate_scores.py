#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断Direct Predict重复分数问题
"""

import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def main():
    """诊断重复分数问题"""
    print("=" * 80)
    print("Direct Predict重复分数问题诊断")
    print("=" * 80)
    
    # 1. 检查当前snapshot
    print("\n[1] 检查当前Snapshot")
    latest_snapshot_file = project_root / "latest_snapshot_id.txt"
    if latest_snapshot_file.exists():
        snapshot_id = latest_snapshot_file.read_text(encoding="utf-8").strip()
        print(f"  [OK] Snapshot ID: {snapshot_id}")
        
        # 检查snapshot是否存在
        try:
            from bma_models.model_registry import load_manifest
            manifest = load_manifest(snapshot_id)
            print(f"  [OK] Snapshot存在，创建时间: {manifest.get('created_at', 'N/A')}")
            
            # 检查模型文件
            paths = manifest.get('paths', {})
            print(f"\n  [模型文件检查]")
            models_to_check = {
                'elastic_net_pkl': 'ElasticNet',
                'xgb_json': 'XGBoost',
                'catboost_cbm': 'CatBoost',
                'lambdarank_txt': 'LambdaRank',
                'meta_ranker_txt': 'MetaRankerStacker'
            }
            
            for key, name in models_to_check.items():
                path = paths.get(key)
                if path:
                    file_path = Path(path)
                    if file_path.exists():
                        size = file_path.stat().st_size
                        print(f"    [OK] {name}: {file_path.name} ({size:,} bytes)")
                    else:
                        print(f"    [ERROR] {name}: 文件不存在 - {path}")
                else:
                    print(f"    [WARN] {name}: 路径未找到")
        except Exception as e:
            print(f"  [ERROR] 无法加载snapshot: {e}")
    else:
        print(f"  [ERROR] latest_snapshot_id.txt不存在")
        return 1
    
    # 2. 检查监控数据库中的预测记录
    print("\n[2] 检查最近的预测记录")
    try:
        import sqlite3
        db_path = project_root / "data" / "monitoring.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            
            # 检查表是否存在
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='direct_predictions'")
            if cur.fetchone():
                # 获取最近的预测记录
                cur.execute("""
                    SELECT ts, snapshot_id, ticker, score 
                    FROM direct_predictions 
                    ORDER BY ts DESC 
                    LIMIT 50
                """)
                rows = cur.fetchall()
                
                if rows:
                    df = pd.DataFrame(rows, columns=['ts', 'snapshot_id', 'ticker', 'score'])
                    unique_scores = df['score'].nunique()
                    print(f"  [INFO] 最近50条预测记录")
                    print(f"  [INFO] 唯一分数数量: {unique_scores}")
                    print(f"  [INFO] 分数范围: min={df['score'].min():.6f}, max={df['score'].max():.6f}")
                    
                    if unique_scores == 1:
                        print(f"  [ERROR] 所有预测分数相同: {df['score'].iloc[0]:.6f}")
                        print(f"  [ERROR] 这确认了重复分数问题！")
                    else:
                        print(f"  [OK] 分数有变化，问题可能已解决")
                    
                    # 显示前10个不同的分数
                    score_counts = df['score'].value_counts().head(10)
                    print(f"\n  [分数分布] (前10个最常见的分数)")
                    for score, count in score_counts.items():
                        print(f"    {score:.6f}: {count} 次")
                else:
                    print(f"  [INFO] 没有预测记录")
            else:
                print(f"  [INFO] direct_predictions表不存在")
            
            conn.close()
        else:
            print(f"  [INFO] monitoring.db不存在")
    except Exception as e:
        print(f"  [WARN] 检查监控数据库失败: {e}")
    
    # 3. 建议
    print("\n[3] 诊断建议")
    print("  [建议1] 查看Direct Predict的完整日志")
    print("    查找: [SNAPSHOT] CRITICAL: All predictions have the same value")
    print("    查找: [SNAPSHOT] pred_series unique values")
    print("    查找: [SNAPSHOT] LambdaRank non-null values")
    print("    查找: [SNAPSHOT] CatBoost non-null values")
    print("\n  [建议2] 检查第一层模型预测")
    print("    如果第一层预测都相同 -> 问题在第一层模型")
    print("    如果第一层预测不同，但最终预测相同 -> 问题在MetaRankerStacker")
    print("\n  [建议3] 验证特征数据")
    print("    检查传递给predict_with_snapshot的feature_data是否有变化")
    print("\n  [建议4] 尝试重新加载snapshot")
    print("    重启Direct Predict，重新加载模型")
    
    print("\n" + "=" * 80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
