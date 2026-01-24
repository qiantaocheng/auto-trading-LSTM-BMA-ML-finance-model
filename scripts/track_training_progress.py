"""
跟踪训练进度 - 检查当前训练进行到哪一步
"""
import sys
from pathlib import Path
from datetime import datetime
import subprocess

def check_training_progress():
    """检查训练进度"""
    print("="*80)
    print("训练进度跟踪")
    print("="*80)
    
    project_root = Path(__file__).resolve().parent.parent
    
    # 检查最新的训练运行
    train_dir = project_root / "results" / "full_dataset_training"
    if not train_dir.exists():
        print("[WARN] 训练目录不存在")
        return
    
    runs = sorted(train_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not runs:
        print("[WARN] 没有找到训练运行")
        return
    
    latest_run = runs[0]
    print(f"\n最新训练运行: {latest_run.name}")
    print(f"创建时间: {datetime.fromtimestamp(latest_run.stat().st_mtime)}")
    
    # 检查snapshot ID
    snapshot_file = latest_run / "snapshot_id.txt"
    if snapshot_file.exists():
        snapshot_id = snapshot_file.read_text(encoding='utf-8').strip()
        print(f"\n[OK] 训练已完成!")
        print(f"Snapshot ID: {snapshot_id}")
        return
    
    # 检查Python进程
    print("\n[训练进行中]")
    print("检查Python进程...")
    
    try:
        import psutil
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'create_time', 'memory_info']):
            try:
                if 'python' in proc.info['name'].lower():
                    python_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if python_processes:
            print(f"找到 {len(python_processes)} 个Python进程")
            for proc in python_processes[:3]:  # 只显示前3个
                runtime = datetime.now() - datetime.fromtimestamp(proc.info['create_time'])
                mem_mb = proc.info['memory_info'].rss / 1024 / 1024
                print(f"  PID: {proc.info['pid']}, 运行时长: {runtime}, 内存: {mem_mb:.1f} MB")
        else:
            print("  未找到Python进程（可能已结束）")
    except ImportError:
        print("  [INFO] psutil未安装，无法检查进程详情")
        print("  安装方法: pip install psutil")
    
    # 检查训练目录中的文件
    print("\n检查训练目录文件...")
    files = list(latest_run.glob("*"))
    if files:
        print(f"找到 {len(files)} 个文件:")
        for f in files[:10]:  # 只显示前10个
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            print(f"  {f.name}: {mtime}")
    else:
        print("  目录为空（训练可能刚刚开始）")
    
    # 训练阶段说明
    print("\n" + "="*80)
    print("训练阶段说明")
    print("="*80)
    print("""
训练过程包括以下阶段：

1. 数据加载和预处理
   - 读取parquet文件
   - 验证MultiIndex格式
   - 计算特征

2. 第一层模型训练（每个模型6折CV）
   - ElasticNet (6次训练)
   - XGBoost (6次训练)
   - CatBoost (6次训练)
   - LambdaRank (6次训练)

3. 第二层模型训练
   - MetaRankerStacker (Ridge Stacking)
   - LambdaRank Stacker

4. 模型保存
   - 保存所有模型到snapshot
   - 生成snapshot_id.txt

预计总时间: 1.5-2小时（子集数据）
    """)
    
    print("\n" + "="*80)
    print("如何查看详细日志")
    print("="*80)
    print("""
如果训练脚本有输出重定向，可以查看：
- 终端输出
- 日志文件（如果有）

或者使用以下命令检查进程输出：
  Get-Process python | Select-Object Id, StartTime
    """)

if __name__ == "__main__":
    check_training_progress()
