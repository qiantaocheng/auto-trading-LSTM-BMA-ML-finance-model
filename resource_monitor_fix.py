#!/usr/bin/env python3
"""
ResourceMonitor修复脚本
解决内存增长误报问题，调整监控阈值到合理水平
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autotrader.resource_monitor import get_resource_monitor

def fix_resource_monitor():
    """修复ResourceMonitor配置，减少误报"""
    print("🔧 修复ResourceMonitor配置...")
    
    # 获取全局资源监控器
    monitor = get_resource_monitor()
    
    # 🎯 调整内存监控阈值（关键修复）
    monitor.adjust_memory_thresholds(
        warning_threshold=0.5,    # 50%增长才警告（原来20%太敏感）
        cleanup_threshold=1.0,    # 100%增长才清理（原来50%太早）
        min_memory_mb=512        # 512MB以下不警告（原来100MB基线太低）
    )
    
    # 🔇 临时抑制内存增长警告（用于测试环境）
    # monitor.suppress_warning('memory_growth', True)
    
    # 📊 调整监控间隔和历史窗口
    monitor._memory_analysis_window = 20  # 20个数据点（10分钟@30s间隔）
    monitor._monitor_interval = 30.0      # 30秒监控间隔
    
    # ⚡ 优化垃圾回收频率
    monitor._gc_threshold = 600  # 10分钟执行一次GC（原来5分钟太频繁）
    
    print("✅ ResourceMonitor配置已优化")
    print(f"   - 内存警告阈值: 50% 增长")
    print(f"   - 内存清理阈值: 100% 增长") 
    print(f"   - 最小警告内存: 512 MB")
    print(f"   - 分析时间窗口: 10 分钟")
    print(f"   - 垃圾回收间隔: 10 分钟")
    
    return monitor

def apply_production_settings():
    """应用生产环境设置"""
    print("🚀 应用生产环境ResourceMonitor设置...")
    
    monitor = get_resource_monitor()
    
    # 生产环境更保守的设置
    monitor.adjust_memory_thresholds(
        warning_threshold=0.8,    # 80%增长才警告
        cleanup_threshold=1.5,    # 150%增长才清理  
        min_memory_mb=1024       # 1GB以下不警告
    )
    
    # 延长监控间隔
    monitor._monitor_interval = 60.0  # 1分钟监控间隔
    monitor._gc_threshold = 1800      # 30分钟GC间隔
    
    print("✅ 生产环境设置已应用")
    
    return monitor

def test_memory_monitoring():
    """测试内存监控功能"""
    print("🧪 测试内存监控...")
    
    monitor = get_resource_monitor()
    
    # 获取统计信息
    stats = monitor.get_stats()
    memory_mb = stats['memory']['rss_mb']
    
    print(f"当前内存使用: {memory_mb:.1f} MB")
    print(f"内存使用百分比: {stats['memory']['percent']:.1f}%")
    
    # 手动执行检查
    monitor._check_memory()
    
    # 检查最近警告
    recent_alerts = monitor.get_recent_alerts(hours=1)
    memory_alerts = [a for a in recent_alerts if a['type'] == 'memory_growth']
    
    print(f"最近1小时内存警告数: {len(memory_alerts)}")
    
    if memory_alerts:
        latest = memory_alerts[-1]
        print(f"最新警告: {latest['data']}")
    else:
        print("✅ 无内存增长警告（正常）")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ResourceMonitor修复工具")
    parser.add_argument("--mode", choices=["fix", "production", "test"], 
                       default="fix", help="运行模式")
    
    args = parser.parse_args()
    
    if args.mode == "fix":
        fix_resource_monitor()
    elif args.mode == "production":
        apply_production_settings()
    elif args.mode == "test":
        test_memory_monitoring()
    
    print("🎯 ResourceMonitor修复完成！")