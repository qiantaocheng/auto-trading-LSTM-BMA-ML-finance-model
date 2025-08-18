#!/usr/bin/env python3
"""
优化版训练启动脚本
安全启动2800股票训练
"""

import os
import sys
import psutil
import logging
from pathlib import Path

def check_system_requirements():
    """检查系统需求"""
    print("=== 系统需求检查 ===")
    
    # 检查内存
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    total_gb = memory.total / (1024**3)
    
    print(f"总内存: {total_gb:.1f} GB")
    print(f"可用内存: {available_gb:.1f} GB")
    
    if available_gb < 2.5:
        print("❌ 警告: 可用内存不足，建议至少3GB")
        return False
    else:
        print("✅ 内存检查通过")
    
    # 检查磁盘空间
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / (1024**3)
    print(f"可用磁盘空间: {free_gb:.1f} GB")
    
    if free_gb < 5.0:
        print("❌ 警告: 磁盘空间不足，建议至少5GB")
        return False
    else:
        print("✅ 磁盘空间检查通过")
    
    # 检查CPU
    cpu_count = psutil.cpu_count()
    print(f"CPU核心数: {cpu_count}")
    
    if cpu_count < 4:
        print("⚠️ 建议: CPU核心数较少，训练时间可能较长")
    else:
        print("✅ CPU检查通过")
    
    return True

def setup_environment():
    """设置环境"""
    print("\n=== 环境设置 ===")
    
    # 设置内存环境变量
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['NUMEXPR_NUM_THREADS'] = '4'
    os.environ['PYTHONHASHSEED'] = '42'
    
    # 创建必要目录
    dirs_to_create = [
        'cache/optimized_bma',
        'cache/training',
        'cache/data',
        'cache/models',
        'logs/progress',
        'result'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ 目录创建: {dir_path}")
    
    print("✅ 环境设置完成")

def get_training_config():
    """获取训练配置"""
    print("\n=== 训练配置 ===")
    
    # 自动配置批次大小
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    if available_gb >= 4.0:
        batch_size = 500
        memory_limit = 3.5
    elif available_gb >= 3.0:
        batch_size = 400
        memory_limit = 2.5
    else:
        batch_size = 300
        memory_limit = 2.0
    
    config = {
        'batch_size': batch_size,
        'memory_limit_gb': memory_limit,
        'enable_caching': True,
        'start_date': '2022-01-01',
        'end_date': '2024-12-31'
    }
    
    print(f"批次大小: {config['batch_size']}")
    print(f"内存限制: {config['memory_limit_gb']} GB")
    print(f"启用缓存: {config['enable_caching']}")
    print(f"数据范围: {config['start_date']} 到 {config['end_date']}")
    
    return config

def main():
    """主函数"""
    print("🚀 优化版BMA训练启动器")
    print("=" * 50)
    
    # 检查系统需求
    if not check_system_requirements():
        print("\n❌ 系统需求不满足，建议:")
        print("1. 释放更多内存")
        print("2. 清理磁盘空间")
        print("3. 关闭其他程序")
        
        response = input("\n是否继续运行? (y/N): ")
        if response.lower() != 'y':
            print("训练已取消")
            return
    
    # 设置环境
    setup_environment()
    
    # 获取配置
    config = get_training_config()
    
    # 确认启动
    print(f"\n准备训练最多2800股票...")
    print("预估训练时间: 1-3小时 (取决于系统性能)")
    
    response = input("是否开始训练? (y/N): ")
    if response.lower() != 'y':
        print("训练已取消")
        return
    
    try:
        print("\n🎯 开始训练...")
        
        # 导入并运行优化训练器
        from bma_models.optimized_bma_trainer import OptimizedBMATrainer
        
        trainer = OptimizedBMATrainer(
            batch_size=config['batch_size'],
            memory_limit_gb=config['memory_limit_gb'],
            enable_caching=config['enable_caching']
        )
        
        # 加载股票清单
        universe = trainer.load_universe("stocks.txt")
        
        # 限制股票数量
        if len(universe) > 2800:
            universe = universe[:2800]
            print(f"限制训练股票数量为: 2800")
        else:
            print(f"训练股票数量: {len(universe)}")
        
        # 开始训练
        results = trainer.train_universe(
            universe=universe,
            start_date=config['start_date'],
            end_date=config['end_date']
        )
        
        # 保存结果
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"result/optimized_bma_results_{timestamp}.json"
        trainer.save_results(results, output_file)
        
        # 显示成功信息
        print("\n🎉 训练成功完成!")
        print("=" * 50)
        print(f"总股票数: {results.get('total_stocks', 0)}")
        print(f"成功预测: {len(results.get('predictions', {}))}")
        print(f"总训练时间: {results.get('total_training_time', 0) / 60:.1f} 分钟")
        print(f"结果文件: {output_file}")
        
        # 显示性能统计
        if 'optimization_stats' in results:
            stats = results['optimization_stats']
            if 'memory_stats' in stats:
                mem_stats = stats['memory_stats']
                print(f"内存清理次数: {mem_stats.get('cleanup_count', 0)}")
                print(f"内存警告次数: {mem_stats.get('warning_count', 0)}")
            
            if 'cache_stats' in stats:
                cache_stats = stats['cache_stats']
                if 'cache_hit_rate_percent' in cache_stats:
                    print(f"缓存命中率: {cache_stats['cache_hit_rate_percent']:.1f}%")
        
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断训练")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            trainer.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()