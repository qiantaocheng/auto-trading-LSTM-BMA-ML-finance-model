#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比训练：有/无 obv_divergence 因子的性能差异
使用 1/5 股票子集进行训练
"""
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def run_training_with_obv_divergence(subset_file: Path, output_dir: Path) -> Dict[str, Any]:
    """运行包含 obv_divergence 的训练"""
    print("="*80)
    print("训练配置: WITH obv_divergence")
    print("="*80)
    
    # 确保使用 T10 factors (horizon >= 10)
    train_cmd = [
        sys.executable,
        "scripts/train_full_dataset.py",
        "--train-data", str(subset_file),
        "--top-n", "50",
        "--horizon-days", "10",  # 确保使用 T10 factors
        "--log-level", "INFO"
    ]
    
    print(f"执行命令: {' '.join(train_cmd)}")
    print(f"输出目录: {output_dir}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            train_cmd,
            cwd=Path(__file__).parent.parent,
            check=True,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time
        
        # 查找训练结果文件
        results = {
            'success': True,
            'elapsed_time': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        print(f"\n[OK] 训练完成! 耗时: {elapsed/60:.1f}分钟")
        return results
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] 训练失败 (耗时: {elapsed/60:.1f}分钟)")
        return {
            'success': False,
            'elapsed_time': elapsed,
            'stdout': e.stdout if hasattr(e, 'stdout') else '',
            'stderr': e.stderr if hasattr(e, 'stderr') else str(e)
        }

def run_training_without_obv_divergence(subset_file: Path, output_dir: Path) -> Dict[str, Any]:
    """运行不包含 obv_divergence 的训练（临时修改因子列表）"""
    print("="*80)
    print("训练配置: WITHOUT obv_divergence")
    print("="*80)
    
    # 需要临时修改 simple_25_factor_engine.py 中的 T10_ALPHA_FACTORS
    # 或者通过环境变量/配置来排除 obv_divergence
    # 最简单的方法：临时修改文件
    
    factor_engine_file = Path(__file__).parent.parent / "bma_models" / "simple_25_factor_engine.py"
    
    # 备份原文件
    backup_file = factor_engine_file.with_suffix('.py.backup_obv_test')
    
    try:
        # 读取原文件
        with open(factor_engine_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 备份
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 修改 T10_ALPHA_FACTORS，移除 obv_divergence
        modified_content = content.replace(
            "'obv_divergence',  # OBV divergence",
            "# 'obv_divergence',  # OBV divergence - TEMPORARILY REMOVED FOR TESTING"
        )
        
        # 写入修改后的文件
        with open(factor_engine_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"✅ 临时修改因子文件: 移除 obv_divergence")
        print(f"   备份文件: {backup_file}")
        
        # 运行训练
        train_cmd = [
            sys.executable,
            "scripts/train_full_dataset.py",
            "--train-data", str(subset_file),
            "--top-n", "50",
            "--horizon-days", "10",
            "--log-level", "INFO"
        ]
        
        print(f"执行命令: {' '.join(train_cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                train_cmd,
                cwd=Path(__file__).parent.parent,
                check=True,
                capture_output=True,
                text=True
            )
            elapsed = time.time() - start_time
            
            results = {
                'success': True,
                'elapsed_time': elapsed,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            print(f"\n[OK] 训练完成! 耗时: {elapsed/60:.1f}分钟")
            
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"\n[ERROR] 训练失败 (耗时: {elapsed/60:.1f}分钟)")
            results = {
                'success': False,
                'elapsed_time': elapsed,
                'stdout': e.stdout if hasattr(e, 'stdout') else '',
                'stderr': e.stderr if hasattr(e, 'stderr') else str(e)
            }
        
        finally:
            # 恢复原文件
            print(f"\n恢复原文件...")
            with open(backup_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            with open(factor_engine_file, 'w', encoding='utf-8') as f:
                f.write(original_content)
            print(f"✅ 文件已恢复")
            
            # 删除备份（可选）
            # backup_file.unlink()
        
        return results
        
    except Exception as e:
        print(f"[ERROR] 文件操作失败: {e}")
        # 尝试恢复
        if backup_file.exists():
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                with open(factor_engine_file, 'w', encoding='utf-8') as f:
                    f.write(original_content)
            except:
                pass
        return {
            'success': False,
            'error': str(e)
        }

def compare_results(with_results: Dict[str, Any], without_results: Dict[str, Any], output_file: Path):
    """对比训练结果"""
    print("\n" + "="*80)
    print("结果对比")
    print("="*80)
    
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'with_obv_divergence': {
            'success': with_results.get('success', False),
            'elapsed_time_minutes': with_results.get('elapsed_time', 0) / 60,
        },
        'without_obv_divergence': {
            'success': without_results.get('success', False),
            'elapsed_time_minutes': without_results.get('elapsed_time', 0) / 60,
        },
        'difference': {
            'time_diff_minutes': (with_results.get('elapsed_time', 0) - without_results.get('elapsed_time', 0)) / 60,
        }
    }
    
    print(f"\n训练时间对比:")
    print(f"  包含 obv_divergence: {comparison['with_obv_divergence']['elapsed_time_minutes']:.1f} 分钟")
    print(f"  不包含 obv_divergence: {comparison['without_obv_divergence']['elapsed_time_minutes']:.1f} 分钟")
    print(f"  时间差: {comparison['difference']['time_diff_minutes']:.1f} 分钟")
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 对比结果已保存: {output_file}")
    
    return comparison

def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    subset_file = project_root / "data" / "factor_exports" / "polygon_factors_all_filtered_clean_final_v2_subset_1_5_tickers.parquet"
    
    if not subset_file.exists():
        print(f"[ERROR] 子集文件不存在: {subset_file}")
        return False
    
    output_dir = project_root / "results" / "obv_divergence_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("OBV_DIVERGENCE 因子对比训练")
    print("="*80)
    print(f"子集文件: {subset_file}")
    print(f"输出目录: {output_dir}")
    print(f"开始时间: {datetime.now()}")
    
    # 实验1: 包含 obv_divergence
    print("\n" + "="*80)
    print("实验 1/2: 包含 obv_divergence")
    print("="*80)
    with_results = run_training_with_obv_divergence(subset_file, output_dir)
    
    # 实验2: 不包含 obv_divergence
    print("\n" + "="*80)
    print("实验 2/2: 不包含 obv_divergence")
    print("="*80)
    without_results = run_training_without_obv_divergence(subset_file, output_dir)
    
    # 对比结果
    comparison_file = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    comparison = compare_results(with_results, without_results, comparison_file)
    
    print("\n" + "="*80)
    print("实验完成")
    print("="*80)
    print(f"结束时间: {datetime.now()}")
    print(f"结果文件: {comparison_file}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
