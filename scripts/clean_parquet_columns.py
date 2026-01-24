#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理parquet文件中的不需要的列：
- target_new, Close_new
- roa, ebit
- 所有 _new 后缀的列
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Fix module import path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("clean_parquet")


def clean_parquet_columns(parquet_file: str, backup: bool = True):
    """清理parquet文件中的不需要的列"""
    parquet_path = Path(parquet_file)
    
    if not parquet_path.exists():
        logger.error(f"❌ File not found: {parquet_file}")
        return False
    
    logger.info("=" * 80)
    logger.info(f"清理文件: {parquet_file}")
    logger.info("=" * 80)
    
    # 创建备份
    if backup:
        backup_path = parquet_path.with_suffix('.parquet.backup')
        logger.info(f"创建备份: {backup_path}")
        import shutil
        shutil.copy2(parquet_path, backup_path)
        logger.info("✅ 备份完成")
    
    # 加载数据
    logger.info("加载数据...")
    df = pd.read_parquet(parquet_path)
    
    logger.info(f"原始数据形状: {df.shape}")
    logger.info(f"原始列数: {len(df.columns)}")
    logger.info(f"原始列: {list(df.columns)}")
    
    # 确定要删除的列
    columns_to_drop = []
    
    # 1. target_new, Close_new
    if 'target_new' in df.columns:
        columns_to_drop.append('target_new')
    if 'Close_new' in df.columns:
        columns_to_drop.append('Close_new')
    
    # 2. roa, ebit
    if 'roa' in df.columns:
        columns_to_drop.append('roa')
    if 'ebit' in df.columns:
        columns_to_drop.append('ebit')
    
    # 3. 所有 _new 后缀的列
    new_suffix_cols = [col for col in df.columns if col.endswith('_new')]
    columns_to_drop.extend(new_suffix_cols)
    
    # 去重
    columns_to_drop = list(set(columns_to_drop))
    
    if not columns_to_drop:
        logger.info("✅ 没有需要删除的列")
        return True
    
    logger.info(f"\n要删除的列 ({len(columns_to_drop)}):")
    for col in sorted(columns_to_drop):
        logger.info(f"  - {col}")
    
    # 删除列
    df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
    
    logger.info(f"\n清理后数据形状: {df_cleaned.shape}")
    logger.info(f"清理后列数: {len(df_cleaned.columns)}")
    logger.info(f"删除的列数: {len(df.columns) - len(df_cleaned.columns)}")
    
    # 保存
    logger.info(f"\n保存清理后的数据到: {parquet_path}")
    df_cleaned.to_parquet(parquet_path, index=True)
    
    logger.info("✅ 清理完成！")
    
    # 验证
    logger.info("\n验证...")
    df_verify = pd.read_parquet(parquet_path)
    remaining_new_cols = [col for col in df_verify.columns if col.endswith('_new')]
    remaining_bad_cols = [col for col in df_verify.columns if col in ['target_new', 'Close_new', 'roa', 'ebit']]
    
    if remaining_new_cols or remaining_bad_cols:
        logger.warning(f"⚠️ 仍有未删除的列: {remaining_new_cols + remaining_bad_cols}")
        return False
    else:
        logger.info("✅ 验证通过：所有目标列已删除")
        return True


def main():
    parser = argparse.ArgumentParser(description="清理parquet文件中的不需要的列")
    parser.add_argument("--parquet-file", type=str, required=True,
                       help="Parquet文件路径")
    parser.add_argument("--no-backup", action="store_true",
                       help="不创建备份")
    
    args = parser.parse_args()
    
    success = clean_parquet_columns(
        parquet_file=args.parquet_file,
        backup=not args.no_backup
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
