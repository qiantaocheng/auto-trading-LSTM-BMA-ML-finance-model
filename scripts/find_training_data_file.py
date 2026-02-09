#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find the data file used for training and prediction"""
import json
from pathlib import Path

# Load snapshot info
snapshot_id = "00428c79-3f69-4846-a414-63e9ee8cc4d9"

print("="*80)
print("FINDING DATA FILE USED FOR TRAINING AND PREDICTION")
print("="*80)

print(f"\nSnapshot ID: {snapshot_id}")

# Check model registry for snapshot info
try:
    from bma_models.model_registry import load_manifest
    manifest = load_manifest(snapshot_id)
    
    print(f"\nSnapshot Manifest:")
    print(f"  Training date: {manifest.get('training_date', 'N/A')}")
    print(f"  Data file: {manifest.get('data_file', 'N/A')}")
    print(f"  Training data path: {manifest.get('training_data_path', 'N/A')}")
    
    if 'data_file' in manifest:
        data_file = manifest['data_file']
        print(f"\n[OK] Data file from manifest: {data_file}")
        
        # Check if file exists
        data_path = Path(data_file)
        if data_path.exists():
            print(f"   File exists: {data_path.absolute()}")
        else:
            print(f"   [WARNING] File does not exist: {data_path.absolute()}")
    
    if 'training_data_path' in manifest:
        train_path = manifest['training_data_path']
        print(f"\n[OK] Training data path from manifest: {train_path}")
        
        train_path_obj = Path(train_path)
        if train_path_obj.exists():
            print(f"   Path exists: {train_path_obj.absolute()}")
        else:
            print(f"   [WARNING] Path does not exist: {train_path_obj.absolute()}")
            
except Exception as e:
    print(f"\nCould not load manifest: {e}")

# Check default data files
print(f"\n" + "="*80)
print("DEFAULT DATA FILES (from time_split_80_20_oos_eval.py)")
print("="*80)

default_files = [
    r"D:\trade\data\factor_exports\polygon_factors_all_2021_2026_T5_final.parquet"
]

for file_path in default_files:
    path = Path(file_path)
    if path.exists():
        print(f"\n[OK] {path.name}")
        print(f"   Path: {path.absolute()}")
        print(f"   Size: {path.stat().st_size / (1024*1024):.2f} MB")
    else:
        print(f"\n[NOT FOUND] {path.name} (not found)")

# Check what files exist in data directory
print(f"\n" + "="*80)
print("FILES IN data/factor_exports/")
print("="*80)

data_dir = Path("data/factor_exports")
if data_dir.exists():
    parquet_files = list(data_dir.glob("*.parquet"))
    print(f"\nFound {len(parquet_files)} parquet files:")
    for f in sorted(parquet_files):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")
else:
    print(f"\n[WARNING] Directory not found: {data_dir.absolute()}")

print("\n" + "="*80)
