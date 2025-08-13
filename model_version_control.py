#!/usr/bin/env python3
"""
模型版本控制系统 - 生产级模型管理
"""

import os
import json
import joblib
import hashlib
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModelVersionControl:
    """模型版本控制系统"""
    
    def __init__(self, base_dir: str = "models"):
        """
        初始化版本控制系统
        
        Args:
            base_dir: 模型存储基础目录
        """
        self.base_dir = base_dir
        self.registry_file = os.path.join(base_dir, "model_registry.json")
        
        # 创建目录结构
        os.makedirs(base_dir, exist_ok=True)
        
        # 初始化注册表
        self._load_registry()
        
        logger.info(f"模型版本控制系统初始化完成，基础目录: {base_dir}")
    
    def _load_registry(self):
        """加载模型注册表"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                'models': {},
                'latest_version': None,
                'production_version': None,
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_models': 0
                }
            }
    
    def _save_registry(self):
        """保存模型注册表"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
    
    def generate_version(self) -> str:
        """生成新的版本号"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 添加短hash确保唯一性
        hash_suffix = hashlib.md5(f"{timestamp}_{np.random.random()}".encode()).hexdigest()[:6]
        return f"v{timestamp}_{hash_suffix}"
    
    def get_git_info(self) -> Dict[str, str]:
        """获取Git信息"""
        try:
            # 获取当前commit hash
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # 获取分支名
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            # 检查是否有未提交的更改
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            
            return {
                'commit_hash': commit_hash,
                'branch': branch,
                'has_uncommitted_changes': bool(status),
                'status': status[:200] if status else "clean"
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("无法获取Git信息，可能不在Git仓库中")
            return {
                'commit_hash': 'unknown',
                'branch': 'unknown', 
                'has_uncommitted_changes': False,
                'status': 'git_unavailable'
            }
    
    def compute_data_hash(self, data: Optional[pd.DataFrame] = None) -> str:
        """计算数据哈希值"""
        if data is None:
            return "no_data"
        
        try:
            # 使用数据的形状、列名和部分内容计算hash
            data_info = {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'sample_hash': hashlib.md5(
                    str(data.head(100).values.tobytes())[:1000].encode()
                ).hexdigest()[:16]
            }
            
            return hashlib.md5(str(data_info).encode()).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"数据哈希计算失败: {e}")
            return "hash_error"
    
    def save_model(self, model: Any, metadata: Dict[str, Any], 
                   training_data: Optional[pd.DataFrame] = None,
                   model_type: str = "quantitative_model") -> str:
        """
        保存模型和元数据
        
        Args:
            model: 要保存的模型对象
            metadata: 模型元数据
            training_data: 训练数据（用于计算数据hash）
            model_type: 模型类型
            
        Returns:
            版本号
        """
        version = self.generate_version()
        version_dir = os.path.join(self.base_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(version_dir, f"{model_type}.pkl")
        joblib.dump(model, model_path)
        
        # 计算模型文件hash
        with open(model_path, 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()
        
        # 构建完整元数据
        full_metadata = {
            'version': version,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'model_hash': model_hash,
            'model_size_mb': os.path.getsize(model_path) / (1024 * 1024),
            'git_info': self.get_git_info(),
            'data_hash': self.compute_data_hash(training_data),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            'user_metadata': metadata.copy()
        }
        
        # 保存元数据
        metadata_path = os.path.join(version_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)
        
        # 更新注册表
        self.registry['models'][version] = full_metadata
        self.registry['latest_version'] = version
        self.registry['metadata']['total_models'] += 1
        self.registry['metadata']['last_updated'] = datetime.now().isoformat()
        
        self._save_registry()
        
        logger.info(f"模型已保存，版本: {version}")
        logger.info(f"模型路径: {model_path}")
        logger.info(f"模型大小: {full_metadata['model_size_mb']:.2f} MB")
        
        return version
    
    def load_model(self, version: Optional[str] = None, 
                   model_type: str = "quantitative_model") -> tuple:
        """
        加载模型
        
        Args:
            version: 版本号，None表示加载最新版本
            model_type: 模型类型
            
        Returns:
            (model, metadata) 元组
        """
        if version is None:
            version = self.get_latest_version()
        
        if version is None:
            raise ValueError("没有找到任何模型版本")
        
        if version not in self.registry['models']:
            raise ValueError(f"版本 {version} 不存在")
        
        version_dir = os.path.join(self.base_dir, version)
        model_path = os.path.join(version_dir, f"{model_type}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        model = joblib.load(model_path)
        
        # 加载元数据
        metadata = self.registry['models'][version]
        
        logger.info(f"已加载模型版本: {version}")
        
        return model, metadata
    
    def get_latest_version(self) -> Optional[str]:
        """获取最新版本号"""
        return self.registry.get('latest_version')
    
    def get_production_version(self) -> Optional[str]:
        """获取生产版本号"""
        return self.registry.get('production_version')
    
    def set_production_version(self, version: str):
        """设置生产版本"""
        if version not in self.registry['models']:
            raise ValueError(f"版本 {version} 不存在")
        
        self.registry['production_version'] = version
        self._save_registry()
        
        logger.info(f"生产版本已设置为: {version}")
    
    def list_versions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """列出模型版本"""
        versions = []
        for version, metadata in self.registry['models'].items():
            version_info = {
                'version': version,
                'timestamp': metadata.get('timestamp'),
                'model_type': metadata.get('model_type'),
                'model_size_mb': metadata.get('model_size_mb'),
                'performance': metadata.get('user_metadata', {}).get('performance_metrics', {}),
                'is_production': version == self.registry.get('production_version'),
                'is_latest': version == self.registry.get('latest_version')
            }
            versions.append(version_info)
        
        # 按时间戳排序
        versions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return versions[:limit]
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """比较两个版本"""
        if version1 not in self.registry['models']:
            raise ValueError(f"版本 {version1} 不存在")
        if version2 not in self.registry['models']:
            raise ValueError(f"版本 {version2} 不存在")
        
        meta1 = self.registry['models'][version1]
        meta2 = self.registry['models'][version2]
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'timestamp_diff': meta2['timestamp'],  # 较新的版本
            'size_diff_mb': meta2.get('model_size_mb', 0) - meta1.get('model_size_mb', 0),
            'performance_diff': {},
            'git_diff': {
                'commit_changed': meta1.get('git_info', {}).get('commit_hash') != 
                                meta2.get('git_info', {}).get('commit_hash'),
                'branch_changed': meta1.get('git_info', {}).get('branch') != 
                                meta2.get('git_info', {}).get('branch')
            },
            'data_changed': meta1.get('data_hash') != meta2.get('data_hash')
        }
        
        # 比较性能指标
        perf1 = meta1.get('user_metadata', {}).get('performance_metrics', {})
        perf2 = meta2.get('user_metadata', {}).get('performance_metrics', {})
        
        for metric in set(perf1.keys()) | set(perf2.keys()):
            if metric in perf1 and metric in perf2:
                comparison['performance_diff'][metric] = perf2[metric] - perf1[metric]
            elif metric in perf2:
                comparison['performance_diff'][metric] = f"新增: {perf2[metric]}"
            else:
                comparison['performance_diff'][metric] = f"移除: {perf1[metric]}"
        
        return comparison
    
    def delete_version(self, version: str, confirm: bool = False):
        """删除模型版本"""
        if not confirm:
            raise ValueError("请设置 confirm=True 以确认删除操作")
        
        if version not in self.registry['models']:
            raise ValueError(f"版本 {version} 不存在")
        
        if version == self.registry.get('production_version'):
            raise ValueError("不能删除生产版本")
        
        # 删除文件
        version_dir = os.path.join(self.base_dir, version)
        if os.path.exists(version_dir):
            import shutil
            shutil.rmtree(version_dir)
        
        # 更新注册表
        del self.registry['models'][version]
        
        if version == self.registry.get('latest_version'):
            # 重新计算最新版本
            if self.registry['models']:
                latest = max(self.registry['models'].keys(), 
                           key=lambda v: self.registry['models'][v]['timestamp'])
                self.registry['latest_version'] = latest
            else:
                self.registry['latest_version'] = None
        
        self.registry['metadata']['total_models'] -= 1
        self._save_registry()
        
        logger.info(f"版本 {version} 已删除")
    
    def get_version_info(self, version: str) -> Dict[str, Any]:
        """获取版本详细信息"""
        if version not in self.registry['models']:
            raise ValueError(f"版本 {version} 不存在")
        
        return self.registry['models'][version].copy()
    
    def export_model_info(self, output_file: str):
        """导出模型信息到文件"""
        export_data = {
            'registry': self.registry,
            'export_timestamp': datetime.now().isoformat(),
            'export_version': '1.0'
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型信息已导出到: {output_file}")


def example_usage():
    """示例用法"""
    print("📦 模型版本控制系统示例")
    
    # 初始化版本控制
    mvc = ModelVersionControl("test_models")
    
    # 模拟保存模型
    from sklearn.linear_model import LinearRegression
    
    # 创建一个简单模型
    model = LinearRegression()
    
    # 模拟训练数据
    training_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randn(100)
    })
    
    # 模拟性能指标
    metadata = {
        'model_name': 'test_linear_model',
        'training_samples': len(training_data),
        'features': list(training_data.columns[:-1]),
        'performance_metrics': {
            'r2_score': 0.75,
            'mse': 0.25,
            'mae': 0.15
        },
        'hyperparameters': {
            'fit_intercept': True,
            'normalize': False
        },
        'notes': '这是一个测试模型'
    }
    
    # 保存模型
    version = mvc.save_model(
        model=model,
        metadata=metadata,
        training_data=training_data,
        model_type='linear_regression'
    )
    
    print(f"✅ 模型已保存，版本: {version}")
    
    # 列出版本
    versions = mvc.list_versions()
    print(f"\n📋 模型版本列表:")
    for v in versions:
        print(f"  {v['version']}: {v['timestamp']} "
              f"({'生产' if v['is_production'] else '开发'})")
    
    # 加载模型
    loaded_model, loaded_metadata = mvc.load_model()
    print(f"\n📥 已加载最新模型版本: {loaded_metadata['version']}")
    
    return mvc


if __name__ == "__main__":
    example_usage()
