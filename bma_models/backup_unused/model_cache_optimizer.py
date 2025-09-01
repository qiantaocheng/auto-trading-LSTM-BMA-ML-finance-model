#!/usr/bin/env python3
"""
模型缓存优化器
智能管理训练好的模型，避免重复训练
"""

import gc
import os
import pickle
import hashlib
import logging
import sqlite3
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ModelCacheEntry:
    """模型缓存条目"""
    model_id: str
    model_type: str
    features_hash: str
    data_hash: str
    hyperparameters_hash: str
    file_path: str
    creation_time: float
    last_access_time: float
    access_count: int
    file_size_bytes: int
    performance_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ModelCacheOptimizer:
    """模型缓存优化器"""
    
    def __init__(self,
                 cache_dir: str = "cache/models",
                 max_cache_size_gb: float = 2.0,
                 max_model_age_days: int = 30,
                 enable_compression: bool = True,
                 cleanup_interval_hours: float = 6.0):
        """
        初始化模型缓存优化器
        
        Args:
            cache_dir: 缓存目录
            max_cache_size_gb: 最大缓存大小(GB)
            max_model_age_days: 模型最大保存天数
            enable_compression: 启用压缩
            cleanup_interval_hours: 清理间隔(小时)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.max_model_age_seconds = max_model_age_days * 24 * 3600
        self.enable_compression = enable_compression
        self.cleanup_interval_seconds = cleanup_interval_hours * 3600
        
        # 缓存数据库
        self.db_path = self.cache_dir / "model_cache.db"
        self._init_database()
        
        # 内存缓存
        self.memory_cache: Dict[str, Any] = {}
        self.memory_cache_metadata: Dict[str, ModelCacheEntry] = {}
        self.max_memory_models = 5  # 内存中最多保存5个模型
        
        # 统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_saves = 0
        self.cache_evictions = 0
        
        # 清理线程
        self.cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()
        
    def _init_database(self):
        """初始化缓存数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_cache (
                    model_id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    features_hash TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    hyperparameters_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    creation_time REAL NOT NULL,
                    last_access_time REAL NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    file_size_bytes INTEGER NOT NULL,
                    performance_score REAL,
                    metadata TEXT
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_type ON model_cache(model_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_features_hash ON model_cache(features_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_access ON model_cache(last_access_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance ON model_cache(performance_score)")
    
    def _generate_model_id(self, 
                          model_type: str,
                          features_hash: str,
                          data_hash: str,
                          hyperparameters_hash: str) -> str:
        """生成模型ID"""
        combined = f"{model_type}_{features_hash}_{data_hash}_{hyperparameters_hash}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _hash_object(self, obj: Any) -> str:
        """计算对象哈希"""
        if isinstance(obj, pd.DataFrame):
            # 对DataFrame计算哈希
            return hashlib.md5(
                pd.util.hash_pandas_object(obj, index=True).values.tobytes()
            ).hexdigest()[:16]
        elif isinstance(obj, np.ndarray):
            # 对numpy数组计算哈希
            return hashlib.md5(obj.tobytes()).hexdigest()[:16]
        elif isinstance(obj, dict):
            # 对字典计算哈希（排序后）
            sorted_items = sorted(obj.items())
            return hashlib.md5(str(sorted_items).encode()).hexdigest()[:16]
        else:
            # 对其他对象计算哈希
            return hashlib.md5(str(obj).encode()).hexdigest()[:16]
    
    def _save_model_to_disk(self, model: Any, model_id: str) -> str:
        """将模型保存到磁盘"""
        filename = f"model_{model_id}.pkl"
        if self.enable_compression:
            filename += ".gz"
        
        file_path = self.cache_dir / filename
        
        try:
            if self.enable_compression:
                import gzip
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"保存模型到磁盘失败: {e}")
            raise
    
    def _load_model_from_disk(self, file_path: str) -> Any:
        """从磁盘加载模型"""
        try:
            if file_path.endswith('.gz'):
                import gzip
                with gzip.open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"从磁盘加载模型失败: {e}")
            raise
    
    def cache_model(self,
                   model: Any,
                   model_type: str,
                   features: Union[pd.DataFrame, np.ndarray],
                   training_data: Union[pd.DataFrame, np.ndarray],
                   hyperparameters: Dict[str, Any],
                   performance_score: Optional[float] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """缓存模型"""
        
        # 计算哈希
        features_hash = self._hash_object(features)
        data_hash = self._hash_object(training_data)
        hyperparameters_hash = self._hash_object(hyperparameters)
        
        # 生成模型ID
        model_id = self._generate_model_id(
            model_type, features_hash, data_hash, hyperparameters_hash
        )
        
        # 检查是否已存在
        if self._model_exists(model_id):
            logger.debug(f"模型已存在于缓存: {model_id}")
            return model_id
        
        try:
            # 保存到磁盘
            file_path = self._save_model_to_disk(model, model_id)
            file_size = os.path.getsize(file_path)
            
            # 创建缓存条目
            cache_entry = ModelCacheEntry(
                model_id=model_id,
                model_type=model_type,
                features_hash=features_hash,
                data_hash=data_hash,
                hyperparameters_hash=hyperparameters_hash,
                file_path=file_path,
                creation_time=time.time(),
                last_access_time=time.time(),
                access_count=0,
                file_size_bytes=file_size,
                performance_score=performance_score,
                metadata=metadata
            )
            
            # 保存到数据库
            self._save_cache_entry(cache_entry)
            
            # 加载到内存缓存
            self._add_to_memory_cache(model_id, model, cache_entry)
            
            self.cache_saves += 1
            logger.info(f"模型已缓存: {model_id} ({file_size / 1024 / 1024:.1f}MB)")
            
            # 检查缓存大小限制
            self._check_cache_size_limit()
            
            return model_id
            
        except Exception as e:
            logger.error(f"缓存模型失败: {e}")
            raise
    
    def get_model(self,
                 model_type: str,
                 features: Union[pd.DataFrame, np.ndarray],
                 training_data: Union[pd.DataFrame, np.ndarray],
                 hyperparameters: Dict[str, Any]) -> Optional[Any]:
        """获取缓存的模型"""
        
        # 计算哈希
        features_hash = self._hash_object(features)
        data_hash = self._hash_object(training_data)
        hyperparameters_hash = self._hash_object(hyperparameters)
        
        # 生成模型ID
        model_id = self._generate_model_id(
            model_type, features_hash, data_hash, hyperparameters_hash
        )
        
        # 检查内存缓存
        if model_id in self.memory_cache:
            self._update_access_time(model_id)
            self.cache_hits += 1
            logger.debug(f"内存缓存命中: {model_id}")
            return self.memory_cache[model_id]
        
        # 检查磁盘缓存
        cache_entry = self._get_cache_entry(model_id)
        if cache_entry is None:
            self.cache_misses += 1
            return None
        
        try:
            # 从磁盘加载模型
            model = self._load_model_from_disk(cache_entry.file_path)
            
            # 加载到内存缓存
            self._add_to_memory_cache(model_id, model, cache_entry)
            
            # 更新访问时间
            self._update_access_time(model_id)
            
            self.cache_hits += 1
            logger.debug(f"磁盘缓存命中: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"加载缓存模型失败: {e}")
            # 如果加载失败，从缓存中删除
            self._remove_cache_entry(model_id)
            self.cache_misses += 1
            return None
    
    def _model_exists(self, model_id: str) -> bool:
        """检查模型是否存在"""
        return self._get_cache_entry(model_id) is not None
    
    def _get_cache_entry(self, model_id: str) -> Optional[ModelCacheEntry]:
        """从数据库获取缓存条目"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM model_cache WHERE model_id = ?",
                    (model_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return ModelCacheEntry(
                        model_id=row[0],
                        model_type=row[1],
                        features_hash=row[2],
                        data_hash=row[3],
                        hyperparameters_hash=row[4],
                        file_path=row[5],
                        creation_time=row[6],
                        last_access_time=row[7],
                        access_count=row[8],
                        file_size_bytes=row[9],
                        performance_score=row[10],
                        metadata=eval(row[11]) if row[11] else None
                    )
                    
        except Exception as e:
            logger.error(f"获取缓存条目失败: {e}")
        
        return None
    
    def _save_cache_entry(self, entry: ModelCacheEntry):
        """保存缓存条目到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_cache 
                    (model_id, model_type, features_hash, data_hash, hyperparameters_hash,
                     file_path, creation_time, last_access_time, access_count, file_size_bytes,
                     performance_score, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.model_id,
                    entry.model_type,
                    entry.features_hash,
                    entry.data_hash,
                    entry.hyperparameters_hash,
                    entry.file_path,
                    entry.creation_time,
                    entry.last_access_time,
                    entry.access_count,
                    entry.file_size_bytes,
                    entry.performance_score,
                    str(entry.metadata) if entry.metadata else None
                ))
                
        except Exception as e:
            logger.error(f"保存缓存条目失败: {e}")
            raise
    
    def _update_access_time(self, model_id: str):
        """更新访问时间"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE model_cache 
                    SET last_access_time = ?, access_count = access_count + 1
                    WHERE model_id = ?
                """, (time.time(), model_id))
                
        except Exception as e:
            logger.error(f"更新访问时间失败: {e}")
    
    def _add_to_memory_cache(self, model_id: str, model: Any, cache_entry: ModelCacheEntry):
        """添加到内存缓存"""
        # 检查内存缓存大小
        if len(self.memory_cache) >= self.max_memory_models:
            self._evict_from_memory_cache()
        
        self.memory_cache[model_id] = model
        self.memory_cache_metadata[model_id] = cache_entry
    
    def _evict_from_memory_cache(self):
        """从内存缓存中驱逐模型"""
        if not self.memory_cache:
            return
        
        # 找到最少使用的模型
        lru_model_id = min(
            self.memory_cache_metadata.keys(),
            key=lambda mid: self.memory_cache_metadata[mid].last_access_time
        )
        
        # 从内存缓存中删除
        del self.memory_cache[lru_model_id]
        del self.memory_cache_metadata[lru_model_id]
        
        self.cache_evictions += 1
        logger.debug(f"从内存缓存中驱逐模型: {lru_model_id}")
        
        # 强制垃圾回收
        gc.collect()
    
    def _remove_cache_entry(self, model_id: str):
        """删除缓存条目"""
        try:
            # 从数据库删除
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_path FROM model_cache WHERE model_id = ?",
                    (model_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    file_path = row[0]
                    
                    # 删除文件
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    # 从数据库删除记录
                    conn.execute("DELETE FROM model_cache WHERE model_id = ?", (model_id,))
            
            # 从内存缓存删除
            if model_id in self.memory_cache:
                del self.memory_cache[model_id]
            if model_id in self.memory_cache_metadata:
                del self.memory_cache_metadata[model_id]
                
        except Exception as e:
            logger.error(f"删除缓存条目失败: {e}")
    
    def _check_cache_size_limit(self):
        """检查缓存大小限制"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT SUM(file_size_bytes) FROM model_cache")
                row = cursor.fetchone()
                total_size = row[0] or 0
                
                if total_size > self.max_cache_size_bytes:
                    logger.info(f"缓存大小超限 ({total_size / 1024 / 1024 / 1024:.2f}GB)，开始清理")
                    self._cleanup_old_models()
                    
        except Exception as e:
            logger.error(f"检查缓存大小失败: {e}")
    
    def _cleanup_old_models(self):
        """清理旧模型"""
        try:
            current_time = time.time()
            cutoff_time = current_time - self.max_model_age_seconds
            
            with sqlite3.connect(self.db_path) as conn:
                # 获取需要清理的模型
                cursor = conn.execute("""
                    SELECT model_id, file_path FROM model_cache 
                    WHERE last_access_time < ? 
                    ORDER BY last_access_time ASC
                """, (cutoff_time,))
                
                models_to_remove = cursor.fetchall()
                
                for model_id, file_path in models_to_remove:
                    # 删除文件
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    # 从数据库删除
                    conn.execute("DELETE FROM model_cache WHERE model_id = ?", (model_id,))
                    
                    # 从内存缓存删除
                    if model_id in self.memory_cache:
                        del self.memory_cache[model_id]
                    if model_id in self.memory_cache_metadata:
                        del self.memory_cache_metadata[model_id]
                
                if models_to_remove:
                    logger.info(f"清理了 {len(models_to_remove)} 个过期模型")
                    
        except Exception as e:
            logger.error(f"清理模型失败: {e}")
    
    def _start_cleanup_thread(self):
        """启动清理线程"""
        def cleanup_loop():
            while not self._stop_cleanup.wait(self.cleanup_interval_seconds):
                try:
                    self._cleanup_old_models()
                    self._check_cache_size_limit()
                except Exception as e:
                    logger.error(f"定期清理失败: {e}")
        
        self.cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 总体统计
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_models,
                        SUM(file_size_bytes) as total_size_bytes,
                        AVG(access_count) as avg_access_count,
                        MAX(performance_score) as best_performance
                    FROM model_cache
                """)
                row = cursor.fetchone()
                
                # 按模型类型统计
                cursor = conn.execute("""
                    SELECT model_type, COUNT(*), SUM(file_size_bytes)
                    FROM model_cache
                    GROUP BY model_type
                """)
                type_stats = {row[0]: {'count': row[1], 'size_bytes': row[2]} 
                             for row in cursor.fetchall()}
                
                cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100 
                                if (self.cache_hits + self.cache_misses) > 0 else 0)
                
                return {
                    'total_models': row[0],
                    'total_size_mb': (row[1] or 0) / 1024 / 1024,
                    'average_access_count': row[2] or 0,
                    'best_performance_score': row[3],
                    'memory_cached_models': len(self.memory_cache),
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'cache_hit_rate_percent': cache_hit_rate,
                    'cache_saves': self.cache_saves,
                    'cache_evictions': self.cache_evictions,
                    'type_statistics': type_stats
                }
                
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {'error': str(e)}
    
    def clear_cache(self):
        """清理所有缓存"""
        try:
            # 清理内存缓存
            self.memory_cache.clear()
            self.memory_cache_metadata.clear()
            
            # 清理磁盘文件和数据库
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT file_path FROM model_cache")
                for (file_path,) in cursor.fetchall():
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
                conn.execute("DELETE FROM model_cache")
            
            # 重置统计
            self.cache_hits = 0
            self.cache_misses = 0
            self.cache_saves = 0
            self.cache_evictions = 0
            
            logger.info("所有模型缓存已清理")
            
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
    
    def stop(self):
        """停止缓存优化器"""
        self._stop_cleanup.set()
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)


def create_model_cache_optimizer(**kwargs) -> ModelCacheOptimizer:
    """创建模型缓存优化器的工厂函数"""
    return ModelCacheOptimizer(**kwargs)