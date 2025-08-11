#!/usr/bin/env python3
"""
统一数据源管理器
解决stocks.txt、数据库tickers、HotConfig universe之间的数据源混乱问题
"""

import sqlite3
import json
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from pathlib import Path
from threading import RLock
from collections import defaultdict
import time

class UnifiedDataSourceManager:
    """统一的数据源管理器"""
    
    def __init__(self, db_path: str = "data/autotrader_stocks.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger("DataSourceManager")
        self.lock = RLock()
        
        # 缓存
        self._universe_cache: Optional[List[str]] = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 300.0  # 5分钟缓存
        
        # 数据源优先级（数字越大优先级越高）
        self.source_priority = {
            'manual_input': 1,      # 手动输入
            'file_import': 2,       # 文件导入（stocks.txt）
            'database_global': 3,   # 数据库全局tickers
            'config_hotload': 4,    # HotConfig配置
            'runtime_override': 5   # 运行时覆盖（最高优先级）
        }
        
        # 各数据源的内容
        self._sources: Dict[str, Set[str]] = {
            'manual_input': set(),
            'file_import': set(),
            'database_global': set(),
            'config_hotload': set(),
            'runtime_override': set()
        }
        
        # 统计信息
        self._stats = {
            'last_sync': 0,
            'sync_count': 0,
            'conflicts_resolved': 0,
            'total_symbols': 0
        }
        
        self._init_database()
        self._initial_sync()
    
    def _init_database(self):
        """初始化数据库表"""
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建统一的数据源表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS unified_data_sources (
                        symbol TEXT NOT NULL,
                        source TEXT NOT NULL,
                        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        priority INTEGER DEFAULT 1,
                        metadata TEXT,
                        PRIMARY KEY (symbol, source)
                    )
                """)
                
                # 创建索引
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sources_symbol 
                    ON unified_data_sources(symbol)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sources_priority 
                    ON unified_data_sources(priority DESC)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sources_active 
                    ON unified_data_sources(is_active)
                """)
                
                conn.commit()
                self.logger.info("数据源管理表初始化完成")
                
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
    
    def _initial_sync(self):
        """初始数据同步"""
        try:
            self.sync_from_all_sources()
            self.logger.info("初始数据同步完成")
        except Exception as e:
            self.logger.error(f"初始数据同步失败: {e}")
    
    def sync_from_all_sources(self):
        """从所有数据源同步数据"""
        with self.lock:
            self.logger.info("开始从所有数据源同步...")
            
            # 1. 从stocks.txt同步
            self._sync_from_file()
            
            # 2. 从数据库tickers表同步
            self._sync_from_database_tickers()
            
            # 3. 从HotConfig同步
            self._sync_from_hotconfig()
            
            # 4. 更新统一表
            self._update_unified_table()
            
            # 5. 更新统计
            self._stats['last_sync'] = time.time()
            self._stats['sync_count'] += 1
            
            # 6. 清理缓存
            self._invalidate_cache()
            
            self.logger.info("数据源同步完成")
    
    def _sync_from_file(self):
        """从stocks.txt文件同步"""
        stocks_file = Path("stocks.txt")
        
        if stocks_file.exists():
            try:
                with open(stocks_file, 'r', encoding='utf-8') as f:
                    symbols = {line.strip().upper() for line in f if line.strip()}
                
                self._sources['file_import'] = symbols
                self.logger.info(f"从stocks.txt同步 {len(symbols)} 只股票")
                
            except Exception as e:
                self.logger.error(f"从stocks.txt同步失败: {e}")
        else:
            self.logger.debug("stocks.txt文件不存在")
    
    def _sync_from_database_tickers(self):
        """从数据库tickers表同步"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 检查tickers表是否存在
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='tickers'
                """)
                
                if cursor.fetchone():
                    # 兼容不同的数据库schema
                    try:
                        cursor.execute("SELECT symbol FROM tickers WHERE is_active = 1")
                    except sqlite3.OperationalError:
                        # 如果没有is_active列，直接查询symbol
                        cursor.execute("SELECT symbol FROM tickers")
                    symbols = {row[0].upper() for row in cursor.fetchall()}
                    
                    self._sources['database_global'] = symbols
                    self.logger.info(f"从数据库tickers表同步 {len(symbols)} 只股票")
                else:
                    self.logger.debug("数据库tickers表不存在")
                    
        except Exception as e:
            self.logger.error(f"从数据库同步失败: {e}")
    
    def _sync_from_hotconfig(self):
        """从HotConfig同步"""
        config_file = Path("config.json")
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                universe = config.get('CONFIG', {}).get('scanner', {}).get('universe', [])
                if universe:
                    symbols = {symbol.upper() for symbol in universe}
                    self._sources['config_hotload'] = symbols
                    self.logger.info(f"从HotConfig同步 {len(symbols)} 只股票")
                
            except Exception as e:
                self.logger.error(f"从HotConfig同步失败: {e}")
        else:
            self.logger.debug("config.json文件不存在")
    
    def _update_unified_table(self):
        """更新统一数据源表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 清理旧数据
                cursor.execute("DELETE FROM unified_data_sources")
                
                # 插入所有源的数据
                for source, symbols in self._sources.items():
                    priority = self.source_priority.get(source, 1)
                    
                    for symbol in symbols:
                        cursor.execute("""
                            INSERT INTO unified_data_sources 
                            (symbol, source, priority, is_active)
                            VALUES (?, ?, ?, 1)
                        """, (symbol, source, priority))
                
                conn.commit()
                self.logger.info("统一数据源表更新完成")
                
        except Exception as e:
            self.logger.error(f"更新统一表失败: {e}")
    
    def get_universe(self, force_refresh: bool = False) -> List[str]:
        """获取统一的股票列表（按优先级合并）"""
        with self.lock:
            # 检查缓存
            if not force_refresh and self._universe_cache:
                if time.time() - self._cache_timestamp < self._cache_ttl:
                    return self._universe_cache.copy()
            
            # 按优先级合并
            universe_dict = {}  # symbol -> (priority, source)
            
            for source, symbols in self._sources.items():
                priority = self.source_priority.get(source, 1)
                
                for symbol in symbols:
                    if symbol not in universe_dict or priority > universe_dict[symbol][0]:
                        universe_dict[symbol] = (priority, source)
            
            # 提取最终列表
            final_universe = sorted(list(universe_dict.keys()))
            
            # 更新缓存
            self._universe_cache = final_universe
            self._cache_timestamp = time.time()
            self._stats['total_symbols'] = len(final_universe)
            
            self.logger.info(f"合并得到最终universe: {len(final_universe)} 只股票")
            
            # 记录来源统计
            source_stats = defaultdict(int)
            for symbol, (priority, source) in universe_dict.items():
                source_stats[source] += 1
            
            self.logger.debug(f"来源统计: {dict(source_stats)}")
            
            return final_universe
    
    def add_symbols(self, symbols: List[str], source: str = 'manual_input', save_to_db: bool = True):
        """添加股票到指定数据源"""
        with self.lock:
            if source not in self._sources:
                self.logger.warning(f"未知数据源: {source}")
                return 0
            
            # 标准化符号
            normalized_symbols = {symbol.upper().strip() for symbol in symbols if symbol.strip()}
            
            # 添加到内存
            added_count = 0
            for symbol in normalized_symbols:
                if symbol not in self._sources[source]:
                    self._sources[source].add(symbol)
                    added_count += 1
            
            # 保存到数据库
            if save_to_db and added_count > 0:
                try:
                    self._update_unified_table()
                    self.logger.info(f"添加 {added_count} 只股票到 {source}")
                except Exception as e:
                    self.logger.error(f"保存到数据库失败: {e}")
            
            # 清理缓存
            self._invalidate_cache()
            
            return added_count
    
    def remove_symbols(self, symbols: List[str], source: Optional[str] = None, save_to_db: bool = True):
        """移除股票"""
        with self.lock:
            normalized_symbols = {symbol.upper().strip() for symbol in symbols if symbol.strip()}
            removed_count = 0
            
            if source:
                # 从特定源移除
                if source in self._sources:
                    for symbol in normalized_symbols:
                        if symbol in self._sources[source]:
                            self._sources[source].discard(symbol)
                            removed_count += 1
            else:
                # 从所有源移除
                for src_name, src_symbols in self._sources.items():
                    for symbol in normalized_symbols:
                        if symbol in src_symbols:
                            src_symbols.discard(symbol)
                            removed_count += 1
            
            # 保存到数据库
            if save_to_db and removed_count > 0:
                try:
                    self._update_unified_table()
                    self.logger.info(f"移除 {removed_count} 只股票")
                except Exception as e:
                    self.logger.error(f"保存到数据库失败: {e}")
            
            # 清理缓存
            self._invalidate_cache()
            
            return removed_count
    
    def set_runtime_universe(self, symbols: List[str]):
        """设置运行时universe（最高优先级）"""
        with self.lock:
            normalized_symbols = {symbol.upper().strip() for symbol in symbols if symbol.strip()}
            self._sources['runtime_override'] = normalized_symbols
            self._invalidate_cache()
            
            self.logger.info(f"设置运行时universe: {len(normalized_symbols)} 只股票")
    
    def clear_runtime_universe(self):
        """清除运行时universe"""
        with self.lock:
            self._sources['runtime_override'].clear()
            self._invalidate_cache()
            self.logger.info("已清除运行时universe")
    
    def get_source_breakdown(self) -> Dict[str, Any]:
        """获取数据源详细分解"""
        with self.lock:
            breakdown = {}
            
            for source, symbols in self._sources.items():
                if symbols:
                    breakdown[source] = {
                        'count': len(symbols),
                        'priority': self.source_priority.get(source, 1),
                        'symbols': sorted(list(symbols))
                    }
            
            return breakdown
    
    def analyze_conflicts(self) -> Dict[str, Any]:
        """分析数据源冲突"""
        with self.lock:
            all_symbols = set()
            for symbols in self._sources.values():
                all_symbols.update(symbols)
            
            conflicts = []
            
            # 找出只在某些源中存在的股票
            for symbol in all_symbols:
                sources_with_symbol = [
                    source for source, symbols in self._sources.items()
                    if symbol in symbols
                ]
                
                if len(sources_with_symbol) == 1:
                    conflicts.append({
                        'symbol': symbol,
                        'type': 'unique_to_source',
                        'source': sources_with_symbol[0],
                        'priority': self.source_priority.get(sources_with_symbol[0], 1)
                    })
            
            # 计算覆盖度
            coverage = {}
            for source, symbols in self._sources.items():
                if symbols:
                    coverage[source] = len(symbols & all_symbols) / len(all_symbols) * 100
            
            return {
                'total_unique_symbols': len(all_symbols),
                'source_coverage': coverage,
                'unique_conflicts': conflicts,
                'conflict_count': len(conflicts)
            }
    
    def reconcile_all_sources(self, target_source: str = 'database_global'):
        """统一所有数据源到目标源"""
        with self.lock:
            if target_source not in self._sources:
                self.logger.error(f"目标源 {target_source} 不存在")
                return False
            
            # 获取最终合并的universe
            final_universe = set(self.get_universe(force_refresh=True))
            
            if not final_universe:
                self.logger.warning("没有可统一的数据")
                return False
            
            try:
                # 1. 更新stocks.txt
                stocks_file = Path("stocks.txt")
                with open(stocks_file, 'w', encoding='utf-8') as f:
                    for symbol in sorted(final_universe):
                        f.write(f"{symbol}\n")
                
                self.logger.info(f"已更新stocks.txt: {len(final_universe)} 只股票")
                
                # 2. 更新数据库tickers表
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 清空并重建
                    cursor.execute("DELETE FROM tickers")
                    
                    for symbol in final_universe:
                        try:
                            cursor.execute("""
                                INSERT INTO tickers (symbol, is_active) 
                                VALUES (?, 1)
                            """, (symbol,))
                        except sqlite3.OperationalError:
                            # 如果没有is_active列，只插入symbol
                            cursor.execute("""
                                INSERT INTO tickers (symbol) 
                                VALUES (?)
                            """, (symbol,))
                    
                    conn.commit()
                
                self.logger.info(f"已更新数据库tickers表: {len(final_universe)} 只股票")
                
                # 3. 更新HotConfig
                config_file = Path("config.json")
                
                config = {}
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                    except Exception:
                        pass
                
                if 'CONFIG' not in config:
                    config['CONFIG'] = {}
                if 'scanner' not in config['CONFIG']:
                    config['CONFIG']['scanner'] = {}
                
                config['CONFIG']['scanner']['universe'] = sorted(list(final_universe))
                
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"已更新HotConfig: {len(final_universe)} 只股票")
                
                # 4. 重新同步
                self.sync_from_all_sources()
                
                self._stats['conflicts_resolved'] += 1
                self.logger.info("数据源统一完成")
                
                return True
                
            except Exception as e:
                self.logger.error(f"数据源统一失败: {e}")
                return False
    
    def _invalidate_cache(self):
        """使缓存失效"""
        self._universe_cache = None
        self._cache_timestamp = 0
    
    def export_to_file(self, filepath: str, source: Optional[str] = None):
        """导出数据源到文件"""
        with self.lock:
            try:
                if source:
                    symbols = sorted(list(self._sources.get(source, set())))
                else:
                    symbols = self.get_universe()
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    for symbol in symbols:
                        f.write(f"{symbol}\n")
                
                self.logger.info(f"已导出 {len(symbols)} 只股票到 {filepath}")
                return True
                
            except Exception as e:
                self.logger.error(f"导出失败: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            source_stats = {}
            for source, symbols in self._sources.items():
                source_stats[source] = len(symbols)
            
            return {
                'sources': source_stats,
                'cache_valid': self._universe_cache is not None,
                'cache_age_seconds': time.time() - self._cache_timestamp if self._cache_timestamp else 0,
                **self._stats
            }


# 全局数据源管理器
_global_data_source_manager: Optional[UnifiedDataSourceManager] = None

def get_data_source_manager() -> UnifiedDataSourceManager:
    """获取全局数据源管理器"""
    global _global_data_source_manager
    if _global_data_source_manager is None:
        _global_data_source_manager = UnifiedDataSourceManager()
    return _global_data_source_manager

def get_unified_universe() -> List[str]:
    """获取统一的股票列表"""
    manager = get_data_source_manager()
    return manager.get_universe()

def sync_all_data_sources():
    """同步所有数据源"""
    manager = get_data_source_manager()
    manager.sync_from_all_sources()

def reconcile_data_sources():
    """统一所有数据源"""
    manager = get_data_source_manager()
    return manager.reconcile_all_sources()
