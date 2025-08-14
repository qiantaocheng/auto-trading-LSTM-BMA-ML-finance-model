#!/usr/bin/env python3
"""
统一数据源管理器
解决stocks.txt、数据库tickers、配置文件universe之间数据源混乱问题
集成Polygon.io API作as统一数据源，支持T+5预测
"""

import json
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from pathlib import Path
from threading import RLock
from collections import defaultdict
import time
import sys
import os

# 添加上级目录to路径，以便导入polygon模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from polygon_client import polygon_client, download, Ticker
    from polygon_factors import PolygonFactorIntegrator, PolygonShortTermFactors
    POLYGON_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Polygon模块导入failed: {e}")
    POLYGON_AVAILABLE = False

class UnifiedDataSourceManager:
    """统一数据源管理器"""
    
    def __init__(self, db_path: str = "data/autotrader_stocks.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger("DataSourceManager")
        self.lock = RLock()
        
        # Polygon集成
        self.polygon_available = POLYGON_AVAILABLE
        if self.polygon_available:
            self.polygon_client = polygon_client
            self.polygon_integrator = PolygonFactorIntegrator()
            self.polygon_short_term = PolygonShortTermFactors()
            self.logger.info("Polygon数据源集成")
        else:
            self.logger.warning("Polygon数据源notcanuse，使use传统数据源")
        
        # 缓存
        self._universe_cache: Optional[List[str]] = None
        self._cache_timestamp: float = 0
        self._cache_ttl: float = 300.0  # 5分钟缓存
        self._factor_cache: Dict[str, Any] = {}  # 因子缓存
        
        # 数据源优先级（数字越大优先级越高）
        self.source_priority = {
            'manual_input': 1,      # 手动输入
            'file_import': 2,       # 文件导入（stocks.txt）
            'database_global': 3,   # 数据库全局tickers
            'config_hotload': 4,    # 配置文件
            'runtime_override': 5   # 运行when覆盖（最高优先级）
        }
        
        # 各数据源内容
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
        """初始化数据库表（通过database.py）"""
        try:
            # 延迟导入避免循环导入
            from .database import StockDatabase
            
            # 通过StockDatabase确保基础表结构存in
            db = StockDatabase()
            self.logger.info("数据源管理器初始化completed")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def _initial_sync(self):
        """初始数据同步"""
        try:
            self.sync_from_all_sources()
            self.logger.info("初始数据同步completed")
        except Exception as e:
            self.logger.error(f"初始数据同步failed: {e}")
    
    def sync_from_all_sources(self):
        """from所has数据源同步数据"""
        with self.lock:
            self.logger.info("startingfrom所has数据源同步...")
            
            # 1. fromstocks.txt同步
            self._sync_from_file()
            
            # 2. from数据库tickers表同步
            self._sync_from_database_tickers()
            
            # 3. from配置文件同步
            self._sync_from_config_file()
            
            # 4. updates统一表
            self._update_unified_table()
            
            # 5. updates统计
            self._stats['last_sync'] = time.time()
            self._stats['sync_count'] += 1
            
            # 6. 清理缓存
            self._invalidate_cache()
            
            self.logger.info("数据源同步completed")
    
    def _sync_from_file(self):
        """fromstocks.txt文件同步"""
        stocks_file = Path("stocks.txt")
        
        if stocks_file.exists():
            try:
                with open(stocks_file, 'r', encoding='utf-8') as f:
                    symbols = {line.strip().upper() for line in f if line.strip()}
                
                self._sources['file_import'] = symbols
                self.logger.info(f"fromstocks.txt同步 {len(symbols)} 只股票")
                
            except Exception as e:
                self.logger.error(f"fromstocks.txt同步failed: {e}")
        else:
            self.logger.debug("stocks.txt文件not存in")
    
    def _sync_from_database_tickers(self):
        """from数据库tickers表同步（通过database.py统一访问）"""
        try:
            # 延迟导入避免循环导入
            from .database import StockDatabase
            
            db = StockDatabase()
            symbols = set(db.get_stock_universe())
            
            if symbols:
                self._sources['database_global'] = symbols
                self.logger.info(f"from数据库tickers表同步 {len(symbols)} 只股票")
            else:
                self.logger.debug("数据库inno股票数据")
                    
        except Exception as e:
            self.logger.error(f"数据库同步failed: {e}")
    
    def _sync_from_config_file(self):
        """from配置文件同步"""
        config_file = Path("config.json")
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                universe = config.get('CONFIG', {}).get('scanner', {}).get('universe', [])
                if universe:
                    symbols = {symbol.upper() for symbol in universe}
                    self._sources['config_hotload'] = symbols
                    self.logger.info(f"from配置文件同步 {len(symbols)} 只股票")
                
            except Exception as e:
                self.logger.error(f"from配置文件同步failed: {e}")
        else:
            self.logger.debug("config.json文件not存in")
    
    def _update_unified_table(self):
        """updates统一数据源表（通过database.py）"""
        try:
            # 延迟导入避免循环导入
            from .database import StockDatabase
            
            db = StockDatabase()
            
            # 构建所has符号列表，按优先级排序
            all_symbols = []
            for source, symbols in self._sources.items():
                priority = self.source_priority.get(source, 1)
                for symbol in symbols:
                    all_symbols.append({
                        'symbol': symbol,
                        'source': source,
                        'priority': priority
                    })
            
            # 通过database.pyupdates（这里需要添加相应方法todatabase.py）
            # 暂when记录日志，具体实现待database.py扩展
            self.logger.debug(f"准备updates统一数据源表: {len(all_symbols)}  records记录")
                
        except Exception as e:
            self.logger.error(f"updates统一数据源表failed: {e}")
    
    def get_universe(self, force_refresh: bool = False) -> List[str]:
        """retrieval统一股票列表（按优先级合并）"""
        with self.lock:
            # check缓存
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
            
            # updates缓存
            self._universe_cache = final_universe
            self._cache_timestamp = time.time()
            self._stats['total_symbols'] = len(final_universe)
            
            self.logger.info(f"合并得to最终universe: {len(final_universe)} 只股票")
            
            # 记录来源统计
            source_stats = defaultdict(int)
            for symbol, (priority, source) in universe_dict.items():
                source_stats[source] += 1
            
            self.logger.debug(f"来源统计: {dict(source_stats)}")
            
            return final_universe
    
    def add_symbols(self, symbols: List[str], source: str = 'manual_input', save_to_db: bool = True):
        """添加股票to指定数据源"""
        with self.lock:
            if source not in self._sources:
                self.logger.warning(f"未知数据源: {source}")
                return 0
            
            # 标准化符号
            normalized_symbols = {symbol.upper().strip() for symbol in symbols if symbol.strip()}
            
            # 添加to内存
            added_count = 0
            for symbol in normalized_symbols:
                if symbol not in self._sources[source]:
                    self._sources[source].add(symbol)
                    added_count += 1
            
            # 保存to数据库
            if save_to_db and added_count > 0:
                try:
                    self._update_unified_table()
                    self.logger.info(f"添加 {added_count} 只股票to {source}")
                except Exception as e:
                    self.logger.error(f"保存to数据库failed: {e}")
            
            # 清理缓存
            self._invalidate_cache()
            
            return added_count
    
    def remove_symbols(self, symbols: List[str], source: Optional[str] = None, save_to_db: bool = True):
        """移除股票"""
        with self.lock:
            normalized_symbols = {symbol.upper().strip() for symbol in symbols if symbol.strip()}
            removed_count = 0
            
            if source:
                # from特定源移除
                if source in self._sources:
                    for symbol in normalized_symbols:
                        if symbol in self._sources[source]:
                            self._sources[source].discard(symbol)
                            removed_count += 1
            else:
                # from所has源移除
                for src_name, src_symbols in self._sources.items():
                    for symbol in normalized_symbols:
                        if symbol in src_symbols:
                            src_symbols.discard(symbol)
                            removed_count += 1
            
            # 保存to数据库
            if save_to_db and removed_count > 0:
                try:
                    self._update_unified_table()
                    self.logger.info(f"移除 {removed_count} 只股票")
                except Exception as e:
                    self.logger.error(f"保存to数据库failed: {e}")
            
            # 清理缓存
            self._invalidate_cache()
            
            return removed_count
    
    def set_runtime_universe(self, symbols: List[str]):
        """settings运行whenuniverse（最高优先级）"""
        with self.lock:
            normalized_symbols = {symbol.upper().strip() for symbol in symbols if symbol.strip()}
            self._sources['runtime_override'] = normalized_symbols
            self._invalidate_cache()
            
            self.logger.info(f"settings运行whenuniverse: {len(normalized_symbols)} 只股票")
    
    def clear_runtime_universe(self):
        """清除运行whenuniverse"""
        with self.lock:
            self._sources['runtime_override'].clear()
            self._invalidate_cache()
            self.logger.info("清除运行whenuniverse")
    
    def get_source_breakdown(self) -> Dict[str, Any]:
        """retrieval数据源详细分解"""
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
            
            # 找出只in某些源in存in股票
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
        """统一所has数据源to目标源"""
        with self.lock:
            if target_source not in self._sources:
                self.logger.error(f"目标源 {target_source} not存in")
                return False
            
            # retrieval最终合并universe
            final_universe = set(self.get_universe(force_refresh=True))
            
            if not final_universe:
                self.logger.warning("没hascan统一数据")
                return False
            
            try:
                # 1. updatesstocks.txt
                stocks_file = Path("stocks.txt")
                with open(stocks_file, 'w', encoding='utf-8') as f:
                    for symbol in sorted(final_universe):
                        f.write(f"{symbol}\n")
                
                self.logger.info(f"updatesstocks.txt: {len(final_universe)} 只股票")
                
                # 2. updates数据库tickers表（通过database.py）
                try:
                    from .database import StockDatabase
                    
                    db = StockDatabase()
                    # 使usedatabase.py方法来批量updates股票列表
                    db.clear_tickers()
                    db.batch_add_tickers(list(final_universe))
                    
                    self.logger.info(f"updates数据库tickers表: {len(final_universe)} 只股票")
                except Exception as e:
                    self.logger.error(f"updates数据库failed: {e}")
                
                # 3. updatesHotConfig
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
                
                self.logger.info(f"updates配置文件: {len(final_universe)} 只股票")
                
                # 4. 重新同步
                self.sync_from_all_sources()
                
                self._stats['conflicts_resolved'] += 1
                self.logger.info("数据源统一completed")
                
                return True
                
            except Exception as e:
                self.logger.error(f"数据源统一failed: {e}")
                return False
    
    def _invalidate_cache(self):
        """使缓存失效"""
        self._universe_cache = None
        self._cache_timestamp = 0
    
    def export_to_file(self, filepath: str, source: Optional[str] = None):
        """导出数据源to文件"""
        with self.lock:
            try:
                if source:
                    symbols = sorted(list(self._sources.get(source, set())))
                else:
                    symbols = self.get_universe()
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    for symbol in symbols:
                        f.write(f"{symbol}\n")
                
                self.logger.info(f"导出 {len(symbols)} 只股票to {filepath}")
                return True
                
            except Exception as e:
                self.logger.error(f"导出failed: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """retrieval统计信息"""
        with self.lock:
            source_stats = {}
            for source, symbols in self._sources.items():
                source_stats[source] = len(symbols)
            
            return {
                'sources': source_stats,
                'cache_valid': self._universe_cache is not None,
                'cache_age_seconds': time.time() - self._cache_timestamp if self._cache_timestamp else 0,
                'polygon_available': self.polygon_available,
                'factor_cache_size': len(self._factor_cache),
                **self._stats
            }
    
    # ===============================
    # Polygon数据集成方法
    # ===============================
    
    def get_polygon_market_data(self, symbol: str, period: str = "1d", limit: int = 100) -> Dict[str, Any]:
        """retrievalPolygon市场数据"""
        if not self.polygon_available:
            self.logger.error("Polygon数据源notcanuse")
            return {}
        
        try:
            # 使usePolygon客户端retrieval历史数据
            end_date = time.strftime("%Y-%m-%d")
            start_date = time.strftime("%Y-%m-%d", time.gmtime(time.time() - limit * 24 * 3600))
            
            data = download(symbol, start=start_date, end=end_date)
            
            if len(data) > 0:
                return {
                    'symbol': symbol,
                    'data': data,
                    'source': 'polygon',
                    'timestamp': time.time(),
                    'records': len(data)
                }
            else:
                self.logger.warning(f"Polygon未返回{symbol}数据")
                return {}
                
        except Exception as e:
            self.logger.error(f"retrievalPolygon数据failed {symbol}: {e}")
            return {}
    
    def calculate_t5_factors(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """计算T+5短期预测因子"""
        if not self.polygon_available:
            self.logger.error("Polygon因子库notcanuse")
            return {}
        
        cache_key = f"t5_factors_{symbol}"
        
        # check缓存
        if use_cache and cache_key in self._factor_cache:
            cached_data = self._factor_cache[cache_key]
            if time.time() - cached_data['timestamp'] < 3600:  # 1小when缓存
                return cached_data['factors']
        
        try:
            # 计算所hasT+5因子
            factors = self.polygon_short_term.calculate_all_short_term_factors(symbol)
            
            if factors:
                # 创建T+5预测
                prediction = self.polygon_short_term.create_t_plus_5_prediction(symbol, factors)
                
                result = {
                    'symbol': symbol,
                    'factors': factors,
                    'prediction': prediction,
                    'timestamp': time.time(),
                    'factor_count': len(factors)
                }
                
                # 缓存结果
                if use_cache:
                    self._factor_cache[cache_key] = {
                        'factors': result,
                        'timestamp': time.time()
                    }
                
                self.logger.info(f"计算T+5因子completed {symbol}: {len(factors)} 个因子")
                return result
            else:
                self.logger.warning(f"未能计算{symbol}T+5因子")
                return {}
                
        except Exception as e:
            self.logger.error(f"计算T+5因子failed {symbol}: {e}")
            return {}
    
    def get_polygon_factors_batch(self, symbols: List[str], factor_types: List[str] = None) -> Dict[str, Any]:
        """批量retrievalPolygon因子"""
        if not self.polygon_available:
            self.logger.error("Polygon因子库notcanuse")
            return {}
        
        if factor_types is None:
            factor_types = ['microstructure', 'technical', 'momentum', 'volume']
        
        try:
            results = {}
            
            for symbol in symbols:
                try:
                    # 使use因子集成器
                    factor_matrix = self.polygon_integrator.create_factor_matrix(
                        [symbol], 
                        factors=factor_types
                    )
                    
                    if len(factor_matrix) > 0:
                        results[symbol] = {
                            'factor_matrix': factor_matrix,
                            'timestamp': time.time(),
                            'factor_types': factor_types
                        }
                    
                    time.sleep(0.1)  # API限制
                    
                except Exception as e:
                    self.logger.error(f"retrieval{symbol}因子failed: {e}")
                    continue
            
            self.logger.info(f"批量retrieval因子completed: {len(results)}/{len(symbols)} 个股票")
            return results
            
        except Exception as e:
            self.logger.error(f"批量retrieval因子failed: {e}")
            return {}
    
    def validate_polygon_data_quality(self, symbols: List[str]) -> Dict[str, Any]:
        """验证Polygon数据质量"""
        if not self.polygon_available:
            return {'error': 'Polygonnotcanuse'}
        
        quality_report = {
            'total_symbols': len(symbols),
            'successful': 0,
            'failed': 0,
            'data_quality_scores': {},
            'issues': []
        }
        
        for symbol in symbols:
            try:
                # retrieval基本数据
                data = self.get_polygon_market_data(symbol, limit=30)
                
                if data and 'data' in data:
                    df = data['data']
                    
                    # 数据质量check
                    quality_score = 1.0
                    issues = []
                    
                    # check数据完整性
                    if len(df) < 20:
                        quality_score -= 0.3
                        issues.append("数据点not足")
                    
                    # checkprice异常
                    if (df['Close'] <= 0).any():
                        quality_score -= 0.4
                        issues.append("存inno效price")
                    
                    # checkexecution量异常
                    if (df['Volume'] <= 0).any():
                        quality_score -= 0.2
                        issues.append("存inno效execution量")
                    
                    # check数据连续性
                    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
                    if missing_ratio > 0.1:
                        quality_score -= missing_ratio
                        issues.append(f"缺失数据比例: {missing_ratio:.2%}")
                    
                    quality_report['data_quality_scores'][symbol] = max(0, quality_score)
                    if issues:
                        quality_report['issues'].append({'symbol': symbol, 'issues': issues})
                    
                    quality_report['successful'] += 1
                else:
                    quality_report['failed'] += 1
                    quality_report['issues'].append({'symbol': symbol, 'issues': ['数据retrievalfailed']})
                
            except Exception as e:
                quality_report['failed'] += 1
                quality_report['issues'].append({'symbol': symbol, 'issues': [str(e)]})
        
        # 计算整体质量分数
        if quality_report['data_quality_scores']:
            quality_report['overall_quality'] = sum(quality_report['data_quality_scores'].values()) / len(quality_report['data_quality_scores'])
        else:
            quality_report['overall_quality'] = 0.0
        
        return quality_report
    
    def clear_factor_cache(self):
        """清理因子缓存"""
        with self.lock:
            self._factor_cache.clear()
            self.logger.info("因子缓存清理")


# 全局数据源管理器
_global_data_source_manager: Optional[UnifiedDataSourceManager] = None

def get_data_source_manager() -> UnifiedDataSourceManager:
    """retrieval全局数据源管理器"""
    global _global_data_source_manager
    if _global_data_source_manager is None:
        _global_data_source_manager = UnifiedDataSourceManager()
    return _global_data_source_manager

def get_unified_universe() -> List[str]:
    """retrieval统一股票列表"""
    manager = get_data_source_manager()
    return manager.get_universe()

def sync_all_data_sources():
    """同步所has数据源"""
    manager = get_data_source_manager()
    manager.sync_from_all_sources()

def reconcile_data_sources():
    """统一所has数据源"""
    manager = get_data_source_manager()
    return manager.reconcile_all_sources()
