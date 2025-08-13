#!/usr/bin/env python3
"""
统一数据源管理器
解决stocks.txt、数据库tickers、配置文件universe之间的数据源混乱问题
集成Polygon.io API作为统一数据源，支持T+5预测
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

# 添加上级目录到路径，以便导入polygon模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from polygon_client import polygon_client, download, Ticker
    from polygon_factors import PolygonFactorIntegrator, PolygonShortTermFactors
    POLYGON_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Polygon模块导入失败: {e}")
    POLYGON_AVAILABLE = False

class UnifiedDataSourceManager:
    """统一的数据源管理器"""
    
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
            self.logger.info("Polygon数据源已集成")
        else:
            self.logger.warning("Polygon数据源不可用，使用传统数据源")
        
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
        """初始化数据库表（通过database.py）"""
        try:
            # 延迟导入避免循环导入
            from .database import StockDatabase
            
            # 通过StockDatabase确保基础表结构存在
            db = StockDatabase()
            self.logger.info("数据源管理器初始化完成")
                
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
            
            # 3. 从配置文件同步
            self._sync_from_config_file()
            
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
        """从数据库tickers表同步（通过database.py统一访问）"""
        try:
            # 延迟导入避免循环导入
            from .database import StockDatabase
            
            db = StockDatabase()
            symbols = set(db.get_stock_universe())
            
            if symbols:
                self._sources['database_global'] = symbols
                self.logger.info(f"从数据库tickers表同步 {len(symbols)} 只股票")
            else:
                self.logger.debug("数据库中无股票数据")
                    
        except Exception as e:
            self.logger.error(f"数据库同步失败: {e}")
    
    def _sync_from_config_file(self):
        """从配置文件同步"""
        config_file = Path("config.json")
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                universe = config.get('CONFIG', {}).get('scanner', {}).get('universe', [])
                if universe:
                    symbols = {symbol.upper() for symbol in universe}
                    self._sources['config_hotload'] = symbols
                    self.logger.info(f"从配置文件同步 {len(symbols)} 只股票")
                
            except Exception as e:
                self.logger.error(f"从配置文件同步失败: {e}")
        else:
            self.logger.debug("config.json文件不存在")
    
    def _update_unified_table(self):
        """更新统一数据源表（通过database.py）"""
        try:
            # 延迟导入避免循环导入
            from .database import StockDatabase
            
            db = StockDatabase()
            
            # 构建所有符号的列表，按优先级排序
            all_symbols = []
            for source, symbols in self._sources.items():
                priority = self.source_priority.get(source, 1)
                for symbol in symbols:
                    all_symbols.append({
                        'symbol': symbol,
                        'source': source,
                        'priority': priority
                    })
            
            # 通过database.py更新（这里需要添加相应的方法到database.py）
            # 暂时记录日志，具体实现待database.py扩展
            self.logger.debug(f"准备更新统一数据源表: {len(all_symbols)} 条记录")
                
        except Exception as e:
            self.logger.error(f"更新统一数据源表失败: {e}")
    
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
                
                # 2. 更新数据库tickers表（通过database.py）
                try:
                    from .database import StockDatabase
                    
                    db = StockDatabase()
                    # 使用database.py的方法来批量更新股票列表
                    db.clear_tickers()
                    db.batch_add_tickers(list(final_universe))
                    
                    self.logger.info(f"已更新数据库tickers表: {len(final_universe)} 只股票")
                except Exception as e:
                    self.logger.error(f"更新数据库失败: {e}")
                
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
                
                self.logger.info(f"已更新配置文件: {len(final_universe)} 只股票")
                
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
                'polygon_available': self.polygon_available,
                'factor_cache_size': len(self._factor_cache),
                **self._stats
            }
    
    # ===============================
    # Polygon数据集成方法
    # ===============================
    
    def get_polygon_market_data(self, symbol: str, period: str = "1d", limit: int = 100) -> Dict[str, Any]:
        """获取Polygon市场数据"""
        if not self.polygon_available:
            self.logger.error("Polygon数据源不可用")
            return {}
        
        try:
            # 使用Polygon客户端获取历史数据
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
                self.logger.warning(f"Polygon未返回{symbol}的数据")
                return {}
                
        except Exception as e:
            self.logger.error(f"获取Polygon数据失败 {symbol}: {e}")
            return {}
    
    def calculate_t5_factors(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """计算T+5短期预测因子"""
        if not self.polygon_available:
            self.logger.error("Polygon因子库不可用")
            return {}
        
        cache_key = f"t5_factors_{symbol}"
        
        # 检查缓存
        if use_cache and cache_key in self._factor_cache:
            cached_data = self._factor_cache[cache_key]
            if time.time() - cached_data['timestamp'] < 3600:  # 1小时缓存
                return cached_data['factors']
        
        try:
            # 计算所有T+5因子
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
                
                self.logger.info(f"计算T+5因子完成 {symbol}: {len(factors)} 个因子")
                return result
            else:
                self.logger.warning(f"未能计算{symbol}的T+5因子")
                return {}
                
        except Exception as e:
            self.logger.error(f"计算T+5因子失败 {symbol}: {e}")
            return {}
    
    def get_polygon_factors_batch(self, symbols: List[str], factor_types: List[str] = None) -> Dict[str, Any]:
        """批量获取Polygon因子"""
        if not self.polygon_available:
            self.logger.error("Polygon因子库不可用")
            return {}
        
        if factor_types is None:
            factor_types = ['microstructure', 'technical', 'momentum', 'volume']
        
        try:
            results = {}
            
            for symbol in symbols:
                try:
                    # 使用因子集成器
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
                    self.logger.error(f"获取{symbol}因子失败: {e}")
                    continue
            
            self.logger.info(f"批量获取因子完成: {len(results)}/{len(symbols)} 个股票")
            return results
            
        except Exception as e:
            self.logger.error(f"批量获取因子失败: {e}")
            return {}
    
    def validate_polygon_data_quality(self, symbols: List[str]) -> Dict[str, Any]:
        """验证Polygon数据质量"""
        if not self.polygon_available:
            return {'error': 'Polygon不可用'}
        
        quality_report = {
            'total_symbols': len(symbols),
            'successful': 0,
            'failed': 0,
            'data_quality_scores': {},
            'issues': []
        }
        
        for symbol in symbols:
            try:
                # 获取基本数据
                data = self.get_polygon_market_data(symbol, limit=30)
                
                if data and 'data' in data:
                    df = data['data']
                    
                    # 数据质量检查
                    quality_score = 1.0
                    issues = []
                    
                    # 检查数据完整性
                    if len(df) < 20:
                        quality_score -= 0.3
                        issues.append("数据点不足")
                    
                    # 检查价格异常
                    if (df['Close'] <= 0).any():
                        quality_score -= 0.4
                        issues.append("存在无效价格")
                    
                    # 检查成交量异常
                    if (df['Volume'] <= 0).any():
                        quality_score -= 0.2
                        issues.append("存在无效成交量")
                    
                    # 检查数据连续性
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
                    quality_report['issues'].append({'symbol': symbol, 'issues': ['数据获取失败']})
                
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
            self.logger.info("因子缓存已清理")


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
