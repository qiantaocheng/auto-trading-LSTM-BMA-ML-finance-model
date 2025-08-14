#!/usr/bin/env python3
"""
统一配置管理器，解决配置冲突问题
解决HotConfig、数据库配置、GUI配置相互冲突问题
"""

import json
import sqlite3
import logging
from typing import Dict, Any, Optional, List
# 清理：移除未使use导入
# import os
# from typing import Union
from pathlib import Path
from threading import RLock
from copy import deepcopy
import time

class UnifiedConfigManager:
    """统一配置管理器，解决配置冲突"""
    
    # 配置优先级（数字越大优先级越高）
    PRIORITY = {
        'default': 1,
        'file': 2,
        'database': 3,
        'hotconfig': 4,
        'runtime': 5  # GUIor命令行参数
    }
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger("UnifiedConfig")
        self.lock = RLock()
        
        # 分层配置存储
        self._configs: Dict[str, Dict[str, Any]] = {
            'default': {},
            'file': {},
            'database': {},
            'hotconfig': {},
            'runtime': {}
        }
        
        # 合并after配置缓存
        self._merged_config: Optional[Dict[str, Any]] = None
        self._cache_valid = False
        self._last_update = 0
        
        # 配置路径
        self.paths = {
            'hotconfig': self.base_dir / 'config.json',
            'risk': self.base_dir / 'data' / 'risk_config.json',
            'connection': self.base_dir / 'data' / 'connection.json',
            'database': self.base_dir / 'data' / 'autotrader_stocks.db'
        }
        
        # 监控配置变化
        self._file_mtimes: Dict[str, float] = {}
        
        # 初始化默认配置
        self._init_defaults()
        
        # 自动加载
        self.load_all()
    
    def _init_defaults(self):
        """初始化默认配置"""
        self._configs['default'] = {
            'connection': {
                'host': '127.0.0.1',
                'port': 7497,
                'client_id': 3130,
                'account_id': 'c2dvdongg',
                'use_delayed_if_no_realtime': True,
                'timeout': 20.0,
                'max_reconnect_attempts': 10
            },
            'capital': {
                'initial_capital': 100000.0,
                'cash_reserve_pct': 0.15,
                'max_single_position_pct': 0.12,
                'max_portfolio_exposure': 0.85,
                'require_account_ready': True
            },
            'orders': {
                'smart_price_mode': 'midpoint',
                'default_stop_loss_pct': 0.02,
                'default_take_profit_pct': 0.05,
                'min_order_value_usd': 500.0,
                'verify_tolerance_usd': 100.0,
                'daily_order_limit': 20
            },
            'scanner': {
                'universe': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                'max_stocks': 100,
                'enable_filtering': True
            },
            'signals': {
                'acceptance_threshold': 0.6,
                'enable_realtime': True,
                'signal_timeout': 300,
                'min_signal_strength': 0.3
            },
            'sizing': {
                'per_trade_risk_pct': 0.02,
                'max_position_pct_of_equity': 0.15,
                'min_position_usd': 1000.0,
                'min_shares': 1,
                'notional_round_lots': True
            },
            'risk_controls': {
                'daily_order_limit': 20,
                'sector_exposure_limit': 0.30,
                'max_correlation': 0.70,
                'enable_dynamic_stops': True
            },
            'performance': {
                'cache_indicators': True,
                'cache_ttl_seconds': 300,
                'max_history_bars': 500,
                'parallel_processing': True
            }
        }
    
    def load_all(self, force: bool = False):
        """加载所has配置源"""
        with self.lock:
            # 添加加载防抖机制，避免过频繁加载
            current_time = time.time()
            if not force and hasattr(self, '_last_load_attempt'):
                if current_time - self._last_load_attempt < 5.0:  # 5秒内不重复加载
                    return
            self._last_load_attempt = current_time
            
            # check文件修改when间；仅在确实需要加载时打印日志
            if not force and not self._files_changed():
                return
            
            self.logger.info("Starting to load all configuration sources...")
            
            # 1. 加载HotConfig
            if self.paths['hotconfig'].exists():
                try:
                    with open(self.paths['hotconfig'], 'r', encoding='utf-8') as f:
                        hot_config = json.load(f)
                        if 'CONFIG' in hot_config:
                            self._configs['hotconfig'] = hot_config['CONFIG']
                            self.logger.info(f"加载HotConfig: {self.paths['hotconfig']}")
                except Exception as e:
                    self.logger.error(f"加载HotConfigfailed: {e}")
                finally:
                    # 无论成功失败都更新文件修改时间，避免无限重试
                    self._file_mtimes['hotconfig'] = self.paths['hotconfig'].stat().st_mtime
            
            # 2. 加载风险配置文件
            if self.paths['risk'].exists():
                try:
                    with open(self.paths['risk'], 'r', encoding='utf-8') as f:
                        risk_config = json.load(f)
                        if 'risk_management' in risk_config:
                            self._configs['file']['risk_management'] = risk_config['risk_management']
                        else:
                            self._configs['file']['risk_management'] = risk_config
                        self.logger.info(f"Risk configuration loaded: {self.paths['risk']}")
                except Exception as e:
                    self.logger.error(f"加载风险配置failed: {e}")
                finally:
                    # 无论成功失败都更新文件修改时间，避免无限重试
                    self._file_mtimes['risk'] = self.paths['risk'].stat().st_mtime
            
            # 3. 加载connection配置文件
            if self.paths['connection'].exists():
                try:
                    with open(self.paths['connection'], 'r', encoding='utf-8') as f:
                        connection_data = json.load(f)
                        self._configs['file']['connection'] = connection_data
                        self.logger.info(f"加载connection配置: {self.paths['connection']}")
                except Exception as e:
                    self.logger.error(f"加载connection配置failed: {e}")
                finally:
                    # 无论成功失败都更新文件修改时间，避免无限重试
                    self._file_mtimes['connection'] = self.paths['connection'].stat().st_mtime
            
            # 4. 加载数据库配置
            if self.paths['database'].exists():
                try:
                    self._load_database_config()
                except Exception as e:
                    self.logger.error(f"加载数据库配置failed: {e}")
                finally:
                    # 无论成功失败都更新文件修改时间，避免无限重试
                    self._file_mtimes['database'] = self.paths['database'].stat().st_mtime
            
            # 标记缓存失效
            self._cache_valid = False
            self._last_update = time.time()
            
            self.logger.info("Configuration loading completed")
    
    def _files_changed(self) -> bool:
        """check配置文件is否修改"""
        for file_key, path in self.paths.items():
            if path.exists():
                current_mtime = path.stat().st_mtime
                if file_key not in self._file_mtimes or current_mtime > self._file_mtimes[file_key]:
                    return True
        return False
    
    def _load_database_config(self):
        """from数据库加载配置"""
        conn = sqlite3.connect(self.paths['database'])
        cursor = conn.cursor()
        
        try:
            # 加载风险配置
            cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='risk_configs'
            """)
            
            if cursor.fetchone():
                # 修复：使use实际表结构 (config_json列而notiskey,value列)
                cursor.execute("""
                    SELECT config_json FROM risk_configs 
                    WHERE name = '默认风险配置'
                """)
                
                result = cursor.fetchone()
                if result:
                    try:
                        # 解析JSON配置
                        db_risk_config = json.loads(result[0])
                        if db_risk_config:
                            self._configs['database']['risk_management'] = db_risk_config
                            self.logger.info(f"Loaded database risk configuration: {len(db_risk_config)} items")
                    except Exception as e:
                        self.logger.warning(f"解析数据库风险配置failed: {e}")
            
            # 加载全局tickers作asuniverse（兼容not同表结构）
            try:
                cursor.execute("SELECT symbol FROM tickers WHERE is_active = 1")
                tickers = [row[0] for row in cursor.fetchall()]
            except Exception:
                # if果没hasis_active列，直接查询所hassymbol
                try:
                    cursor.execute("SELECT symbol FROM tickers")
                    tickers = [row[0] for row in cursor.fetchall()]
                except Exception as e:
                    self.logger.debug(f"tickers表查询failed: {e}")
                    tickers = []
            
            if tickers:
                self._configs['database']['scanner'] = {'universe': tickers}
                self.logger.info(f"Loaded database tickers: {len(tickers)} records")
            
            # 加载引擎配置（can选表，can能not存in）
            try:
                cursor.execute("""
                    SELECT name FROM sqlite_master WHERE type='table' AND name='engine_configs'
                """)
                
                if cursor.fetchone():
                    cursor.execute("SELECT section, key, value FROM engine_configs")
                    engine_config = {}
                    
                    for section, key, value in cursor.fetchall():
                        if section not in engine_config:
                            engine_config[section] = {}
                        try:
                            engine_config[section][key] = json.loads(value)
                        except:
                            engine_config[section][key] = value
                    
                    if engine_config:
                        self._configs['database'].update(engine_config)
                        self.logger.info(f"加载数据库引擎配置: {len(engine_config)}节")
                else:
                    self.logger.debug("engine_configs表not存in，跳过引擎配置加载")
            except Exception as e:
                self.logger.debug(f"引擎配置加载failed: {e}")
                
        finally:
            conn.close()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """retrieval配置值（支持点号路径）"""
        with self.lock:
            # 自动重加载check
            if time.time() - self._last_update > 60:  # 60 secondscheck一次
                # 仅在文件有变化时才触发加载并打印日志
                if self._files_changed():
                    self.load_all(force=True)
            
            config = self._get_merged_config()
            
            # 支持点号路径访问
            keys = key_path.split('.')
            value = config
            
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                    if value is None:
                        return default
                else:
                    return default
            
            return value if value is not None else default
    
    def set_runtime(self, key_path: str, value: Any):
        """settings运行when配置（最高优先级）"""
        with self.lock:
            keys = key_path.split('.')
            config = self._configs['runtime']
            
            # 创建嵌套结构
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            config[keys[-1]] = value
            self._cache_valid = False
            
            self.logger.debug(f"settings运行when配置: {key_path} = {value}")
    
    def update_runtime_config(self, updates: Dict[str, Any]):
        """批量updates运行when配置"""
        with self.lock:
            for key_path, value in updates.items():
                self.set_runtime(key_path, value)
                
    def save_to_file(self, config_type: str = 'hotconfig') -> bool:
        """保存配置to文件（持久化）"""
        try:
            with self.lock:
                if config_type == 'hotconfig':
                    # 保存合并after配置tohotconfig文件
                    merged = self._get_merged_config()
                    
                    # 确保目录存in
                    self.paths['hotconfig'].parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(self.paths['hotconfig'], 'w', encoding='utf-8') as f:
                        json.dump(merged, f, indent=2, ensure_ascii=False)
                    
                    # updates文件层配置
                    self._configs['file'] = merged.copy()
                    self.logger.info(f"配置保存to {self.paths['hotconfig']}")
                    return True
                    
                elif config_type == 'connection':
                    # 保存connection配置
                    conn_config = self.get_connection_params()
                    
                    self.paths['connection'].parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(self.paths['connection'], 'w', encoding='utf-8') as f:
                        json.dump(conn_config, f, indent=2, ensure_ascii=False)
                    
                    self.logger.info(f"connection配置保存to {self.paths['connection']}")
                    return True
                    
        except Exception as e:
            self.logger.error(f"保存配置failed: {e}")
            return False
            
    def persist_runtime_changes(self):
        """will运行when配置持久化to文件"""
        try:
            with self.lock:
                if self._configs['runtime']:
                    # willruntime配置合并tofile配置in
                    runtime_config = deepcopy(self._configs['runtime'])
                    file_config = deepcopy(self._configs['file'])
                    
                    # depth合并
                    def deep_merge(base, updates):
                        for key, value in updates.items():
                            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                                deep_merge(base[key], value)
                            else:
                                base[key] = value
                    
                    deep_merge(file_config, runtime_config)
                    
                    # 保存tohotconfig文件
                    self.paths['hotconfig'].parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(self.paths['hotconfig'], 'w', encoding='utf-8') as f:
                        json.dump(file_config, f, indent=2, ensure_ascii=False)
                    
                    # updatesfile配置并清空runtime
                    self._configs['file'] = file_config
                    self._configs['runtime'] = {}
                    self._cache_valid = False
                    
                    self.logger.info("运行when配置持久化")
                    return True
                    
        except Exception as e:
            self.logger.error(f"持久化运行when配置failed: {e}")
            return False
    
    def _get_merged_config(self) -> Dict[str, Any]:
        """retrieval合并after配置"""
        if self._cache_valid and self._merged_config:
            # 缓存有效时不触发任何加载日志，直接返回
            return self._merged_config
        
        # 按优先级合并配置
        merged = {}
        for source in ['default', 'file', 'database', 'hotconfig', 'runtime']:
            if self._configs[source]:
                self._deep_merge(merged, self._configs[source])
                self.logger.debug(f"合并配置源: {source}")
        
        # 原子性settings缓存
        self._merged_config = merged
        self._cache_valid = True
        
        return merged
    
    def _deep_merge(self, target: Dict, source: Dict):
        """depth合并字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = deepcopy(value)
    
    def get_universe(self) -> List[str]:
        """retrieval统一股票列表（使use数据源管理器）"""
        try:
            from .data_source_manager import get_data_source_manager
            
            # 使use统一数据源管理器
            data_manager = get_data_source_manager()
            universe = data_manager.get_universe()
            
            if universe:
                return universe
            
        except Exception as e:
            self.logger.warning(f"数据源管理器retrievalfailed: {e}")
        
        # 降级to配置文件
        universe = self.get('scanner.universe', [])
        
        if not universe:
            self.logger.warning("未配置股票列表，使use默认")
            universe = self._configs['default']['scanner']['universe']
        
        # 确保唯一性并排序
        return sorted(list(set(universe)))
    
    def get_connection_params(self, auto_allocate_client_id: bool = True) -> Dict[str, Any]:
        """retrievalconnection参数"""
        host = self.get('connection.host')
        port = self.get('connection.port')
        
        if auto_allocate_client_id:
            # 使use动态ClientID分配
            try:
                from .client_id_manager import allocate_dynamic_client_id
                preferred_id = self.get('connection.client_id')
                client_id = allocate_dynamic_client_id(host, port, preferred_id)
                self.logger.info(f"Assigned dynamic ClientID: {client_id}")
            except Exception as e:
                self.logger.warning(f"动态ClientID分配failed，使use配置值: {e}")
                client_id = self.get('connection.client_id')
        else:
            client_id = self.get('connection.client_id')
        
        return {
            'host': host,
            'port': port,
            'client_id': client_id,
            'account_id': self.get('connection.account_id'),
            'use_delayed_if_no_realtime': self.get('connection.use_delayed_if_no_realtime'),
            'timeout': self.get('connection.timeout')
        }
    
    def save_to_file(self, filepath: Optional[str] = None):
        """保存当before合并配置to文件"""
        if not filepath:
            filepath = self.base_dir / 'unified_config.json'
        
        with self.lock:
            config = self._get_merged_config()
            
            # 创建目录
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"配置保存to: {filepath}")
    
    def save_runtime_to_hotconfig(self):
        """will运行when配置保存toHotConfig文件"""
        with self.lock:
            if not self._configs['runtime']:
                return
            
            # 读取现hasHotConfig
            hot_config = {}
            if self.paths['hotconfig'].exists():
                try:
                    with open(self.paths['hotconfig'], 'r', encoding='utf-8') as f:
                        hot_config = json.load(f)
                except Exception as e:
                    self.logger.error(f"读取HotConfigfailed: {e}")
            
            # 合并运行when配置
            if 'CONFIG' not in hot_config:
                hot_config['CONFIG'] = {}
            
            self._deep_merge(hot_config['CONFIG'], self._configs['runtime'])
            
            # 保存
            try:
                self.paths['hotconfig'].parent.mkdir(parents=True, exist_ok=True)
                with open(self.paths['hotconfig'], 'w', encoding='utf-8') as f:
                    json.dump(hot_config, f, indent=2, ensure_ascii=False)
                
                self.logger.info("运行when配置保存toHotConfig")
            except Exception as e:
                self.logger.error(f"保存HotConfigfailed: {e}")
    
    def validate(self) -> List[str]:
        """验证配置一致性"""
        issues = []
        config = self._get_merged_config()
        
        # check必要字段
        if not config.get('connection', {}).get('host'):
            issues.append("缺少connection主机地址")
        
        if not config.get('connection', {}).get('client_id'):
            issues.append("缺少客户端ID")
        
        # check数值范围
        cash_reserve = config.get('capital', {}).get('cash_reserve_pct', 0)
        if not 0 <= cash_reserve <= 1:
            issues.append(f"现金预留比例异常: {cash_reserve}")
        
        # checkuniverse
        universe = config.get('scanner', {}).get('universe', [])
        if len(universe) == 0:
            issues.append("股票列表as空")
        elif len(universe) > 500:
            issues.append(f"股票列表过大: {len(universe)}")
        
        # check端口范围
        port = config.get('connection', {}).get('port', 0)
        if port not in [4001, 4002, 7496, 7497]:
            issues.append(f"not标准IBKR端口: {port}")
        
        return issues
    
    def print_config_sources(self):
        """打印所has配置源内容"""
        with self.lock:
            print("\n" + "="*60)
            print("配置源详细信息")
            print("="*60)
            
            for source, config in self._configs.items():
                if config:
                    print(f"\n=== {source.upper()} (优先级: {self.PRIORITY.get(source, 0)}) ===")
                    self._print_dict(config, indent=2)
                else:
                    print(f"\n=== {source.upper()} === (空)")
    
    def _print_dict(self, d: Dict, indent: int = 0):
        """递归打印字典"""
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                self._print_dict(value, indent + 2)
            elif isinstance(value, list) and len(value) > 5:
                print(" " * indent + f"{key}: [{len(value)} items] {value[:3]}...")
            else:
                print(" " * indent + f"{key}: {value}")
    
    def get_config_conflicts(self) -> Dict[str, List[str]]:
        """检测配置冲突"""
        conflicts = {}
        
        # check同一个配置 itemsinnot同源innot同值
        for source1 in self._configs:
            for source2 in self._configs:
                if source1 >= source2:  # 避免重复比较
                    continue
                
                conflicts_found = self._find_conflicts(
                    self._configs[source1], 
                    self._configs[source2],
                    source1, source2
                )
                
                if conflicts_found:
                    key = f"{source1}_vs_{source2}"
                    conflicts[key] = conflicts_found
        
        return conflicts
    
    def _find_conflicts(self, config1: Dict, config2: Dict, source1: str, source2: str, path: str = "") -> List[str]:
        """查找两个配置字典间冲突"""
        conflicts = []
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key in config1 and key in config2:
                val1, val2 = config1[key], config2[key]
                
                if isinstance(val1, dict) and isinstance(val2, dict):
                    # 递归check嵌套字典
                    sub_conflicts = self._find_conflicts(val1, val2, source1, source2, current_path)
                    conflicts.extend(sub_conflicts)
                elif val1 != val2:
                    # 值not同
                    conflicts.append(f"{current_path}: {source1}={val1} vs {source2}={val2}")
        
        return conflicts
    
    def clear_runtime_config(self):
        """清空运行when配置"""
        with self.lock:
            self._configs['runtime'].clear()
            self._cache_valid = False
            self.logger.info("清空运行when配置")


# 全局配置管理器实例
_global_config_manager: Optional[UnifiedConfigManager] = None

def get_unified_config() -> UnifiedConfigManager:
    """retrieval全局统一配置管理器"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = UnifiedConfigManager()
    return _global_config_manager

def reload_all_configs():
    """重新加载所has配置"""
    global _global_config_manager
    if _global_config_manager:
        _global_config_manager.load_all(force=True)
