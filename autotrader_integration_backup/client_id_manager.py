#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClientID 动态分配管理器
解决多实例运行whenClientID冲突问题
"""

import os
import time
import json
import random
import logging
from pathlib import Path
from typing import Optional, Dict
# 清理：移除未使use导入
# import socket
# from typing import Set
from threading import Lock
from dataclasses import dataclass, asdict

# 跨平台文件锁支持
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # Windows环境下fcntlnotcanuse，使use替代方案
    HAS_FCNTL = False

@dataclass
class ClientIDRegistration:
    """ClientID注册信息"""
    client_id: int
    process_id: int
    host: str
    port: int
    timestamp: float
    heartbeat: float
    process_name: str

class DynamicClientIDManager:
    """动态ClientID分配管理器"""
    
    def __init__(self, registry_file: str = "data/client_ids.json"):
        self.registry_file = Path(registry_file)
        self.logger = logging.getLogger("ClientIDManager")
        self.lock = Lock()
        
        # ClientID范围配置
        try:
            from .config_manager import get_config_manager
            config = get_config_manager()
            self.min_client_id = config.get('ibkr.min_client_id', 1000)
            self.max_client_id = config.get('ibkr.max_client_id', 9999)
            reserved_ports = config.get('ibkr.reserved_ports', [7496, 7497])
            self.reserved_ids = set(reserved_ports)
        except:
            # Fallback to defaults
            self.min_client_id = 1000
            self.max_client_id = 9999
            self.reserved_ids = {7496, 7497}
        
        # 当before分配ID
        self.current_client_id: Optional[int] = None
        self.registration: Optional[ClientIDRegistration] = None
        
        # 确保注册文件目录存in
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 清理过期注册
        self._cleanup_expired_registrations()
    
    def allocate_client_id(self, host: Optional[str] = None, port: Optional[int] = None, 
                          preferred_id: Optional[int] = None) -> int:
        """分配一个canuseClientID"""
        with self.lock:
            try:
                # Use defaults from config if not provided
                try:
                    from .config_manager import get_config_manager
                    config = get_config_manager()
                    host = host or config.get('ibkr.host', '127.0.0.1')
                    port = port or config.get('ibkr.port', 7497)
                except:
                    host = host or '127.0.0.1'
                    port = port or 7497
                
                # if果指定了首选ID且canuse，优先使use
                if preferred_id and self._is_client_id_available(preferred_id, host, port):
                    client_id = preferred_id
                else:
                    # 查找canuseID
                    client_id = self._find_available_client_id(host, port)
                
                # 注册ClientID
                self._register_client_id(client_id, host, port)
                self.current_client_id = client_id
                
                self.logger.info(f"Assigned ClientID: {client_id} (Host: {host}:{port})")
                return client_id
                
            except Exception as e:
                self.logger.error(f"ClientID分配failed: {e}")
                # 回退to随机ID
                fallback_id = self._generate_fallback_id()
                self.logger.warning(f"使use回退ClientID: {fallback_id}")
                return fallback_id
    
    def _find_available_client_id(self, host: str, port: int) -> int:
        """查找canuseClientID"""
        active_registrations = self._load_active_registrations()
        
        # retrieval使useID
        used_ids = set()
        for reg in active_registrations.values():
            if reg.host == host and reg.port == port:
                used_ids.add(reg.client_id)
        
        # 添加保留ID
        used_ids.update(self.reserved_ids)
        
        # 查找canuseID
        for client_id in range(self.min_client_id, self.max_client_id + 1):
            if client_id not in used_ids:
                return client_id
        
        # if果没hascanuseID，使use随机ID
        return self._generate_fallback_id()
    
    def _is_client_id_available(self, client_id: int, host: str, port: int) -> bool:
        """checkClientIDis否canuse"""
        if client_id in self.reserved_ids:
            return False
        
        active_registrations = self._load_active_registrations()
        
        for reg in active_registrations.values():
            if (reg.client_id == client_id and 
                reg.host == host and 
                reg.port == port):
                # check进程is否还活着
                if self._is_process_alive(reg.process_id):
                    return False
        
        return True
    
    def _register_client_id(self, client_id: int, host: str, port: int):
        """注册ClientID"""
        registration = ClientIDRegistration(
            client_id=client_id,
            process_id=os.getpid(),
            host=host,
            port=port,
            timestamp=time.time(),
            heartbeat=time.time(),
            process_name=f"autotrader-{os.getpid()}"
        )
        
        self.registration = registration
        self._save_registration(registration)
    
    def update_heartbeat(self):
        """updates心跳when间"""
        if self.registration:
            self.registration.heartbeat = time.time()
            self._save_registration(self.registration)
    
    def release_client_id(self):
        """释放ClientID"""
        if self.current_client_id and self.registration:
            self._remove_registration(self.registration)
            self.logger.info(f"释放ClientID: {self.current_client_id}")
            self.current_client_id = None
            self.registration = None
    
    def _load_active_registrations(self) -> Dict[str, ClientIDRegistration]:
        """加载活跃注册信息"""
        if not self.registry_file.exists():
            return {}
        
        try:
            with open(self.registry_file, 'r') as f:
                # 使use文件锁（if果canuse）
                if HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = json.load(f)
            
            registrations = {}
            current_time = time.time()
            
            for key, reg_data in data.items():
                reg = ClientIDRegistration(**reg_data)
                
                # check注册is否过期（30分钟no心跳）
                if (current_time - reg.heartbeat < 1800 and 
                    self._is_process_alive(reg.process_id)):
                    registrations[key] = reg
            
            return registrations
            
        except Exception as e:
            self.logger.warning(f"加载注册文件failed: {e}")
            return {}
    
    def _save_registration(self, registration: ClientIDRegistration):
        """保存注册信息"""
        try:
            # 加载现has注册
            registrations = self._load_active_registrations()
            
            # 添加/updates当before注册
            key = f"{registration.process_id}_{registration.client_id}"
            registrations[key] = registration
            
            # 保存to文件
            with open(self.registry_file, 'w') as f:
                if HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(
                    {k: asdict(v) for k, v in registrations.items()}, 
                    f, 
                    indent=2
                )
            
        except Exception as e:
            self.logger.error(f"保存注册信息failed: {e}")
    
    def _remove_registration(self, registration: ClientIDRegistration):
        """移除注册信息"""
        try:
            registrations = self._load_active_registrations()
            key = f"{registration.process_id}_{registration.client_id}"
            
            if key in registrations:
                del registrations[key]
                
                with open(self.registry_file, 'w') as f:
                    if HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    json.dump(
                        {k: asdict(v) for k, v in registrations.items()}, 
                        f, 
                        indent=2
                    )
            
        except Exception as e:
            self.logger.error(f"移除注册信息failed: {e}")
    
    def _cleanup_expired_registrations(self):
        """清理过期注册信息"""
        try:
            if not self.registry_file.exists():
                return
            
            active_registrations = self._load_active_registrations()
            
            # 重新保存活跃注册（自动清理过期）
            with open(self.registry_file, 'w') as f:
                if HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(
                    {k: asdict(v) for k, v in active_registrations.items()}, 
                    f, 
                    indent=2
                )
            
            self.logger.debug(f"清理completed，活跃注册数: {len(active_registrations)}")
            
        except Exception as e:
            self.logger.warning(f"清理过期注册failed: {e}")
    
    def _is_process_alive(self, pid: int) -> bool:
        """check进程is否还活着"""
        try:
            # Windows兼容进程check
            import platform
            if platform.system() == "Windows":
                try:
                    import psutil
                    return psutil.pid_exists(pid)
                except ImportError:
                    # 回退toos.kill方法，但加强异常处理
                    try:
                        os.kill(pid, 0)
                        return True
                    except (OSError, PermissionError, ProcessLookupError):
                        return False
            else:
                # Unix/Linux系统
                os.kill(pid, 0)
                return True
        except Exception:
            # 所has其他异常都视as进程not存in
            return False
    
    def _generate_fallback_id(self) -> int:
        """生成回退ClientID"""
        from .production_random_control import get_random_control
        control = get_random_control()
        return int(control.controlled_random_call('random_uniform', low=5000, high=8999))
    
    def get_registry_status(self) -> Dict:
        """retrieval注册状态信息"""
        active_registrations = self._load_active_registrations()
        
        return {
            'current_client_id': self.current_client_id,
            'registry_file': str(self.registry_file),
            'active_registrations': len(active_registrations),
            'registrations': [
                {
                    'client_id': reg.client_id,
                    'process_id': reg.process_id,
                    'host_port': f"{reg.host}:{reg.port}",
                    'process_name': reg.process_name,
                    'uptime': time.time() - reg.timestamp
                }
                for reg in active_registrations.values()
            ]
        }

# 全局实例
_global_client_id_manager: Optional[DynamicClientIDManager] = None

def get_client_id_manager() -> DynamicClientIDManager:
    """retrieval全局ClientID管理器实例"""
    global _global_client_id_manager
    if _global_client_id_manager is None:
        _global_client_id_manager = DynamicClientIDManager()
    return _global_client_id_manager

def allocate_dynamic_client_id(host: Optional[str] = None, port: Optional[int] = None, 
                              preferred_id: Optional[int] = None) -> int:
    """便捷函数：Assigned dynamic ClientID"""
    manager = get_client_id_manager()
    return manager.allocate_client_id(host, port, preferred_id)

def release_dynamic_client_id():
    """便捷函数：释放ClientID"""
    manager = get_client_id_manager()
    manager.release_client_id()

def update_client_id_heartbeat():
    """便捷函数：updates心跳"""
    manager = get_client_id_manager()
    manager.update_heartbeat()
