#!/usr/bin/env python3
"""
连接管理简化脚本
将分散的连接管理统一到unified_connection_manager.py
"""

import logging
from typing import Optional, Dict, Any

# 设置统一连接管理器为主管理器
from .unified_connection_manager import UnifiedConnectionManager, ConnectionConfig

logger = logging.getLogger(__name__)

class ConnectionSimplifier:
    """连接管理简化器 - 提供统一的连接接口"""
    
    def __init__(self):
        self._primary_manager = None
        self._config = ConnectionConfig(
            host="127.0.0.1",
            port=4002,  # 使用统一配置的端口
            client_id=3130,
            timeout=20.0
        )
        
    def get_primary_connection_manager(self) -> UnifiedConnectionManager:
        """获取主连接管理器（单例模式）"""
        if self._primary_manager is None:
            # 创建一个模拟的IB客户端用于测试
            try:
                from ib_async import IB
                mock_ib = IB()
            except ImportError:
                # 如果ib_insync不可用，创建一个模拟对象
                mock_ib = type('MockIB', (), {})()
            
            self._primary_manager = UnifiedConnectionManager(
                ib_client=mock_ib,
                config=self._config
            )
        return self._primary_manager
    
    async def connect_unified(self) -> bool:
        """统一连接方法 - 替换所有分散的连接函数"""
        try:
            manager = self.get_primary_connection_manager()
            return await manager.connect()
        except Exception as e:
            logger.error(f"统一连接失败: {e}")
            return False
    
    async def disconnect_unified(self) -> bool:
        """统一断开连接方法"""
        try:
            manager = self.get_primary_connection_manager()
            return await manager.disconnect()
        except Exception as e:
            logger.error(f"统一断开连接失败: {e}")
            return False
    
    def get_connection_status(self) -> str:
        """获取连接状态"""
        if self._primary_manager is None:
            return "DISCONNECTED"
        # 检查管理器是否有状态属性
        if hasattr(self._primary_manager, 'state'):
            return self._primary_manager.state.value
        elif hasattr(self._primary_manager, '_state'):
            return self._primary_manager._state.value
        else:
            return "UNKNOWN"
    
    async def test_connection_unified(self) -> bool:
        """统一连接测试方法"""
        try:
            manager = self.get_primary_connection_manager()
            return await manager.test_connection()
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False

# 全局简化器实例
_connection_simplifier = None

def get_connection_simplifier() -> ConnectionSimplifier:
    """获取连接简化器单例"""
    global _connection_simplifier
    if _connection_simplifier is None:
        _connection_simplifier = ConnectionSimplifier()
    return _connection_simplifier

# 提供向后兼容的接口函数
async def unified_connect() -> bool:
    """统一连接接口"""
    simplifier = get_connection_simplifier()
    return await simplifier.connect_unified()

async def unified_disconnect() -> bool:
    """统一断开连接接口"""
    simplifier = get_connection_simplifier()
    return await simplifier.disconnect_unified()

async def unified_test_connection() -> bool:
    """统一连接测试接口"""
    simplifier = get_connection_simplifier()
    return await simplifier.test_connection_unified()

def unified_get_connection_status() -> str:
    """统一获取连接状态接口"""
    simplifier = get_connection_simplifier()
    return simplifier.get_connection_status()