#!/usr/bin/env python3
"""
系统状态管理器 - 保存和恢复交易系统状态
支持崩溃恢复、热重启、状态审计
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from .system_paths import resolve_data_path


class StateManager:
    """交易系统状态管理器"""

    def __init__(self, state_file: str = "autotrader_state.json"):
        """
        初始化状态管理器

        Args:
            state_file: 状态文件名（将保存在data目录）
        """
        self.state_file = resolve_data_path(state_file)
        self.logger = logging.getLogger("StateManager")
        self.state = self.load_state()

    def save_state(self, trader: Any, engine: Any = None) -> bool:
        """
        保存当前交易状态

        Args:
            trader: IbkrAutoTrader实例
            engine: Engine实例（可选）

        Returns:
            是否保存成功
        """
        try:
            state = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'version': '1.0',

                # 账户信息
                'account': {
                    'net_liq': getattr(trader, 'net_liq', 0.0),
                    'cash_balance': getattr(trader, 'cash_balance', 0.0),
                    'buying_power': getattr(trader, 'buying_power', 0.0),
                },

                # 持仓信息（从position_manager获取）
                'positions': self._extract_positions(trader),

                # 未完成订单
                'open_orders': self._extract_open_orders(trader),

                # 运行统计
                'stats': {
                    'daily_order_count': getattr(trader, '_daily_order_count', 0),
                    'connection_status': 'connected' if getattr(trader, 'is_connected', False) else 'disconnected',
                },

                # 风险状态
                'risk': self._extract_risk_state(trader),
            }

            # 保存到文件
            path = Path(self.state_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')

            self.logger.info(f"状态已保存: {len(state.get('positions', {}))}个持仓, {len(state.get('open_orders', []))}个未完成订单")
            return True

        except Exception as e:
            self.logger.error(f"保存状态失败: {e}", exc_info=True)
            return False

    def load_state(self) -> Dict[str, Any]:
        """
        加载上次保存的状态

        Returns:
            状态字典，如果加载失败或状态过期则返回空字典
        """
        try:
            path = Path(self.state_file)
            if not path.exists():
                self.logger.info("状态文件不存在，首次运行")
                return {}

            state = json.loads(path.read_text(encoding='utf-8'))

            # 检查状态新鲜度（超过24小时视为过期）
            timestamp = state.get('timestamp', 0)
            age_hours = (time.time() - timestamp) / 3600

            if age_hours > 24:
                self.logger.warning(f"状态文件过期（{age_hours:.1f}小时前），丢弃")
                return {}

            self.logger.info(f"加载状态成功: 保存于 {state.get('datetime')}")
            return state

        except Exception as e:
            self.logger.error(f"加载状态失败: {e}")
            return {}

    def restore_state(self, trader: Any) -> bool:
        """
        恢复状态到trader（仅验证，不自动修改）

        Args:
            trader: IbkrAutoTrader实例

        Returns:
            是否有状态可以恢复
        """
        if not self.state:
            self.logger.info("无可恢复状态")
            return False

        try:
            # 1. 验证账户信息
            saved_account = self.state.get('account', {})
            current_net_liq = getattr(trader, 'net_liq', 0.0)

            self.logger.info(f"保存的账户净值: ${saved_account.get('net_liq', 0):.2f}")
            self.logger.info(f"当前账户净值: ${current_net_liq:.2f}")

            # 2. 比对持仓
            saved_positions = self.state.get('positions', {})
            if saved_positions and hasattr(trader, 'position_manager'):
                actual_positions = {
                    symbol: pos.quantity
                    for symbol, pos in trader.position_manager.get_all_positions().items()
                }

                # 检查差异
                for symbol, saved_qty in saved_positions.items():
                    actual_qty = actual_positions.get(symbol, 0)
                    if actual_qty != saved_qty:
                        self.logger.warning(
                            f"持仓差异 {symbol}: 保存={saved_qty}, 实际={actual_qty}"
                        )

                # 检查新增持仓
                for symbol in actual_positions:
                    if symbol not in saved_positions:
                        self.logger.warning(
                            f"新增持仓 {symbol}: {actual_positions[symbol]}"
                        )

            # 3. 检查未完成订单
            saved_orders = self.state.get('open_orders', [])
            if saved_orders:
                self.logger.info(f"保存了{len(saved_orders)}个未完成订单，请手动核对")

            return True

        except Exception as e:
            self.logger.error(f"恢复状态失败: {e}", exc_info=True)
            return False

    def _extract_positions(self, trader: Any) -> Dict[str, int]:
        """提取持仓信息"""
        try:
            if hasattr(trader, 'position_manager'):
                return {
                    symbol: pos.quantity
                    for symbol, pos in trader.position_manager.get_all_positions().items()
                }
        except Exception as e:
            self.logger.warning(f"提取持仓失败: {e}")

        return {}

    def _extract_open_orders(self, trader: Any) -> list:
        """提取未完成订单信息"""
        try:
            if hasattr(trader, 'open_orders'):
                orders = []
                for order_id, order_info in getattr(trader, 'open_orders', {}).items():
                    orders.append({
                        'order_id': order_id,
                        'symbol': order_info.get('symbol', ''),
                        'action': order_info.get('action', ''),
                        'quantity': order_info.get('quantity', 0),
                        'status': order_info.get('status', ''),
                    })
                return orders
        except Exception as e:
            self.logger.warning(f"提取订单失败: {e}")

        return []

    def _extract_risk_state(self, trader: Any) -> Dict[str, Any]:
        """提取风险状态"""
        try:
            if hasattr(trader, 'risk_manager'):
                risk_manager = trader.risk_manager
                return {
                    'daily_order_count': getattr(risk_manager, 'daily_order_count', 0),
                    'daily_pnl': getattr(risk_manager, 'daily_pnl', 0.0),
                }
        except Exception as e:
            self.logger.warning(f"提取风险状态失败: {e}")

        return {}

    def clear_state(self) -> bool:
        """清除保存的状态"""
        try:
            path = Path(self.state_file)
            if path.exists():
                path.unlink()
                self.logger.info("状态文件已清除")
            self.state = {}
            return True
        except Exception as e:
            self.logger.error(f"清除状态失败: {e}")
            return False

    def get_state_age(self) -> Optional[float]:
        """获取状态年龄（小时）"""
        if not self.state:
            return None

        timestamp = self.state.get('timestamp', 0)
        if timestamp == 0:
            return None

        return (time.time() - timestamp) / 3600


# 全局实例（单例）
_state_manager: Optional[StateManager] = None


def get_state_manager(state_file: str = "autotrader_state.json") -> StateManager:
    """获取全局状态管理器实例（单例模式）"""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager(state_file)
    return _state_manager
