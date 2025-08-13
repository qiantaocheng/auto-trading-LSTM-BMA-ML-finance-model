#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
账户数据管理器 - 解决账户数据获取的竞态条件和多币种支持
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from threading import Lock
from collections import defaultdict
from enum import Enum

class CurrencyPriority(Enum):
    """货币优先级"""
    BASE = 1      # 基础货币
    USD = 2       # 美元
    PRIMARY = 3   # 主要货币
    ANY = 4       # 任意货币

@dataclass
class AccountSnapshot:
    """账户快照"""
    timestamp: float
    cash_balance: float
    net_liquidation: float
    buying_power: float
    positions: Dict[str, int] = field(default_factory=dict)
    account_values: Dict[str, str] = field(default_factory=dict)
    currency: str = "USD"
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

class RobustAccountDataManager:
    """稳健的账户数据管理器"""
    
    def __init__(self, ib_client, account_id: str = ""):
        self.ib = ib_client
        self.account_id = account_id
        self.logger = logging.getLogger("AccountDataManager")
        
        # 数据保护锁
        self.data_lock = Lock()
        self.refresh_lock = asyncio.Lock()
        
        # 缓存和历史
        self.current_snapshot: Optional[AccountSnapshot] = None
        self.snapshot_history: List[AccountSnapshot] = []
        self.max_history = 100
        
        # 配置
        self.refresh_timeout = 15.0
        self.validation_enabled = True
        self.currency_priorities = [
            CurrencyPriority.BASE,
            CurrencyPriority.USD,
            CurrencyPriority.PRIMARY,
            CurrencyPriority.ANY
        ]
        
        # 统计
        self.refresh_count = 0
        self.error_count = 0
        self.last_successful_refresh = 0.0
    
    async def refresh_with_retry(self, max_retries: int = 3, 
                               timeout: Optional[float] = None) -> AccountSnapshot:
        """带重试的账户数据刷新"""
        timeout = timeout or self.refresh_timeout
        
        async with self.refresh_lock:
            for attempt in range(max_retries):
                try:
                    # 执行刷新
                    snapshot = await asyncio.wait_for(
                        self._perform_refresh(), 
                        timeout=timeout
                    )
                    
                    # 验证数据
                    if self.validation_enabled:
                        validation_result = self._validate_snapshot(snapshot)
                        snapshot.is_valid = validation_result['is_valid']
                        snapshot.validation_errors = validation_result['errors']
                        
                        if not snapshot.is_valid and attempt < max_retries - 1:
                            self.logger.warning(f"账户数据验证失败 (尝试 {attempt + 1}): {snapshot.validation_errors}")
                            await asyncio.sleep(1.0 * (attempt + 1))
                            continue
                    
                    # 保存快照
                    self._save_snapshot(snapshot)
                    self.refresh_count += 1
                    self.last_successful_refresh = time.time()
                    
                    self.logger.debug(f"账户数据刷新成功 (尝试 {attempt + 1})")
                    return snapshot
                    
                except asyncio.TimeoutError:
                    self.error_count += 1
                    if attempt == max_retries - 1:
                        self.logger.error(f"账户数据刷新超时 ({timeout}秒)")
                        raise
                    else:
                        self.logger.warning(f"账户数据刷新超时，重试 {attempt + 1}/{max_retries}")
                        await asyncio.sleep(2.0 * (attempt + 1))
                        
                except Exception as e:
                    self.error_count += 1
                    if attempt == max_retries - 1:
                        self.logger.error(f"账户数据刷新失败: {e}")
                        raise
                    else:
                        self.logger.warning(f"账户数据刷新失败，重试 {attempt + 1}/{max_retries}: {e}")
                        await asyncio.sleep(1.0 * (attempt + 1))
    
    async def _perform_refresh(self) -> AccountSnapshot:
        """执行实际的账户数据刷新"""
        refresh_start = time.time()
        
        # 请求账户摘要
        await self.ib.reqAccountUpdatesAsync(account=self.account_id)
        
        # 等待数据更新
        await asyncio.sleep(0.5)
        
        # 提取账户数据
        account_values = {}
        for av in self.ib.accountValues():
            if not self.account_id or av.account == self.account_id:
                key = f"{av.tag}:{av.currency}" if av.currency else av.tag
                account_values[key] = av.value
        
        # 多币种支持的数值提取
        cash_balance = self._extract_account_numeric(account_values, "CashBalance")
        net_liquidation = self._extract_account_numeric(account_values, "NetLiquidation") 
        buying_power = self._extract_account_numeric(account_values, "BuyingPower")
        
        # 提取持仓信息
        positions = {}
        for pos in self.ib.positions():
            if not self.account_id or pos.account == self.account_id:
                positions[pos.contract.symbol] = int(pos.position)
        
        # 确定主要货币
        primary_currency = self._determine_primary_currency(account_values)
        
        snapshot = AccountSnapshot(
            timestamp=time.time(),
            cash_balance=cash_balance,
            net_liquidation=net_liquidation,
            buying_power=buying_power,
            positions=positions,
            account_values=account_values,
            currency=primary_currency
        )
        
        self.logger.debug(f"账户刷新耗时: {time.time() - refresh_start:.2f}秒")
        return snapshot
    
    def _extract_account_numeric(self, account_values: Dict[str, str], 
                               tag: str) -> float:
        """多币种支持的账户数值提取"""
        candidates: Dict[CurrencyPriority, List[Tuple[str, float]]] = defaultdict(list)
        
        # 按货币优先级分类候选值
        for key, value in account_values.items():
            if key.startswith(f"{tag}:"):
                currency = key.split(":", 1)[1]
                try:
                    numeric_value = float(value)
                    priority = self._get_currency_priority(currency)
                    candidates[priority].append((currency, numeric_value))
                except (ValueError, TypeError):
                    continue
        
        # 按优先级选择最佳值
        for priority in self.currency_priorities:
            if candidates[priority]:
                # 如果有BASE货币，优先选择
                if priority == CurrencyPriority.BASE:
                    return candidates[priority][0][1]
                
                # 对于USD和其他货币，选择最大值（通常更准确）
                best_value = max(candidates[priority], key=lambda x: abs(x[1]))
                return best_value[1]
        
        # 如果没有找到匹配的tag，尝试不带货币的版本
        if tag in account_values:
            try:
                return float(account_values[tag])
            except (ValueError, TypeError):
                pass
        
        self.logger.warning(f"无法提取账户数值: {tag}")
        return 0.0
    
    def _get_currency_priority(self, currency: str) -> CurrencyPriority:
        """获取货币优先级"""
        currency = currency.upper()
        
        if currency == "BASE":
            return CurrencyPriority.BASE
        elif currency == "USD":
            return CurrencyPriority.USD
        elif currency in ["EUR", "GBP", "JPY", "CHF", "CAD", "AUD"]:
            return CurrencyPriority.PRIMARY
        else:
            return CurrencyPriority.ANY
    
    def _determine_primary_currency(self, account_values: Dict[str, str]) -> str:
        """确定账户的主要货币"""
        # 检查BASE货币
        for key in account_values:
            if ":BASE" in key:
                return "BASE"
        
        # 检查最大的NetLiquidation货币
        max_netliq = 0.0
        primary_currency = "USD"
        
        for key, value in account_values.items():
            if key.startswith("NetLiquidation:"):
                currency = key.split(":", 1)[1]
                try:
                    netliq = float(value)
                    if abs(netliq) > abs(max_netliq):
                        max_netliq = netliq
                        primary_currency = currency
                except (ValueError, TypeError):
                    continue
        
        return primary_currency
    
    def _validate_snapshot(self, snapshot: AccountSnapshot) -> Dict[str, Any]:
        """验证账户快照数据"""
        errors = []
        is_valid = True
        
        # 基本数值检查
        if snapshot.net_liquidation <= 0:
            errors.append("净资产值异常")
            is_valid = False
        
        if snapshot.cash_balance < 0 and abs(snapshot.cash_balance) > snapshot.net_liquidation:
            errors.append("现金余额异常")
            is_valid = False
        
        # 与历史数据对比
        if self.current_snapshot:
            prev = self.current_snapshot
            
            # 检查异常变化
            netliq_change = abs(snapshot.net_liquidation - prev.net_liquidation) / max(prev.net_liquidation, 1)
            if netliq_change > 0.5:  # 50%变化
                errors.append(f"净资产异常变化: {netliq_change:.1%}")
                is_valid = False
            
            # 检查时间戳
            if snapshot.timestamp <= prev.timestamp:
                errors.append("时间戳异常")
                is_valid = False
        
        return {
            'is_valid': is_valid,
            'errors': errors
        }
    
    def _save_snapshot(self, snapshot: AccountSnapshot):
        """保存快照到历史"""
        with self.data_lock:
            self.current_snapshot = snapshot
            self.snapshot_history.append(snapshot)
            
            # 限制历史长度
            if len(self.snapshot_history) > self.max_history:
                self.snapshot_history = self.snapshot_history[-self.max_history:]
    
    def get_current_data(self) -> Optional[AccountSnapshot]:
        """获取当前账户数据"""
        with self.data_lock:
            return self.current_snapshot
    
    def get_account_numeric(self, tag: str) -> float:
        """兼容性方法：获取账户数值"""
        snapshot = self.get_current_data()
        if not snapshot:
            return 0.0
        
        return self._extract_account_numeric(snapshot.account_values, tag)
    
    def get_positions(self) -> Dict[str, int]:
        """获取当前持仓"""
        snapshot = self.get_current_data()
        return snapshot.positions if snapshot else {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        with self.data_lock:
            return {
                'refresh_count': self.refresh_count,
                'error_count': self.error_count,
                'error_rate': self.error_count / max(self.refresh_count, 1),
                'last_successful_refresh': self.last_successful_refresh,
                'current_snapshot_valid': self.current_snapshot.is_valid if self.current_snapshot else None,
                'history_length': len(self.snapshot_history),
                'primary_currency': self.current_snapshot.currency if self.current_snapshot else None
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 尝试快速刷新
            snapshot = await self.refresh_with_retry(max_retries=1, timeout=5.0)
            
            return {
                'status': 'healthy',
                'data_age': time.time() - snapshot.timestamp,
                'is_valid': snapshot.is_valid,
                'errors': snapshot.validation_errors
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_success_age': time.time() - self.last_successful_refresh
            }
