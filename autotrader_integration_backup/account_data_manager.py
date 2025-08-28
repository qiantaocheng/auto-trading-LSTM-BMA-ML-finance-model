#!/usr/bin/env python3
"""
Account Data Manager Compatibility Module
Provides compatibility for the missing account_data_manager module
"""

import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class RobustAccountDataManager:
    """
    Compatibility wrapper for account data management
    Provides basic account management functionality
    """
    
    def __init__(self, ib_client, account_id: str):
        self.ib = ib_client
        self.account_id = account_id
        self.logger = logger
        
        # Cache for account data
        self._account_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 60  # 1 minute cache
        
        logger.info(f"RobustAccountDataManager initialized for account: {account_id}")
    
    async def refresh_account_data(self) -> Dict[str, Any]:
        """Refresh and return account data"""
        try:
            # Check cache freshness
            now = time.time()
            if now - self._cache_timestamp < self._cache_ttl and self._account_cache:
                return self._account_cache
            
            # Get fresh account data from IB
            account_data = {
                'account_id': self.account_id,
                'total_cash': 0.0,
                'available_funds': 0.0,
                'net_liquidation': 0.0,
                'positions': {},
                'timestamp': now
            }
            
            # Try to get account summary if connected
            if self.ib and hasattr(self.ib, 'accountSummary'):
                try:
                    summary = self.ib.accountSummary()
                    for item in summary:
                        if item.tag == 'TotalCashValue':
                            account_data['total_cash'] = float(item.value or 0)
                        elif item.tag == 'AvailableFunds':
                            account_data['available_funds'] = float(item.value or 0)
                        elif item.tag == 'NetLiquidation':
                            account_data['net_liquidation'] = float(item.value or 0)
                except Exception as e:
                    logger.warning(f"Failed to get account summary: {e}")
            
            # Try to get positions if connected
            if self.ib and hasattr(self.ib, 'positions'):
                try:
                    positions = self.ib.positions()
                    for pos in positions:
                        if pos.position != 0:
                            symbol = pos.contract.symbol
                            account_data['positions'][symbol] = int(pos.position)
                except Exception as e:
                    logger.warning(f"Failed to get positions: {e}")
            
            # Update cache
            self._account_cache = account_data
            self._cache_timestamp = now
            
            return account_data
            
        except Exception as e:
            logger.error(f"Failed to refresh account data: {e}")
            return self._account_cache or {
                'account_id': self.account_id,
                'total_cash': 0.0,
                'available_funds': 0.0,
                'net_liquidation': 0.0,
                'positions': {},
                'timestamp': time.time()
            }
    
    def get_account_data(self) -> Dict[str, Any]:
        """Get cached account data (synchronous)"""
        return self._account_cache or {
            'account_id': self.account_id,
            'total_cash': 0.0,
            'available_funds': 0.0,
            'net_liquidation': 0.0,
            'positions': {},
            'timestamp': time.time()
        }
    
    def get_cash_balance(self) -> float:
        """Get available cash balance"""
        data = self.get_account_data()
        return data.get('available_funds', 0.0)
    
    def get_net_liquidation(self) -> float:
        """Get net liquidation value"""
        data = self.get_account_data()
        return data.get('net_liquidation', 0.0)
    
    def get_positions(self) -> Dict[str, int]:
        """Get current positions"""
        data = self.get_account_data()
        return data.get('positions', {})
    
    def get_position_quantity(self, symbol: str) -> int:
        """Get position quantity for a specific symbol"""
        positions = self.get_positions()
        return positions.get(symbol, 0)