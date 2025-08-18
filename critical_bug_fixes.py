#!/usr/bin/env python3
"""
Critical Bug Fixes for Trading System
Apply these fixes immediately to prevent system failures and financial losses
"""

import logging
import sqlite3
import threading
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class SafeExceptionHandler:
    """Replace bare except handlers with safe, specific exception handling"""
    
    @staticmethod
    def safe_database_operation(operation, *args, **kwargs):
        """Safely execute database operations with proper exception handling"""
        try:
            return operation(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                logger.warning(f"Database locked, will retry: {e}")
                raise  # Let caller handle retry
            else:
                logger.error(f"Database operational error: {e}")
                raise
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error in database operation: {e}")
            raise

    @staticmethod
    def safe_encoding_operation(operation, message, *args, **kwargs):
        """Safely handle encoding operations"""
        try:
            return operation(message, *args, **kwargs)
        except UnicodeEncodeError as e:
            # Handle encoding errors gracefully
            safe_message = str(message).encode('ascii', errors='ignore').decode('ascii')
            logger.warning(f"Encoding error, using safe message: {e}")
            return operation(safe_message, *args, **kwargs)
        except UnicodeDecodeError as e:
            logger.error(f"Decoding error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected encoding error: {e}")
            raise

class InputValidator:
    """Comprehensive input validation for trading operations"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate trading symbol"""
        if not isinstance(symbol, str):
            raise ValueError(f"Symbol must be string, got {type(symbol)}")
        
        symbol = symbol.strip().upper()
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        # Check for valid characters (alphanumeric and common symbols)
        if not symbol.replace('.', '').replace('-', '').isalnum():
            raise ValueError(f"Invalid symbol format: {symbol}")
        
        if len(symbol) > 10:  # Reasonable limit
            raise ValueError(f"Symbol too long: {symbol}")
        
        return symbol

    @staticmethod
    def validate_order_side(side: str) -> str:
        """Validate order side"""
        if not isinstance(side, str):
            raise ValueError(f"Order side must be string, got {type(side)}")
        
        side = side.strip().upper()
        if side not in ['BUY', 'SELL']:
            raise ValueError(f"Invalid order side: {side}. Must be BUY or SELL")
        
        return side

    @staticmethod
    def validate_quantity(quantity: Any) -> int:
        """Validate order quantity"""
        try:
            qty = int(quantity)
        except (ValueError, TypeError):
            raise ValueError(f"Quantity must be integer, got {type(quantity)}: {quantity}")
        
        if qty <= 0:
            raise ValueError(f"Quantity must be positive, got {qty}")
        
        if qty > 1_000_000:  # Reasonable upper limit
            raise ValueError(f"Quantity too large: {qty}")
        
        return qty

    @staticmethod
    def validate_price(price: Any, allow_none: bool = True) -> Optional[float]:
        """Validate price"""
        if price is None:
            if allow_none:
                return None
            else:
                raise ValueError("Price cannot be None")
        
        try:
            price_val = float(price)
        except (ValueError, TypeError):
            raise ValueError(f"Price must be numeric, got {type(price)}: {price}")
        
        if price_val <= 0:
            raise ValueError(f"Price must be positive, got {price_val}")
        
        if price_val > 1_000_000:  # Reasonable upper limit
            raise ValueError(f"Price too large: {price_val}")
        
        return price_val

class SafeDatabaseManager:
    """Enhanced database manager with proper connection handling"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection_count = 0
        self._lock = threading.RLock()
    
    @contextmanager
    def safe_connection(self, timeout: float = 30.0):
        """Safe database connection with automatic cleanup"""
        conn = None
        try:
            with self._lock:
                self._connection_count += 1
                logger.debug(f"Opening database connection #{self._connection_count}")
            
            conn = sqlite3.connect(
                self.db_path,
                timeout=timeout,
                isolation_level="DEFERRED",
                check_same_thread=False
            )
            
            # Configure for better performance and safety
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            conn.execute("PRAGMA busy_timeout=30000;")
            conn.execute("PRAGMA foreign_keys=ON;")
            
            yield conn
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
            raise
        finally:
            if conn:
                try:
                    conn.close()
                    with self._lock:
                        self._connection_count -= 1
                        logger.debug(f"Closed database connection, remaining: {self._connection_count}")
                except sqlite3.Error as e:
                    logger.warning(f"Error closing database connection: {e}")
    
    def check_connection_leaks(self) -> bool:
        """Check for connection leaks"""
        with self._lock:
            if self._connection_count > 0:
                logger.warning(f"Potential connection leak: {self._connection_count} connections still active")
                return True
            return False

class SafeConfigManager:
    """Thread-safe configuration manager with simplified locking"""
    
    def __init__(self):
        self._config = {}
        self._lock = threading.RLock()
        self._cache_valid = False
        self._last_update = 0
    
    def set_config(self, key: str, value: Any):
        """Thread-safe config setting"""
        with self._lock:
            self._config[key] = value
            self._invalidate_cache()
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Thread-safe config getting"""
        with self._lock:
            return self._config.get(key, default)
    
    def _invalidate_cache(self):
        """Simple cache invalidation"""
        # Already holding lock when called
        self._cache_valid = False
        self._last_update = time.time()

@contextmanager
def error_context(operation: str):
    """Context manager for better error handling and logging"""
    start_time = time.time()
    logger.info(f"Starting operation: {operation}")
    
    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Completed operation: {operation} (took {duration:.2f}s)")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed operation: {operation} after {duration:.2f}s - {e}")
        raise

def apply_critical_fixes():
    """Apply all critical fixes to the system"""
    logger.info("Applying critical bug fixes...")
    
    # Fix 1: Setup proper logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler('trading_system_fixes.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Fix 2: Validate that critical modules can be imported safely
    try:
        with error_context("Module import validation"):
            import sqlite3
            import threading
            import asyncio
            logger.info("Critical modules validated successfully")
    except ImportError as e:
        logger.critical(f"Critical module import failed: {e}")
        raise
    
    # Fix 3: Test database connection safety
    try:
        with error_context("Database connection test"):
            db_manager = SafeDatabaseManager(":memory:")  # Test with in-memory DB
            with db_manager.safe_connection() as conn:
                conn.execute("SELECT 1")
                logger.info("Database connection test passed")
    except Exception as e:
        logger.critical(f"Database connection test failed: {e}")
        raise
    
    logger.info("Critical bug fixes applied successfully")

# Usage example:
def fixed_order_validation_example(symbol, side, quantity, price=None):
    """Example of proper order validation using the fixes"""
    try:
        with error_context(f"Order validation for {symbol}"):
            # Apply input validation
            validated_symbol = InputValidator.validate_symbol(symbol)
            validated_side = InputValidator.validate_order_side(side)
            validated_quantity = InputValidator.validate_quantity(quantity)
            validated_price = InputValidator.validate_price(price)
            
            logger.info(f"Order validated: {validated_symbol} {validated_side} {validated_quantity} @ {validated_price}")
            
            return {
                'symbol': validated_symbol,
                'side': validated_side,
                'quantity': validated_quantity,
                'price': validated_price
            }
    except ValueError as e:
        logger.error(f"Order validation failed: {e}")
        raise
    except Exception as e:
        logger.critical(f"Unexpected error in order validation: {e}")
        raise

if __name__ == "__main__":
    apply_critical_fixes()
    
    # Test the fixes
    try:
        test_order = fixed_order_validation_example("AAPL", "buy", "100", 150.50)
        print(f"✅ Order validation test passed: {test_order}")
    except Exception as e:
        print(f"❌ Order validation test failed: {e}")