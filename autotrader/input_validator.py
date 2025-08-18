#!/usr/bin/env python3
"""
Input validation module for trading operations
Ensures all trading parameters are safe and valid
"""

import re
from typing import Any, Optional, Union
from decimal import Decimal, InvalidOperation
import logging

logger = logging.getLogger(__name__)

class ValidationError(ValueError):
    """Custom exception for validation errors"""
    pass

class InputValidator:
    """Comprehensive input validation for trading operations"""
    
    # Valid symbols pattern (alphanumeric, dots, hyphens)
    SYMBOL_PATTERN = re.compile(r'^[A-Z0-9.-]+$')
    
    # Reasonable limits
    MAX_SYMBOL_LENGTH = 10
    MAX_QUANTITY = 1_000_000
    MAX_PRICE = 1_000_000.0
    MIN_QUANTITY = 1
    MIN_PRICE = 0.01
    
    @classmethod
    def validate_symbol(cls, symbol: Any) -> str:
        """
        Validate trading symbol
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            str: Validated and normalized symbol
            
        Raises:
            ValidationError: If symbol is invalid
        """
        if not isinstance(symbol, str):
            raise ValidationError(f"Symbol must be string, got {type(symbol).__name__}")
        
        symbol = symbol.strip().upper()
        if not symbol:
            raise ValidationError("Symbol cannot be empty")
        
        if len(symbol) > cls.MAX_SYMBOL_LENGTH:
            raise ValidationError(f"Symbol too long: {symbol} (max {cls.MAX_SYMBOL_LENGTH} chars)")
        
        if not cls.SYMBOL_PATTERN.match(symbol):
            raise ValidationError(f"Invalid symbol format: {symbol}")
        
        return symbol

    @classmethod
    def validate_order_side(cls, side: Any) -> str:
        """
        Validate order side (BUY/SELL)
        
        Args:
            side: Order side to validate
            
        Returns:
            str: Validated order side
            
        Raises:
            ValidationError: If side is invalid
        """
        if not isinstance(side, str):
            raise ValidationError(f"Order side must be string, got {type(side).__name__}")
        
        side = side.strip().upper()
        if side not in ['BUY', 'SELL']:
            raise ValidationError(f"Invalid order side: {side}. Must be BUY or SELL")
        
        return side

    @classmethod
    def validate_quantity(cls, quantity: Any) -> int:
        """
        Validate order quantity
        
        Args:
            quantity: Quantity to validate
            
        Returns:
            int: Validated quantity
            
        Raises:
            ValidationError: If quantity is invalid
        """
        try:
            if isinstance(quantity, float):
                if not quantity.is_integer():
                    raise ValidationError(f"Quantity must be whole number, got {quantity}")
                qty = int(quantity)
            else:
                qty = int(quantity)
        except (ValueError, TypeError):
            raise ValidationError(f"Quantity must be integer, got {type(quantity).__name__}: {quantity}")
        
        if qty < cls.MIN_QUANTITY:
            raise ValidationError(f"Quantity too small: {qty} (minimum {cls.MIN_QUANTITY})")
        
        if qty > cls.MAX_QUANTITY:
            raise ValidationError(f"Quantity too large: {qty} (maximum {cls.MAX_QUANTITY:,})")
        
        return qty

    @classmethod
    def validate_price(cls, price: Any, allow_none: bool = True) -> Optional[float]:
        """
        Validate price
        
        Args:
            price: Price to validate
            allow_none: Whether None is allowed
            
        Returns:
            Optional[float]: Validated price
            
        Raises:
            ValidationError: If price is invalid
        """
        if price is None:
            if allow_none:
                return None
            else:
                raise ValidationError("Price cannot be None")
        
        try:
            # Use Decimal for precise validation
            decimal_price = Decimal(str(price))
            price_val = float(decimal_price)
        except (ValueError, TypeError, InvalidOperation):
            raise ValidationError(f"Price must be numeric, got {type(price).__name__}: {price}")
        
        if price_val <= 0:
            raise ValidationError(f"Price must be positive, got {price_val}")
        
        if price_val < cls.MIN_PRICE:
            raise ValidationError(f"Price too small: ${price_val} (minimum ${cls.MIN_PRICE})")
        
        if price_val > cls.MAX_PRICE:
            raise ValidationError(f"Price too large: ${price_val:,.2f} (maximum ${cls.MAX_PRICE:,.2f})")
        
        # Check for reasonable decimal places (max 4)
        decimal_str = str(decimal_price)
        if '.' in decimal_str:
            decimal_places = len(decimal_str.split('.')[1])
            if decimal_places > 4:
                raise ValidationError(f"Price has too many decimal places: {decimal_places} (max 4)")
        
        return price_val

    @classmethod
    def validate_client_id(cls, client_id: Any) -> int:
        """
        Validate client ID
        
        Args:
            client_id: Client ID to validate
            
        Returns:
            int: Validated client ID
            
        Raises:
            ValidationError: If client ID is invalid
        """
        try:
            cid = int(client_id)
        except (ValueError, TypeError):
            raise ValidationError(f"Client ID must be integer, got {type(client_id).__name__}: {client_id}")
        
        if cid < 0:
            raise ValidationError(f"Client ID must be non-negative, got {cid}")
        
        if cid > 999999:  # Reasonable upper limit
            raise ValidationError(f"Client ID too large: {cid}")
        
        return cid

    @classmethod
    def validate_port(cls, port: Any) -> int:
        """
        Validate port number
        
        Args:
            port: Port to validate
            
        Returns:
            int: Validated port
            
        Raises:
            ValidationError: If port is invalid
        """
        try:
            port_val = int(port)
        except (ValueError, TypeError):
            raise ValidationError(f"Port must be integer, got {type(port).__name__}: {port}")
        
        if port_val < 1 or port_val > 65535:
            raise ValidationError(f"Port out of range: {port_val} (must be 1-65535)")
        
        # Check for common IBKR ports
        valid_ibkr_ports = [4001, 4002, 7496, 7497]
        if port_val not in valid_ibkr_ports:
            logger.warning(f"Non-standard IBKR port: {port_val}. Common ports: {valid_ibkr_ports}")
        
        return port_val

    @classmethod
    def validate_host(cls, host: Any) -> str:
        """
        Validate host address
        
        Args:
            host: Host to validate
            
        Returns:
            str: Validated host
            
        Raises:
            ValidationError: If host is invalid
        """
        if not isinstance(host, str):
            raise ValidationError(f"Host must be string, got {type(host).__name__}")
        
        host = host.strip()
        if not host:
            raise ValidationError("Host cannot be empty")
        
        # Basic validation - allow IP addresses and hostnames
        if not re.match(r'^[a-zA-Z0-9.-]+$', host):
            raise ValidationError(f"Invalid host format: {host}")
        
        return host

    @classmethod
    def validate_percentage(cls, percentage: Any, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """
        Validate percentage value
        
        Args:
            percentage: Percentage to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            float: Validated percentage
            
        Raises:
            ValidationError: If percentage is invalid
        """
        try:
            pct_val = float(percentage)
        except (ValueError, TypeError):
            raise ValidationError(f"Percentage must be numeric, got {type(percentage).__name__}: {percentage}")
        
        if pct_val < min_val or pct_val > max_val:
            raise ValidationError(f"Percentage out of range: {pct_val}% (must be {min_val}%-{max_val}%)")
        
        return pct_val

    @classmethod
    def validate_order_params(cls, symbol: str, side: str, quantity: int, price: Optional[float] = None) -> dict:
        """
        Validate complete order parameters
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Number of shares
            price: Order price (optional for market orders)
            
        Returns:
            dict: Validated parameters
            
        Raises:
            ValidationError: If any parameter is invalid
        """
        try:
            validated = {
                'symbol': cls.validate_symbol(symbol),
                'side': cls.validate_order_side(side),
                'quantity': cls.validate_quantity(quantity),
                'price': cls.validate_price(price, allow_none=True)
            }
            
            # Additional business logic validation
            order_value = quantity * (price or 0)
            if price and order_value > 1_000_000:  # $1M order value limit
                raise ValidationError(f"Order value too large: ${order_value:,.2f}")
            
            logger.debug(f"Order validation passed: {validated}")
            return validated
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Unexpected validation error: {e}")

# Decorator for automatic input validation
def validate_trading_inputs(**validations):
    """
    Decorator to automatically validate function inputs
    
    Usage:
        @validate_trading_inputs(symbol='symbol', side='side', quantity='quantity')
        def place_order(symbol, side, quantity, **kwargs):
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Apply validations
            for param_name, validation_type in validations.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    
                    if validation_type == 'symbol':
                        bound_args.arguments[param_name] = InputValidator.validate_symbol(value)
                    elif validation_type == 'side':
                        bound_args.arguments[param_name] = InputValidator.validate_order_side(value)
                    elif validation_type == 'quantity':
                        bound_args.arguments[param_name] = InputValidator.validate_quantity(value)
                    elif validation_type == 'price':
                        bound_args.arguments[param_name] = InputValidator.validate_price(value)
                    elif validation_type == 'client_id':
                        bound_args.arguments[param_name] = InputValidator.validate_client_id(value)
                    elif validation_type == 'port':
                        bound_args.arguments[param_name] = InputValidator.validate_port(value)
                    elif validation_type == 'host':
                        bound_args.arguments[param_name] = InputValidator.validate_host(value)
            
            return func(*bound_args.args, **bound_args.kwargs)
        
        return wrapper
    return decorator

# Usage examples:
if __name__ == "__main__":
    # Test the validator
    try:
        # Valid inputs
        params = InputValidator.validate_order_params("AAPL", "BUY", 100, 150.50)
        print(f"✅ Valid order: {params}")
        
        # Invalid inputs
        try:
            InputValidator.validate_symbol("")
        except ValidationError as e:
            print(f"❌ Invalid symbol: {e}")
        
        try:
            InputValidator.validate_quantity(-10)
        except ValidationError as e:
            print(f"❌ Invalid quantity: {e}")
            
    except Exception as e:
        print(f"Test error: {e}")