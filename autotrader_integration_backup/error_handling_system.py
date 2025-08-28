#!/usr/bin/env python3
"""
Enhanced Error Handling System
Provides comprehensive error handling, logging, and recovery mechanisms
"""

import logging
import traceback
import functools
import threading
import time
from typing import Dict, Any, Optional, List, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from contextlib import contextmanager
import json
import os


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """Error categories for classification"""
    CONNECTION = "connection"
    TRADING = "trading"
    DATA = "data"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    VALIDATION = "validation"
    SECURITY = "security"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    operation: str
    component: str
    user_data: Dict[str, Any] = field(default_factory=dict)
    system_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    thread_id: str = field(default_factory=lambda: str(threading.current_thread().ident))


@dataclass 
class ErrorRecord:
    """Complete error record for tracking and analysis"""
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    traceback_str: str
    context: ErrorContext
    recovery_attempted: bool = False
    recovery_success: bool = False
    recovery_details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_notes: Optional[str] = None


class ErrorHandlerRegistry:
    """Registry for error handlers and recovery strategies"""
    
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._recovery_strategies: Dict[str, Callable] = {}
        self._circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.logger = logging.getLogger("ErrorHandlerRegistry")
    
    def register_handler(self, error_pattern: str, handler: Callable, category: ErrorCategory = None):
        """Register an error handler for specific error patterns"""
        key = f"{category.value if category else 'generic'}:{error_pattern}"
        self._handlers[key] = handler
        self.logger.info(f"Registered error handler: {key}")
    
    def register_recovery_strategy(self, operation: str, strategy: Callable):
        """Register a recovery strategy for specific operations"""
        self._recovery_strategies[operation] = strategy
        self.logger.info(f"Registered recovery strategy: {operation}")
    
    def get_handler(self, error_type: str, category: ErrorCategory = None) -> Optional[Callable]:
        """Get appropriate error handler"""
        # Try specific category first
        if category:
            key = f"{category.value}:{error_type}"
            if key in self._handlers:
                return self._handlers[key]
        
        # Fall back to generic handler
        generic_key = f"generic:{error_type}"
        return self._handlers.get(generic_key)
    
    def get_recovery_strategy(self, operation: str) -> Optional[Callable]:
        """Get recovery strategy for operation"""
        return self._recovery_strategies.get(operation)
    
    def get_circuit_breaker(self, operation: str) -> 'CircuitBreaker':
        """Get or create circuit breaker for operation"""
        if operation not in self._circuit_breakers:
            self._circuit_breakers[operation] = CircuitBreaker(operation)
        return self._circuit_breakers[operation]


class CircuitBreaker:
    """Circuit breaker pattern for error handling"""
    
    def __init__(self, operation: str, failure_threshold: int = 5, reset_timeout: int = 60):
        self.operation = operation
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = logging.getLogger(f"CircuitBreaker.{operation}")
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
                self.logger.info(f"Circuit breaker {self.operation} transitioning to HALF_OPEN")
                return True
            return False
        
        if self.state == "HALF_OPEN":
            return True
        
        return False
    
    def record_success(self):
        """Record successful operation"""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self.logger.info(f"Circuit breaker {self.operation} reset to CLOSED")
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker {self.operation} opened after {self.failure_count} failures")


class EnhancedErrorHandler:
    """Enhanced error handling system with logging, recovery, and circuit breakers"""
    
    def __init__(self, log_file: str = "logs/error_handling.log"):
        self.registry = ErrorHandlerRegistry()
        self.error_records: List[ErrorRecord] = []
        self._record_lock = threading.RLock()
        self._setup_logging(log_file)
        self._register_default_handlers()
        self.stats = {
            'total_errors': 0,
            'by_severity': {s.value: 0 for s in ErrorSeverity},
            'by_category': {c.value: 0 for c in ErrorCategory},
            'recoveries_attempted': 0,
            'recoveries_successful': 0
        }
    
    def _setup_logging(self, log_file: str):
        """Setup enhanced logging"""
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        self.logger = logging.getLogger("EnhancedErrorHandler")
        
        # File handler for error details
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important errors
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.WARNING)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.DEBUG)
    
    def _register_default_handlers(self):
        """Register default error handlers"""
        
        # Connection error handlers
        self.registry.register_handler(
            "ConnectionError", 
            self._handle_connection_error,
            ErrorCategory.CONNECTION
        )
        
        self.registry.register_handler(
            "TimeoutError",
            self._handle_timeout_error, 
            ErrorCategory.CONNECTION
        )
        
        # Trading error handlers
        self.registry.register_handler(
            "InsufficientFunds",
            self._handle_insufficient_funds,
            ErrorCategory.TRADING
        )
        
        # Data error handlers
        self.registry.register_handler(
            "DataError",
            self._handle_data_error,
            ErrorCategory.DATA
        )
        
        # Configuration error handlers
        self.registry.register_handler(
            "ConfigurationError", 
            self._handle_config_error,
            ErrorCategory.CONFIGURATION
        )
        
        # Register recovery strategies
        self.registry.register_recovery_strategy("ibkr_connection", self._recover_ibkr_connection)
        self.registry.register_recovery_strategy("data_fetch", self._recover_data_fetch)
        self.registry.register_recovery_strategy("order_submission", self._recover_order_submission)
    
    def handle_error(self, 
                    exception: Exception, 
                    context: ErrorContext,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.SYSTEM,
                    attempt_recovery: bool = True) -> ErrorRecord:
        """Main error handling entry point"""
        
        # Generate unique error ID
        error_id = f"{category.value}_{int(time.time())}_{id(exception)}"
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            severity=severity,
            category=category,
            message=str(exception),
            exception_type=type(exception).__name__,
            traceback_str=traceback.format_exc(),
            context=context
        )
        
        # Store error record
        with self._record_lock:
            self.error_records.append(error_record)
            self.stats['total_errors'] += 1
            self.stats['by_severity'][severity.value] += 1
            self.stats['by_category'][category.value] += 1
        
        # Log error details
        self._log_error(error_record)
        
        # Check circuit breaker
        circuit_breaker = self.registry.get_circuit_breaker(context.operation)
        if not circuit_breaker.can_execute():
            self.logger.error(f"Circuit breaker OPEN for {context.operation} - operation blocked")
            return error_record
        
        # Attempt recovery if enabled
        if attempt_recovery:
            recovery_success = self._attempt_recovery(error_record, circuit_breaker)
            error_record.recovery_attempted = True
            error_record.recovery_success = recovery_success
            
            if recovery_success:
                circuit_breaker.record_success()
            else:
                circuit_breaker.record_failure()
        else:
            circuit_breaker.record_failure()
        
        return error_record
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level"""
        log_msg = (
            f"[{error_record.error_id}] {error_record.category.value.upper()} ERROR: "
            f"{error_record.message} (Operation: {error_record.context.operation})"
        )
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_msg)
            self.logger.critical(f"Stack trace: {error_record.traceback_str}")
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_msg)
            self.logger.debug(f"Stack trace: {error_record.traceback_str}")
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_msg)
            self.logger.debug(f"Stack trace: {error_record.traceback_str}")
        else:
            self.logger.info(log_msg)
    
    def _attempt_recovery(self, error_record: ErrorRecord, circuit_breaker: CircuitBreaker) -> bool:
        """Attempt to recover from error"""
        try:
            self.stats['recoveries_attempted'] += 1
            
            # Try specific error handler first
            handler = self.registry.get_handler(
                error_record.exception_type, 
                error_record.category
            )
            
            if handler:
                self.logger.info(f"Attempting specific recovery for {error_record.error_id}")
                result = handler(error_record)
                if result:
                    self.stats['recoveries_successful'] += 1
                    error_record.recovery_details = "Specific handler successful"
                    self.logger.info(f"Recovery successful for {error_record.error_id}")
                    return True
            
            # Try general recovery strategy
            recovery_strategy = self.registry.get_recovery_strategy(error_record.context.operation)
            if recovery_strategy:
                self.logger.info(f"Attempting general recovery for {error_record.error_id}")
                result = recovery_strategy(error_record)
                if result:
                    self.stats['recoveries_successful'] += 1
                    error_record.recovery_details = "General strategy successful"
                    self.logger.info(f"Recovery successful for {error_record.error_id}")
                    return True
            
            error_record.recovery_details = "No recovery strategy available"
            return False
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed for {error_record.error_id}: {recovery_error}")
            error_record.recovery_details = f"Recovery failed: {recovery_error}"
            return False
    
    # Default error handlers
    def _handle_connection_error(self, error_record: ErrorRecord) -> bool:
        """Handle connection errors"""
        self.logger.info(f"Handling connection error: {error_record.error_id}")
        # Implement connection retry logic
        return False  # Placeholder
    
    def _handle_timeout_error(self, error_record: ErrorRecord) -> bool:
        """Handle timeout errors"""
        self.logger.info(f"Handling timeout error: {error_record.error_id}")
        # Implement timeout recovery logic
        return False  # Placeholder
    
    def _handle_insufficient_funds(self, error_record: ErrorRecord) -> bool:
        """Handle insufficient funds errors"""
        self.logger.warning(f"Insufficient funds detected: {error_record.error_id}")
        # Log for manual review
        return False  # Cannot auto-recover from insufficient funds
    
    def _handle_data_error(self, error_record: ErrorRecord) -> bool:
        """Handle data errors"""
        self.logger.info(f"Handling data error: {error_record.error_id}")
        # Implement data retry/fallback logic
        return False  # Placeholder
    
    def _handle_config_error(self, error_record: ErrorRecord) -> bool:
        """Handle configuration errors"""
        self.logger.error(f"Configuration error detected: {error_record.error_id}")
        # Configuration errors typically require manual intervention
        return False
    
    # Recovery strategies
    def _recover_ibkr_connection(self, error_record: ErrorRecord) -> bool:
        """Recover IBKR connection"""
        self.logger.info(f"Attempting IBKR connection recovery for {error_record.error_id}")
        # Implement IBKR reconnection logic
        return False  # Placeholder
    
    def _recover_data_fetch(self, error_record: ErrorRecord) -> bool:
        """Recover from data fetch errors"""
        self.logger.info(f"Attempting data fetch recovery for {error_record.error_id}")
        # Implement data retry logic
        return False  # Placeholder
    
    def _recover_order_submission(self, error_record: ErrorRecord) -> bool:
        """Recover from order submission errors"""
        self.logger.info(f"Attempting order submission recovery for {error_record.error_id}")
        # Implement order retry logic
        return False  # Placeholder
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        with self._record_lock:
            recent_errors = [
                r for r in self.error_records 
                if r.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            return {
                **self.stats,
                'recent_24h': len(recent_errors),
                'total_recorded': len(self.error_records),
                'recovery_rate': (
                    self.stats['recoveries_successful'] / max(self.stats['recoveries_attempted'], 1) * 100
                )
            }
    
    def get_recent_errors(self, hours: int = 1) -> List[ErrorRecord]:
        """Get recent errors within specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        with self._record_lock:
            return [r for r in self.error_records if r.timestamp > cutoff_time]
    
    def export_error_report(self, filepath: str = None) -> str:
        """Export error report to JSON"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"logs/error_report_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_error_statistics(),
            'recent_errors': [
                {
                    'error_id': r.error_id,
                    'severity': r.severity.value,
                    'category': r.category.value,
                    'message': r.message,
                    'operation': r.context.operation,
                    'component': r.context.component,
                    'timestamp': r.timestamp.isoformat(),
                    'recovery_attempted': r.recovery_attempted,
                    'recovery_success': r.recovery_success,
                    'resolved': r.resolved
                }
                for r in self.get_recent_errors(24)
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Error report exported to: {filepath}")
        return filepath


# Decorators for enhanced error handling
def with_error_handling(
    operation: str,
    component: str = "unknown",
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    attempt_recovery: bool = True,
    reraise: bool = False
):
    """Decorator for automatic error handling"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    operation=operation,
                    component=component,
                    user_data={'args': str(args), 'kwargs': str(kwargs)},
                    system_data={'function': func.__name__}
                )
                
                error_handler = get_error_handler()
                error_record = error_handler.handle_error(
                    e, context, severity, category, attempt_recovery
                )
                
                if reraise:
                    raise
                
                return None  # or appropriate default value
        return wrapper
    return decorator


@contextmanager
def error_handling_context(
    operation: str,
    component: str = "unknown",
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM
):
    """Context manager for error handling"""
    try:
        yield
    except Exception as e:
        context = ErrorContext(
            operation=operation,
            component=component
        )
        
        error_handler = get_error_handler()
        error_handler.handle_error(e, context, severity, category)
        raise


# Global error handler instance
_global_error_handler: Optional[EnhancedErrorHandler] = None
_error_handler_lock = threading.RLock()


def get_error_handler() -> EnhancedErrorHandler:
    """Get global error handler instance"""
    global _global_error_handler
    
    with _error_handler_lock:
        if _global_error_handler is None:
            _global_error_handler = EnhancedErrorHandler()
        return _global_error_handler


def reset_error_handler():
    """Reset global error handler (for testing)"""
    global _global_error_handler
    with _error_handler_lock:
        _global_error_handler = None


if __name__ == "__main__":
    # Test the error handling system
    logging.basicConfig(level=logging.INFO)
    
    error_handler = EnhancedErrorHandler()
    
    # Test error handling
    try:
        raise ValueError("Test error message")
    except Exception as e:
        context = ErrorContext(
            operation="test_operation",
            component="test_component"
        )
        
        error_record = error_handler.handle_error(
            e, context, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM
        )
        
        print(f"Error handled: {error_record.error_id}")
    
    # Test decorator
    @with_error_handling("decorated_test", "test_component")
    def test_function():
        raise RuntimeError("Decorated function error")
    
    test_function()  # Should handle error gracefully
    
    # Print statistics
    stats = error_handler.get_error_statistics()
    print(f"Error statistics: {stats}")
    
    # Export report
    report_file = error_handler.export_error_report()
    print(f"Error report exported to: {report_file}")
    
    print("Error handling system test completed")