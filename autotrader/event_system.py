"""
Event system for autotrader
"""
import asyncio
import logging
from typing import Dict, List, Callable, Any
import threading

logger = logging.getLogger(__name__)


class EventBus:
    """Simple event bus for inter-component communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.lock = threading.Lock()
        self.running = True
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type"""
        with self.lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event type"""
        with self.lock:
            if event_type in self.subscribers:
                self.subscribers[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type}")
    
    def emit(self, event_type: str, data: Any = None):
        """Emit an event to all subscribers"""
        if not self.running:
            return
            
        with self.lock:
            subscribers = self.subscribers.get(event_type, [])
        
        for callback in subscribers:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in event callback for {event_type}: {e}")
    
    def shutdown(self):
        """Shutdown the event bus"""
        self.running = False
        with self.lock:
            self.subscribers.clear()
        logger.info("Event bus shut down")


# Global event bus instance
_event_bus = None


def get_event_bus() -> EventBus:
    """Get global event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def shutdown_event_bus():
    """Shutdown the global event bus"""
    global _event_bus
    if _event_bus is not None:
        _event_bus.shutdown()
        _event_bus = None