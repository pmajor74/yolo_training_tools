"""Base class for singleton QObjects that works with Python 3.13"""

from PyQt6.QtCore import QObject
from typing import TypeVar, Type, Dict, Any
import threading

T = TypeVar('T')


def singleton(cls):
    """Decorator to make a QObject-based class a singleton."""
    instances = {}
    lock = threading.Lock()
    
    class SingletonWrapper(cls):
        def __new__(cls_inner, *args, **kwargs):
            if cls not in instances:
                with lock:
                    if cls not in instances:
                        # Create the actual instance
                        instance = object.__new__(cls_inner)
                        instances[cls] = instance
            return instances[cls]
        
        def __init__(self, *args, **kwargs):
            # Only initialize once
            if hasattr(self, '_singleton_initialized'):
                return
            super().__init__(*args, **kwargs)
            self._singleton_initialized = True
    
    # Copy class attributes
    SingletonWrapper.__name__ = cls.__name__
    SingletonWrapper.__qualname__ = cls.__qualname__
    SingletonWrapper.__module__ = cls.__module__
    
    # Add instance class method
    @classmethod
    def instance(cls_method):
        """Get the singleton instance."""
        return SingletonWrapper()
    
    SingletonWrapper.instance = instance
    
    return SingletonWrapper