"""Base class for application modes."""

from abc import abstractmethod
from typing import Optional
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import pyqtSignal


class BaseMode(QWidget):
    """Abstract base class for all application modes."""
    
    # Common signals
    statusMessage = pyqtSignal(str, int)  # message, timeout
    progressUpdate = pyqtSignal(int, int)  # current, total
    modeActivated = pyqtSignal()
    modeDeactivated = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_active = False
        self._setup_ui()
    
    @abstractmethod
    def _setup_ui(self):
        """Setup the UI for this mode."""
        pass
    
    def activate(self):
        """Activate this mode."""
        self._is_active = True
        self._on_activate()
        self.modeActivated.emit()
    
    def deactivate(self) -> bool:
        """
        Deactivate this mode.
        
        Returns:
            bool: True if deactivation is allowed, False to cancel
        """
        # Check if deactivation is allowed
        can_deactivate = self._on_deactivate()
        if can_deactivate is None:
            can_deactivate = True  # Default to allowing deactivation
            
        if can_deactivate:
            self._is_active = False
            self.modeDeactivated.emit()
            
        return can_deactivate
    
    def is_active(self) -> bool:
        """Check if mode is active."""
        return self._is_active
    
    @abstractmethod
    def _on_activate(self):
        """Called when mode is activated."""
        pass
    
    @abstractmethod
    def _on_deactivate(self) -> Optional[bool]:
        """
        Called when mode is deactivated.
        
        Returns:
            Optional[bool]: True to allow deactivation, False to cancel, None for default (True)
        """
        pass
    
    @abstractmethod
    def get_mode_name(self) -> str:
        """Get the display name of this mode."""
        pass
    
    def has_unsaved_changes(self) -> bool:
        """Check if mode has unsaved changes."""
        return False
    
    def save_changes(self) -> bool:
        """Save any pending changes."""
        return True
    
    def can_switch_mode(self) -> bool:
        """Check if it's safe to switch away from this mode."""
        if self.has_unsaved_changes():
            # In real implementation, would show dialog
            return False
        return True