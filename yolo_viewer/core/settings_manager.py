"""Settings manager using QSettings for persistence."""

from typing import Any, Optional
from PyQt6.QtCore import QObject, QSettings, pyqtSignal
from .constants import ORGANIZATION, APP_NAME, DEFAULT_CONFIDENCE, DEFAULT_IOU


class SettingsManager(QObject):
    """Singleton settings manager with Qt integration."""
    
    # Signals
    settingChanged = pyqtSignal(str, object)  # key, value
    
    def __init__(self):
        super().__init__()
        
        # Initialize instance variables
        self._settings = QSettings(ORGANIZATION, APP_NAME)
        
        # Set defaults if not present
        self._set_defaults()
    
    def _set_defaults(self):
        """Set default values if they don't exist."""
        defaults = {
            'confidence_threshold': DEFAULT_CONFIDENCE,
            'iou_threshold': DEFAULT_IOU,
            'last_dataset_path': '',
            'last_model_path': '',
            'last_inference_folder': '',
            'theme': 'light',
            'auto_save': True,
            'show_confidence': True,
            'show_labels': True,
            'window_geometry': None,
            'splitter_sizes': {},
        }
        
        for key, value in defaults.items():
            if not self._settings.contains(key):
                self._settings.setValue(key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.
        
        Args:
            key: Setting key
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        return self._settings.value(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set a setting value.
        
        Args:
            key: Setting key
            value: Setting value
        """
        old_value = self.get(key)
        self._settings.setValue(key, value)
        
        if old_value != value:
            self.settingChanged.emit(key, value)
    
    def get_confidence_threshold(self) -> float:
        """Get confidence threshold."""
        return float(self.get('confidence_threshold', DEFAULT_CONFIDENCE))
    
    def set_confidence_threshold(self, value: float):
        """Set confidence threshold."""
        self.set('confidence_threshold', value)
    
    def get_iou_threshold(self) -> float:
        """Get IOU threshold."""
        return float(self.get('iou_threshold', DEFAULT_IOU))
    
    def set_iou_threshold(self, value: float):
        """Set IOU threshold."""
        self.set('iou_threshold', value)
    
    def get_last_dataset_path(self) -> str:
        """Get last used dataset path."""
        return self.get('last_dataset_path', '')
    
    def set_last_dataset_path(self, path: str):
        """Set last used dataset path."""
        self.set('last_dataset_path', path)
    
    def get_last_model_path(self) -> str:
        """Get last used model path."""
        return self.get('last_model_path', '')
    
    def set_last_model_path(self, path: str):
        """Set last used model path."""
        self.set('last_model_path', path)
    
    def get_window_geometry(self) -> Optional[bytes]:
        """Get saved window geometry."""
        return self.get('window_geometry')
    
    def set_window_geometry(self, geometry: bytes):
        """Save window geometry."""
        self.set('window_geometry', geometry)
    
    def get_splitter_sizes(self, splitter_id: str) -> Optional[list]:
        """Get saved splitter sizes."""
        sizes = self.get('splitter_sizes', {})
        return sizes.get(splitter_id)
    
    def set_splitter_sizes(self, splitter_id: str, sizes: list):
        """Save splitter sizes."""
        all_sizes = self.get('splitter_sizes', {})
        all_sizes[splitter_id] = sizes
        self.set('splitter_sizes', all_sizes)
    
    def sync(self):
        """Force sync settings to disk."""
        self._settings.sync()