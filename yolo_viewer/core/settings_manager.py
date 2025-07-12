"""Settings manager using JSON file for persistence."""

import json
import os
import base64
from typing import Any, Optional
from PyQt6.QtCore import QObject, pyqtSignal
from .constants import DEFAULT_CONFIDENCE, DEFAULT_IOU


class SettingsManager(QObject):
    """Singleton settings manager with Qt integration."""
    
    # Signals
    settingChanged = pyqtSignal(str, object)  # key, value
    
    def __init__(self):
        super().__init__()
        
        # Settings file path in the root of the application
        self._settings_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'settings.json')
        
        # Load settings from file
        self._settings = self._load_settings()
        
        # Set defaults if not present
        self._set_defaults()
        
        # Save initial settings
        self._save_settings()
    
    def _load_settings(self) -> dict:
        """Load settings from JSON file."""
        if os.path.exists(self._settings_file):
            try:
                with open(self._settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                
                # Decode special types
                settings = {}
                for key, value in loaded_settings.items():
                    if isinstance(value, dict) and value.get('_type') == 'bytes':
                        # Decode base64 string back to bytes
                        settings[key] = base64.b64decode(value['data'])
                    else:
                        settings[key] = value
                
                return settings
            except (json.JSONDecodeError, IOError):
                # If file is corrupted or can't be read, return empty dict
                return {}
        return {}
    
    def _save_settings(self):
        """Save settings to JSON file."""
        try:
            # Create a copy of settings to handle special types
            settings_to_save = {}
            for key, value in self._settings.items():
                if isinstance(value, bytes):
                    # Encode bytes as base64 string for JSON
                    settings_to_save[key] = {
                        '_type': 'bytes',
                        'data': base64.b64encode(value).decode('utf-8')
                    }
                else:
                    settings_to_save[key] = value
            
            with open(self._settings_file, 'w') as f:
                json.dump(settings_to_save, f, indent=2)
        except IOError as e:
            print(f"Error saving settings: {e}")
    
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
            if key not in self._settings:
                self._settings[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value.
        
        Args:
            key: Setting key
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        return self._settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set a setting value.
        
        Args:
            key: Setting key
            value: Setting value
        """
        old_value = self.get(key)
        self._settings[key] = value
        
        # Save to file
        self._save_settings()
        
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
        self._save_settings()