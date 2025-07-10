"""Singleton model cache for YOLO models."""

from typing import Optional, Dict, Any
from pathlib import Path
from ultralytics import YOLO
from PyQt6.QtCore import QObject, pyqtSignal
import torch


class ModelCache(QObject):
    """Singleton cache for YOLO models with Qt signals."""
    
    # Signals
    modelLoaded = pyqtSignal(str)  # Emits model path when loaded
    modelCleared = pyqtSignal()
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._model: Optional[YOLO] = None
        self._model_path: Optional[str] = None
        self._model_info: Dict[str, Any] = {}
        self._device: Optional[str] = None
        self._initialized = True
    
    def _get_best_device(self) -> str:
        """Detect and return the best available device."""
        try:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except Exception:
            return "cpu"
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a YOLO model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Detect best device
            self._device = self._get_best_device()
            
            # Load model with device specification
            self._model = YOLO(model_path)
            
            # Move model to device
            if self._device != "cpu":
                self._model.to(self._device)
            
            self._model_path = model_path
            
            # Extract model info
            self._model_info = {
                'path': model_path,
                'name': Path(model_path).name,
                'type': 'YOLOv8',  # Could be detected from model
                'classes': getattr(self._model.model, 'names', {}),
                'num_classes': len(getattr(self._model.model, 'names', {})),
                'device': self._device
            }
            
            # Log device usage
            device_name = self._device.upper()
            if self._device == "cuda":
                device_name = f"GPU (CUDA) - {torch.cuda.get_device_name(0)}"
            elif self._device == "mps":
                device_name = "GPU (Apple Silicon)"
            print(f"Model loaded on: {device_name}")
            
            self.modelLoaded.emit(model_path)
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.clear_model()
            return False
    
    def clear_model(self):
        """Clear the cached model."""
        self._model = None
        self._model_path = None
        self._model_info = {}
        self._device = None
        self.modelCleared.emit()
    
    def get_model(self) -> Optional[YOLO]:
        """Get the cached model."""
        return self._model
    
    def get_model_path(self) -> Optional[str]:
        """Get the path of the loaded model."""
        return self._model_path
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return self._model_info.copy()
    
    def is_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._model is not None
    
    def get_device(self) -> Optional[str]:
        """Get the device the model is loaded on."""
        return self._device