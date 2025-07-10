"""Centralized dataset manager for sharing dataset information across modes."""

from pathlib import Path
from typing import Optional, Dict, List
from PyQt6.QtCore import QObject, pyqtSignal
from ..utils.yolo_format import load_data_yaml


class DatasetManager(QObject):
    """
    Singleton manager for dataset information.
    
    Ensures dataset information is shared across all application modes.
    """
    
    # Signals
    datasetLoaded = pyqtSignal(Path)  # Emitted when a new dataset is loaded
    datasetUpdated = pyqtSignal()  # Emitted when dataset info is updated
    
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
        self._initialized = True
        
        # Dataset information
        self._yaml_path: Optional[Path] = None
        self._dataset_root: Optional[Path] = None
        self._class_names: Dict[int, str] = {}
        self._num_classes: int = 0
        self._splits: Dict[str, str] = {}  # split name -> relative path
        self._raw_data: Optional[Dict] = None  # Raw YAML data
        
    def load_dataset(self, yaml_path: Path) -> bool:
        """
        Load dataset from data.yaml file.
        
        Args:
            yaml_path: Path to data.yaml file
            
        Returns:
            True if successful, False otherwise
        """
        # Check if this is the same dataset
        if self._yaml_path == yaml_path:
            return True
            
        try:
            # Load YAML data
            data = load_data_yaml(yaml_path)
            if not data:
                return False
                
            # Store raw data
            self._raw_data = data
            self._yaml_path = yaml_path
            
            # Extract dataset root
            yaml_dir = yaml_path.parent
            if 'path' in data:
                self._dataset_root = yaml_dir / data['path']
            else:
                self._dataset_root = yaml_dir
                
            # Extract class information
            self._num_classes = data.get('nc', 0)
            
            # Extract class names - handle both list and dict formats
            names_data = data.get('names', {})
            if isinstance(names_data, dict):
                # Names is a dict mapping class_id to name
                self._class_names = {int(k): str(v) for k, v in names_data.items()}
            else:
                # Names is a list
                self._class_names = {i: str(name) for i, name in enumerate(names_data)}
            
            # Extract splits
            self._splits = {}
            for split in ['train', 'val', 'test']:
                if split in data:
                    self._splits[split] = data[split]
                    
            # Emit signal
            self.datasetLoaded.emit(yaml_path)
            self.datasetUpdated.emit()
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def get_yaml_path(self) -> Optional[Path]:
        """Get current dataset YAML path."""
        return self._yaml_path
    
    def get_dataset_root(self) -> Optional[Path]:
        """Get dataset root directory."""
        return self._dataset_root
    
    def get_class_names(self) -> Dict[int, str]:
        """Get class names dictionary."""
        return self._class_names.copy()
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name for given ID."""
        return self._class_names.get(class_id, f"Class {class_id}")
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self._num_classes
    
    def get_splits(self) -> Dict[str, str]:
        """Get dataset splits."""
        return self._splits.copy()
    
    def get_split_path(self, split: str) -> Optional[Path]:
        """Get full path for a split."""
        if split in self._splits and self._dataset_root:
            return self._dataset_root / self._splits[split]
        return None
    
    def get_raw_data(self) -> Optional[Dict]:
        """Get raw YAML data."""
        return self._raw_data.copy() if self._raw_data else None
    
    def has_dataset(self) -> bool:
        """Check if a dataset is loaded."""
        return self._yaml_path is not None
    
    def clear(self):
        """Clear dataset information."""
        self._yaml_path = None
        self._dataset_root = None
        self._class_names = {}
        self._num_classes = 0
        self._splits = {}
        self._raw_data = None
        self.datasetUpdated.emit()