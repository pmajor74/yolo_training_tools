"""Image cache manager for efficient image loading and caching."""

from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QPixmap
import cv2


class ImageCache(QObject):
    """Singleton image cache with memory management."""
    
    # Signals
    cacheUpdated = pyqtSignal(int, int)  # num_images, size_mb
    
    _max_cache_size_mb = 500  # Maximum cache size in MB
    
    def __init__(self):
        super().__init__()
        
        # Initialize instance variables
        self._image_cache: Dict[str, np.ndarray] = {}
        self._pixmap_cache: Dict[str, QPixmap] = {}
        self._thumbnail_cache: Dict[str, QPixmap] = {}
        self._cache_order: List[str] = []
        self._inference_cache: Optional[Dict[str, Any]] = None  # For inference mode data
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image with caching.
        
        Args:
            image_path: Path to the image
            
        Returns:
            numpy array of the image or None
        """
        if image_path in self._image_cache:
            # Move to end (LRU)
            self._cache_order.remove(image_path)
            self._cache_order.append(image_path)
            return self._image_cache[image_path]
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Add to cache
            self._add_to_cache(image_path, image)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def get_pixmap(self, image_path: str, max_size: Optional[Tuple[int, int]] = None) -> Optional['QPixmap']:
        """
        Get QPixmap for an image with caching.
        
        Args:
            image_path: Path to the image
            max_size: Optional maximum size (width, height)
            
        Returns:
            QPixmap or None
        """
        # Delayed import to avoid issues before QApplication is created
        from PyQt6.QtGui import QPixmap, QImage
        from PyQt6.QtCore import Qt
        
        cache_key = f"{image_path}_{max_size}" if max_size else image_path
        
        if cache_key in self._pixmap_cache:
            return self._pixmap_cache[cache_key]
        
        # Load image first
        image = self.load_image(image_path)
        if image is None:
            return None
        
        try:
            # Convert to QPixmap
            height, width, channel = image.shape
            
            # Check for valid dimensions
            if width <= 0 or height <= 0:
                return None
                
            # Ensure image data is contiguous in memory
            if not image.flags['C_CONTIGUOUS']:
                image = np.ascontiguousarray(image)
                
            bytes_per_line = 3 * width
            # Create QImage with a copy of the data to ensure it persists
            image_data = image.data.tobytes()
            q_image = QImage(image_data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Make a deep copy to ensure data persists
            q_image = q_image.copy()
            
            # Check if QImage is valid
            if q_image.isNull():
                print(f"Warning: QImage is null for {image_path}")
                return None
                
            pixmap = QPixmap.fromImage(q_image)
            
            # Check if pixmap is valid
            if pixmap.isNull():
                return None
            
            # Scale if needed
            if max_size and max_size[0] > 0 and max_size[1] > 0:
                pixmap = pixmap.scaled(
                    max_size[0], max_size[1],
                    aspectRatioMode=1,  # KeepAspectRatio
                    transformMode=1     # SmoothTransformation
                )
            
            self._pixmap_cache[cache_key] = pixmap
            return pixmap
            
        except Exception as e:
            print(f"Error creating pixmap: {e}")
            return None
    
    def get_thumbnail(self, image_path: str, size: int = 150) -> Optional['QPixmap']:
        """
        Get thumbnail for an image.
        
        Args:
            image_path: Path to the image
            size: Thumbnail size
            
        Returns:
            QPixmap thumbnail or None
        """
        return self.get_pixmap(image_path, (size, size))
    
    def _add_to_cache(self, path: str, image: np.ndarray):
        """Add image to cache with LRU eviction."""
        self._image_cache[path] = image
        self._cache_order.append(path)
        
        # Check cache size and evict if necessary
        self._evict_if_needed()
        
        # Emit update signal
        cache_size_mb = self._calculate_cache_size()
        self.cacheUpdated.emit(len(self._image_cache), cache_size_mb)
    
    def _evict_if_needed(self):
        """Evict oldest items if cache is too large."""
        while self._calculate_cache_size() > self._max_cache_size_mb and self._cache_order:
            oldest_path = self._cache_order.pop(0)
            
            # Remove from all caches
            self._image_cache.pop(oldest_path, None)
            
            # Remove related pixmaps
            keys_to_remove = [k for k in self._pixmap_cache.keys() if k.startswith(oldest_path)]
            for key in keys_to_remove:
                self._pixmap_cache.pop(key, None)
    
    def _calculate_cache_size(self) -> int:
        """Calculate approximate cache size in MB."""
        size_bytes = 0
        
        # Image cache
        for image in self._image_cache.values():
            size_bytes += image.nbytes
        
        # Rough estimate for pixmap cache (4 bytes per pixel)
        for pixmap in self._pixmap_cache.values():
            size_bytes += pixmap.width() * pixmap.height() * 4
        
        return size_bytes // (1024 * 1024)
    
    def clear_cache(self):
        """Clear all caches."""
        self._image_cache.clear()
        self._pixmap_cache.clear()
        self._thumbnail_cache.clear()
        self._cache_order.clear()
        self.cacheUpdated.emit(0, 0)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'num_images': len(self._image_cache),
            'num_pixmaps': len(self._pixmap_cache),
            'size_mb': self._calculate_cache_size(),
            'max_size_mb': self._max_cache_size_mb
        }
    
    def cache_inference_data(self, data: Dict[str, Any]):
        """Cache inference mode data."""
        self._inference_cache = data
    
    def get_inference_cache(self) -> Optional[Dict[str, Any]]:
        """Get cached inference data."""
        return self._inference_cache
    
    def clear_inference_cache(self):
        """Clear only the inference cache."""
        self._inference_cache = None