"""Widget for sorting and filtering thumbnail galleries."""

from typing import List, Optional, Dict, Tuple, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QComboBox, QPushButton,
    QLabel, QSpinBox, QGroupBox, QCheckBox, QToolButton, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt

class SortOption(Enum):
    """Available sorting options."""
    DEFAULT = "Default"
    NAME = "Name"
    DATE_MODIFIED = "Date Modified"
    FILE_SIZE = "File Size"
    DETECTION_COUNT = "Detection Count"  # Only for auto-annotation
    CLASS_COUNT = "Class Count"  # Only for auto-annotation

class SortOrder(Enum):
    """Sort order options."""
    ASCENDING = "Ascending"
    DESCENDING = "Descending"

class SortFilterWidget(QWidget):
    """Widget for sorting and filtering image galleries."""
    
    # Signals
    sortingChanged = pyqtSignal()  # Emitted when sorting changes
    filteringChanged = pyqtSignal()  # Emitted when filtering changes
    
    def __init__(self, parent=None, enable_detection_filters=False, start_collapsed=True):
        """
        Initialize the sort/filter widget.
        
        Args:
            parent: Parent widget
            enable_detection_filters: Whether to show detection-based filters
            start_collapsed: Whether to start in collapsed state
        """
        super().__init__(parent)
        self._enable_detection_filters = enable_detection_filters
        self._start_collapsed = start_collapsed
        self._setup_ui()
        
    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header with collapse/expand button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        self.toggle_btn = QToolButton()
        self.toggle_btn.setText("▼ Sort and Filter Options")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(not self._start_collapsed)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.toggle_btn.setSizePolicy(self.toggle_btn.sizePolicy().horizontalPolicy(), self.toggle_btn.sizePolicy().verticalPolicy())
        self.toggle_btn.setMinimumWidth(200)  # Ensure enough space for full text
        self.toggle_btn.setStyleSheet("""
            QToolButton {
                border: none;
                background: transparent;
                text-align: left;
                font-weight: bold;
                padding: 2px;
                min-width: 200px;
            }
            QToolButton:hover {
                background-color: rgba(0, 0, 0, 0.1);
            }
        """)
        self.toggle_btn.clicked.connect(self._toggle_content)
        header_layout.addWidget(self.toggle_btn)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Collapsible content frame
        self.content_frame = QFrame()
        self.content_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        self.content_frame.setVisible(not self._start_collapsed)
        
        content_layout = QVBoxLayout(self.content_frame)
        content_layout.setContentsMargins(5, 5, 5, 5)
        
        # Sorting controls
        sort_group = QGroupBox("Sort Options")
        sort_layout = QHBoxLayout(sort_group)
        
        # Sort by dropdown
        sort_layout.addWidget(QLabel("Sort by:"))
        self.sort_combo = QComboBox()
        
        # Add basic sort options
        self.sort_combo.addItem(SortOption.DEFAULT.value)
        self.sort_combo.addItem(SortOption.NAME.value)
        self.sort_combo.addItem(SortOption.DATE_MODIFIED.value)
        self.sort_combo.addItem(SortOption.FILE_SIZE.value)
        
        # Add detection-based sort options if enabled
        if self._enable_detection_filters:
            self.sort_combo.addItem(SortOption.DETECTION_COUNT.value)
            self.sort_combo.addItem(SortOption.CLASS_COUNT.value)
        
        self.sort_combo.currentTextChanged.connect(self._on_sorting_changed)
        sort_layout.addWidget(self.sort_combo)
        
        # Sort order radio buttons
        sort_layout.addWidget(QLabel("Order:"))
        self.order_combo = QComboBox()
        self.order_combo.addItem(SortOrder.ASCENDING.value)
        self.order_combo.addItem(SortOrder.DESCENDING.value)
        self.order_combo.currentTextChanged.connect(self._on_sorting_changed)
        sort_layout.addWidget(self.order_combo)
        
        # Clear sort button
        self.clear_sort_btn = QPushButton("Clear Sort")
        self.clear_sort_btn.clicked.connect(self.clear_sorting)
        sort_layout.addWidget(self.clear_sort_btn)
        
        sort_layout.addStretch()
        content_layout.addWidget(sort_group)
        
        # Detection filters (only if enabled)
        if self._enable_detection_filters:
            filter_group = QGroupBox("Detection Filters")
            filter_layout = QVBoxLayout(filter_group)
            
            # Minimum detections filter
            min_det_layout = QHBoxLayout()
            self.min_detections_checkbox = QCheckBox("Minimum detections:")
            self.min_detections_checkbox.toggled.connect(self._on_filter_toggled)
            min_det_layout.addWidget(self.min_detections_checkbox)
            
            self.min_detections_spin = QSpinBox()
            self.min_detections_spin.setRange(0, 999)
            self.min_detections_spin.setValue(1)
            self.min_detections_spin.setEnabled(False)
            self.min_detections_spin.valueChanged.connect(self._on_filtering_changed)
            min_det_layout.addWidget(self.min_detections_spin)
            min_det_layout.addStretch()
            filter_layout.addLayout(min_det_layout)
            
            # Minimum classes filter
            min_class_layout = QHBoxLayout()
            self.min_classes_checkbox = QCheckBox("Minimum classes:")
            self.min_classes_checkbox.toggled.connect(self._on_filter_toggled)
            min_class_layout.addWidget(self.min_classes_checkbox)
            
            self.min_classes_spin = QSpinBox()
            self.min_classes_spin.setRange(0, 99)
            self.min_classes_spin.setValue(1)
            self.min_classes_spin.setEnabled(False)
            self.min_classes_spin.valueChanged.connect(self._on_filtering_changed)
            min_class_layout.addWidget(self.min_classes_spin)
            min_class_layout.addStretch()
            filter_layout.addLayout(min_class_layout)
            
            # Clear filters button
            clear_filter_layout = QHBoxLayout()
            self.clear_filter_btn = QPushButton("Clear Filters")
            self.clear_filter_btn.clicked.connect(self.clear_filters)
            clear_filter_layout.addWidget(self.clear_filter_btn)
            clear_filter_layout.addStretch()
            filter_layout.addLayout(clear_filter_layout)
            
            content_layout.addWidget(filter_group)
        
        layout.addWidget(self.content_frame)
    
    def _toggle_content(self, checked):
        """Toggle the visibility of the content frame."""
        self.content_frame.setVisible(checked)
        if checked:
            self.toggle_btn.setText("▼ Sort and Filter Options")
        else:
            self.toggle_btn.setText("▶ Sort and Filter Options")
    
    def _on_sorting_changed(self):
        """Handle sorting change."""
        self.sortingChanged.emit()
        # Also emit filtering changed if detection filters are enabled
        # This ensures auto-annotation mode updates on all changes
        if self._enable_detection_filters:
            self.filteringChanged.emit()
    
    def _on_filtering_changed(self):
        """Handle filtering change."""
        if self._enable_detection_filters:
            self.filteringChanged.emit()
            # Also emit sorting changed to ensure consistent updates
            self.sortingChanged.emit()
    
    def _on_filter_toggled(self, checked):
        """Handle filter checkbox toggle."""
        if self.sender() == self.min_detections_checkbox:
            self.min_detections_spin.setEnabled(checked)
        elif self.sender() == self.min_classes_checkbox:
            self.min_classes_spin.setEnabled(checked)
        self._on_filtering_changed()
    
    def clear_sorting(self):
        """Clear sorting to default."""
        self.sort_combo.setCurrentIndex(0)  # Default
        self.order_combo.setCurrentIndex(0)  # Ascending
        self._on_sorting_changed()
    
    def clear_filters(self):
        """Clear all filters."""
        if self._enable_detection_filters:
            self.min_detections_checkbox.setChecked(False)
            self.min_classes_checkbox.setChecked(False)
            self.min_detections_spin.setValue(1)
            self.min_classes_spin.setValue(1)
            self._on_filtering_changed()
    
    def get_sort_option(self) -> SortOption:
        """Get current sort option."""
        text = self.sort_combo.currentText()
        for option in SortOption:
            if option.value == text:
                return option
        return SortOption.DEFAULT
    
    def get_sort_order(self) -> SortOrder:
        """Get current sort order."""
        text = self.order_combo.currentText()
        for order in SortOrder:
            if order.value == text:
                return order
        return SortOrder.ASCENDING
    
    def get_filter_settings(self) -> Dict:
        """Get current filter settings."""
        settings = {}
        if self._enable_detection_filters:
            if self.min_detections_checkbox.isChecked():
                settings['min_detections'] = self.min_detections_spin.value()
            if self.min_classes_checkbox.isChecked():
                settings['min_classes'] = self.min_classes_spin.value()
        return settings
    
    def sort_image_data(self, image_data: List[Tuple], annotations_dict: Optional[Dict] = None) -> List[Tuple]:
        """
        Sort image data based on current settings.
        
        Args:
            image_data: List of (path, metadata) tuples
            annotations_dict: Optional dict of path -> annotations
            
        Returns:
            Sorted list of image data
        """
        sort_option = self.get_sort_option()
        sort_order = self.get_sort_order()
        
        if sort_option == SortOption.DEFAULT:
            # For default sorting, only apply order if it's descending
            if sort_order == SortOrder.DESCENDING:
                # Reverse the original order
                return list(reversed(image_data))
            else:
                # Keep original order
                return image_data
        
        # Define sort key functions
        def get_sort_key(item):
            path, metadata = item if isinstance(item, tuple) else (item, {})
            path_obj = Path(path)
            
            if sort_option == SortOption.NAME:
                return path_obj.name.lower()
            elif sort_option == SortOption.DATE_MODIFIED:
                try:
                    return path_obj.stat().st_mtime
                except:
                    return 0
            elif sort_option == SortOption.FILE_SIZE:
                try:
                    return path_obj.stat().st_size
                except:
                    return 0
            elif sort_option == SortOption.DETECTION_COUNT and annotations_dict:
                return len(annotations_dict.get(path, []))
            elif sort_option == SortOption.CLASS_COUNT and annotations_dict:
                annotations = annotations_dict.get(path, [])
                unique_classes = set(ann[0] for ann in annotations if len(ann) > 0)
                return len(unique_classes)
            else:
                return 0
        
        # Sort the data
        reverse = (sort_order == SortOrder.DESCENDING)
        sorted_data = sorted(image_data, key=get_sort_key, reverse=reverse)
        
        return sorted_data
    
    def filter_image_data(self, image_data: List[Tuple], annotations_dict: Optional[Dict] = None) -> List[Tuple]:
        """
        Filter image data based on current settings.
        
        Args:
            image_data: List of (path, metadata) tuples
            annotations_dict: Optional dict of path -> annotations
            
        Returns:
            Filtered list of image data
        """
        if not self._enable_detection_filters:
            return image_data
        
        filter_settings = self.get_filter_settings()
        if not filter_settings:
            return image_data
        
        filtered_data = []
        for item in image_data:
            path = item[0] if isinstance(item, tuple) else item
            
            if annotations_dict and path in annotations_dict:
                annotations = annotations_dict[path]
                
                # Check minimum detections
                if 'min_detections' in filter_settings:
                    if len(annotations) < filter_settings['min_detections']:
                        continue
                
                # Check minimum classes
                if 'min_classes' in filter_settings:
                    unique_classes = set(ann[0] for ann in annotations if len(ann) > 0)
                    if len(unique_classes) < filter_settings['min_classes']:
                        continue
            else:
                # No annotations - only skip if filters require annotations
                skip_item = False
                if 'min_detections' in filter_settings:
                    # If requiring minimum detections and this image has no annotations, skip it
                    skip_item = True
                if 'min_classes' in filter_settings:
                    # If requiring minimum classes and this image has no annotations, skip it
                    skip_item = True
                
                if skip_item:
                    continue
            
            filtered_data.append(item)
        
        return filtered_data
    
    def apply_sort_and_filter(self, image_data: List, annotations_dict: Optional[Dict] = None) -> List:
        """
        Apply both sorting and filtering to image data.
        
        Args:
            image_data: List of image paths or (path, metadata) tuples
            annotations_dict: Optional dict of path -> annotations
            
        Returns:
            Sorted and filtered list
        """
        # Remember if we started with simple paths
        return_simple_paths = image_data and not isinstance(image_data[0], tuple)
        
        # Convert simple paths to tuples if needed
        if return_simple_paths:
            image_data = [(path, {}) for path in image_data]
        
        # Apply filtering first
        filtered = self.filter_image_data(image_data, annotations_dict)
        
        # Then apply sorting
        sorted_filtered = self.sort_image_data(filtered, annotations_dict)
        
        # Extract just the paths if that's what we started with
        if return_simple_paths:
            return [path for path, _ in sorted_filtered]
        
        return sorted_filtered