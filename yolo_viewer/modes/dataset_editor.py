"""Dataset Editor mode for creating and editing YOLO annotations."""

from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QComboBox,
    QLabel, QPushButton, QGroupBox, QSpinBox, QMessageBox,
    QFileDialog, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QRectF
from PyQt6.QtGui import QPixmap

from .base_mode import BaseMode
from ..widgets import EnhancedThumbnailGallery, AnnotationCanvas
from ..widgets.annotation_canvas import Annotation
from ..widgets.sort_filter_widget import SortFilterWidget, SortOption
from ..utils.yolo_format import (
    parse_yolo_annotation, save_yolo_annotation, load_data_yaml,
    get_image_paths_from_dataset, get_annotation_path, denormalize_bbox
)
from ..core.constants import IMAGE_EXTENSIONS
from ..core import DatasetManager
from ..dialogs.dataset_config_dialog import DatasetConfigDialog


@dataclass
class DatasetInfo:
    """Information about loaded dataset."""
    yaml_path: Path
    base_path: Path
    class_names: List[str]
    splits: Dict[str, str]  # split name -> relative path


class DatasetEditorMode(BaseMode):
    """Dataset editor mode for annotating images."""
    
    # Custom signals
    datasetLoaded = pyqtSignal(Path)
    annotationsSaved = pyqtSignal(int)  # number of files saved
    
    def __init__(self, parent=None):
        # Initialize attributes before calling super().__init__()
        self._current_image_path: Optional[Path] = None
        self._dataset_info: Optional[DatasetInfo] = None
        self._modified_files: set = set()
        self._current_split = 'test'
        self._loading = False  # Prevent double loading
        self._deactivating = False  # Flag to prevent save dialogs during deactivation
        self._dataset_manager = DatasetManager()
        
        # Now call super which will call _setup_ui
        super().__init__(parent)
        
        # Connect to dataset manager signals
        self._dataset_manager.datasetLoaded.connect(self._on_dataset_manager_loaded)
        self._dataset_manager.datasetUpdated.connect(self._on_dataset_manager_updated)
    
    def _setup_ui(self):
        """Setup the UI for dataset editor mode."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Top controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        # Dataset info
        self.dataset_label = QLabel("No dataset loaded")
        self.dataset_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        controls_layout.addWidget(self.dataset_label)
        
        controls_layout.addWidget(QLabel("|"))
        
        controls_layout.addWidget(QLabel("Split:"))
        self.split_combo = QComboBox()
        self.split_combo.addItems(['train', 'val', 'test'])
        # Connect signal before setting current text to avoid double loading
        self.split_combo.currentTextChanged.connect(self._on_split_changed)
        self.split_combo.setCurrentText('val')  # Start with val split so that the user does not have to wait if there's a giant train set
        self.split_combo.setMinimumWidth(100)
        controls_layout.addWidget(self.split_combo)
        
        # Add stretch to push class dropdown more to center
        controls_layout.addStretch(1)
        
        controls_layout.addWidget(QLabel("|"))
        
        # Make class label and dropdown more prominent
        class_label = QLabel("Class:")
        class_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        controls_layout.addWidget(class_label)
        
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        self.class_combo.setMinimumWidth(200)  # Increased width
        controls_layout.addWidget(self.class_combo)
        
        # Add another stretch to balance the layout
        controls_layout.addStretch(2)
        
        self.save_btn = QPushButton("💾 Save Changes")
        self.save_btn.clicked.connect(self._save_annotations)
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip("Save all annotation changes")
        controls_layout.addWidget(self.save_btn)
        
        layout.addLayout(controls_layout)
        
        # Main content - splitter with gallery and canvas
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Gallery with header
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        
        gallery_group = QGroupBox("Image Gallery")
        gallery_layout = QVBoxLayout(gallery_group)
        gallery_layout.setContentsMargins(5, 5, 5, 5)
        gallery_layout.setSpacing(5)
        
        # Gallery controls
        gallery_controls = QHBoxLayout()
        gallery_controls.setContentsMargins(0, 0, 0, 0)
        
        self.load_dataset_btn = QPushButton("📁 Load Dataset")
        self.load_dataset_btn.clicked.connect(self.open_dataset_dialog)
        self.load_dataset_btn.setToolTip("Load a YOLO dataset from data.yaml")
        gallery_controls.addWidget(self.load_dataset_btn)
        
        self.create_dataset_btn = QPushButton("➕ Create Dataset")
        self.create_dataset_btn.clicked.connect(self.create_dataset_dialog)
        self.create_dataset_btn.setToolTip("Create a new YOLO dataset configuration")
        gallery_controls.addWidget(self.create_dataset_btn)
        
        self.edit_dataset_btn = QPushButton("✏️ Edit Dataset")
        self.edit_dataset_btn.clicked.connect(self.edit_dataset_dialog)
        self.edit_dataset_btn.setToolTip("Edit current dataset configuration")
        self.edit_dataset_btn.setEnabled(False)  # Disabled until dataset is loaded
        gallery_controls.addWidget(self.edit_dataset_btn)
        
        gallery_controls.addStretch()
        
        gallery_layout.addLayout(gallery_controls)
        
        # Add sort/filter widget (collapsed by default)
        self.sort_filter_widget = SortFilterWidget(self, enable_detection_filters=False, start_collapsed=True)
        self.sort_filter_widget.sortingChanged.connect(self._on_sorting_changed)
        gallery_layout.addWidget(self.sort_filter_widget)
        
        # Thumbnail gallery - takes up all available space
        self.thumbnail_gallery = EnhancedThumbnailGallery()
        self.thumbnail_gallery.imageSelected.connect(self._on_image_selected)
        gallery_layout.addWidget(self.thumbnail_gallery, 1)  # stretch factor 1
        
        left_layout.addWidget(gallery_group)
        self.splitter.addWidget(left_panel)
        
        # Right panel - Canvas with header
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        canvas_group = QGroupBox("Annotation Canvas")
        canvas_layout = QVBoxLayout(canvas_group)
        canvas_layout.setContentsMargins(5, 5, 5, 5)
        canvas_layout.setSpacing(5)
        
        # Canvas info bar
        canvas_info_layout = QHBoxLayout()
        canvas_info_layout.setContentsMargins(0, 0, 0, 0)
        
        self.image_info_label = QLabel("No image selected")
        self.image_info_label.setStyleSheet("font-weight: bold;")
        canvas_info_layout.addWidget(self.image_info_label)
        
        canvas_info_layout.addWidget(QLabel("|"))
        
        self.annotation_count_label = QLabel("Annotations: 0")
        canvas_info_layout.addWidget(self.annotation_count_label)
        
        canvas_info_layout.addStretch()
        
        # Show class names checkbox
        self.show_names_checkbox = QCheckBox("Show Class Names")
        self.show_names_checkbox.setChecked(True)
        self.show_names_checkbox.toggled.connect(self._on_show_names_toggled)
        self.show_names_checkbox.setToolTip("Toggle between showing class names and IDs")
        canvas_info_layout.addWidget(self.show_names_checkbox)
        
        # Drawing tools hint
        hint_label = QLabel("✏️ Left-click drag to draw | 🗑️ DEL to delete | 🔢 0-9 keys to change class | 🖱️ Ctrl+click or Middle mouse to pan | 🔍 Scroll to zoom")
        hint_label.setStyleSheet("color: #888888; font-size: 12px;")
        canvas_info_layout.addWidget(hint_label)
        
        canvas_layout.addLayout(canvas_info_layout)
        
        # Annotation canvas - takes up all available space
        self.annotation_canvas = AnnotationCanvas()
        self.annotation_canvas.annotationAdded.connect(self._on_annotation_changed)
        self.annotation_canvas.annotationModified.connect(self._on_annotation_changed)
        self.annotation_canvas.annotationDeleted.connect(self._on_annotation_changed)
        self.annotation_canvas.selectionChanged.connect(self._on_annotation_selection_changed)
        canvas_layout.addWidget(self.annotation_canvas, 1)  # stretch factor 1
        
        right_layout.addWidget(canvas_group)
        self.splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (35% / 65%)
        self.splitter.setSizes([350, 650])
        
        layout.addWidget(self.splitter, 1)  # stretch factor 1 for splitter
        
        # Status bar
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(5, 5, 5, 0)
        
        # Status info
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        # Modified files indicator
        self.modified_label = QLabel("")
        self.modified_label.setStyleSheet("color: #ff9800; font-weight: bold;")
        status_layout.addWidget(self.modified_label)
        
        layout.addLayout(status_layout)
    
    def _on_activate(self):
        """Called when mode is activated."""
        self.statusMessage.emit("Dataset Editor mode activated - Draw annotations on images", 3000)
        
        # Subscribe to dataset manager signals
        self._dataset_manager.datasetLoaded.connect(self._on_dataset_manager_loaded)
        self._dataset_manager.datasetUpdated.connect(self._on_dataset_manager_updated)
        
        # Check if dataset manager has a dataset loaded
        if self._dataset_manager.has_dataset() and not self._dataset_info:
            # Load dataset from dataset manager
            yaml_path = self._dataset_manager.get_yaml_path()
            if yaml_path:
                self._load_from_dataset_manager()
        
        self._update_ui_state()
        
        # Show hint if no dataset loaded
        if not self._dataset_info:
            self.status_label.setText("📁 Load a dataset to start editing annotations")
    
    def _on_deactivate(self) -> Optional[bool]:
        """Called when mode is deactivated."""
        # Disconnect dataset manager signals
        try:
            self._dataset_manager.datasetLoaded.disconnect(self._on_dataset_manager_loaded)
            self._dataset_manager.datasetUpdated.disconnect(self._on_dataset_manager_updated)
        except:
            pass  # Signals might not be connected
            
        # Set flag to prevent image selection dialogs
        self._deactivating = True
        
        try:
            # Check for unsaved changes
            if self.has_unsaved_changes():
                # Build message with count of modified files
                count = len(self._modified_files)
                msg = f"You have unsaved changes to {count} image{'s' if count > 1 else ''}.\n\nDo you want to save them?"
                
                reply = QMessageBox.question(
                    self, "Unsaved Changes",
                    msg,
                    QMessageBox.StandardButton.Save | 
                    QMessageBox.StandardButton.Discard | 
                    QMessageBox.StandardButton.Cancel
                )
                
                if reply == QMessageBox.StandardButton.Save:
                    self._save_all_modifications()
                elif reply == QMessageBox.StandardButton.Discard:
                    # Reload original annotations for all modified files
                    for img_path in list(self._modified_files):  # Copy the set to avoid modification during iteration
                        img_path_obj = Path(img_path)
                        ann_path = get_annotation_path(img_path_obj)
                        if ann_path.exists():
                            # Reload original annotations from disk
                            annotations = parse_yolo_annotation(ann_path)
                            self.thumbnail_gallery.update_image_annotations(
                                img_path, annotations, is_modified=False
                            )
                        else:
                            # No annotations on disk
                            self.thumbnail_gallery.update_image_annotations(
                                img_path, [], is_modified=False
                            )
                    
                    # Clear the modified files without saving
                    self._modified_files.clear()
                    
                    # If current image was modified, reload it from disk  
                    if self._current_image_path:
                        # Reload annotations in the canvas without triggering selection
                        ann_path = get_annotation_path(self._current_image_path)
                        if ann_path.exists():
                            annotations = parse_yolo_annotation(ann_path)
                            # Convert to canvas annotations
                            canvas_annotations = []
                            pixmap = self.annotation_canvas._pixmap_item.pixmap()
                            for ann in annotations:
                                class_id, x_center, y_center, width, height = ann
                                x, y, w, h = denormalize_bbox(
                                    x_center, y_center, width, height,
                                    pixmap.width(), pixmap.height()
                                )
                                canvas_ann = Annotation(
                                    class_id=class_id,
                                    rect=QRectF(x, y, w, h)
                                )
                                canvas_annotations.append(canvas_ann)
                            self.annotation_canvas.set_annotations(canvas_annotations)
                        else:
                            self.annotation_canvas.set_annotations([])
                    
                    self._update_ui_state()
                elif reply == QMessageBox.StandardButton.Cancel:
                    return False
            
            return True
        finally:
            # Always reset the flag
            self._deactivating = False
    
    def get_mode_name(self) -> str:
        """Get the display name of this mode."""
        return "Dataset Editor"
    
    def has_unsaved_changes(self) -> bool:
        """Check if mode has unsaved changes."""
        return len(self._modified_files) > 0
    
    def save_changes(self) -> bool:
        """Save any pending changes."""
        try:
            self._save_all_modifications()
            return True
        except Exception as e:
            print(f"Error saving changes: {e}")
            return False
    
    def load_dataset(self, yaml_path: Path) -> bool:
        """
        Load a YOLO dataset from data.yaml.
        
        Args:
            yaml_path: Path to data.yaml file
            
        Returns:
            True if successful
        """
        self._loading = True
        try:
            # First update the dataset manager
            # Force reload if we're reloading the same dataset (e.g., after editing)
            force_reload = self._dataset_info and self._dataset_info.yaml_path == yaml_path
            if not self._dataset_manager.load_dataset(yaml_path, force_reload=force_reload):
                self.statusMessage.emit("Failed to load data.yaml", 5000)
                return False
            
            # Load configuration
            data = self._dataset_manager.get_raw_data()
            if not data:
                self.statusMessage.emit("Failed to load data.yaml", 5000)
                return False
            
            # Extract dataset info
            yaml_dir = yaml_path.parent
            
            # Check if there's a 'path' field in data.yaml (dataset root)
            if 'path' in data:
                base_path = yaml_dir / data['path']
            else:
                base_path = yaml_dir
                
            # Extract class names - handle both list and dict formats
            names_data = data.get('names', [])
            if isinstance(names_data, dict):
                # Names is a dict mapping class_id to name
                # Convert to a list ensuring correct order
                max_id = max(int(k) for k in names_data.keys()) if names_data else -1
                class_names = [''] * (max_id + 1)
                for k, v in names_data.items():
                    class_names[int(k)] = v
            else:
                # Names is already a list
                class_names = names_data
            splits = {}
            
            for split in ['train', 'val', 'test']:
                if split in data:
                    splits[split] = data[split]
            
            # Store dataset info
            self._dataset_info = DatasetInfo(
                yaml_path=yaml_path,
                base_path=base_path,
                class_names=class_names,
                splits=splits
            )
            
            # Update UI
            self._update_class_combo()
            self.dataset_label.setText(f"{yaml_path.name}")
            
            # Load images from current split
            self._load_split_images()
            
            # Update UI state to enable edit button
            self._update_ui_state()
            
            self.datasetLoaded.emit(yaml_path)
            self.statusMessage.emit(f"Loaded dataset: {yaml_path.name}", 3000)
            return True
            
        except Exception as e:
            self.statusMessage.emit(f"Error loading dataset: {str(e)}", 5000)
            return False
        finally:
            self._loading = False
    
    def _update_class_combo(self):
        """Update class combo box with dataset classes."""
        self.class_combo.clear()
        
        if self._dataset_info and self._dataset_info.class_names:
            # Build class names dict for annotation canvas
            class_names_dict = {}
            for i, name in enumerate(self._dataset_info.class_names):
                self.class_combo.addItem(f"{i}: {name}", i)
                class_names_dict[i] = name
            
            # Update annotation canvas with new class names
            if hasattr(self, 'annotation_canvas'):
                self.annotation_canvas.set_class_names(class_names_dict)
        else:
            self.class_combo.addItem("Class 0", 0)
            # Set default class name
            if hasattr(self, 'annotation_canvas'):
                self.annotation_canvas.set_class_names({0: "Class 0"})
    
    def _load_split_images(self):
        """Load images for current split."""
        if not self._dataset_info:
            return
        
        # Get image paths
        image_paths = get_image_paths_from_dataset(
            self._dataset_info.yaml_path, 
            self._current_split
        )
        
        if not image_paths:
            self.statusMessage.emit(f"No images found in {self._current_split} split", 3000)
            return
        
        # Load annotations
        annotations_dict = {}
        for img_path in image_paths:
            ann_path = get_annotation_path(img_path)
            if ann_path.exists():
                annotations = parse_yolo_annotation(ann_path)
                annotations_dict[str(img_path)] = annotations
        
        # Update gallery
        unique_paths = [str(p) for p in image_paths]
        self.thumbnail_gallery.load_images(unique_paths, annotations_dict)
        
        self.statusMessage.emit(f"Loaded {len(image_paths)} images from {self._current_split} split", 3000)
    
    @pyqtSlot(str)
    def _on_split_changed(self, split: str):
        """Handle split selection change."""
        # Check if we're still initializing or loading
        if not hasattr(self, '_loading') or self._loading or split == self._current_split:
            return
            
        if split != self._current_split:
            # Check for unsaved changes
            if self.has_unsaved_changes():
                reply = QMessageBox.question(
                    self, "Unsaved Changes",
                    "Save changes before switching split?",
                    QMessageBox.StandardButton.Save | 
                    QMessageBox.StandardButton.Discard | 
                    QMessageBox.StandardButton.Cancel
                )
                
                if reply == QMessageBox.StandardButton.Save:
                    self._save_all_modifications()
                elif reply == QMessageBox.StandardButton.Cancel:
                    # Restore previous selection
                    self.split_combo.setCurrentText(self._current_split)
                    return
            
            self._current_split = split
            self._load_split_images()
    
    @pyqtSlot(str)
    def _on_image_selected(self, image_path: str):
        """Handle image selection from gallery."""
        # Skip if we're in the process of deactivating (tab switch)
        if self._deactivating:
            return
        
        # Don't save automatically - keep changes in memory until user saves or discards
        # This allows users to work on multiple images before saving
        
        # Load new image
        self._current_image_path = Path(image_path)
        pixmap = QPixmap(image_path)
        
        if pixmap.isNull():
            self.statusMessage.emit(f"Failed to load image: {image_path}", 3000)
            return
        
        # Update canvas
        self.annotation_canvas.load_image(pixmap)
        
        # Check if we have unsaved changes for this image
        if str(self._current_image_path) in self._modified_files:
            # Load from thumbnail gallery's current state (which has the unsaved changes)
            gallery_annotations = self.thumbnail_gallery.get_image_annotations(str(self._current_image_path))
            if gallery_annotations:
                # Convert to canvas annotations
                canvas_annotations = []
                for ann in gallery_annotations:
                    class_id, x_center, y_center, width, height = ann
                    
                    # Denormalize to pixel coordinates
                    x, y, w, h = denormalize_bbox(
                        x_center, y_center, width, height,
                        pixmap.width(), pixmap.height()
                    )
                    
                    canvas_ann = Annotation(
                        class_id=class_id,
                        rect=QRectF(x, y, w, h)
                    )
                    canvas_annotations.append(canvas_ann)
                
                self.annotation_canvas.set_annotations(canvas_annotations)
            else:
                self.annotation_canvas.set_annotations([])
        else:
            # Load annotations from disk
            ann_path = get_annotation_path(self._current_image_path)
            if ann_path.exists():
                annotations = parse_yolo_annotation(ann_path)
                
                # Convert to canvas annotations
                canvas_annotations = []
                for ann in annotations:
                    class_id, x_center, y_center, width, height = ann
                    
                    # Denormalize to pixel coordinates
                    x, y, w, h = denormalize_bbox(
                        x_center, y_center, width, height,
                        pixmap.width(), pixmap.height()
                    )
                    
                    canvas_ann = Annotation(
                        class_id=class_id,
                        rect=QRectF(x, y, w, h)
                    )
                    canvas_annotations.append(canvas_ann)
                
                self.annotation_canvas.set_annotations(canvas_annotations)
            else:
                self.annotation_canvas.set_annotations([])
        
        # Update info
        self.image_info_label.setText(f"Image: {self._current_image_path.name} ({pixmap.width()}x{pixmap.height()})")
        self._update_annotation_count()
    
    @pyqtSlot(Annotation)
    def _on_annotation_changed(self, annotation: Annotation):
        """Handle annotation changes."""
        if self._current_image_path:
            self._modified_files.add(str(self._current_image_path))
            
            # Update thumbnail
            annotations = self.annotation_canvas.get_annotations()
            yolo_annotations = []
            
            pixmap = self.annotation_canvas._pixmap_item.pixmap()
            for ann in annotations:
                yolo_ann = ann.to_yolo_format(pixmap.width(), pixmap.height())
                yolo_annotations.append(yolo_ann)
            
            self.thumbnail_gallery.update_image_annotations(
                str(self._current_image_path), 
                yolo_annotations,
                is_modified=True
            )
            
            self._update_ui_state()
            self._update_annotation_count()
    
    @pyqtSlot(int)
    def _on_class_changed(self, index: int):
        """Handle class selection change."""
        if index >= 0:
            class_id = self.class_combo.itemData(index)
            # This will update all selected annotations and emit annotationModified signals
            self.annotation_canvas.set_current_class(class_id)
    
    @pyqtSlot(bool)
    def _on_show_names_toggled(self, checked: bool):
        """Toggle between showing class names and IDs."""
        self.annotation_canvas.set_show_class_names(checked)
    
    def _on_sorting_changed(self):
        """Handle sorting/filtering changes."""
        if not self._dataset_info:
            return
        
        # Remember current selection
        current_selection = self.thumbnail_gallery.get_current_selected_path()
        
        # Get all image paths and annotations
        all_paths = self.thumbnail_gallery.get_all_image_paths()
        annotations_dict = self.thumbnail_gallery.get_annotations_dict()
        
        # Apply sorting and filtering
        sorted_paths = self.sort_filter_widget.apply_sort_and_filter(
            all_paths, annotations_dict
        )
        
        # Update gallery with sorted/filtered paths
        self.thumbnail_gallery.apply_sort_and_filter(sorted_paths)
        
        # Handle selection after sorting/filtering
        self._handle_selection_after_change(current_selection)
    
    def _handle_selection_after_change(self, previous_selection: Optional[str]):
        """Handle thumbnail selection after sorting/filtering changes.
        
        Args:
            previous_selection: Path of previously selected image, or None
        """
        if previous_selection:
            # Try to maintain current selection
            if not self.thumbnail_gallery.select_and_scroll_to_path(previous_selection):
                # Previous selection not in filtered results, select first
                self.thumbnail_gallery.select_first_item()
        else:
            # No previous selection, select first item
            self.thumbnail_gallery.select_first_item()
    
    def _on_annotation_selection_changed(self, selected_annotations: List[Annotation]):
        """Handle annotation selection changes - update class dropdown."""
        if selected_annotations and len(selected_annotations) == 1:
            # Single selection - update dropdown to show the annotation's class
            annotation = selected_annotations[0]
            
            # Find and set the combo box item that matches this class ID
            for i in range(self.class_combo.count()):
                if self.class_combo.itemData(i) == annotation.class_id:
                    # Block signals to avoid triggering class change
                    self.class_combo.blockSignals(True)
                    self.class_combo.setCurrentIndex(i)
                    self.class_combo.blockSignals(False)
                    break
        elif len(selected_annotations) > 1:
            # Multiple selection - check if all have same class
            class_ids = set(ann.class_id for ann in selected_annotations)
            if len(class_ids) == 1:
                # All annotations have the same class, update dropdown
                class_id = class_ids.pop()
                for i in range(self.class_combo.count()):
                    if self.class_combo.itemData(i) == class_id:
                        self.class_combo.blockSignals(True)
                        self.class_combo.setCurrentIndex(i)
                        self.class_combo.blockSignals(False)
                        break
    
    def _save_current_annotations(self):
        """Save annotations for current image if modified."""
        if self._current_image_path and str(self._current_image_path) in self._modified_files:
            ann_path = get_annotation_path(self._current_image_path)
            annotations = self.annotation_canvas.get_annotations()
            
            # Convert to YOLO format
            yolo_annotations = []
            pixmap = self.annotation_canvas._pixmap_item.pixmap()
            
            for ann in annotations:
                yolo_ann = ann.to_yolo_format(pixmap.width(), pixmap.height())
                yolo_annotations.append(yolo_ann)
            
            # Save to file
            save_yolo_annotation(ann_path, yolo_annotations)
            
            # Update thumbnail
            self.thumbnail_gallery.update_image_annotations(
                str(self._current_image_path),
                yolo_annotations,
                is_modified=False
            )
            
            # Remove from modified files since we saved it
            self._modified_files.discard(str(self._current_image_path))
    
    @pyqtSlot()
    def _save_annotations(self):
        """Save all modified annotations."""
        self._save_all_modifications()
    
    def _save_all_modifications(self):
        """Save all modified annotation files."""
        # Save current image first
        self._save_current_annotations()
        
        # Clear modified set
        saved_count = len(self._modified_files)
        self._modified_files.clear()
        
        self._update_ui_state()
        self.annotationsSaved.emit(saved_count)
        self.statusMessage.emit(f"Saved {saved_count} annotation files", 3000)
    
    def _update_ui_state(self):
        """Update UI elements based on current state."""
        has_changes = self.has_unsaved_changes()
        self.save_btn.setEnabled(has_changes)
        
        if has_changes:
            self.modified_label.setText(f"⚠️ Modified: {len(self._modified_files)} files")
        else:
            self.modified_label.setText("")
            
        # Update status
        if self._dataset_info:
            self.status_label.setText(f"Editing {self._current_split} split")
            self.edit_dataset_btn.setEnabled(True)
        else:
            self.status_label.setText("No dataset loaded")
            self.edit_dataset_btn.setEnabled(False)
    
    def _update_annotation_count(self):
        """Update annotation count display."""
        count = len(self.annotation_canvas.get_annotations())
        self.annotation_count_label.setText(f"📦 Annotations: {count}")
    
    def open_dataset_dialog(self):
        """Open dialog to select data.yaml file."""
        # Start in project directory (parent of parent of parent of this file)
        start_dir = str(Path(__file__).parent.parent.parent)
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Dataset",
            start_dir,
            "YAML files (*.yaml *.yml)"
        )
        
        if path:
            self.load_dataset(Path(path))
    
    @pyqtSlot(Path)
    def _on_dataset_manager_loaded(self, yaml_path: Path):
        """Handle dataset loaded in dataset manager."""
        # Only update if we don't have a dataset or it's different
        if not self._dataset_info or self._dataset_info.yaml_path != yaml_path:
            # Only load if UI exists (mode has been activated)
            if hasattr(self, 'dataset_label'):
                self._load_from_dataset_manager()
    
    @pyqtSlot()
    def _on_dataset_manager_updated(self):
        """Handle dataset updated in dataset manager (e.g., after editing data.yaml)."""
        # Refresh the dataset to get updated class names
        if self._dataset_info and hasattr(self, 'dataset_label'):
            # Get current yaml path
            yaml_path = self._dataset_manager.get_yaml_path()
            if yaml_path and yaml_path == self._dataset_info.yaml_path:
                # Same dataset was updated, refresh it
                self._load_from_dataset_manager()
    
    def _load_from_dataset_manager(self):
        """Load dataset information from dataset manager."""
        yaml_path = self._dataset_manager.get_yaml_path()
        if not yaml_path:
            return
            
        # Create dataset info from manager
        self._dataset_info = DatasetInfo(
            yaml_path=yaml_path,
            base_path=self._dataset_manager.get_dataset_root(),
            class_names=list(self._dataset_manager.get_class_names().values()),
            splits=self._dataset_manager.get_splits()
        )
        
        # Update UI
        self._update_class_combo()
        self.dataset_label.setText(f"{yaml_path.name}")
        
        # Load images from current split
        self._load_split_images()
        
        # Update UI state to enable edit button
        self._update_ui_state()
        
        self.datasetLoaded.emit(yaml_path)
        self.statusMessage.emit(f"Loaded dataset: {yaml_path.name}", 3000)
    
    def refresh_from_dataset_manager(self):
        """Refresh dataset from dataset manager (called when already in this mode)."""
        self._load_from_dataset_manager()
    
    def create_dataset_dialog(self):
        """Open dialog to create new dataset configuration."""
        dialog = DatasetConfigDialog(self)
        dialog.datasetSaved.connect(self._on_dataset_created)
        dialog.exec()
    
    def edit_dataset_dialog(self):
        """Open dialog to edit current dataset configuration."""
        if not self._dataset_info:
            return
            
        dialog = DatasetConfigDialog(self, yaml_path=self._dataset_info.yaml_path)
        dialog.datasetSaved.connect(self._on_dataset_edited)
        dialog.exec()
    
    @pyqtSlot(Path)
    def _on_dataset_created(self, yaml_path: Path):
        """Handle newly created dataset."""
        # Load the newly created dataset
        self.load_dataset(yaml_path)
    
    @pyqtSlot(Path) 
    def _on_dataset_edited(self, yaml_path: Path):
        """Handle edited dataset."""
        # Reload the dataset to reflect changes
        self.load_dataset(yaml_path)
        self.statusMessage.emit("Dataset configuration updated", 3000)