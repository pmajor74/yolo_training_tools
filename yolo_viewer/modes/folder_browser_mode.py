"""Folder Browser mode for annotating images in any folder without dataset structure."""

from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple
from enum import Enum
import shutil
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSplitter, QFileDialog, QMessageBox, QFrame,
    QComboBox, QProgressBar, QSlider, QSpinBox, QCheckBox,
    QProgressDialog, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QRectF, QEvent, QThread, QTimer
from PyQt6.QtGui import QPixmap

from .base_mode import BaseMode
from ..core import ImageCache, ModelCache, DatasetManager
from ..widgets.thumbnail_gallery import ThumbnailGallery
from ..widgets.annotation_canvas import AnnotationCanvas, Annotation
from ..widgets.image_viewer import ImageViewer
from ..utils.yolo_format import (
    parse_yolo_annotation, save_yolo_annotation, 
    get_annotation_path, denormalize_bbox, normalize_bbox
)
from ..core.constants import IMAGE_EXTENSIONS

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class ImageFilter(Enum):
    """Filter types for image display."""
    ALL = "all"
    ANNOTATED = "annotated"
    UNANNOTATED = "unannotated"


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    confidence: float
    x: float  # pixel coordinates
    y: float
    width: float
    height: float
    
    def to_yolo_format(self, img_width: int, img_height: int) -> Tuple[int, float, float, float, float]:
        """Convert to YOLO format (class_id, x_center, y_center, width, height) normalized."""
        x_center, y_center, norm_width, norm_height = normalize_bbox(
            self.x, self.y, self.width, self.height, img_width, img_height
        )
        return (self.class_id, x_center, y_center, norm_width, norm_height)


class InferenceThread(QThread):
    """Thread for running batch inference on provided images."""
    
    progress = pyqtSignal(int, int)  # current, total
    imageProcessed = pyqtSignal(str, list)  # image_path, detections
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, model, image_paths: List[Path], confidence_threshold: float):
        super().__init__()
        self.model = model
        self.image_paths = image_paths  # Can be a subset of all images
        self.confidence_threshold = confidence_threshold
        self._is_running = True
    
    def run(self):
        """Run inference on provided images."""
        try:
            total = len(self.image_paths)
            
            for i, img_path in enumerate(self.image_paths):
                if not self._is_running:
                    break
                    
                # Run inference (model already on GPU/CPU from ModelCache)
                results = self.model(str(img_path), conf=0.01)  # Low conf to get all detections
                
                # Extract detections
                detections = []
                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        boxes = result.boxes
                        for j in range(len(boxes)):
                            conf = float(boxes.conf[j])
                            cls = int(boxes.cls[j])
                            x1, y1, x2, y2 = boxes.xyxy[j].tolist()
                            
                            detection = Detection(
                                class_id=cls,
                                confidence=conf,
                                x=x1,
                                y=y1,
                                width=x2 - x1,
                                height=y2 - y1
                            )
                            detections.append(detection)
                
                self.imageProcessed.emit(str(img_path), detections)
                self.progress.emit(i + 1, total)
            
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))
    
    def stop(self):
        """Stop the inference thread."""
        self._is_running = False


class FolderBrowserMode(BaseMode):
    """
    Mode for browsing and annotating images in any folder.
    
    Features:
    - Browse any folder of images
    - Uses class names from loaded model
    - Filter by annotation status
    - Save annotations alongside images
    - Track progress and modifications
    """
    
    # Signals
    folderLoaded = pyqtSignal(str)  # folder path
    annotationsSaved = pyqtSignal(int)  # number of files saved
    progressUpdated = pyqtSignal(int, int)  # annotated, total
    
    def __init__(self, parent=None):
        # Initialize attributes before super().__init__()
        self._current_folder: Optional[Path] = None
        self._current_image_path: Optional[Path] = None
        self._class_names: Dict[int, str] = {}
        self._modified_files: Set[str] = set()
        self._pending_annotations: Dict[str, List[Annotation]] = {}  # Store pending changes per image
        self._original_yolo_annotations: Dict[str, List[tuple]] = {}  # Store original YOLO format to preserve precision
        self._current_filter = ImageFilter.ALL
        self._all_image_paths: List[Path] = []
        self._annotated_paths: Set[str] = set()
        self._loading = False
        
        # Inference-related attributes
        self._inference_mode = False
        self._detections: Dict[str, List[Detection]] = {}
        self._filtered_detections: Dict[str, List[Detection]] = {}
        self._confidence_threshold = 0.25
        self._inference_thread: Optional[InferenceThread] = None
        self._show_below_threshold = False
        
        super().__init__(parent)
        
        # Enable keyboard focus for the widget
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Check for model on init
        QTimer.singleShot(100, self._check_model_status)
    
    def _setup_ui(self):
        """Setup the UI for folder browser mode."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Create three-panel layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Controls (20% width)
        self._controls_panel = self._create_controls_panel()
        splitter.addWidget(self._controls_panel)
        
        # Center panel - Thumbnail gallery (40% width)
        self._gallery_panel = self._create_gallery_panel()
        splitter.addWidget(self._gallery_panel)
        
        # Right panel - Annotation editor (40% width)
        self._editor_panel = self._create_editor_panel()
        splitter.addWidget(self._editor_panel)
        
        # Set splitter sizes (20%, 40%, 40%)
        splitter.setSizes([200, 400, 400])
        
        layout.addWidget(splitter)
    
    def _create_controls_panel(self) -> QWidget:
        """Create the controls panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Folder selection
        folder_group = QGroupBox("Folder Selection")
        folder_layout = QVBoxLayout()
        
        self._browse_folder_btn = QPushButton("📁 Browse Folder")
        self._browse_folder_btn.clicked.connect(self._browse_folder)
        folder_layout.addWidget(self._browse_folder_btn)
        
        self._load_dataset_btn = QPushButton("📊 Load Dataset")
        self._load_dataset_btn.clicked.connect(self._load_dataset)
        self._load_dataset_btn.setToolTip("Load a YOLO dataset to use its class names")
        folder_layout.addWidget(self._load_dataset_btn)
        
        self._current_folder_label = QLabel("No folder selected")
        self._current_folder_label.setWordWrap(True)
        self._current_folder_label.setStyleSheet("color: #888;")
        folder_layout.addWidget(self._current_folder_label)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        
        # Model status
        model_group = QGroupBox("Model Status")
        model_layout = QVBoxLayout()
        
        self._model_status_label = QLabel("No model loaded")
        self._model_status_label.setWordWrap(True)
        self._model_status_label.setStyleSheet("color: #ff6b6b;")
        model_layout.addWidget(self._model_status_label)
        
        self._class_info_label = QLabel("")
        self._class_info_label.setWordWrap(True)
        self._class_info_label.setStyleSheet("color: #888; font-size: 11px;")
        model_layout.addWidget(self._class_info_label)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self._progress_bar = QProgressBar()
        progress_layout.addWidget(self._progress_bar)
        
        self._progress_label = QLabel("0/0 images annotated")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self._progress_label)
        
        self._modified_label = QLabel("")
        self._modified_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._modified_label.setStyleSheet("color: #ff9800; font-weight: bold;")
        progress_layout.addWidget(self._modified_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Class Statistics
        stats_group = QGroupBox("Class Statistics")
        stats_layout = QVBoxLayout()
        
        self._class_stats_label = QLabel("No classes loaded")
        self._class_stats_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        self._class_stats_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        stats_layout.addWidget(self._class_stats_label)
        
        # Total annotations count
        self._total_annotations_label = QLabel("Total annotations: 0")
        self._total_annotations_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        stats_layout.addWidget(self._total_annotations_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        self._save_all_btn = QPushButton("💾 Save All Changes")
        self._save_all_btn.clicked.connect(self._save_all_annotations)
        self._save_all_btn.setEnabled(False)
        actions_layout.addWidget(self._save_all_btn)
        
        self._clear_current_btn = QPushButton("🗑️ Clear Current")
        self._clear_current_btn.clicked.connect(self._clear_current_annotations)
        self._clear_current_btn.setToolTip("Clear all annotations from current image")
        actions_layout.addWidget(self._clear_current_btn)
        
        self._export_dataset_btn = QPushButton("📦 Export as Dataset")
        self._export_dataset_btn.clicked.connect(self._export_as_dataset)
        self._export_dataset_btn.setToolTip("Export annotated images as YOLO dataset")
        actions_layout.addWidget(self._export_dataset_btn)
        
        self._reject_selected_btn = QPushButton("❌ Reject Selected Image(s)")
        self._reject_selected_btn.clicked.connect(self._reject_selected_images)
        self._reject_selected_btn.setToolTip("Move selected images to 'rejected' folder")
        actions_layout.addWidget(self._reject_selected_btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Inference controls
        inference_group = QGroupBox("Inference")
        inference_layout = QVBoxLayout()
        
        self._run_inference_btn = QPushButton("🚀 Run Inference")
        self._run_inference_btn.clicked.connect(self._run_inference)
        self._run_inference_btn.setToolTip("Run model predictions on all images")
        self._run_inference_btn.setEnabled(False)
        inference_layout.addWidget(self._run_inference_btn)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        
        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(1, 100)
        self._conf_slider.setValue(int(self._confidence_threshold * 100))
        self._conf_slider.valueChanged.connect(self._on_confidence_changed)
        conf_layout.addWidget(self._conf_slider)
        
        self._conf_spinbox = QSpinBox()
        self._conf_spinbox.setRange(1, 100)
        self._conf_spinbox.setValue(int(self._confidence_threshold * 100))
        self._conf_spinbox.setSuffix("%")
        self._conf_spinbox.valueChanged.connect(self._on_confidence_spin_changed)
        conf_layout.addWidget(self._conf_spinbox)
        
        inference_layout.addLayout(conf_layout)
        
        # Show below threshold checkbox
        self._show_below_cb = QCheckBox("Show Below Threshold")
        self._show_below_cb.setChecked(self._show_below_threshold)  # Will be False by default
        self._show_below_cb.stateChanged.connect(self._on_show_below_changed)
        self._show_below_cb.setToolTip("Display detections below confidence threshold in red")
        inference_layout.addWidget(self._show_below_cb)
        
        # Export annotations button
        self._export_annotations_btn = QPushButton("💾 Export Selected Annotations")
        self._export_annotations_btn.clicked.connect(self._export_annotations)
        self._export_annotations_btn.setToolTip("Export annotations for selected images")
        self._export_annotations_btn.setEnabled(False)
        inference_layout.addWidget(self._export_annotations_btn)
        
        inference_group.setLayout(inference_layout)
        layout.addWidget(inference_group)
        
        layout.addStretch()
        return panel
    
    def _create_gallery_panel(self) -> QWidget:
        """Create the thumbnail gallery panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Gallery header with filters
        header_layout = QHBoxLayout()
        self._gallery_label = QLabel("Images")
        self._gallery_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self._gallery_label)
        
        header_layout.addStretch()
        
        # Filter buttons
        self._filter_all_btn = QPushButton("All")
        self._filter_all_btn.setCheckable(True)
        self._filter_all_btn.setChecked(True)
        self._filter_all_btn.clicked.connect(lambda: self._set_filter(ImageFilter.ALL))
        self._filter_all_btn.setMaximumWidth(80)
        header_layout.addWidget(self._filter_all_btn)
        
        self._filter_annotated_btn = QPushButton("Annotated")
        self._filter_annotated_btn.setCheckable(True)
        self._filter_annotated_btn.clicked.connect(lambda: self._set_filter(ImageFilter.ANNOTATED))
        self._filter_annotated_btn.setMaximumWidth(100)
        header_layout.addWidget(self._filter_annotated_btn)
        
        self._filter_unannotated_btn = QPushButton("Unannotated")
        self._filter_unannotated_btn.setCheckable(True)
        self._filter_unannotated_btn.clicked.connect(lambda: self._set_filter(ImageFilter.UNANNOTATED))
        self._filter_unannotated_btn.setMaximumWidth(120)
        header_layout.addWidget(self._filter_unannotated_btn)
        
        layout.addLayout(header_layout)
        
        # Selection controls
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Selection:"))
        
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.clicked.connect(self._select_all_images)
        self._select_all_btn.setMaximumWidth(100)
        selection_layout.addWidget(self._select_all_btn)
        
        self._select_none_btn = QPushButton("Select None")
        self._select_none_btn.clicked.connect(self._select_none_images)
        self._select_none_btn.setMaximumWidth(100)
        selection_layout.addWidget(self._select_none_btn)
        
        selection_layout.addStretch()
        layout.addLayout(selection_layout)
        
        # Thumbnail gallery
        self._gallery = ThumbnailGallery()
        self._gallery.imageSelected.connect(self._on_image_selected)
        self._gallery.selectionChanged.connect(self._on_gallery_selection_changed)
        
        # Install event filter to save annotations before selecting new image
        self._gallery.installEventFilter(self)
        
        layout.addWidget(self._gallery)
        
        return panel
    
    def _create_editor_panel(self) -> QWidget:
        """Create the annotation editor panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Editor header
        header_layout = QHBoxLayout()
        
        self._image_info_label = QLabel("No image selected")
        self._image_info_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self._image_info_label)
        
        header_layout.addStretch()
        
        # Mode toggle button
        self._mode_toggle_btn = QPushButton("🎨 Switch to Inference")
        self._mode_toggle_btn.setCheckable(True)
        self._mode_toggle_btn.setChecked(True)
        self._mode_toggle_btn.clicked.connect(self._toggle_mode)
        self._mode_toggle_btn.setToolTip("Toggle between Edit and Inference modes")
        header_layout.addWidget(self._mode_toggle_btn)
        
        # Class selection (for edit mode)
        self._class_label = QLabel("Class:")
        header_layout.addWidget(self._class_label)
        self._class_combo = QComboBox()
        self._class_combo.setMinimumWidth(150)
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)
        header_layout.addWidget(self._class_combo)
        
        # Annotation/Detection count
        self._annotation_count_label = QLabel("Annotations: 0")
        header_layout.addWidget(self._annotation_count_label)
        
        layout.addLayout(header_layout)
        
        # Drawing hint (for edit mode)
        self._hint_label = QLabel("✏️ Left-click drag to draw | 🗑️ DEL to delete | 🔢 0-9 keys to change class | 🔍 Ctrl+scroll to zoom")
        self._hint_label.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(self._hint_label)
        
        # Stacked widget to switch between canvas and viewer
        from PyQt6.QtWidgets import QStackedWidget
        self._editor_stack = QStackedWidget()
        
        # Annotation canvas (index 0)
        self._canvas = AnnotationCanvas()
        self._canvas.annotationAdded.connect(self._on_annotation_changed)
        self._canvas.annotationModified.connect(self._on_annotation_changed)
        self._canvas.annotationDeleted.connect(self._on_annotation_changed)
        self._canvas.selectionChanged.connect(self._on_annotation_selection_changed)
        self._canvas.installEventFilter(self)
        self._editor_stack.addWidget(self._canvas)
        
        # Image viewer (index 1)
        self._viewer = ImageViewer()
        self._editor_stack.addWidget(self._viewer)
        
        layout.addWidget(self._editor_stack)
        
        return panel
    
    def _browse_folder(self):
        """Browse and select a folder containing images."""
        # Save current annotations before browsing new folder
        if self._current_image_path and str(self._current_image_path) in self._modified_files:
            self._save_current_annotations()
            self._update_progress()
        
        # Start from current folder or project root
        start_dir = str(self._current_folder) if self._current_folder else str(Path(__file__).parent.parent.parent)
        
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", start_dir
        )
        
        if folder:
            self._load_folder(Path(folder))
    
    def _load_folder(self, folder: Path):
        """Load images from the specified folder."""
        # Set busy cursor
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            self._current_folder = folder
            self._current_folder_label.setText(f"📁 {folder.name}")
            self._current_folder_label.setStyleSheet("color: #0d7377;")
            
            # Find all image files (using a set to avoid duplicates)
            image_paths_set = set()
            for ext in IMAGE_EXTENSIONS:
                image_paths_set.update(folder.glob(f"*{ext}"))
                image_paths_set.update(folder.glob(f"*{ext.upper()}"))
            
            # Convert to sorted list
            self._all_image_paths = sorted(list(image_paths_set))
            
            # Check which images have annotations
            self._annotated_paths.clear()
            self._original_yolo_annotations.clear()  # Clear cached original annotations
            annotations_dict = {}
            
            for img_path in self._all_image_paths:
                ann_path = get_annotation_path(img_path)
                if ann_path.exists():
                    self._annotated_paths.add(str(img_path))
                    annotations = parse_yolo_annotation(ann_path)
                    if annotations:
                        annotations_dict[str(img_path)] = annotations
            
            # Update gallery with all images
            self._apply_filter()
            
            # Update progress
            self._update_progress()
            
            # Update class statistics
            if self._class_names:
                self._update_class_statistics()
            
            # Enable/disable controls
            self._update_ui_state()
            
            self.folderLoaded.emit(str(folder))
        finally:
            # Restore normal cursor
            QApplication.restoreOverrideCursor()
        self.statusMessage.emit(f"Loaded {len(self._all_image_paths)} images from {folder.name}", 3000)
    
    def _load_dataset(self):
        """Load a YOLO dataset to get class names."""
        # Start from the project root directory (where the main script is)
        start_dir = str(Path(__file__).parent.parent.parent)
        
        # Ask user to select data.yaml file
        yaml_file, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset YAML File",
            start_dir,
            "YAML Files (*.yaml *.yml);;All Files (*.*)"
        )
        
        if not yaml_file:
            return
        
        yaml_path = Path(yaml_file)
        
        # Load dataset using DatasetManager
        dataset_manager = DatasetManager()
        if dataset_manager.load_dataset(yaml_path):
            # Get class names from dataset
            self._class_names = dataset_manager.get_class_names()
            
            # Update UI components
            self._update_class_combo()
            self._canvas.set_class_names(self._class_names)
            self._viewer.set_class_names(self._class_names)
            
            # Update status
            self._model_status_label.setText(f"✅ Dataset loaded with {len(self._class_names)} classes")
            self._model_status_label.setStyleSheet("color: #51cf66;")
            
            # Show class names
            class_list = ", ".join([f"{k}: {v}" for k, v in sorted(self._class_names.items())[:5]])
            if len(self._class_names) > 5:
                class_list += f" ... and {len(self._class_names) - 5} more"
            self._class_info_label.setText(f"Classes: {class_list}")
            
            # Update class statistics if we have a folder loaded
            if self._current_folder:
                self._update_class_statistics()
            
            # Update UI state
            self._update_ui_state()
            
            self.statusMessage.emit(f"Loaded dataset: {yaml_path.name}", 3000)
        else:
            QMessageBox.warning(
                self, "Dataset Load Failed",
                f"Failed to load dataset from:\n{yaml_path}\n\n"
                "Make sure it's a valid YOLO data.yaml file."
            )
    
    def _update_class_combo(self):
        """Update class combo box."""
        self._class_combo.clear()
        
        if self._class_names:
            for class_id, class_name in sorted(self._class_names.items()):
                self._class_combo.addItem(f"{class_id}: {class_name}", class_id)
        else:
            self._class_combo.addItem("(Load model first)", None)
            self._class_combo.setEnabled(False)
    
    def _set_filter(self, filter_type: ImageFilter):
        """Set the current image filter."""
        # Save current annotations before changing filter
        if self._current_image_path:
            self._save_current_annotations()
            # Note: _save_current_annotations already updates progress and statistics
        
        self._current_filter = filter_type
        
        # Update button states
        self._filter_all_btn.setChecked(filter_type == ImageFilter.ALL)
        self._filter_annotated_btn.setChecked(filter_type == ImageFilter.ANNOTATED)
        self._filter_unannotated_btn.setChecked(filter_type == ImageFilter.UNANNOTATED)
        
        # Apply filter
        self._apply_filter()
    
    def _apply_filter(self):
        """Apply the current filter to the gallery."""
        if not self._all_image_paths:
            return
        
        # Filter paths based on current filter
        if self._current_filter == ImageFilter.ALL:
            filtered_paths = self._all_image_paths
        elif self._current_filter == ImageFilter.ANNOTATED:
            filtered_paths = [p for p in self._all_image_paths if str(p) in self._annotated_paths]
        else:  # UNANNOTATED
            filtered_paths = [p for p in self._all_image_paths if str(p) not in self._annotated_paths]
        
        # Load annotations for filtered images
        annotations_dict = {}
        for img_path in filtered_paths:
            if str(img_path) in self._annotated_paths:
                ann_path = get_annotation_path(img_path)
                if ann_path.exists():
                    annotations = parse_yolo_annotation(ann_path)
                    if annotations:
                        annotations_dict[str(img_path)] = annotations
        
        # Update gallery
        self._gallery.load_images([str(p) for p in filtered_paths], annotations_dict)
        
        # Update label
        total = len(self._all_image_paths)
        shown = len(filtered_paths)
        if self._current_filter == ImageFilter.ALL:
            self._gallery_label.setText(f"All Images ({shown})")
        elif self._current_filter == ImageFilter.ANNOTATED:
            self._gallery_label.setText(f"Annotated ({shown}/{total})")
        else:
            self._gallery_label.setText(f"Unannotated ({shown}/{total})")
    
    @pyqtSlot(str)
    def _on_image_selected(self, image_path: str):
        """Handle image selection."""
        # Store current annotations in memory before switching (only in edit mode)
        if not self._inference_mode and self._current_image_path and self._current_image_path != Path(image_path):
            self._store_current_annotations()
        
        # Load new image
        self._current_image_path = Path(image_path)
        pixmap = QPixmap(image_path)
        
        if pixmap.isNull():
            self.statusMessage.emit(f"Failed to load image: {image_path}", 3000)
            return
        
        # Update info
        self._image_info_label.setText(f"{self._current_image_path.name} ({pixmap.width()}x{pixmap.height()})")
        
        if self._inference_mode:
            # Display in viewer with detections
            self._display_image_with_detections(self._current_image_path)
        else:
            # Update canvas for editing
            self._canvas.load_image(pixmap)
            
            # Prepare canvas annotations
            canvas_annotations = []
            
            # Check if we have pending annotations for this image
            if str(self._current_image_path) in self._pending_annotations:
                # Use pending annotations
                canvas_annotations = self._pending_annotations[str(self._current_image_path)]
            else:
                # Load saved annotations if they exist
                ann_path = get_annotation_path(self._current_image_path)
                if ann_path.exists():
                    annotations = parse_yolo_annotation(ann_path)
                    
                    # Store original YOLO format to preserve precision
                    self._original_yolo_annotations[str(self._current_image_path)] = annotations.copy()
                    
                    # Convert to canvas annotations
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
            
            # Then, add filtered detections if we have them (so user can edit inference results)
            # This allows users to refine the detection results
            filtered_detections = self._filtered_detections.get(str(self._current_image_path), [])
            for det in filtered_detections:
                # Convert detection to canvas annotation
                x, y, w, h = det.x, det.y, det.width, det.height
                
                canvas_ann = Annotation(
                    class_id=det.class_id,
                    rect=QRectF(x, y, w, h)
                )
                canvas_annotations.append(canvas_ann)
            
            self._canvas.set_annotations(canvas_annotations)
            self._update_annotation_count()
        
        # Update UI state to ensure mode toggle button is enabled
        self._update_ui_state()
    
    @pyqtSlot(Annotation)
    def _on_annotation_changed(self, annotation: Annotation):
        """Handle annotation changes."""
        if not self._class_names:
            QMessageBox.warning(
                self, "No Dataset Loaded",
                "Please load a dataset first to get the class names for annotation.\n\n"
                "Use the 'Load Dataset' button to select a dataset with defined classes."
            )
            self._canvas.clear_annotations()
            return
            
        if self._current_image_path:
            self._modified_files.add(str(self._current_image_path))
            self._update_ui_state()
            self._update_annotation_count()
            
            # Update thumbnail with current annotations
            annotations = self._canvas.get_annotations()
            yolo_annotations = []
            
            pixmap = self._canvas._pixmap_item.pixmap()
            for ann in annotations:
                yolo_ann = ann.to_yolo_format(pixmap.width(), pixmap.height())
                yolo_annotations.append(yolo_ann)
            
            self._gallery.update_image_annotations(
                str(self._current_image_path),
                yolo_annotations,
                is_modified=True
            )
    
    @pyqtSlot(int)
    def _on_class_changed(self, index: int):
        """Handle class selection change."""
        if index >= 0:
            class_id = self._class_combo.itemData(index)
            if class_id is not None:
                self._canvas.set_current_class(class_id)
    
    @pyqtSlot(list)
    def _on_annotation_selection_changed(self, selected_annotations: List[Annotation]):
        """Handle annotation selection changes - update class dropdown."""
        if selected_annotations and len(selected_annotations) == 1:
            # Single selection - update dropdown to show the annotation's class
            annotation = selected_annotations[0]
            
            # Find and set the combo box item that matches this class ID
            for i in range(self._class_combo.count()):
                if self._class_combo.itemData(i) == annotation.class_id:
                    # Block signals to avoid triggering class change
                    self._class_combo.blockSignals(True)
                    self._class_combo.setCurrentIndex(i)
                    self._class_combo.blockSignals(False)
                    break
    
    @pyqtSlot()
    def _on_gallery_selection_changed(self):
        """Handle gallery selection change to update UI state."""
        # Update the reject button state based on selection
        if hasattr(self, '_reject_selected_btn') and self._current_folder:
            has_selection = len(self._gallery.get_selected_paths()) > 0
            self._reject_selected_btn.setEnabled(has_selection)
        
        # Update UI state to handle export predictions button
        self._update_ui_state()
    
    def _select_all_images(self):
        """Select all images in the gallery."""
        if hasattr(self, '_gallery'):
            self._gallery.select_all()
            self._update_ui_state()
    
    def _select_none_images(self):
        """Deselect all images in the gallery."""
        if hasattr(self, '_gallery'):
            self._gallery.clear_selection()
            self._update_ui_state()
    
    def _store_current_annotations(self):
        """Store current annotations in memory without saving to disk."""
        if not self._current_image_path:
            return
        
        current_annotations = self._canvas.get_annotations()
        
        # Check if annotations have actually changed
        has_changed = False
        
        # First check if we already have pending annotations for this image
        if str(self._current_image_path) in self._pending_annotations:
            # Already marked as modified, just update
            self._pending_annotations[str(self._current_image_path)] = current_annotations
            has_changed = True
        else:
            # Compare with saved annotations to see if there are changes
            ann_path = get_annotation_path(self._current_image_path)
            if ann_path.exists():
                saved_annotations = parse_yolo_annotation(ann_path)
                # Compare counts first
                if len(current_annotations) != len(saved_annotations):
                    has_changed = True
                else:
                    # Compare each annotation (accounting for potential reordering)
                    pixmap = self._canvas._pixmap_item.pixmap()
                    current_yolo = []
                    for ann in current_annotations:
                        yolo_ann = ann.to_yolo_format(pixmap.width(), pixmap.height())
                        current_yolo.append(yolo_ann)
                    
                    # Sort both lists for comparison
                    current_yolo.sort()
                    saved_annotations.sort()
                    
                    # Compare with tolerance for floating point precision
                    for curr, saved in zip(current_yolo, saved_annotations):
                        if curr[0] != saved[0]:  # Different class
                            has_changed = True
                            break
                        # Check coordinates with small tolerance
                        for i in range(1, 5):
                            if abs(curr[i] - saved[i]) > 1e-6:
                                has_changed = True
                                break
                        if has_changed:
                            break
            else:
                # No saved file, so any annotations mean changes
                has_changed = len(current_annotations) > 0
        
        # Only store and mark as modified if actually changed
        if has_changed:
            self._pending_annotations[str(self._current_image_path)] = current_annotations
            self._modified_files.add(str(self._current_image_path))
            
            # Update gallery to show modified state
            if current_annotations:
                yolo_annotations = []
                pixmap = self._canvas._pixmap_item.pixmap()
                for ann in current_annotations:
                    yolo_ann = ann.to_yolo_format(pixmap.width(), pixmap.height())
                    yolo_annotations.append(yolo_ann)
                
                self._gallery.update_image_annotations(
                    str(self._current_image_path),
                    yolo_annotations,
                    is_modified=True
                )
            else:
                self._gallery.update_image_annotations(
                    str(self._current_image_path),
                    [],
                    is_modified=True
                )
    
    def _save_current_annotations(self):
        """Save annotations for current image to disk."""
        if not self._current_image_path:
            return
        
        # Get annotations from canvas or pending annotations
        if str(self._current_image_path) in self._pending_annotations:
            annotations = self._pending_annotations[str(self._current_image_path)]
        else:
            annotations = self._canvas.get_annotations()
        
        ann_path = get_annotation_path(self._current_image_path)
        
        if annotations:
            # Convert to YOLO format
            yolo_annotations = []
            # Need to get image dimensions
            pixmap = QPixmap(str(self._current_image_path))
            
            for ann in annotations:
                yolo_ann = ann.to_yolo_format(pixmap.width(), pixmap.height())
                yolo_annotations.append(yolo_ann)
            
            # Save to file
            save_yolo_annotation(ann_path, yolo_annotations)
            
            # Mark as annotated
            self._annotated_paths.add(str(self._current_image_path))
        else:
            # No annotations, remove file if it exists
            if ann_path.exists():
                ann_path.unlink()
            self._annotated_paths.discard(str(self._current_image_path))
        
        # Remove from modified set and pending annotations
        self._modified_files.discard(str(self._current_image_path))
        self._pending_annotations.pop(str(self._current_image_path), None)
        
        # Update gallery to show saved state
        if annotations:
            yolo_annotations = []
            pixmap = QPixmap(str(self._current_image_path))
            for ann in annotations:
                yolo_ann = ann.to_yolo_format(pixmap.width(), pixmap.height())
                yolo_annotations.append(yolo_ann)
            
            self._gallery.update_image_annotations(
                str(self._current_image_path),
                yolo_annotations,
                is_modified=False
            )
        else:
            # Update gallery to show no annotations
            self._gallery.update_image_annotations(
                str(self._current_image_path),
                [],
                is_modified=False
            )
        
        # Update progress first (this updates the annotated count)
        self._update_progress()
        
        # Then update class statistics after saving
        if self._class_names:
            self._update_class_statistics()
    
    def _save_all_annotations(self):
        """Save all modified annotations."""
        # Store current annotations first
        if self._current_image_path:
            self._store_current_annotations()
        
        saved_count = 0
        
        # Save all pending annotations
        for image_path_str in list(self._pending_annotations.keys()):
            image_path = Path(image_path_str)
            annotations = self._pending_annotations[image_path_str]
            ann_path = get_annotation_path(image_path)
            
            if annotations:
                # Check if we should use original YOLO format
                # This happens when annotations match the original but were marked as modified
                use_original = False
                if image_path_str in self._original_yolo_annotations:
                    original = self._original_yolo_annotations[image_path_str]
                    if len(annotations) == len(original):
                        # Quick check - if counts match, do detailed comparison
                        pixmap = QPixmap(image_path_str)
                        current_yolo = []
                        for ann in annotations:
                            yolo_ann = ann.to_yolo_format(pixmap.width(), pixmap.height())
                            current_yolo.append(yolo_ann)
                        
                        # Sort for comparison
                        current_yolo.sort()
                        original_sorted = sorted(original)
                        
                        # Check if essentially the same
                        matches = True
                        for curr, orig in zip(current_yolo, original_sorted):
                            if curr[0] != orig[0]:  # Different class
                                matches = False
                                break
                            # Check coordinates with tolerance
                            for i in range(1, 5):
                                if abs(curr[i] - orig[i]) > 1e-4:
                                    matches = False
                                    break
                            if not matches:
                                break
                        
                        use_original = matches
                
                if use_original:
                    # Use original to preserve precision
                    yolo_annotations = self._original_yolo_annotations[image_path_str]
                else:
                    # Convert to YOLO format
                    yolo_annotations = []
                    pixmap = QPixmap(image_path_str)
                    
                    for ann in annotations:
                        yolo_ann = ann.to_yolo_format(pixmap.width(), pixmap.height())
                        yolo_annotations.append(yolo_ann)
                
                # Save to file
                save_yolo_annotation(ann_path, yolo_annotations)
                
                # Mark as annotated
                self._annotated_paths.add(image_path_str)
            else:
                # No annotations, remove file if it exists
                if ann_path.exists():
                    ann_path.unlink()
                self._annotated_paths.discard(image_path_str)
            
            # Update gallery to show saved state
            if annotations:
                self._gallery.update_image_annotations(
                    image_path_str,
                    yolo_annotations,
                    is_modified=False
                )
            else:
                self._gallery.update_image_annotations(
                    image_path_str,
                    [],
                    is_modified=False
                )
            
            saved_count += 1
        
        # Clear pending annotations and modified files
        self._pending_annotations.clear()
        self._modified_files.clear()
        # Don't clear original annotations - they're still valid for unchanged files
        
        self._update_ui_state()
        self._update_progress()
        
        # Update class statistics after saving
        if self._class_names:
            self._update_class_statistics()
        
        self.annotationsSaved.emit(saved_count)
        self.statusMessage.emit(f"Saved {saved_count} annotation files", 3000)
    
    def _clear_current_annotations(self):
        """Clear all annotations from current image."""
        if not self._current_image_path:
            return
        
        reply = QMessageBox.question(
            self, "Clear Annotations",
            "Clear all annotations from current image?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._canvas.clear_annotations()
            self._modified_files.add(str(self._current_image_path))
            self._update_ui_state()
            self._update_annotation_count()
            
            # Update thumbnail to show cleared state
            self._gallery.update_image_annotations(
                str(self._current_image_path),
                [],  # Empty annotations
                is_modified=True
            )
    
    def _export_as_dataset(self):
        """Export annotated images as a YOLO dataset."""
        if not self._annotated_paths:
            QMessageBox.information(self, "No Annotations", 
                                  "No annotated images to export.")
            return
        
        # TODO: Implement dataset export functionality
        QMessageBox.information(self, "Export Dataset", 
                              "Dataset export functionality coming soon!")
    
    def _reject_selected_images(self):
        """Move selected images and their annotations to 'rejected' folder."""
        # Get selected images from gallery
        selected_paths = self._gallery.get_selected_paths()
        
        if not selected_paths:
            QMessageBox.information(self, "No Selection", 
                                  "Please select one or more images to reject.")
            return
        
        # Confirm action
        reply = QMessageBox.question(
            self, "Reject Images",
            f"Move {len(selected_paths)} selected image(s) to 'rejected' folder?\n\n"
            "This will move both the images and their annotation files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Create rejected folder if it doesn't exist
        rejected_folder = self._current_folder / "rejected"
        rejected_folder.mkdir(exist_ok=True)
        
        moved_count = 0
        errors = []
        
        for image_path_str in selected_paths:
            try:
                image_path = Path(image_path_str)
                
                # Skip if file doesn't exist
                if not image_path.exists():
                    continue
                
                # Destination paths
                dest_image = rejected_folder / image_path.name
                
                # Move image file
                shutil.move(str(image_path), str(dest_image))
                moved_count += 1
                
                # Move annotation file if it exists
                ann_path = get_annotation_path(image_path)
                if ann_path.exists():
                    dest_ann = rejected_folder / ann_path.name
                    shutil.move(str(ann_path), str(dest_ann))
                
                # Remove from internal tracking
                self._all_image_paths = [p for p in self._all_image_paths if p != image_path]
                self._annotated_paths.discard(str(image_path))
                
                # If this was the current image, clear the canvas
                if self._current_image_path == image_path:
                    self._canvas.clear_canvas()
                    self._current_image_path = None
                    self._image_info_label.setText("No image selected")
                    self._update_annotation_count()
                
            except Exception as e:
                errors.append(f"{image_path.name}: {str(e)}")
        
        # Reload the gallery to reflect changes
        self._apply_filter()
        
        # Update progress and statistics
        self._update_progress()
        if self._class_names:
            self._update_class_statistics()
        
        # Show result message
        if errors:
            error_msg = "\n".join(errors[:5])  # Show first 5 errors
            if len(errors) > 5:
                error_msg += f"\n... and {len(errors) - 5} more errors"
            
            QMessageBox.warning(
                self, "Rejection Completed with Errors",
                f"Moved {moved_count} image(s) to 'rejected' folder.\n\n"
                f"Errors encountered:\n{error_msg}"
            )
        else:
            self.statusMessage.emit(
                f"Moved {moved_count} image(s) to 'rejected' folder", 3000
            )
    
    def _update_progress(self):
        """Update progress display."""
        if not self._all_image_paths:
            return
        
        total = len(self._all_image_paths)
        annotated = len(self._annotated_paths)
        
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(annotated)
        self._progress_label.setText(f"{annotated}/{total} images annotated")
        
        # Calculate percentage
        percentage = (annotated / total * 100) if total > 0 else 0
        self._progress_bar.setFormat(f"{percentage:.1f}%")
        
        self.progressUpdated.emit(annotated, total)
    
    def _update_annotation_count(self):
        """Update annotation count display."""
        if self._current_image_path:
            # Count saved annotations
            saved_count = 0
            ann_path = get_annotation_path(self._current_image_path)
            if ann_path.exists():
                saved_annotations = parse_yolo_annotation(ann_path)
                saved_count = len(saved_annotations)
            
            # Count detections
            filtered_count = len(self._filtered_detections.get(str(self._current_image_path), []))
            total_det_count = len(self._detections.get(str(self._current_image_path), []))
            
            if self._inference_mode:
                # Show both saved and detection counts
                if total_det_count > 0:
                    self._annotation_count_label.setText(
                        f"📝 Saved: {saved_count} | 🎯 Detections: {filtered_count}/{total_det_count}"
                    )
                else:
                    self._annotation_count_label.setText(f"📝 Saved: {saved_count}")
            else:
                # In edit mode, show total annotations in canvas
                canvas_count = len(self._canvas.get_annotations())
                if filtered_count > 0:
                    self._annotation_count_label.setText(
                        f"✏️ Editing: {canvas_count} (📝 {saved_count} saved + 🎯 {filtered_count} detections)"
                    )
                else:
                    self._annotation_count_label.setText(f"✏️ Editing: {canvas_count}")
        else:
            self._annotation_count_label.setText("Annotations: 0")
    
    def _update_class_statistics(self):
        """Update the class statistics display."""
        if not self._class_names:
            self._class_stats_label.setText("No classes loaded")
            self._total_annotations_label.setText("Total annotations: 0")
            return
        
        # Initialize fresh counters for each class
        class_counts = {}
        detection_counts = {}
        for class_id in self._class_names.keys():
            class_counts[class_id] = 0
            detection_counts[class_id] = 0
            
        total_annotations = 0
        total_detections = 0
        images_with_class = {}
        for class_id in self._class_names.keys():
            images_with_class[class_id] = set()
        
        # Count annotations across all images - fresh count from disk
        for img_path in self._all_image_paths:
            ann_path = get_annotation_path(img_path)
            if ann_path.exists():
                try:
                    annotations = parse_yolo_annotation(ann_path)
                    for ann in annotations:
                        class_id = int(ann[0])  # First element is class_id
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                            total_annotations += 1
                            # Track unique images containing this class
                            images_with_class[class_id].add(str(img_path))
                except Exception as e:
                    print(f"Error parsing annotations for {ann_path}: {e}")
        
        # Count detections if we have inference results
        if self._filtered_detections:
            for img_path, detections in self._filtered_detections.items():
                for det in detections:
                    if det.class_id in detection_counts:
                        detection_counts[det.class_id] += 1
                        total_detections += 1
        
        # Build statistics text
        stats_lines = []
        for class_id, class_name in sorted(self._class_names.items()):
            count = class_counts.get(class_id, 0)
            det_count = detection_counts.get(class_id, 0)
            
            # Format with fixed width for alignment
            if det_count > 0:
                stats_lines.append(f"{class_id:2d}: {class_name:15s} {count:4d} | 🎯{det_count:4d}")
            else:
                stats_lines.append(f"{class_id:2d}: {class_name:15s} {count:4d}")
        
        # Update labels
        self._class_stats_label.setText("\n".join(stats_lines))
        
        if total_detections > 0:
            self._total_annotations_label.setText(
                f"Total annotations: {total_annotations}\n"
                f"Total detections: {total_detections}"
            )
        else:
            self._total_annotations_label.setText(f"Total annotations: {total_annotations}")
    
    def _update_ui_state(self):
        """Update UI element states."""
        has_folder = self._current_folder is not None
        has_classes = bool(self._class_names)
        has_modifications = bool(self._modified_files)
        has_model = ModelCache().get_model() is not None
        
        # Enable/disable controls
        self._save_all_btn.setEnabled(has_modifications and not self._inference_mode)
        self._clear_current_btn.setEnabled(has_folder and self._current_image_path is not None and not self._inference_mode)
        self._export_dataset_btn.setEnabled(has_folder and bool(self._annotated_paths))
        self._reject_selected_btn.setEnabled(has_folder)
        self._class_combo.setEnabled(has_classes and not self._inference_mode)
        
        # Mode toggle button should always be enabled when we have a folder and image
        self._mode_toggle_btn.setEnabled(has_folder and self._current_image_path is not None)
        
        # Inference controls
        self._run_inference_btn.setEnabled(has_folder and has_model)
        
        # Export annotations button - requires selected images with annotations (saved or detections)
        has_selection = False
        if hasattr(self, '_gallery') and self._gallery:
            selected_paths = self._gallery.get_selected_paths()
            if selected_paths:
                # Check if any selected image has annotations
                for path in selected_paths:
                    path_obj = Path(path)
                    has_saved = get_annotation_path(path_obj).exists()
                    has_detections = path in self._filtered_detections and self._filtered_detections[path]
                    if has_saved or has_detections:
                        has_selection = True
                        break
        
        self._export_annotations_btn.setEnabled(has_selection)
        
        # Update modified indicator
        if has_modifications:
            self._modified_label.setText(f"⚠️ {len(self._modified_files)} unsaved changes")
        else:
            self._modified_label.setText("")
        
        # Update filter buttons
        self._filter_all_btn.setEnabled(has_folder)
        self._filter_annotated_btn.setEnabled(has_folder)
        self._filter_unannotated_btn.setEnabled(has_folder)
        
        # Update selection buttons
        self._select_all_btn.setEnabled(has_folder and len(self._all_image_paths) > 0)
        self._select_none_btn.setEnabled(has_folder and len(self._gallery.get_selected_paths()) > 0 if hasattr(self, '_gallery') else False)
    
    def _on_activate(self):
        """Called when mode is activated."""
        self.statusMessage.emit("Folder Browser - Annotate images in any folder", 3000)
        
        # Check if model is loaded and get class names
        model_cache = ModelCache()
        model = model_cache.get_model()
        
        if model and hasattr(model, 'model') and hasattr(model.model, 'names'):
            # Get class names from model
            self._class_names = model.model.names
            self._update_class_combo()
            self._canvas.set_class_names(self._class_names)
            self._viewer.set_class_names(self._class_names)
            
            # Update status
            self._model_status_label.setText(f"✅ Model loaded with {len(self._class_names)} classes")
            self._model_status_label.setStyleSheet("color: #51cf66;")
            
            # Show class names
            class_list = ", ".join([f"{k}: {v}" for k, v in sorted(self._class_names.items())[:5]])
            if len(self._class_names) > 5:
                class_list += f" ... and {len(self._class_names) - 5} more"
            self._class_info_label.setText(f"Classes: {class_list}")
            
            # Update class statistics if we have a folder loaded
            if self._current_folder:
                self._update_class_statistics()
        else:
            # No model loaded
            self._class_names = {}
            self._update_class_combo()
            self._canvas.set_class_names(self._class_names)
            self._viewer.set_class_names(self._class_names)
            
            self._model_status_label.setText("⚠️ No model loaded")
            self._model_status_label.setStyleSheet("color: #ff6b6b;")
            self._class_info_label.setText("Load a model in Model Management tab to enable annotation and inference")
        
        self._update_ui_state()
    
    def _on_deactivate(self) -> Optional[bool]:
        """Called when mode is deactivated."""
        # Stop any running inference
        if self._inference_thread and self._inference_thread.isRunning():
            self._inference_thread.stop()
            self._inference_thread.wait()
        
        # Store current annotations if needed
        if self._current_image_path and self._canvas.get_annotations():
            self._store_current_annotations()
        
        # Check for unsaved changes (both modified files and pending annotations)
        if self._modified_files or self._pending_annotations:
            unsaved_count = len(self._modified_files) + len(self._pending_annotations)
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                f"You have {unsaved_count} unsaved changes.\n\nSave before leaving?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                self._save_all_annotations()
            elif reply == QMessageBox.StandardButton.Cancel:
                return False
        
        return True
    
    def get_mode_name(self) -> str:
        """Get the display name of this mode."""
        return "Folder Browser"
    
    def has_unsaved_changes(self) -> bool:
        """Check if mode has unsaved changes."""
        # Store current annotations if needed
        if self._current_image_path and self._canvas.get_annotations():
            current_annotations = self._canvas.get_annotations()
            # Check if current annotations are different from saved
            ann_path = get_annotation_path(self._current_image_path)
            if ann_path.exists():
                saved_annotations = parse_yolo_annotation(ann_path)
                # Compare counts as a simple check
                if len(current_annotations) != len(saved_annotations):
                    self._store_current_annotations()
            elif current_annotations:
                # Has annotations but no saved file
                self._store_current_annotations()
        
        return bool(self._modified_files) or bool(self._pending_annotations)
    
    def save_changes(self) -> bool:
        """Save any pending changes."""
        try:
            self._save_all_annotations()
            return True
        except Exception as e:
            print(f"Error saving changes: {e}")
            return False
    
    def eventFilter(self, obj, event):
        """Filter events for the gallery."""
        # We no longer auto-save annotations, just pass the event through
        return super().eventFilter(obj, event)
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        # Check if Delete key was pressed
        if event.key() == Qt.Key.Key_Delete:
            # Check if gallery has focus or if we have selected images
            selected_paths = self._gallery.get_selected_paths()
            if selected_paths:
                # Trigger the reject action
                self._reject_selected_images()
                return
        
        super().keyPressEvent(event)
    
    def _check_model_status(self):
        """Check if a model is loaded and update UI accordingly."""
        model_cache = ModelCache()
        model = model_cache.get_model()
        
        if not model:
            self._model_status_label.setText("⚠️ No model loaded")
            self._model_status_label.setStyleSheet("color: #ff6b6b;")
            self._class_info_label.setText("Load a model in Model Management tab to enable annotation and inference")
    
    def _toggle_mode(self):
        """Toggle between edit and inference mode."""
        self._inference_mode = not self._inference_mode
        
        if self._inference_mode:
            # Switch to inference mode
            self._mode_toggle_btn.setText("🌌 Switch to Edit")
            self._mode_toggle_btn.setChecked(False)  # Update button state
            self._editor_stack.setCurrentIndex(1)  # Show viewer
            self._class_combo.setVisible(False)
            self._class_label.setVisible(False)
            self._hint_label.setVisible(False)
            
            # Display current image with detections if any
            if self._current_image_path:
                self._display_image_with_detections(self._current_image_path)
        else:
            # Switch to edit mode
            self._mode_toggle_btn.setText("🎨 Switch to Inference")
            self._mode_toggle_btn.setChecked(True)  # Update button state
            self._editor_stack.setCurrentIndex(0)  # Show canvas
            self._class_combo.setVisible(True)
            self._class_label.setVisible(True)
            self._hint_label.setVisible(True)
            
            # Reload current image in canvas if needed
            if self._current_image_path:
                self._on_image_selected(str(self._current_image_path))
    
    def _run_inference(self):
        """Run inference only on images without existing annotations."""
        model_cache = ModelCache()
        model = model_cache.get_model()
        
        if not model:
            QMessageBox.warning(
                self, "No Model",
                "Please load a model in the Model Management tab first.\n\n"
                "The model is required for running inference."
            )
            return
        
        if not self._all_image_paths:
            return
        
        # Filter to only images without existing annotations
        unannotated_paths = []
        for img_path in self._all_image_paths:
            ann_path = get_annotation_path(img_path)
            if not ann_path.exists():
                unannotated_paths.append(img_path)
        
        if not unannotated_paths:
            QMessageBox.information(
                self, "All Images Annotated",
                "All images in this folder already have annotations.\n"
                "No inference needed."
            )
            return
        
        # Show confirmation with counts
        annotated_count = len(self._all_image_paths) - len(unannotated_paths)
        msg = f"Run inference on {len(unannotated_paths)} unannotated images?\n\n"
        if annotated_count > 0:
            msg += f"Note: {annotated_count} images with existing annotations will be skipped."
        
        reply = QMessageBox.question(
            self, "Run Inference",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Automatically switch to inference mode
        if not self._inference_mode:
            self._toggle_mode()
        
        # Create progress dialog
        progress = QProgressDialog(
            f"Running inference on {len(unannotated_paths)} images...", 
            "Cancel", 0, len(unannotated_paths), self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        
        # Create and start inference thread with only unannotated images
        self._inference_thread = InferenceThread(
            model, unannotated_paths, self._confidence_threshold
        )
        
        # Connect signals
        self._inference_thread.progress.connect(
            lambda current, total: progress.setValue(current)
        )
        self._inference_thread.imageProcessed.connect(self._on_image_processed)
        self._inference_thread.finished.connect(progress.close)
        self._inference_thread.finished.connect(self._on_inference_finished)
        self._inference_thread.error.connect(
            lambda msg: QMessageBox.critical(self, "Inference Error", msg)
        )
        
        # Handle cancel
        progress.canceled.connect(self._inference_thread.stop)
        
        # Start inference
        self._inference_thread.start()
    
    @pyqtSlot(str, list)
    def _on_image_processed(self, image_path: str, detections: List[Detection]):
        """Handle inference results for one image."""
        self._detections[image_path] = detections
        
        # Apply filtering
        filtered = [d for d in detections if d.confidence >= self._confidence_threshold]
        self._filtered_detections[image_path] = filtered
        
        # Update thumbnail with detections
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            yolo_annotations = []
            if filtered:
                for det in filtered:
                    yolo_ann = det.to_yolo_format(pixmap.width(), pixmap.height())
                    yolo_annotations.append(yolo_ann)
            
            self._gallery.update_image_annotations(
                image_path, yolo_annotations, is_modified=False
            )
        
        # If this is the current image and we're in inference mode, update display
        if self._inference_mode and self._current_image_path and str(self._current_image_path) == image_path:
            self._display_image_with_detections(self._current_image_path)
    
    def _on_inference_finished(self):
        """Handle inference completion."""
        total_detections = sum(len(dets) for dets in self._filtered_detections.values())
        images_with_detections = sum(1 for dets in self._filtered_detections.values() if dets)
        
        self.statusMessage.emit(
            f"Inference complete: {images_with_detections} images with {total_detections} detections", 
            5000
        )
        
        # Update class statistics to include detections
        if self._class_names:
            self._update_class_statistics()
        
        # If in inference mode and viewing an image, refresh the display
        if self._inference_mode and self._current_image_path:
            self._display_image_with_detections(self._current_image_path)
        
        # Update UI state to ensure all buttons are properly enabled
        self._update_ui_state()
    
    @pyqtSlot(int)
    def _on_confidence_changed(self, value: int):
        """Handle confidence slider change."""
        self._confidence_threshold = value / 100.0
        self._conf_spinbox.setValue(value)
        self._apply_filtering()
    
    @pyqtSlot(int)
    def _on_confidence_spin_changed(self, value: int):
        """Handle confidence spinbox change."""
        self._confidence_threshold = value / 100.0
        self._conf_slider.setValue(value)
        self._apply_filtering()
    
    @pyqtSlot()
    def _on_show_below_changed(self):
        """Handle show below threshold checkbox change."""
        self._show_below_threshold = self._show_below_cb.isChecked()
        # Redisplay current image if in inference mode
        if self._inference_mode and self._current_image_path:
            self._display_image_with_detections(self._current_image_path)
    
    def _apply_filtering(self):
        """Apply confidence filtering to all detections."""
        # Update filtered detections
        for img_path, detections in self._detections.items():
            filtered = [d for d in detections if d.confidence >= self._confidence_threshold]
            self._filtered_detections[img_path] = filtered
            
            # Update thumbnail
            if filtered:
                pixmap = QPixmap(img_path)
                if not pixmap.isNull():
                    yolo_annotations = []
                    for det in filtered:
                        yolo_ann = det.to_yolo_format(pixmap.width(), pixmap.height())
                        yolo_annotations.append(yolo_ann)
                    
                    self._gallery.update_image_annotations(
                        img_path, yolo_annotations, is_modified=False
                    )
            else:
                # Clear annotations if no detections pass threshold
                self._gallery.update_image_annotations(img_path, [], is_modified=False)
        
        # Update current image display if in inference mode
        if self._inference_mode and self._current_image_path:
            self._display_image_with_detections(self._current_image_path)
        
        # Update statistics
        if self._class_names:
            self._update_class_statistics()
    
    def _display_image_with_detections(self, image_path: Path):
        """Display image with both saved annotations and detections in viewer."""
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            return
        
        # Prepare combined annotations for viewer
        annotations = []
        confidences = []
        below_threshold_flags = []
        
        # First, add saved annotations (these will be shown without confidence)
        ann_path = get_annotation_path(image_path)
        if ann_path.exists():
            saved_annotations = parse_yolo_annotation(ann_path)
            for ann in saved_annotations:
                annotations.append(ann)
                confidences.append(None)  # No confidence for saved annotations
                below_threshold_flags.append(False)
        
        # Then, add detection results if we have them
        all_detections = self._detections.get(str(image_path), [])
        for det in all_detections:
            if det.confidence >= self._confidence_threshold:
                # Above threshold - always show
                yolo_ann = det.to_yolo_format(pixmap.width(), pixmap.height())
                annotations.append(yolo_ann)
                confidences.append(det.confidence)
                below_threshold_flags.append(False)
            elif self._show_below_threshold:
                # Below threshold - show if checkbox is checked
                yolo_ann = det.to_yolo_format(pixmap.width(), pixmap.height())
                annotations.append(yolo_ann)
                confidences.append(det.confidence)
                below_threshold_flags.append(True)
        
        # Load image and annotations
        self._viewer.load_image(pixmap)
        if annotations:
            self._viewer.set_annotations(
                annotations, 
                confidences=confidences,
                below_threshold_flags=below_threshold_flags,
                confidence_threshold=self._confidence_threshold
            )
        else:
            # Clear annotations if no detections
            self._viewer.set_annotations([])
        
        # Update annotation count
        self._update_annotation_count()
    
    def _export_annotations(self):
        """Export annotations for selected images (both saved annotations and inference results)."""
        # Get selected images from gallery
        selected_paths = self._gallery.get_selected_paths()
        
        if not selected_paths:
            QMessageBox.information(
                self, "No Selection",
                "Please select one or more images to export annotations."
            )
            return
        
        # Count images with annotations (saved or detections)
        images_with_annotations = 0
        annotation_sources = []  # Track where annotations come from
        
        for img_path in selected_paths:
            img_path_obj = Path(img_path)
            has_saved = get_annotation_path(img_path_obj).exists()
            has_detections = img_path in self._filtered_detections and self._filtered_detections[img_path]
            
            if has_saved or has_detections:
                images_with_annotations += 1
                if has_saved and has_detections:
                    annotation_sources.append("both")
                elif has_saved:
                    annotation_sources.append("saved")
                else:
                    annotation_sources.append("detection")
        
        if images_with_annotations == 0:
            QMessageBox.information(
                self, "No Annotations",
                "None of the selected images have annotations to export."
            )
            return
        
        # Build informative message
        saved_count = annotation_sources.count("saved") + annotation_sources.count("both")
        detection_count = annotation_sources.count("detection") + annotation_sources.count("both")
        
        msg = f"Export annotations for {images_with_annotations} selected images?\n\n"
        msg += f"• {saved_count} images have saved annotations\n"
        msg += f"• {detection_count} images have detection results\n\n"
        msg += "Note: For images with both, saved annotations take priority."
        
        reply = QMessageBox.question(
            self, "Export Selected Annotations",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Ask for output folder
        output_folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for Annotations",
            str(self._current_folder) if self._current_folder else str(Path.home())
        )
        
        if not output_folder:
            return
        
        output_path = Path(output_folder)
        exported_count = 0
        
        # Export annotations for selected images
        for img_path in selected_paths:
            img_path_obj = Path(img_path)
            existing_ann_path = get_annotation_path(img_path_obj)
            
            # Create output annotation file path
            ann_filename = img_path_obj.stem + ".txt"
            ann_path = output_path / ann_filename
            
            # Priority: saved annotations over detections
            if existing_ann_path.exists():
                # Copy existing annotation file
                shutil.copy2(existing_ann_path, ann_path)
                exported_count += 1
            elif img_path in self._filtered_detections and self._filtered_detections[img_path]:
                # Export detection results
                filtered_dets = self._filtered_detections[img_path]
                pixmap = QPixmap(img_path)
                if not pixmap.isNull():
                    yolo_annotations = []
                    for det in filtered_dets:
                        yolo_ann = det.to_yolo_format(pixmap.width(), pixmap.height())
                        yolo_annotations.append(yolo_ann)
                    
                    # Save annotations
                    save_yolo_annotation(ann_path, yolo_annotations)
                    exported_count += 1
        
        self.statusMessage.emit(f"Exported {exported_count} annotation files", 3000)