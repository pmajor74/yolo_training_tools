"""Auto-annotation mode for automated dataset annotation with quality control."""

from pathlib import Path
from typing import Optional, Dict, List, Set, Any
import shutil
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QFileDialog, QMessageBox, QApplication,
    QListWidgetItem, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QObject, QEvent, QPoint, QTimer
from PyQt6.QtGui import QCursor, QAction

from .base_mode import BaseMode
from ..core import ModelCache, ImageCache, DatasetManager
from ..core.constants import IMAGE_EXTENSIONS
from ..widgets.annotation_canvas import Annotation
from ..utils.yolo_format import load_data_yaml

# Import refactored modules
from .auto_annotation_data_classes import ConfidenceCategory, SessionStats, WorkflowState
from .auto_annotation_dataset_handler import DatasetHandler
from .auto_annotation_training_handler import TrainingHandler
from .auto_annotation_ui_builder import UIBuilder
from .auto_annotation_image_processor import ImageProcessor
from ..utils.tif_converter import TifFormatChecker
from ..widgets.sort_filter_widget import SortOption


class AutoAnnotationMode(BaseMode):
    """
    Mode for automated annotation workflow with quality control.
    
    Features:
    - Three-panel layout (Controls | Thumbnails | Annotation Editor)
    - Multi-tier confidence thresholds  
    - Batch processing with progress monitoring
    - Interactive annotation review and editing
    - Quality control assessments
    - Active learning sample selection
    - Export approved annotations to dataset
    """
    
    # Signals
    sessionStarted = pyqtSignal(str)  # folder path
    sessionProgress = pyqtSignal(int, int)  # current, total
    sessionCompleted = pyqtSignal()
    annotationsExported = pyqtSignal(str)  # export path
    datasetSplitCompleted = pyqtSignal(str)  # output path
    trainingStarted = pyqtSignal(str)  # config path
    trainingProgress = pyqtSignal(int, int, dict)  # epoch, total_epochs, metrics
    trainingCompleted = pyqtSignal(str)  # model path
    workflowStageChanged = pyqtSignal(str)  # new stage
    
    def __init__(self, parent=None):
        # Initialize attributes before super().__init__()
        self._current_folder: Optional[Path] = None
        self._session_stats = SessionStats()
        self._workflow_state = WorkflowState()
        self._workflow_enabled = True
        self._is_processing = False
        self._current_image_path: Optional[str] = None
        self._current_filter: Optional[ConfidenceCategory] = None
        self._dataset_yaml_path: Optional[Path] = None
        self._dataset_class_names: Dict[int, str] = {}
        self._detected_categories: Dict[str, Set[int]] = {}
        self._selected_category_filters: Set[int] = set()
        self._last_exported_paths: List[str] = []
        self._pending_iteration_message = False
        self._gallery_expanded = False
        self._saved_splitter_sizes = None
        
        # Initialize handlers first (before super().__init__ which calls _setup_ui)
        # Note: We can't pass 'self' to DatasetHandler yet as super().__init__ hasn't been called
        self._dataset_handler = DatasetHandler()
        self._training_handler = TrainingHandler()
        self._image_processor = ImageProcessor()
        self._ui_builder = UIBuilder()
        
        super().__init__(parent)
        
        # Now set the parent for dataset handler after initialization
        self._dataset_handler._parent = self
        
        self._setup_handler_connections()
        
    def _setup_ui(self):
        """Set up the UI."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create three-panel layout
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Controls panel (left)
        self._controls_panel = self._ui_builder.create_controls_panel()
        self._splitter.addWidget(self._controls_panel)
        
        # Gallery panel (center)
        self._gallery_panel = self._ui_builder.create_gallery_panel()
        self._splitter.addWidget(self._gallery_panel)
        
        # Editor panel (right)
        self._editor_panel = self._ui_builder.create_editor_panel()
        self._splitter.addWidget(self._editor_panel)
        
        # Set initial sizes (20%, 40%, 40%)
        self._splitter.setSizes([300, 600, 600])
        
        # Add splitter to main layout
        main_layout.addWidget(self._splitter)
        
        # Store references to UI elements
        self._store_ui_references()
        
        # Connect UI signals
        self._connect_ui_signals()
        
        # Set up event filters
        self._gallery.installEventFilter(self)
        self._gallery.list_view.installEventFilter(self)
        
        # Initialize UI state
        self._update_ui_state()
        self._update_requirements_status()
        self._update_threshold_labels()
        self._populate_class_combo()
        
    def _store_ui_references(self):
        """Store references to UI elements from the builder."""
        # Controls
        self._select_folder_btn = self._ui_builder.select_folder_btn
        self._folder_label = self._ui_builder.folder_label
        self._start_btn = self._ui_builder.start_btn
        self._stop_btn = self._ui_builder.stop_btn
        self._include_annotated_checkbox = self._ui_builder.include_annotated_checkbox
        
        # Workflow
        self._workflow_checkbox = self._ui_builder.workflow_checkbox
        self._iteration_label = self._ui_builder.iteration_label
        self._stage_label = self._ui_builder.stage_label
        self._model_history_combo = self._ui_builder.model_history_combo
        
        # Progress
        self._progress_bar = self._ui_builder.progress_bar
        self._progress_label = self._ui_builder.progress_label
        self._stats_label = self._ui_builder.stats_label
        
        # Requirements
        self._model_status_label = self._ui_builder.model_status_label
        
        # Thresholds
        self._high_conf_slider = self._ui_builder.high_conf_slider
        self._high_conf_label = self._ui_builder.high_conf_label
        self._med_conf_slider = self._ui_builder.med_conf_slider
        self._med_conf_label = self._ui_builder.med_conf_label
        self._refresh_thresholds_btn = self._ui_builder.refresh_thresholds_btn
        
        # Augmentation
        self._enable_augmentation_checkbox = self._ui_builder.enable_augmentation_checkbox
        self._augmentation_settings = self._ui_builder.augmentation_settings
        self._augmentation_scroll = self._ui_builder.augmentation_scroll
        
        # Dataset
        self._dataset_btn = self._ui_builder.dataset_btn
        self._dataset_info_label = self._ui_builder.dataset_info_label
        self._train_split_spin = self._ui_builder.train_split_spin
        self._val_split_spin = self._ui_builder.val_split_spin
        self._test_split_spin = self._ui_builder.test_split_spin
        
        # Training
        self._training_model_combo = self._ui_builder.training_model_combo
        self._epochs_spin = self._ui_builder.epochs_spin
        self._batch_spin = self._ui_builder.batch_spin
        self._image_size_combo = self._ui_builder.image_size_combo
        self._lr_spin = self._ui_builder.lr_spin
        self._start_training_btn = self._ui_builder.start_training_btn
        self._stop_training_btn = self._ui_builder.stop_training_btn
        self._training_progress_bar = self._ui_builder.training_progress_bar
        self._training_status_label = self._ui_builder.training_status_label
        self._training_console = self._ui_builder.training_console
        
        # Actions
        self._approve_selected_btn = self._ui_builder.approve_selected_btn
        self._reject_selected_btn = self._ui_builder.reject_selected_btn
        self._export_btn = self._ui_builder.export_btn
        self._quality_btn = self._ui_builder.quality_btn
        self._active_learning_btn = self._ui_builder.active_learning_btn
        self._move_to_rejected_btn = self._ui_builder.move_to_rejected_btn
        
        # Gallery
        self._gallery = self._ui_builder.gallery
        self._gallery_label = self._ui_builder.gallery_label
        self._center_stack = self._ui_builder.center_stack
        self._training_charts = self._ui_builder.training_charts
        self._select_all_btn = self._ui_builder.select_all_btn
        self._select_none_btn = self._ui_builder.select_none_btn
        self._expand_gallery_btn = self._ui_builder.expand_gallery_btn
        self._category_filter_combo = self._ui_builder.category_filter_combo
        self._category_list_widget = self._ui_builder.category_list_widget
        self._filter_review_btn = self._ui_builder.filter_review_btn
        self._filter_approved_btn = self._ui_builder.filter_approved_btn
        self._filter_rejected_btn = self._ui_builder.filter_rejected_btn
        self._filter_no_detections_btn = self._ui_builder.filter_no_detections_btn
        
        # Sort/Filter widget
        self._sort_filter_widget = self._ui_builder.sort_filter_widget
        
        # Editor
        self._canvas = self._ui_builder.annotation_canvas
        self._annotation_count_label = self._ui_builder.annotation_count_label
        self._class_combo = self._ui_builder.class_combo
        
        # Add info label for current image
        self._info_label = self._gallery_label  # Reuse gallery label for now
        
    def _connect_ui_signals(self):
        """Connect UI signals to handlers."""
        # UI Builder signals
        self._ui_builder.selectFolder.connect(self._on_select_folder)
        self._ui_builder.startClicked.connect(self._on_start_clicked)
        self._ui_builder.stopClicked.connect(self._on_stop_clicked)
        self._ui_builder.thresholdChanged.connect(self._update_threshold_labels)
        self._ui_builder.refreshThresholds.connect(self._refresh_thresholds)
        self._ui_builder.augmentationToggled.connect(self._on_augmentation_toggled)
        self._ui_builder.workflowToggled.connect(self._toggle_workflow)
        self._ui_builder.modelHistoryChanged.connect(self._load_historical_model)
        self._ui_builder.datasetManage.connect(self._manage_dataset)
        self._ui_builder.splitPercentageChanged.connect(self._update_split_percentages)
        self._ui_builder.startTraining.connect(self._start_training)
        self._ui_builder.stopTraining.connect(self._stop_training)
        self._ui_builder.approveSelected.connect(self._approve_selected)
        self._ui_builder.rejectSelected.connect(self._reject_selected)
        self._ui_builder.exportAnnotations.connect(self._export_annotations)
        self._ui_builder.qualityAssessment.connect(self._run_quality_assessment)
        self._ui_builder.moveToRejected.connect(self._move_selected_to_rejected)
        self._ui_builder.selectAllThumbnails.connect(self._select_all_thumbnails)
        self._ui_builder.selectNoneThumbnails.connect(self._select_none_thumbnails)
        self._ui_builder.filterByCategory.connect(self._filter_by_category)
        self._ui_builder.categoryFilterClicked.connect(self._on_category_filter_clicked)
        self._ui_builder.categoryFilterChanged.connect(self._on_category_filter_changed)
        self._ui_builder.classChanged.connect(self._on_class_changed)
        self._ui_builder.expandGallery.connect(self._toggle_gallery_expansion)
        
        # Sort/Filter signals
        self._sort_filter_widget.sortingChanged.connect(self._on_sorting_changed)
        self._sort_filter_widget.filteringChanged.connect(self._on_sorting_changed)
        
        # Gallery signals
        self._gallery.imageSelected.connect(self._on_image_selected)
        self._gallery.imageDoubleClicked.connect(self._on_image_double_clicked)
        self._gallery.selectionChanged.connect(self._on_gallery_selection_changed)
        
        # Canvas signals
        self._canvas.annotationAdded.connect(self._on_annotation_added)
        self._canvas.annotationModified.connect(self._on_annotation_modified)
        self._canvas.annotationDeleted.connect(self._on_annotation_deleted)
        self._canvas.selectionChanged.connect(self._on_annotation_selection_changed)
        
    def _setup_handler_connections(self):
        """Set up connections between handlers."""
        # Dataset handler signals
        self._dataset_handler.datasetLoaded.connect(self._on_dataset_loaded)
        self._dataset_handler.datasetCreated.connect(self._on_dataset_created)
        self._dataset_handler.splitCompleted.connect(self._on_split_completed)
        self._dataset_handler.error.connect(self._on_handler_error)
        
        # Training handler signals
        self._training_handler.trainingStarted.connect(self._on_training_started)
        self._training_handler.trainingProgress.connect(self._on_training_progress)
        self._training_handler.trainingCompleted.connect(self._on_training_completed)
        self._training_handler.trainingError.connect(self._on_training_error)
        self._training_handler.logMessage.connect(self._on_training_output)
        
        # Image processor signals
        self._image_processor.processingStarted.connect(self._on_processing_started)
        self._image_processor.processingProgress.connect(self._on_inference_progress)
        self._image_processor.processingCompleted.connect(self._on_inference_completed)
        self._image_processor.statsUpdated.connect(self._on_stats_updated)
        self._image_processor.proposalsUpdated.connect(self._on_proposals_updated)
        
    def _update_threshold_labels(self):
        """Update threshold value labels."""
        high_value = self._high_conf_slider.value() / 100
        med_value = self._med_conf_slider.value() / 100
        self._ui_builder.update_threshold_labels(high_value, med_value)
        
    def _refresh_thresholds(self):
        """Refresh thresholds and re-categorize images."""
        if not self._image_processor.annotation_manager:
            return
            
        # Update thresholds in the session
        high_threshold = self._high_conf_slider.value() / 100
        med_threshold = self._med_conf_slider.value() / 100
        
        session = self._image_processor.annotation_manager.current_session
        if session:
            session.high_threshold = high_threshold
            session.medium_threshold = med_threshold
            
            # Re-categorize all images
            self._image_processor.categorize_images()
            
            # Update gallery
            self._reload_gallery_with_filter()
            
            # Update detected categories filter
            self._populate_category_filter()
                                  
    def _load_class_names_from_model(self):
        """Load class names from the currently loaded model."""
        model_cache = ModelCache()
        model = model_cache.get_model()
        if model:
            # Try to get class names from model
            try:
                # YOLO models store names in model.model.names
                if hasattr(model, 'model') and hasattr(model.model, 'names'):
                    self._dataset_class_names = model.model.names
                    self.update_class_names(self._dataset_class_names)
                elif hasattr(model, 'names'):
                    # Fallback to direct names attribute
                    self._dataset_class_names = model.names
                    self.update_class_names(self._dataset_class_names)
            except Exception as e:
                print(f"Could not load class names from model: {e}")
                
    def _on_select_folder(self):
        """Handle folder selection."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Image Folder",
            str(self._current_folder) if self._current_folder else ""
        )
        
        if folder_path:
            self.set_folder(folder_path)
            
    def _on_start_clicked(self):
        """Handle start button click."""
        if not self._current_folder:
            QMessageBox.warning(self, "No Folder", "Please select a folder first.")
            return
            
        # Check model is loaded
        model_cache = ModelCache()
        if not model_cache.is_loaded():
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return
            
        # Check if workflow is enabled but no dataset is loaded
        if self._workflow_enabled and not self._dataset_yaml_path:
            QMessageBox.warning(self, "Dataset Required", 
                              "Automated workflow requires a dataset to be loaded.\n\n"
                              "Please use 'Load Dataset' to load an existing dataset first.")
            return
            
        # Start processing
        self._start_processing()
        
    def _on_stop_clicked(self):
        """Handle stop button click."""
        # Stop any running process
        if self._training_handler.is_training:
            self._stop_training()
        elif self._is_processing:
            self._stop_processing()
        
    def _on_augmentation_toggled(self, checked: bool):
        """Handle augmentation toggle."""
        self._ui_builder.show_augmentation_settings(checked)
        
    def _on_image_selected(self, image_path: str):
        """Handle image selection from gallery."""
        self._load_image_for_editing(image_path)
    
    def _on_image_double_clicked(self, image_path: str):
        """Handle image double-click from gallery."""
        # Load the image
        self._load_image_for_editing(image_path)
        
        # If gallery is expanded, retract it to show the image
        if self._gallery_expanded:
            self._toggle_gallery_expansion()
        
    def _on_gallery_selection_changed(self):
        """Handle gallery selection change."""
        selected = self._gallery.get_selected_paths()
        
        # Update button states based on selection
        has_selection = len(selected) > 0
        self._approve_selected_btn.setEnabled(has_selection)
        self._reject_selected_btn.setEnabled(has_selection)
        self._export_btn.setEnabled(has_selection)
        self._move_to_rejected_btn.setEnabled(has_selection)
        
        # Update gallery label
        if has_selection:
            total = self._gallery._model.rowCount()
            self._gallery_label.setText(f"Images ({len(selected)}/{total} selected)")
        else:
            total = self._gallery._model.rowCount()
            self._gallery_label.setText(f"Images ({total} total)")
            
    def _start_processing(self):
        """Start auto-annotation processing."""
        if not self._current_folder:
            return
        
        # Check and convert TIF files if needed
        if not TifFormatChecker.check_and_convert_if_needed(self._current_folder, self):
            # User cancelled conversion or conversion failed
            QMessageBox.information(self, "Processing Cancelled", 
                                  "Auto-annotation cancelled. TIF files must be in RGB format for YOLO processing.")
            return
            
        # Reset filters and categories for new session
        self._detected_categories.clear()
        self._selected_category_filters.clear()
        self._current_filter = ConfidenceCategory.REQUIRES_REVIEW  # Set default filter
        self._filter_approved_btn.setChecked(False)
        self._filter_review_btn.setChecked(True)
        self._filter_rejected_btn.setChecked(False)
        self._filter_no_detections_btn.setChecked(False)
        
        # Populate category filter with all dataset classes
        if self._dataset_class_names:
            self._populate_category_filter()
        else:
            # Clear category filter if no dataset loaded
            while self._category_list_widget.count() > 3:
                self._category_list_widget.takeItem(3)
            self._category_filter_combo.setCurrentText("All categories")
        
        # Get settings
        include_annotated = self._include_annotated_checkbox.isChecked()
        high_threshold, med_threshold = self._ui_builder.get_threshold_values()
        augment = self._enable_augmentation_checkbox.isChecked()
        augment_settings = self._ui_builder.get_augmentation_settings() if augment else None
        
        # Start processing
        self._image_processor.start_processing(
            self._current_folder,
            include_annotated,
            high_threshold,
            med_threshold,
            augment,
            augment_settings
        )
        
        # Load images to gallery
        self._gallery.load_images(self._image_processor.all_processed_images)
        
        # Update UI state
        self._is_processing = True
        self._update_ui_state()
        
    def _stop_processing(self):
        """Stop auto-annotation processing."""
        self._image_processor.stop_processing()
        self._is_processing = False
        self._update_ui_state()
        
    def _filter_by_category(self, category: Optional[ConfidenceCategory]):
        """Filter gallery by confidence category."""
        # Update button states
        self._filter_approved_btn.setChecked(category == ConfidenceCategory.AUTO_APPROVED)
        self._filter_review_btn.setChecked(category == ConfidenceCategory.REQUIRES_REVIEW)
        self._filter_rejected_btn.setChecked(category == ConfidenceCategory.REJECTED)
        self._filter_no_detections_btn.setChecked(category == ConfidenceCategory.NO_DETECTIONS)
        
        # Store current filter
        self._current_filter = category
        
        # Reload gallery with filtered images
        self._reload_gallery_with_filter()
        
        # Select and load the first image in the filtered view
        self._select_image_at_position(0)
    
    def _on_sorting_changed(self):
        """Handle sorting/filtering changes."""
        # Remember current selection
        current_selection = self._gallery.get_current_selected_path()
        
        # Reload gallery with current filter and apply new sorting
        self._reload_gallery_with_filter()
        
        # Handle selection after sorting/filtering
        self._handle_selection_after_change(current_selection)
    
    def _handle_selection_after_change(self, previous_selection: Optional[str]):
        """Handle thumbnail selection after sorting/filtering changes.
        
        Args:
            previous_selection: Path of previously selected image, or None
        """
        if previous_selection:
            # Try to maintain current selection
            if not self._gallery.select_and_scroll_to_path(previous_selection):
                # Previous selection not in filtered results, select first
                self._gallery.select_first_item()
        else:
            # No previous selection, select first item
            self._gallery.select_first_item()
        
    def _approve_selected(self):
        """Approve annotations for selected images."""
        selected = self._gallery.get_selected_paths()
        if not selected:
            QMessageBox.information(self, "No Selection", "Please select images to approve.")
            return
            
        # Store current index position before reloading
        selected_indices = self._gallery.list_view.selectedIndexes()
        current_row = selected_indices[0].row() if selected_indices else 0
        
        # Approve proposals
        self._image_processor.approve_proposals(selected)
        
        # Update UI
        self._reload_gallery_with_filter()
        
        # Select and load the next available image
        self._select_image_at_position(current_row)
                              
    def _reject_selected(self):
        """Reject annotations for selected images."""
        selected = self._gallery.get_selected_paths()
        if not selected:
            QMessageBox.information(self, "No Selection", "Please select images to reject.")
            return
            
        # Store current index position before reloading
        selected_indices = self._gallery.list_view.selectedIndexes()
        current_row = selected_indices[0].row() if selected_indices else 0
        
        # Reject proposals
        self._image_processor.reject_proposals(selected)
        
        # Update UI
        self._reload_gallery_with_filter()
        
        # Select and load the next available image
        self._select_image_at_position(current_row)
                              
    def _approve_current(self):
        """Approve current image annotations."""
        if self._current_image_path:
            # Find current image position in gallery
            current_row = self._find_image_row_in_gallery(self._current_image_path)
            
            self._image_processor.approve_proposals([self._current_image_path])
            self._reload_gallery_with_filter()
            
            # Select image at same position
            self._select_image_at_position(current_row if current_row >= 0 else 0)
            
    def _reject_current(self):
        """Reject current image annotations."""
        if self._current_image_path:
            # Find current image position in gallery
            current_row = self._find_image_row_in_gallery(self._current_image_path)
            
            self._image_processor.reject_proposals([self._current_image_path])
            self._reload_gallery_with_filter()
            
            # Select image at same position
            self._select_image_at_position(current_row if current_row >= 0 else 0)
            
    def _select_next_image(self):
        """Select the next available image in the gallery."""
        # Get current selection in gallery
        selected_indices = self._gallery.list_view.selectedIndexes()
        if selected_indices:
            current_index = selected_indices[0]
            # Try to select next item
            next_row = current_index.row() + 1
            if next_row < self._gallery._model.rowCount():
                next_index = self._gallery._model.index(next_row, 0)
                self._gallery.list_view.setCurrentIndex(next_index)
                self._gallery.list_view.selectionModel().select(next_index, 
                    self._gallery.list_view.selectionModel().SelectionFlag.ClearAndSelect)
                # Load the next image
                item = self._gallery._model.get_item(next_row)
                if item and item.image_path:
                    self._load_image_for_editing(item.image_path)
            else:
                # No more images, clear current image
                self._current_image_path = None
                self._canvas.clear_canvas()
                self._info_label.setText("No more images to review")
                self._annotation_count_label.setText("Annotations: 0")
                
    def _find_image_row_in_gallery(self, image_path: str) -> int:
        """Find the row index of an image in the gallery."""
        for row in range(self._gallery._model.rowCount()):
            item = self._gallery._model.get_item(row)
            if item and item.image_path == image_path:
                return row
        return -1
        
    def _select_image_at_position(self, position: int):
        """Select and load image at the given position in the gallery."""
        # Use QTimer to ensure gallery is fully updated before selecting
        def do_selection():
            total_items = self._gallery._model.rowCount()
            
            if total_items == 0:
                # No images left in gallery
                self._current_image_path = None
                self._canvas.clear_canvas()
                self._info_label.setText("No more images to review")
                self._annotation_count_label.setText("Annotations: 0")
                return
                
            # Adjust position if it's beyond the current items
            adjusted_position = position
            if adjusted_position >= total_items:
                adjusted_position = total_items - 1
            elif adjusted_position < 0:
                adjusted_position = 0
                
            # Select the item at the position
            index = self._gallery._model.index(adjusted_position, 0)
            self._gallery.list_view.setCurrentIndex(index)
            self._gallery.list_view.selectionModel().select(index, 
                self._gallery.list_view.selectionModel().SelectionFlag.ClearAndSelect)
                
            # Ensure the selection is visible
            self._gallery.list_view.scrollTo(index)
                
            # Load the image - get the actual item data
            item = self._gallery._model.get_item(adjusted_position)
            if item and item.image_path:
                # Load the image directly
                self._load_image_for_editing(item.image_path)
                    
        # Execute selection with a small delay to ensure gallery is ready
        QTimer.singleShot(10, do_selection)
            
    def _export_annotations(self):
        """Export approved annotations."""
        selected = self._gallery.get_selected_paths()
        if not selected:
            QMessageBox.information(self, "No Selection", 
                                  "Please select images to export annotations.")
            return
            
        # For workflow mode, export to current folder
        if self._workflow_enabled:
            output_folder = str(self._current_folder)
        else:
            # Ask for output folder
            output_folder = QFileDialog.getExistingDirectory(
                self, "Select Output Folder", 
                str(self._current_folder) if self._current_folder else ""
            )
            if not output_folder:
                return
                
        # Save current annotations
        if self._current_image_path and self._current_image_path in selected:
            self._save_current_annotations()
            
        # Export annotations
        exported_count = self._image_processor.export_annotations(output_folder, selected)
        
        # Track exported paths
        self._last_exported_paths = selected
        
        # Update stats
        self._session_stats.exported += exported_count
        self._update_stats_display()
        
        # Show confirmation only if workflow is NOT enabled
        if not self._workflow_enabled:
            QMessageBox.information(self, "Export Complete", 
                                  f"Exported annotations for {exported_count} image(s) to:\n{output_folder}")
                              
        self.annotationsExported.emit(output_folder)
        
        # If workflow enabled, proceed to dataset split
        if self._workflow_enabled and self._dataset_yaml_path:
            self._execute_dataset_split()
            
    def _run_quality_assessment(self):
        """Run quality assessment on annotations."""
        if not self._current_folder:
            QMessageBox.warning(self, "No Session", 
                              "Please start an annotation session first.")
            return
            
        # Import here to avoid circular imports
        from ..dialogs import QualityControlDialog
        from ..utils.auto_annotation_manager import AutoAnnotationSession
        
        # Create session object for the dialog
        session = AutoAnnotationSession(
            session_id=datetime.now().isoformat(),
            folder_path=str(self._current_folder),
            created_at=datetime.now(),
            total_images=self._session_stats.total_images,
            processed_images=self._session_stats.processed_images
        )
        
        # Get annotations from the image processor
        if self._image_processor.annotation_manager and self._image_processor.annotation_manager.current_session:
            # Get proposals from the annotation manager's current session
            manager_session = self._image_processor.annotation_manager.current_session
            
            # Copy proposals with their approval status
            for img_path, proposals in manager_session.proposals.items():
                session.proposals[img_path] = proposals
                
            # Also check for any modified annotations in the image processor
            for img_path in self._image_processor.all_processed_images:
                annotations = self._image_processor.get_annotations_for_image(img_path)
                if annotations and img_path not in session.proposals:
                    # Convert canvas annotations to proposals
                    proposals = []
                    for ann in annotations:
                        from ..utils.auto_annotation_manager import AnnotationProposal
                        proposal = AnnotationProposal(
                            class_id=ann.class_id,
                            bbox=(ann.rect.x(), ann.rect.y(), ann.rect.width(), ann.rect.height()),
                            confidence=ann.confidence,
                            image_path=img_path,
                            is_approved=getattr(ann, 'is_approved', False),
                            is_modified=getattr(ann, 'is_modified', False)
                        )
                        proposals.append(proposal)
                    session.proposals[img_path] = proposals
        
        # Show dialog
        dialog = QualityControlDialog(session, self)
        # Pass class names if available
        if self._dataset_class_names:
            dialog.class_names = self._dataset_class_names
        dialog.exec()
                              
    def _load_image_for_editing(self, image_path: str):
        """Load image into annotation editor."""
        # Save current annotations before switching
        if self._current_image_path and self._current_image_path != image_path:
            self._save_current_annotations()
            
        self._current_image_path = image_path
        
        # Load image to canvas
        pixmap = self._image_processor.load_image_for_editing(image_path)
        if pixmap and not pixmap.isNull():
            self._canvas.load_image(pixmap)
        else:
            print(f"Warning: Failed to load image {image_path}")
            return
        
        # Ensure canvas has the current class names
        if self._dataset_class_names:
            self._canvas.set_class_names(self._dataset_class_names)
            
        # Get annotations for this image
        annotations = self._image_processor.get_annotations_for_image(image_path)
        self._canvas.set_annotations(annotations)
        
        # Update info
        self._info_label.setText(f"Editing: {Path(image_path).name}")
        self._update_annotation_count()
        
        # Force canvas update
        self._canvas.update()
        
    def _on_annotation_added(self, annotation: Annotation):
        """Handle annotation added."""
        if self._current_image_path:
            self._save_current_annotations()
            self._update_stats_display()
            self._update_annotation_count()
            
    def _on_annotation_modified(self, annotation: Annotation):
        """Handle annotation modified."""
        if self._current_image_path:
            self._save_current_annotations()
            self._update_stats_display()
            self._update_annotation_count()
            
    def _on_annotation_deleted(self, annotation: Annotation):
        """Handle annotation deleted."""
        if self._current_image_path:
            self._save_current_annotations()
        self._update_stats_display()
        self._update_annotation_count()
        
    @pyqtSlot(int)
    def _on_class_changed(self, index: int):
        """Handle class selection change."""
        if index >= 0:
            # Get the actual class ID from the combo box data
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
                    
    def _populate_class_combo(self):
        """Populate the class combo box."""
        self._class_combo.clear()
        
        # Only use dataset class names - don't fall back to model names
        if self._dataset_class_names:
            # Use dataset class names
            for class_id, class_name in sorted(self._dataset_class_names.items()):
                display_text = f"{class_name} ({class_id})"
                self._class_combo.addItem(display_text, class_id)
            # Also update canvas with these names
            self._canvas.set_class_names(self._dataset_class_names)
            
            # Set current class on canvas
            if self._class_combo.count() > 0:
                first_class_id = self._class_combo.itemData(0)
                if first_class_id is not None:
                    self._canvas.set_current_class(first_class_id)
        else:
            # No classes loaded - show placeholder
            self._class_combo.addItem("(Load model first)", None)
            self._class_combo.setEnabled(False)
            
    def _update_annotation_count(self):
        """Update the annotation count label."""
        annotations = self._canvas.get_annotations()
        count = len(annotations)
        self._annotation_count_label.setText(f"Annotations: {count}")
        
    def _save_current_annotations(self):
        """Save current annotations and update thumbnail."""
        if not self._current_image_path:
            return
            
        # Get current annotations from canvas
        annotations = self._canvas.get_annotations()
        
        # Save annotations
        self._image_processor.save_annotations_for_image(self._current_image_path, annotations)
        
        # Update detected categories for this image based on current annotations
        if annotations:
            detected_classes = set(ann.class_id for ann in annotations)
            self._detected_categories[self._current_image_path] = detected_classes
        else:
            # No annotations, remove from detected categories
            self._detected_categories.pop(self._current_image_path, None)
            
        # Convert to thumbnail format and update gallery
        thumbnail_annotations = []
        dimensions = self._image_processor.get_image_dimensions(self._current_image_path)
        if dimensions:
            img_width, img_height = dimensions
            
            for ann in annotations:
                # Convert from QRectF to normalized coordinates
                rect = ann.rect
                x_center = (rect.x() + rect.width() / 2) / img_width
                y_center = (rect.y() + rect.height() / 2) / img_height
                w_norm = rect.width() / img_width
                h_norm = rect.height() / img_height
                
                thumbnail_annotations.append((
                    ann.class_id,
                    x_center,
                    y_center,
                    w_norm,
                    h_norm,
                    ann.confidence if ann.confidence else 1.0
                ))
                
        # Update thumbnail gallery
        self._gallery.update_image_annotations(
            self._current_image_path, 
            thumbnail_annotations, 
            is_modified=True
        )
        
        # Update category filter counts
        self._update_category_filter_counts()
        
    @pyqtSlot(str)
    def _on_processing_started(self, folder_path: str):
        """Handle processing started."""
        self.sessionStarted.emit(folder_path)
        
    @pyqtSlot(int, int)
    def _on_inference_progress(self, current: int, total: int):
        """Handle inference progress."""
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._progress_label.setText(f"Processing {current}/{total}")
        self.sessionProgress.emit(current, total)
        
    @pyqtSlot()
    def _on_inference_completed(self):
        """Handle inference completion."""
        self._is_processing = False
        self._update_ui_state()
        self._progress_label.setText("Processing complete")
        
        # Restore cursor
        QApplication.restoreOverrideCursor()
        
        # Update gallery with all annotations
        if self._image_processor.annotation_manager and self._image_processor.annotation_manager.current_session:
            # Update gallery thumbnails with proposals
            for image_path in self._image_processor.all_processed_images:
                proposals = self._image_processor.annotation_manager.current_session.proposals.get(image_path, [])
                
                # Convert proposals to thumbnail format
                thumbnail_annotations = []
                dimensions = self._image_processor.get_image_dimensions(image_path)
                if dimensions and proposals:
                    img_width, img_height = dimensions
                    
                    for prop in proposals:
                        # Normalize bbox
                        x_center = (prop.bbox[0] + prop.bbox[2] / 2) / img_width
                        y_center = (prop.bbox[1] + prop.bbox[3] / 2) / img_height
                        w_norm = prop.bbox[2] / img_width
                        h_norm = prop.bbox[3] / img_height
                        
                        thumbnail_annotations.append((
                            prop.class_id,
                            x_center,
                            y_center,
                            w_norm,
                            h_norm,
                            prop.confidence
                        ))
                        
                # Update thumbnail
                self._gallery.update_image_annotations(image_path, thumbnail_annotations)
                
                # Update detected categories based on proposals
                if proposals:
                    detected_classes = set(prop.class_id for prop in proposals)
                    self._detected_categories[image_path] = detected_classes
                    
        # Populate category filter with detected classes
        self._populate_category_filter()
        
        # Show pending iteration message if workflow enabled
        if self._pending_iteration_message:
            self._pending_iteration_message = False
            QMessageBox.information(self, "Ready to Continue", 
                                  f"Completed iteration {self._workflow_state.iteration}.\n"
                                  "Model has been updated and inference completed.\n"
                                  "You can now review and select images for the next iteration.")
        
        # Ensure categorization is complete before applying filter
        # Force categorization if it hasn't happened yet
        if not any(self._image_processor.annotation_categories.values()):
            self._image_processor.categorize_images()
        
        # Apply the default filter (requires review)
        # Set the filter even if there are no images in that category
        self._filter_by_category(ConfidenceCategory.REQUIRES_REVIEW)
        
        # Enable refresh thresholds button
        self._refresh_thresholds_btn.setEnabled(True)
        
        self.sessionCompleted.emit()
        
        # Show dialog to notify user
        if self._workflow_enabled:
            QMessageBox.information(self, "Auto-Annotation Complete", 
                                  "Auto-annotation is complete. Please review the proposed annotations.\n\n"
                                  "The 'Review' category has been selected for you.")
        else:
            # Even without workflow, show a simple notification
            QMessageBox.information(self, "Auto-Annotation Complete", 
                                  "Auto-annotation is complete. The 'Review' category has been selected.")
            
        # Ensure the Review filter is properly applied after the dialog
        # This is important in case the dialog or any other action reset the filter
        self._filter_by_category(ConfidenceCategory.REQUIRES_REVIEW)
        
    @pyqtSlot(SessionStats)
    def _on_stats_updated(self, stats: SessionStats):
        """Handle stats update."""
        self._session_stats = stats
        self._update_stats_display()
        
    @pyqtSlot(str)
    def _on_proposals_updated(self, image_path: str):
        """Handle proposals updated for an image."""
        # Update gallery thumbnail if this image is displayed
        if self._image_processor.annotation_manager and self._image_processor.annotation_manager.current_session:
            proposals = self._image_processor.annotation_manager.current_session.proposals.get(image_path, [])
            
            # Convert proposals to thumbnail format
            thumbnail_annotations = []
            dimensions = self._image_processor.get_image_dimensions(image_path)
            if dimensions and proposals:
                img_width, img_height = dimensions
                
                for prop in proposals:
                    # Normalize bbox
                    x_center = (prop.bbox[0] + prop.bbox[2] / 2) / img_width
                    y_center = (prop.bbox[1] + prop.bbox[3] / 2) / img_height
                    w_norm = prop.bbox[2] / img_width
                    h_norm = prop.bbox[3] / img_height
                    
                    thumbnail_annotations.append((
                        prop.class_id,
                        x_center,
                        y_center,
                        w_norm,
                        h_norm,
                        prop.confidence
                    ))
                    
            # Update thumbnail
            self._gallery.update_image_annotations(image_path, thumbnail_annotations)
            
    def _update_stats_display(self):
        """Update statistics display."""
        s = self._session_stats
        self._stats_label.setText(
            f"Processed: {s.processed_images}/{s.total_images} | "
            f"Auto: {s.auto_approved} | Review: {s.requires_review} | "
            f"Reject: {s.rejected} | None: {s.no_detections} | "
            f"Modified: {s.modified}"
        )
        
    @property
    def _current_image_path(self) -> Optional[str]:
        """Get current image path."""
        return self.__current_image_path
        
    @_current_image_path.setter
    def _current_image_path(self, value: Optional[str]):
        """Set current image path."""
        self.__current_image_path = value
        
    def _on_activate(self):
        """Called when mode is activated."""
        super()._on_activate()
        
        # Subscribe to model events
        model_cache = ModelCache()
        model_cache.modelLoaded.connect(self._on_model_loaded)
        model_cache.modelCleared.connect(self._on_model_cleared)
        
        # Subscribe to dataset events
        dataset_manager = DatasetManager()
        dataset_manager.datasetLoaded.connect(self._on_dataset_manager_loaded)
        
        # Update status
        self._update_requirements_status()
        
        # Load class names if model is already loaded
        if model_cache.is_loaded():
            self._load_class_names_from_model()
            
    def update_class_names(self, class_names: Dict[int, str]):
        """
        Update class names from external source.
        
        Args:
            class_names: Dictionary mapping class IDs to names
        """
        self._dataset_class_names = class_names
        
        # Update canvas
        self._canvas.set_class_names(class_names)
        
        # Repopulate class combo
        self._populate_class_combo()
        
        # Enable class selection
        self._class_combo.setEnabled(True)
        
        # Update any existing annotations with new class names
        if self._current_image_path:
            annotations = self._canvas.get_annotations()
            if annotations:
                # Force a refresh of the canvas display
                self._canvas.update()
        
        # Update category filter to show all classes
        if hasattr(self, '_category_list_widget'):
            if self._category_list_widget.count() <= 3:
                # Only populate if filter is empty
                self._populate_category_filter()
            else:
                # Just update counts if filter already populated
                self._update_category_filter_counts()
                
    def _on_deactivate(self) -> Optional[bool]:
        """Called when mode is being deactivated."""
        # Save any current annotations
        self._save_current_annotations()
        
        # Stop any ongoing processing
        if self._is_processing:
            self._stop_processing()
        
        # Clean up UI builder event filters
        if hasattr(self, '_ui_builder') and self._ui_builder:
            self._ui_builder.cleanup()
            
        return super()._on_deactivate()
        
    def get_mode_name(self) -> str:
        """Get the display name for this mode."""
        return "Auto-Annotation"
        
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Event filter to save annotations when clicking on gallery."""
        # Note: Annotations are now saved immediately when added/modified/deleted,
        # so we only need to save here if switching images
        if event.type() == QEvent.Type.MouseButtonPress:
            if watched == self._gallery or watched == self._gallery.list_view:
                # Only save when clicking in the gallery (image switching)
                if self._current_image_path:
                    self._save_current_annotations()
                
        # Handle right-click on gallery for context menu
        if (watched == self._gallery.list_view and 
            event.type() == QEvent.Type.ContextMenu):
            # Get the position and show context menu
            pos = event.pos()
            self._show_thumbnail_context_menu(pos)
            return True
            
        return super().eventFilter(watched, event)
        
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        # Note: Annotations are now saved immediately when added/modified/deleted
        # This is kept as a safety net for any edge cases
        super().mousePressEvent(event)
        
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Delete:
            # Delete key pressed - move selected to rejected
            self._move_selected_to_rejected()
        elif event.key() == Qt.Key.Key_R:
            # R key pressed (works for both uppercase and lowercase) - reject images
            # Get selected thumbnails in gallery
            selected = self._gallery.get_selected_paths()
            
            if selected:
                # Process all selected thumbnails
                self._reject_selected()
            elif self._current_image_path:
                # No selection in gallery, but we have a current image in editor
                self._reject_current()
        elif event.key() == Qt.Key.Key_A:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Ctrl+A pressed - select all thumbnails
                self._select_all_thumbnails()
            else:
                # A key pressed (works for both uppercase and lowercase) - approve images
                # Get selected thumbnails in gallery
                selected = self._gallery.get_selected_paths()
                
                if selected:
                    # Process all selected thumbnails
                    self._approve_selected()
                elif self._current_image_path:
                    # No selection in gallery, but we have a current image in editor
                    self._approve_current()
        else:
            super().keyPressEvent(event)
            
    def _update_ui_state(self):
        """Update UI element states based on current state."""
        # Check if model is loaded
        model_cache = ModelCache()
        model_loaded = model_cache.is_loaded()
        
        # Processing state
        self._start_btn.setEnabled(
            model_loaded and 
            bool(self._current_folder) and 
            not self._is_processing
        )
        self._stop_btn.setEnabled(self._is_processing or self._training_handler.is_training)
        
        # Filter buttons enabled when we have results
        has_results = bool(self._image_processor.annotation_manager and 
                          self._image_processor.annotation_manager.current_session)
        self._filter_approved_btn.setEnabled(has_results)
        self._filter_review_btn.setEnabled(has_results)
        self._filter_rejected_btn.setEnabled(has_results)
        self._filter_no_detections_btn.setEnabled(has_results)
        
        # Dataset operations
        has_dataset = bool(self._dataset_yaml_path)
        self._start_training_btn.setEnabled(has_dataset and not self._training_handler.is_training)
        
        # Gallery selection based operations
        self._on_gallery_selection_changed()
        
    def _update_requirements_status(self):
        """Update requirements status display."""
        model_cache = ModelCache()
        
        # Model status
        if model_cache.is_loaded():
            self._model_status_label.setText("✅ Model loaded")
            self._model_status_label.setStyleSheet("color: #4CAF50;")
        else:
            self._model_status_label.setText("❌ Model not loaded")
            self._model_status_label.setStyleSheet("color: #ff6b6b;")
            
        # Update start button
        self._start_btn.setEnabled(
            model_cache.is_loaded() and 
            bool(self._current_folder) and 
            not self._is_processing
        )
        
    def _reload_gallery_with_filter(self):
        """Reload gallery with current filter and category filter applied."""
        if not self._image_processor.annotation_manager:
            return
            
        # Get images based on current confidence filter
        if self._current_filter is not None:
            # Filter by confidence category
            category_images = self._image_processor.annotation_categories.get(self._current_filter, set())
            filtered_paths = list(category_images)
        else:
            # Show all processed images
            filtered_paths = self._image_processor.all_processed_images
            
        # Apply detected category filter
        # Exception: NO_DETECTIONS category should always show all images since they have no categories by definition
        if self._current_filter != ConfidenceCategory.NO_DETECTIONS:
            if self._selected_category_filters:
                # Further filter by detected categories when categories are selected
                final_paths = []
                for path in filtered_paths:
                    detected = self._detected_categories.get(path, set())
                    # Check if image contains any of the selected categories
                    if detected.intersection(self._selected_category_filters):
                        final_paths.append(path)
                filtered_paths = final_paths
            else:
                # No categories selected (and not viewing NO_DETECTIONS) = show no images
                filtered_paths = []
        
        # Build annotations dict for sorting/filtering - always needed for detection filters
        annotations_dict = {}
        if self._image_processor.annotation_manager and self._image_processor.annotation_manager.current_session:
            for image_path in filtered_paths:
                # Get proposals for this image
                proposals = self._image_processor.annotation_manager.current_session.proposals.get(image_path, [])
                if proposals:
                    # Convert proposals to thumbnail annotation format
                    thumbnail_annotations = []
                    for proposal in proposals:
                        # Proposals store bbox in pixel coords, convert to normalized
                        dimensions = self._image_processor.get_image_dimensions(image_path)
                        if dimensions:
                            img_width, img_height = dimensions
                            x, y, w, h = proposal.bbox
                            x_center = (x + w/2) / img_width
                            y_center = (y + h/2) / img_height
                            norm_w = w / img_width
                            norm_h = h / img_height
                            thumbnail_annotations.append((proposal.class_id, x_center, y_center, norm_w, norm_h))
                    annotations_dict[image_path] = thumbnail_annotations
        
        # Apply detection filters if active - always check regardless of sort option
        filter_settings = self._sort_filter_widget.get_filter_settings()
        if filter_settings:
            # Apply detection-based filtering
            final_paths = []
            for path in filtered_paths:
                annotations = annotations_dict.get(path, [])
                
                # Check minimum detections
                if 'min_detections' in filter_settings:
                    if len(annotations) < filter_settings['min_detections']:
                        continue
                
                # Check minimum classes
                if 'min_classes' in filter_settings:
                    unique_classes = set(ann[0] for ann in annotations if len(ann) > 0)
                    if len(unique_classes) < filter_settings['min_classes']:
                        continue
                
                final_paths.append(path)
            filtered_paths = final_paths
        
        # Always apply sorting (includes handling of Default + Descending order)
        sorted_data = self._sort_filter_widget.sort_image_data(
            [(path, {}) for path in filtered_paths], annotations_dict
        )
        sorted_paths = [path for path, _ in sorted_data]
            
        # Load filtered and sorted images to gallery  
        self._gallery.load_images(sorted_paths, annotations_dict)
        
        # Update gallery label
        total = len(self._image_processor.all_processed_images)
        showing = len(sorted_paths)
        if showing < total:
            self._gallery_label.setText(f"Images (showing {showing} of {total})")
        else:
            self._gallery_label.setText(f"Images ({total} total)")
            
        # Re-apply annotations to visible thumbnails
        if self._image_processor.annotation_manager and self._image_processor.annotation_manager.current_session:
            for image_path in filtered_paths:
                # Check if we have modified annotations
                if image_path in self._image_processor.modified_annotations:
                    annotations = self._image_processor.modified_annotations[image_path]
                    
                    # Convert to thumbnail format
                    thumbnail_annotations = []
                    dimensions = self._image_processor.get_image_dimensions(image_path)
                    if dimensions:
                        img_width, img_height = dimensions
                        
                        for ann in annotations:
                            rect = ann.rect
                            x_center = (rect.x() + rect.width() / 2) / img_width
                            y_center = (rect.y() + rect.height() / 2) / img_height
                            w_norm = rect.width() / img_width
                            h_norm = rect.height() / img_height
                            
                            thumbnail_annotations.append((
                                ann.class_id,
                                x_center,
                                y_center,
                                w_norm,
                                h_norm,
                                ann.confidence if ann.confidence else 1.0
                            ))
                            
                    self._gallery.update_image_annotations(image_path, thumbnail_annotations, is_modified=True)
                else:
                    # Use proposals
                    proposals = self._image_processor.annotation_manager.current_session.proposals.get(image_path, [])
                    
                    # Convert proposals to thumbnail format
                    thumbnail_annotations = []
                    dimensions = self._image_processor.get_image_dimensions(image_path)
                    if dimensions and proposals:
                        img_width, img_height = dimensions
                        
                        for prop in proposals:
                            # Normalize bbox
                            x_center = (prop.bbox[0] + prop.bbox[2] / 2) / img_width
                            y_center = (prop.bbox[1] + prop.bbox[3] / 2) / img_height
                            w_norm = prop.bbox[2] / img_width
                            h_norm = prop.bbox[3] / img_height
                            
                            thumbnail_annotations.append((
                                prop.class_id,
                                x_center,
                                y_center,
                                w_norm,
                                h_norm,
                                prop.confidence
                            ))
                            
                    self._gallery.update_image_annotations(image_path, thumbnail_annotations)
                    
        # If no filtered images, clear the canvas
        if len(filtered_paths) == 0:
            self._current_image_path = None
            self._canvas.clear_canvas()
            self._info_label.setText("No images in this category")
            self._annotation_count_label.setText("Annotations: 0")
                    
    def _on_dataset_manager_loaded(self, yaml_path: Path):
        """Handle dataset loaded from DatasetManager."""
        # Load the dataset YAML to get class names
        try:
            data = load_data_yaml(yaml_path)
            if 'names' in data:
                self._dataset_class_names = data['names']
                self.update_class_names(data['names'])
        except Exception as e:
            print(f"Error loading dataset YAML: {e}")
            
    def _on_model_loaded(self, model_path: str):
        """Handle model loaded."""
        self._update_requirements_status()
        self._update_ui_state()
        
        # Load class names from model if no dataset is loaded
        if not self._dataset_class_names:
            self._load_class_names_from_model()
            
        # If workflow enabled and we just completed a training iteration
        if (self._workflow_enabled and 
            self._workflow_state.current_stage == "loading_model"):
            # Continue workflow - automatically start new annotation session
            self._set_workflow_stage("idle")
            
            # Don't show message here - wait until after inference
            # Store iteration info for later
            self._pending_iteration_message = True
            
            # Reset filters and categories for new session
            self._detected_categories.clear()
            self._selected_category_filters.clear()
            self._current_filter = ConfidenceCategory.REQUIRES_REVIEW
            self._filter_approved_btn.setChecked(False)
            self._filter_review_btn.setChecked(True)
            self._filter_rejected_btn.setChecked(False)
            self._filter_no_detections_btn.setChecked(False)
            
            # Clear category filter
            while self._category_list_widget.count() > 3:
                self._category_list_widget.takeItem(3)
            self._category_filter_combo.setCurrentText("All categories")
            
            # Automatically start processing with current folder
            if self._current_folder:
                # Re-load images from folder and start processing
                self._start_processing()
                                  
    def _on_model_cleared(self):
        """Handle model cleared."""
        self._update_requirements_status()
        self._update_ui_state()
        
        # Don't clear dataset class names - keep them if loaded from dataset
        if not self._dataset_yaml_path:
            self._dataset_class_names.clear()
            self._populate_class_combo()
            
    def _on_category_filter_clicked(self, item: QListWidgetItem):
        """Handle clicks on category filter items."""
        action = item.data(Qt.ItemDataRole.UserRole)
        
        if action == "select_all":
            # Check all category items
            for i in range(3, self._category_list_widget.count()):
                item = self._category_list_widget.item(i)
                if item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    item.setCheckState(Qt.CheckState.Checked)
                    
        elif action == "select_none":
            # Uncheck all category items
            for i in range(3, self._category_list_widget.count()):
                item = self._category_list_widget.item(i)
                if item and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    item.setCheckState(Qt.CheckState.Unchecked)
                    
        elif action is not None and isinstance(action, int):
            # This is a category item - toggle its checkbox
            if item.checkState() == Qt.CheckState.Checked:
                item.setCheckState(Qt.CheckState.Unchecked)
            else:
                item.setCheckState(Qt.CheckState.Checked)
                
        # Update filter text
        self._update_category_filter_text()
        
        # Apply filter (don't preserve selection for manual filter clicks)
        self._apply_category_filter(preserve_selection=False)
        
    def _update_category_filter_text(self):
        """Update category filter combo box text based on selection."""
        checked_items = []
        for i in range(3, self._category_list_widget.count()):
            item = self._category_list_widget.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                checked_items.append(item.text().split(' ')[0])  # Get class name only
                
        if not checked_items:
            self._category_filter_combo.setCurrentText("No categories selected")
        elif len(checked_items) == self._category_list_widget.count() - 3:
            self._category_filter_combo.setCurrentText("All categories")
        else:
            self._category_filter_combo.setCurrentText(f"{len(checked_items)} categories selected")
            
    def _populate_category_filter(self):
        """Populate the category filter with all dataset classes."""
        # Clear existing items (keep first 3)
        while self._category_list_widget.count() > 3:
            self._category_list_widget.takeItem(3)
            
        # Use all classes from dataset, not just detected ones
        if self._dataset_class_names:
            # Add all dataset classes
            for class_id in sorted(self._dataset_class_names.keys()):
                class_name = self._dataset_class_names[class_id]
                
                # Count images with this class (from detections)
                count = sum(1 for classes in self._detected_categories.values() if class_id in classes)
                
                item = QListWidgetItem(f"{class_name} ({count})")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)  # Default to checked
                item.setData(Qt.ItemDataRole.UserRole, class_id)
                self._category_list_widget.addItem(item)
                
            # Update combo text
            self._update_category_filter_text()
            
            # Initialize selected categories (all classes checked by default)
            self._selected_category_filters = set(self._dataset_class_names.keys())
        else:
            # No dataset loaded - use detected classes as fallback
            all_classes = set()
            for classes in self._detected_categories.values():
                all_classes.update(classes)
                
            # Add detected class items
            for class_id in sorted(all_classes):
                class_name = f"Class {class_id}"
                
                # Count images with this class
                count = sum(1 for classes in self._detected_categories.values() if class_id in classes)
                
                item = QListWidgetItem(f"{class_name} ({count})")
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)  # Default to checked
                item.setData(Qt.ItemDataRole.UserRole, class_id)
                self._category_list_widget.addItem(item)
                
            # Update combo text
            self._update_category_filter_text()
            
            # Initialize selected categories (all checked by default)
            if all_classes:
                self._selected_category_filters = all_classes.copy()
        
    def _apply_category_filter(self, preserve_selection=False):
        """Apply the selected category filters.
        
        Args:
            preserve_selection: If True, try to preserve current selection
        """
        # Save current selection if requested
        current_selection = None
        if preserve_selection and self._current_image_path:
            current_selection = self._current_image_path
            
        # Get selected categories
        self._selected_category_filters.clear()
        for i in range(3, self._category_list_widget.count()):
            item = self._category_list_widget.item(i)
            if item and item.checkState() == Qt.CheckState.Checked:
                class_id = item.data(Qt.ItemDataRole.UserRole)
                if class_id is not None:
                    self._selected_category_filters.add(class_id)
                    
        # Reload gallery with updated filter
        self._reload_gallery_with_filter()
        
        # Restore selection or select first image
        if preserve_selection and current_selection:
            # Try to find and select the previously selected image
            for row in range(self._gallery._model.rowCount()):
                item = self._gallery._model.get_item(row)
                if item and item.image_path == current_selection:
                    self._select_image_at_position(row)
                    return
        
        # Only select first image if explicitly filtering (not preserving)
        if not preserve_selection:
            self._select_image_at_position(0)
        
    def _on_category_filter_changed(self, item: QListWidgetItem):
        """Handle category filter item state changes."""
        # Skip if it's one of the action items
        if item.data(Qt.ItemDataRole.UserRole) in ["select_all", "select_none", None]:
            return
            
        # Update filter text
        self._update_category_filter_text()
        
        # Apply filter (preserve selection for checkbox changes)
        self._apply_category_filter(preserve_selection=True)
        
    def _move_selected_to_rejected(self):
        """Move selected images to rejected folder."""
        selected = self._gallery.get_selected_paths()
        if not selected:
            return
            
        # Confirm action
        reply = QMessageBox.question(
            self, "Move to Rejected",
            f"Move {len(selected)} selected image(s) to 'rejected' folder?\n\n"
            "This will physically move the files and their annotations.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        # Create rejected folder
        rejected_folder = self._current_folder / "rejected"
        rejected_folder.mkdir(exist_ok=True)
        
        # Move files
        moved_count = 0
        for image_path in selected:
            try:
                # Move image and annotation if exists
                img_path = Path(image_path)
                ann_path = img_path.with_suffix('.txt')
                
                # Move image
                dest_img = rejected_folder / img_path.name
                shutil.move(str(img_path), str(dest_img))
                
                # Move annotation if exists
                if ann_path.exists():
                    dest_ann = rejected_folder / ann_path.name
                    shutil.move(str(ann_path), str(dest_ann))
                    
                moved_count += 1
                
                # Clean up from internal tracking
                self._image_processor.clean_up_after_export([image_path])
                
            except Exception as e:
                print(f"Error moving {image_path}: {e}")
                
        # Remove from gallery
        self._gallery._model.remove_items_by_paths(selected)
        
        # Reload gallery
        self._reload_gallery_with_filter()
        
        # Show confirmation
        QMessageBox.information(self, "Move Complete", 
                              f"Moved {moved_count} image(s) to rejected folder.")
                              
    def _select_all_thumbnails(self):
        """Select all thumbnails in gallery."""
        self._gallery.select_all()
        
    def _select_none_thumbnails(self):
        """Deselect all thumbnails in gallery."""
        self._gallery.clear_selection()
        
    def _show_thumbnail_context_menu(self, pos: QPoint):
        """Show context menu for thumbnail gallery."""
        # Get the item at position
        index = self._gallery.list_view.indexAt(pos)
        if not index.isValid():
            return
            
        # Get selected items
        selected = self._gallery.get_selected_paths()
        if not selected:
            return
            
        # Create context menu
        menu = QMenu(self)
        
        # Add actions based on current category
        if len(selected) == 1:
            image_path = selected[0]
            category = self._image_processor.get_image_category(image_path)
            
            if category != ConfidenceCategory.AUTO_APPROVED:
                approve_action = QAction("✓ Approve", self)
                approve_action.triggered.connect(lambda: self._approve_selected())
                menu.addAction(approve_action)
                
            if category != ConfidenceCategory.REJECTED:
                reject_action = QAction("✗ Reject", self)
                reject_action.triggered.connect(lambda: self._reject_selected())
                menu.addAction(reject_action)
        else:
            # Multiple selection
            approve_action = QAction(f"✓ Approve {len(selected)} images", self)
            approve_action.triggered.connect(lambda: self._approve_selected())
            menu.addAction(approve_action)
            
            reject_action = QAction(f"✗ Reject {len(selected)} images", self)
            reject_action.triggered.connect(lambda: self._reject_selected())
            menu.addAction(reject_action)
            
        menu.addSeparator()
        
        # Export action
        export_action = QAction("Export Annotations", self)
        export_action.triggered.connect(lambda: self._export_annotations())
        menu.addAction(export_action)
        
        menu.addSeparator()
        
        # Move to rejected action
        move_action = QAction("Move to Rejected Folder", self)
        move_action.triggered.connect(lambda: self._move_selected_to_rejected())
        menu.addAction(move_action)
        
        # Show menu
        menu.exec(self._gallery.list_view.mapToGlobal(pos))
        
    def _move_image_to_category(self, image_path: str, category: ConfidenceCategory):
        """Move an image to a specific category."""
        # Update in image processor
        old_category = self._image_processor.get_image_category(image_path)
        if old_category != category:
            # Approve or reject based on category
            if category == ConfidenceCategory.AUTO_APPROVED:
                self._image_processor.approve_proposals([image_path])
            elif category == ConfidenceCategory.REJECTED:
                self._image_processor.reject_proposals([image_path])
                
            # Reload gallery if needed
            if self._current_filter is not None and (
                old_category == self._current_filter or 
                category == self._current_filter):
                self._reload_gallery_with_filter()
                
    def _move_single_to_rejected_folder(self, image_path: str):
        """Move a single image to rejected folder."""
        self._move_selected_to_rejected()  # Reuse existing logic
        
    def _manage_dataset(self):
        """Manage dataset - load existing or create new."""
        yaml_path = self._dataset_handler.manage_dataset()
        if yaml_path:
            self._dataset_yaml_path = yaml_path
            self._dataset_info_label.setText(f"Dataset: {yaml_path.name}\nPath: {yaml_path.parent}")
            self._workflow_state.dataset_path = yaml_path.parent
            self._update_ui_state()
            
    def _on_dataset_loaded(self, yaml_path: Path, data: dict):
        """Handle dataset loaded."""
        self._dataset_yaml_path = yaml_path
        self._workflow_state.dataset_path = yaml_path.parent
        
        # Update UI
        self._dataset_info_label.setText(f"Dataset: {yaml_path.name}\nPath: {yaml_path.parent}")
        
        # Load class names if available
        if 'names' in data:
            self._dataset_class_names = data['names']
            self.update_class_names(data['names'])
            
            # Populate category filter with all dataset classes
            if self._image_processor.annotation_manager:
                self._populate_category_filter()
            
        # Enable relevant buttons
        self._update_ui_state()
        
    def _on_dataset_created(self, yaml_path: Path):
        """Handle dataset created."""
        self._dataset_yaml_path = yaml_path
        self._workflow_state.dataset_path = yaml_path.parent
        
        # Update UI
        self._dataset_info_label.setText(f"New dataset created at:\n{yaml_path.parent}")
        
        # Enable relevant buttons
        self._update_ui_state()
        
    def _update_split_percentages(self):
        """Update split percentages to ensure they sum to 100."""
        train = self._train_split_spin.value()
        val = self._val_split_spin.value()
        test = 100 - train - val
        
        self._test_split_spin.blockSignals(True)
        self._test_split_spin.setValue(max(0, test))
        self._test_split_spin.blockSignals(False)
        
    def _execute_dataset_split(self):
        """Execute dataset split operation on the current folder."""
        if not self._current_folder or not self._dataset_yaml_path:
            QMessageBox.warning(self, "Missing Requirements", 
                              "Please ensure you have a folder selected and dataset loaded.")
            self._set_workflow_stage("idle")
            return
            
        self._set_workflow_stage("splitting")
        
        # Get split percentages
        train_pct, val_pct, test_pct = self._ui_builder.get_split_percentages()
        
        # Execute split
        success = self._dataset_handler.execute_dataset_split(
            self._current_folder,
            train_pct, val_pct, test_pct,
            self._workflow_state,
            self._last_exported_paths
        )
        
        if success:
            # Clean up gallery and tracking after successful split
            if self._last_exported_paths:
                self._image_processor.clean_up_after_export(self._last_exported_paths)
                
                # Remove from gallery
                self._gallery._model.remove_items_by_paths(self._last_exported_paths)
                
                # Reload gallery
                self._reload_gallery_with_filter()
                
                # Clear the exported paths list
                self._last_exported_paths = []
                
            # If workflow enabled, proceed to training
            if self._workflow_enabled:
                self._start_training()
            else:
                self._set_workflow_stage("idle")
        else:
            self._set_workflow_stage("idle")
            
    def _on_split_completed(self, message: str, split_counts: dict):
        """Handle dataset split completion."""
        self.datasetSplitCompleted.emit(str(self._dataset_yaml_path.parent))
        
    def _start_training(self):
        """Start training process."""
        if not self._dataset_yaml_path:
            QMessageBox.warning(self, "No Dataset", "Please load or create a dataset first.")
            return
            
        self._set_workflow_stage("training")
        
        # Get training config from UI
        training_config = self._ui_builder.get_training_config()
        
        # Start training
        success = self._training_handler.start_training(
            self._dataset_yaml_path,
            self._workflow_state,
            training_config
        )
        
        if success:
            # Update UI
            self._start_training_btn.setEnabled(False)
            self._stop_training_btn.setEnabled(True)
            self._training_progress_bar.setVisible(True)
            self._training_status_label.setVisible(True)
            self._training_console.setVisible(True)
            self._training_console.clear()
            self._training_progress_bar.setMaximum(training_config['epochs'])
            self._training_progress_bar.setValue(0)
            self._training_status_label.setText("Starting training...")
            
            # Also update main progress bar
            self._progress_bar.setMaximum(training_config['epochs'])
            self._progress_bar.setValue(0)
            self._progress_label.setText("Starting training...")
            
            # Switch to training chart view
            self._center_stack.setCurrentIndex(1)
            self._gallery_label.setText("Training Progress")
            
            # Clear and start monitoring charts
            self._training_charts.clear_data()
            self._training_charts.start_monitoring()
            
            # Connect training output to charts
            self._training_handler.logMessage.connect(self._training_charts.on_metrics_update)
            self._training_handler.trainingProgress.connect(self._on_training_progress_for_chart)
            
    def _stop_training(self):
        """Stop training process."""
        self._training_handler.stop_training()
        self._start_training_btn.setEnabled(True)
        self._stop_training_btn.setEnabled(False)
        self._set_workflow_stage("idle")
        
        # Stop chart monitoring and switch back to gallery
        self._training_charts.stop_monitoring()
        try:
            self._training_handler.logMessage.disconnect(self._training_charts.on_metrics_update)
            self._training_handler.trainingProgress.disconnect(self._on_training_progress_for_chart)
        except:
            pass  # Ignore if already disconnected
        self._center_stack.setCurrentIndex(0)
        self._gallery_label.setText(f"Images ({self._gallery._model.rowCount()} total)")
        
    def _on_training_started(self, config_path: str):
        """Handle training started."""
        self.trainingStarted.emit(config_path)
        
    def _on_training_output(self, output: str):
        """Handle training output."""
        # Append to console
        self._training_console.append(output)
        # Scroll to bottom
        scrollbar = self._training_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Update status label with latest output
        if "Epoch" in output:
            self._training_status_label.setText(output.strip())
            
    def _on_training_progress(self, epoch: int, total_epochs: int, metrics: dict):
        """Handle training progress update."""
        self._training_progress_bar.setValue(epoch)
        
        # Update status with ETA if available
        status_text = f"Epoch {epoch}/{total_epochs}"
        if metrics.get('eta'):
            status_text += f" | ETA: {metrics['eta']}"
        self._training_status_label.setText(status_text)
        
        # Also update the main progress bar in the Progress panel
        self._progress_bar.setMaximum(total_epochs)
        self._progress_bar.setValue(epoch)
        progress_text = f"Training: Epoch {epoch}/{total_epochs}"
        if metrics.get('eta'):
            progress_text += f" - ETA: {metrics['eta']}"
        self._progress_label.setText(progress_text)
        
        self.trainingProgress.emit(epoch, total_epochs, metrics)
        
    def _on_training_progress_for_chart(self, epoch: int, total_epochs: int, metrics: dict):
        """Update training chart with epoch progress."""
        self._training_charts.on_epoch_update(epoch, total_epochs)
        
    def _on_training_completed(self, model_path: str):
        """Handle training completion."""
        self._training_progress_bar.setVisible(False)
        self._training_status_label.setVisible(False)
        self._training_console.setVisible(False)
        self._start_training_btn.setEnabled(True)
        self._stop_training_btn.setEnabled(False)
        
        # Reset main progress bar
        self._progress_bar.setValue(0)
        self._progress_label.setText("Training complete")
        
        # Stop chart monitoring and switch back to gallery
        self._training_charts.stop_monitoring()
        try:
            self._training_handler.logMessage.disconnect(self._training_charts.on_metrics_update)
            self._training_handler.trainingProgress.disconnect(self._on_training_progress_for_chart)
        except:
            pass  # Ignore if already disconnected
        self._center_stack.setCurrentIndex(0)
        self._gallery_label.setText(f"Images ({self._gallery._model.rowCount()} total)")
        
        # Add to model history
        self._workflow_state.iteration += 1
        self._workflow_state.model_history.append((self._workflow_state.iteration, model_path))
        
        # Update UI
        self._iteration_label.setText(f"Iteration: {self._workflow_state.iteration}")
        self._model_history_combo.addItem(f"Iteration {self._workflow_state.iteration}", model_path)
        
        # Load the new model
        if self._workflow_enabled:
            self._set_workflow_stage("loading_model")
            self._training_handler.load_model_to_cache(model_path, self._workflow_enabled)
        else:
            self._set_workflow_stage("idle")
            
        self.trainingCompleted.emit(model_path)
        
    def _on_training_error(self, error: str):
        """Handle training error."""
        self._training_progress_bar.setVisible(False)
        self._training_status_label.setVisible(False)
        self._start_training_btn.setEnabled(True)
        self._stop_training_btn.setEnabled(False)
        self._set_workflow_stage("idle")
        
        # Reset main progress bar
        self._progress_bar.setValue(0)
        self._progress_label.setText("Training failed")
        
        # Stop chart monitoring and switch back to gallery
        self._training_charts.stop_monitoring()
        try:
            self._training_handler.logMessage.disconnect(self._training_charts.on_metrics_update)
            self._training_handler.trainingProgress.disconnect(self._on_training_progress_for_chart)
        except:
            pass  # Ignore if already disconnected
        self._center_stack.setCurrentIndex(0)
        self._gallery_label.setText(f"Images ({self._gallery._model.rowCount()} total)")
        
        # Show error dialog
        console_output = self._training_console.toPlainText()
        self._training_handler.show_training_error(error, console_output)
        
        # Keep console visible for debugging
        self._training_console.append("\n--- TRAINING FAILED ---")
        self._training_console.append(f"Error: {error}")
        
    def _on_handler_error(self, error_msg: str):
        """Handle errors from handlers."""
        QMessageBox.critical(self, "Error", error_msg)
        
    def _toggle_workflow(self, enabled: bool):
        """Toggle workflow automation."""
        self._workflow_enabled = enabled
        if enabled:
            # Check if dataset is loaded
            if not self._dataset_yaml_path:
                # Remind user to load dataset
                QMessageBox.warning(self, "Dataset Required", 
                                  "Automated workflow requires a dataset to be loaded.\n\n"
                                  "Please use 'Load/Create Dataset' to set up your dataset first.")
                # Uncheck the checkbox
                self._workflow_checkbox.setChecked(False)
                self._workflow_enabled = False
                return
            
            QMessageBox.information(self, "Workflow Enabled", 
                                  "Automated workflow is now enabled.\n"
                                  "The system will automatically transition between stages.")
                                  
    def _set_workflow_stage(self, stage: str):
        """Set the current workflow stage."""
        self._workflow_state.current_stage = stage
        stage_display = {
            "idle": "Idle",
            "annotating": "Annotating",
            "exporting": "Exporting",
            "splitting": "Splitting Dataset",
            "training": "Training Model",
            "loading_model": "Loading Model"
        }
        self._stage_label.setText(f"Stage: {stage_display.get(stage, stage)}")
        self.workflowStageChanged.emit(stage)
        
    def _load_historical_model(self, index: int):
        """Load a model from history."""
        if index > 0:  # Skip "Initial model" entry
            model_path = self._model_history_combo.itemData(index)
            if model_path and Path(model_path).exists():
                model_cache = ModelCache()
                model_cache.load_model(model_path)
                
    def set_folder(self, folder_path: str):
        """Set the current folder for processing."""
        # Set busy cursor
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            # Stop any ongoing processing
            if self._is_processing:
                self._stop_processing()
                
            # Clear previous session data
            self._current_folder = Path(folder_path)
            self._folder_label.setText(f"Folder: {self._current_folder.name}")
            self._folder_label.setToolTip(str(self._current_folder))
            
            # Load images from the folder into gallery
            all_image_paths_set = set()
            for ext in IMAGE_EXTENSIONS:
                all_image_paths_set.update(self._current_folder.glob(f'*{ext}'))
                all_image_paths_set.update(self._current_folder.glob(f'*{ext.upper()}'))
            
            # Convert to string paths and sort
            image_paths = sorted([str(p) for p in all_image_paths_set])
            
            # Load images to gallery
            if image_paths:
                self._gallery.load_images(image_paths)
                self._gallery_label.setText(f"Images ({len(image_paths)} total)")
            else:
                self._gallery.load_images([])
                self._gallery_label.setText("Images (0 total)")
            
            # Reset filters and categories
            self._detected_categories.clear()
            self._selected_category_filters.clear()
            self._current_filter = None
            
            # Reset filter buttons
            self._filter_approved_btn.setChecked(False)
            self._filter_review_btn.setChecked(False)
            self._filter_rejected_btn.setChecked(False)
            self._filter_no_detections_btn.setChecked(False)
            
            # Clear category filter
            while self._category_list_widget.count() > 3:
                self._category_list_widget.takeItem(3)
            self._category_filter_combo.setCurrentText("All categories")
            
            # Clear current image if any
            if self._current_image_path:
                self._current_image_path = None
                self._canvas.clear_canvas()
                self._info_label.setText("No image selected")
                self._annotation_count_label.setText("Annotations: 0")
                
            # Clear image processor state
            self._clear_processor_state()
            
            # Update stats
            self._session_stats = SessionStats()
            self._update_stats_display()
            
            # Disable refresh thresholds button until new processing
            self._refresh_thresholds_btn.setEnabled(False)
            
            # Update UI state
            self._update_ui_state()
            
        finally:
            # Restore cursor
            QApplication.restoreOverrideCursor()
        
    def _clear_processor_state(self):
        """Clear image processor state."""
        # Clear annotation manager session by setting the private attribute
        if self._image_processor.annotation_manager:
            self._image_processor.annotation_manager._current_session = None
            
        # Clear processor state
        self._image_processor._all_processed_images.clear()
        self._image_processor._annotation_proposals.clear()
        self._image_processor._modified_annotations.clear()
        self._image_processor._image_dimensions_cache.clear()
        
        # Clear annotation categories
        for category_set in self._image_processor._annotation_categories.values():
            category_set.clear()
    
    def _toggle_gallery_expansion(self):
        """Toggle gallery expansion/retraction."""
        if self._gallery_expanded:
            # Retract gallery - restore saved sizes
            if self._saved_splitter_sizes:
                self._splitter.setSizes(self._saved_splitter_sizes)
            else:
                # Default sizes if no saved state
                self._splitter.setSizes([300, 600, 600])
            self._gallery_expanded = False
            self._expand_gallery_btn.setText("Expand Gallery")
        else:
            # Expand gallery - save current sizes and hide editor
            self._saved_splitter_sizes = self._splitter.sizes()
            # Get total width
            total_width = sum(self._splitter.sizes())
            # Set gallery to take up most space, keep controls visible
            controls_width = self._splitter.sizes()[0]
            self._splitter.setSizes([controls_width, total_width - controls_width, 0])
            self._gallery_expanded = True
            self._expand_gallery_btn.setText("Retract Gallery")
    
    def _update_category_filter_counts(self):
        """Update the counts in category filter without changing selections."""
        if not self._dataset_class_names:
            return
            
        # Save current selection states
        selected_states = {}
        for i in range(3, self._category_list_widget.count()):
            item = self._category_list_widget.item(i)
            if item:
                class_id = item.data(Qt.ItemDataRole.UserRole)
                if class_id is not None:
                    selected_states[class_id] = item.checkState()
        
        # Update counts for each class
        for i in range(3, self._category_list_widget.count()):
            item = self._category_list_widget.item(i)
            if item:
                class_id = item.data(Qt.ItemDataRole.UserRole)
                if class_id is not None and class_id in self._dataset_class_names:
                    # Count images with this class
                    count = sum(1 for classes in self._detected_categories.values() if class_id in classes)
                    class_name = self._dataset_class_names[class_id]
                    
                    # Update item text with new count
                    item.setText(f"{class_name} ({count})")
                    
                    # Restore selection state
                    if class_id in selected_states:
                        item.setCheckState(selected_states[class_id])