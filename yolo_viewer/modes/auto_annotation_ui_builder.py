"""UI building components for auto-annotation mode."""

from typing import Dict, Optional, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFrame, QComboBox, QCheckBox, QListWidget, 
    QListWidgetItem, QSlider, QSpinBox, QDoubleSpinBox,
    QTextEdit, QScrollArea, QProgressBar, QStackedWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject

from ..widgets.thumbnail_gallery import ThumbnailGallery
from ..widgets.annotation_canvas import AnnotationCanvas
from ..widgets.augmentation_settings import InfoLabel, AugmentationSettings
from ..widgets.training_charts import TrainingCharts
from .auto_annotation_data_classes import ConfidenceCategory


class UIBuilder(QObject):
    """Builds UI components for auto-annotation mode."""
    
    # Signals for UI events
    selectFolder = pyqtSignal()
    startClicked = pyqtSignal()
    stopClicked = pyqtSignal()
    thresholdChanged = pyqtSignal()
    refreshThresholds = pyqtSignal()
    augmentationToggled = pyqtSignal(bool)
    workflowToggled = pyqtSignal(bool)
    modelHistoryChanged = pyqtSignal(int)
    datasetManage = pyqtSignal()
    splitPercentageChanged = pyqtSignal()
    startTraining = pyqtSignal()
    stopTraining = pyqtSignal()
    approveSelected = pyqtSignal()
    rejectSelected = pyqtSignal()
    exportAnnotations = pyqtSignal()
    qualityAssessment = pyqtSignal()
    activeLearning = pyqtSignal()
    moveToRejected = pyqtSignal()
    selectAllThumbnails = pyqtSignal()
    selectNoneThumbnails = pyqtSignal()
    filterByCategory = pyqtSignal(object)  # ConfidenceCategory or None
    categoryFilterClicked = pyqtSignal(QListWidgetItem)
    categoryFilterChanged = pyqtSignal(QListWidgetItem)
    classChanged = pyqtSignal(int)
    approveCurrentImage = pyqtSignal()
    rejectCurrentImage = pyqtSignal()
    expandGallery = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self._init_ui_elements()
        
    def _init_ui_elements(self):
        """Initialize UI element references."""
        # Control elements
        self.select_folder_btn: Optional[QPushButton] = None
        self.folder_label: Optional[QLabel] = None
        self.start_btn: Optional[QPushButton] = None
        self.stop_btn: Optional[QPushButton] = None
        self.include_annotated_checkbox: Optional[QCheckBox] = None
        
        # Workflow elements
        self.workflow_checkbox: Optional[QCheckBox] = None
        self.iteration_label: Optional[QLabel] = None
        self.stage_label: Optional[QLabel] = None
        self.model_history_combo: Optional[QComboBox] = None
        
        # Progress elements
        self.progress_bar: Optional[QProgressBar] = None
        self.progress_label: Optional[QLabel] = None
        self.stats_label: Optional[QLabel] = None
        
        # Requirements
        self.model_status_label: Optional[QLabel] = None
        
        # Thresholds
        self.high_conf_slider: Optional[QSlider] = None
        self.high_conf_label: Optional[QLabel] = None
        self.med_conf_slider: Optional[QSlider] = None
        self.med_conf_label: Optional[QLabel] = None
        self.refresh_thresholds_btn: Optional[QPushButton] = None
        
        # Augmentation
        self.enable_augmentation_checkbox: Optional[QCheckBox] = None
        self.augmentation_settings: Optional[AugmentationSettings] = None
        self.augmentation_scroll: Optional[QScrollArea] = None
        
        # Dataset management
        self.dataset_btn: Optional[QPushButton] = None
        self.dataset_info_label: Optional[QLabel] = None
        self.train_split_spin: Optional[QSpinBox] = None
        self.val_split_spin: Optional[QSpinBox] = None
        self.test_split_spin: Optional[QSpinBox] = None
        
        # Training settings
        self.training_model_combo: Optional[QComboBox] = None
        self.epochs_spin: Optional[QSpinBox] = None
        self.batch_spin: Optional[QSpinBox] = None
        self.image_size_combo: Optional[QComboBox] = None
        self.lr_spin: Optional[QDoubleSpinBox] = None
        self.start_training_btn: Optional[QPushButton] = None
        self.stop_training_btn: Optional[QPushButton] = None
        self.training_progress_bar: Optional[QProgressBar] = None
        self.training_status_label: Optional[QLabel] = None
        self.training_console: Optional[QTextEdit] = None
        
        # Actions
        self.approve_selected_btn: Optional[QPushButton] = None
        self.reject_selected_btn: Optional[QPushButton] = None
        self.export_btn: Optional[QPushButton] = None
        self.quality_btn: Optional[QPushButton] = None
        self.active_learning_btn: Optional[QPushButton] = None
        self.move_to_rejected_btn: Optional[QPushButton] = None
        
        # Gallery elements
        self.gallery: Optional[ThumbnailGallery] = None
        self.gallery_label: Optional[QLabel] = None
        self.center_stack: Optional[QStackedWidget] = None
        self.training_charts: Optional[TrainingCharts] = None
        self.select_all_btn: Optional[QPushButton] = None
        self.select_none_btn: Optional[QPushButton] = None
        self.expand_gallery_btn: Optional[QPushButton] = None
        self.category_filter_combo: Optional[QComboBox] = None
        self.category_list_widget: Optional[QListWidget] = None
        self.filter_review_btn: Optional[QPushButton] = None
        self.filter_approved_btn: Optional[QPushButton] = None
        self.filter_rejected_btn: Optional[QPushButton] = None
        self.filter_no_detections_btn: Optional[QPushButton] = None
        
        # Editor elements
        self.annotation_canvas: Optional[AnnotationCanvas] = None
        self.annotation_count_label: Optional[QLabel] = None
        self.class_combo: Optional[QComboBox] = None
        
    def create_controls_panel(self) -> QWidget:
        """Create the controls panel."""
        # Create scrollable container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Add all control groups
        layout.addWidget(self._create_controls_group())
        layout.addWidget(self._create_progress_group())
        layout.addWidget(self._create_workflow_group())
        layout.addWidget(self._create_requirements_group())
        layout.addWidget(self._create_thresholds_group())
        layout.addWidget(self._create_augmentation_group())
        layout.addWidget(self._create_dataset_group())
        layout.addWidget(self._create_training_group())
        layout.addWidget(self._create_actions_group())
        
        layout.addStretch()
        
        # Set the panel as the scroll widget
        scroll.setWidget(panel)
        return scroll
    
    def _create_controls_group(self) -> QGroupBox:
        """Create control buttons group."""
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        
        # Folder selection
        self.select_folder_btn = QPushButton("Select Folder")
        self.select_folder_btn.clicked.connect(self.selectFolder.emit)
        controls_layout.addWidget(self.select_folder_btn)
        
        # Folder label
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("color: #666;")
        controls_layout.addWidget(self.folder_label)
        
        controls_layout.addSpacing(10)
        
        # Dataset selection
        self.dataset_btn = QPushButton("Load Dataset")
        self.dataset_btn.setToolTip("Load existing data.yaml")
        self.dataset_btn.clicked.connect(self.datasetManage.emit)
        controls_layout.addWidget(self.dataset_btn)
        
        controls_layout.addSpacing(10)
        
        # Start button
        self.start_btn = QPushButton("Start Auto-Annotation")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.startClicked.emit)
        controls_layout.addWidget(self.start_btn)
        
        # Stop button
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stopClicked.emit)
        controls_layout.addWidget(self.stop_btn)
        
        controls_layout.addSpacing(10)
        
        # Export
        self.export_btn = QPushButton("Export Selected Annotations")
        self.export_btn.setToolTip("Export selected annotations. With automated workflow enabled, this triggers the next phase of splitting, training, and inferring again.")
        self.export_btn.clicked.connect(self.exportAnnotations.emit)
        controls_layout.addWidget(self.export_btn)
        
        # Add checkbox for including annotated images
        controls_layout.addSpacing(10)
        self.include_annotated_checkbox = QCheckBox("Include already annotated images")
        self.include_annotated_checkbox.setToolTip("When unchecked, images with existing annotations will be skipped")
        self.include_annotated_checkbox.setChecked(False)
        controls_layout.addWidget(self.include_annotated_checkbox)
        
        controls_group.setLayout(controls_layout)
        return controls_group
    
    def _create_workflow_group(self) -> QGroupBox:
        """Create workflow automation group."""
        workflow_group = QGroupBox("Workflow Automation")
        workflow_layout = QVBoxLayout()
        
        # Workflow toggle
        self.workflow_checkbox = QCheckBox("Enable automated workflow")
        self.workflow_checkbox.setToolTip("Automatically transition between annotation, export, split, and training stages")
        self.workflow_checkbox.setChecked(True)
        self.workflow_checkbox.toggled.connect(self.workflowToggled.emit)
        workflow_layout.addWidget(self.workflow_checkbox)
        
        # Iteration info
        self.iteration_label = QLabel("Iteration: 0")
        self.iteration_label.setStyleSheet("font-weight: bold;")
        workflow_layout.addWidget(self.iteration_label)
        
        # Workflow stage
        self.stage_label = QLabel("Stage: Idle")
        self.stage_label.setStyleSheet("color: #14ffec;")
        workflow_layout.addWidget(self.stage_label)
        
        # Model history
        history_layout = QHBoxLayout()
        history_layout.addWidget(QLabel("Model:"))
        self.model_history_combo = QComboBox()
        self.model_history_combo.addItem("Initial model")
        self.model_history_combo.currentIndexChanged.connect(self.modelHistoryChanged.emit)
        history_layout.addWidget(self.model_history_combo)
        workflow_layout.addLayout(history_layout)
        
        workflow_group.setLayout(workflow_layout)
        return workflow_group
    
    def _create_progress_group(self) -> QGroupBox:
        """Create progress group."""
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        progress_layout.addWidget(self.progress_label)
        
        self.stats_label = QLabel("No session")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_label.setStyleSheet("font-size: 10px;")
        progress_layout.addWidget(self.stats_label)
        
        progress_group.setLayout(progress_layout)
        return progress_group
    
    def _create_requirements_group(self) -> QGroupBox:
        """Create requirements status group."""
        requirements_group = QGroupBox("Requirements")
        requirements_layout = QVBoxLayout()
        
        self.model_status_label = QLabel("❌ Model not loaded")
        self.model_status_label.setStyleSheet("color: #ff6b6b;")
        requirements_layout.addWidget(self.model_status_label)
        
        requirements_group.setLayout(requirements_layout)
        return requirements_group
    
    def _create_thresholds_group(self) -> QGroupBox:
        """Create confidence thresholds group."""
        thresholds_group = QGroupBox()
        thresholds_layout = QVBoxLayout()
        
        # Custom title layout with info icon
        title_layout = QHBoxLayout()
        title_label = QLabel("Confidence Thresholds")
        title_label.setStyleSheet("font-weight: bold; color: #14ffec;")
        title_layout.addWidget(title_label)
        
        info_label = InfoLabel(
            "ℹ", 
            "Confidence thresholds categorize detected annotations:\n\n"
            "• Above High threshold → Auto-approved (green)\n"
            "  Highly confident detections that are automatically accepted\n\n"
            "• Between Medium and High → Requires Review (yellow)\n"
            "  Moderately confident detections that need human verification\n\n"
            "• Below Medium threshold → Rejected (red)\n"
            "  Low confidence detections that are automatically rejected\n\n"
            "Adjust these thresholds based on your quality requirements."
        )
        info_label.setStyleSheet("color: #14ffec; font-weight: bold; font-size: 14px;")
        title_layout.addWidget(info_label)
        title_layout.addStretch()
        thresholds_layout.addLayout(title_layout)
        
        # High confidence
        high_layout = QHBoxLayout()
        high_layout.addWidget(QLabel("High:"))
        self.high_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.high_conf_slider.setRange(50, 100)
        self.high_conf_slider.setValue(85)
        self.high_conf_slider.valueChanged.connect(self._update_threshold_labels)
        high_layout.addWidget(self.high_conf_slider)
        self.high_conf_label = QLabel("0.80")
        self.high_conf_label.setFixedWidth(40)
        high_layout.addWidget(self.high_conf_label)
        thresholds_layout.addLayout(high_layout)
        
        # Medium confidence
        med_layout = QHBoxLayout()
        med_layout.addWidget(QLabel("Medium:"))
        self.med_conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.med_conf_slider.setRange(10, 80)
        self.med_conf_slider.setValue(40)
        self.med_conf_slider.valueChanged.connect(self._update_threshold_labels)
        med_layout.addWidget(self.med_conf_slider)
        self.med_conf_label = QLabel("0.40")
        self.med_conf_label.setFixedWidth(40)
        med_layout.addWidget(self.med_conf_label)
        thresholds_layout.addLayout(med_layout)
        
        # Add refresh button
        refresh_layout = QHBoxLayout()
        refresh_layout.addStretch()
        self.refresh_thresholds_btn = QPushButton("Apply Thresholds")
        self.refresh_thresholds_btn.setToolTip("Re-categorize all detected annotations with the new threshold values")
        self.refresh_thresholds_btn.clicked.connect(self.refreshThresholds.emit)
        self.refresh_thresholds_btn.setEnabled(False)  # Disabled until we have results
        refresh_layout.addWidget(self.refresh_thresholds_btn)
        thresholds_layout.addLayout(refresh_layout)
        
        thresholds_group.setLayout(thresholds_layout)
        return thresholds_group
    
    def _create_augmentation_group(self) -> QGroupBox:
        """Create augmentation settings group."""
        augmentation_group = QGroupBox("Augmentation Options")
        augmentation_layout = QVBoxLayout()
        
        # Enable/disable augmentation checkbox
        self.enable_augmentation_checkbox = QCheckBox("Enable augmentation during inference")
        self.enable_augmentation_checkbox.setToolTip(
            "Apply augmentations to images during auto-annotation inference.\n"
            "This can help detect objects at different angles, scales, and conditions.\n"
            "Note: Augmentation is applied only during inference, not saved to images."
        )
        self.enable_augmentation_checkbox.setChecked(True)
        self.enable_augmentation_checkbox.toggled.connect(self.augmentationToggled.emit)
        augmentation_layout.addWidget(self.enable_augmentation_checkbox)
        
        # Augmentation settings widget (initially visible since checkbox is on)
        self.augmentation_settings = AugmentationSettings()
        self.augmentation_settings.setVisible(True)
        
        # Set custom default values
        custom_defaults = {
            'mosaic': 0.0,
            'flipud': 0.5,
            'perspective': 0.1,
            'shear': 0.05,
            'degrees': 45
        }
        self.augmentation_settings.set_settings(custom_defaults)
        
        # Wrap in scroll area for better space management
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.augmentation_settings)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)
        scroll_area.setVisible(True)  # Visible by default since augmentation is on
        self.augmentation_scroll = scroll_area
        
        augmentation_layout.addWidget(scroll_area)
        
        augmentation_group.setLayout(augmentation_layout)
        return augmentation_group
    
    def _create_dataset_group(self) -> QGroupBox:
        """Create dataset management group."""
        dataset_group = QGroupBox("Dataset Management")
        dataset_layout = QVBoxLayout()
        
        # Dataset info
        self.dataset_info_label = QLabel("No dataset loaded")
        self.dataset_info_label.setStyleSheet("font-size: 10px; color: #888888;")
        self.dataset_info_label.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_info_label)
        
        # Add note about workflow
        workflow_note = QLabel("Export saves .txt files next to images.\nSplit COPIES all annotated images (originals remain).")
        workflow_note.setStyleSheet("font-size: 9px; color: #666666; font-style: italic;")
        workflow_note.setWordWrap(True)
        dataset_layout.addWidget(workflow_note)
        
        # Dataset split controls
        split_layout = QVBoxLayout()
        split_label = QLabel("Dataset Split:")
        split_label.setToolTip("Stratified splitting ensures each split (train/val/test) has similar class distributions")
        split_layout.addWidget(split_label)
        
        # Train percentage
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("Train:"))
        self.train_split_spin = QSpinBox()
        self.train_split_spin.setRange(0, 100)
        self.train_split_spin.setValue(80)
        self.train_split_spin.setSuffix("%")
        self.train_split_spin.valueChanged.connect(self.splitPercentageChanged.emit)
        train_layout.addWidget(self.train_split_spin)
        split_layout.addLayout(train_layout)
        
        # Val percentage
        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Val:"))
        self.val_split_spin = QSpinBox()
        self.val_split_spin.setRange(0, 100)
        self.val_split_spin.setValue(20)
        self.val_split_spin.setSuffix("%")
        self.val_split_spin.valueChanged.connect(self.splitPercentageChanged.emit)
        val_layout.addWidget(self.val_split_spin)
        split_layout.addLayout(val_layout)
        
        # Test percentage (disabled)
        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("Test:"))
        self.test_split_spin = QSpinBox()
        self.test_split_spin.setRange(0, 100)
        self.test_split_spin.setValue(0)
        self.test_split_spin.setSuffix("%")
        self.test_split_spin.setEnabled(False)
        test_layout.addWidget(self.test_split_spin)
        split_layout.addLayout(test_layout)
        
        dataset_layout.addLayout(split_layout)
        
        dataset_group.setLayout(dataset_layout)
        return dataset_group
    
    def _create_training_group(self) -> QGroupBox:
        """Create training settings group."""
        training_group = QGroupBox("Quick Training")
        training_layout = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.training_model_combo = QComboBox()
        self.training_model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        model_layout.addWidget(self.training_model_combo)
        training_layout.addLayout(model_layout)
        
        # Training parameters in a grid
        params_layout = QVBoxLayout()
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 300)
        self.epochs_spin.setValue(50)
        epochs_layout.addWidget(self.epochs_spin)
        params_layout.addLayout(epochs_layout)
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        batch_layout.addWidget(self.batch_spin)
        params_layout.addLayout(batch_layout)
        
        # Image size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Image Size:"))
        self.image_size_combo = QComboBox()
        self.image_size_combo.addItems(["320", "416", "512", "640", "768", "1024", "1280"])
        self.image_size_combo.setCurrentText("640")
        size_layout.addWidget(self.image_size_combo)
        params_layout.addLayout(size_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        lr_layout.addWidget(self.lr_spin)
        params_layout.addLayout(lr_layout)
        
        training_layout.addLayout(params_layout)
        
        # Training control buttons
        training_btn_layout = QHBoxLayout()
        self.start_training_btn = QPushButton("Start Training")
        self.start_training_btn.clicked.connect(self.startTraining.emit)
        self.start_training_btn.setEnabled(False)
        self.start_training_btn.setMaximumWidth(120)
        training_btn_layout.addWidget(self.start_training_btn)
        
        self.stop_training_btn = QPushButton("Stop")
        self.stop_training_btn.clicked.connect(self.stopTraining.emit)
        self.stop_training_btn.setEnabled(False)
        self.stop_training_btn.setMaximumWidth(140)
        training_btn_layout.addWidget(self.stop_training_btn)
        
        training_layout.addLayout(training_btn_layout)
        
        # Training progress
        self.training_progress_bar = QProgressBar()
        self.training_progress_bar.setVisible(False)
        training_layout.addWidget(self.training_progress_bar)
        
        self.training_status_label = QLabel("")
        self.training_status_label.setStyleSheet("font-size: 10px;")
        self.training_status_label.setVisible(False)
        training_layout.addWidget(self.training_status_label)
        
        # Training console output
        self.training_console = QTextEdit()
        self.training_console.setReadOnly(True)
        self.training_console.setMaximumHeight(100)
        self.training_console.setStyleSheet("font-family: monospace; font-size: 9px; background-color: #1e1e1e;")
        self.training_console.setPlaceholderText("Training output will appear here...")
        self.training_console.setVisible(False)
        training_layout.addWidget(self.training_console)
        
        training_group.setLayout(training_layout)
        return training_group
    
    def _create_actions_group(self) -> QGroupBox:
        """Create actions group."""
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        # Batch operations
        self.approve_selected_btn = QPushButton("Approve Selected (A)")
        self.approve_selected_btn.setToolTip("Approve annotations for all selected images (A key)")
        self.approve_selected_btn.clicked.connect(self.approveSelected.emit)
        actions_layout.addWidget(self.approve_selected_btn)
        
        self.reject_selected_btn = QPushButton("Reject Selected (R)")
        self.reject_selected_btn.setToolTip("Reject annotations for all selected images (R key)")
        self.reject_selected_btn.clicked.connect(self.rejectSelected.emit)
        actions_layout.addWidget(self.reject_selected_btn)
        
        # Quality control
        self.quality_btn = QPushButton("Quality Assessment")
        self.quality_btn.clicked.connect(self.qualityAssessment.emit)
        actions_layout.addWidget(self.quality_btn)
        
        # Active learning
        self.active_learning_btn = QPushButton("Active Learning")
        self.active_learning_btn.clicked.connect(self.activeLearning.emit)
        actions_layout.addWidget(self.active_learning_btn)
        
        # Add separator
        actions_layout.addSpacing(10)
        
        # Move to rejected folder
        self.move_to_rejected_btn = QPushButton("❌ Move Selected to Rejected Folder")
        self.move_to_rejected_btn.setToolTip("Move selected images to 'rejected' folder (DEL key)")
        self.move_to_rejected_btn.clicked.connect(self.moveToRejected.emit)
        self.move_to_rejected_btn.setStyleSheet("""
            QPushButton {
                background-color: #c9302c;
                color: white;
            }
            QPushButton:hover {
                background-color: #d9413c;
            }
            QPushButton:pressed {
                background-color: #a02622;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """)
        actions_layout.addWidget(self.move_to_rejected_btn)
        
        actions_group.setLayout(actions_layout)
        return actions_group
    
    def create_gallery_panel(self) -> QWidget:
        """Create the thumbnail gallery panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Gallery header with selection controls
        header_layout = QHBoxLayout()
        self.gallery_label = QLabel("Images")
        header_layout.addWidget(self.gallery_label)
        
        header_layout.addStretch()
        
        # Expand/Retract button
        self.expand_gallery_btn = QPushButton("Expand Gallery")
        self.expand_gallery_btn.clicked.connect(self.expandGallery.emit)
        self.expand_gallery_btn.setMaximumWidth(100)
        self.expand_gallery_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a7c8a;
                color: white;
                padding: 4px 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #6a8c9a;
            }
            QPushButton:pressed {
                background-color: #4a6c7a;
            }
        """)
        header_layout.addWidget(self.expand_gallery_btn)
        
        # Selection controls
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.selectAllThumbnails.emit)
        self.select_all_btn.setMaximumWidth(80)
        self.select_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a7c59;
                color: white;
                padding: 4px 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #5a8d69;
            }
            QPushButton:pressed {
                background-color: #3a6c49;
            }
        """)
        header_layout.addWidget(self.select_all_btn)
        
        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self.selectNoneThumbnails.emit)
        self.select_none_btn.setMaximumWidth(80)
        self.select_none_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                padding: 4px 8px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:pressed {
                background-color: #444444;
            }
        """)
        header_layout.addWidget(self.select_none_btn)
        
        layout.addLayout(header_layout)
        
        # Category filter dropdown
        category_filter_layout = QHBoxLayout()
        category_filter_label = QLabel("Filter by category:")
        category_filter_layout.addWidget(category_filter_label)
        
        self.category_filter_combo = QComboBox()
        self.category_filter_combo.setMinimumWidth(200)
        self.category_filter_combo.setMaximumWidth(300)
        self.category_filter_combo.setEditable(False)
        
        # Create a custom widget for the dropdown to support checkboxes
        self.category_list_widget = QListWidget()
        self.category_list_widget.setStyleSheet("""
            QListWidget {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                color: #cccccc;
            }
            QListWidget::item {
                padding: 5px;
            }
            QListWidget::item:hover {
                background-color: #555555;
            }
        """)
        
        # Add "Select All" and "Select None" items (not checkable)
        select_all_item = QListWidgetItem("Select All")
        select_all_item.setData(Qt.ItemDataRole.UserRole, "select_all")
        self.category_list_widget.addItem(select_all_item)
        
        select_none_item = QListWidgetItem("Select None")
        select_none_item.setData(Qt.ItemDataRole.UserRole, "select_none")
        self.category_list_widget.addItem(select_none_item)
        
        # Add separator
        separator_item = QListWidgetItem("─────────────")
        separator_item.setFlags(Qt.ItemFlag.NoItemFlags)
        self.category_list_widget.addItem(separator_item)
        
        self.category_filter_combo.setModel(self.category_list_widget.model())
        self.category_filter_combo.setView(self.category_list_widget)
        self.category_filter_combo.setCurrentText("All categories")
        
        # Connect list widget item clicks and changes
        self.category_list_widget.itemClicked.connect(self.categoryFilterClicked.emit)
        self.category_list_widget.itemChanged.connect(self.categoryFilterChanged.emit)
        
        category_filter_layout.addWidget(self.category_filter_combo)
        category_filter_layout.addStretch()
        layout.addLayout(category_filter_layout)
        
        # Confidence filter buttons
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Confidence filters:")
        filter_layout.addWidget(filter_label)
        
        # Style for filter buttons
        filter_button_style = """
            QPushButton {
                background-color: #444444;
                color: #cccccc;
                border: 1px solid #555555;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #555555;
                border: 1px solid #666666;
            }
            QPushButton:checked {
                background-color: #14ffec;
                color: #000000;
                border: 2px solid #0affdc;
                font-weight: bold;
            }
        """
        
        self.filter_review_btn = QPushButton("Review")
        self.filter_review_btn.setCheckable(True)
        self.filter_review_btn.setStyleSheet(filter_button_style)
        self.filter_review_btn.clicked.connect(
            lambda: self.filterByCategory.emit(ConfidenceCategory.REQUIRES_REVIEW)
        )
        self.filter_review_btn.setMaximumWidth(80)
        filter_layout.addWidget(self.filter_review_btn)
        
        self.filter_approved_btn = QPushButton("Approved")
        self.filter_approved_btn.setCheckable(True)
        self.filter_approved_btn.setStyleSheet(filter_button_style)
        self.filter_approved_btn.clicked.connect(
            lambda: self.filterByCategory.emit(ConfidenceCategory.AUTO_APPROVED)
        )
        self.filter_approved_btn.setMaximumWidth(80)
        filter_layout.addWidget(self.filter_approved_btn)
        
        self.filter_rejected_btn = QPushButton("Rejected")
        self.filter_rejected_btn.setCheckable(True)
        self.filter_rejected_btn.setStyleSheet(filter_button_style)
        self.filter_rejected_btn.clicked.connect(
            lambda: self.filterByCategory.emit(ConfidenceCategory.REJECTED)
        )
        self.filter_rejected_btn.setMaximumWidth(80)
        filter_layout.addWidget(self.filter_rejected_btn)
        
        self.filter_no_detections_btn = QPushButton("No Detections")
        self.filter_no_detections_btn.setCheckable(True)
        self.filter_no_detections_btn.setStyleSheet(filter_button_style)
        self.filter_no_detections_btn.clicked.connect(
            lambda: self.filterByCategory.emit(ConfidenceCategory.NO_DETECTIONS)
        )
        self.filter_no_detections_btn.setMaximumWidth(100)
        filter_layout.addWidget(self.filter_no_detections_btn)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # Create stacked widget for gallery and charts
        self.center_stack = QStackedWidget()
        
        # Page 1: Thumbnail gallery
        self.gallery = ThumbnailGallery()
        self.center_stack.addWidget(self.gallery)
        
        # Page 2: Training charts
        self.training_charts = TrainingCharts()
        self.center_stack.addWidget(self.training_charts)
        
        # Default to gallery view
        self.center_stack.setCurrentIndex(0)
        
        layout.addWidget(self.center_stack)
        
        return panel
    
    def create_editor_panel(self) -> QWidget:
        """Create the annotation editor panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Editor header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        
        self.annotation_count_label = QLabel("Annotations: 0")
        self.annotation_count_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.annotation_count_label)
        
        header_layout.addStretch()
        
        # Class selection with proper spacing
        class_layout = QHBoxLayout()
        class_layout.setSpacing(5)
        class_label = QLabel("Class:")
        class_layout.addWidget(class_label)
        
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self.classChanged.emit)
        class_layout.addWidget(self.class_combo)
        header_layout.addLayout(class_layout)
        
        
        layout.addLayout(header_layout)
        
        # Drawing tools hint
        hint_label = QLabel("✏️ Left-click drag to draw | 🗑️ DEL to delete | 🔢 0-9 keys to change class")
        hint_label.setStyleSheet("color: #888888; font-size: 12px;")
        layout.addWidget(hint_label)
        
        # Annotation canvas
        self.annotation_canvas = AnnotationCanvas()
        layout.addWidget(self.annotation_canvas)
        
        return panel
    
    def _update_threshold_labels(self):
        """Update threshold value labels."""
        self.high_conf_label.setText(f"{self.high_conf_slider.value() / 100:.2f}")
        self.med_conf_label.setText(f"{self.med_conf_slider.value() / 100:.2f}")
        self.thresholdChanged.emit()
    
    def update_threshold_labels(self, high_value: float, med_value: float):
        """Update threshold labels with specific values."""
        self.high_conf_label.setText(f"{high_value:.2f}")
        self.med_conf_label.setText(f"{med_value:.2f}")
    
    def get_threshold_values(self) -> Tuple[float, float]:
        """Get current threshold values."""
        return (
            self.high_conf_slider.value() / 100,
            self.med_conf_slider.value() / 100
        )
    
    def get_split_percentages(self) -> Tuple[int, int, int]:
        """Get dataset split percentages."""
        return (
            self.train_split_spin.value(),
            self.val_split_spin.value(),
            self.test_split_spin.value()
        )
    
    def get_training_config(self) -> Dict:
        """Get training configuration from UI."""
        # Auto-detect best available device
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
            
        return {
            'model': self.training_model_combo.currentText(),
            'epochs': self.epochs_spin.value(),
            'batch': self.batch_spin.value(),
            'imgsz': int(self.image_size_combo.currentText()),
            'lr0': self.lr_spin.value(),
            'device': device
        }
    
    def get_augmentation_settings(self) -> Dict:
        """Get augmentation settings if enabled."""
        if self.enable_augmentation_checkbox.isChecked():
            return self.augmentation_settings.get_settings()
        return {}
    
    def show_augmentation_settings(self, visible: bool):
        """Show or hide augmentation settings."""
        self.augmentation_settings.setVisible(visible)
        self.augmentation_scroll.setVisible(visible)