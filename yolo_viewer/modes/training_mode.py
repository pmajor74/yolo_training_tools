"""Training mode for YOLO model training with real-time monitoring."""

from pathlib import Path
from typing import Optional, Dict, List
import yaml
import json
import sys
import os
import math
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit,
    QTextEdit, QSplitter, QFileDialog, QMessageBox, QProgressBar,
    QCheckBox, QRadioButton, QButtonGroup, QScrollArea, QFrame,
    QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor

from .base_mode import BaseMode
from ..core import ModelCache, SettingsManager, DatasetManager
from ..core.constants import IMAGE_EXTENSIONS
from ..utils.training_process import TrainingProcess
from ..widgets.augmentation_settings import AugmentationSettings
from ..widgets.training_charts import TrainingCharts
from ..widgets.training_results import TrainingResults
from ..utils.tif_converter import TifFormatChecker


class TrainingMode(BaseMode):
    """Training mode for YOLO model training."""
    
    # Custom signals
    trainingStarted = pyqtSignal(str)  # config path
    trainingProgress = pyqtSignal(int, int, dict)  # epoch, total_epochs, metrics
    trainingCompleted = pyqtSignal(str)  # model path
    trainingFailed = pyqtSignal(str)  # error message
    
    def __init__(self, parent=None):
        # Initialize attributes before super().__init__()
        self._training_process: Optional[TrainingProcess] = None
        self._is_training = False
        self._current_epoch = 0
        self._total_epochs = 100
        self._current_step = 0
        self._total_steps = 0
        self._steps_per_epoch = 0
        self._training_start_time: Optional[datetime] = None
        self._dataset_path: Optional[Path] = None
        self._output_dir: Optional[Path] = None
        self._num_train_images = 0
        self._dataset_manager = DatasetManager()
        
        super().__init__(parent)
        
        # Connect to dataset manager signals
        self._dataset_manager.datasetLoaded.connect(self._on_dataset_manager_loaded)
    
    def _setup_ui(self):
        """Setup the UI for training mode."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Configuration (make it scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Dataset selection
        dataset_group = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout()
        
        dataset_path_layout = QHBoxLayout()
        self.dataset_label = QLabel("No dataset selected - Click 'Load Dataset' to select a data.yaml file")
        self.dataset_label.setWordWrap(True)
        dataset_path_layout.addWidget(self.dataset_label, 1)
        
        self.select_dataset_btn = QPushButton("Load Dataset")
        self.select_dataset_btn.clicked.connect(self._select_dataset)
        dataset_path_layout.addWidget(self.select_dataset_btn)
        
        dataset_layout.addLayout(dataset_path_layout)
        
        # Dataset info display
        self.dataset_info_label = QLabel("")
        self.dataset_info_label.setWordWrap(True)
        self.dataset_info_label.setVisible(False)
        dataset_layout.addWidget(self.dataset_info_label)
        
        dataset_group.setLayout(dataset_layout)
        left_layout.addWidget(dataset_group)
        
        # Model selection
        model_group = QGroupBox("Pre-trained Model")
        model_layout = QVBoxLayout()
        
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Base Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt",
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"
        ])
        self.model_combo.setCurrentText("yolov8n.pt")
        model_select_layout.addWidget(self.model_combo)
        model_select_layout.addStretch()
        
        model_layout.addLayout(model_select_layout)
        
        # Model size info
        self.model_info_label = QLabel("n: Nano (fastest) | s: Small | m: Medium | l: Large | x: X-Large (most accurate)")
        model_layout.addWidget(self.model_info_label)
        
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QVBoxLayout()
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        epochs_layout.addWidget(self.epochs_spin)
        epochs_layout.addStretch()
        params_layout.addLayout(epochs_layout)
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addStretch()
        params_layout.addLayout(batch_layout)
        
        # Image size
        imgsz_layout = QHBoxLayout()
        imgsz_layout.addWidget(QLabel("Image Size:"))
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(["320", "640", "1280", "1920"])
        self.imgsz_combo.setCurrentText("640")
        self.imgsz_combo.setEditable(True)
        imgsz_layout.addWidget(self.imgsz_combo)
        imgsz_layout.addStretch()
        params_layout.addLayout(imgsz_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(4)
        lr_layout.addWidget(self.lr_spin)
        lr_layout.addStretch()
        params_layout.addLayout(lr_layout)
        
        # Patience
        patience_layout = QHBoxLayout()
        patience_layout.addWidget(QLabel("Patience:"))
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 100)
        self.patience_spin.setValue(20)
        patience_layout.addWidget(self.patience_spin)
        patience_layout.addStretch()
        params_layout.addLayout(patience_layout)
        
        # Add spacing before buttons
        params_layout.addSpacing(10)
        
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self._start_training)        
        self.start_btn.setMaximumWidth(120)
        controls_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMaximumWidth(140)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        
        params_layout.addLayout(controls_layout)
        
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout()
        
        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Training Device:"))
        self.device_label = QLabel("Detecting...")
        self.device_label.setMinimumWidth(200)
        device_layout.addWidget(self.device_label)
        device_layout.addStretch()
        advanced_layout.addLayout(device_layout)
        
        self.pretrained_check = QCheckBox("Use Pre-trained Weights")
        self.pretrained_check.setChecked(True)
        advanced_layout.addWidget(self.pretrained_check)
        
        self.augment_check = QCheckBox("Enable Data Augmentation")
        self.augment_check.setChecked(False)
        self.augment_check.setToolTip(
            "Apply augmentations to images during training.\n"
            "This helps the model generalize better by seeing variations of the training data."
        )
        self.augment_check.toggled.connect(self._on_augmentation_toggled)
        advanced_layout.addWidget(self.augment_check)
        
        self.cache_check = QCheckBox("Cache Images")
        self.cache_check.setChecked(True)
        advanced_layout.addWidget(self.cache_check)
        
        self.export_onnx_check = QCheckBox("Export to ONNX")
        self.export_onnx_check.setChecked(False)
        advanced_layout.addWidget(self.export_onnx_check)
        
        advanced_group.setLayout(advanced_layout)
        left_layout.addWidget(advanced_group)
        
        # Augmentation settings (initially hidden)
        augment_group = QGroupBox("Augmentation Settings")
        augment_layout = QVBoxLayout()
        
        self.augment_settings = AugmentationSettings()
        augment_layout.addWidget(self.augment_settings)
        
        augment_group.setLayout(augment_layout)
        augment_group.setVisible(False)
        self.augment_group = augment_group
        left_layout.addWidget(augment_group)
        
        left_layout.addStretch()
        
        # Set left widget as scroll area content
        left_scroll.setWidget(left_widget)
        
        # Right panel - Progress and logs
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Progress section
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        # Overall progress
        overall_layout = QHBoxLayout()
        overall_layout.addWidget(QLabel("Overall:"))
        self.overall_progress = QProgressBar()
        overall_layout.addWidget(self.overall_progress)
        self.step_label = QLabel("Epoch 0/0 • 0/0 steps")
        self.step_label.setMinimumWidth(200)
        overall_layout.addWidget(self.step_label)
        progress_layout.addLayout(overall_layout)
        
        # Current epoch progress
        epoch_layout = QHBoxLayout()
        epoch_layout.addWidget(QLabel("Current Epoch:"))
        self.epoch_progress = QProgressBar()
        epoch_layout.addWidget(self.epoch_progress)
        progress_layout.addLayout(epoch_layout)
        
        # Time estimate
        self.time_label = QLabel("Time: --:--:-- elapsed, --:--:-- remaining")
        progress_layout.addWidget(self.time_label)
        
        # Metrics display
        self.metrics_label = QLabel("Loss: --, mAP: --")
        self.metrics_label.setFont(QFont("Courier", 10))
        progress_layout.addWidget(self.metrics_label)
        
        progress_group.setLayout(progress_layout)
        right_layout.addWidget(progress_group)
        
        # Tab widget for log and charts
        self.output_tabs = QTabWidget()

        
        # Training log tab
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(5, 5, 5, 5)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_text)
        
        self.output_tabs.addTab(log_widget, "📄 Training Log")
        
        # Charts tab
        self.charts_widget = TrainingCharts()
        self.output_tabs.addTab(self.charts_widget, "📊 Charts")
        
        # Results tab
        self.results_widget = TrainingResults()
        self.output_tabs.addTab(self.results_widget, "📁 Results")
        
        right_layout.addWidget(self.output_tabs)
        
        # Add panels to splitter
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])
        
        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_progress)
        
        # Detect and display device
        self._update_device_label()
        
    @pyqtSlot()
    def _select_dataset(self):
        """Select dataset for training."""
        # Get the directory from last dataset path
        last_path = SettingsManager().get_last_dataset_path()
        start_dir = str(Path(last_path).parent) if last_path else ""
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset Configuration (data.yaml)",
            start_dir,
            "YAML Files (*.yaml *.yml)"
        )
        
        if file_path:
            self._dataset_path = Path(file_path)
            self.dataset_label.setText(str(self._dataset_path.name))
            self.dataset_label.setToolTip(str(self._dataset_path))
            SettingsManager().set_last_dataset_path(str(self._dataset_path))
            
            # Parse and display dataset info
            self._parse_and_display_dataset_info()
            
            # Try to count training images
            self._count_training_images()
    
    @pyqtSlot(bool)
    def _on_augmentation_toggled(self, checked: bool):
        """Handle augmentation enable/disable toggle."""
        self.augment_group.setVisible(checked)
    
    def _parse_and_display_dataset_info(self):
        """Parse and display dataset information from the YAML file."""
        try:
            with open(self._dataset_path, 'r') as f:
                data = yaml.safe_load(f)
            
            info_parts = []
            
            # Number of classes
            if 'nc' in data:
                info_parts.append(f"Classes: {data['nc']}")
            
            # Class names
            if 'names' in data:
                names = data['names']
                if isinstance(names, dict):
                    # Convert dict to list format
                    class_names = [names.get(i, f"class_{i}") for i in range(len(names))]
                elif isinstance(names, list):
                    class_names = names
                else:
                    class_names = []
                
                if class_names:
                    names_str = ", ".join(class_names[:5])  # Show first 5 classes
                    if len(class_names) > 5:
                        names_str += f", ... ({len(class_names) - 5} more)"
                    info_parts.append(f"Class Names: {names_str}")
            
            # Dataset paths
            paths_info = []
            for split in ['train', 'val', 'test']:
                if split in data:
                    if isinstance(data[split], list):
                        paths_info.append(f"{split.capitalize()}: {len(data[split])} paths")
                    else:
                        paths_info.append(f"{split.capitalize()}: ✓")
            
            if paths_info:
                info_parts.append("Splits: " + ", ".join(paths_info))
            
            # Base path
            if 'path' in data:
                info_parts.append(f"Base Path: {data['path']}")
            
            # Display the info
            if info_parts:
                self.dataset_info_label.setText("\n".join(info_parts))
                self.dataset_info_label.setVisible(True)
            else:
                self.dataset_info_label.setVisible(False)
                
        except Exception as e:
            self._log(f"Error parsing dataset YAML: {e}")
            self.dataset_info_label.setText(f"Error reading dataset: {str(e)}")
            self.dataset_info_label.setVisible(True)
    
    def _count_training_images(self):
        """Count the number of training images in the dataset."""
        try:
            print(f"[DEBUG] _count_training_images: Reading YAML from: {self._dataset_path}")
            with open(self._dataset_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Get the base path
            yaml_dir = self._dataset_path.parent
            print(f"[DEBUG] _count_training_images: YAML directory: {yaml_dir}")
            
            if 'path' in data and data['path']:
                # If path is absolute, use it; otherwise make it relative to yaml dir
                path_val = Path(data['path'])
                print(f"[DEBUG] _count_training_images: Path value from YAML: '{data['path']}'")
                
                if path_val.is_absolute():
                    base_path = path_val
                    print(f"[DEBUG] _count_training_images: Using absolute path: {base_path}")
                else:
                    base_path = yaml_dir / path_val
                    print(f"[DEBUG] _count_training_images: Using relative path: {base_path}")
            else:
                base_path = yaml_dir
                print(f"[DEBUG] _count_training_images: No path in YAML, using YAML directory: {base_path}")
            
            # Ensure base_path exists
            if not base_path.exists():
                print(f"[WARNING] _count_training_images: Base path doesn't exist: {base_path}")
                print(f"[WARNING] _count_training_images: Falling back to YAML directory: {yaml_dir}")
                base_path = yaml_dir
            
            # Get train path
            self._num_train_images = 0
            if 'train' in data:
                if isinstance(data['train'], str):
                    # Single train path
                    train_paths = [data['train']]
                elif isinstance(data['train'], list):
                    # Multiple train paths
                    train_paths = data['train']
                else:
                    return
                
                # Count images in each train path
                print(f"[DEBUG] _count_training_images: Train paths to check: {train_paths}")
                for train_path_str in train_paths:
                    # Build full path
                    if Path(train_path_str).is_absolute():
                        full_train_path = Path(train_path_str)
                        print(f"[DEBUG] _count_training_images: Absolute train path: {full_train_path}")
                    else:
                        full_train_path = base_path / train_path_str
                        print(f"[DEBUG] _count_training_images: Relative train path: {train_path_str} -> {full_train_path}")
                    
                    if full_train_path.exists():
                        print(f"[DEBUG] _count_training_images: Train path exists: {full_train_path}")
                        # First check if this is directly an images folder or contains an images subfolder
                        if full_train_path.name == 'images' or (full_train_path / 'images').exists():
                            # Images are in 'images' subdirectory
                            img_dir = full_train_path / 'images' if full_train_path.name != 'images' else full_train_path
                            print(f"[DEBUG] _count_training_images: Looking for images in 'images' subdirectory: {img_dir}")
                        else:
                            # Images are directly in the train directory
                            img_dir = full_train_path
                            print(f"[DEBUG] _count_training_images: Looking for images directly in: {img_dir}")
                        
                        # Count only image files (not .txt annotation files)
                        if img_dir.exists():
                            for file in img_dir.iterdir():
                                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                                    self._num_train_images += 1
                            print(f"[DEBUG] _count_training_images: Found {self._num_train_images} images in {img_dir}")
                        else:
                            print(f"[ERROR] _count_training_images: Image directory doesn't exist: {img_dir}")
                    else:
                        print(f"[ERROR] _count_training_images: Train path doesn't exist: {full_train_path}")
                
                # Update label with count
                if self._num_train_images > 0:
                    self.dataset_label.setText(f"📁 {self._dataset_path.name} ({self._num_train_images} training images)")
                else:
                    self.dataset_label.setText(f"📁 {self._dataset_path.name} (no images found)")
            
        except Exception as e:
            print(f"Error counting training images: {e}")
            self._num_train_images = 0
            self.dataset_label.setText(f"📁 {self._dataset_path.name} (count error)")
    
    @pyqtSlot()
    def _start_training(self):
        """Start the training process."""
        import traceback
        
        try:
            print("=== TRAINING START DEBUG ===")
            print(f"[INFO] TrainingMode._start_training: Method called")
            print(f"[INFO] TrainingMode._start_training: dataset_path = {self._dataset_path}")
            
            # Validate inputs
            if not self._dataset_path or not self._dataset_path.exists():
                print(f"[ERROR] TrainingMode._start_training: Dataset validation failed")
                print(f"[ERROR] TrainingMode._start_training: path={self._dataset_path}")
                print(f"[ERROR] TrainingMode._start_training: exists={self._dataset_path.exists() if self._dataset_path else 'None'}")
                QMessageBox.warning(self, "Warning", "Please select a valid dataset")
                return
            
            print(f"[INFO] TrainingMode._start_training: Dataset validation passed: {self._dataset_path}")
            
            # Get the actual dataset directory from the YAML file
            try:
                actual_dataset_path = self._get_dataset_directory_from_yaml()
                print(f"[INFO] TrainingMode._start_training: Actual dataset directory: {actual_dataset_path}")
            except Exception as e:
                print(f"[ERROR] TrainingMode._start_training: Failed to get dataset directory from YAML")
                print(f"[ERROR] TrainingMode._start_training: Exception: {type(e).__name__}: {e}")
                traceback.print_exc()
                self._log(f"Error getting dataset directory: {e}")
                QMessageBox.critical(self, "Error", f"Failed to read dataset configuration:\n{e}")
                return
        
            # Check and convert TIF files if needed
            tif_check_msg = f"Checking TIF file formats in dataset: {actual_dataset_path}"
            self._log(tif_check_msg)
            print(f"[INFO] TrainingMode._start_training: {tif_check_msg}")
            
            try:
                print(f"[INFO] TrainingMode._start_training: Calling TifFormatChecker.check_and_convert_if_needed...")
                conversion_result = TifFormatChecker.check_and_convert_if_needed(actual_dataset_path, self, self._log)
                print(f"[INFO] TrainingMode._start_training: TIF check result: {conversion_result}")
                
                if not conversion_result:
                    # User cancelled conversion or conversion failed
                    cancel_msg = "Training cancelled: TIF file conversion declined or failed"
                    self._log(cancel_msg)
                    print(f"[INFO] TrainingMode._start_training: {cancel_msg}")
                    QMessageBox.information(self, "Training Cancelled", 
                                          "Training cancelled. TIF files must be in RGB format for YOLO training.")
                    return
            except Exception as e:
                error_msg = f"Error during TIF format check: {e}"
                self._log(error_msg)
                print(f"[ERROR] TrainingMode._start_training: {error_msg}")
                print(f"[ERROR] TrainingMode._start_training: Exception type: {type(e).__name__}")
                print(f"[ERROR] TrainingMode._start_training: Exception details: {e}")
                traceback.print_exc()
                QMessageBox.warning(self, "Error", f"Error checking TIF files: {e}")
                return
            
            print(f"[INFO] TrainingMode._start_training: TIF check completed successfully, continuing with training...")
        
            # Check if model is loaded for training
            print(f"[INFO] TrainingMode._start_training: Checking model availability...")
            model_cache = ModelCache()
            if not self.pretrained_check.isChecked() and not model_cache.model:
                print(f"[ERROR] TrainingMode._start_training: No model loaded and pretrained not checked")
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    "Please load a model in Model Management tab or enable 'Use Pre-trained Weights'"
                )
                return
            
            # Create output directory first
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._output_dir = Path("runs/train") / f"train_{timestamp}"
                self._output_dir.mkdir(parents=True, exist_ok=True)
                print(f"[INFO] TrainingMode._start_training: Created output directory: {self._output_dir}")
            except Exception as e:
                print(f"[ERROR] TrainingMode._start_training: Failed to create output directory")
                print(f"[ERROR] TrainingMode._start_training: Exception: {type(e).__name__}: {e}")
                traceback.print_exc()
                self._log(f"Error creating output directory: {e}")
                QMessageBox.critical(self, "Error", f"Failed to create output directory:\n{e}")
                return
            
            # Prepare training configuration (now that output_dir exists)
            try:
                print(f"[INFO] TrainingMode._start_training: Preparing training configuration...")
                config = self._prepare_training_config()
                print(f"[INFO] TrainingMode._start_training: Configuration prepared: {config}")
            except Exception as e:
                print(f"[ERROR] TrainingMode._start_training: Failed to prepare training configuration")
                print(f"[ERROR] TrainingMode._start_training: Exception: {type(e).__name__}: {e}")
                traceback.print_exc()
                self._log(f"Error preparing training configuration: {e}")
                QMessageBox.critical(self, "Error", f"Failed to prepare training configuration:\n{e}")
                return
            
            # Save configuration
            try:
                config_path = self._output_dir / "training_config.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
                print(f"[INFO] TrainingMode._start_training: Saved configuration to: {config_path}")
            except Exception as e:
                print(f"[ERROR] TrainingMode._start_training: Failed to save configuration")
                print(f"[ERROR] TrainingMode._start_training: Exception: {type(e).__name__}: {e}")
                traceback.print_exc()
                self._log(f"Error saving configuration: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{e}")
                return
        
            # Update UI state
            print(f"[INFO] TrainingMode._start_training: Updating UI state...")
            self._is_training = True
            self._current_epoch = 0
            self._total_epochs = config['epochs']
            self._current_step = 0
            
            # Calculate total steps
            batch_size = config['batch']
            if self._num_train_images > 0:
                # Use ceiling division to match how YOLO calculates batches
                import math
                self._steps_per_epoch = math.ceil(self._num_train_images / batch_size)
                self._total_steps = self._steps_per_epoch * self._total_epochs
                # Log the calculation for debugging
                self._log(f"[DEBUG] Initial step calculation: {self._num_train_images} images / {batch_size} batch size = {self._steps_per_epoch} steps/epoch")
                self._log(f"[DEBUG] Total steps: {self._steps_per_epoch} × {self._total_epochs} epochs = {self._total_steps}")
                print(f"[INFO] TrainingMode._start_training: Steps calculation: {self._num_train_images} images / {batch_size} batch = {self._steps_per_epoch} steps/epoch")
            else:
                # Estimate if we couldn't count images
                self._steps_per_epoch = 100  # Default estimate
                self._total_steps = self._steps_per_epoch * self._total_epochs
                self._log(f"[DEBUG] Using estimated steps: {self._steps_per_epoch} steps/epoch × {self._total_epochs} epochs = {self._total_steps}")
                print(f"[INFO] TrainingMode._start_training: Using estimated steps: {self._steps_per_epoch} steps/epoch")
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self._training_start_time = datetime.now()
            
            # Clear log and reset progress
            self.log_text.clear()
            self.overall_progress.setValue(0)
            self.overall_progress.setMaximum(self._total_steps)
            self.epoch_progress.setValue(0) 
            self.epoch_progress.setMaximum(100)
            self.step_label.setText(f"Epoch 0/{self._total_epochs} • 0/{self._total_steps} steps")
            self.metrics_label.setText("Loss: --, mAP: --")
            
            # Clear and start charts
            self.charts_widget.clear_data()
            self.charts_widget.start_monitoring()
            
            # Set output directory for results widget (delay to avoid conflicts)
            # The results will be loaded when training completes
            # self.results_widget.set_output_directory(self._output_dir)
            
            # Automatically switch to Charts tab
            self.output_tabs.setCurrentIndex(1)  # Charts is the second tab (index 1)
            
            self._log("=" * 60)
            self._log("Training started at " + self._training_start_time.strftime("%Y-%m-%d %H:%M:%S"))
            self._log(f"Output directory: {self._output_dir}")
            
            # Log device information
            device = config['device']
            device_name = "CPU"
            if device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        device_name = f"GPU (CUDA) - {torch.cuda.get_device_name(0)}"
                except:
                    device_name = "GPU (CUDA)"
            elif device == "mps":
                device_name = "GPU (Apple Silicon)"
            self._log(f"Training device: {device_name}")
            
            if self._num_train_images > 0:
                self._log(f"Dataset: {self._num_train_images} images, batch size: {batch_size}")
                self._log(f"Steps per epoch: {self._num_train_images} ÷ {batch_size} = {self._steps_per_epoch}")
                self._log(f"Total training steps: {self._total_steps} ({self._steps_per_epoch} steps/epoch × {self._total_epochs} epochs)")
            self._log("=" * 60)
            
            # Start update timer
            self.update_timer.start(1000)  # Update every second
        
            # Create and start training process
            try:
                print(f"[INFO] TrainingMode._start_training: Creating TrainingProcess...")
                self._training_process = TrainingProcess()
                self._training_process.logMessage.connect(self._log)
                self._training_process.progressUpdate.connect(self._on_progress_update)
                self._training_process.epochProgress.connect(self._on_epoch_progress)
                self._training_process.batchProgress.connect(self._on_batch_progress)
                self._training_process.metricsUpdate.connect(self._on_metrics_update)
                self._training_process.trainingCompleted.connect(self._on_training_completed)
                self._training_process.trainingFailed.connect(self._on_training_failed)
                self._training_process.stepInfoDetected.connect(self._on_steps_detected)
                self._training_process.trainingStopped.connect(self._on_training_stopped)
                
                # Connect charts
                self._training_process.progressUpdate.connect(self.charts_widget.on_epoch_update)
                
                print(f"[INFO] TrainingMode._start_training: Starting training process...")
                print(f"[INFO] TrainingMode._start_training: Config: {config}")
                print(f"[INFO] TrainingMode._start_training: Output dir: {self._output_dir}")
                print(f"[INFO] TrainingMode._start_training: Export ONNX: {self.export_onnx_check.isChecked()}")
                
                self._training_process.start_training(config, self._output_dir, self.export_onnx_check.isChecked())
                
                # Emit signal
                self.trainingStarted.emit(str(config_path))
                print(f"[INFO] TrainingMode._start_training: Training started successfully")
                
            except Exception as e:
                print(f"[ERROR] TrainingMode._start_training: Failed to start training process")
                print(f"[ERROR] TrainingMode._start_training: Exception: {type(e).__name__}: {e}")
                traceback.print_exc()
                self._log(f"Error starting training process: {e}")
                
                # Reset UI state on error
                self._is_training = False
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.update_timer.stop()
                
                QMessageBox.critical(self, "Error", f"Failed to start training process:\n{e}")
                return
                
        except Exception as e:
            # Catch-all for any unexpected exceptions
            print(f"[ERROR] TrainingMode._start_training: Unexpected error in _start_training")
            print(f"[ERROR] TrainingMode._start_training: Exception: {type(e).__name__}: {e}")
            print(f"[ERROR] TrainingMode._start_training: Full exception details:")
            traceback.print_exc()
            
            # Log to UI
            self._log(f"Unexpected error: {e}")
            
            # Reset UI state
            try:
                self._is_training = False
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                if hasattr(self, 'update_timer'):
                    self.update_timer.stop()
            except:
                pass
            
            # Show error dialog
            QMessageBox.critical(self, "Unexpected Error", 
                               f"An unexpected error occurred while starting training:\n\n{type(e).__name__}: {e}\n\nPlease check the console for more details.")
    
    @pyqtSlot()
    def _stop_training(self):
        """Stop the training process."""
        if self._training_process:
            # Update UI to show stopping state
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping...")
            self._log("\n" + "=" * 60)
            self._log("STOPPING TRAINING...")
            self._log("Waiting for current batch to complete...")
            self._log("=" * 60)
            
            # Update status in metrics label
            self.metrics_label.setText("Status: Stopping training...")
            
            self._training_process.stop_training()
        
        # Note: The actual cleanup will happen in _on_training_stopped
    
    def _update_device_label(self):
        """Update the device label with detected device info."""
        device = self._get_best_device()
        device_info = "CPU (No GPU available)"
        
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    cuda_version = torch.version.cuda
                    device_info = f"GPU: {gpu_name} (CUDA {cuda_version})"
            except:
                device_info = "GPU (CUDA)"
        elif device == "mps":
            device_info = "GPU (Apple Silicon)"
        
        self.device_label.setText(device_info)
        
        # Set color based on device type
        if device in ["cuda", "mps"]:
            self.device_label.setStyleSheet("color: #4CAF50; font-weight: bold;")  # Green for GPU
        else:
            self.device_label.setStyleSheet("color: #FF9800; font-weight: bold;")  # Orange for CPU
    
    def _get_best_device(self) -> str:
        """Detect and return the best available device for training."""
        try:
            import torch
            if torch.cuda.is_available():
                # GPU is available
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Apple Silicon GPU
                return "mps"
            else:
                # CPU fallback
                return "cpu"
        except ImportError:
            # If torch is not available, default to cpu
            return "cpu"
    
    def _get_dataset_directory_from_yaml(self) -> Path:
        """Extract the actual dataset directory from the YAML file."""
        try:
            import yaml
            print(f"[DEBUG] _get_dataset_directory_from_yaml: Reading YAML from: {self._dataset_path}")
            with open(self._dataset_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"[DEBUG] _get_dataset_directory_from_yaml: YAML contents: {config}")
                
            # Get the path from the YAML config
            dataset_root = config.get('path', '.')
            print(f"[DEBUG] _get_dataset_directory_from_yaml: Dataset root from YAML: '{dataset_root}'")
            
            # If it's a relative path, make it relative to the YAML file location
            yaml_dir = self._dataset_path.parent
            print(f"[DEBUG] _get_dataset_directory_from_yaml: YAML directory: {yaml_dir}")
            
            if not Path(dataset_root).is_absolute():
                actual_path = yaml_dir / dataset_root
                print(f"[DEBUG] _get_dataset_directory_from_yaml: Relative path detected, resolving to: {actual_path}")
            else:
                actual_path = Path(dataset_root)
                print(f"[DEBUG] _get_dataset_directory_from_yaml: Absolute path detected: {actual_path}")
                
            resolved_path = actual_path.resolve()
            print(f"[DEBUG] _get_dataset_directory_from_yaml: Final resolved path: {resolved_path}")
            print(f"[DEBUG] _get_dataset_directory_from_yaml: Path exists: {resolved_path.exists()}")
            
            if not resolved_path.exists():
                print(f"[ERROR] _get_dataset_directory_from_yaml: Resolved path does not exist!")
                print(f"[ERROR] _get_dataset_directory_from_yaml: Looking for alternative paths...")
                # Try without resolve() in case of symlink issues
                if actual_path.exists():
                    print(f"[INFO] _get_dataset_directory_from_yaml: Unresolved path exists: {actual_path}")
                    return actual_path
                    
            return resolved_path
            
        except Exception as e:
            print(f"[ERROR] _get_dataset_directory_from_yaml: Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to using the YAML file's directory
            fallback_path = self._dataset_path.parent
            print(f"[ERROR] _get_dataset_directory_from_yaml: Falling back to YAML parent directory: {fallback_path}")
            return fallback_path

    def _prepare_training_config(self) -> Dict:
        """Prepare training configuration."""
        # Determine which model to use
        model_cache = ModelCache()
        if self.pretrained_check.isChecked():
            # Use selected pre-trained model
            model = self.model_combo.currentText()
        elif model_cache.model_path:
            # Use loaded model for transfer learning
            model = str(model_cache.model_path)
        else:
            model = self.model_combo.currentText()  # Fallback to selected model
            
        config = {
            "task": "detect",
            "mode": "train",
            "model": model,
            "data": str(self._dataset_path),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": int(self.imgsz_combo.currentText()),
            "lr0": self.lr_spin.value(),
            "patience": self.patience_spin.value(),
            "pretrained": self.pretrained_check.isChecked(),
            "augment": self.augment_check.isChecked(),
            "cache": self.cache_check.isChecked(),
            "project": str(self._output_dir.parent),
            "name": self._output_dir.name,
            "exist_ok": True,
            "device": self._get_best_device()  # Automatically detect best device
        }
        
        # Add augmentation parameters if augmentation is enabled
        if self.augment_check.isChecked():
            augment_params = self.augment_settings.get_settings()
            config.update(augment_params)
        
        return config
    
    def _update_progress(self):
        """Update progress display."""
        if not self._is_training:
            return
        
        # Update time display
        if self._training_start_time:
            elapsed = datetime.now() - self._training_start_time
            elapsed_str = str(elapsed).split('.')[0]
            
            # Estimate remaining time based on current progress
            if self._current_step > 0 and self._total_steps > 0:
                # Calculate average time per step
                elapsed_seconds = elapsed.total_seconds()
                avg_time_per_step = elapsed_seconds / self._current_step
                remaining_steps = self._total_steps - self._current_step
                remaining_seconds = avg_time_per_step * remaining_steps
                
                # Format remaining time
                remaining_hours = int(remaining_seconds // 3600)
                remaining_minutes = int((remaining_seconds % 3600) // 60)
                remaining_secs = int(remaining_seconds % 60)
                remaining_str = f"{remaining_hours:02d}:{remaining_minutes:02d}:{remaining_secs:02d}"
                
                self.time_label.setText(f"Time: {elapsed_str} elapsed, {remaining_str} remaining")
            else:
                self.time_label.setText(f"Time: {elapsed_str} elapsed, calculating...")
    
    def _log(self, message: str):
        """Add message to training log."""
        self.log_text.append(message)
        # Auto-scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
    
    def _on_activate(self):
        """Called when mode is activated."""
        self.statusMessage.emit("Training Mode - Configure and monitor model training", 5000)
        
        # Check if dataset manager has a dataset loaded
        if self._dataset_manager.has_dataset() and not self._dataset_path:
            # Auto-populate with dataset manager's yaml
            yaml_path = self._dataset_manager.get_yaml_path()
            if yaml_path:
                self._dataset_path = yaml_path
                self.dataset_label.setText(str(yaml_path.name))
                self.dataset_label.setToolTip(str(yaml_path))
                # Parse and display dataset info
                self._parse_and_display_dataset_info()
                # Try to count training images
                self._count_training_images()
    
    def _on_deactivate(self) -> Optional[bool]:
        """Called when mode is deactivated."""
        if self._is_training:
            reply = QMessageBox.question(
                self,
                "Training in Progress",
                "Training is currently in progress. Stop training and switch modes?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._stop_training()
                return True
            else:
                return False
        return True
    
    def get_mode_name(self) -> str:
        """Get the display name of this mode."""
        return "Training"
    
    def has_unsaved_changes(self) -> bool:
        """Check if mode has unsaved changes."""
        return self._is_training
    
    @pyqtSlot(int, int)
    def _on_progress_update(self, current_epoch: int, total_epochs: int):
        """Handle progress updates from training process."""
        # self._log(f"[DEBUG] _on_progress_update called: epoch {current_epoch}/{total_epochs}")
        
        if current_epoch != self._current_epoch:
            self._log(f"\n>>> Starting Epoch {current_epoch}/{total_epochs}")
        self._current_epoch = current_epoch
        # Note: We don't update _total_epochs here to avoid changing our step calculations
        # The total epochs for progress calculation should remain as configured
        
        # Don't recalculate total steps here - keep the original calculation
        # This prevents the total from changing if YOLO reports different epoch count
    
    @pyqtSlot(int)
    def _on_epoch_progress(self, percent: int):
        """Handle epoch progress updates."""
        # Just update the epoch progress bar
        # The actual step counting is done in _on_batch_progress for accuracy
        self.epoch_progress.setValue(percent)
    
    @pyqtSlot(int, int)
    def _on_batch_progress(self, current_batch: int, total_batches: int):
        """Handle direct batch progress updates."""
        # self._log(f"[DEBUG] _on_batch_progress called: batch {current_batch}/{total_batches}, epoch {self._current_epoch}")
        
        # If we get batch progress but epoch is still 0, set it to 1
        if self._current_epoch == 0 and current_batch > 0:
            self._current_epoch = 1
            self._log(">>> Starting Epoch 1")
            
        # This gives us the most accurate progress tracking
        if self._current_epoch > 0:
            # If we haven't set steps per epoch yet, use the total_batches
            if self._steps_per_epoch == 0 or (total_batches > 0 and total_batches != self._steps_per_epoch):
                # Update steps per epoch if different
                if total_batches > 0 and total_batches != self._steps_per_epoch:
                    self._steps_per_epoch = total_batches
                    self._total_steps = self._steps_per_epoch * self._total_epochs
                    self.overall_progress.setMaximum(self._total_steps)
                    self._log(f"[INFO] Updated steps per epoch from batch counter: {total_batches}")
            
            # Calculate exact step number
            completed_steps = (self._current_epoch - 1) * self._steps_per_epoch
            new_step = completed_steps + current_batch
            
            # Only update if step actually changed
            if new_step != self._current_step:
                self._current_step = new_step
                
                # Update progress
                self.overall_progress.setValue(self._current_step)
                self.step_label.setText(f"Epoch {self._current_epoch}/{self._total_epochs} • {self._current_step}/{self._total_steps} steps")
                
                # Debug log - commented out
                # if current_batch <= 3 or current_batch == total_batches:
                #     self._log(f"[DEBUG] Progress updated: step {self._current_step}/{self._total_steps}, epoch progress {current_batch}/{total_batches}")
                
                # Update epoch progress based on batch
                if total_batches > 0:
                    percent = int((current_batch / total_batches) * 100)
                    self.epoch_progress.setValue(percent)
    
    @pyqtSlot(str)
    def _on_training_completed(self, model_path: str):
        """Handle training completion."""
        self._is_training = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_timer.stop()
        
        # Stop chart monitoring
        self.charts_widget.stop_monitoring()
        
        # Load results in Results tab
        if self._output_dir and self._output_dir.exists():
            self.results_widget.set_output_directory(self._output_dir)
            # Switch to Results tab
            self.output_tabs.setCurrentIndex(2)  # Results is the 3rd tab (index 2)
        
        # Update progress to full
        self.overall_progress.setValue(self._total_steps)
        self.step_label.setText(f"{self._total_steps}/{self._total_steps} steps")
        self.epoch_progress.setValue(100)
        
        # Refresh charts to show final state
        self.charts_widget.refresh_chart()
        
        # Emit signal
        self.trainingCompleted.emit(model_path)
    
    @pyqtSlot(str)
    def _on_training_failed(self, error: str):
        """Handle training failure."""
        self._is_training = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_timer.stop()
        
        # Stop chart monitoring
        self.charts_widget.stop_monitoring()
        
        # Refresh charts to show final state
        self.charts_widget.refresh_chart()
        
        # Emit signal
        self.trainingFailed.emit(error)
    
    @pyqtSlot(str)
    def _on_metrics_update(self, metrics_str: str):
        """Handle metrics update from training process."""
        # Update the label
        self.metrics_label.setText(metrics_str)
        
        # Send to charts
        self.charts_widget.on_metrics_update(metrics_str)
    
    @pyqtSlot(int)
    def _on_steps_detected(self, steps_per_epoch: int):
        """Handle detection of actual steps per epoch from training output."""
        if steps_per_epoch > 0 and steps_per_epoch != self._steps_per_epoch:
            # Store the original configured epochs to avoid using potentially updated value
            original_epochs = self.epochs_spin.value()
            
            # Sanity check: if we calculated steps from image count, verify the detected value is reasonable
            expected_steps = None
            if self._num_train_images > 0:
                expected_steps = math.ceil(self._num_train_images / self.batch_spin.value())
                # If detected steps are more than 2x expected, something might be wrong
                if steps_per_epoch > expected_steps * 2:
                    self._log(f"[WARNING] Detected steps ({steps_per_epoch}) seems too high for {self._num_train_images} images with batch {self.batch_spin.value()}")
                    self._log(f"[WARNING] Expected ~{expected_steps} steps/epoch, ignoring detected value")
                    return
            
            old_steps = self._steps_per_epoch
            self._steps_per_epoch = steps_per_epoch
            self._total_steps = self._steps_per_epoch * original_epochs
            self.overall_progress.setMaximum(self._total_steps)
            self._log(f"Updated steps per epoch: {steps_per_epoch} (was {old_steps})")
            self._log(f"Total training steps: {self._total_steps} ({steps_per_epoch} × {original_epochs} epochs)")
            # Update the label
            self.step_label.setText(f"{self._current_step}/{self._total_steps} steps")
    
    @pyqtSlot(Path)
    def _on_dataset_manager_loaded(self, yaml_path: Path):
        """Handle dataset loaded in dataset manager."""
        # Only update if we don't have a dataset path set
        if not self._dataset_path:
            self._dataset_path = yaml_path
            # Only update UI if it exists (mode has been activated)
            if hasattr(self, 'dataset_label'):
                self.dataset_label.setText(str(yaml_path.name))
                self.dataset_label.setToolTip(str(yaml_path))
                # Parse and display dataset info
                self._parse_and_display_dataset_info()
                # Try to count training images
                self._count_training_images()
    
    @pyqtSlot(str)
    def _on_training_stopped(self, message: str):
        """Handle training stopped by user."""
        self._is_training = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("Stop")
        self.update_timer.stop()
        
        # Stop chart monitoring
        self.charts_widget.stop_monitoring()
        
        # Refresh charts to show final state
        self.charts_widget.refresh_chart()
        
        # Update status
        self.metrics_label.setText("Status: Training stopped")
        
        # Log final message
        self._log("\n" + "=" * 60)
        self._log("TRAINING STOPPED")
        self._log(message)
        self._log("=" * 60)
        
        # Show message box notification
        QMessageBox.information(
            self,
            "Training Stopped",
            message,
            QMessageBox.StandardButton.Ok
        )