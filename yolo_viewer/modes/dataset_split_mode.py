"""Dataset Split mode for splitting YOLO datasets into train/val/test sets."""

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass
import random
import shutil
from datetime import datetime
from collections import defaultdict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSplitter, QFileDialog, QMessageBox, QProgressBar,
    QSlider, QSpinBox, QTextEdit, QScrollArea, QCheckBox, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QThread, QTimer
from PyQt6.QtGui import QFont

from .base_mode import BaseMode
from ..utils.yolo_format import load_data_yaml, parse_yolo_annotation
from ..core.constants import IMAGE_EXTENSIONS
from ..core import DatasetManager


@dataclass
class ClassDistribution:
    """Class distribution information for stratified splitting."""
    total_images: int = 0
    class_counts: Dict[int, int] = None  # class_id -> count
    images_per_class: Dict[int, List[Path]] = None  # class_id -> list of image paths
    images_by_class_combo: Dict[str, List[Tuple[Path, Path]]] = None  # "0,1" -> [(img, lbl)]
    class_combinations: Dict[str, int] = None  # "0,1" -> count
    
    def __post_init__(self):
        if self.class_counts is None:
            self.class_counts = {}
        if self.images_per_class is None:
            self.images_per_class = defaultdict(list)
        if self.images_by_class_combo is None:
            self.images_by_class_combo = defaultdict(list)
        if self.class_combinations is None:
            self.class_combinations = defaultdict(int)


@dataclass
class DatasetStructure:
    """Information about dataset structure."""
    is_structured: bool  # True if has separate images/labels folders
    image_folder: Optional[Path] = None
    label_folder: Optional[Path] = None
    image_files: List[Path] = None
    label_files: List[Path] = None
    matched_pairs: List[Tuple[Path, Optional[Path]]] = None  # (image, label)
    orphaned_labels: List[Path] = None  # Labels without corresponding images
    images_without_labels: List[Path] = None  # Images without labels
    class_distribution: Optional[ClassDistribution] = None  # Class distribution analysis


class SplitWorker(QThread):
    """Worker thread for splitting dataset."""
    progress = pyqtSignal(int, int)  # current, total
    statusUpdate = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, source_dir: Path, output_dir: Path, yaml_path: Path,
                 splits: Dict[str, List[Tuple[Path, Optional[Path]]]], 
                 structure: DatasetStructure,
                 rejected_files: Dict[str, List[Path]]):
        super().__init__()
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.yaml_path = yaml_path
        self.splits = splits
        self.structure = structure
        self.rejected_files = rejected_files  # {"images_without_labels": [...], "orphaned_labels": [...]}
        self._cancelled = False
    
    def run(self):
        """Execute the split operation."""
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy data.yaml to output if it exists
            if self.yaml_path and self.yaml_path.exists():
                shutil.copy2(self.yaml_path, self.output_dir / "data.yaml")
            
            # Calculate total files including rejected
            total_rejected = len(self.rejected_files.get("images_without_labels", [])) + \
                           len(self.rejected_files.get("orphaned_labels", []))
            total_files = sum(len(pairs) * 2 for pairs in self.splits.values()) + total_rejected  # x2 for image+label
            current = 0
            
            # Process rejected files first
            if total_rejected > 0:
                self.statusUpdate.emit(f"Moving {total_rejected} rejected files to 'rejected' folder...")
                
                # Create rejected directories
                rejected_dir = self.output_dir / "rejected"
                rejected_dir.mkdir(parents=True, exist_ok=True)
                
                # Process images without labels
                if self.rejected_files.get("images_without_labels"):
                    img_no_label_dir = rejected_dir / "images_without_labels"
                    img_no_label_dir.mkdir(parents=True, exist_ok=True)
                    
                    for img_path in self.rejected_files["images_without_labels"]:
                        if self._cancelled:
                            break
                        dest_img = img_no_label_dir / img_path.name
                        shutil.copy2(img_path, dest_img)
                        current += 1
                        self.progress.emit(current, total_files)
                
                # Process orphaned labels
                if self.rejected_files.get("orphaned_labels"):
                    orphaned_label_dir = rejected_dir / "labels_without_images"
                    orphaned_label_dir.mkdir(parents=True, exist_ok=True)
                    
                    for lbl_path in self.rejected_files["orphaned_labels"]:
                        if self._cancelled:
                            break
                        dest_lbl = orphaned_label_dir / lbl_path.name
                        shutil.copy2(lbl_path, dest_lbl)
                        current += 1
                        self.progress.emit(current, total_files)
            
            # Process each split
            for split_name, pairs in self.splits.items():
                if self._cancelled:
                    break
                
                # Skip empty splits (e.g., when test ratio is 0)
                if not pairs:
                    continue
                    
                self.statusUpdate.emit(f"Processing {split_name} split...")
                
                # Create split directories
                if self.structure.is_structured:
                    img_dir = self.output_dir / split_name / "images"
                    lbl_dir = self.output_dir / split_name / "labels"
                else:
                    img_dir = self.output_dir / split_name
                    lbl_dir = img_dir
                
                img_dir.mkdir(parents=True, exist_ok=True)
                if self.structure.is_structured:
                    lbl_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy files (only pairs with both image and label)
                for img_path, lbl_path in pairs:
                    if self._cancelled:
                        break
                    
                    # Copy image
                    dest_img = img_dir / img_path.name
                    shutil.copy2(img_path, dest_img)
                    current += 1
                    self.progress.emit(current, total_files)
                    
                    # Copy label (should always exist for valid pairs)
                    if lbl_path and lbl_path.exists():
                        dest_lbl = lbl_dir / lbl_path.name
                        shutil.copy2(lbl_path, dest_lbl)
                    current += 1
                    self.progress.emit(current, total_files)
            
            if not self._cancelled:
                self.finished.emit(True, "Dataset split completed successfully!")
            else:
                self.finished.emit(False, "Split operation cancelled")
                
        except Exception as e:
            self.finished.emit(False, f"Error during split: {str(e)}")
    
    def cancel(self):
        """Cancel the operation."""
        self._cancelled = True


class DatasetSplitMode(BaseMode):
    """
    Mode for splitting YOLO datasets into train/val/test sets.
    
    Features:
    - Support for both structured and flat datasets
    - Configurable split ratios
    - Random seed for reproducible splits
    - Preview before execution
    - Progress monitoring
    - Automatic summary report generation
    """
    
    # Signals
    splitStarted = pyqtSignal(str)  # output path
    splitProgress = pyqtSignal(int, int)  # current, total
    splitCompleted = pyqtSignal(str)  # output path
    
    def __init__(self, parent=None):
        # Initialize attributes before super().__init__()
        self._source_dir: Optional[Path] = None
        self._output_dir: Optional[Path] = None
        self._yaml_path: Optional[Path] = None
        self._dataset_structure: Optional[DatasetStructure] = None
        self._split_ratios = {"train": 0.7, "val": 0.2, "test": 0.1}
        self._random_seed = 42
        self._split_worker: Optional[SplitWorker] = None
        self._is_splitting = False
        self._dataset_manager = DatasetManager()
        
        super().__init__(parent)
        
        # Connect to dataset manager signals
        self._dataset_manager.datasetLoaded.connect(self._on_dataset_manager_loaded)
    
    def _setup_ui(self):
        """Setup the UI for dataset split mode."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QGroupBox {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5a5d;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #555555;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0d7377;
                border: 1px solid #0d7377;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                font-family: monospace;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #0d7377;
                border-radius: 3px;
            }
        """)
        
        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Configuration
        left_panel = self._create_config_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Information
        right_panel = self._create_info_panel()
        splitter.addWidget(right_panel)
        
        # Set initial sizes (40% / 60%)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
    
    def _create_config_panel(self) -> QWidget:
        """Create configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Dataset selection group
        dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QVBoxLayout(dataset_group)
        
        # Source dataset
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source Dataset:"))
        self._source_label = QLabel("Not selected")
        self._source_label.setStyleSheet("color: #888888;")
        source_layout.addWidget(self._source_label, 1)
        self._browse_source_btn = QPushButton("Browse")
        self._browse_source_btn.clicked.connect(self._browse_source_dataset)
        source_layout.addWidget(self._browse_source_btn)
        dataset_layout.addLayout(source_layout)
        
        # Output directory
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self._output_label = QLabel("Not selected")
        self._output_label.setStyleSheet("color: #888888;")
        output_layout.addWidget(self._output_label, 1)
        self._browse_output_btn = QPushButton("Browse")
        self._browse_output_btn.clicked.connect(self._browse_output_directory)
        output_layout.addWidget(self._browse_output_btn)
        dataset_layout.addLayout(output_layout)
        
        layout.addWidget(dataset_group)
        
        # Split configuration group
        split_group = QGroupBox("Split Configuration")
        split_layout = QVBoxLayout(split_group)
        
        # Train ratio
        self._train_slider, self._train_label = self._create_ratio_slider(
            "Train:", 0.7, self._on_train_ratio_changed
        )
        split_layout.addLayout(self._create_slider_layout(
            "Train:", self._train_slider, self._train_label
        ))
        
        # Val ratio
        self._val_slider, self._val_label = self._create_ratio_slider(
            "Validation:", 0.2, self._on_val_ratio_changed
        )
        split_layout.addLayout(self._create_slider_layout(
            "Validation:", self._val_slider, self._val_label
        ))
        
        # Test ratio
        self._test_slider, self._test_label = self._create_ratio_slider(
            "Test:", 0.1, self._on_test_ratio_changed
        )
        split_layout.addLayout(self._create_slider_layout(
            "Test:", self._test_slider, self._test_label
        ))
        
        # Random seed
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Random Seed:"))
        self._seed_spinbox = QSpinBox()
        self._seed_spinbox.setRange(0, 99999)
        self._seed_spinbox.setValue(42)
        self._seed_spinbox.valueChanged.connect(self._on_seed_changed)
        seed_layout.addWidget(self._seed_spinbox)
        seed_layout.addStretch()
        split_layout.addLayout(seed_layout)
        
        # Stratified splitting option
        self._stratified_checkbox = QCheckBox("Use Stratified Splitting")
        self._stratified_checkbox.setToolTip(
            "Maintain class distribution across splits.\n"
            "Ensures each split has similar proportions of each class combination."
        )
        self._stratified_checkbox.setChecked(True)  # Default to stratified
        split_layout.addWidget(self._stratified_checkbox)
        
        # Ignore unannotated images option
        self._ignore_unannotated_checkbox = QCheckBox("Ignore Images Without Annotations")
        self._ignore_unannotated_checkbox.setToolTip(
            "When checked: Only images with annotations will be included in the dataset split.\n"
            "When unchecked: All images will be included, even those without annotations."
        )
        self._ignore_unannotated_checkbox.setChecked(True)  # Default to ignore unannotated
        self._ignore_unannotated_checkbox.stateChanged.connect(self._on_ignore_unannotated_changed)
        split_layout.addWidget(self._ignore_unannotated_checkbox)
        
        layout.addWidget(split_group)
        
        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self._preview_btn = QPushButton("👁️ Preview Split")
        self._preview_btn.clicked.connect(self._preview_split)
        self._preview_btn.setEnabled(False)
        actions_layout.addWidget(self._preview_btn)
        
        self._execute_btn = QPushButton("▶️ Execute Split")
        self._execute_btn.clicked.connect(self._execute_split)
        self._execute_btn.setEnabled(False)
        self._execute_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
        """)
        actions_layout.addWidget(self._execute_btn)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        actions_layout.addWidget(self._progress_bar)
        
        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        actions_layout.addWidget(self._status_label)
        
        layout.addWidget(actions_group)
        layout.addStretch()
        
        return panel
    
    def _create_info_panel(self) -> QWidget:
        """Create information panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Info display
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout(info_group)
        
        # Add header with clear button
        header_layout = QHBoxLayout()
        header_layout.addStretch()
        
        clear_btn = QPushButton("Clear")
        clear_btn.setMaximumWidth(60)
        clear_btn.clicked.connect(self._clear_info_display)
        header_layout.addWidget(clear_btn)
        
        info_layout.addLayout(header_layout)
        
        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setPlainText("No dataset loaded")
        info_layout.addWidget(self._info_text)
        
        layout.addWidget(info_group)
        
        return panel
    
    def _create_ratio_slider(self, label: str, initial: float, callback) -> Tuple[QSlider, QLabel]:
        """Create a ratio slider with label."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(initial * 100))
        slider.valueChanged.connect(callback)
        
        label = QLabel(f"{int(initial * 100)}%")
        label.setMinimumWidth(45)
        label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        return slider, label
    
    def _create_slider_layout(self, title: str, slider: QSlider, label: QLabel) -> QHBoxLayout:
        """Create layout for slider with title and percentage."""
        layout = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setMinimumWidth(80)
        layout.addWidget(title_label)
        layout.addWidget(slider, 1)
        layout.addWidget(label)
        return layout
    
    @pyqtSlot()
    def _browse_source_dataset(self):
        """Browse for source dataset directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Source Dataset Directory"
        )
        if path:
            self._source_dir = Path(path)
            self._source_label.setText(str(self._source_dir))
            self._source_label.setStyleSheet("color: #cccccc;")
            self._scan_dataset()
    
    @pyqtSlot()
    def _browse_output_directory(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if path:
            self._output_dir = Path(path)
            self._output_label.setText(str(self._output_dir))
            self._output_label.setStyleSheet("color: #cccccc;")
            self._update_ui_state()
    
    
    def _scan_dataset(self):
        """Scan dataset structure and find files."""
        if not self._source_dir:
            return
        
        self._status_label.setText("Scanning dataset...")
        QTimer.singleShot(100, self._do_scan)
    
    def _do_scan(self):
        """Perform the actual scanning."""
        try:
            structure = DatasetStructure(is_structured=False)
            
            # Check for structured dataset (separate images/labels folders)
            images_dir = self._source_dir / "images"
            labels_dir = self._source_dir / "labels"
            
            if images_dir.exists() and labels_dir.exists():
                structure.is_structured = True
                structure.image_folder = images_dir
                structure.label_folder = labels_dir
                
                # Get all image files
                image_files = []
                for ext in IMAGE_EXTENSIONS:
                    image_files.extend(images_dir.glob(f"**/*{ext}"))
                structure.image_files = sorted(image_files)
                
                # Get all label files
                structure.label_files = sorted(labels_dir.glob("**/*.txt"))
            else:
                # Flat structure - everything in one folder
                structure.image_folder = self._source_dir
                structure.label_folder = self._source_dir
                
                # Get all image files
                image_files = []
                for ext in IMAGE_EXTENSIONS:
                    image_files.extend(self._source_dir.glob(f"*{ext}"))
                structure.image_files = sorted(image_files)
                
                # Get all label files
                structure.label_files = sorted(self._source_dir.glob("*.txt"))
            
            # Match image-label pairs and find orphans
            matched_pairs = []
            images_without_labels = []
            label_dict = {lbl.stem: lbl for lbl in structure.label_files}
            image_stems = {img.stem for img in structure.image_files}
            
            # Find images without labels
            for img_file in structure.image_files:
                label_file = label_dict.get(img_file.stem)
                if label_file:
                    matched_pairs.append((img_file, label_file))
                else:
                    images_without_labels.append(img_file)
            
            # Find orphaned labels (labels without images)
            orphaned_labels = []
            for lbl_file in structure.label_files:
                if lbl_file.stem not in image_stems:
                    orphaned_labels.append(lbl_file)
            
            structure.matched_pairs = matched_pairs
            structure.images_without_labels = images_without_labels
            structure.orphaned_labels = orphaned_labels
            self._dataset_structure = structure
            
            # Analyze class distribution
            self._analyze_class_distribution()
            
            # Update info display
            self._update_dataset_info()
            
            # Look for data.yaml if not already selected
            if not self._yaml_path:
                yaml_candidates = list(self._source_dir.glob("*.yaml")) + \
                                list(self._source_dir.glob("*.yml"))
                if yaml_candidates:
                    self._yaml_path = yaml_candidates[0]
                    self._load_yaml_info()
            
            self._status_label.setText("Dataset scanned successfully")
            self._update_ui_state()
            
        except Exception as e:
            self._status_label.setText(f"Error scanning dataset: {str(e)}")
            self.statusMessage.emit(f"Error scanning dataset: {str(e)}", 5000)
    
    def _analyze_class_distribution(self):
        """Analyze class distribution in the dataset."""
        if not self._dataset_structure or not self._dataset_structure.matched_pairs:
            return
        
        distribution = ClassDistribution()
        distribution.total_images = len(self._dataset_structure.matched_pairs)
        
        # Analyze each image-label pair
        for img_path, lbl_path in self._dataset_structure.matched_pairs:
            if lbl_path and lbl_path.exists():
                annotations = parse_yolo_annotation(lbl_path)
                if annotations:
                    # Get unique classes in this image
                    classes_in_image = set()
                    for ann in annotations:
                        class_id = int(ann[0])
                        classes_in_image.add(class_id)
                        
                        # Count total annotations per class
                        if class_id not in distribution.class_counts:
                            distribution.class_counts[class_id] = 0
                        distribution.class_counts[class_id] += 1
                        
                        # Track which images contain each class
                        distribution.images_per_class[class_id].append(img_path)
                    
                    # Create class combination key (sorted for consistency)
                    combo_key = ",".join(map(str, sorted(classes_in_image)))
                    distribution.class_combinations[combo_key] += 1
                    distribution.images_by_class_combo[combo_key].append((img_path, lbl_path))
        
        self._dataset_structure.class_distribution = distribution
    
    def _on_activate(self):
        """Called when mode is activated."""
        self.statusMessage.emit("Dataset Split mode activated - Configure and split your dataset", 3000)
        
        # Check if dataset manager has a dataset loaded
        if self._dataset_manager.has_dataset() and not self._yaml_path:
            # Auto-populate with dataset manager's yaml
            yaml_path = self._dataset_manager.get_yaml_path()
            if yaml_path:
                self._yaml_path = yaml_path
                self._load_yaml_info()
                
                # If source directory is not set, try to infer it
                if not self._source_dir:
                    # Use the parent directory of the yaml file as default
                    self._source_dir = yaml_path.parent
                    self._source_label.setText(str(self._source_dir))
                    self._source_label.setStyleSheet("color: #cccccc;")
                    self._scan_dataset()
    
    def _on_deactivate(self) -> Optional[bool]:
        """Called when mode is deactivated."""
        if self._is_splitting:
            reply = QMessageBox.question(
                self, "Split in Progress",
                "Dataset split is in progress. Cancel and switch mode?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return False
            else:
                if self._split_worker:
                    self._split_worker.cancel()
                    self._split_worker.wait()
        return True
    
    def _on_train_ratio_changed(self, value):
        """Handle train ratio change."""
        self._split_ratios["train"] = value / 100.0
        self._train_label.setText(f"{value}%")
        self._validate_ratios()
    
    def _on_val_ratio_changed(self, value):
        """Handle validation ratio change."""
        self._split_ratios["val"] = value / 100.0
        self._val_label.setText(f"{value}%")
        self._validate_ratios()
    
    def _on_test_ratio_changed(self, value):
        """Handle test ratio change."""
        self._split_ratios["test"] = value / 100.0
        self._test_label.setText(f"{value}%")
        self._validate_ratios()
    
    def _validate_ratios(self):
        """Validate that ratios sum to 1.0."""
        total = sum(self._split_ratios.values())
        if abs(total - 1.0) > 0.01:
            self._status_label.setText(f"⚠️ Ratios sum to {total:.0%} (must be 100%)")
            self._status_label.setStyleSheet("color: #ff9800;")
            self._preview_btn.setEnabled(False)
            self._execute_btn.setEnabled(False)
        else:
            self._status_label.setText("Ready")
            self._status_label.setStyleSheet("color: #cccccc;")
            self._update_ui_state()
    
    def _on_seed_changed(self, value):
        """Handle seed change."""
        self._random_seed = value
    
    def _on_ignore_unannotated_changed(self, state):
        """Handle ignore unannotated checkbox change."""
        # Update the dataset info display to reflect the new setting
        if self._dataset_structure:
            self._update_dataset_info()
    
    def _update_ui_state(self):
        """Update UI element states."""
        has_required = (
            self._source_dir is not None and
            self._output_dir is not None and
            self._dataset_structure is not None
        )
        
        ratios_valid = abs(sum(self._split_ratios.values()) - 1.0) <= 0.01
        
        self._preview_btn.setEnabled(has_required and ratios_valid and not self._is_splitting)
        self._execute_btn.setEnabled(has_required and ratios_valid and not self._is_splitting)
    
    def _create_stratified_splits(self, valid_pairs: List[Tuple[Path, Path]]) -> Dict[str, List[Tuple[Path, Path]]]:
        """Create stratified splits maintaining class distribution."""
        if not self._dataset_structure.class_distribution:
            # Fallback to random split if no class distribution
            return self._create_random_splits(valid_pairs)
        
        dist = self._dataset_structure.class_distribution
        splits = {"train": [], "val": [], "test": []}
        
        # Split each class combination proportionally
        for combo_key, pairs in dist.images_by_class_combo.items():
            if not pairs:
                continue
                
            # Shuffle pairs for this combination
            combo_pairs = pairs.copy()
            random.shuffle(combo_pairs)
            
            # Calculate split sizes for this combination
            combo_total = len(combo_pairs)
            train_size = int(combo_total * self._split_ratios["train"])
            val_size = int(combo_total * self._split_ratios["val"])
            test_size = int(combo_total * self._split_ratios["test"])
            
            # Handle special cases based on combo_total and ratios
            if combo_total == 1:
                # Only 1 sample - assign based on highest ratio
                if self._split_ratios["train"] >= self._split_ratios["val"] and self._split_ratios["train"] >= self._split_ratios["test"]:
                    train_size, val_size, test_size = 1, 0, 0
                elif self._split_ratios["val"] >= self._split_ratios["test"]:
                    train_size, val_size, test_size = 0, 1, 0
                else:
                    train_size, val_size, test_size = 0, 0, 1
            elif combo_total == 2:
                # 2 samples - split between two largest ratios
                ratios = [("train", self._split_ratios["train"]), 
                          ("val", self._split_ratios["val"]), 
                          ("test", self._split_ratios["test"])]
                ratios.sort(key=lambda x: x[1], reverse=True)
                
                # Reset sizes
                train_size, val_size, test_size = 0, 0, 0
                
                # Assign 1 to each of the top 2 ratios (if > 0)
                assigned = 0
                for split_name, ratio in ratios:
                    if ratio > 0 and assigned < 2:
                        if split_name == "train":
                            train_size = 1
                        elif split_name == "val":
                            val_size = 1
                        elif split_name == "test":
                            test_size = 1
                        assigned += 1
            else:
                # 3 or more samples - ensure sizes sum to combo_total
                current_sum = train_size + val_size + test_size
                if current_sum < combo_total:
                    diff = combo_total - current_sum
                    # Add remainder based on ratio priority
                    if self._split_ratios["train"] > 0:
                        train_size += diff
                    elif self._split_ratios["val"] > 0:
                        val_size += diff
                    elif self._split_ratios["test"] > 0:
                        test_size += diff
            
            # Assign to splits based on calculated sizes
            train_end = train_size
            val_end = train_end + val_size
            
            splits["train"].extend(combo_pairs[:train_end])
            splits["val"].extend(combo_pairs[train_end:val_end])
            # Only add to test if test ratio > 0
            if self._split_ratios["test"] > 0 and test_size > 0:
                splits["test"].extend(combo_pairs[val_end:val_end + test_size])
        
        return splits
    
    def _create_random_splits(self, valid_pairs: List[Tuple[Path, Path]]) -> Dict[str, List[Tuple[Path, Path]]]:
        """Create random splits without stratification."""
        # Shuffle all pairs
        shuffled_pairs = valid_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # Calculate split sizes
        total = len(shuffled_pairs)
        train_size = int(total * self._split_ratios["train"])
        val_size = int(total * self._split_ratios["val"])
        test_size = int(total * self._split_ratios["test"])
        
        # Handle rounding errors - ensure sizes sum to total
        # Priority: train > val > test
        current_sum = train_size + val_size + test_size
        if current_sum < total:
            diff = total - current_sum
            if self._split_ratios["train"] > 0:
                train_size += diff
            elif self._split_ratios["val"] > 0:
                val_size += diff
            elif self._split_ratios["test"] > 0:
                test_size += diff
        
        # Create splits
        train_end = train_size
        val_end = train_end + val_size
        test_end = val_end + test_size
        
        return {
            "train": shuffled_pairs[:train_end],
            "val": shuffled_pairs[train_end:val_end],
            "test": shuffled_pairs[val_end:test_end] if self._split_ratios["test"] > 0 else []
        }
    
    @pyqtSlot()
    def _preview_split(self):
        """Preview the split without copying files."""
        if not self._dataset_structure or not self._dataset_structure.matched_pairs:
            return
        
        # Set random seed
        random.seed(self._random_seed)
        
        # Get valid pairs
        valid_pairs = self._dataset_structure.matched_pairs.copy()
        
        # Handle images without annotations based on checkbox setting
        if self._ignore_unannotated_checkbox.isChecked():
            # Simply ignore images without labels - they won't be included or rejected
            images_ignored = len(self._dataset_structure.images_without_labels or [])
            images_to_reject = []
        else:
            # Images without labels will be moved to rejected folder
            images_ignored = 0
            images_to_reject = self._dataset_structure.images_without_labels or []
        
        # Create splits based on stratification setting
        if self._stratified_checkbox.isChecked() and self._dataset_structure.class_distribution:
            splits = self._create_stratified_splits(valid_pairs)
        else:
            splits = self._create_random_splits(valid_pairs)
        
        # Calculate total valid pairs
        total_valid = len(valid_pairs)
        
        # Display preview
        preview_text = f"Dataset Split Preview\n"
        preview_text += f"{'='*50}\n\n"
        preview_text += f"Valid Image-Label Pairs: {total_valid}\n"
        if images_ignored > 0:
            preview_text += f"Images Ignored (no annotations): {images_ignored}\n"
        preview_text += f"Random Seed: {self._random_seed}\n"
        preview_text += f"Split Method: {'Stratified' if self._stratified_checkbox.isChecked() else 'Random'}\n"
        preview_text += f"Ignore Unannotated: {'Yes' if self._ignore_unannotated_checkbox.isChecked() else 'No'}\n\n"
        
        # Show validation issues
        total_rejected = len(images_to_reject) + len(self._dataset_structure.orphaned_labels)
        if total_rejected > 0:
            preview_text += f"⚠️ Files to be Rejected: {total_rejected} (will be moved to 'rejected' folder)\n"
            if images_to_reject:
                preview_text += f"  - Images without labels: {len(images_to_reject)}\n"
            if self._dataset_structure.orphaned_labels:
                preview_text += f"  - Labels without images: {len(self._dataset_structure.orphaned_labels)}\n"
            preview_text += f"\n"
        
        # Get actual split sizes
        train_size = len(splits["train"])
        val_size = len(splits["val"])
        test_size = len(splits["test"])
        
        preview_text += f"Split Distribution (Valid Pairs Only):\n"
        preview_text += f"  Train: {train_size} ({train_size/total_valid*100:.1f}%)\n"
        preview_text += f"  Val:   {val_size} ({val_size/total_valid*100:.1f}%)\n"
        preview_text += f"  Test:  {test_size} ({test_size/total_valid*100:.1f}%)\n\n"
        
        # Show class distribution per split if stratified
        if self._stratified_checkbox.isChecked() and self._dataset_structure.class_distribution:
            preview_text += f"Class Distribution per Split:\n"
            
            # Get class names if available
            class_names = {}
            if self._yaml_path and self._yaml_path.exists():
                try:
                    data = load_data_yaml(self._yaml_path)
                    if data and 'names' in data:
                        names = data['names']
                        if isinstance(names, dict):
                            class_names = {int(k): v for k, v in names.items()}
                        else:
                            class_names = {i: name for i, name in enumerate(names)}
                except:
                    pass
            
            # Analyze each split
            for split_name, split_pairs in splits.items():
                if split_pairs:
                    # Count class combinations in this split
                    combo_counts = defaultdict(int)
                    for img_path, lbl_path in split_pairs:
                        # Find which combination this belongs to
                        for combo_key, combo_pairs in self._dataset_structure.class_distribution.images_by_class_combo.items():
                            if any(p[0] == img_path for p in combo_pairs):
                                combo_counts[combo_key] += 1
                                break
                    
                    preview_text += f"\n  {split_name.upper()}:\n"
                    for combo, count in sorted(combo_counts.items()):
                        if combo:
                            class_ids = [int(x) for x in combo.split(',')]
                            class_names_str = ", ".join([class_names.get(cid, f"Class {cid}") for cid in class_ids])
                            preview_text += f"    [{combo}] {class_names_str}: {count} images\n"
            
            preview_text += f"\n"
        
        # Show output directory structure
        preview_text += f"Output Directory Structure:\n"
        preview_text += f"  {self._output_dir.name}/\n"
        preview_text += f"    ├── train/\n"
        if self._dataset_structure.is_structured:
            preview_text += f"    │   ├── images/\n"
            preview_text += f"    │   └── labels/\n"
        preview_text += f"    ├── val/\n"
        if self._dataset_structure.is_structured:
            preview_text += f"    │   ├── images/\n"
            preview_text += f"    │   └── labels/\n"
        preview_text += f"    ├── test/\n"
        if self._dataset_structure.is_structured:
            preview_text += f"    │   ├── images/\n"
            preview_text += f"    │   └── labels/\n"
        if total_rejected > 0:
            preview_text += f"    ├── rejected/  (contains {total_rejected} problematic files)\n"
            if images_to_reject:
                preview_text += f"    │   ├── images_without_labels/  ({len(images_to_reject)} files)\n"
            if self._dataset_structure.orphaned_labels:
                preview_text += f"    │   └── labels_without_images/  ({len(self._dataset_structure.orphaned_labels)} files)\n"
        preview_text += f"    └── data.yaml\n\n"
        
        # Show sample files from each split
        preview_text += f"Sample Files:\n"
        for split_name, split_pairs in splits.items():
            preview_text += f"\n{split_name.upper()} (first 3):\n"
            for i, (img, lbl) in enumerate(split_pairs[:3]):
                preview_text += f"  - {img.name}\n"
        
        self._info_text.setPlainText(preview_text)
        self._status_label.setText("Preview generated")
    
    @pyqtSlot()
    def _execute_split(self):
        """Execute the actual split."""
        if not self._dataset_structure or not self._output_dir:
            return
        
        # Check for existing train/test/val folders
        existing_split_dirs = []
        for split_name in ["train", "test", "val"]:
            split_dir = self._output_dir / split_name
            if split_dir.exists() and any(split_dir.iterdir()):
                existing_split_dirs.append(split_name)
        
        if existing_split_dirs:
            # Create custom message box with Yes, No, Cancel options
            msg = QMessageBox(self)
            msg.setWindowTitle("Existing Split Folders Found")
            msg.setText(f"The following split folders already exist and contain files:\n"
                       f"{', '.join(existing_split_dirs)}\n\n"
                       f"Do you want to delete these existing folders before splitting?")
            msg.setInformativeText("Yes: Delete existing folders and continue\n"
                                 "No: Keep existing folders (files may be overwritten)\n"
                                 "Cancel: Abort the split operation")
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | 
                                 QMessageBox.StandardButton.No | 
                                 QMessageBox.StandardButton.Cancel)
            msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
            
            reply = msg.exec()
            
            if reply == QMessageBox.StandardButton.Cancel:
                return
            elif reply == QMessageBox.StandardButton.Yes:
                # Delete existing split directories
                try:
                    for split_name in ["train", "test", "val"]:
                        split_dir = self._output_dir / split_name
                        if split_dir.exists():
                            shutil.rmtree(split_dir)
                            self._status_label.setText(f"Deleted existing {split_name} folder")
                            QApplication.processEvents()  # Update UI immediately
                except Exception as e:
                    QMessageBox.critical(self, "Error", 
                                       f"Failed to delete existing folders: {str(e)}")
                    return
            # If No is selected, continue without deleting (files may be overwritten)
        
        # Set random seed
        random.seed(self._random_seed)
        
        # Get valid pairs
        valid_pairs = self._dataset_structure.matched_pairs.copy()
        
        # Handle images without annotations based on checkbox setting
        if self._ignore_unannotated_checkbox.isChecked():
            # Simply ignore images without labels - don't include them in split or rejected
            rejected_files = {
                "images_without_labels": [],  # Don't move to rejected, just ignore
                "orphaned_labels": self._dataset_structure.orphaned_labels or []
            }
        else:
            # Move images without labels to rejected folder
            rejected_files = {
                "images_without_labels": self._dataset_structure.images_without_labels or [],
                "orphaned_labels": self._dataset_structure.orphaned_labels or []
            }
        
        # Create splits based on stratification setting
        if self._stratified_checkbox.isChecked() and self._dataset_structure.class_distribution:
            splits = self._create_stratified_splits(valid_pairs)
        else:
            splits = self._create_random_splits(valid_pairs)
        
        # Create and start worker thread
        self._split_worker = SplitWorker(
            self._source_dir,
            self._output_dir,
            self._yaml_path or self._source_dir / "data.yaml",
            splits,
            self._dataset_structure,
            rejected_files
        )
        
        self._split_worker.progress.connect(self._on_split_progress)
        self._split_worker.statusUpdate.connect(self._on_split_status)
        self._split_worker.finished.connect(self._on_split_finished)
        
        # Update UI
        self._is_splitting = True
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._update_ui_state()
        
        # Start split
        self._split_worker.start()
        self.splitStarted.emit(str(self._output_dir))
    
    @pyqtSlot(int, int)
    def _on_split_progress(self, current, total):
        """Handle split progress update."""
        if total > 0:
            progress = int((current / total) * 100)
            self._progress_bar.setValue(progress)
    
    @pyqtSlot(str)
    def _on_split_status(self, message):
        """Handle split status update."""
        self._status_label.setText(message)
    
    @pyqtSlot(bool, str)
    def _on_split_finished(self, success, message):
        """Handle split completion."""
        self._is_splitting = False
        self._progress_bar.setVisible(False)
        self._update_ui_state()
        
        if success:
            self._status_label.setText("Split completed successfully!")
            self._status_label.setStyleSheet("color: #27ae60;")
            
            # Generate summary report
            self._generate_summary_report()
            
            # Generate validation report if there were issues
            if (self._dataset_structure.images_without_labels or 
                self._dataset_structure.orphaned_labels):
                self._generate_validation_report()
            
            self.splitCompleted.emit(str(self._output_dir))
            self.statusMessage.emit(f"Dataset split completed: {self._output_dir.name}", 5000)
            
            # Build completion message
            msg = f"Dataset has been split successfully!\n\n"
            msg += f"Output: {self._output_dir}\n\n"
            
            QMessageBox.information(
                self, "Split Complete", msg
            )
        else:
            self._status_label.setText(f"Split failed: {message}")
            self._status_label.setStyleSheet("color: #e74c3c;")
            self.statusMessage.emit(f"Split failed: {message}", 5000)
            
            QMessageBox.critical(
                self, "Split Failed",
                f"Failed to split dataset:\n\n{message}"
            )
        
        # Reset status after delay
        QTimer.singleShot(5000, lambda: self._status_label.setStyleSheet("color: #cccccc;"))
    
    def _generate_summary_report(self):
        """Generate a summary report file."""
        try:
            report_path = self._output_dir / "dataset_info.txt"
            
            # Count files in each split
            split_counts = {}
            for split in ["train", "val", "test"]:
                if self._dataset_structure.is_structured:
                    img_dir = self._output_dir / split / "images"
                else:
                    img_dir = self._output_dir / split
                
                if img_dir.exists():
                    count = len(list(img_dir.glob("*")))
                    split_counts[split] = count
            
            # Count rejected files
            rejected_counts = {}
            rejected_dir = self._output_dir / "rejected"
            if rejected_dir.exists():
                img_no_label_dir = rejected_dir / "images_without_labels"
                if img_no_label_dir.exists():
                    rejected_counts["images_without_labels"] = len(list(img_no_label_dir.glob("*")))
                
                orphaned_label_dir = rejected_dir / "labels_without_images"
                if orphaned_label_dir.exists():
                    rejected_counts["labels_without_images"] = len(list(orphaned_label_dir.glob("*")))
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"Dataset Split Summary\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source: {self._source_dir}\n")
                f.write(f"Output: {self._output_dir}\n")
                f.write(f"Random Seed: {self._random_seed}\n")
                f.write(f"Structure: {'Structured' if self._dataset_structure.is_structured else 'Flat'}\n\n")
                
                f.write(f"Valid Pairs Split Distribution:\n")
                total = sum(split_counts.values())
                for split, count in split_counts.items():
                    percentage = (count / total * 100) if total > 0 else 0
                    f.write(f"  {split}: {count} ({percentage:.1f}%)\n")
                
                f.write(f"\nTotal Valid Images: {total}\n")
                
                # Report rejected files
                if rejected_counts:
                    f.write(f"\nRejected Files:\n")
                    total_rejected = sum(rejected_counts.values())
                    for category, count in rejected_counts.items():
                        f.write(f"  {category}: {count}\n")
                    f.write(f"Total Rejected: {total_rejected}\n")
                
                # Dataset validation summary
                f.write(f"\nDataset Validation Summary:\n")
                f.write(f"  Original Images: {len(self._dataset_structure.image_files)}\n")
                f.write(f"  Original Labels: {len(self._dataset_structure.label_files)}\n")
                f.write(f"  Valid Pairs: {len(self._dataset_structure.matched_pairs)}\n")
                
                # Detailed validation issues
                if self._dataset_structure.images_without_labels:
                    f.write(f"\n  Images without labels: {len(self._dataset_structure.images_without_labels)}\n")
                    # List all images without labels
                    for img in self._dataset_structure.images_without_labels:
                        expected_label = img.stem + ".txt"
                        f.write(f"    - {img.name} → missing {expected_label}\n")
                        
                if self._dataset_structure.orphaned_labels:
                    f.write(f"\n  Labels without images: {len(self._dataset_structure.orphaned_labels)}\n")
                    # List all orphaned labels
                    for lbl in self._dataset_structure.orphaned_labels:
                        possible_imgs = []
                        for ext in IMAGE_EXTENSIONS:
                            possible_imgs.append(lbl.stem + ext)
                        f.write(f"    - {lbl.name} → missing image ({', '.join(possible_imgs[:3])}...)\n")
                
                if self._yaml_path and self._yaml_path.exists():
                    f.write(f"\ndata.yaml: {self._yaml_path.name}\n")
        
        except Exception as e:
            print(f"Error generating summary report: {e}")
    
    def _generate_validation_report(self):
        """Generate a detailed validation report for problematic files."""
        try:
            report_path = self._output_dir / "validation_issues.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"Dataset Validation Issues Report\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source: {self._source_dir}\n\n")
                
                # Summary
                total_images_no_labels = len(self._dataset_structure.images_without_labels)
                total_labels_no_images = len(self._dataset_structure.orphaned_labels)
                total_issues = total_images_no_labels + total_labels_no_images
                
                f.write(f"Summary:\n")
                f.write(f"  Total Issues Found: {total_issues}\n")
                f.write(f"  Images without labels: {total_images_no_labels}\n")
                f.write(f"  Labels without images: {total_labels_no_images}\n\n")
                
                # Detailed list of images without labels
                if self._dataset_structure.images_without_labels:
                    f.write(f"Images Without Labels ({total_images_no_labels}):\n")
                    f.write(f"{'-'*40}\n")
                    for img in sorted(self._dataset_structure.images_without_labels):
                        expected_label = img.stem + ".txt"
                        label_path = img.parent / expected_label if not self._dataset_structure.is_structured else \
                                     self._dataset_structure.label_folder / expected_label
                        f.write(f"Image: {img.name}\n")
                        f.write(f"  Expected label: {expected_label}\n")
                        f.write(f"  Expected path: {label_path}\n")
                        f.write(f"  Status: NOT FOUND\n\n")
                
                # Detailed list of orphaned labels
                if self._dataset_structure.orphaned_labels:
                    f.write(f"\nLabels Without Images ({total_labels_no_images}):\n")
                    f.write(f"{'-'*40}\n")
                    for lbl in sorted(self._dataset_structure.orphaned_labels):
                        f.write(f"Label: {lbl.name}\n")
                        f.write(f"  Expected image names:\n")
                        for ext in IMAGE_EXTENSIONS:
                            expected_img = lbl.stem + ext
                            img_path = lbl.parent / expected_img if not self._dataset_structure.is_structured else \
                                      self._dataset_structure.image_folder / expected_img
                            f.write(f"    - {expected_img} (at {img_path})\n")
                        f.write(f"  Status: NO MATCHING IMAGE FOUND\n\n")
                
                # Recommendations
                f.write(f"\nRecommendations:\n")
                f.write(f"{'-'*40}\n")
                f.write(f"1. Check the 'rejected' folder for all problematic files\n")
                f.write(f"2. For images without labels:\n")
                f.write(f"   - Create annotations using the Dataset Editor\n")
                f.write(f"   - Or use Auto-Annotation mode if you have a trained model\n")
                f.write(f"3. For orphaned labels:\n")
                f.write(f"   - Verify if the image files were accidentally deleted\n")
                f.write(f"   - Check if image extensions match (e.g., .jpg vs .png)\n")
                f.write(f"   - Remove orphaned label files if images are permanently missing\n")
                
        except Exception as e:
            print(f"Error generating validation report: {e}")
    
    
    
    def _load_yaml_info(self):
        """Load and display YAML information."""
        if not self._yaml_path or not self._yaml_path.exists():
            return
        
        try:
            data = load_data_yaml(self._yaml_path)
            if data:
                # Get existing text and add separator if needed
                existing_text = self._info_text.toPlainText()
                if existing_text and not existing_text.endswith('\n'):
                    existing_text += '\n'
                if existing_text and existing_text.strip() != "No dataset loaded":
                    existing_text += '\n'  # Add extra line for separation
                
                yaml_text = f"data.yaml Contents:\n"
                yaml_text += f"{'='*50}\n\n"
                
                if 'path' in data:
                    yaml_text += f"Path: {data['path']}\n"
                
                if 'train' in data:
                    yaml_text += f"Train: {data['train']}\n"
                if 'val' in data:
                    yaml_text += f"Val: {data['val']}\n"
                if 'test' in data:
                    yaml_text += f"Test: {data['test']}\n"
                
                yaml_text += f"\nClasses: {data.get('nc', 'Unknown')}\n"
                
                names = data.get('names', {})
                if names:
                    yaml_text += f"\nClass Names:\n"
                    if isinstance(names, dict):
                        for idx, name in sorted(names.items(), key=lambda x: int(x[0])):
                            yaml_text += f"  {idx}: {name}\n"
                    else:
                        for idx, name in enumerate(names):
                            yaml_text += f"  {idx}: {name}\n"
                
                self._info_text.setPlainText(existing_text + yaml_text)
        except Exception as e:
            # Append error to existing text
            existing_text = self._info_text.toPlainText()
            if existing_text and not existing_text.endswith('\n'):
                existing_text += '\n\n'
            self._info_text.setPlainText(existing_text + f"Error loading YAML: {str(e)}")
    
    def _update_dataset_info(self):
        """Update dataset information display."""
        if not self._dataset_structure:
            return
        
        # Check if we should replace or append
        existing_text = self._info_text.toPlainText()
        should_replace = existing_text.strip() == "No dataset loaded" or not existing_text.strip()
        
        info_text = f"Dataset Structure Information\n"
        info_text += f"{'='*50}\n\n"
        
        info_text += f"Type: {'Structured' if self._dataset_structure.is_structured else 'Flat'}\n"
        info_text += f"Total Images: {len(self._dataset_structure.image_files)}\n"
        info_text += f"Total Labels: {len(self._dataset_structure.label_files)}\n"
        info_text += f"Valid Pairs: {len(self._dataset_structure.matched_pairs)}\n"
        
        # Dataset validity check
        info_text += f"\n📋 Dataset Validity Check:\n"
        info_text += f"{'='*30}\n"
        
        # Images without labels
        if self._dataset_structure.images_without_labels:
            action = "will be ignored" if self._ignore_unannotated_checkbox.isChecked() else "will be moved to rejected folder"
            info_text += f"\n⚠️ Images without labels: {len(self._dataset_structure.images_without_labels)} ({action})\n"
            # Show first 5 examples with expected label file
            for i, img in enumerate(self._dataset_structure.images_without_labels[:5]):
                expected_label = img.stem + ".txt"
                info_text += f"  - {img.name} → missing {expected_label}\n"
            if len(self._dataset_structure.images_without_labels) > 5:
                info_text += f"  ... and {len(self._dataset_structure.images_without_labels) - 5} more\n"
        else:
            info_text += f"✅ All images have labels\n"
        
        # Orphaned labels
        if self._dataset_structure.orphaned_labels:
            info_text += f"\n⚠️ Labels without images: {len(self._dataset_structure.orphaned_labels)}\n"
            # Show first 5 examples with expected image files
            for i, lbl in enumerate(self._dataset_structure.orphaned_labels[:5]):
                # Try to guess the expected image extension
                possible_imgs = []
                for ext in IMAGE_EXTENSIONS:
                    possible_imgs.append(lbl.stem + ext)
                info_text += f"  - {lbl.name} → missing image ({', '.join(possible_imgs[:3])}...)\n"
            if len(self._dataset_structure.orphaned_labels) > 5:
                info_text += f"  ... and {len(self._dataset_structure.orphaned_labels) - 5} more\n"
        else:
            info_text += f"✅ All labels have corresponding images\n"
        
        # Summary
        total_issues = len(self._dataset_structure.images_without_labels) + len(self._dataset_structure.orphaned_labels)
        if total_issues > 0:
            info_text += f"\n⚠️ Total issues found: {total_issues} files (will be moved to 'rejected' folder)\n"
        else:
            info_text += f"\n✅ Dataset is valid - no issues found\n"
        
        # Show file extensions found
        extensions = set(f.suffix for f in self._dataset_structure.image_files)
        info_text += f"\nImage Types: {', '.join(sorted(extensions))}\n"
        
        # Show class distribution if available
        if self._dataset_structure.class_distribution:
            dist = self._dataset_structure.class_distribution
            info_text += f"\n📊 Class Distribution Analysis:\n"
            info_text += f"{'='*30}\n"
            
            # Get class names if yaml is loaded
            class_names = {}
            if self._yaml_path and self._yaml_path.exists():
                try:
                    data = load_data_yaml(self._yaml_path)
                    if data and 'names' in data:
                        names = data['names']
                        if isinstance(names, dict):
                            class_names = {int(k): v for k, v in names.items()}
                        else:
                            class_names = {i: name for i, name in enumerate(names)}
                except:
                    pass
            
            # Show annotations per class
            info_text += f"\nAnnotations per class:\n"
            for class_id in sorted(dist.class_counts.keys()):
                class_name = class_names.get(class_id, f"Class {class_id}")
                count = dist.class_counts[class_id]
                img_count = len(set(dist.images_per_class[class_id]))
                info_text += f"  {class_id}: {class_name} - {count} annotations in {img_count} images\n"
            
            # Show class combinations
            info_text += f"\nClass combinations in images:\n"
            for combo, count in sorted(dist.class_combinations.items(), key=lambda x: x[1], reverse=True):
                if combo:
                    class_ids = [int(x) for x in combo.split(',')]
                    class_names_str = ", ".join([class_names.get(cid, f"Class {cid}") for cid in class_ids])
                    info_text += f"  [{combo}] {class_names_str}: {count} images\n"
            
            # Show warning if imbalanced
            if dist.class_counts:
                max_count = max(dist.class_counts.values())
                min_count = min(dist.class_counts.values())
                if max_count > min_count * 3:  # More than 3x difference
                    info_text += f"\n⚠️ Class Imbalance Detected!\n"
                    info_text += f"  Largest class has {max_count} annotations\n"
                    info_text += f"  Smallest class has {min_count} annotations\n"
                    info_text += f"  Consider stratified splitting to maintain balance\n"
        
        # Either replace or append based on context
        if should_replace:
            self._info_text.setPlainText(info_text)
        else:
            # Append to existing text with separator
            if not existing_text.endswith('\n'):
                existing_text += '\n'
            existing_text += '\n'  # Add extra line for separation
            self._info_text.setPlainText(existing_text + info_text)
    
    def _clear_info_display(self):
        """Clear the information display."""
        self._info_text.setPlainText("No dataset loaded")
    
    @pyqtSlot(Path)
    def _on_dataset_manager_loaded(self, yaml_path: Path):
        """Handle dataset loaded in dataset manager."""
        # Only update if we don't have a YAML path set
        if not self._yaml_path:
            self._yaml_path = yaml_path
            self._load_yaml_info()
            
            # If source directory is not set, try to infer it
            if not self._source_dir:
                # Use the parent directory of the yaml file as default
                self._source_dir = yaml_path.parent
                if hasattr(self, '_source_label'):
                    self._source_label.setText(str(self._source_dir))
                    self._source_label.setStyleSheet("color: #cccccc;")
                self._scan_dataset()