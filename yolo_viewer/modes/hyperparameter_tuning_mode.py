"""Hyperparameter tuning mode for automated YOLO model optimization."""

from pathlib import Path
from typing import Optional, Dict, Tuple, List
import yaml
import json
import csv
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit,
    QTextEdit, QSplitter, QFileDialog, QMessageBox, QProgressBar,
    QCheckBox, QScrollArea, QFrame, QSlider, QTableWidget,
    QTableWidgetItem, QHeaderView, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QTimer, QFileSystemWatcher
from PyQt6.QtGui import QFont

# Import matplotlib
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] Matplotlib not available, plots will be disabled")

from .base_mode import BaseMode
from ..core import ModelCache, DatasetManager
from ..widgets.augmentation_settings import InfoLabel
from ..utils.tuning_process import TuningProcess


class SearchSpaceWidget(QWidget):
    """Widget for configuring hyperparameter search space."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parameter_ranges = {}
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the UI for search space configuration."""
        layout = QVBoxLayout(self)
        
        # Preset selector
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Quick", "Standard", "Comprehensive", "Custom"])
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        layout.addLayout(preset_layout)
        
        # Scrollable area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        params_widget = QWidget()
        self.params_layout = QVBoxLayout(params_widget)
        
        # Add parameter ranges
        self._add_parameter_range("Learning Rate (lr0)", "lr0", 0.00001, 0.1, 0.001, 0.01)
        self._add_parameter_range("Final LR Factor (lrf)", "lrf", 0.01, 1.0, 0.01, 0.1)
        self._add_parameter_range("Momentum", "momentum", 0.6, 0.98, 0.6, 0.937)
        self._add_parameter_range("Weight Decay", "weight_decay", 0.0, 0.001, 0.0, 0.0005)
        self._add_parameter_range("Warmup Epochs", "warmup_epochs", 0.0, 5.0, 0.0, 3.0)
        self._add_parameter_range("Box Loss", "box", 0.02, 0.2, 0.02, 0.05)
        self._add_parameter_range("Class Loss", "cls", 0.2, 4.0, 0.5, 1.0)
        self._add_parameter_range("DFL Loss", "dfl", 0.5, 2.0, 1.0, 1.5)
        
        scroll.setWidget(params_widget)
        layout.addWidget(scroll)
        
    def _add_parameter_range(self, label: str, param_name: str, 
                            min_val: float, max_val: float, 
                            default_min: float, default_max: float):
        """Add a parameter range configuration."""
        group = QGroupBox(label)
        group_layout = QVBoxLayout()
        
        # Enable checkbox
        enable_check = QCheckBox(f"Include {label} in tuning")
        enable_check.setChecked(True)
        group_layout.addWidget(enable_check)
        
        # Min value
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min:"))
        min_spin = QDoubleSpinBox()
        min_spin.setRange(min_val, max_val)
        min_spin.setValue(default_min)
        min_spin.setDecimals(5)
        min_spin.setSingleStep(0.00001)
        min_layout.addWidget(min_spin)
        group_layout.addLayout(min_layout)
        
        # Max value
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max:"))
        max_spin = QDoubleSpinBox()
        max_spin.setRange(min_val, max_val)
        max_spin.setValue(default_max)
        max_spin.setDecimals(5)
        max_spin.setSingleStep(0.00001)
        max_layout.addWidget(max_spin)
        group_layout.addLayout(max_layout)
        
        group.setLayout(group_layout)
        self.params_layout.addWidget(group)
        
        # Store references
        self._parameter_ranges[param_name] = {
            'enable': enable_check,
            'min': min_spin,
            'max': max_spin,
            'group': group
        }
        
        # Connect enable checkbox
        enable_check.toggled.connect(lambda checked: self._on_param_toggled(param_name, checked))
        
    def _on_param_toggled(self, param_name: str, enabled: bool):
        """Handle parameter enable/disable."""
        widgets = self._parameter_ranges[param_name]
        widgets['min'].setEnabled(enabled)
        widgets['max'].setEnabled(enabled)
        
    def _on_preset_changed(self, preset: str):
        """Handle preset selection change."""
        if preset == "Quick":
            # Enable only essential parameters
            for param in self._parameter_ranges:
                self._parameter_ranges[param]['enable'].setChecked(param in ['lr0', 'momentum'])
        elif preset == "Standard":
            # Enable common parameters
            for param in self._parameter_ranges:
                self._parameter_ranges[param]['enable'].setChecked(
                    param in ['lr0', 'lrf', 'momentum', 'weight_decay', 'box', 'cls']
                )
        elif preset == "Comprehensive":
            # Enable all parameters
            for param in self._parameter_ranges:
                self._parameter_ranges[param]['enable'].setChecked(True)
        # Custom does nothing - user configures manually
        
    def get_search_space(self) -> Dict[str, Tuple[float, float]]:
        """Get the configured search space."""
        space = {}
        for param_name, widgets in self._parameter_ranges.items():
            if widgets['enable'].isChecked():
                space[param_name] = (widgets['min'].value(), widgets['max'].value())
        return space


class HyperparameterTuningMode(BaseMode):
    """Mode for hyperparameter tuning with genetic algorithms."""
    
    # Custom signals
    tuningStarted = pyqtSignal(str)  # config path
    tuningProgress = pyqtSignal(int, int, dict)  # iteration, total_iterations, metrics
    tuningCompleted = pyqtSignal(str)  # results path
    tuningFailed = pyqtSignal(str)  # error message
    
    def __init__(self, parent=None):
        # Initialize attributes before super().__init__()
        self._dataset_path: Optional[Path] = None
        self._dataset_info: Dict = {}
        self._is_tuning = False
        self._tuning_process = None
        self._current_iteration = 0
        self._total_iterations = 300
        self._best_fitness = 0.0
        self._results_path: Optional[Path] = None
        self.csv_watcher = None
        self.csv_path = None
        
        super().__init__(parent)
        
    def _setup_ui(self):
        """Setup the UI for hyperparameter tuning mode."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Create splitter for three panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Configuration
        left_panel = self._create_config_panel()
        splitter.addWidget(left_panel)
        
        # Center panel - Progress and logging
        center_panel = self._create_progress_panel()
        splitter.addWidget(center_panel)
        
        # Right panel - Results
        right_panel = self._create_results_panel()
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (30%, 40%, 30%)
        splitter.setSizes([400, 500, 400])
        
        main_layout.addWidget(splitter)
        
    def _create_config_panel(self) -> QWidget:
        """Create the configuration panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Dataset configuration (TOP PRIORITY)
        dataset_group = QGroupBox("Dataset Configuration")
        dataset_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        dataset_layout = QVBoxLayout()
        
        # Dataset selection
        dataset_select_layout = QHBoxLayout()
        dataset_select_layout.addWidget(QLabel("Dataset:"))
        self.dataset_path_edit = QLineEdit()
        self.dataset_path_edit.setReadOnly(True)
        self.dataset_path_edit.setPlaceholderText("No dataset selected")
        dataset_select_layout.addWidget(self.dataset_path_edit)
        self.browse_dataset_btn = QPushButton("Browse")
        self.browse_dataset_btn.clicked.connect(self._on_browse_dataset)
        dataset_select_layout.addWidget(self.browse_dataset_btn)
        dataset_layout.addLayout(dataset_select_layout)
        
        # Dataset info display
        self.dataset_info_label = QLabel("Please select a dataset YAML file")
        self.dataset_info_label.setWordWrap(True)
        self.dataset_info_label.setStyleSheet("QLabel { padding: 5px; background: rgba(255,255,255,10); }")
        dataset_layout.addWidget(self.dataset_info_label)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Model configuration
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        # Model selection
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", 
            "yolov8l.pt", "yolov8x.pt"
        ])
        model_select_layout.addWidget(self.model_combo)
        
        self.use_loaded_check = QCheckBox("Use loaded model")
        model_select_layout.addWidget(self.use_loaded_check)
        model_layout.addLayout(model_select_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Tuning parameters
        tuning_group = QGroupBox("Tuning Parameters")
        tuning_layout = QVBoxLayout()
        
        # Epochs per iteration
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs per iteration:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 100)
        self.epochs_spin.setValue(30)
        epochs_layout.addWidget(self.epochs_spin)
        epochs_layout.addWidget(InfoLabel(
            "ℹ", 
            "Number of training epochs for each hyperparameter combination.\n"
            "Lower values = faster but less accurate\n"
            "Higher values = slower but more accurate"
        ))
        tuning_layout.addLayout(epochs_layout)
        
        # Number of iterations
        iterations_layout = QHBoxLayout()
        iterations_layout.addWidget(QLabel("Iterations:"))
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(10, 1000)
        self.iterations_spin.setValue(300)
        iterations_layout.addWidget(self.iterations_spin)
        iterations_layout.addWidget(InfoLabel(
            "ℹ", 
            "Number of hyperparameter combinations to test.\n"
            "More iterations = better optimization but longer time"
        ))
        tuning_layout.addLayout(iterations_layout)
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        self.batch_spin.setSingleStep(2)
        batch_layout.addWidget(self.batch_spin)
        batch_layout.addWidget(InfoLabel(
            "ℹ", 
            "Batch size for training.\n"
            "Larger = faster training but more memory\n"
            "Smaller = slower but less memory\n"
            "Common values: 8, 16, 32"
        ))
        tuning_layout.addLayout(batch_layout)
        
        # Image size
        image_size_layout = QHBoxLayout()
        image_size_layout.addWidget(QLabel("Image Size:"))
        self.image_size_combo = QComboBox()
        self.image_size_combo.setEditable(True)
        self.image_size_combo.addItems(["640", "1280", "1920"])
        self.image_size_combo.setCurrentText("640")
        image_size_layout.addWidget(self.image_size_combo)
        image_size_layout.addWidget(InfoLabel(
            "ℹ", 
            "Input image size for training.\n"
            "Must be multiple of 32\n"
            "Larger = better accuracy but slower\n"
            "Common values: 640, 1280, 1920"
        ))
        tuning_layout.addLayout(image_size_layout)
        
        # Optimizer selection
        optimizer_layout = QHBoxLayout()
        optimizer_layout.addWidget(QLabel("Optimizer:"))
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["AdamW", "Adam", "SGD", "RMSprop"])
        optimizer_layout.addWidget(self.optimizer_combo)
        tuning_layout.addLayout(optimizer_layout)
        
        tuning_group.setLayout(tuning_layout)
        layout.addWidget(tuning_group)
        
        # Search space configuration (collapsible)
        search_group = QGroupBox("Search Space Configuration")
        search_layout = QVBoxLayout()
        
        self.search_space_widget = SearchSpaceWidget()
        search_layout.addWidget(self.search_space_widget)
        
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)
        
        # Resume options (hidden by default)
        self.resume_group = QGroupBox("Resume Previous Session")
        self.resume_group.setVisible(False)
        resume_layout = QVBoxLayout()
        
        self.resume_combo = QComboBox()
        resume_layout.addWidget(QLabel("Select session to resume:"))
        resume_layout.addWidget(self.resume_combo)
        
        self.resume_info_label = QLabel()
        self.resume_info_label.setWordWrap(True)
        resume_layout.addWidget(self.resume_info_label)
        
        self.resume_group.setLayout(resume_layout)
        layout.addWidget(self.resume_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Tuning")
        self.start_btn.clicked.connect(self._on_start_tuning)
        self.start_btn.setEnabled(False)  # Disabled until dataset is selected
        button_layout.addWidget(self.start_btn)
        
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.clicked.connect(self._on_resume_tuning)
        self.resume_btn.setEnabled(False)
        self.resume_btn.setVisible(False)  # Hidden until resumable sessions found
        button_layout.addWidget(self.resume_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop_tuning)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
        layout.addStretch()
        return panel
        
    def _create_progress_panel(self) -> QWidget:
        """Create the progress and logging panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Progress section
        progress_group = QGroupBox("Tuning Progress")
        progress_layout = QVBoxLayout()
        
        # Overall progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        # Status labels
        self.iteration_label = QLabel("Iteration: 0 / 0")
        progress_layout.addWidget(self.iteration_label)
        
        # Best fitness with info
        fitness_layout = QHBoxLayout()
        self.best_fitness_label = QLabel("Best Fitness: N/A")
        fitness_layout.addWidget(self.best_fitness_label)
        
        # Add info button for fitness explanation
        fitness_info = InfoLabel(
            "Fitness Metric",
            "Fitness is a weighted combination of model accuracy metrics:\n\n"
            "Fitness = 0.1 × mAP50 + 0.9 × mAP50-95\n\n"
            "• mAP50: Mean Average Precision at IoU threshold 0.5\n"
            "• mAP50-95: Mean Average Precision averaged from IoU 0.5 to 0.95\n\n"
            "Higher fitness values indicate better model performance.\n"
            "The genetic algorithm optimizes hyperparameters to maximize this metric."
        )
        fitness_layout.addWidget(fitness_info)
        fitness_layout.addStretch()
        
        progress_layout.addLayout(fitness_layout)
        
        # Time tracking labels  
        time_layout = QHBoxLayout()
        self.time_label = QLabel("Time Elapsed: 00:00:00")
        time_layout.addWidget(self.time_label)
        time_layout.addStretch()
        self.eta_label = QLabel("ETA: Calculating...")
        time_layout.addWidget(self.eta_label)
        progress_layout.addLayout(time_layout)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Current parameters
        params_group = QGroupBox("Current Parameters Being Tested")
        params_layout = QVBoxLayout()
        
        self.current_params_text = QTextEdit()
        self.current_params_text.setReadOnly(True)
        self.current_params_text.setMaximumHeight(150)
        params_layout.addWidget(self.current_params_text)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Iterations table
        iterations_group = QGroupBox("Iteration History")
        iterations_layout = QVBoxLayout()
        
        self.iteration_table = QTableWidget()
        self.iteration_table.setColumnCount(3)
        self.iteration_table.setHorizontalHeaderLabels(["Iteration", "Fitness", "Parameters"])
        self.iteration_table.horizontalHeader().setStretchLastSection(True)
        self.iteration_table.setMaximumHeight(200)
        iterations_layout.addWidget(self.iteration_table)
        
        iterations_group.setLayout(iterations_layout)
        layout.addWidget(iterations_group)
        
        # Console output
        console_group = QGroupBox("Console Output")
        console_layout = QVBoxLayout()
        
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setFont(QFont("Consolas", 9))
        console_layout.addWidget(self.console_output)
        
        console_group.setLayout(console_layout)
        layout.addWidget(console_group)
        
        return panel
        
    def _create_results_panel(self) -> QWidget:
        """Create the results visualization panel."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Best parameters tab
        best_params_widget = QWidget()
        best_params_layout = QVBoxLayout(best_params_widget)
        
        # Results text display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText("Best parameters will appear here after tuning...")
        best_params_layout.addWidget(self.results_text)
        
        # Export button
        export_btn = QPushButton("Export Best Parameters")
        export_btn.clicked.connect(self._on_export_results)
        best_params_layout.addWidget(export_btn)
        
        self.results_tabs.addTab(best_params_widget, "Best Parameters")
        
        # Plots tab
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        
        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib figure and canvas
            self.plot_figure = Figure(figsize=(10, 6))
            self.plot_canvas = FigureCanvas(self.plot_figure)
            plots_layout.addWidget(self.plot_canvas)
            
            # CSV watcher for live updates
            self.csv_watcher = None
            self.csv_path = None
        else:
            self.plots_label = QLabel("Matplotlib not installed. Run: pip install matplotlib")
            self.plots_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            plots_layout.addWidget(self.plots_label)
            
        self.results_tabs.addTab(plots_widget, "Plots")
        
        # History tab
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["Iteration", "Fitness", "mAP50", "Time"])
        history_layout.addWidget(self.history_table)
        
        self.results_tabs.addTab(history_widget, "History")
        
        layout.addWidget(self.results_tabs)
        
        # Open results folder button
        self.open_results_btn = QPushButton("Open Results Folder")
        self.open_results_btn.clicked.connect(self._on_open_results_folder)
        self.open_results_btn.setEnabled(False)
        layout.addWidget(self.open_results_btn)
        
        return panel
        
    def _on_browse_dataset(self):
        """Handle dataset browsing."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Dataset YAML",
            "",
            "YAML Files (*.yaml *.yml)"
        )
        
        if file_path:
            self._load_dataset(Path(file_path))
            
    def _check_resumable_sessions(self):
        """Check for resumable tuning sessions."""
        runs_dir = Path.cwd() / "runs" / "tuning"
        if not runs_dir.exists():
            return
            
        resumable_sessions = []
        
        # Scan for tune directories
        for session_dir in runs_dir.glob("tune_*"):
            if session_dir.is_dir():
                # Check if this session has a tune subdirectory with results
                tune_dir = session_dir / "tune"
                if tune_dir.exists():
                    # Check for results CSV to see progress
                    results_csv = tune_dir / "tune_results.csv"
                    if results_csv.exists():
                        try:
                            # Read CSV to get iteration count
                            with open(results_csv, 'r') as f:
                                lines = f.readlines()
                                iterations_done = len(lines) - 1  # Subtract header
                                
                            # Get session info
                            session_info = {
                                'name': session_dir.name,
                                'path': session_dir,
                                'iterations_done': iterations_done,
                                'timestamp': session_dir.stat().st_mtime
                            }
                            
                            # Check if there's a config saved
                            config_file = session_dir / "tuning_config.json"
                            if config_file.exists():
                                with open(config_file, 'r') as f:
                                    session_info['config'] = json.load(f)
                            
                            resumable_sessions.append(session_info)
                        except Exception as e:
                            self._log(f"Error checking session {session_dir.name}: {e}")
        
        # Sort by timestamp (most recent first)
        resumable_sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Update UI if sessions found
        if resumable_sessions:
            self.resume_group.setVisible(True)
            self.resume_btn.setVisible(True)
            self.resume_combo.clear()
            
            for session in resumable_sessions:
                timestamp = datetime.fromtimestamp(session['timestamp']).strftime("%Y-%m-%d %H:%M")
                label = f"{session['name']} - {session['iterations_done']} iterations done ({timestamp})"
                self.resume_combo.addItem(label, session)
                
            # Show info for selected session
            self._on_resume_session_changed()
            self.resume_combo.currentIndexChanged.connect(self._on_resume_session_changed)
            
    def _on_resume_session_changed(self):
        """Update info when resume session selection changes."""
        if self.resume_combo.count() == 0:
            return
            
        session = self.resume_combo.currentData()
        if session and 'config' in session:
            config = session['config']
            info = f"Dataset: {Path(config.get('data', 'Unknown')).name}\n"
            info += f"Model: {Path(config.get('model', 'Unknown')).name}\n"
            info += f"Total iterations: {config.get('iterations', 'Unknown')}\n"
            info += f"Completed: {session['iterations_done']}"
            self.resume_info_label.setText(info)
            
            # Enable resume button if valid
            self.resume_btn.setEnabled(session['iterations_done'] < config.get('iterations', 0))
            
    def _load_dataset(self, yaml_path: Path):
        """Load and validate dataset."""
        try:
            # Load YAML
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                
            # Validate required fields
            required_fields = ['train', 'val', 'nc', 'names']
            missing = [f for f in required_fields if f not in data]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")
                
            # Store dataset info
            self._dataset_path = yaml_path
            self._dataset_info = data
            
            # Update UI
            self.dataset_path_edit.setText(str(yaml_path))
            
            # Count images if possible
            dataset_dir = yaml_path.parent
            
            # Handle 'path' field if present (YOLO dataset structure)
            if 'path' in data:
                # Check if path is absolute or relative
                base_path = Path(data['path'])
                if base_path.is_absolute():
                    dataset_root = base_path
                else:
                    dataset_root = dataset_dir / data['path']
            else:
                dataset_root = dataset_dir
            
            # Get train and val paths
            train_rel = data.get('train', 'train')
            val_rel = data.get('val', 'val')
            
            # Build full paths
            train_path = dataset_root / train_rel if not Path(train_rel).is_absolute() else Path(train_rel)
            val_path = dataset_root / val_rel if not Path(val_rel).is_absolute() else Path(val_rel)
            
            # Count images in the directories
            train_count = 0
            val_count = 0
            
            if train_path.exists():
                # Check for images directly in the folder
                train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png')) + \
                              list(train_path.glob('*.jpeg')) + list(train_path.glob('*.tif')) + \
                              list(train_path.glob('*.tiff'))
                train_count = len(train_images)
                
                # If no images found, check for 'images' subfolder (common YOLO structure)
                if train_count == 0:
                    images_path = train_path / 'images'
                    if images_path.exists():
                        train_images = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')) + \
                                     list(images_path.glob('*.jpeg')) + list(images_path.glob('*.tif')) + \
                                     list(images_path.glob('*.tiff'))
                        train_count = len(train_images)
                        
            if val_path.exists():
                # Check for images directly in the folder
                val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png')) + \
                            list(val_path.glob('*.jpeg')) + list(val_path.glob('*.tif')) + \
                            list(val_path.glob('*.tiff'))
                val_count = len(val_images)
                
                # If no images found, check for 'images' subfolder
                if val_count == 0:
                    images_path = val_path / 'images'
                    if images_path.exists():
                        val_images = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')) + \
                                   list(images_path.glob('*.jpeg')) + list(images_path.glob('*.tif')) + \
                                   list(images_path.glob('*.tiff'))
                        val_count = len(val_images)
            
            # Debug logging
            self._log(f"[DEBUG] Dataset root: {dataset_root}")
            self._log(f"[DEBUG] Train path: {train_path} (exists: {train_path.exists()})")
            self._log(f"[DEBUG] Val path: {val_path} (exists: {val_path.exists()})")
            
            # Update info label
            info_text = f"✓ Dataset loaded successfully\n"
            info_text += f"Classes: {data['nc']} ({', '.join(data['names'].values() if isinstance(data['names'], dict) else data['names'][:3])}...)\n"
            info_text += f"Training images: {train_count}\n"
            info_text += f"Validation images: {val_count}"
            
            self.dataset_info_label.setText(info_text)
            self.dataset_info_label.setStyleSheet("QLabel { padding: 5px; background: rgba(0,255,0,20); }")
            
            # Enable start button
            self.start_btn.setEnabled(True)
            
            self._log(f"Dataset loaded: {yaml_path.name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Dataset Error", f"Failed to load dataset:\n{str(e)}")
            self.dataset_info_label.setText(f"Error: {str(e)}")
            self.dataset_info_label.setStyleSheet("QLabel { padding: 5px; background: rgba(255,0,0,20); }")
            self.start_btn.setEnabled(False)
            
    def _on_start_tuning(self):
        """Start hyperparameter tuning."""
        # Validate dataset
        if not self._dataset_path:
            QMessageBox.warning(self, "No Dataset", "Please select a dataset YAML file first")
            return
            
        # Get model
        model_cache = ModelCache()
        if self.use_loaded_check.isChecked() and model_cache.is_loaded():
            model = str(model_cache.model_path)
        else:
            model = self.model_combo.currentText()
            
        # Get image size (validate it's a multiple of 32)
        try:
            image_size = int(self.image_size_combo.currentText())
            if image_size % 32 != 0:
                QMessageBox.warning(self, "Invalid Image Size", 
                                  f"Image size must be a multiple of 32. Got: {image_size}")
                return
        except ValueError:
            QMessageBox.warning(self, "Invalid Image Size", 
                              "Please enter a valid integer for image size")
            return
            
        # Prepare configuration
        config = {
            'model': model,
            'data': str(self._dataset_path),
            'epochs': self.epochs_spin.value(),
            'iterations': self.iterations_spin.value(),
            'batch': self.batch_spin.value(),
            'imgsz': image_size,
            'optimizer': self.optimizer_combo.currentText(),
            'space': self.search_space_widget.get_search_space(),
            'plots': True,
            'save': True,
            'val': True
        }
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path.cwd() / "runs" / "tuning" / f"tune_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config for potential resume
        config_file = output_dir / "tuning_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create tuning process
        self._tuning_process = TuningProcess(self)
        self._tuning_process.logMessage.connect(self._log)
        self._tuning_process.progressUpdate.connect(self._update_progress)
        self._tuning_process.iterationComplete.connect(self._on_iteration_complete)
        self._tuning_process.tuningComplete.connect(self._on_tuning_complete)
        self._tuning_process.errorOccurred.connect(self._on_tuning_error)
        self._tuning_process.bestFitnessUpdate.connect(self._on_best_fitness_update)
        self._tuning_process.timeUpdate.connect(self._on_time_update)
        
        # Update UI state
        self._is_tuning = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_dataset_btn.setEnabled(False)
        self._total_iterations = config['iterations']
        
        # Clear previous results
        self.console_output.clear()
        self.results_text.clear()
        self.iteration_table.setRowCount(0)
        
        # Start tuning
        self._log("Starting hyperparameter tuning...")
        self._log(f"Dataset: {self._dataset_path.name}")
        self._log(f"Model: {model}")
        self._log(f"Iterations: {config['iterations']}")
        self._log(f"Epochs per iteration: {config['epochs']}")
        self._log(f"Batch size: {config['batch']}")
        self._log(f"Image size: {config['imgsz']}")
        self._log(f"Optimizer: {config['optimizer']}")
        self._log(f"Search space: {len(config['space'])} parameters")
        self._log(f"Output directory: {output_dir}")
        
        # Start the tuning process
        if not self._tuning_process.start_tuning(config, output_dir):
            self._on_tuning_error("Failed to start tuning process")
            self._reset_ui_state()
        else:
            self.tuningStarted.emit(str(self._dataset_path))
            # Start monitoring for CSV updates (give it a moment to be created)
            QTimer.singleShot(2000, lambda: self._start_csv_monitoring(output_dir / "tune"))
        
    def _on_resume_tuning(self):
        """Resume a previous tuning session."""
        if self.resume_combo.count() == 0:
            return
            
        session = self.resume_combo.currentData()
        if not session or 'config' not in session:
            QMessageBox.warning(self, "Invalid Session", "Cannot resume this session - missing configuration")
            return
            
        config = session['config']
        session_path = session['path']
        
        # Create tuning process
        self._tuning_process = TuningProcess(self)
        self._tuning_process.logMessage.connect(self._log)
        self._tuning_process.progressUpdate.connect(self._update_progress)
        self._tuning_process.iterationComplete.connect(self._on_iteration_complete)
        self._tuning_process.tuningComplete.connect(self._on_tuning_complete)
        self._tuning_process.errorOccurred.connect(self._on_tuning_error)
        self._tuning_process.bestFitnessUpdate.connect(self._on_best_fitness_update)
        self._tuning_process.timeUpdate.connect(self._on_time_update)
        
        # Update UI state
        self._is_tuning = True
        self.start_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_dataset_btn.setEnabled(False)
        self._total_iterations = config['iterations']
        self._current_iteration = session['iterations_done']
        
        # Clear previous results
        self.console_output.clear()
        self.results_text.clear()
        
        # Start resumed tuning
        self._log(f"Resuming tuning session: {session['name']}")
        self._log(f"Continuing from iteration {session['iterations_done']}/{config['iterations']}")
        
        # Set the dataset path from config
        if 'data' in config:
            self._dataset_path = Path(config['data'])
            
        # Start the resumed tuning process with resume flag
        if not self._tuning_process.start_tuning(config, session_path, resume=True):
            self._on_tuning_error("Failed to resume tuning process")
            self._reset_ui_state()
        else:
            # Emit signal
            if self._dataset_path:
                self.tuningStarted.emit(str(self._dataset_path))
            # Start monitoring for CSV updates
            QTimer.singleShot(2000, lambda: self._start_csv_monitoring(session_path / "tune"))
            # Load existing results
            self._load_results_from_path(session_path / "tune")
        
    def _on_time_update(self, elapsed_time: str, eta: str):
        """Update time tracking displays."""
        self.time_label.setText(f"Time Elapsed: {elapsed_time}")
        self.eta_label.setText(f"ETA: {eta}")
        
    def _on_stop_tuning(self):
        """Stop hyperparameter tuning."""
        if self._is_tuning and self._tuning_process:
            reply = QMessageBox.question(
                self, 
                "Stop Tuning",
                f"Stop tuning at iteration {self._current_iteration}/{self._total_iterations}?\n\n"
                f"The best parameters found so far will be saved.\n"
                f"You can export the results or resume later.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._log("Stopping tuning and saving results...")
                self._tuning_process.stop_tuning()
                # Don't reset UI here - wait for process to finish and save
                
    def _on_export_results(self):
        """Export tuning results."""
        if not self._results_path:
            QMessageBox.information(self, "No Results", "No tuning results available to export")
            return
            
        # Try to find the best parameters from CSV or existing YAML
        best_params = self._extract_best_parameters()
        
        if not best_params:
            QMessageBox.warning(self, "No Parameters", "Could not find best parameters to export")
            return
            
        # Ask user where to save
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Export Best Hyperparameters",
            "best_hyperparameters.yaml",
            "YAML Files (*.yaml *.yml)"
        )
        
        if file_path:
            try:
                # Add header with fitness info
                export_data = {
                    '# Generated by YOLO Hyperparameter Tuning': None,
                    f'# Best Fitness': self._best_fitness,
                    f'# Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    '# Hyperparameters': None,
                    **best_params
                }
                
                with open(file_path, 'w') as f:
                    # Write with comments
                    f.write("# Best Hyperparameters from Tuning\n")
                    f.write(f"# Fitness: {self._best_fitness:.4f}\n")
                    f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # Write parameters
                    yaml.dump(best_params, f, default_flow_style=False, sort_keys=False)
                    
                self._log(f"Exported best parameters to: {file_path}")
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Best hyperparameters exported to:\n{file_path}\n\n"
                    f"Fitness: {self._best_fitness:.4f}"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export parameters:\n{e}")
                
    def _extract_best_parameters(self) -> Optional[Dict]:
        """Extract best parameters from results."""
        # First try existing best_hyperparameters.yaml
        if self._results_path:
            best_params_file = self._results_path / 'best_hyperparameters.yaml'
            if best_params_file.exists():
                try:
                    with open(best_params_file, 'r') as f:
                        return yaml.safe_load(f)
                except:
                    pass
                    
        # Try to extract from CSV
        if self._results_path:
            csv_path = self._results_path / "tune_results.csv"
            if not csv_path.exists():
                csv_path = self._results_path.parent / "tune_results.csv"
                
            if csv_path.exists():
                try:
                    best_row = None
                    best_fitness = -1
                    
                    with open(csv_path, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            fitness = float(row.get('fitness', 0))
                            if fitness > best_fitness:
                                best_fitness = fitness
                                best_row = row
                                
                    if best_row:
                        # Extract hyperparameters from best row
                        params = {}
                        for key, value in best_row.items():
                            # Skip non-parameter columns
                            if key in ['iteration', 'fitness', 'time', 'runtime']:
                                continue
                            if key.startswith('metrics/'):
                                continue
                            # Try to convert to appropriate type
                            try:
                                if '.' in value:
                                    params[key] = float(value)
                                else:
                                    params[key] = int(value)
                            except:
                                params[key] = value
                        return params
                        
                except Exception as e:
                    print(f"[ERROR] Failed to extract from CSV: {e}")
                    
        return None
        
    def _on_open_results_folder(self):
        """Open the results folder in file explorer."""
        if self._results_path and self._results_path.exists():
            import os
            import platform
            if platform.system() == 'Windows':
                os.startfile(self._results_path)
            elif platform.system() == 'Darwin':  # macOS
                os.system(f'open "{self._results_path}"')
            else:  # Linux
                os.system(f'xdg-open "{self._results_path}"')
                
    def _log(self, message: str):
        """Add message to console output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_output.append(f"[{timestamp}] {message}")
        
    def _update_progress(self, current: int, total: int):
        """Update progress bar."""
        self._current_iteration = current
        self._total_iterations = total
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.iteration_label.setText(f"Iteration {current}/{total}")
        
    def _on_iteration_complete(self, iteration: int, fitness: float, params: Dict):
        """Handle iteration completion."""
        # Add to table
        row = self.iteration_table.rowCount()
        self.iteration_table.insertRow(row)
        self.iteration_table.setItem(row, 0, QTableWidgetItem(str(iteration)))
        self.iteration_table.setItem(row, 1, QTableWidgetItem(f"{fitness:.4f}"))
        
        # Show truncated params
        params_str = json.dumps(params, indent=2)
        if len(params_str) > 100:
            params_str = params_str[:97] + "..."
        self.iteration_table.setItem(row, 2, QTableWidgetItem(params_str))
        
        # Auto-scroll to latest
        self.iteration_table.scrollToBottom()
        
    def _on_best_fitness_update(self, fitness: float, params: Dict):
        """Handle best fitness update."""
        self._best_fitness = fitness
        self.best_fitness_label.setText(f"Best Fitness: {fitness:.4f}")
        
        # Update results display
        results_text = f"Best Fitness: {fitness:.4f}\n\n"
        results_text += "Best Parameters:\n"
        results_text += json.dumps(params, indent=2)
        self.results_text.setPlainText(results_text)
        
    def _on_tuning_complete(self, results_path: str):
        """Handle tuning completion (full or partial)."""
        self._results_path = Path(results_path)
        
        # Check if this was a user stop or full completion
        was_stopped = self._current_iteration < self._total_iterations
        
        # Don't fully reset UI - keep progress visible
        self._is_tuning = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_dataset_btn.setEnabled(True)
        
        if was_stopped:
            self._log(f"Tuning stopped at iteration {self._current_iteration}/{self._total_iterations}")
            self._log(f"Partial results saved to: {results_path}")
            status_msg = f"Stopped at iteration {self._current_iteration}/{self._total_iterations}"
        else:
            self._log(f"Tuning completed all {self._total_iterations} iterations!")
            self._log(f"Results saved to: {results_path}")
            status_msg = "Completed successfully"
        
        # Load and display results (partial or full)
        self._load_results_from_path(self._results_path)
        
        # Update iteration label to show stopped state
        if was_stopped:
            self.iteration_label.setText(f"Stopped: {self._current_iteration}/{self._total_iterations}")
        
        # Show appropriate message
        if was_stopped:
            QMessageBox.information(
                self,
                "Tuning Stopped",
                f"Hyperparameter tuning stopped at iteration {self._current_iteration}.\n\n"
                f"Best fitness so far: {self._best_fitness:.4f}\n"
                f"Partial results saved to: {results_path}\n\n"
                f"You can export the best parameters found or resume later."
            )
        else:
            QMessageBox.information(
                self,
                "Tuning Complete",
                f"Hyperparameter tuning completed all {self._total_iterations} iterations!\n\n"
                f"Best fitness: {self._best_fitness:.4f}\n"
                f"Results saved to: {results_path}"
            )
        
    def _on_tuning_error(self, error_msg: str):
        """Handle tuning error."""
        # Don't show error dialog for certain expected messages
        if "crashed" in error_msg.lower() or "failed" in error_msg.lower():
            self._log(f"[ERROR] {error_msg}")
            self._reset_ui_state()
            
            QMessageBox.critical(
                self,
                "Tuning Error",
                f"An error occurred during tuning:\n{error_msg}"
            )
        else:
            # Just log it without dialog (might be informational)
            self._log(f"[INFO] {error_msg}")
        
    def _reset_ui_state(self):
        """Reset UI to idle state."""
        self._is_tuning = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_dataset_btn.setEnabled(True)
        
    def _load_results_from_path(self, results_path: Path):
        """Load results from a tuning run."""
        try:
            # Look for best parameters file
            best_params_file = results_path / 'best_hyperparameters.yaml'
            if best_params_file.exists():
                with open(best_params_file, 'r') as f:
                    best_params = yaml.safe_load(f)
                    
                # Display in results
                results_text = f"Results from: {results_path.name}\n\n"
                results_text += "Best Hyperparameters:\n"
                results_text += json.dumps(best_params, indent=2)
                self.results_text.setPlainText(results_text)
                
                self._log(f"Loaded results from {results_path}")
            
            # Start CSV monitoring for plots
            self._start_csv_monitoring(results_path)
            
            # Populate history table
            self._populate_history_table(results_path)
                
        except Exception as e:
            self._log(f"Error loading results: {e}")
        
    def _on_activate(self):
        """Called when mode is activated."""
        # Check if a model is loaded
        model_cache = ModelCache()
        if model_cache.is_loaded():
            self.use_loaded_check.setEnabled(True)
            self.use_loaded_check.setText(f"Use loaded model ({Path(model_cache.model_path).name})")
        else:
            self.use_loaded_check.setEnabled(False)
            self.use_loaded_check.setText("Use loaded model (none loaded)")
            
        # Check for resumable sessions
        self._check_resumable_sessions()
            
    def _on_deactivate(self) -> Optional[bool]:
        """Called when mode is deactivated."""
        if self._is_tuning:
            reply = QMessageBox.question(
                self,
                "Tuning in Progress",
                "Hyperparameter tuning is in progress. Do you want to stop it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return False  # Cancel deactivation
            else:
                self._on_stop_tuning()
        return True
        
    def _start_csv_monitoring(self, results_path: Path):
        """Start monitoring CSV file for plot updates."""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        csv_path = results_path / "tune_results.csv"
        if not csv_path.exists():
            # Try parent directory
            csv_path = results_path.parent / "tune_results.csv"
            
        if csv_path.exists():
            self.csv_path = csv_path
            print(f"[DEBUG] Monitoring CSV: {csv_path}")
            
            # Set up file watcher
            if self.csv_watcher:
                self.csv_watcher.deleteLater()
                
            self.csv_watcher = QFileSystemWatcher()
            self.csv_watcher.addPath(str(csv_path))
            self.csv_watcher.fileChanged.connect(self._update_plots)
            
            # Initial plot update
            self._update_plots()
            
    def _update_plots(self):
        """Update plots from CSV data."""
        if not MATPLOTLIB_AVAILABLE or not self.csv_path:
            return
            
        try:
            # Read CSV data
            iterations = []
            fitness_values = []
            map50_values = []
            map50_95_values = []
            
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        iterations.append(int(row.get('iteration', 0)))
                        fitness_values.append(float(row.get('fitness', 0)))
                        map50_values.append(float(row.get('metrics/mAP50(B)', 0)))
                        map50_95_values.append(float(row.get('metrics/mAP50-95(B)', 0)))
                    except (ValueError, KeyError):
                        continue
                        
            if not iterations:
                return
                
            # Clear and create subplots
            self.plot_figure.clear()
            
            # Create 2x2 subplot grid
            ax1 = self.plot_figure.add_subplot(2, 2, 1)
            ax2 = self.plot_figure.add_subplot(2, 2, 2)
            ax3 = self.plot_figure.add_subplot(2, 2, 3)
            ax4 = self.plot_figure.add_subplot(2, 2, 4)
            
            # Plot fitness
            ax1.plot(iterations, fitness_values, 'b-', marker='o', markersize=4)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Fitness Progress')
            ax1.grid(True, alpha=0.3)
            
            # Highlight best fitness
            if fitness_values:
                best_idx = fitness_values.index(max(fitness_values))
                ax1.plot(iterations[best_idx], fitness_values[best_idx], 
                        'r*', markersize=15, label=f'Best: {fitness_values[best_idx]:.4f}')
                ax1.legend()
            
            # Plot mAP50
            ax2.plot(iterations, map50_values, 'g-', marker='s', markersize=4)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('mAP50')
            ax2.set_title('mAP50 Progress')
            ax2.grid(True, alpha=0.3)
            
            # Plot mAP50-95
            ax3.plot(iterations, map50_95_values, 'm-', marker='^', markersize=4)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('mAP50-95')
            ax3.set_title('mAP50-95 Progress')
            ax3.grid(True, alpha=0.3)
            
            # Combined metrics
            ax4.plot(iterations, fitness_values, 'b-', label='Fitness', alpha=0.7)
            ax4.plot(iterations, map50_values, 'g-', label='mAP50', alpha=0.7)
            ax4.plot(iterations, map50_95_values, 'm-', label='mAP50-95', alpha=0.7)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Value')
            ax4.set_title('All Metrics')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Adjust layout and refresh
            self.plot_figure.tight_layout()
            self.plot_canvas.draw()
            
            print(f"[DEBUG] Updated plots with {len(iterations)} iterations")
            
        except Exception as e:
            print(f"[ERROR] Failed to update plots: {e}")
            
    def _populate_history_table(self, results_path: Path):
        """Populate history table from CSV data."""
        csv_path = results_path / "tune_results.csv"
        if not csv_path.exists():
            csv_path = results_path.parent / "tune_results.csv"
            
        if not csv_path.exists():
            return
            
        try:
            # Clear existing rows
            self.history_table.setRowCount(0)
            
            # Read CSV and populate table
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_idx = self.history_table.rowCount()
                    self.history_table.insertRow(row_idx)
                    
                    # Add data to columns
                    self.history_table.setItem(row_idx, 0, 
                        QTableWidgetItem(str(row.get('iteration', ''))))
                    self.history_table.setItem(row_idx, 1, 
                        QTableWidgetItem(f"{float(row.get('fitness', 0)):.4f}"))
                    self.history_table.setItem(row_idx, 2, 
                        QTableWidgetItem(f"{float(row.get('metrics/mAP50(B)', 0)):.4f}"))
                    
                    # Add time if available
                    time_str = row.get('time', '')
                    if not time_str and 'runtime' in row:
                        time_str = row['runtime']
                    self.history_table.setItem(row_idx, 3, QTableWidgetItem(time_str))
                    
            # Highlight best row
            if self.history_table.rowCount() > 0:
                best_fitness = -1
                best_row = -1
                for i in range(self.history_table.rowCount()):
                    fitness_item = self.history_table.item(i, 1)
                    if fitness_item:
                        fitness = float(fitness_item.text())
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_row = i
                            
                # Highlight best row in light green
                if best_row >= 0:
                    for col in range(self.history_table.columnCount()):
                        item = self.history_table.item(best_row, col)
                        if item:
                            item.setBackground(Qt.GlobalColor.lightGray)
                            
            print(f"[DEBUG] Populated history table with {self.history_table.rowCount()} rows")
            
        except Exception as e:
            print(f"[ERROR] Failed to populate history table: {e}")
    
    def get_mode_name(self) -> str:
        """Get the display name of this mode."""
        return "Hyper Param Tuning"