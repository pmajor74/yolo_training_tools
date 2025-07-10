"""Model Management mode for loading and managing YOLO models."""

from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QMessageBox, QGroupBox, QTextEdit, QSplitter, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QThread, QTimer, QDateTime
from PyQt6.QtGui import QFont

from .base_mode import BaseMode
from ..core import ModelCache

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class DateTableWidgetItem(QTableWidgetItem):
    """Custom table widget item for proper date sorting."""
    
    def __init__(self, text: str, sort_date: Optional[datetime] = None):
        super().__init__(text)
        self._sort_date = sort_date
    
    def __lt__(self, other):
        """Custom sorting comparison."""
        if isinstance(other, DateTableWidgetItem):
            # Handle None values
            if self._sort_date is None and other._sort_date is None:
                return self.text() < other.text()
            elif self._sort_date is None:
                return True  # None comes first
            elif other._sort_date is None:
                return False
            else:
                return self._sort_date < other._sort_date
        return super().__lt__(other)


class ModelDiscoveryThread(QThread):
    """Thread for discovering models in project directory."""
    
    modelFound = pyqtSignal(str)  # path
    finished = pyqtSignal()
    
    def __init__(self, search_path: Path):
        super().__init__()
        self.search_path = search_path
        self._is_running = True
    
    def run(self):
        """Search for .pt files."""
        try:
            # Walk through directory tree
            for root, dirs, files in os.walk(self.search_path):
                if not self._is_running:
                    break
                    
                # Skip hidden directories and common excluded folders
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', '__pycache__', 'node_modules']]
                
                # Check each file
                for file in files:
                    if not self._is_running:
                        break
                        
                    if file.endswith('.pt'):
                        full_path = os.path.join(root, file)
                        self.modelFound.emit(full_path)
            
            self.finished.emit()
            
        except Exception as e:
            print(f"Error during model discovery: {e}")
    
    def stop(self):
        """Stop the discovery thread."""
        self._is_running = False


class ModelManagementMode(BaseMode):
    """Model Management mode for loading and managing models."""
    
    # Custom signals
    modelLoaded = pyqtSignal(str)  # model path
    
    def __init__(self, parent=None):
        # Initialize attributes before calling super().__init__()
        self._discovered_models: List[str] = []
        self._model_info: Dict[str, Dict] = {}
        self._discovery_thread: Optional[ModelDiscoveryThread] = None
        
        super().__init__(parent)
    
    def _setup_ui(self):
        """Setup the UI for model management mode."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Browse button
        self.browse_btn = QPushButton("📁 Browse for Model")
        self.browse_btn.clicked.connect(self._browse_model)
        controls_layout.addWidget(self.browse_btn)
        
        # Discover button
        self.discover_btn = QPushButton("🔍 Find Models in Project")
        self.discover_btn.clicked.connect(self._discover_models)
        controls_layout.addWidget(self.discover_btn)
        
        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self._refresh_current_model)
        controls_layout.addWidget(self.refresh_btn)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Main content - single panel layout (no splitter needed)
        main_panel = QWidget()
        main_layout = QVBoxLayout(main_panel)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Model list group
        model_group = QGroupBox("Available Models")
        model_layout = QVBoxLayout(model_group)
        
        # Model table
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(4)
        self.model_table.setHorizontalHeaderLabels(["Name", "Path", "Size", "Modified"])
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.model_table.setAlternatingRowColors(False)
        self.model_table.setSortingEnabled(True)
        self.model_table.setMinimumHeight(300)
        # Make table read-only
        self.model_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        
        # Set column resize modes
        header = self.model_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)  # Path - stretch to fill
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Size
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Modified
        
        # Set minimum column widths
        self.model_table.setColumnWidth(0, 150)  # Name
        self.model_table.setColumnWidth(2, 80)   # Size
        self.model_table.setColumnWidth(3, 150)  # Modified
        
        # Connect double-click to load
        self.model_table.itemDoubleClicked.connect(self._on_model_double_clicked)
        self.model_table.itemSelectionChanged.connect(self._on_selection_changed)
        
        model_layout.addWidget(self.model_table)
        
        # Load button
        self.load_btn = QPushButton("Load Selected Model")
        self.load_btn.clicked.connect(self._load_selected_model)
        self.load_btn.setEnabled(False)
        model_layout.addWidget(self.load_btn)
        
        main_layout.addWidget(model_group)
        
        # Current model section below Load button
        current_model_layout = QHBoxLayout()
        
        # Current model info (compact)
        current_model_layout.addWidget(QLabel("Current Model:"))
        self.status_label = QLabel("No model loaded")
        # Removed inline style - rely on global styles
        self.status_label.setObjectName("modelStatusLabel")
        current_model_layout.addWidget(self.status_label)
        
        current_model_layout.addStretch()
        
        # Action buttons
        self.unload_btn = QPushButton("Unload Model")
        self.unload_btn.clicked.connect(self._unload_model)
        self.unload_btn.setEnabled(False)
        current_model_layout.addWidget(self.unload_btn)
        
        self.open_folder_btn = QPushButton("Open Model Folder")
        self.open_folder_btn.clicked.connect(self._open_model_folder)
        self.open_folder_btn.setEnabled(False)
        current_model_layout.addWidget(self.open_folder_btn)
        
        main_layout.addLayout(current_model_layout)
        
        # Model details (expandable)
        details_group = QGroupBox("Model Details")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(200)
        details_layout.addWidget(self.details_text)
        
        main_layout.addWidget(details_group)
        
        # Pre-trained models info
        pretrained_group = QGroupBox("Pre-trained Models")
        pretrained_layout = QVBoxLayout(pretrained_group)
        
        pretrained_info = QTextEdit()
        pretrained_info.setReadOnly(True)
        pretrained_info.setMaximumHeight(100)
        pretrained_info.setHtml("""
        <p><b>Ultralytics Pre-trained Models:</b></p>
        <p style="margin-left: 20px;">
        <b>yolov8n.pt</b> - Nano (fastest, least accurate)<br>
        <b>yolov8s.pt</b> - Small (fast, good accuracy)<br>
        <b>yolov8m.pt</b> - Medium (balanced)<br>
        <b>yolov8l.pt</b> - Large (slower, more accurate)<br>
        <b>yolov8x.pt</b> - Extra Large (slowest, most accurate)
        </p>
        <p><i>These will be downloaded automatically when first used.</i></p>
        """)
        
        pretrained_layout.addWidget(pretrained_info)
        main_layout.addWidget(pretrained_group)
        
        layout.addWidget(main_panel)
        
        # Add default models
        self._add_default_models()
    
    def _on_activate(self):
        """Called when mode is activated."""
        self.statusMessage.emit("Model Management mode activated", 3000)
        self._refresh_current_model()
    
    def _on_deactivate(self) -> Optional[bool]:
        """Called when mode is deactivated."""
        # Stop discovery thread if running
        if self._discovery_thread and self._discovery_thread.isRunning():
            self._discovery_thread.stop()
            self._discovery_thread.wait()
        return True
    
    def get_mode_name(self) -> str:
        """Get the display name of this mode."""
        return "Model Management"
    
    def has_unsaved_changes(self) -> bool:
        """Check if mode has unsaved changes."""
        return False  # No unsaved changes in model management
    
    def save_changes(self) -> bool:
        """Save any pending changes."""
        return True  # Nothing to save
    
    def _add_default_models(self):
        """Add pre-trained model options to the list."""
        # Add common pre-trained models
        pretrained_models = [
            ("yolov8n.pt", "Pre-trained", "~6MB", "N/A"),
            ("yolov8s.pt", "Pre-trained", "~22MB", "N/A"),
            ("yolov8m.pt", "Pre-trained", "~52MB", "N/A"),
            ("yolov8l.pt", "Pre-trained", "~87MB", "N/A"),
            ("yolov8x.pt", "Pre-trained", "~137MB", "N/A"),
        ]
        
        for name, path, size, modified in pretrained_models:
            self._add_model_to_table(name, path, size, modified)
    
    def _add_model_to_table(self, name: str, path: str, size: str, modified: str, sort_date: Optional[datetime] = None):
        """Add a model to the table."""
        row = self.model_table.rowCount()
        self.model_table.insertRow(row)
        
        # Name
        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.ItemDataRole.UserRole, path)  # Store full path
        self.model_table.setItem(row, 0, name_item)
        
        # Path - with tooltip for long paths
        path_item = QTableWidgetItem(path)
        path_item.setToolTip(path)  # Show full path on hover
        self.model_table.setItem(row, 1, path_item)
        
        # Size
        self.model_table.setItem(row, 2, QTableWidgetItem(size))
        
        # Modified - with sortable date
        modified_item = DateTableWidgetItem(modified, sort_date)
        self.model_table.setItem(row, 3, modified_item)
    
    @pyqtSlot()
    def _browse_model(self):
        """Browse for a model file."""
        # Start in project directory
        start_dir = str(Path(__file__).parent.parent.parent)
        
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File",
            start_dir,
            "Model files (*.pt)"
        )
        
        if path:
            self._load_model(path)
    
    @pyqtSlot()
    def _discover_models(self):
        """Discover models in project directory."""
        if self._discovery_thread and self._discovery_thread.isRunning():
            return
        
        # Clear discovered models from table (keep pre-trained)
        rows_to_remove = []
        for row in range(self.model_table.rowCount()):
            path_item = self.model_table.item(row, 1)
            if path_item and path_item.text() not in ["Pre-trained"]:
                rows_to_remove.append(row)
        
        # Remove in reverse order
        for row in reversed(rows_to_remove):
            self.model_table.removeRow(row)
        
        # Start discovery
        project_path = Path(__file__).parent.parent.parent
        self.statusMessage.emit(f"Searching for models in {project_path}...", 3000)
        
        self._discovery_thread = ModelDiscoveryThread(project_path)
        self._discovery_thread.modelFound.connect(self._on_model_discovered)
        self._discovery_thread.finished.connect(self._on_discovery_finished)
        
        self.discover_btn.setEnabled(False)
        self._discovery_thread.start()
    
    @pyqtSlot(str)
    def _on_model_discovered(self, path: str):
        """Handle discovered model."""
        path_obj = Path(path)
        
        # Get file info
        try:
            stat = path_obj.stat()
            size_mb = stat.st_size / (1024 * 1024)
            size_str = f"{size_mb:.1f}MB"
            
            modified = datetime.fromtimestamp(stat.st_mtime)
            modified_str = modified.strftime("%Y-%m-%d %H:%M")
            
            # Add to table with sortable date
            self._add_model_to_table(
                path_obj.name,
                str(path_obj),
                size_str,
                modified_str,
                modified  # Pass datetime for sorting
            )
            
        except Exception as e:
            print(f"Error getting info for {path}: {e}")
    
    @pyqtSlot()
    def _on_discovery_finished(self):
        """Handle discovery completion."""
        self.discover_btn.setEnabled(True)
        row_count = self.model_table.rowCount()
        
        # Sort by date column (newest first)
        self.model_table.sortItems(3, Qt.SortOrder.DescendingOrder)
        
        self.statusMessage.emit(f"Found {row_count} models", 3000)
    
    @pyqtSlot()
    def _on_selection_changed(self):
        """Handle selection change in model table."""
        selected = self.model_table.selectedItems()
        self.load_btn.setEnabled(len(selected) > 0)
    
    @pyqtSlot()
    def _on_model_double_clicked(self):
        """Handle double-click on model."""
        self._load_selected_model()
    
    @pyqtSlot()
    def _load_selected_model(self):
        """Load the selected model."""
        current_row = self.model_table.currentRow()
        if current_row < 0:
            return
        
        name_item = self.model_table.item(current_row, 0)
        path_item = self.model_table.item(current_row, 1)
        
        if not name_item:
            return
        
        # Get the stored path from UserRole
        model_path = name_item.data(Qt.ItemDataRole.UserRole)
        
        # Handle pre-trained models
        if path_item and path_item.text() == "Pre-trained":
            model_name = name_item.text()
            self._load_model(model_name)
        else:
            self._load_model(model_path)
    
    def _load_model(self, path: str):
        """Load a model."""
        try:
            # Show loading message
            self.statusMessage.emit(f"Loading model: {path}", 0)
            
            # Load through ModelCache
            model_cache = ModelCache()
            if model_cache.load_model(path):
                self.modelLoaded.emit(path)
                self.statusMessage.emit(f"Model loaded successfully: {Path(path).name}", 3000)
                self._refresh_current_model()
            else:
                QMessageBox.warning(
                    self, "Load Failed",
                    f"Failed to load model: {path}\n\n"
                    "Please ensure the file is a valid YOLO model."
                )
                
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Error loading model:\n{str(e)}"
            )
    
    @pyqtSlot()
    def _unload_model(self):
        """Unload the current model."""
        model_cache = ModelCache()
        model_cache.clear_model()
        self.statusMessage.emit("Model unloaded", 3000)
        self._refresh_current_model()
    
    @pyqtSlot()
    def _open_model_folder(self):
        """Open the folder containing the current model."""
        model_cache = ModelCache()
        model_info = model_cache.get_model_info()
        
        if model_info and 'path' in model_info:
            model_path = Path(model_info['path'])
            folder = model_path.parent
            
            # Open folder in system file explorer
            import subprocess
            import platform
            
            try:
                if platform.system() == 'Windows':
                    subprocess.run(['explorer', str(folder)])
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', str(folder)])
                else:  # Linux
                    subprocess.run(['xdg-open', str(folder)])
            except Exception as e:
                QMessageBox.warning(
                    self, "Error",
                    f"Could not open folder:\n{str(e)}"
                )
    
    @pyqtSlot()
    def _refresh_current_model(self):
        """Refresh the current model display."""
        model_cache = ModelCache()
        model = model_cache.get_model()
        model_info = model_cache.get_model_info()
        
        if model and model_info:
            # Update status
            self.status_label.setText(f"Loaded: {model_info['name']}")
            self.status_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
            
            # Build compact details HTML
            details_html = f"""
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                td {{ padding: 4px 8px; }}
                td:first-child {{ font-weight: bold; width: 120px; }}
            </style>
            <table>
            <tr><td>Path:</td><td style="word-wrap: break-word;">{model_info['path']}</td></tr>
            <tr><td>Type:</td><td>{model_info.get('type', 'Unknown')}</td></tr>
            <tr><td>Task:</td><td>{model_info.get('task', 'detect')}</td></tr>
            """
            
            # Add model-specific info if available
            if hasattr(model, 'model') and hasattr(model.model, 'names'):
                classes = model.model.names
                class_list = ', '.join(f'{i}: {name}' for i, name in classes.items())
                if len(class_list) > 100:
                    class_list = class_list[:100] + '...'
                details_html += f"""
                <tr><td>Classes:</td><td>{len(classes)}</td></tr>
                <tr><td>Class Names:</td><td>{class_list}</td></tr>
                """
            
            # Add model architecture info if available
            if hasattr(model, 'model'):
                try:
                    total_params = sum(p.numel() for p in model.model.parameters())
                    details_html += f"""
                    <tr><td>Parameters:</td><td>{total_params:,}</td></tr>
                    """
                except:
                    pass
            
            details_html += "</table>"
            
            self.details_text.setHtml(details_html)
            
            # Enable buttons
            self.unload_btn.setEnabled(True)
            self.open_folder_btn.setEnabled(True)
            
        else:
            # No model loaded
            self.status_label.setText("No model loaded")
            self.status_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            
            self.details_text.setHtml("""
            <p>No model is currently loaded.</p>
            <p style="color: #888;">To load a model:</p>
            <ul style="color: #888;">
            <li>Click "Browse for Model" to select a .pt file</li>
            <li>Click "Find Models in Project" to discover local models</li>
            <li>Double-click a pre-trained model to download it</li>
            </ul>
            """)
            
            # Disable buttons
            self.unload_btn.setEnabled(False)
            self.open_folder_btn.setEnabled(False)