"""Main application window for YOLO Dataset Viewer."""

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QMenuBar, QMenu,
    QStatusBar, QMessageBox, QFileDialog, QToolBar, QLabel,
    QPushButton, QWidget, QVBoxLayout, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence

from .core import ModelCache, SettingsManager, ImageCache, DatasetManager
from .core.constants import (
    APP_NAME, APP_VERSION, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT, AppMode, MODE_NAMES
)
from .modes.dataset_editor import DatasetEditorMode
from .modes.folder_browser_mode import FolderBrowserMode
from .modes.model_management import ModelManagementMode
from .modes.training_mode import TrainingMode
from .modes.auto_annotation_mode import AutoAnnotationMode
from .modes.dataset_split_mode import DatasetSplitMode
from .widgets.device_status_widget import DeviceStatusWidget


class MainApplication(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize singletons
        self.model_cache = ModelCache()
        self.settings = SettingsManager()
        self.image_cache = ImageCache()
        self.dataset_manager = DatasetManager()
        
        # Mode widgets (will be populated as we implement them)
        self.mode_widgets = {}
        
        # Setup UI
        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_statusbar()
        
        # Initialize modes
        self._initialize_modes()
        
        # Hide all tabs immediately after creation to prevent early painting
        for i in range(self.tab_widget.count()):
            self.tab_widget.widget(i).setVisible(False)
        
        # Connect signals
        self._connect_signals()
        
        # Restore geometry
        self._restore_geometry()
        
        # Set initial mode after everything is set up
        self._current_tab_index = 0
        self._stylesheet_loaded = False
    
    def _setup_ui(self):
        """Setup main UI."""
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        
        # Central widget with tabs
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # Style the tab widget
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
    
    def _setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        self.save_action = QAction("&Save", self)
        self.save_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self._on_save)
        file_menu.addAction(self.save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self._on_undo)
        edit_menu.addAction(self.undo_action)
        
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self._on_redo)
        edit_menu.addAction(self.redo_action)
        
        edit_menu.addSeparator()
        
        self.settings_action = QAction("&Settings...", self)
        self.settings_action.setShortcut("Ctrl+,")
        self.settings_action.triggered.connect(self._on_settings)
        edit_menu.addAction(self.settings_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        self.theme_action = QAction("Toggle &Dark Mode", self)
        self.theme_action.setShortcut("Ctrl+D")
        self.theme_action.triggered.connect(self._on_toggle_theme)
        view_menu.addAction(self.theme_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Model status
        self.model_status_label = QLabel("No model loaded")
        toolbar.addWidget(self.model_status_label)
    
    def _setup_statusbar(self):
        """Setup status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Add device status widget to the right side of status bar
        self.device_status_widget = DeviceStatusWidget()
        self.statusbar.addPermanentWidget(self.device_status_widget)
    
    def _initialize_modes(self):
        """Initialize application modes."""
        # Create Model Management mode FIRST
        model_mode = ModelManagementMode()
        self.tab_widget.addTab(model_mode, MODE_NAMES[AppMode.MODEL_MANAGEMENT])
        self.mode_widgets[AppMode.MODEL_MANAGEMENT] = model_mode
        
        # Connect Model Management signals
        model_mode.statusMessage.connect(
            lambda msg, timeout: self.statusbar.showMessage(msg, timeout)
        )
        model_mode.modelLoaded.connect(
            lambda path: self._update_model_status()
        )
        
        # Create Dataset Editor mode
        dataset_editor = DatasetEditorMode()
        self.tab_widget.addTab(dataset_editor, MODE_NAMES[AppMode.DATASET_EDITOR])
        self.mode_widgets[AppMode.DATASET_EDITOR] = dataset_editor
        
        # Connect Dataset Editor signals
        dataset_editor.statusMessage.connect(
            lambda msg, timeout: self.statusbar.showMessage(msg, timeout)
        )
        dataset_editor.datasetLoaded.connect(self._on_dataset_loaded)
        
        # Create Folder Browser mode
        folder_browser = FolderBrowserMode()
        self.tab_widget.addTab(folder_browser, MODE_NAMES[AppMode.FOLDER_BROWSER])
        self.mode_widgets[AppMode.FOLDER_BROWSER] = folder_browser
        
        # Connect Folder Browser signals
        folder_browser.statusMessage.connect(
            lambda msg, timeout: self.statusbar.showMessage(msg, timeout)
        )
        folder_browser.folderLoaded.connect(
            lambda path: self.statusbar.showMessage(f"Loaded folder: {Path(path).name}", 3000)
        )
        folder_browser.annotationsSaved.connect(
            lambda count: self.statusbar.showMessage(f"Saved {count} annotation files", 3000)
        )
        
        # Create Training mode
        training_mode = TrainingMode()
        self.tab_widget.addTab(training_mode, MODE_NAMES[AppMode.TRAINING])
        self.mode_widgets[AppMode.TRAINING] = training_mode
        
        # Connect Training mode signals
        training_mode.statusMessage.connect(
            lambda msg, timeout: self.statusbar.showMessage(msg, timeout)
        )
        training_mode.trainingStarted.connect(
            lambda config: (
                self.statusbar.showMessage("Training started", 3000),
                self.device_status_widget.refresh()
            )
        )
        training_mode.trainingCompleted.connect(
            lambda model_path: self.statusbar.showMessage(f"Training completed: {Path(model_path).name}", 5000)
        )
        training_mode.trainingFailed.connect(
            lambda error: self.statusbar.showMessage(f"Training failed: {error}", 5000)
        )
        
        # Create Auto-Annotation mode
        auto_annotation_mode = AutoAnnotationMode()
        self.tab_widget.addTab(auto_annotation_mode, MODE_NAMES[AppMode.AUTO_ANNOTATION])
        self.mode_widgets[AppMode.AUTO_ANNOTATION] = auto_annotation_mode
        
        # Connect Auto-Annotation mode signals
        auto_annotation_mode.sessionStarted.connect(
            lambda folder: self.statusbar.showMessage(f"Auto-annotation started: {Path(folder).name}", 3000)
        )
        auto_annotation_mode.sessionCompleted.connect(
            lambda: self.statusbar.showMessage("Auto-annotation complete", 3000)
        )
        auto_annotation_mode.annotationsExported.connect(
            lambda path: self.statusbar.showMessage(f"Annotations exported to: {Path(path).name}", 5000)
        )
        
        # Create Dataset Split mode
        dataset_split_mode = DatasetSplitMode()
        self.tab_widget.addTab(dataset_split_mode, MODE_NAMES[AppMode.DATASET_SPLIT])
        self.mode_widgets[AppMode.DATASET_SPLIT] = dataset_split_mode
        
        # Connect Dataset Split mode signals
        dataset_split_mode.statusMessage.connect(
            lambda msg, timeout: self.statusbar.showMessage(msg, timeout)
        )
        dataset_split_mode.splitStarted.connect(
            lambda path: self.statusbar.showMessage(f"Dataset split started: {Path(path).name}", 3000)
        )
        dataset_split_mode.splitCompleted.connect(
            lambda path: self.statusbar.showMessage(f"Dataset split completed: {Path(path).name}", 5000)
        )
        
        # Create placeholder tab for remaining modes (if any)
        for mode in []:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(50, 50, 50, 50)
            
            # Create a styled group box
            group = QGroupBox(f"{MODE_NAMES[mode]} - Coming Soon")
            group.setStyleSheet("""
                QGroupBox {
                    font-size: 24px;
                    font-weight: bold;
                    color: #0d7377;
                    border: 2px solid #0d7377;
                    border-radius: 10px;
                    margin-top: 20px;
                    padding-top: 20px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 20px;
                    padding: 0 10px 0 10px;
                }
            """)
            
            group_layout = QVBoxLayout(group)
            group_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Add icon and description
            icon_label = QLabel("✂️")
            desc_label = QLabel("Split your dataset into train, validation, and test sets.\nConfigure split ratios and stratification options.")
            
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setStyleSheet("font-size: 72px;")
            
            desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc_label.setStyleSheet("font-size: 16px; color: #666; line-height: 1.5;")
            desc_label.setWordWrap(True)
            
            group_layout.addStretch()
            group_layout.addWidget(icon_label)
            group_layout.addSpacing(20)
            group_layout.addWidget(desc_label)
            group_layout.addStretch()
            
            layout.addWidget(group)
            
            # Add to tab widget
            self.tab_widget.addTab(widget, MODE_NAMES[mode])
            self.mode_widgets[mode] = widget
    
    def _connect_signals(self):
        """Connect signals."""
        # Model cache signals
        self.model_cache.modelLoaded.connect(self._on_model_loaded)
        self.model_cache.modelCleared.connect(self._on_model_cleared)
        
        # Image cache signals
        # (Cache update signals removed - no longer displaying cache info)
        
        # Dataset manager signals
        self.dataset_manager.datasetLoaded.connect(self._on_dataset_loaded_global)
        
        # Tab changing
        self._current_tab_index = 0
        self._switching_programmatically = False
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
    
    def _restore_geometry(self):
        """Restore window geometry from settings."""
        geometry = self.settings.get_window_geometry()
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    
    def _load_stylesheet(self):
        """Load application stylesheet."""
        try:
            resources_dir = Path(__file__).parent / "resources"
            stylesheet_path = resources_dir / "styles.qss"
            if stylesheet_path.exists():
                with open(stylesheet_path, 'r') as f:
                    stylesheet = f.read()
                    # Apply stylesheet to the application, not just this window
                    QApplication.instance().setStyleSheet(stylesheet)
        except Exception as e:
            print(f"Failed to load stylesheet: {e}")
    
    def showEvent(self, event):
        """Handle show event - called when window is first shown."""
        super().showEvent(event)
        
        # Load stylesheet only once, after window is shown
        if not self._stylesheet_loaded:
            self._stylesheet_loaded = True
            # Use QTimer to ensure window is fully rendered
            QTimer.singleShot(1, self._on_window_ready)
    
    def closeEvent(self, event):
        """Handle close event."""
        # Save geometry
        self.settings.set_window_geometry(self.saveGeometry())
        
        # Check for unsaved changes (placeholder)
        # TODO: Implement unsaved changes check
        
        event.accept()
    
    # Slots
    
    @pyqtSlot()
    def _on_save(self):
        """Save current work."""
        # Get current mode widget
        current_index = self.tab_widget.currentIndex()
        current_mode = list(self.mode_widgets.keys())[current_index]
        current_widget = self.mode_widgets[current_mode]
        
        # Call save if available
        if hasattr(current_widget, 'save_changes'):
            if current_widget.save_changes():
                self.statusbar.showMessage("Changes saved", 3000)
            else:
                self.statusbar.showMessage("Failed to save changes", 3000)
        else:
            self.statusbar.showMessage("Save not available in this mode", 3000)
    
    @pyqtSlot()
    def _on_undo(self):
        """Undo last action."""
        # TODO: Implement undo
        self.statusbar.showMessage("Undo", 1000)
    
    @pyqtSlot()
    def _on_redo(self):
        """Redo last action."""
        # TODO: Implement redo
        self.statusbar.showMessage("Redo", 1000)
    
    @pyqtSlot()
    def _on_settings(self):
        """Open settings dialog."""
        # TODO: Implement settings dialog
        QMessageBox.information(self, "Settings", "Settings dialog coming soon!")
    
    @pyqtSlot()
    def _on_toggle_theme(self):
        """Toggle dark mode."""
        # TODO: Implement theme switching
        current_theme = self.settings.get('theme', 'light')
        new_theme = 'dark' if current_theme == 'light' else 'light'
        self.settings.set('theme', new_theme)
        self.statusbar.showMessage(f"Switched to {new_theme} mode", 2000)
    
    @pyqtSlot()
    def _on_about(self):
        """Show about dialog."""
        # Get PyTorch information
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_info = f"GPU (CUDA {torch.version.cuda})"
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
                device_info += f"\n   Device: {gpu_name}"
            else:
                device_info = "CPU"
        except ImportError:
            torch_version = "Not installed"
            device_info = "Unknown"
        
        QMessageBox.about(
            self, "About",
            f"{APP_NAME} v{APP_VERSION}\n\n"
            "A comprehensive YOLO dataset management tool.\n\n"
            "Built with PyQt6 and Ultralytics.\n\n"
            f"PyTorch: {torch_version}\n"
            f"Compute: {device_info}"
        )
    
    
    @pyqtSlot(str)
    def _on_model_loaded(self, path):
        """Handle model loaded signal."""
        self._update_model_status()
    
    @pyqtSlot()
    def _on_model_cleared(self):
        """Handle model cleared signal."""
        self._update_model_status()
    
    @pyqtSlot(int)
    def _on_tab_changed(self, index):
        """Handle tab change."""
        # Safety check
        if not self.mode_widgets or index < 0:
            return
            
        # Get list of modes in tab order
        mode_list = list(self.mode_widgets.keys())
        if index >= len(mode_list):
            return
        
        # If we're already on this tab, nothing to do
        if index == self._current_tab_index:
            return
        
        # Check if we can leave the current tab
        if self._current_tab_index < len(mode_list):
            current_mode = mode_list[self._current_tab_index]
            current_widget = self.mode_widgets[current_mode]
            
            # Check if current mode can be deactivated
            if hasattr(current_widget, 'deactivate'):
                can_deactivate = current_widget.deactivate()
                if not can_deactivate:
                    # Restore previous tab
                    self.tab_widget.blockSignals(True)
                    self.tab_widget.setCurrentIndex(self._current_tab_index)
                    self.tab_widget.blockSignals(False)
                    return
        
        # Update current index
        self._current_tab_index = index
        
        # Activate new mode
        current_mode = mode_list[index]
        current_widget = self.mode_widgets[current_mode]
        if hasattr(current_widget, 'activate'):
            current_widget.activate()
        
        mode_name = self.tab_widget.tabText(index)
        self.statusbar.showMessage(f"Switched to {mode_name}", 2000)
    
    @pyqtSlot(Path)
    def _on_dataset_loaded(self, yaml_path):
        """Handle dataset loaded signal from Dataset Editor."""
        self.statusbar.showMessage(f"Dataset loaded: {yaml_path.name}", 3000)
    
    @pyqtSlot(Path)
    def _on_dataset_loaded_global(self, yaml_path):
        """Handle dataset loaded signal from DatasetManager."""
        # This is called whenever a dataset is loaded from anywhere
        # Each mode will handle its own update via the DatasetManager signals
    
    def _update_model_status(self):
        """Update model status in toolbar."""
        model_cache = ModelCache()
        model_info = model_cache.get_model_info()
        
        if model_info:
            self.model_status_label.setText(f"Model: {model_info['name']}")
            self.model_status_label.setStyleSheet("color: green;")
        else:
            self.model_status_label.setText("No model loaded")
            self.model_status_label.setStyleSheet("")
            
        # Refresh device status when model is loaded/unloaded
        self.device_status_widget.refresh()
    
    def _on_window_ready(self):
        """Called after window is shown and ready."""
        # Load stylesheet now that window has a valid paint device
        self._load_stylesheet()
        
        # Then activate the initial tab
        self._activate_initial_tab()
    
    def _activate_initial_tab(self):
        """Activate the initial tab after window is shown."""
        # Make first tab visible
        if self.tab_widget.count() > 0:
            self.tab_widget.widget(0).setVisible(True)
            self.tab_widget.setCurrentIndex(0)
            # Activate the mode
            current_widget = self.mode_widgets.get(AppMode.MODEL_MANAGEMENT)
            if current_widget and hasattr(current_widget, 'activate'):
                current_widget.activate()


def main():
    """Main entry point."""
    import os
    from PyQt6.QtCore import qInstallMessageHandler, QtMsgType
    
    # Note: We're not setting AA_ShareOpenGLContexts as it causes issues
    # and is typically only needed for specific OpenGL use cases
    
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("YOLOTools")
    
    # Install custom message handler to suppress QPainter warnings
    def qt_message_handler(msg_type, context, msg):
        # Suppress specific QPainter warnings that are harmless
        if "QPainter" in msg and any(x in msg for x in [
            "Paint device returned engine == 0",
            "Painter not active",
            "aborted"
        ]):
            return  # Suppress these warnings
        
        # Let other messages through
        if msg_type == QtMsgType.QtWarningMsg:
            print(f"Qt Warning: {msg}")
        elif msg_type == QtMsgType.QtCriticalMsg:
            print(f"Qt Critical: {msg}")
        elif msg_type == QtMsgType.QtFatalMsg:
            print(f"Qt Fatal: {msg}")
            sys.exit(1)
    
    qInstallMessageHandler(qt_message_handler)
    
    # Set Windows style to ensure proper arrow rendering on Windows
    QApplication.setStyle("Windows")
    
    # Create main window but don't show it yet
    window = MainApplication()
    
    # Ensure the window has a valid size before showing
    if window.size().width() == 0 or window.size().height() == 0:
        window.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    
    # Process events to ensure all widgets are initialized
    app.processEvents()
    
    # Show window after a brief delay to ensure all painting is ready
    def show_window():
        window.show()
        # Force update to ensure proper painting
        window.update()
    
    QTimer.singleShot(50, show_window)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()