"""Options dialog for application settings."""

from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QGroupBox, QFormLayout, QMessageBox,
    QCheckBox, QFrame, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
import os

from ..core import SettingsManager


class OptionsDialog(QDialog):
    """Dialog for configuring application options."""
    
    # Signal emitted when options are saved
    optionsSaved = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.setModal(True)
        self.resize(450, 300)
        
        # Get settings manager
        self.settings = SettingsManager()
        
        # Setup UI
        self._setup_ui()
        
        # Load current settings
        self._load_settings()
    
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Processing options section
        processing_section = self._create_section("Processing Options")
        processing_layout = QFormLayout()
        processing_layout.setSpacing(10)
        processing_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        processing_layout.setContentsMargins(5, 5, 5, 5)
        
        # Workers setting
        self.workers_spin = QSpinBox()
        self.workers_spin.setMinimum(0)
        self.workers_spin.setMaximum(32)
        self.workers_spin.setSpecialValueText("Auto (0)")
        self.workers_spin.setToolTip(
            "Number of worker processes for data loading.\n"
            "0 = Auto (recommended for Windows)\n"
            "Higher values may improve performance on Linux/Mac\n"
            "but can cause issues on Windows."
        )
        self.workers_spin.setStyleSheet("""
            QSpinBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
                color: #333333;
                min-width: 80px;
            }
            QSpinBox:hover {
                border: 1px solid #999999;
            }
            QSpinBox:focus {
                border: 1px solid #0078d4;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #f0f0f0;
                border: 1px solid #cccccc;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #e0e0e0;
            }
        """)
        
        # Create wrapper widget for workers controls
        workers_widget = QWidget()
        workers_widget.setStyleSheet("background-color: transparent;")
        workers_layout = QHBoxLayout(workers_widget)
        workers_layout.setContentsMargins(0, 0, 0, 0)
        workers_layout.setSpacing(10)
        workers_layout.addWidget(self.workers_spin)
        
        # Add platform-specific recommendation
        if os.name == 'nt':  # Windows
            rec_label = QLabel("(Windows detected: 0 recommended)")
        else:
            rec_label = QLabel("(Linux/Mac: 4-8 recommended)")
        rec_label.setStyleSheet("color: #666666; font-style: italic; background-color: transparent;")
        workers_layout.addWidget(rec_label)
        workers_layout.addStretch()
        
        workers_label = QLabel("Data Loading Workers:")
        workers_label.setStyleSheet("font-weight: 500; color: #333333; background-color: transparent;")
        processing_layout.addRow(workers_label, workers_widget)
        
        # Add note about workers
        note_widget = QFrame()
        note_widget.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border: 1px solid #dddddd;
                border-radius: 4px;
                padding: 10px;
                margin-top: 10px;
            }
        """)
        note_layout = QVBoxLayout(note_widget)
        note_layout.setContentsMargins(5, 5, 5, 5)
        
        note_label = QLabel(
            "<b>Note:</b> This setting affects both training and inference operations. "
            "If you experience crashes or freezing, try setting this to 0."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #444444;")
        note_layout.addWidget(note_label)
        
        processing_layout.addRow(note_widget)
        
        # Add the processing layout to the section's content area
        processing_section.content_widget.setLayout(processing_layout)
        layout.addWidget(processing_section)
        
        # Additional options section
        additional_section = self._create_section("Additional Options")
        additional_layout = QFormLayout()
        additional_layout.setSpacing(10)
        
        # Placeholder checkbox
        self.gpu_memory_checkbox = QCheckBox("Limit GPU memory usage")
        self.gpu_memory_checkbox.setEnabled(False)
        self.gpu_memory_checkbox.setToolTip("Coming soon: Option to limit GPU memory usage")
        self.gpu_memory_checkbox.setStyleSheet("""
            QCheckBox {
                color: #999999;
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #f5f5f5;
            }
        """)
        additional_layout.addRow(self.gpu_memory_checkbox)
        
        # Add the additional layout to the section's content area
        additional_section.content_widget.setLayout(additional_layout)
        layout.addWidget(additional_section)
        
        # Add stretch to push buttons to bottom
        layout.addStretch()
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.Apply
        )
        button_box.accepted.connect(self._on_ok)
        button_box.rejected.connect(self.reject)
        
        # Connect Apply button
        apply_button = button_box.button(QDialogButtonBox.StandardButton.Apply)
        apply_button.clicked.connect(self._apply_settings)
        
        layout.addWidget(button_box)
    
    def _create_section(self, title: str) -> QWidget:
        """Create a styled section frame with title."""
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Section frame
        section = QFrame()
        section.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #dddddd;
                border-radius: 6px;
            }
        """)
        
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(15, 15, 15, 15)
        section_layout.setSpacing(10)
        
        # Section title
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333333;
                padding: 0px 0px 10px 0px;
                border-bottom: 1px solid #eeeeee;
                margin-bottom: 10px;
            }
        """)
        section_layout.addWidget(title_label)
        
        # Content widget
        content_widget = QWidget()
        content_widget.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
        """)
        section_layout.addWidget(content_widget)
        
        container_layout.addWidget(section)
        
        # Store reference to content widget for later use
        container.content_widget = content_widget
        
        return container
    
    def _load_settings(self):
        """Load current settings into the dialog."""
        # Load workers setting (default to 4 for all platforms)
        workers = self.settings.get('data_loading_workers', 4)
        self.workers_spin.setValue(workers)
    
    def _apply_settings(self):
        """Apply the current settings without closing the dialog."""
        # Save workers setting
        self.settings.set('data_loading_workers', self.workers_spin.value())
        
        # Emit signal that options were saved
        self.optionsSaved.emit()
        
        # Show confirmation
        QMessageBox.information(
            self, "Settings Applied",
            "Settings have been applied successfully.\n"
            "Note: Some settings may require restarting the current operation to take effect."
        )
    
    def _on_ok(self):
        """Handle OK button - apply settings and close."""
        self._apply_settings()
        self.accept()