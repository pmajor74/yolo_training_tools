"""Options dialog for application settings."""

from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QGroupBox, QFormLayout, QMessageBox,
    QCheckBox
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
        
        # Processing group
        processing_group = QGroupBox("Processing Options")
        processing_layout = QFormLayout()
        
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
        
        workers_layout = QHBoxLayout()
        workers_layout.addWidget(self.workers_spin)
        
        # Add platform-specific recommendation
        if os.name == 'nt':  # Windows
            rec_label = QLabel("(Windows detected: 0 recommended)")
            rec_label.setStyleSheet("color: #666666; font-style: italic;")
            workers_layout.addWidget(rec_label)
        else:
            rec_label = QLabel("(Linux/Mac: 4-8 recommended)")
            rec_label.setStyleSheet("color: #666666; font-style: italic;")
            workers_layout.addWidget(rec_label)
        
        workers_layout.addStretch()
        
        processing_layout.addRow("Data Loading Workers:", workers_layout)
        
        # Add note about workers
        note_label = QLabel(
            "<i>Note: This setting affects both training and inference operations. "
            "If you experience crashes or freezing, try setting this to 0.</i>"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #666666; margin-top: 10px;")
        processing_layout.addRow(note_label)
        
        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)
        
        # Future options group (placeholder for additional settings)
        future_group = QGroupBox("Additional Options")
        future_layout = QFormLayout()
        
        # Placeholder checkbox
        self.gpu_memory_checkbox = QCheckBox("Limit GPU memory usage")
        self.gpu_memory_checkbox.setEnabled(False)
        self.gpu_memory_checkbox.setToolTip("Coming soon: Option to limit GPU memory usage")
        future_layout.addRow(self.gpu_memory_checkbox)
        
        future_group.setLayout(future_layout)
        layout.addWidget(future_group)
        
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
    
    def _load_settings(self):
        """Load current settings into the dialog."""
        # Load workers setting (default to 0 for Windows, 8 for others)
        default_workers = 0 if os.name == 'nt' else 8
        workers = self.settings.get('data_loading_workers', default_workers)
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