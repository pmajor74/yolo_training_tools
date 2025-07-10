"""Device status widget for displaying GPU/CPU usage in status bar."""

from typing import Optional
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
import torch


class DeviceStatusWidget(QWidget):
    """Widget that displays current compute device (GPU/CPU) status."""
    
    device_changed = pyqtSignal(str)  # Emitted when device changes
    
    def __init__(self, parent=None):
        """Initialize the device status widget."""
        super().__init__(parent)
        self._current_device = None
        self._setup_ui()
        self._update_device_status()
        
        # Set up timer for periodic updates
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_device_status)
        self._update_timer.start(5000)  # Update every 5 seconds
        
    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 0, 5, 0)
        
        # Device label
        self._device_label = QLabel("Device:")
        layout.addWidget(self._device_label)
        
        # Status label
        self._status_label = QLabel("Detecting...")
        self._status_label.setMinimumWidth(100)
        font = QFont()
        font.setBold(True)
        self._status_label.setFont(font)
        layout.addWidget(self._status_label)
        
        self.setLayout(layout)
        
    def _get_device_info(self) -> tuple[str, str]:
        """Get current device information.
        
        Returns:
            Tuple of (device_type, device_name)
        """
        try:
            if torch.cuda.is_available():
                device_type = "cuda"
                device_name = torch.cuda.get_device_name(0)
                # Shorten common GPU names
                if "NVIDIA" in device_name:
                    device_name = device_name.replace("NVIDIA ", "")
                if "GeForce" in device_name:
                    device_name = device_name.replace("GeForce ", "")
                return device_type, f"GPU: {device_name}"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps", "GPU: Apple Silicon"
            else:
                return "cpu", "CPU"
        except Exception:
            return "cpu", "CPU"
            
    def _update_device_status(self):
        """Update the device status display."""
        device_type, device_name = self._get_device_info()
        
        # Update status label text
        self._status_label.setText(device_name)
        
        # Update status label style based on device type
        if device_type in ["cuda", "mps"]:
            # Green for GPU
            self._status_label.setStyleSheet("""
                QLabel {
                    color: #28a745;
                    background-color: rgba(40, 167, 69, 0.1);
                    padding: 2px 8px;
                    border: 1px solid #28a745;
                    border-radius: 3px;
                }
            """)
        else:
            # Red for CPU
            self._status_label.setStyleSheet("""
                QLabel {
                    color: #dc3545;
                    background-color: rgba(220, 53, 69, 0.1);
                    padding: 2px 8px;
                    border: 1px solid #dc3545;
                    border-radius: 3px;
                }
            """)
            
        # Emit signal if device changed
        if self._current_device != device_type:
            self._current_device = device_type
            self.device_changed.emit(device_type)
            
    def refresh(self):
        """Manually refresh the device status."""
        self._update_device_status()
        
    def get_current_device(self) -> str:
        """Get the current device type.
        
        Returns:
            Device type string ('cuda', 'mps', or 'cpu')
        """
        return self._current_device or "cpu"