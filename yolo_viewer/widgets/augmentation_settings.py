"""Augmentation settings widget for training configuration."""

from typing import Dict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QToolTip
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont, QEnterEvent, QCursor


class InfoLabel(QLabel):
    """Custom label for info icons with enhanced tooltip support."""
    
    def __init__(self, text: str, tooltip: str, parent=None):
        super().__init__(text, parent)
        self._tooltip_text = tooltip
        self.setToolTip(tooltip)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.setMouseTracking(True)
        
    def enterEvent(self, event: QEnterEvent):
        """Show tooltip on hover."""
        # Convert newlines to HTML breaks for better formatting
        formatted_tooltip = self._tooltip_text.replace('\n', '<br/>')
        QToolTip.showText(QCursor.pos(), formatted_tooltip, self)
        super().enterEvent(event)


class AugmentationSettings(QWidget):
    """Augmentation settings widget with scrollable parameters."""
    
    settingsChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._apply_dark_theme()
    
    def _apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                color: #cccccc;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QToolTip {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 3px;
                font-size: 12px;
            }
            QDoubleSpinBox {
                background-color: #4a4a4a;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
                color: #cccccc;
                min-height: 20px;
            }
            QDoubleSpinBox:hover {
                border-color: #0d7377;
            }
        """)
    
    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Augmentation parameters with tooltips
        augment_params = [
            ("Degrees", "degrees", 0, 180, 45.0, "Image rotation range (+/- degrees)\nRotates images randomly within this range"),
            ("Translate", "translate", 0.0, 0.9, 0.1, "Image translation (+/- fraction)\nShifts images horizontally and vertically by fraction of image size"),
            ("Scale", "scale", 0.0, 0.9, 0.5, "Image scale (+/- gain)\nScales images up or down by this factor"),
            ("Shear", "shear", 0.0, 10.0, 0.0, "Image shear (+/- degrees)\nApplies shear transformation to images"),
            ("Perspective", "perspective", 0.0, 0.001, 0.0, "Image perspective (+/- fraction)\nApplies perspective transformation"),
            ("Flip LR", "fliplr", 0.0, 1.0, 0.5, "Horizontal flip probability\n0.5 = 50% chance to flip horizontally"),
            ("Flip UD", "flipud", 0.0, 1.0, 0.5, "Vertical flip probability\n0.0 = no vertical flipping"),
            ("Mosaic", "mosaic", 0.0, 1.0, 0.0, "Mosaic augmentation probability\nCombines 4 images into 1"),
            ("Mixup", "mixup", 0.0, 1.0, 0.0, "Mixup augmentation probability\nBlends two images together"),
            ("HSV H", "hsv_h", 0.0, 1.0, 0.015, "HSV-Hue augmentation (fraction)\nAdjusts image hue randomly"),
            ("HSV S", "hsv_s", 0.0, 1.0, 0.7, "HSV-Saturation augmentation (fraction)\nAdjusts image saturation randomly"),
            ("HSV V", "hsv_v", 0.0, 1.0, 0.4, "HSV-Value augmentation (fraction)\nAdjusts image brightness randomly")
        ]
        
        self.spinboxes = {}
        for label, key, min_val, max_val, default, tooltip in augment_params:
            param_layout = QHBoxLayout()
            param_layout.setSpacing(10)
            
            # Parameter label
            param_label = QLabel(f"{label}:")
            param_label.setFixedWidth(100)
            param_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            param_label.setToolTip(tooltip)
            param_layout.addWidget(param_label)
            
            # Spinbox
            spinbox = QDoubleSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setSingleStep(0.01 if max_val <= 1.0 else 0.1)
            spinbox.setValue(default)
            spinbox.setDecimals(3 if max_val <= 0.001 else 2)
            spinbox.setToolTip(tooltip)
            spinbox.setFixedWidth(100)
            spinbox.valueChanged.connect(lambda: self.settingsChanged.emit())
            self.spinboxes[key] = spinbox
            param_layout.addWidget(spinbox)
            
            # Add info icon with tooltip
            info_label = InfoLabel("ⓘ", tooltip)
            info_label.setObjectName("info_icon")
            info_label.setStyleSheet("""
                QLabel#info_icon { 
                    color: #0d7377; 
                    font-size: 16px;
                    font-weight: bold;
                    padding: 2px;
                    background-color: transparent;
                }
                QLabel#info_icon:hover {
                    color: #14ffec;
                }
            """)
            param_layout.addWidget(info_label)
            
            param_layout.addStretch()
            
            layout.addLayout(param_layout)
    
    def get_settings(self) -> Dict[str, float]:
        """Get all augmentation settings."""
        return {key: spinbox.value() for key, spinbox in self.spinboxes.items()}
    
    def set_settings(self, settings: Dict[str, float]):
        """Set augmentation settings."""
        for key, value in settings.items():
            if key in self.spinboxes:
                self.spinboxes[key].setValue(value)