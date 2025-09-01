"""Training results visualization widget."""

from pathlib import Path
from typing import Optional, List, Dict
import csv
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QGridLayout, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QFileDialog, QMessageBox,
    QSplitter, QListWidget, QListWidgetItem, QTextEdit,
    QToolTip
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QEvent, QPoint
from PyQt6.QtGui import QPixmap, QFont, QPalette, QColor, QCursor
import pyqtgraph as pg


class FileListItemWidget(QWidget):
    """Custom widget for file list items with info icon."""
    
    clicked = pyqtSignal()
    
    def __init__(self, text: str, tooltip: str, file_path: Path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.tooltip_text = tooltip
        self.setup_ui(text)
    
    def cleanup(self):
        """Clean up event filters before deletion."""
        try:
            self.removeEventFilter(self)
        except Exception as e:
            print(f"[DEBUG] FileListItemWidget.cleanup: Error removing event filter: {e}")
        
    def setup_ui(self, text: str):
        """Setup the UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(5)
        
        # File name label
        self.label = QLabel(text)
        self.label.setStyleSheet("color: #cccccc; padding: 3px;")
        layout.addWidget(self.label, 1)
        
        # Info icon
        self.info_label = QLabel("ℹ️")
        self.info_label.setStyleSheet("""
            QLabel {
                color: #14ffec;
                font-size: 14px;
                padding: 2px;
            }
            QLabel:hover {
                color: #ffffff;
                background-color: #0d7377;
                border-radius: 3px;
            }
        """)
        self.info_label.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.info_label.setToolTip(self.tooltip_text)
        layout.addWidget(self.info_label)
        
        # Install event filter for the whole widget
        self.installEventFilter(self)
        
    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            
    def eventFilter(self, obj, event):
        """Show tooltip on hover over info icon."""
        if event.type() == QEvent.Type.Enter:
            self.setStyleSheet("background-color: #3c3c3c; border-radius: 3px;")
        elif event.type() == QEvent.Type.Leave:
            self.setStyleSheet("")
        return super().eventFilter(obj, event)


class ImageCard(QWidget):
    """Widget to display an image with title."""
    
    def __init__(self, title: str, image_path: Path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setup_ui(title)
    
    def setup_ui(self, title: str):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #14ffec;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 1px solid #555555;
                border-radius: 4px;
                background-color: #2b2b2b;
                padding: 5px;
            }
        """)
        
        # Load and scale image
        if self.image_path.exists():
            pixmap = QPixmap(str(self.image_path))
            if not pixmap.isNull():
                # Scale to larger size while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    800, 600,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setText("Failed to load image")
        else:
            self.image_label.setText("Image not found")
        
        layout.addWidget(self.image_label)
        
        # View full size button
        view_btn = QPushButton("🔍 View Full Size")
        view_btn.clicked.connect(self.view_full_size)
        view_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                border: none;
                border-radius: 4px;
                color: white;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #14ffec;
                color: #1e1e1e;
            }
        """)
        layout.addWidget(view_btn)
    
    def view_full_size(self):
        """Open image in default viewer."""
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Windows':
                subprocess.run(['start', str(self.image_path)], shell=True)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', str(self.image_path)])
            else:  # Linux
                subprocess.run(['xdg-open', str(self.image_path)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open image: {e}")


class TrainingResults(QWidget):
    """Widget for displaying training results."""
    
    resultsLoaded = pyqtSignal(Path)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_dir: Optional[Path] = None
        self._result_images: List[Path] = []
        self._results_data: List[Dict] = []
        self.setup_ui()
        self.setMinimumSize(800, 600)
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            if hasattr(self, '_cleanup_file_list'):
                self._cleanup_file_list()
        except Exception as e:
            print(f"[DEBUG] TrainingResults.__del__: Error during cleanup: {e}")
    
    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Apply styling
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QGroupBox {
                background-color: #2b2b2b;
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
                color: #14ffec;
            }
            QTableWidget {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                gridline-color: #3c3c3c;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #0d7377;
            }
            QHeaderView::section {
                background-color: #3c3c3c;
                border: none;
                padding: 5px;
                font-weight: bold;
            }
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #555555;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #3c3c3c;
            }
            QListWidget::item:selected {
                background-color: #0d7377;
            }
            QListWidget::item:hover {
                background-color: #3c3c3c;
            }
        """)
        
        # Header
        header_layout = QHBoxLayout()
        
        self.status_label = QLabel("No results loaded")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #14ffec;")
        header_layout.addWidget(self.status_label)
        
        header_layout.addStretch()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_results)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                border: none;
                border-radius: 4px;
                color: white;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #14ffec;
                color: #1e1e1e;
            }
        """)
        header_layout.addWidget(refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - File list
        left_panel = QGroupBox("Output Files")
        left_layout = QVBoxLayout(left_panel)
        
        # Add hint at top
        hint_label = QLabel("ℹ️ Hover over icons for details")
        hint_label.setStyleSheet("color: #888888; font-size: 11px; padding: 5px;")
        left_layout.addWidget(hint_label)
        
        self.file_list = QListWidget()
        self.file_list.setSpacing(2)
        self.file_list.setStyleSheet("""
            QListWidget::item {
                padding: 0px;
                margin: 0px;
            }
        """)
        left_layout.addWidget(self.file_list)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Content viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area for content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll_area.setWidget(self.content_widget)
        right_layout.addWidget(scroll_area)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 650])
        
        layout.addWidget(splitter, 1)  # Give splitter all available space
    
    def set_output_directory(self, output_dir: Path):
        """Set the output directory and load results."""
        try:
            self._output_dir = output_dir
            # Check if widget is properly initialized
            if not hasattr(self, 'file_list'):
                print(f"[WARNING] TrainingResults.set_output_directory: file_list not initialized yet")
                return
            # Delay refresh slightly to ensure widget is ready
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, self.refresh_results)
        except Exception as e:
            print(f"[ERROR] TrainingResults.set_output_directory: {e}")
            import traceback
            traceback.print_exc()
    
    def _cleanup_file_list(self):
        """Properly clean up file list widgets before clearing."""
        try:
            # Check if file_list exists
            if not hasattr(self, 'file_list') or not self.file_list:
                return
                
            # Clean up all custom widgets before clearing
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item:
                    widget = self.file_list.itemWidget(item)
                    if widget and isinstance(widget, FileListItemWidget):
                        try:
                            widget.cleanup()
                            self.file_list.removeItemWidget(item)
                        except Exception as e:
                            print(f"[DEBUG] TrainingResults._cleanup_file_list: Error cleaning widget {i}: {e}")
            # Now safe to clear
            self.file_list.clear()
        except RuntimeError as e:
            # Handle the "wrapped C/C++ object has been deleted" error
            if "deleted" in str(e):
                print(f"[DEBUG] TrainingResults._cleanup_file_list: Widget already deleted, skipping cleanup")
            else:
                print(f"[ERROR] TrainingResults._cleanup_file_list: RuntimeError during cleanup: {e}")
        except Exception as e:
            print(f"[ERROR] TrainingResults._cleanup_file_list: Error during cleanup: {e}")
            import traceback
            traceback.print_exc()
            # Try to clear anyway
            try:
                if hasattr(self, 'file_list') and self.file_list:
                    self.file_list.clear()
            except:
                pass
    
    def refresh_results(self):
        """Refresh the results display."""
        try:
            # Safety checks
            if not hasattr(self, 'status_label') or not hasattr(self, 'file_list'):
                print(f"[WARNING] TrainingResults.refresh_results: Widgets not initialized")
                return
                
            if not self._output_dir or not self._output_dir.exists():
                self.status_label.setText("No results directory set")
                return
            
            self.status_label.setText(f"Results from: {self._output_dir.name}")
            
            # Clear current content with proper cleanup
            self._cleanup_file_list()
            self.clear_content()
        except RuntimeError as e:
            if "deleted" in str(e):
                print(f"[WARNING] TrainingResults.refresh_results: Widget was deleted, skipping refresh")
                return
            else:
                raise
        except Exception as e:
            print(f"[ERROR] TrainingResults.refresh_results: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Find all relevant files
        files_to_show = []
        
        # Common result files
        result_patterns = [
            ("results.csv", "📊 Training Metrics"),
            ("results.png", "📈 Training Curves"),
            ("confusion_matrix.png", "🎯 Confusion Matrix"),
            ("confusion_matrix_normalized.png", "🎯 Normalized Confusion Matrix"),
            ("F1_curve.png", "📉 F1 Curve"),
            ("P_curve.png", "📉 Precision Curve"),
            ("R_curve.png", "📉 Recall Curve"),
            ("PR_curve.png", "📉 PR Curve"),
            ("labels.jpg", "🏷️ Ground Truth Labels"),
            ("labels_correlogram.jpg", "📊 Label Correlogram"),
            ("train_batch*.jpg", "🖼️ Training Batches"),
            ("val_batch*.jpg", "🖼️ Validation Batches"),
        ]
        
        # Tooltip descriptions for file types
        file_tooltips = {
            "results.csv": """<b>Training Metrics CSV</b><br><br>
            Contains all training metrics logged during the training process:<br>
            • <b>epoch</b>: Training epoch number<br>
            • <b>train/box_loss</b>: Bounding box regression loss<br>
            • <b>train/cls_loss</b>: Classification loss<br>
            • <b>train/dfl_loss</b>: Distribution focal loss<br>
            • <b>metrics/precision(B)</b>: Detection precision on validation set<br>
            • <b>metrics/recall(B)</b>: Detection recall on validation set<br>
            • <b>metrics/mAP50(B)</b>: Mean Average Precision at IoU=0.5<br>
            • <b>metrics/mAP50-95(B)</b>: Mean Average Precision at IoU=0.5:0.95<br>
            • <b>val/box_loss</b>: Validation box loss<br>
            • <b>val/cls_loss</b>: Validation classification loss<br>
            • <b>val/dfl_loss</b>: Validation DFL loss<br><br>
            Use this data to analyze training progression and identify issues.""",
            
            "results.png": """<b>Training Curves</b><br><br>
            Multi-plot visualization showing all key metrics over training epochs.<br>
            Helps you quickly assess model training quality and convergence.""",
            
            "confusion_matrix.png": """<b>Confusion Matrix</b><br><br>
            Shows actual vs predicted classes for validation data.<br>
            Diagonal = correct predictions, off-diagonal = errors.<br>
            Helps identify which classes are being confused.""",
            
            "confusion_matrix_normalized.png": """<b>Normalized Confusion Matrix</b><br><br>
            Same as confusion matrix but normalized to percentages.<br>
            Each row sums to 100%, making class comparison easier.""",
            
            "F1_curve.png": """<b>F1 Score Curve</b><br><br>
            F1 score (harmonic mean of precision & recall) vs confidence threshold.<br>
            Peak shows optimal threshold for balanced performance.""",
            
            "P_curve.png": """<b>Precision Curve</b><br><br>
            Precision vs confidence threshold.<br>
            Use when false positives are costly.""",
            
            "R_curve.png": """<b>Recall Curve</b><br><br>
            Recall vs confidence threshold.<br>
            Use when false negatives are costly.""",
            
            "PR_curve.png": """<b>Precision-Recall Curve</b><br><br>
            Shows precision-recall trade-off.<br>
            Area under curve indicates overall detection quality.""",
            
            "labels.jpg": """<b>Ground Truth Labels</b><br><br>
            Sample images with ground truth annotations.<br>
            Verify annotation quality and dataset diversity.""",
            
            "labels_correlogram.jpg": """<b>Label Correlogram</b><br><br>
            Correlation matrix of bounding box properties.<br>
            Low correlations indicate good dataset diversity.""",
            
            "train_batch": """<b>Training Batch Sample</b><br><br>
            Shows augmented training images as fed to model.<br>
            Verify augmentations are appropriate.""",
            
            "val_batch": """<b>Validation Batch Sample</b><br><br>
            Shows validation images without augmentation.<br>
            Should represent your target deployment data."""
        }
        
        # Add files to list
        try:
            for pattern, display_name in result_patterns:
                if '*' in pattern:
                    # Handle wildcards
                    base_pattern = pattern.replace('*', '')
                    for f in self._output_dir.glob(pattern):
                        # Find tooltip for batch files
                        tooltip = ""
                        for key, tt in file_tooltips.items():
                            if key in f.name:
                                tooltip = tt
                                break
                        
                        # Create custom widget
                        widget = FileListItemWidget(f"🖼️ {f.name}", tooltip, f)
                        widget.clicked.connect(lambda checked=False, p=f: self._on_file_clicked(p))
                        
                        # Add to list
                        item = QListWidgetItem()
                        item.setSizeHint(widget.sizeHint())
                        item.setData(Qt.ItemDataRole.UserRole, f)
                        self.file_list.addItem(item)
                        self.file_list.setItemWidget(item, widget)
                else:
                    file_path = self._output_dir / pattern
                    if file_path.exists():
                        tooltip = file_tooltips.get(pattern, "")
                        
                        # Create custom widget
                        widget = FileListItemWidget(display_name, tooltip, file_path)
                        widget.clicked.connect(lambda checked=False, p=file_path: self._on_file_clicked(p))
                        
                        # Add to list
                        item = QListWidgetItem()
                        item.setSizeHint(widget.sizeHint())
                        item.setData(Qt.ItemDataRole.UserRole, file_path)
                        self.file_list.addItem(item)
                        self.file_list.setItemWidget(item, widget)
            
            # Also check for model files
            for pt_file in self._output_dir.glob("*.pt"):
                # Model tooltip
                model_tooltip = f"""<b>YOLO Model File: {pt_file.name}</b><br><br>
            This is a trained YOLO model checkpoint.<br><br>
            <b>Common model files:</b><br>
            • <b>best.pt</b>: Best performing model (lowest validation loss)<br>
            • <b>last.pt</b>: Most recent checkpoint<br>
            • <b>epoch##.pt</b>: Checkpoint from specific epoch<br><br>
            <b>Usage:</b><br>
            • Load in Model Management tab for inference<br>
            • Use as starting point for continued training<br>
            • Export to other formats (ONNX, TensorFlow, etc.)<br><br>
            Size: {pt_file.stat().st_size / (1024*1024):.1f} MB"""
            
                # Create custom widget
                widget = FileListItemWidget(f"🤖 {pt_file.name}", model_tooltip, pt_file)
                widget.clicked.connect(lambda checked=False, p=pt_file: self._on_file_clicked(p))
                
                # Add to list
                item = QListWidgetItem()
                item.setSizeHint(widget.sizeHint())
                item.setData(Qt.ItemDataRole.UserRole, pt_file)
                self.file_list.addItem(item)
                self.file_list.setItemWidget(item, widget)
        except RuntimeError as e:
            if "deleted" in str(e):
                print(f"[WARNING] TrainingResults.refresh_results: Widget deleted during refresh")
                return
            else:
                raise
        except Exception as e:
            print(f"[ERROR] TrainingResults.refresh_results: Error populating file list: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Auto-select results.png if available
        results_png = self._output_dir / "results.png"
        if results_png.exists():
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == results_png:
                    self.file_list.setCurrentItem(item)
                    self._on_file_clicked(results_png)
                    break
        
        self.resultsLoaded.emit(self._output_dir)
    
    def _on_file_clicked(self, file_path: Path):
        """Handle file click from custom widget."""
        # Find and select the corresponding list item
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == file_path:
                self.file_list.setCurrentItem(item)
                self.on_file_selected(item)
                break
    
    def clear_content(self):
        """Clear the content area."""
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    @pyqtSlot(QListWidgetItem)
    def on_file_selected(self, item: QListWidgetItem):
        """Handle file selection."""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if not file_path or not file_path.exists():
            return
        
        self.clear_content()
        
        # Handle different file types
        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            self.display_image(file_path)
        elif file_path.suffix.lower() == '.csv':
            self.display_csv(file_path)
        elif file_path.suffix.lower() == '.pt':
            self.display_model_info(file_path)
        else:
            self.display_text_file(file_path)
    
    def display_image(self, image_path: Path):
        """Display an image file."""
        # Create container for image and description
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(10)
        
        # Create image card
        title = image_path.stem.replace('_', ' ').title()
        card = ImageCard(title, image_path)
        container_layout.addWidget(card)
        
        # Add description based on file name
        desc_label = QLabel()
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 15px;
                font-size: 12px;
                line-height: 1.5;
            }
        """)
        
        descriptions = {
            "results": """Training curves showing loss and metrics over epochs.
            
            This image displays multiple metrics plotted against training epochs:
            • Box Loss (red): Measures how well the model predicts bounding box coordinates
            • Cls Loss (blue): Classification loss - how accurately objects are classified
            • DFL Loss (green): Distribution Focal Loss for improved box regression
            • Precision (orange): Ratio of true positives to all positive predictions
            • Recall (purple): Ratio of true positives to all actual positives
            • mAP@0.5 (pink): Mean Average Precision at 50% IoU threshold
            • mAP@0.5:0.95 (brown): Average mAP across IoU thresholds from 0.5 to 0.95
            
            Lower loss values and higher precision/recall/mAP values indicate better performance.""",
            
            "confusion_matrix": """Shows prediction accuracy for each class.
            
            The confusion matrix visualizes how well the model classifies each object type:
            • Rows represent actual (ground truth) classes
            • Columns represent predicted classes
            • Diagonal values show correct predictions
            • Off-diagonal values show misclassifications
            • Darker colors indicate higher counts
            
            A good model has a strong diagonal (high values) with minimal off-diagonal values.""",
            
            "confusion_matrix_normalized": """Normalized confusion matrix showing prediction rates.
            
            Similar to the regular confusion matrix but normalized by row (true class):
            • Values range from 0.0 to 1.0 (0% to 100%)
            • Each row sums to 1.0
            • Shows the percentage of each true class that was predicted as each class
            • Helps identify which classes are most often confused with each other
            
            Ideal performance shows 1.0 on the diagonal and 0.0 elsewhere.""",
            
            "F1_curve": """F1 score vs confidence threshold.
            
            The F1 score is the harmonic mean of precision and recall:
            • X-axis: Confidence threshold (0 to 1)
            • Y-axis: F1 score (0 to 1)
            • Shows how F1 score changes as you adjust the detection confidence threshold
            • Peak indicates optimal confidence threshold for balanced performance
            
            Use this to find the best confidence threshold for your application.""",
            
            "P_curve": """Precision vs confidence threshold.
            
            Shows how precision changes with different confidence thresholds:
            • X-axis: Confidence threshold (0 to 1)
            • Y-axis: Precision (0 to 1)
            • Higher thresholds typically yield higher precision but lower recall
            • Precision = True Positives / (True Positives + False Positives)
            
            If false positives are costly, choose a threshold with high precision.""",
            
            "R_curve": """Recall vs confidence threshold.
            
            Shows how recall changes with different confidence thresholds:
            • X-axis: Confidence threshold (0 to 1)
            • Y-axis: Recall (0 to 1)
            • Lower thresholds typically yield higher recall but lower precision
            • Recall = True Positives / (True Positives + False Negatives)
            
            If missing detections are costly, choose a threshold with high recall.""",
            
            "PR_curve": """Precision-Recall curve showing the trade-off.
            
            Visualizes the precision-recall trade-off:
            • X-axis: Recall (0 to 1)
            • Y-axis: Precision (0 to 1)
            • Area under the curve (AUC) indicates overall performance
            • Perfect detector would have AUC = 1.0
            • Random detector would follow the diagonal
            
            A curve that stays high and to the right indicates good performance.""",
            
            "labels": """Distribution of ground truth labels in the dataset.
            
            Shows sample images with their ground truth bounding boxes:
            • Each bounding box is labeled with its class
            • Helps verify that annotations are correct
            • Shows the variety of objects in your dataset
            • Useful for identifying annotation errors or biases
            
            Check that boxes tightly fit objects and classes are correctly labeled.""",
            
            "labels_correlogram": """Correlation between label dimensions.
            
            Analyzes relationships between bounding box properties:
            • Shows correlations between x, y positions, width, and height
            • Helps identify biases in object locations or sizes
            • Strong correlations might indicate dataset limitations
            • Each cell shows correlation coefficient (-1 to 1)
            
            Ideally, you want diverse object positions and sizes (low correlations).""",
            
            "train_batch": """Sample training batch with augmentations.
            
            Shows a batch of training images after augmentation:
            • Displays the actual augmented images fed to the model
            • Includes any enabled augmentations (rotation, scaling, color changes, etc.)
            • Bounding boxes are adjusted to match augmentations
            • Helps verify augmentation settings are appropriate
            
            Check that augmentations are realistic and don't distort objects excessively.""",
            
            "val_batch": """Sample validation batch without augmentations.
            
            Shows a batch of validation images:
            • No augmentations applied (original images)
            • Used to evaluate model performance
            • Should represent real-world data distribution
            • Bounding boxes show ground truth annotations
            
            Validation data should be representative of your deployment scenario."""
        }
        
        for key, desc in descriptions.items():
            if key in image_path.stem:
                desc_label.setText(desc)
                container_layout.addWidget(desc_label)
                break
        
        # Add stretch to push content to top
        container_layout.addStretch()
        
        # Add container to main layout
        self.content_layout.addWidget(container)
    
    def display_csv(self, csv_path: Path):
        """Display CSV file as a graph and table."""
        try:
            # Read CSV data
            data = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
            
            if not data:
                self.content_layout.addWidget(QLabel("No data in CSV file"))
                return
            
            # Create tabs for graph and table view
            from PyQt6.QtWidgets import QTabWidget
            tabs = QTabWidget()
            tabs.setStyleSheet("""
                QTabWidget::pane {
                    border: 1px solid #555555;
                    background-color: #2b2b2b;
                }
                QTabBar::tab {
                    background-color: #3c3c3c;
                    color: #cccccc;
                    padding: 8px 16px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #0d7377;
                    color: #ffffff;
                }
            """)
            
            # Graph tab
            graph_widget = self.create_csv_graph(data)
            tabs.addTab(graph_widget, "📊 Graph View")
            
            # Table tab
            table_widget = self.create_csv_table(data)
            tabs.addTab(table_widget, "📋 Table View")
            
            self.content_layout.addWidget(tabs)
            
        except Exception as e:
            error_label = QLabel(f"Error reading CSV: {str(e)}")
            error_label.setStyleSheet("color: #ff6666;")
            self.content_layout.addWidget(error_label)
    
    def create_csv_graph(self, data: List[Dict]) -> QWidget:
        """Create a graph visualization of CSV data."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create plot widget
        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('#2b2b2b')
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Style the plot
        plot_widget.setLabel('bottom', 'Epoch', color='#cccccc', size='12pt')
        plot_widget.setLabel('left', 'Value', color='#cccccc', size='12pt')
        plot_widget.setTitle('Training Metrics Over Time', color='#14ffec', size='14pt')
        plot_widget.getAxis('bottom').setTextPen('#cccccc')
        plot_widget.getAxis('left').setTextPen('#cccccc')
        
        # Common metrics to plot with better descriptions and more distinguishable colors
        metrics_to_plot = [
            ('train/box_loss', 'Box Loss (Bounding Box Regression)', '#ff3838'),  # Bright red
            ('train/cls_loss', 'Class Loss (Classification Accuracy)', '#00d4aa'),  # Teal/cyan
            ('train/dfl_loss', 'DFL Loss (Distribution Focal Loss)', '#ffa726'),  # Orange
            ('metrics/precision(B)', 'Precision (True Positives / All Positives)', '#66bb6a'),  # Green
            ('metrics/recall(B)', 'Recall (True Positives / All Ground Truth)', '#ab47bc'),  # Purple
            ('metrics/mAP50(B)', 'mAP@50 (Mean Avg Precision @ IoU=0.5)', '#42a5f5'),  # Light blue
            ('metrics/mAP50-95(B)', 'mAP@50-95 (Mean Avg Precision @ IoU=0.5:0.95)', '#ffee58'),  # Yellow
        ]
        
        # Extract epochs
        epochs = []
        for row in data:
            epoch_str = row.get('epoch', '').strip()
            if epoch_str:
                try:
                    epochs.append(float(epoch_str))
                except ValueError:
                    continue
        
        if not epochs:
            layout.addWidget(QLabel("No epoch data found"))
            return widget
        
        # Plot each metric
        legend_items = []
        for metric_key, full_label, color in metrics_to_plot:
            values = []
            for row in data:
                value_str = row.get(metric_key, '').strip()
                if value_str:
                    try:
                        values.append(float(value_str))
                    except ValueError:
                        values.append(None)
                else:
                    values.append(None)
            
            # Filter out None values
            valid_points = [(e, v) for e, v in zip(epochs, values) if v is not None]
            
            if valid_points:
                x_vals, y_vals = zip(*valid_points)
                pen = pg.mkPen(color=color, width=2)
                # Use shorter label for legend, full label is in metrics_to_plot
                short_label = full_label.split('(')[0].strip()
                line = plot_widget.plot(x_vals, y_vals, pen=pen, name=short_label)
                legend_items.append((line, short_label))
        
        # Add legend with better styling and positioning
        if legend_items:
            legend = plot_widget.addLegend(offset=(10, 10))
            legend.setParentItem(plot_widget.getPlotItem())
            # Style the legend
            legend.setBrush(pg.mkBrush(color=(43, 43, 43, 200)))  # Semi-transparent dark background
            legend.setPen(pg.mkPen(color=(85, 85, 85), width=1))  # Border
            # Position legend in top-right corner
            legend.anchor((1, 0), (1, 0), offset=(-10, 10))
        
        layout.addWidget(plot_widget)
        
        # Add detailed metric key/legend
        metric_key = QLabel("""<b>Metric Key:</b><br>
<span style='color: #ff3838;'>━━</span> <b>Box Loss:</b> Bounding box regression accuracy (lower is better)<br>
<span style='color: #00d4aa;'>━━</span> <b>Class Loss:</b> Object classification accuracy (lower is better)<br>
<span style='color: #ffa726;'>━━</span> <b>DFL Loss:</b> Distribution focal loss for precise box edges (lower is better)<br>
<span style='color: #66bb6a;'>━━</span> <b>Precision:</b> % of detections that are correct (higher is better, 0-1 scale)<br>
<span style='color: #ab47bc;'>━━</span> <b>Recall:</b> % of actual objects detected (higher is better, 0-1 scale)<br>
<span style='color: #42a5f5;'>━━</span> <b>mAP@50:</b> Mean Average Precision at 50% IoU threshold (higher is better, 0-1 scale)<br>
<span style='color: #ffee58;'>━━</span> <b>mAP@50-95:</b> Mean Average Precision averaged from 50-95% IoU (higher is better, 0-1 scale)<br>
<br>
<i>💡 Tip: Use mouse wheel to zoom, drag to pan. Validation metrics (Precision, Recall, mAP) appear periodically during validation phases.</i>""")
        metric_key.setWordWrap(True)
        metric_key.setTextFormat(Qt.TextFormat.RichText)
        metric_key.setStyleSheet("""
            QLabel {
                color: #cccccc;
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 10px;
                font-size: 11px;
            }
        """)
        layout.addWidget(metric_key)
        
        return widget
    
    def create_csv_table(self, data: List[Dict]) -> QTableWidget:
        """Create a table view of CSV data."""
        table = QTableWidget()
        
        if not data:
            return table
        
        # Set up columns
        columns = list(data[0].keys())
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        
        # Add rows
        table.setRowCount(len(data))
        for row_idx, row_data in enumerate(data):
            for col_idx, col_name in enumerate(columns):
                value = row_data.get(col_name, '')
                item = QTableWidgetItem(str(value))
                
                # Format numbers nicely
                try:
                    float_val = float(value)
                    if '.' in value:
                        item.setText(f"{float_val:.4f}")
                except ValueError:
                    pass
                
                table.setItem(row_idx, col_idx, item)
        
        # Adjust column widths
        table.resizeColumnsToContents()
        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        
        return table
    
    def display_model_info(self, model_path: Path):
        """Display model file information."""
        info_text = f"""
        <h3 style='color: #14ffec;'>Model File: {model_path.name}</h3>
        <p><b>Path:</b> {model_path}</p>
        <p><b>Size:</b> {model_path.stat().st_size / (1024*1024):.1f} MB</p>
        <p><b>Modified:</b> {datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}</p>
        <br>
        <p style='color: #888888;'>This is the trained YOLO model. You can load it in the Model Management tab for inference or further training.</p>
        """
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setTextFormat(Qt.TextFormat.RichText)
        self.content_layout.addWidget(info_label)
        
        # Copy path button
        copy_btn = QPushButton("📋 Copy Model Path")
        copy_btn.clicked.connect(lambda: self.copy_to_clipboard(str(model_path)))
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                border: none;
                border-radius: 4px;
                color: white;
                padding: 8px 16px;
                max-width: 200px;
            }
            QPushButton:hover {
                background-color: #14ffec;
                color: #1e1e1e;
            }
        """)
        self.content_layout.addWidget(copy_btn)
    
    def display_text_file(self, file_path: Path):
        """Display a text file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setPlainText(content)
            text_edit.setFont(QFont("Courier", 10))
            self.content_layout.addWidget(text_edit)
            
        except Exception as e:
            error_label = QLabel(f"Error reading file: {str(e)}")
            error_label.setStyleSheet("color: #ff6666;")
            self.content_layout.addWidget(error_label)
    
    def copy_to_clipboard(self, text: str):
        """Copy text to clipboard."""
        from PyQt6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        
        # Show temporary message
        QMessageBox.information(self, "Success", "Path copied to clipboard!", QMessageBox.StandardButton.Ok)