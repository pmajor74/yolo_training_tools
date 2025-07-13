"""Real-time training charts widget using matplotlib."""

from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
import re

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QToolTip, QPushButton, QMenu
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QTimer, QEvent, QPoint
from PyQt6.QtGui import QCursor, QAction, QPixmap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from ..core.constants import COLOR_MANAGER


class TrainingCharts(QWidget):
    """Widget for displaying real-time training metrics charts."""
    
    # Signals
    chartUpdated = pyqtSignal()
    
    # Chart information for tooltips
    CHART_INFO = {
        "Loss Overview": {
            "description": "Overall training loss trend",
            "details": """Shows the total loss value over training steps. This is the sum of all individual losses that the model is trying to minimize.

<b>What is Loss?</b>
Loss measures how wrong the model's predictions are. Lower values mean better predictions.

<b>Components Combined:</b>
• Box regression loss (location accuracy)
• Classification loss (object type accuracy)  
• Distribution focal loss (improved box predictions)

<b>Reading the Chart:</b>
• X-axis: Training steps (batches processed)
• Y-axis: Total loss value
• Blue line: Actual loss values
• Red dashed: Smoothed trend (if enough data)""",
            "good": "• Steadily decreasing trend\n• Smooth curve without spikes\n• Plateaus after initial drop\n• Final loss < 1.0 is typically good",
            "bad": "• Increasing or erratic loss\n• Sharp spikes indicate instability\n• No improvement over time\n• Loss stuck above 5.0"
        },
        "Detailed Losses": {
            "description": "Individual loss components breakdown",
            "details": """Displays separate loss values to diagnose specific issues:

<b>Box Loss (Red):</b>
Measures how accurately the model predicts bounding box coordinates.
• Should be lowest of the three losses
• Values typically 0.5-2.0 when converged

<b>Class Loss (Cyan):</b>
Measures how accurately the model classifies objects.
• Higher with more classes
• Should decrease steadily

<b>DFL Loss (Blue):</b>
Distribution Focal Loss improves box edge predictions.
• Usually similar to box loss
• Helps with precise boundaries

<b>Total Loss (Green):</b>
Sum of all losses - overall optimization target.""",
            "good": "• All losses decrease together\n• Box loss typically lowest\n• Smooth convergence\n• Proportional decrease rates",
            "bad": "• One loss dominates others\n• Losses diverging\n• Unstable oscillations\n• Class loss not decreasing (classification issues)"
        },
        "Train vs Val Loss": {
            "description": "Training vs Validation loss comparison",
            "details": """Compares training and validation losses to detect overfitting.

<b>Training Loss (Blue):</b>
Loss calculated on training data that the model learns from.

<b>Validation Loss (Red with markers):</b>
Loss on held-out validation data that tests generalization.

<b>Key Insights:</b>
• Gap indicates overfitting potential
• Validation only calculated periodically (hence markers)
• Both should decrease together ideally

<b>Overfitting Signs:</b>
The red shaded area appears when validation loss increases while training loss decreases - a clear sign the model is memorizing rather than learning.""",
            "good": "• Both losses decrease\n• Val loss close to train loss\n• Gap < 20% of train loss\n• Parallel downward trends",
            "bad": "• Val loss increases while train decreases (overfitting)\n• Large gap between losses (> 50%)\n• Val loss plateaus early\n• Diverging trends"
        },
        "mAP Metrics": {
            "description": "Mean Average Precision scores",
            "details": """mAP (Mean Average Precision) is the primary metric for object detection quality.

<b>mAP@50 (Green with circles):</b>
• Measures detection accuracy when IoU ≥ 0.5
• IoU = Intersection over Union (overlap between predicted and actual boxes)
• 50% overlap required for "correct" detection
• Good for general detection tasks

<b>mAP@50-95 (Purple with squares):</b>
• Average mAP from IoU 0.5 to 0.95 (step 0.05)
• Much stricter - requires precise boundaries
• Industry standard metric
• Always lower than mAP@50

<b>Value Interpretation:</b>
• 0.0-0.3: Poor detection
• 0.3-0.5: Acceptable
• 0.5-0.7: Good
• 0.7-0.9: Excellent
• >0.9: State-of-the-art

⚠️ These appear only during validation (typically every 10 epochs).""",
            "good": "• Steadily increasing values\n• mAP@50 > 0.5\n• mAP@50-95 > 0.3\n• Consistent improvement\n• Gap between metrics < 0.4",
            "bad": "• Decreasing or stuck at 0\n• Large gap between metrics (> 0.5)\n• No improvement after many epochs\n• Sudden drops"
        },
        "Precision & Recall": {
            "description": "Detection quality metrics",
            "details": """Fundamental metrics showing different aspects of detection performance.

<b>Precision (Cyan with triangles):</b>
• Of all detections made, what percentage are correct?
• High precision = few false positives
• Important when false alarms are costly
• Formula: True Positives / (True Positives + False Positives)

<b>Recall (Yellow with inverted triangles):</b>
• Of all actual objects, what percentage are detected?
• High recall = few missed detections  
• Important when missing objects is costly
• Formula: True Positives / (True Positives + False Negatives)

<b>Trade-off:</b>
• Increasing confidence threshold → Higher precision, lower recall
• Decreasing confidence threshold → Higher recall, lower precision
• F1 score balances both (2 × Precision × Recall / (Precision + Recall))

⚠️ Calculated during validation phases only.""",
            "good": "• Both values > 0.7\n• Balanced (within 0.1 of each other)\n• Steady improvement\n• Final values > 0.8",
            "bad": "• Values stuck below 0.5\n• Large imbalance (> 0.2 difference)\n• Decreasing trends\n• Precision high but recall very low (missing objects)"
        },
        "All Metrics": {
            "description": "Combined view of all training metrics",
            "details": """Overview showing all metrics in a 2x2 grid for comprehensive monitoring.

<b>Top Left - Losses:</b>
Shows total loss trend for quick training health check.

<b>Top Right - mAP Metrics:</b>
Shows detection accuracy metrics when available.

<b>Bottom panels:</b>
Reserved for additional metrics as they become available.

<b>Usage:</b>
• Quick overall training assessment
• Identify which aspect needs attention
• Spot correlations between metrics
• Monitor multiple aspects simultaneously""",
            "good": "• Losses decreasing\n• mAP increasing\n• Smooth curves\n• All metrics improving together",
            "bad": "• Contradictory trends\n• Flat lines (no learning)\n• High volatility\n• Some metrics improving while others worsen"
        }
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage
        self._epochs: List[int] = []
        self._steps: List[int] = []
        self._metrics: Dict[str, List[float]] = {
            'loss': [],
            'box_loss': [],
            'cls_loss': [],
            'dfl_loss': [],
            'mAP50': [],
            'mAP50-95': [],
            'precision': [],
            'recall': [],
            'val_loss': [],
            'val_box_loss': [],
            'val_cls_loss': [],
            'val_dfl_loss': []
        }
        
        # Keep last N points for real-time display
        self._max_points = 1000
        self._current_epoch = 0
        self._total_steps = 0
        
        # Chart update control
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_charts)
        self._update_timer.setInterval(1000)  # Update every second
        self._pending_update = False
        self._paused = False
        
        # Interactive features
        self._crosshair_enabled = False
        self._grid_enabled = True
        self._legend_items = {}  # Store legend line references
        
        
        # Get graph colors from color manager
        self._graph_colors = COLOR_MANAGER.get_graph_colors()
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Chart Type:"))
        self.chart_combo = QComboBox()
        self.chart_combo.addItems([
            "Loss Overview",
            "Detailed Losses",
            "Train vs Val Loss",
            "mAP Metrics", 
            "Precision & Recall",
            "All Metrics"
        ])
        self.chart_combo.currentIndexChanged.connect(self._on_chart_type_changed)
        controls_layout.addWidget(self.chart_combo)
        
        # Info hint
        info_label = QLabel("ℹ️")
        info_label.setToolTip("Hover over the chart for detailed information")
        info_label.setStyleSheet("color: #14ffec; font-size: 16px; padding: 0 5px;")
        controls_layout.addWidget(info_label)
        
        # Pause/Resume button
        self.pause_btn = QPushButton("⏸")
        self.pause_btn.setToolTip("Pause/Resume real-time updates")
        self.pause_btn.setMaximumWidth(30)
        self.pause_btn.clicked.connect(self._toggle_pause)
        controls_layout.addWidget(self.pause_btn)
        
        # Export button
        self.export_btn = QPushButton("💾")
        self.export_btn.setToolTip("Export chart")
        self.export_btn.setMaximumWidth(30)
        self.export_btn.clicked.connect(self._show_export_menu)
        controls_layout.addWidget(self.export_btn)
        
        # Options button
        self.options_btn = QPushButton("⚙")
        self.options_btn.setToolTip("Chart options")
        self.options_btn.setMaximumWidth(30)
        self.options_btn.clicked.connect(self._show_options_menu)
        controls_layout.addWidget(self.options_btn)
        
        controls_layout.addStretch()
        
        # Status label
        self.status_label = QLabel("No data")
        self.status_label.setStyleSheet("color: #888888; font-size: 11px;")
        controls_layout.addWidget(self.status_label)
        
        layout.addLayout(controls_layout)
        
        # Create matplotlib figure
        plt.style.use('dark_background')  # Dark theme
        self.figure = Figure(figsize=(10, 6), dpi=80)
        self.figure.patch.set_facecolor('#2b2b2b')
        
        # Create canvas
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #2b2b2b;")
        
        # Add matplotlib navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("""
            NavigationToolbar2QT {
                background-color: #2b2b2b;
                border: none;
            }
            NavigationToolbar2QT QToolButton {
                background-color: #3c3c3c;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
                margin: 2px;
            }
            NavigationToolbar2QT QToolButton:hover {
                background-color: #4c4c4c;
                border-color: #14ffec;
            }
            NavigationToolbar2QT QToolButton:checked {
                background-color: #14ffec;
                color: #2b2b2b;
            }
        """)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        
        # Install event filter for hover tooltips
        self.canvas.installEventFilter(self)
        self.canvas.setMouseTracking(True)
        self.canvas.setToolTip("Hover for chart information")
        
        # Connect mouse events for interactive features
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self._on_mouse_click)
        
        # Initialize with empty plot
        self._init_plot()
        
    def _init_plot(self):
        """Initialize empty plot."""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.grid(self._grid_enabled, alpha=0.3, linestyle='--')
        self.ax.set_xlabel('Steps', fontsize=10)
        self.ax.set_ylabel('Value', fontsize=10)
        self.ax.set_title('Training Metrics', fontsize=12, fontweight='bold')
        
        # Style the axes
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#555555')
        self.ax.spines['bottom'].set_color('#555555')
        self.ax.tick_params(colors='#888888', labelsize=9)
        
        # Initialize crosshair lines (hidden by default)
        self.crosshair_vline = None
        self.crosshair_hline = None
        self.value_text = None
        self.ax = None
        
        self.canvas.draw()
        
    def start_monitoring(self):
        """Start monitoring for chart updates."""
        self._update_timer.start()
        
    def stop_monitoring(self):
        """Stop monitoring."""
        self._update_timer.stop()
        # Do a final update to ensure the last data is displayed
        if self._pending_update:
            self._update_charts()
        
    def clear_data(self):
        """Clear all chart data."""
        self._epochs.clear()
        self._steps.clear()
        for key in self._metrics:
            self._metrics[key].clear()
        self._current_epoch = 0
        self._total_steps = 0
        self._init_plot()
        self.status_label.setText("No data")
        
    def refresh_chart(self):
        """Force refresh the current chart display."""
        self._pending_update = True
        self._update_charts()
        
    @pyqtSlot(int, int)
    def on_epoch_update(self, current_epoch: int, total_epochs: int):
        """Handle epoch update."""
        self._current_epoch = current_epoch
        
    @pyqtSlot(str)
    def on_metrics_update(self, metrics_str: str):
        """Parse and store metrics from training output."""
        # Debug logging - commented out
        # if "val" in metrics_str.lower() or "Val" in metrics_str:
        #     print(f"[CHART DEBUG] Received metrics with 'val': {metrics_str}")
            
        # Parse metrics string like "Loss: 1.234 | Box: 0.567 | Cls: 0.890"
        if not metrics_str or metrics_str == "Loss: --, mAP: --":
            return
        
        # Check if this is a validation update (don't increment steps for validation)
        is_validation_update = "Val Loss:" in metrics_str or "val" in metrics_str.lower()
        
        # Only increment steps for training updates, not validation
        if not is_validation_update and "Loss:" in metrics_str:
            self._total_steps += 1
            self._steps.append(self._total_steps)
            self._epochs.append(self._current_epoch)
        
        # Parse different metric formats
        metrics_found = False
        
        # Parse loss values (skip if this is a validation update)
        if "Loss:" in metrics_str and not is_validation_update:
            match = re.search(r'Loss:\s*([\d.]+)', metrics_str)
            if match:
                self._add_metric('loss', float(match.group(1)))
                metrics_found = True
                
        # Parse box loss (skip if this is a validation update)
        if "Box:" in metrics_str and not is_validation_update:
            match = re.search(r'Box:\s*([\d.]+)', metrics_str)
            if match:
                self._add_metric('box_loss', float(match.group(1)))
                metrics_found = True
                
        # Parse cls loss (skip if this is a validation update)
        if "Cls:" in metrics_str and not is_validation_update:
            match = re.search(r'Cls:\s*([\d.]+)', metrics_str)
            if match:
                self._add_metric('cls_loss', float(match.group(1)))
                metrics_found = True
                
        # Parse mAP values
        if "mAP50:" in metrics_str:
            match = re.search(r'mAP50:\s*([\d.]+)', metrics_str)
            if match:
                self._add_metric('mAP50', float(match.group(1)))
                metrics_found = True
                
        if "mAP:" in metrics_str and "mAP50:" not in metrics_str:
            match = re.search(r'mAP:\s*([\d.]+)', metrics_str)
            if match:
                self._add_metric('mAP50-95', float(match.group(1)))
                metrics_found = True
                
        # Parse precision and recall
        if "P:" in metrics_str:
            match = re.search(r'P:\s*([\d.]+)', metrics_str)
            if match:
                self._add_metric('precision', float(match.group(1)))
                metrics_found = True
                
        if "R:" in metrics_str:
            match = re.search(r'R:\s*([\d.]+)', metrics_str)
            if match:
                self._add_metric('recall', float(match.group(1)))
                metrics_found = True
                
        # Parse validation loss - multiple patterns
        # Pattern 1: "Val Loss: X.XXX"
        if "Val Loss:" in metrics_str:
            match = re.search(r'Val Loss:\s*([\d.]+)', metrics_str)
            if match:
                val_loss = float(match.group(1))
                # print(f"[CHART DEBUG] Found Val Loss: {val_loss}")
                self._add_metric('val_loss', val_loss)
                metrics_found = True
                # print(f"[CHART DEBUG] val_loss list now has {len(self._metrics['val_loss'])} values")
                
        # Pattern 2: Validation losses in other formats
        elif "val" in metrics_str.lower() and "loss" in metrics_str.lower():
            match = re.search(r'val.*loss:\s*([\d.]+)', metrics_str, re.IGNORECASE)
            if match:
                val_loss = float(match.group(1))
                # print(f"[CHART DEBUG] Found val loss pattern 2: {val_loss}")
                self._add_metric('val_loss', val_loss)
                metrics_found = True
                
        # Pattern 3: Individual validation losses
        if "val/box_loss" in metrics_str:
            match = re.search(r'val/box_loss:\s*([\d.]+)', metrics_str)
            if match:
                self._add_metric('val_box_loss', float(match.group(1)))
                metrics_found = True
                
        if "val/cls_loss" in metrics_str:
            match = re.search(r'val/cls_loss:\s*([\d.]+)', metrics_str)
            if match:
                self._add_metric('val_cls_loss', float(match.group(1)))
                metrics_found = True
                
        if "val/dfl_loss" in metrics_str:
            match = re.search(r'val/dfl_loss:\s*([\d.]+)', metrics_str)
            if match:
                self._add_metric('val_dfl_loss', float(match.group(1)))
                metrics_found = True
                
        # If we have all three validation losses but no total, calculate it
        if (self._metrics['val_box_loss'] and self._metrics['val_cls_loss'] and 
            self._metrics['val_dfl_loss'] and not self._metrics['val_loss']):
            # Use the latest values to calculate total
            total_val_loss = (self._metrics['val_box_loss'][-1] + 
                            self._metrics['val_cls_loss'][-1] + 
                            self._metrics['val_dfl_loss'][-1])
            self._add_metric('val_loss', total_val_loss)
        
        if metrics_found:
            self._pending_update = True
            self.status_label.setText(f"Step {self._total_steps} | Epoch {self._current_epoch}")
            
    def _add_metric(self, metric_name: str, value: float):
        """Add a metric value, maintaining max points limit."""
        if metric_name in self._metrics:
            self._metrics[metric_name].append(value)
            
            # Trim to max points
            if len(self._metrics[metric_name]) > self._max_points:
                self._metrics[metric_name] = self._metrics[metric_name][-self._max_points:]
                
    @pyqtSlot()
    def _update_charts(self):
        """Update charts if there's new data."""
        if not self._pending_update or self._paused:
            return
            
        self._pending_update = False
        chart_type = self.chart_combo.currentText()
        
        # Clear figure
        self.figure.clear()
        
        if chart_type == "Loss Overview":
            self._plot_loss_overview()
        elif chart_type == "Detailed Losses":
            self._plot_detailed_losses()
        elif chart_type == "Train vs Val Loss":
            self._plot_train_val_loss()
        elif chart_type == "mAP Metrics":
            self._plot_map_metrics()
        elif chart_type == "Precision & Recall":
            self._plot_precision_recall()
        else:  # All Metrics
            self._plot_all_metrics()
            
        # Redraw
        self.figure.tight_layout()
        self.canvas.draw()
        self.chartUpdated.emit()
        
    def _plot_loss_overview(self):
        """Plot overall loss trend."""
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self._legend_items.clear()
        
        if self._metrics['loss']:
            steps = self._steps[-len(self._metrics['loss']):]
            line, = self.ax.plot(steps, self._metrics['loss'], color=self._graph_colors[0], linewidth=2, label='Total Loss')
            self._legend_items['Total Loss'] = line
            
            # Add smoothed line
            if len(self._metrics['loss']) > 10:
                try:
                    from scipy.ndimage import gaussian_filter1d
                    smoothed = gaussian_filter1d(self._metrics['loss'], sigma=5)
                    smoothed_line, = self.ax.plot(steps, smoothed, color=self._graph_colors[1], linestyle='--', linewidth=1, alpha=0.7, label='Smoothed')
                    self._legend_items['Smoothed'] = smoothed_line
                except ImportError:
                    # scipy not available, skip smoothing
                    pass
                
        self.ax.set_xlabel('Training Steps', fontsize=10)
        self.ax.set_ylabel('Loss', fontsize=10)
        self.ax.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
        self.ax.grid(self._grid_enabled, alpha=0.3, linestyle='--')
        
        # Create interactive legend
        legend = self.ax.legend(loc='upper right', fontsize=9)
        self._make_legend_interactive(legend)
        
        self._style_axes(self.ax)
        
    def _plot_train_val_loss(self):
        """Plot training vs validation loss comparison."""
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self._legend_items.clear()
        
        has_data = False
        has_val_data = False
        
        if self._metrics['loss']:
            steps = self._steps[-len(self._metrics['loss']):]
            line, = self.ax.plot(steps, self._metrics['loss'], color=self._graph_colors[0], linewidth=2, 
                   label='Training Loss', alpha=0.8)
            self._legend_items['Training Loss'] = line
            has_data = True
            
        if self._metrics['val_loss']:
            steps = self._steps[-len(self._metrics['val_loss']):]
            line, = self.ax.plot(steps, self._metrics['val_loss'], color=self._graph_colors[1], linewidth=2,
                   label='Validation Loss', marker='o', markersize=4)
            self._legend_items['Validation Loss'] = line
            has_data = True
            has_val_data = True
        else:
            # Add placeholder for validation loss in legend
            line, = self.ax.plot([], [], color=self._graph_colors[1], linewidth=2, label='Validation Loss (pending)', 
                               marker='o', markersize=4, alpha=0.5)
            self._legend_items['Validation Loss'] = line
            
            # Highlight overfitting region if val loss increases while train decreases
            if len(self._metrics['loss']) > 10 and len(self._metrics['val_loss']) > 2:
                # Simple overfitting detection
                recent_train = self._metrics['loss'][-10:]
                recent_val = self._metrics['val_loss'][-2:]
                if len(recent_val) >= 2 and recent_val[-1] > recent_val[-2] and recent_train[-1] < recent_train[0]:
                    self.ax.axvspan(steps[-50] if len(steps) > 50 else steps[0], steps[-1], 
                             alpha=0.1, color='red', label='Potential Overfitting')
        
        if not has_data:
            self.ax.text(0.5, 0.5, 'Waiting for training data...', 
                   transform=self.ax.transAxes, ha='center', va='center',
                   fontsize=14, color='#888888', style='italic')
                    
        self.ax.set_xlabel('Training Steps', fontsize=10)
        self.ax.set_ylabel('Loss', fontsize=10)
        self.ax.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
        self.ax.grid(self._grid_enabled, alpha=0.3, linestyle='--')
        
        # Always show legend to indicate what lines will appear
        legend = self.ax.legend(loc='upper right', fontsize=9)
        self._make_legend_interactive(legend)
        
        # Add note if validation hasn't occurred yet
        if not has_val_data and has_data:
            self.ax.text(0.02, 0.02, 'Validation occurs at end of each epoch', 
                        transform=self.ax.transAxes, fontsize=8, color='#888888',
                        verticalalignment='bottom', style='italic')
            
        self._style_axes(self.ax)
        
    def _plot_detailed_losses(self):
        """Plot individual loss components."""
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self._legend_items.clear()
        
        # Plot available losses
        loss_types = [
            ('box_loss', 'Box Loss', '#ff6b6b'),
            ('cls_loss', 'Class Loss', '#4ecdc4'),
            ('dfl_loss', 'DFL Loss', '#45b7d1'),
            ('loss', 'Total Loss', '#96ceb4')
        ]
        
        for key, label, color in loss_types:
            if self._metrics[key]:
                steps = self._steps[-len(self._metrics[key]):]
                line, = self.ax.plot(steps, self._metrics[key], color=color, linewidth=2, 
                       label=label, alpha=0.8)
                self._legend_items[label] = line
                
        self.ax.set_xlabel('Training Steps', fontsize=10)
        self.ax.set_ylabel('Loss Value', fontsize=10)
        self.ax.set_title('Detailed Loss Components', fontsize=12, fontweight='bold')
        self.ax.grid(self._grid_enabled, alpha=0.3, linestyle='--')
        
        if self._legend_items:
            legend = self.ax.legend(loc='upper right', fontsize=9)
            self._make_legend_interactive(legend)
        
        self._style_axes(self.ax)
        
    def _plot_map_metrics(self):
        """Plot mAP metrics."""
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self._legend_items.clear()
        
        has_data = False
        
        if self._metrics['mAP50']:
            steps = self._steps[-len(self._metrics['mAP50']):]
            line, = self.ax.plot(steps, self._metrics['mAP50'], color=self._graph_colors[2], linewidth=2, 
                   label='mAP@50', marker='o', markersize=3)
            self._legend_items['mAP@50'] = line
            has_data = True
            
        if self._metrics['mAP50-95']:
            steps = self._steps[-len(self._metrics['mAP50-95']):]
            line, = self.ax.plot(steps, self._metrics['mAP50-95'], color=self._graph_colors[3], linewidth=2,
                   label='mAP@50-95', marker='s', markersize=3)
            self._legend_items['mAP@50-95'] = line
            has_data = True
        
        if not has_data:
            # Show informative message
            self.ax.text(0.5, 0.5, 'Validation metrics will appear\nduring validation phases', 
                   transform=self.ax.transAxes, ha='center', va='center',
                   fontsize=14, color='#888888', style='italic')
                   
        self.ax.set_xlabel('Training Steps', fontsize=10)
        self.ax.set_ylabel('mAP Value', fontsize=10)
        self.ax.set_title('Mean Average Precision', fontsize=12, fontweight='bold')
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(self._grid_enabled, alpha=0.3, linestyle='--')
        
        # Only add legend if we have data
        if self._metrics['mAP50'] or self._metrics['mAP50-95']:
            legend = self.ax.legend(loc='lower right', fontsize=9)
            self._make_legend_interactive(legend)
        
        self._style_axes(self.ax)
        
    def _plot_precision_recall(self):
        """Plot precision and recall metrics."""
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self._legend_items.clear()
        
        has_data = False
        
        if self._metrics['precision']:
            steps = self._steps[-len(self._metrics['precision']):]
            line, = self.ax.plot(steps, self._metrics['precision'], color=self._graph_colors[4], linewidth=2,
                   label='Precision', marker='^', markersize=3)
            self._legend_items['Precision'] = line
            has_data = True
                   
        if self._metrics['recall']:
            steps = self._steps[-len(self._metrics['recall']):]
            line, = self.ax.plot(steps, self._metrics['recall'], color=self._graph_colors[5], linewidth=2,
                   label='Recall', marker='v', markersize=3)
            self._legend_items['Recall'] = line
            has_data = True
        
        if not has_data:
            # Show informative message
            self.ax.text(0.5, 0.5, 'Validation metrics will appear\nduring validation phases', 
                   transform=self.ax.transAxes, ha='center', va='center',
                   fontsize=14, color='#888888', style='italic')
                   
        self.ax.set_xlabel('Training Steps', fontsize=10)
        self.ax.set_ylabel('Value', fontsize=10)
        self.ax.set_title('Precision & Recall', fontsize=12, fontweight='bold')
        self.ax.set_ylim(0, 1.0)
        self.ax.grid(self._grid_enabled, alpha=0.3, linestyle='--')
        
        # Only add legend if we have data
        if self._metrics['precision'] or self._metrics['recall']:
            legend = self.ax.legend(loc='lower right', fontsize=9)
            self._make_legend_interactive(legend)
        
        self._style_axes(self.ax)
        
    def _plot_all_metrics(self):
        """Plot all available metrics in subplots."""
        self._legend_items.clear()
        # Create 2x2 subplot grid
        axes = []
        for i in range(4):
            ax = self.figure.add_subplot(2, 2, i+1)
            ax.set_facecolor('#1e1e1e')
            axes.append(ax)
            
        # Plot losses in first subplot
        if any(self._metrics[k] for k in ['loss', 'box_loss', 'cls_loss', 'dfl_loss']):
            ax = axes[0]
            if self._metrics['loss']:
                steps = self._steps[-len(self._metrics['loss']):]
                line, = ax.plot(steps, self._metrics['loss'], color=self._graph_colors[0], linewidth=1.5, label='Total')
                self._legend_items['Total'] = line
            if self._metrics['box_loss']:
                steps = self._steps[-len(self._metrics['box_loss']):]
                line, = ax.plot(steps, self._metrics['box_loss'], color=self._graph_colors[1], linewidth=1.5, label='Box', alpha=0.7)
                self._legend_items['Box'] = line
            if self._metrics['cls_loss']:
                steps = self._steps[-len(self._metrics['cls_loss']):]
                line, = ax.plot(steps, self._metrics['cls_loss'], color=self._graph_colors[2], linewidth=1.5, label='Class', alpha=0.7)
                self._legend_items['Class'] = line
            if self._metrics['dfl_loss']:
                steps = self._steps[-len(self._metrics['dfl_loss']):]
                line, = ax.plot(steps, self._metrics['dfl_loss'], color=self._graph_colors[3], linewidth=1.5, label='DFL', alpha=0.7)
                self._legend_items['DFL'] = line
            ax.set_title('Loss Components', fontsize=10)
            ax.set_xlabel('Steps', fontsize=8)
            ax.set_ylabel('Loss', fontsize=8)
            ax.grid(self._grid_enabled, alpha=0.3, linestyle='--')
            if self._legend_items:
                legend = ax.legend(fontsize=8, loc='upper right')
                # Store reference for crosshair
                if i == 0:
                    self.ax = ax
            self._style_axes(ax)
            
        # Plot mAP in second subplot
        if any(self._metrics[k] for k in ['mAP50', 'mAP50-95']):
            ax = axes[1]
            if self._metrics['mAP50']:
                steps = self._steps[-len(self._metrics['mAP50']):]
                line, = ax.plot(steps, self._metrics['mAP50'], color=self._graph_colors[2], linewidth=1.5, label='mAP@50', marker='o', markersize=3)
                self._legend_items['mAP@50'] = line
            if self._metrics['mAP50-95']:
                steps = self._steps[-len(self._metrics['mAP50-95']):]
                line, = ax.plot(steps, self._metrics['mAP50-95'], color=self._graph_colors[3], linewidth=1.5, label='mAP@50-95', marker='s', markersize=3)
                self._legend_items['mAP@50-95'] = line
            ax.set_title('mAP Metrics', fontsize=10)
            ax.set_xlabel('Steps', fontsize=8)
            ax.set_ylabel('mAP', fontsize=8)
            ax.set_ylim(0, 1.0)
            ax.grid(self._grid_enabled, alpha=0.3, linestyle='--')
            if self._metrics['mAP50'] or self._metrics['mAP50-95']:
                ax.legend(fontsize=8, loc='lower right')
            self._style_axes(ax)
            
        # Plot precision & recall in third subplot
        if any(self._metrics[k] for k in ['precision', 'recall']):
            ax = axes[2]
            if self._metrics['precision']:
                steps = self._steps[-len(self._metrics['precision']):]
                line, = ax.plot(steps, self._metrics['precision'], color=self._graph_colors[4], linewidth=1.5, label='Precision', marker='^', markersize=3)
                self._legend_items['Precision'] = line
            if self._metrics['recall']:
                steps = self._steps[-len(self._metrics['recall']):]
                line, = ax.plot(steps, self._metrics['recall'], color=self._graph_colors[5], linewidth=1.5, label='Recall', marker='v', markersize=3)
                self._legend_items['Recall'] = line
            ax.set_title('Precision & Recall', fontsize=10)
            ax.set_xlabel('Steps', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.set_ylim(0, 1.0)
            ax.grid(self._grid_enabled, alpha=0.3, linestyle='--')
            if self._metrics['precision'] or self._metrics['recall']:
                ax.legend(fontsize=8, loc='lower right')
            self._style_axes(ax)
        else:
            axes[2].set_visible(False)
            
        # Plot train vs validation loss in fourth subplot
        if any(self._metrics[k] for k in ['loss', 'val_loss']):
            ax = axes[3]
            if self._metrics['loss']:
                steps = self._steps[-len(self._metrics['loss']):]
                line, = ax.plot(steps, self._metrics['loss'], color=self._graph_colors[0], linewidth=1.5, label='Train', alpha=0.8)
                self._legend_items['Train Loss'] = line
            if self._metrics['val_loss']:
                steps = self._steps[-len(self._metrics['val_loss']):]
                line, = ax.plot(steps, self._metrics['val_loss'], color=self._graph_colors[1], linewidth=1.5, label='Val', marker='o', markersize=3)
                self._legend_items['Val Loss'] = line
            ax.set_title('Train vs Val Loss', fontsize=10)
            ax.set_xlabel('Steps', fontsize=8)
            ax.set_ylabel('Loss', fontsize=8)
            ax.grid(self._grid_enabled, alpha=0.3, linestyle='--')
            if self._metrics['loss'] or self._metrics['val_loss']:
                ax.legend(fontsize=8, loc='upper right')
            self._style_axes(ax)
        else:
            axes[3].set_visible(False)
            
        self.figure.suptitle('Training Metrics Overview', fontsize=12, fontweight='bold', y=0.98)
        
    def _style_axes(self, ax):
        """Apply consistent styling to axes."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#555555')
        ax.spines['bottom'].set_color('#555555')
        ax.tick_params(colors='#888888', labelsize=9)
        ax.grid(self._grid_enabled, alpha=0.3, linestyle='--')
        
    @pyqtSlot()
    def _on_chart_type_changed(self):
        """Handle chart type change."""
        self._pending_update = True
        # If monitoring is stopped, update immediately
        if not self._update_timer.isActive():
            self._update_charts()
        
    def save_chart(self, filepath: str):
        """Save current chart to file."""
        self.figure.savefig(filepath, dpi=150, bbox_inches='tight',
                           facecolor='#2b2b2b', edgecolor='none')
    
    def eventFilter(self, obj, event):
        """Event filter to show tooltips on hover."""
        if obj == self.canvas and event.type() == QEvent.Type.ToolTip:
            # Get current chart type
            chart_type = self.chart_combo.currentText()
            if chart_type in self.CHART_INFO:
                info = self.CHART_INFO[chart_type]
                
                # Create rich tooltip text
                tooltip_html = f"""
                <div style='background-color: #2b2b2b; padding: 10px; max-width: 400px;'>
                    <h3 style='color: #14ffec; margin-top: 0;'>{info['description']}</h3>
                    <p style='color: #cccccc; margin: 10px 0;'>{info['details']}</p>
                    <div style='background-color: #1e1e1e; padding: 8px; margin: 5px 0; border-radius: 4px;'>
                        <h4 style='color: #28a745; margin: 5px 0;'>✓ Good Signs:</h4>
                        <p style='color: #28a745; margin: 5px 0; white-space: pre-line;'>{info['good']}</p>
                    </div>
                    <div style='background-color: #1e1e1e; padding: 8px; margin: 5px 0; border-radius: 4px;'>
                        <h4 style='color: #dc3545; margin: 5px 0;'>✗ Warning Signs:</h4>
                        <p style='color: #dc3545; margin: 5px 0; white-space: pre-line;'>{info['bad']}</p>
                    </div>
                </div>
                """
                
                QToolTip.showText(QCursor.pos(), tooltip_html, self.canvas)
                return True
                
        return super().eventFilter(obj, event)
    
    def _toggle_pause(self):
        """Toggle pause/resume for real-time updates."""
        self._paused = not self._paused
        self.pause_btn.setText("▶" if self._paused else "⏸")
        self.pause_btn.setToolTip("Resume updates" if self._paused else "Pause updates")
        self.status_label.setText(self.status_label.text() + " (Paused)" if self._paused else self.status_label.text().replace(" (Paused)", ""))
    
    def _show_export_menu(self):
        """Show export options menu."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2b2b2b;
                border: 1px solid #555555;
            }
            QMenu::item {
                padding: 5px 20px;
                color: #cccccc;
            }
            QMenu::item:selected {
                background-color: #14ffec;
                color: #2b2b2b;
            }
        """)
        
        save_action = QAction("💾 Save as PNG", self)
        save_action.triggered.connect(lambda: self._export_chart('png'))
        menu.addAction(save_action)
        
        save_svg_action = QAction("📐 Save as SVG", self)
        save_svg_action.triggered.connect(lambda: self._export_chart('svg'))
        menu.addAction(save_svg_action)
        
        copy_action = QAction("📋 Copy to Clipboard", self)
        copy_action.triggered.connect(self._copy_to_clipboard)
        menu.addAction(copy_action)
        
        menu.exec(self.export_btn.mapToGlobal(self.export_btn.rect().bottomLeft()))
    
    def _show_options_menu(self):
        """Show chart options menu."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2b2b2b;
                border: 1px solid #555555;
            }
            QMenu::item {
                padding: 5px 20px;
                color: #cccccc;
            }
            QMenu::item:selected {
                background-color: #14ffec;
                color: #2b2b2b;
            }
            QMenu::separator {
                height: 1px;
                background-color: #555555;
                margin: 5px 10px;
            }
        """)
        
        grid_action = QAction("📊 Toggle Grid", self)
        grid_action.setCheckable(True)
        grid_action.setChecked(self._grid_enabled)
        grid_action.triggered.connect(self._toggle_grid)
        menu.addAction(grid_action)
        
        crosshair_action = QAction("✚ Toggle Crosshair", self)
        crosshair_action.setCheckable(True)
        crosshair_action.setChecked(self._crosshair_enabled)
        crosshair_action.triggered.connect(self._toggle_crosshair)
        menu.addAction(crosshair_action)
        
        menu.addSeparator()
        
        reset_action = QAction("🔄 Reset View", self)
        reset_action.triggered.connect(self._reset_view)
        menu.addAction(reset_action)
        
        menu.exec(self.options_btn.mapToGlobal(self.options_btn.rect().bottomLeft()))
    
    def _export_chart(self, format='png'):
        """Export chart to file."""
        from PyQt6.QtWidgets import QFileDialog
        
        file_filter = f"{format.upper()} Files (*.{format})" 
        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Save Chart as {format.upper()}", 
            f"training_chart.{format}", file_filter
        )
        
        if filepath:
            self.figure.savefig(filepath, dpi=150, bbox_inches='tight',
                              facecolor='#2b2b2b', edgecolor='none')
    
    def _copy_to_clipboard(self):
        """Copy chart to clipboard."""
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QBuffer
        import io
        
        # Save to bytes buffer
        buf = io.BytesIO()
        self.figure.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                           facecolor='#2b2b2b', edgecolor='none')
        buf.seek(0)
        
        # Convert to QPixmap and copy to clipboard
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        QApplication.clipboard().setPixmap(pixmap)
        
        # Show brief notification
        self.status_label.setText("Chart copied to clipboard")
        QTimer.singleShot(2000, lambda: self.status_label.setText(f"Step {self._total_steps} | Epoch {self._current_epoch}"))
    
    def _toggle_grid(self):
        """Toggle grid visibility."""
        self._grid_enabled = not self._grid_enabled
        if hasattr(self, 'ax') and self.ax:
            self.ax.grid(self._grid_enabled, alpha=0.3, linestyle='--')
            self.canvas.draw()
    
    def _toggle_crosshair(self):
        """Toggle crosshair cursor."""
        self._crosshair_enabled = not self._crosshair_enabled
        if not self._crosshair_enabled and self.crosshair_vline and self.crosshair_vline.axes:
            self.crosshair_vline.set_visible(False)
            self.crosshair_hline.set_visible(False)
            if self.value_text:
                self.value_text.set_visible(False)
            self.canvas.draw()
    
    def _reset_view(self):
        """Reset chart view to home."""
        self.toolbar.home()
    
    def _on_mouse_move(self, event):
        """Handle mouse movement for crosshair and value display."""
        if not self._crosshair_enabled or not event.inaxes:
            return
            
        ax = event.inaxes
        
        # Update or create crosshair lines
        if not self.crosshair_vline:
            self.crosshair_vline = ax.axvline(x=event.xdata, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
            self.crosshair_hline = ax.axhline(y=event.ydata, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
        else:
            self.crosshair_vline.set_xdata(event.xdata)
            self.crosshair_hline.set_ydata(event.ydata)
            self.crosshair_vline.set_visible(True)
            self.crosshair_hline.set_visible(True)
        
        # Find nearest data point
        nearest_point = self._find_nearest_point(event.xdata, event.ydata, ax)
        if nearest_point:
            x, y, label = nearest_point
            
            # Update or create value text
            if not self.value_text:
                self.value_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                        verticalalignment='top', horizontalalignment='left',
                                        bbox=dict(boxstyle='round', facecolor='#2b2b2b', alpha=0.8),
                                        color='white', fontsize=9)
            
            self.value_text.set_text(f'{label}: x={x:.0f}, y={y:.3f}')
            self.value_text.set_visible(True)
        
        self.canvas.draw_idle()
    
    def _find_nearest_point(self, x, y, ax):
        """Find the nearest data point to the cursor."""
        min_distance = float('inf')
        nearest = None
        
        for line in ax.get_lines():
            if not line.get_visible() or not line.get_label():
                continue
                
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            
            if len(xdata) == 0:
                continue
                
            # Find closest x point
            idx = np.searchsorted(xdata, x)
            if idx > 0 and (idx == len(xdata) or abs(x - xdata[idx-1]) < abs(x - xdata[idx])):
                idx = idx - 1
                
            if idx < len(xdata):
                distance = abs(x - xdata[idx])
                if distance < min_distance:
                    min_distance = distance
                    nearest = (xdata[idx], ydata[idx], line.get_label())
        
        return nearest if min_distance < 50 else None  # 50 pixel threshold
    
    def _on_mouse_click(self, event):
        """Handle mouse clicks for legend interaction."""
        if not event.inaxes:
            return
            
        # Check if click is on legend
        ax = event.inaxes
        if hasattr(ax, 'get_legend') and ax.get_legend():
            legend = ax.get_legend()
            
            # Simple check if click is near legend area
            bbox = legend.get_window_extent()
            if bbox.contains(event.x, event.y):
                # Toggle visibility of clicked legend item
                for legline, origline in zip(legend.get_lines(), self._legend_items.values()):
                    if legline.contains(event)[0]:
                        visible = not origline.get_visible()
                        origline.set_visible(visible)
                        legline.set_alpha(1.0 if visible else 0.2)
                        self.canvas.draw()
                        break
    
    def _make_legend_interactive(self, legend):
        """Make legend items clickable to toggle visibility."""
        if not legend:
            return
            
        # Map legend lines to original lines
        leg_lines = legend.get_lines()
        
        for leg_line, (label, orig_line) in zip(leg_lines, self._legend_items.items()):
            leg_line.set_picker(True)
            leg_line.set_pickradius(5)