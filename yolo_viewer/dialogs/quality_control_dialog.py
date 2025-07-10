"""Quality control dialog for auto-annotation assessment."""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextBrowser, QProgressBar, QLabel, QDialogButtonBox,
    QFileDialog, QGroupBox, QListWidget, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QTextCharFormat, QColor

from ..utils.auto_annotation_manager import AutoAnnotationSession, AnnotationProposal


class QualityStatus(Enum):
    """Quality status levels."""
    GOOD = "good"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class QualityMetric:
    """Individual quality metric."""
    name: str
    value: float
    status: QualityStatus
    description: str
    
    def to_html(self) -> str:
        """Convert to HTML representation."""
        if self.status == QualityStatus.GOOD:
            color = "#28a745"
            icon = "✓"
        elif self.status == QualityStatus.WARNING:
            color = "#ffc107"
            icon = "⚠"
        else:
            color = "#dc3545"
            icon = "✗"
            
        return f"""
        <tr>
            <td style="color: {color};">{icon}</td>
            <td><b>{self.name}</b></td>
            <td>{self.value:.2f}</td>
            <td>{self.description}</td>
        </tr>
        """


class QualityAssessmentWorker(QThread):
    """Worker thread for quality assessment."""
    
    progress = pyqtSignal(int, int, str)  # current, total, message
    metricCalculated = pyqtSignal(QualityMetric)
    finished = pyqtSignal(list)  # metrics
    
    def __init__(self, session: AutoAnnotationSession):
        super().__init__()
        self.session = session
        
    def run(self):
        """Run quality assessment."""
        metrics = []
        
        # 1. Coverage metric
        self.progress.emit(1, 5, "Calculating coverage...")
        coverage = self._calculate_coverage()
        metrics.append(coverage)
        self.metricCalculated.emit(coverage)
        
        # 2. Confidence distribution
        self.progress.emit(2, 5, "Analyzing confidence distribution...")
        conf_dist = self._calculate_confidence_distribution()
        metrics.append(conf_dist)
        self.metricCalculated.emit(conf_dist)
        
        # 3. Class balance
        self.progress.emit(3, 5, "Checking class balance...")
        class_balance = self._calculate_class_balance()
        metrics.append(class_balance)
        self.metricCalculated.emit(class_balance)
        
        # 4. Annotation density
        self.progress.emit(4, 5, "Measuring annotation density...")
        density = self._calculate_annotation_density()
        metrics.append(density)
        self.metricCalculated.emit(density)
        
        # 5. Modification rate
        self.progress.emit(5, 5, "Analyzing modifications...")
        mod_rate = self._calculate_modification_rate()
        metrics.append(mod_rate)
        self.metricCalculated.emit(mod_rate)
        
        self.finished.emit(metrics)
        
    def _calculate_coverage(self) -> QualityMetric:
        """Calculate dataset coverage."""
        total = self.session.total_images
        processed = self.session.processed_images
        
        if total == 0:
            coverage = 0
        else:
            coverage = processed / total
            
        if coverage >= 0.95:
            status = QualityStatus.GOOD
            desc = "Excellent coverage"
        elif coverage >= 0.8:
            status = QualityStatus.WARNING
            desc = "Good coverage, some images not processed"
        else:
            status = QualityStatus.ERROR
            desc = "Low coverage, many images not processed"
            
        return QualityMetric("Coverage", coverage, status, desc)
        
    def _calculate_confidence_distribution(self) -> QualityMetric:
        """Calculate confidence distribution balance."""
        stats = self.session.get_statistics()
        total = stats['total_proposals']
        
        if total == 0:
            return QualityMetric("Confidence Distribution", 0, 
                               QualityStatus.ERROR, "No proposals found")
        
        high_ratio = stats['high_confidence'] / total
        
        if high_ratio >= 0.6:
            status = QualityStatus.GOOD
            desc = "Good model confidence"
        elif high_ratio >= 0.3:
            status = QualityStatus.WARNING
            desc = "Moderate model confidence"
        else:
            status = QualityStatus.ERROR
            desc = "Low model confidence, consider retraining"
            
        return QualityMetric("High Confidence Ratio", high_ratio, status, desc)
        
    def _calculate_class_balance(self) -> QualityMetric:
        """Calculate class distribution balance."""
        class_counts = {}
        
        for proposals in self.session.proposals.values():
            for prop in proposals:
                if prop.is_approved:
                    class_counts[prop.class_id] = class_counts.get(prop.class_id, 0) + 1
                    
        if not class_counts:
            return QualityMetric("Class Balance", 0, 
                               QualityStatus.ERROR, "No approved annotations")
        
        # Calculate imbalance ratio
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        
        if min_count == 0:
            balance = 0
        else:
            balance = min_count / max_count
            
        if balance >= 0.5:
            status = QualityStatus.GOOD
            desc = "Well-balanced classes"
        elif balance >= 0.2:
            status = QualityStatus.WARNING
            desc = "Some class imbalance"
        else:
            status = QualityStatus.ERROR
            desc = "Severe class imbalance"
            
        return QualityMetric("Class Balance", balance, status, desc)
        
    def _calculate_annotation_density(self) -> QualityMetric:
        """Calculate average annotations per image."""
        total_annotations = sum(len(props) for props in self.session.proposals.values())
        images_with_annotations = len([p for p in self.session.proposals.values() if p])
        
        if images_with_annotations == 0:
            density = 0
        else:
            density = total_annotations / images_with_annotations
            
        if density >= 2:
            status = QualityStatus.GOOD
            desc = f"Good density ({density:.1f} annotations/image)"
        elif density >= 1:
            status = QualityStatus.WARNING
            desc = f"Low density ({density:.1f} annotations/image)"
        else:
            status = QualityStatus.ERROR
            desc = "Very sparse annotations"
            
        return QualityMetric("Annotation Density", density, status, desc)
        
    def _calculate_modification_rate(self) -> QualityMetric:
        """Calculate manual modification rate."""
        stats = self.session.get_statistics()
        approved = stats['approved']
        modified = stats['modified']
        
        if approved == 0:
            mod_rate = 0
        else:
            mod_rate = modified / approved
            
        if mod_rate <= 0.1:
            status = QualityStatus.GOOD
            desc = "Low modification rate, good model performance"
        elif mod_rate <= 0.3:
            status = QualityStatus.WARNING
            desc = "Moderate modifications needed"
        else:
            status = QualityStatus.ERROR
            desc = "High modification rate, model may need improvement"
            
        return QualityMetric("Modification Rate", mod_rate, status, desc)


class QualityControlDialog(QDialog):
    """Dialog for quality control assessment."""
    
    def __init__(self, session: AutoAnnotationSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.metrics: List[QualityMetric] = []
        self.setWindowTitle("Quality Control Assessment")
        self.setModal(True)
        self.resize(800, 600)
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("<h2>Quality Control Assessment</h2>")
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        
        self._run_btn = QPushButton("Run Assessment")
        self._run_btn.clicked.connect(self._run_assessment)
        header_layout.addWidget(self._run_btn)
        
        layout.addLayout(header_layout)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)
        
        self._progress_label = QLabel()
        self._progress_label.setVisible(False)
        layout.addWidget(self._progress_label)
        
        # Main content area
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Metrics list
        metrics_group = QGroupBox("Quality Metrics")
        metrics_layout = QVBoxLayout()
        self._metrics_list = QListWidget()
        metrics_layout.addWidget(self._metrics_list)
        metrics_group.setLayout(metrics_layout)
        splitter.addWidget(metrics_group)
        
        # Report view
        report_group = QGroupBox("Assessment Report")
        report_layout = QVBoxLayout()
        self._report_view = QTextBrowser()
        report_layout.addWidget(self._report_view)
        report_group.setLayout(report_layout)
        splitter.addWidget(report_group)
        
        splitter.setSizes([300, 500])
        layout.addWidget(splitter)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self._export_btn = QPushButton("Export Report")
        self._export_btn.clicked.connect(self._export_report)
        self._export_btn.setEnabled(False)
        button_layout.addWidget(self._export_btn)
        
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self._close_btn)
        
        layout.addLayout(button_layout)
        
    @pyqtSlot()
    def _run_assessment(self):
        """Run quality assessment."""
        self._run_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_label.setVisible(True)
        self._metrics_list.clear()
        
        # Start worker
        self._worker = QualityAssessmentWorker(self.session)
        self._worker.progress.connect(self._update_progress)
        self._worker.metricCalculated.connect(self._add_metric)
        self._worker.finished.connect(self._on_assessment_complete)
        self._worker.start()
        
    @pyqtSlot(int, int, str)
    def _update_progress(self, current: int, total: int, message: str):
        """Update progress display."""
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._progress_label.setText(message)
        
    @pyqtSlot(QualityMetric)
    def _add_metric(self, metric: QualityMetric):
        """Add metric to list."""
        self.metrics.append(metric)
        
        # Add to list widget
        if metric.status == QualityStatus.GOOD:
            icon = "✓"
            color = "#28a745"
        elif metric.status == QualityStatus.WARNING:
            icon = "⚠"
            color = "#ffc107"
        else:
            icon = "✗"
            color = "#dc3545"
            
        item_text = f"{icon} {metric.name}: {metric.value:.2f}"
        self._metrics_list.addItem(item_text)
        
    @pyqtSlot(list)
    def _on_assessment_complete(self, metrics: List[QualityMetric]):
        """Handle assessment completion."""
        self._progress_bar.setVisible(False)
        self._progress_label.setVisible(False)
        self._run_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        
        # Generate report
        self._generate_report()
        
    def _generate_report(self):
        """Generate HTML report."""
        stats = self.session.get_statistics()
        
        # Count status
        good_count = sum(1 for m in self.metrics if m.status == QualityStatus.GOOD)
        warning_count = sum(1 for m in self.metrics if m.status == QualityStatus.WARNING)
        error_count = sum(1 for m in self.metrics if m.status == QualityStatus.ERROR)
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                table {{ border-collapse: collapse; width: 100%; }}
                td, th {{ padding: 8px; text-align: left; }}
                .good {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <h2>Quality Assessment Report</h2>
            
            <h3>Summary</h3>
            <table>
                <tr>
                    <td class="good">✓ Good: {good_count}</td>
                    <td class="warning">⚠ Warning: {warning_count}</td>
                    <td class="error">✗ Error: {error_count}</td>
                </tr>
            </table>
            
            <h3>Session Statistics</h3>
            <table>
                <tr><td>Total Images:</td><td>{stats['total_images']}</td></tr>
                <tr><td>Processed:</td><td>{stats['processed_images']}</td></tr>
                <tr><td>Total Proposals:</td><td>{stats['total_proposals']}</td></tr>
                <tr><td>Approved:</td><td>{stats['approved']}</td></tr>
            </table>
            
            <h3>Quality Metrics</h3>
            <table>
                <tr>
                    <th>Status</th>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
                {''.join(m.to_html() for m in self.metrics)}
            </table>
            
            <h3>Recommendations</h3>
            <ul>
        """
        
        # Add recommendations based on metrics
        for metric in self.metrics:
            if metric.status == QualityStatus.ERROR:
                if "Coverage" in metric.name:
                    html += "<li>Process more images to improve dataset coverage</li>"
                elif "Confidence" in metric.name:
                    html += "<li>Consider retraining model with more diverse data</li>"
                elif "Balance" in metric.name:
                    html += "<li>Add more examples of underrepresented classes</li>"
                elif "Modification" in metric.name:
                    html += "<li>Review model training parameters and dataset quality</li>"
                    
        html += """
            </ul>
        </body>
        </html>
        """
        
        self._report_view.setHtml(html)
        
    @pyqtSlot()
    def _export_report(self):
        """Export report to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "quality_report.html", "HTML Files (*.html)"
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(self._report_view.toHtml())